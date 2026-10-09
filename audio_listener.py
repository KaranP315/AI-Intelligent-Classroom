"""
audio_listener.py - Real-time Whisper Transcription + SSCon Dialogue Act Classification

Records audio from the microphone in chunks, transcribes with Whisper,
classifies each utterance with the SSCon model, and attributes the
speech to the student currently visible in the camera (via attentiveness_log.csv).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import threading
import time
import json
import tempfile
import csv
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import torch
from scipy import signal
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from xgboost import XGBClassifier
import xgboost as xgb

# Monkeypatch for scikit-learn wrapper compatibility
XGBClassifier._estimator_type = "classifier"

# Ensure train_biencoder is importable (for torch.load of bimodel.pt)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_biencoder import HiddenLayerBiEncoder
from setup_models import get_torch_device


# SSCon act mapping (from the paper / inference.py)
ACT_MAPPING = {
    0: "Relating to another student",
    1: "Asking for more information",
    2: "Making a claim",
    3: "Providing evidence",
}

# Map SSCon acts to our Tier system for the point system
ACT_TO_TIER_LABEL = {
    0: "Relating to another student",   # Tier 1 (+3)
    1: "Asking for more information",    # Tier 2 (+1)
    2: "Making a claim",                 # Tier 2 (+1)
    3: "Providing evidence",             # Tier 1 (+3)
}


def get_last_visible_student(cv_log_path):
    """
    Reads the last few lines of the CV attentiveness log to find
    the most recently detected student name from the camera.
    Returns the student name string, or 'Unknown' if unavailable.
    """
    try:
        with open(cv_log_path, 'r') as f:
            lines = f.readlines()
        # Skip header, get the last non-empty line
        data_lines = [l.strip() for l in lines[1:] if l.strip()]
        if data_lines:
            last_line = data_lines[-1]
            parts = last_line.split(',')
            if len(parts) >= 2:
                return parts[1]  # Person column
    except (FileNotFoundError, IndexError):
        pass
    return "Unknown"


class AudioListener:
    """
    Real-time audio listener that records, transcribes, classifies,
    and attributes dialogue acts to students.
    """

    def __init__(
        self,
        cv_log_path,
        sscon_output_path,
        chunk_duration=8,
        sample_rate=16000,
        whisper_model_name=None,
        mic_device=None,
    ):
        """
        Args:
            cv_log_path: Path to attentiveness_log.csv (written by the camera thread)
            sscon_output_path: Path to sscon_output.json (output of this listener)
            chunk_duration: Duration of each audio chunk in seconds
            sample_rate: Audio sample rate (Whisper expects 16kHz)
            whisper_model_name: Whisper model name, or AUDIO_WHISPER_MODEL
                when omitted
            mic_device: Input device index, or AUDIO_INPUT_DEVICE when omitted
        """
        self.cv_log_path = cv_log_path
        self.sscon_output_path = sscon_output_path
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.whisper_model_name = whisper_model_name or os.getenv(
            "AUDIO_WHISPER_MODEL", "small.en"
        )
        configured_mic = os.getenv("AUDIO_INPUT_DEVICE")
        self.mic_device = mic_device
        if self.mic_device is None and configured_mic:
            try:
                self.mic_device = int(configured_mic)
            except ValueError:
                print(f"[Audio] Ignoring invalid AUDIO_INPUT_DEVICE={configured_mic!r}")
        self.stop_event = threading.Event()
        self.dialogue_log = []
        self.history = []  # SSCon dialogue context history

        # Initialize models
        self._load_models()

    def _load_models(self):
        """Load all ML models needed for the pipeline."""
        print(f"[Audio] Loading Whisper model ({self.whisper_model_name})...")
        self.whisper_model = whisper.load_model(self.whisper_model_name)

        # SSCon Models
        self.device = get_torch_device()
        print(f"[Audio] SSCon device: {self.device}")

        print("[Audio] Loading Sentence-BERT (all-MiniLM-L6-v2)...")
        self.sbert_model = SentenceTransformer("./all-MiniLM-L6-v2")

        print("[Audio] Loading DialoGPT-medium...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.dialog_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium"
        ).to(self.device)
        self.dialog_model.eval()

        print("[Audio] Loading trained SSCon bi-encoder and classifier...")
        base_dir = os.path.dirname(__file__)
        bimodel_path = os.path.join(base_dir, "trained_models", "bimodel.pt")
        xgb_path = os.path.join(base_dir, "trained_models", "xgboost_classifier.json")

        if not os.path.exists(bimodel_path) or not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"SSCon trained models not found at {bimodel_path} or {xgb_path}. "
                "Please train the model first."
            )

        self.bimodel = torch.load(bimodel_path, map_location=self.device, weights_only=False).to(self.device)
        self.bimodel.eval()

        self.xgb_clf = XGBClassifier()
        self.xgb_clf.load_model(xgb_path)

        print("[Audio] All models loaded successfully!")

    def classify_utterance(self, text):
        """
        Run the full SSCon inference pipeline on a single utterance.
        Returns: (predicted_class_index, act_label, probabilities)
        """
        # Add to dialogue context history
        self.history.append(text)
        context_turns = self.history[-10:]  # Keep last 10 turns

        # 1. Generate DialoGPT Contextual Embedding (1024-dim)
        context_text = "".join(
            [turn + self.tokenizer.eos_token for turn in context_turns]
        )
        inputs = self.tokenizer(context_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dialog_model(**inputs, output_hidden_states=True)
        hi = outputs.hidden_states[12][0, -1, :].cpu().numpy()

        # 2. Generate Sentence-BERT Embedding (384-dim)
        si = self.sbert_model.encode(text)

        # 3. Concatenate (1408-dim)
        combined_embed = np.concatenate([hi, si], axis=0).astype(np.float32)

        # 4. Bi-Encoder projection (384-dim)
        x_in = torch.tensor(combined_embed).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x_proj = self.bimodel(x_in).cpu().numpy().astype(np.float32)

        x_proj_contiguous = np.ascontiguousarray(x_proj)

        # 5. XGBoost classification
        dmat = xgb.DMatrix(x_proj_contiguous)
        preds_prob = self.xgb_clf.get_booster().predict(dmat)[0]
        pred_class = int(np.argmax(preds_prob))

        act_label = ACT_MAPPING.get(pred_class, "Unknown")
        return pred_class, act_label, preds_prob

    def _save_log(self):
        """Save the current dialogue log to sscon_output.json."""
        with open(self.sscon_output_path, 'w') as f:
            json.dump(self.dialogue_log, f, indent=2)

    def _find_working_mic(self):
        """Auto-detect the best working microphone device."""
        import numpy as np
        print("[Audio] Scanning for a working microphone...")
        devices = sd.query_devices()

        if self.mic_device is not None:
            if 0 <= self.mic_device < len(devices):
                device = devices[self.mic_device]
                if device["max_input_channels"] > 0:
                    native_sr = int(device["default_samplerate"])
                    print(
                        f"[Audio] Using configured mic: Device {self.mic_device} "
                        f"({device['name']}) at {native_sr} Hz"
                    )
                    return self.mic_device, native_sr
            print(f"[Audio] Configured mic {self.mic_device} is unavailable; auto-detecting.")

        candidates = []
        
        # Blacklist output/loopback and low-bitrate telephony drivers
        blacklist_keywords = ["speaker", "output", "mapper", "stereo mix", "wave", "surround", "@system32", "bthhfenum"]
        
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                name_lower = d['name'].lower()
                if any(kw in name_lower for kw in blacklist_keywords):
                    continue
                    
                native_sr = int(d['default_samplerate'])
                try:
                    audio = sd.rec(int(0.5 * native_sr),
                                   samplerate=native_sr,
                                   channels=1, dtype='float32', device=i)
                    sd.wait()
                    vol = float(np.max(np.abs(audio)))
                    if 0.0005 < vol <= 1.0 and not np.isnan(vol):
                        candidates.append((i, d['name'], native_sr, vol))
                        print(f"  [OK] Device {i}: {d['name']} (sr={native_sr}, vol={vol:.4f})")
                except Exception:
                    pass
                    
        if candidates:
            # Prioritize clean headsets / external mics
            headset_candidates = [c for c in candidates if "headset" in c[1].lower() or "buds" in c[1].lower()]
            if headset_candidates:
                best = max(headset_candidates, key=lambda x: x[3])
            else:
                best = max(candidates, key=lambda x: x[3])
            print(f"[Audio] Selected mic: Device {best[0]} ({best[1]}) at {best[2]} Hz")
            return best[0], best[2]
            
        print("[Audio] Checking high-quality headset inputs directly...")
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                name_lower = d['name'].lower()
                if any(kw in name_lower for kw in blacklist_keywords):
                    continue
                if "headset" in name_lower or "oneplus" in name_lower or "buds" in name_lower:
                    native_sr = int(d['default_samplerate'])
                    print(f"[Audio] Selected mic (direct match): Device {i} ({d['name']}) at {native_sr} Hz")
                    return i, native_sr

        return None, self.sample_rate

    def _prepare_audio(self, audio_array, mic_native_sr):
        """Filter, resample, and conservatively level one microphone chunk."""
        audio_array = np.nan_to_num(
            audio_array.flatten().astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Remove low-frequency rumble before measuring speech energy.
        if mic_native_sr > 200:
            highpass = signal.butter(
                4, 80.0, btype="highpass", fs=mic_native_sr, output="sos"
            )
            audio_array = signal.sosfiltfilt(highpass, audio_array).astype(np.float32)

        if mic_native_sr != self.sample_rate:
            num_samples = int(len(audio_array) * self.sample_rate / mic_native_sr)
            audio_array = signal.resample(audio_array, num_samples).astype(np.float32)

        rms = float(np.sqrt(np.mean(np.square(audio_array))))
        if rms < 0.004:
            return None

        # Avoid peak normalization, which can turn steady background noise into speech.
        target_rms = 0.08
        gain = min(target_rms / rms, 3.0)
        audio_array = np.clip(audio_array * gain, -1.0, 1.0)

        frame_length = max(1, int(self.sample_rate * 0.03))
        frame_count = len(audio_array) // frame_length
        if frame_count == 0:
            return None
        frame_rms = np.sqrt(
            np.mean(
                np.square(audio_array[: frame_count * frame_length]).reshape(
                    frame_count, frame_length
                ),
                axis=1,
            )
        )
        speech_threshold = max(0.012, float(np.median(frame_rms) * 2.0))
        if float(np.mean(frame_rms > speech_threshold)) < 0.08:
            return None

        return np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

    def _load_subject_prompt(self):
        """Load subject.txt and build an initial prompt to prime Whisper."""
        subject_path = os.path.join(os.path.dirname(__file__), "subject.txt")
        if os.path.exists(subject_path):
            with open(subject_path, "r", encoding="utf-8") as f:
                subject = f.read().strip()
            # Build a short prompt with key vocabulary from the subject
            return f"Classroom discussion about: {subject[:200]}"
        return ""

    def run(self):
        """
        Main recording loop. Records audio in chunks, transcribes,
        classifies, and attributes to the visible student.
        """
        mic_device, mic_native_sr = self._find_working_mic()
        whisper_prompt = self._load_subject_prompt()
        if whisper_prompt:
            print(f"[Audio] Whisper primed with subject context ({len(whisper_prompt)} chars)")

        print(f"\n[Audio]  Microphone recording started ({self.chunk_duration}s chunks)")
        print("[Audio] Speak into the microphone. Speech will be classified in real-time.\n")

        while not self.stop_event.is_set():
            try:
                # Record at the mic's NATIVE sample rate for best quality
                rec_kwargs = dict(
                    samplerate=mic_native_sr,
                    channels=1,
                    dtype='float32'
                )
                if mic_device is not None:
                    rec_kwargs['device'] = mic_device
                audio_data = sd.rec(
                    int(self.chunk_duration * mic_native_sr),
                    **rec_kwargs
                )
                # Wait for recording to complete, but check stop_event periodically
                for _ in range(self.chunk_duration * 10):  # Check every 100ms
                    if self.stop_event.is_set():
                        sd.stop()
                        return
                    time.sleep(0.1)

                sd.wait()  # Ensure recording is done

                # Check if audio is silent
                import numpy as np
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

                audio_array = self._prepare_audio(audio_data, mic_native_sr)
                if audio_array is None:
                    print("[Audio] No speech detected in chunk; skipping transcription.")
                    continue

                # Save to disk for debugging
                import soundfile as sf
                sf.write("debug_last_chunk.wav", audio_array, 16000)

                # Transcribe with Whisper directly from RAM
                transcribe_kwargs = dict(
                    language="en",
                    fp16=False,  # MPS/CPU compatibility
                    condition_on_previous_text=False,
                    temperature=0.0,
                    beam_size=5,
                )
                if whisper_prompt:
                    transcribe_kwargs['initial_prompt'] = whisper_prompt
                result = self.whisper_model.transcribe(audio_array, **transcribe_kwargs)
                text = result["text"].strip()

                # Skip empty or noise-only transcriptions
                if not text or len(text) < 3:
                    continue

                # Get the student currently visible in the camera
                speaker = get_last_visible_student(self.cv_log_path)

                # Classify the utterance with SSCon
                pred_class, act_label, probs = self.classify_utterance(text)

                timestamp = datetime.now().isoformat()

                # Log the result
                entry = {
                    "timestamp": timestamp,
                    "speaker": speaker,
                    "text": text,
                    "dialogue_act": act_label,
                    "dialogue_act_index": pred_class,
                    "probabilities": [float(p) for p in probs]
                }
                self.dialogue_log.append(entry)
                self._save_log()

                # LIVE Semantic Relevance Check
                import double_gated_scorer
                import os
                
                subject_text = ""
                subject_path = os.path.join(os.path.dirname(__file__), "subject.txt")
                if os.path.exists(subject_path):
                    with open(subject_path, "r", encoding="utf-8") as f:
                        subject_text = f.read().strip()
                
                is_relevant = double_gated_scorer.check_audio_relevance(text, subject_text)

                # Print live output
                print(f"  [{speaker}] \"{text}\"")
                print(f"       SSCon: {act_label} (conf: {probs[pred_class]*100:.1f}%) | Relevant to Subject: {is_relevant}")

            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Audio] Error: {e}")
                continue

    def stop(self):
        """Signal the listener to stop."""
        self.stop_event.set()
        sd.stop()
        self._save_log()
        print(f"\n[Audio]  Microphone stopped. {len(self.dialogue_log)} utterances classified.")
        print(f"[Audio] SSCon log saved to: {self.sscon_output_path}")


if __name__ == "__main__":
    # Standalone test mode
    print("=" * 50)
    print("Audio Listener - Standalone Test Mode")
    print("=" * 50)

    cv_path = os.path.join(os.path.dirname(__file__),
                           "Intelligent-Classroom-System", "attentiveness_log.csv")
    sscon_path = os.path.join(os.path.dirname(__file__), "sscon_output.json")

    listener = AudioListener(cv_log_path=cv_path, sscon_output_path=sscon_path)

    try:
        listener.run()
    except KeyboardInterrupt:
        listener.stop()
