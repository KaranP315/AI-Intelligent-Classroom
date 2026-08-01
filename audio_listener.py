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

    def __init__(self, cv_log_path, sscon_output_path, chunk_duration=10, sample_rate=16000):
        """
        Args:
            cv_log_path: Path to attentiveness_log.csv (written by the camera thread)
            sscon_output_path: Path to sscon_output.json (output of this listener)
            chunk_duration: Duration of each audio chunk in seconds
            sample_rate: Audio sample rate (Whisper expects 16kHz)
        """
        self.cv_log_path = cv_log_path
        self.sscon_output_path = sscon_output_path
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.stop_event = threading.Event()
        self.dialogue_log = []
        self.history = []  # SSCon dialogue context history

        # Initialize models
        self._load_models()

    def _load_models(self):
        """Load all ML models needed for the pipeline."""
        print("[Audio] Loading Whisper model (base)...")
        self.whisper_model = whisper.load_model("base")

        # SSCon Models
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Audio] SSCon device: {self.device}")

        print("[Audio] Loading Sentence-BERT (all-MiniLM-L6-v2)...")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

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

        self.bimodel = torch.load(bimodel_path, weights_only=False).to(self.device)
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

    def run(self):
        """
        Main recording loop. Records audio in chunks, transcribes,
        classifies, and attributes to the visible student.
        """
        print(f"\n[Audio] 🎤 Microphone recording started ({self.chunk_duration}s chunks)")
        print("[Audio] Speak into the microphone. Speech will be classified in real-time.\n")

        while not self.stop_event.is_set():
            try:
                # Record a chunk of audio
                audio_data = sd.rec(
                    int(self.chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32'
                )
                # Wait for recording to complete, but check stop_event periodically
                for _ in range(self.chunk_duration * 10):  # Check every 100ms
                    if self.stop_event.is_set():
                        sd.stop()
                        return
                    time.sleep(0.1)

                sd.wait()  # Ensure recording is done

                # Save to temporary WAV file for Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    sf.write(tmp_path, audio_data, self.sample_rate)

                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    tmp_path,
                    language="en",
                    fp16=False,  # MPS/CPU compatibility
                    condition_on_previous_text=False # Stops Whisper from hallucinating random YouTube subtitles during silence
                )
                text = result["text"].strip()

                # Clean up temp file
                os.unlink(tmp_path)

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

                # Print live output
                print(f"  [🎤 {speaker}] \"{text}\"")
                print(f"      → SSCon: {act_label} (class {pred_class}, conf: {probs[pred_class]*100:.1f}%)")

            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Audio] Error: {e}")
                continue

    def stop(self):
        """Signal the listener to stop."""
        self.stop_event.set()
        sd.stop()
        self._save_log()
        print(f"\n[Audio] 🎤 Microphone stopped. {len(self.dialogue_log)} utterances classified.")
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
