"""
session_runner.py - Unified Session Runner

Launches the CV camera pipeline and Audio+SSCon pipeline simultaneously.
When the session ends (ESC or Ctrl+C), generates the final fused attentiveness report.

Usage:
    cd "/Users/bvtejas/Documents/pi labs"
    source Intelligent-Classroom-System/venv/bin/activate
    python session_runner.py
"""

import os
import sys
import threading
import time
import signal
import subprocess
import platform

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.join(BASE_DIR, "Intelligent-Classroom-System")
CV_LOG = os.path.join(CV_DIR, "attentiveness_log.csv")
SSCON_OUTPUT = os.path.join(BASE_DIR, "sscon_output.json")


def clear_old_logs():
    """Clear previous session data so we start fresh."""
    os.makedirs(CV_DIR, exist_ok=True)
    # Clear CV log (keep header)
    with open(CV_LOG, 'w') as f:
        f.write("Timestamp,Person,Behavior,Confidence\n")
    print("[Session] Cleared attentiveness_log.csv")

    # Clear SSCon log
    if os.path.exists(SSCON_OUTPUT):
        os.remove(SSCON_OUTPUT)
        print("[Session] Cleared sscon_output.json")


def _cv_python_executable():
    """Resolve the Python executable for the CV subprocess (Windows + macOS/Linux)."""
    if platform.system() == "Windows":
        candidates = [
            sys.executable,
            os.path.join(CV_DIR, "venv", "Scripts", "python.exe"),
            os.path.join(BASE_DIR, "venv", "Scripts", "python.exe"),
        ]
    else:
        candidates = [
            sys.executable,
            os.path.join(CV_DIR, "venv", "bin", "python"),
            os.path.join(BASE_DIR, "venv", "bin", "python"),
        ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def run_cv_pipeline(stop_event):
    """
    Run the CV camera pipeline as a subprocess.
    We use subprocess so it runs in its own process with its own imports
    (TensorFlow + PyTorch in the same process can cause issues).
    """
    print("[Session] Starting CV camera pipeline...")
    venv_python = _cv_python_executable()

    proc = subprocess.Popen(
        [venv_python, "main.py"],
        cwd=CV_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output while checking for stop signal
    while not stop_event.is_set():
        line = proc.stdout.readline()
        if line:
            # Print CV output with a prefix for clarity
            print(f"  [] {line.rstrip()}")
        if proc.poll() is not None:
            # Process exited (user pressed ESC in the video window)
            stop_event.set()
            break
        time.sleep(0.01)

    # Cleanup
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=5)
    print("[Session] CV camera pipeline stopped.")


def run_audio_pipeline(listener, stop_event):
    """
    Run the Audio+SSCon pipeline.
    """
    try:
        # Run in a thread so we can monitor stop_event
        audio_thread = threading.Thread(target=listener.run, daemon=True)
        audio_thread.start()

        # Wait until stop is signaled
        while not stop_event.is_set():
            time.sleep(0.5)

        listener.stop()
        audio_thread.join(timeout=5)
    except Exception as e:
        import traceback
        print(f"\n[Audio] FATAL ERROR IN AUDIO PIPELINE:\n{traceback.format_exc()}")


def generate_report():
    """Run the scoring script to generate the final fused report."""
    print("\n" + "=" * 60)
    print("GENERATING FINAL ATTENTIVENESS REPORT...")
    print("=" * 60)
    sys.stdout.flush()

    try:
        # Import and run directly
        sys.path.insert(0, BASE_DIR)
        from score_attentiveness import generate_final_report
        generate_final_report(SSCON_OUTPUT, CV_LOG)
    except Exception as e:
        import traceback
        print(f"\n[Report Error]: {e}\n{traceback.format_exc()}")
    sys.stdout.flush()


def main():
    from setup_models import ensure_trained_models

    ensure_trained_models(BASE_DIR)

    print("=" * 60)
    print(" INTELLIGENT CLASSROOM SESSION RUNNER")
    print("=" * 60)
    print()
    print("This will start:")
    print("   Camera pipeline   tracks physical behavior (listen/write/trance/drink)")
    print("   Microphone        records & classifies dialogue acts via SSCon")
    print()
    print("Both run simultaneously. To end the session:")
    print("   Click the video window and press ESC, or")
    print("   Press Ctrl+C in this terminal")
    print()

    # Clear old data
    clear_old_logs()

    # Shared stop event
    stop_event = threading.Event()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[Session] Ctrl+C detected. Stopping session...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # Start CV pipeline in a thread (as a subprocess)
    cv_thread = threading.Thread(target=run_cv_pipeline, args=(stop_event,), daemon=True)
    cv_thread.start()

    print("[Session] Loading Audio ML Models (this takes 15-30 seconds)...")
    from audio_listener import AudioListener
    listener = AudioListener(
        cv_log_path=CV_LOG,
        sscon_output_path=SSCON_OUTPUT,
        chunk_duration=8,
        sample_rate=16000
    )

    # Start Audio+SSCon pipeline
    print("[Session] Starting Audio+SSCon pipeline...")
    audio_thread = threading.Thread(target=run_audio_pipeline, args=(listener, stop_event,), daemon=True)
    audio_thread.start()

    # Preload Relevance Gateway model into memory so shutdown report is instant
    import double_gated_scorer
    double_gated_scorer._get_model()

    print("\n" + "=" * 60)
    print(" SESSION IS LIVE  Camera + Microphone active!")
    print("=" * 60 + "\n")

    # Wait for session to end
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    # Wait for threads to finish
    print("\n[Session] Shutting down...")
    cv_thread.join(timeout=10)
    audio_thread.join(timeout=10)

    # Generate the final fused report
    generate_report()

    print("\n" + "=" * 60)
    print(" SESSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
