"""
VoiceClip entry point.

Usage:
    python -m voiceclip
"""

import logging
import multiprocessing
import sys
import time

from voiceclip import __version__
from voiceclip.config import MODEL, ENGLISH_ONLY, validate
from voiceclip.recorder import Recorder, cleanup_stale_temps
from voiceclip.transcriber import preload_model
from voiceclip.hotkey import HotkeyHandler
from voiceclip.macos import cleanup_sounds


def setup_logging():
    """Configure logging with timestamps and levels."""
    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)


def main():
    multiprocessing.set_start_method("spawn", force=True)
    setup_logging()
    validate()

    log = logging.getLogger("voiceclip")

    print("=" * 50)
    print(f"  🎙️  VoiceClip v{__version__}")
    print("  Local voice → clipboard on Apple Silicon")
    print("=" * 50)
    print(f"\n  Model:        {MODEL}")
    print(f"  English only: {ENGLISH_ONLY}")

    # Clean up temp files from previous runs
    cleanup_stale_temps()

    # Start the audio recorder child process
    print("\n  Starting audio recorder...")
    recorder = Recorder()
    try:
        recorder.start()
    except RuntimeError as e:
        log.error("Failed to start recorder: %s", e)
        sys.exit(1)
    print("  ✅ Recorder ready")

    # Preload the Whisper model so first transcription is fast
    print("\n  Preloading Whisper model...")
    preload_model()
    print("  ✅ Model ready")

    # List available mics
    print("\n  Microphones:")
    try:
        print(recorder.list_devices())
    except RuntimeError as e:
        log.warning("Could not list devices: %s", e)

    print()
    print("  ⌨️  Hold Right Option (⌥) to record")
    print("     Release to transcribe & copy to clipboard")
    print("     Ctrl+C to quit")
    print()

    # Start the hotkey listener
    handler = HotkeyHandler(recorder)
    handler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 VoiceClip stopped.")
        handler.stop()
        recorder.shutdown()
        cleanup_sounds()
        sys.exit(0)


if __name__ == "__main__":
    main()
