"""
VoiceClip entry point.

Usage:
    python -m voiceclip
"""

import logging
import multiprocessing
import signal
import sys
import time

from voiceclip import __version__
from voiceclip.config import MODEL, ENGLISH_ONLY, validate
from voiceclip.macos import cleanup_sounds, check_accessibility, check_microphone


def setup_logging():
    """Configure logging with timestamps and levels."""
    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)


def _check_dependencies():
    """Verify critical dependencies are importable. Exit with a friendly
    message if anything is missing (instead of a cryptic child-process crash)."""
    missing = []
    for mod in ("sounddevice", "soundfile", "mlx_whisper", "pynput", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)


def main():
    multiprocessing.set_start_method("spawn", force=True)
    setup_logging()
    validate()
    _check_dependencies()

    log = logging.getLogger("voiceclip")

    # Import after dependency check so we get friendly errors
    from voiceclip.recorder import Recorder, cleanup_stale_temps
    from voiceclip.transcriber import preload_model
    from voiceclip.hotkey import HotkeyHandler
    from voiceclip.formatter import load_dictionary

    print("=" * 50)
    print(f"  🎙️  VoiceClip v{__version__}")
    print("  Local voice → clipboard on Apple Silicon")
    print("=" * 50)
    print(f"\n  Model:        {MODEL}")
    print(f"  English only: {ENGLISH_ONLY}")

    # Check macOS permissions early
    check_microphone()
    has_accessibility = check_accessibility()
    if not has_accessibility:
        print("  ⚠️  Accessibility not granted — auto-paste disabled")
        print("     Transcriptions will still be copied to clipboard")

    # Load user dictionary for word replacements
    load_dictionary()

    # Clean up temp files from previous runs
    cleanup_stale_temps()

    # Start the audio recorder child process
    print("\n  Starting audio recorder...")
    recorder = Recorder()

    # Set up a cleanup handler so Ctrl+C at any point cleans up the recorder
    def _shutdown(signum=None, frame=None):
        print("\n👋 VoiceClip stopped.")
        try:
            handler.stop()
        except Exception:
            pass
        recorder.shutdown()
        cleanup_sounds()
        sys.exit(0)

    # handler is set later, but _shutdown references it — use a mutable container
    handler = type("H", (), {"stop": lambda self: None})()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        recorder.start()
    except RuntimeError as e:
        log.error("Failed to start recorder: %s", e)
        sys.exit(1)
    print("  ✅ Recorder ready")

    # Preload the Whisper model so first transcription is fast
    print("\n  Preloading Whisper model (first run downloads ~3 GB)...")
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
    real_handler = HotkeyHandler(recorder)
    real_handler.start()
    # Replace the stub so _shutdown can stop it
    handler = real_handler

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
