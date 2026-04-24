"""
VoiceClip entry point.

Usage:
    python -m voiceclip              # Start recording mode
    python -m voiceclip history      # View transcription history
"""

import argparse
import logging
import multiprocessing
import signal
import sys
import time

from voiceclip import __version__
from voiceclip.macos import cleanup_sounds, check_accessibility, check_microphone


def setup_logging():
    """Configure logging with timestamps and levels."""
    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)


def _check_dependencies():
    """Verify critical dependencies are importable."""
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voiceclip",
        description="Local voice-to-clipboard for macOS",
    )
    parser.add_argument(
        "--version", action="version", version=f"VoiceClip {__version__}"
    )

    sub = parser.add_subparsers(dest="command")

    # history subcommand
    hist = sub.add_parser("history", help="View transcription history")
    hist.add_argument("--today", action="store_true", help="Show today's transcriptions")
    hist.add_argument("--yesterday", action="store_true", help="Show yesterday's transcriptions")
    hist.add_argument("--search", type=str, metavar="TERM", help="Search by keyword")
    hist.add_argument("--copy", type=int, metavar="ID", help="Copy entry by ID to clipboard")
    hist.add_argument("--clear", action="store_true", help="Delete all history")
    hist.add_argument("--count", action="store_true", help="Show total count")
    hist.add_argument("-n", type=int, default=10, help="Number of results (default: 10)")

    return parser


def _handle_history(args):
    """Handle the history subcommand."""
    from voiceclip import config
    config.load()

    if not config.HISTORY_ENABLED:
        print("History is not enabled. Set \"history\": true in ~/.voiceclip/config.json")
        sys.exit(0)

    from voiceclip.history import (
        init, query_recent, query_today, query_yesterday,
        query_search, get_by_id, clear_all, count,
    )
    init()

    if args.clear:
        clear_all()
    elif args.count:
        print(f"  {count()} transcriptions in history")
    elif args.copy:
        text = get_by_id(args.copy)
        if text:
            from voiceclip.macos import copy_to_clipboard
            copy_to_clipboard(text)
            print(f"  Copied to clipboard: \"{text[:100]}\"")
        else:
            print(f"  Entry {args.copy} not found")
    elif args.search:
        print(query_search(args.search, limit=args.n))
    elif args.today:
        print(query_today())
    elif args.yesterday:
        print(query_yesterday())
    else:
        print(query_recent(limit=args.n))


def _run_voiceclip():
    """Main recording mode."""
    multiprocessing.set_start_method("spawn", force=True)
    setup_logging()
    _check_dependencies()

    from voiceclip import config
    config.load()
    config.validate()

    log = logging.getLogger("voiceclip")

    from voiceclip.recorder import Recorder, cleanup_stale_temps
    from voiceclip.transcriber import preload_model
    from voiceclip.hotkey import HotkeyHandler
    from voiceclip.formatter import build_patterns

    print("=" * 50)
    print(f"  🎙️  VoiceClip v{__version__}")
    print("  Local voice → clipboard on Apple Silicon")
    print("=" * 50)
    print(f"\n  Model:        {config.MODEL}")
    print(f"  English only: {config.ENGLISH_ONLY}")
    print(f"  Persona:      {config.PERSONA}")
    print(f"  Hotkey:       {config.hotkey_display_name()} ({config.HOTKEY_MODE} mode)")
    print(f"  Dictionary:   {len(config.DICTIONARY)} entries")
    print(f"  History:      {'✅ enabled' if config.HISTORY_ENABLED else 'off'}")
    print(f"  Config:       {config.CONFIG_PATH}")

    # Initialize history if enabled
    if config.HISTORY_ENABLED:
        from voiceclip.history import init as init_history, cleanup as cleanup_history
        init_history()
        cleanup_history(config.HISTORY_MAX_DAYS)

    check_microphone()
    has_accessibility = check_accessibility()
    if not has_accessibility:
        print("  ⚠️  Accessibility not granted — auto-paste disabled")
        print("     Transcriptions will still be copied to clipboard")

    build_patterns()
    cleanup_stale_temps()

    print("\n  Starting audio recorder...")
    recorder = Recorder()

    def _shutdown(signum=None, frame=None):
        print("\n👋 VoiceClip stopped.")
        try:
            handler.stop()
        except Exception:
            pass
        recorder.shutdown()
        cleanup_sounds()
        # Close history DB if open
        if config.HISTORY_ENABLED:
            try:
                from voiceclip.history import close as close_history
                close_history()
            except Exception:
                pass
        sys.exit(0)

    handler = type("H", (), {"stop": lambda self: None})()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        recorder.start()
    except RuntimeError as e:
        log.error("Failed to start recorder: %s", e)
        sys.exit(1)
    print("  ✅ Recorder ready")

    print("\n  Preloading Whisper model (first run downloads ~3 GB)...")
    preload_model()
    print("  ✅ Model ready")

    print("\n  Microphones:")
    try:
        print(recorder.list_devices())
    except RuntimeError as e:
        log.warning("Could not list devices: %s", e)

    print()
    hotkey_name = config.hotkey_display_name()
    if config.HOTKEY_MODE == "toggle":
        print(f"  ⌨️  Press {hotkey_name} to start recording")
        print(f"     Press again to stop & transcribe")
    else:
        print(f"  ⌨️  Hold {hotkey_name} to record")
        print("     Release to transcribe & copy to clipboard")
    if config.HISTORY_ENABLED:
        print("     History: voiceclip history")
    print("     Ctrl+C to quit")
    print()

    real_handler = HotkeyHandler(recorder)
    real_handler.start()
    handler = real_handler

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "history":
        _handle_history(args)
    else:
        _run_voiceclip()


if __name__ == "__main__":
    main()
