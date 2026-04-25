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
    hist = sub.add_parser("history", help="View transcription / reflection history")
    hist.add_argument("--today", action="store_true", help="Show today's entries")
    hist.add_argument("--yesterday", action="store_true", help="Show yesterday's entries")
    hist.add_argument("--search", type=str, metavar="TERM", help="Search by keyword")
    hist.add_argument("--copy", type=int, metavar="ID", help="Copy entry by ID to clipboard")
    hist.add_argument("--clear", action="store_true", help="Delete history")
    hist.add_argument("--count", action="store_true", help="Show total count")
    hist.add_argument("-n", type=int, default=10, help="Number of results (default: 10)")
    # Kind filters
    hist.add_argument("--reflections", action="store_true",
                      help="Only show reflection entries")
    hist.add_argument("--transcriptions", action="store_true",
                      help="Only show transcription entries")
    # Promote helpers
    hist.add_argument("--reflect-last", action="store_true",
                      help="Promote the most recent transcription to a reflection")
    hist.add_argument("--reflect-id", type=int, metavar="ID",
                      help="Promote entry ID to a reflection")

    # view subcommand — local web viewer
    view = sub.add_parser("view", help="Open the web viewer (localhost only)")
    view.add_argument("--port", type=int, default=8723, help="Port (default: 8723)")
    view.add_argument("--no-browser", action="store_true",
                      help="Don't auto-open a browser")

    # summarize subcommand — LLM day digest
    sumz = sub.add_parser("summarize", help="Generate / cache a daily summary")
    sumz.add_argument("--day", type=str, default="today",
                      help="'today', 'yesterday', or YYYY-MM-DD (default: today)")
    sumz.add_argument("--force", action="store_true",
                      help="Regenerate even if cached")

    return parser


def _kind_filter(args) -> str | None:
    """Return the kind filter or None. If both flags set, returns None."""
    if args.reflections and args.transcriptions:
        return None
    if args.reflections:
        return "reflection"
    if args.transcriptions:
        return "transcription"
    return None


def _handle_history(args):
    """Handle the history subcommand."""
    from voiceclip import config
    config.load()

    if not config.HISTORY_ENABLED:
        print("History is not enabled. Set \"history\": true in ~/.voiceclip/config.json")
        sys.exit(0)

    from voiceclip.history import (
        init, query_recent, query_today, query_yesterday,
        query_search, get_by_id, get_entry_full, clear_all, clear_kind,
        count, promote_to_reflection,
    )
    init()

    kind = _kind_filter(args)

    # Promote handlers come first (they're orthogonal to query flags)
    if args.reflect_last or args.reflect_id is not None:
        target = "last" if args.reflect_last else f"id={args.reflect_id}"
        result = promote_to_reflection(
            entry_id=args.reflect_id,
            last=args.reflect_last,
        )
        if result is None:
            print(f"  Could not promote {target} (not found or already a reflection).")
        else:
            preview = result["text"][:80] + ("..." if len(result["text"]) > 80 else "")
            print(f"  💭 Promoted [{result['id']}] {result['timestamp']}: \"{preview}\"")
        return

    if args.clear:
        if kind is None:
            clear_all()
        else:
            clear_kind(kind)
    elif args.count:
        if kind is None:
            print(f"  {count()} entries in history")
        else:
            print(f"  {count(kind=kind)} {kind} entries in history")
    elif args.copy:
        entry = get_entry_full(args.copy)
        if entry:
            from voiceclip.macos import copy_to_clipboard
            copy_to_clipboard(entry["text"])
            marker = "reflection" if entry["kind"] == "reflection" else "transcription"
            preview = entry["text"][:100]
            print(f"  Copied {marker} [{entry['id']}] to clipboard: \"{preview}\"")
        else:
            print(f"  Entry {args.copy} not found")
    elif args.search:
        print(query_search(args.search, limit=args.n, kind=kind))
    elif args.today:
        print(query_today(kind=kind))
    elif args.yesterday:
        print(query_yesterday(kind=kind))
    else:
        print(query_recent(limit=args.n, kind=kind))


def _resolve_day_arg(value: str) -> str:
    """Convert 'today' / 'yesterday' / YYYY-MM-DD to YYYY-MM-DD."""
    from datetime import datetime, timedelta
    v = (value or "today").strip().lower()
    if v == "today":
        return datetime.now().strftime("%Y-%m-%d")
    if v == "yesterday":
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        datetime.strptime(v, "%Y-%m-%d")
        return v
    except ValueError:
        print(f"  Unrecognized day: {value!r}. Use 'today', 'yesterday', or YYYY-MM-DD.")
        sys.exit(2)


def _handle_summarize(args):
    """Handle the summarize subcommand."""
    setup_logging()
    from voiceclip import config
    config.load()

    if not config.HISTORY_ENABLED:
        print("History is not enabled. Set \"history\": true in ~/.voiceclip/config.json")
        sys.exit(0)

    if config.SUMMARIES_PROVIDER == "none":
        print(
            "Summaries are off. Enable them in ~/.voiceclip/config.json:\n\n"
            '  "summaries": {\n'
            '    "provider": "local"        // or "openai", or "anthropic"\n'
            "  }\n\n"
            "Local runs on your Mac with mlx-lm (install it with: pip install mlx-lm).\n"
            "OpenAI and Anthropic send your day's entries to their APIs.\n"
            "API keys come from env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY."
        )
        sys.exit(0)

    from voiceclip.history import init
    from voiceclip.summarizer import summarize_day, cloud_provider_warning

    init()

    warn = cloud_provider_warning()
    if warn:
        print(f"  ⚠️  {warn}\n")

    date = _resolve_day_arg(args.day)
    print(f"  Summarizing {date} via {config.SUMMARIES_PROVIDER}...")
    try:
        result = summarize_day(date, force=args.force)
    except RuntimeError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    if result is None:
        print(f"  No entries for {date}.")
        return

    print(f"\n  Summary ({result['entry_count']} entries, {result['model']}):\n")
    for line in result["summary"].splitlines() or [result["summary"]]:
        print(f"    {line}")
    print()


def _handle_view(args):
    """Handle the view subcommand — start the local web viewer."""
    setup_logging()
    from voiceclip import config
    config.load()

    if not config.HISTORY_ENABLED:
        print("History is not enabled. Set \"history\": true in ~/.voiceclip/config.json")
        sys.exit(0)

    from voiceclip.summarizer import cloud_provider_warning
    warn = cloud_provider_warning()
    if warn:
        print(f"  ⚠️  {warn}")

    from voiceclip.viewer import serve
    serve(port=args.port, open_browser=not args.no_browser)


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

    # Decide whether the reflection hotkey will be registered (requires history on).
    reflection_active = False
    reflection_key_obj = None
    if config.HISTORY_ENABLED and config.REFLECTION_HOTKEY:
        if config.REFLECTION_HOTKEY.strip().lower() == config.HOTKEY.strip().lower():
            log.error(
                "reflection_hotkey '%s' collides with hotkey '%s'; "
                "reflection hotkey disabled",
                config.REFLECTION_HOTKEY, config.HOTKEY,
            )
        else:
            reflection_key_obj = config.resolve_hotkey(config.REFLECTION_HOTKEY)
            reflection_active = True
            print(
                f"  Reflections:  ✅ {config.hotkey_display_name(config.REFLECTION_HOTKEY)} "
                f"({config.REFLECTION_HOTKEY_MODE} mode)"
            )

    # Initialize history if enabled
    if config.HISTORY_ENABLED:
        from voiceclip.history import init as init_history, cleanup as cleanup_history
        init_history()
        cleanup_history(
            config.HISTORY_MAX_DAYS,
            reflection_max_days=config.REFLECTION_MAX_DAYS,
        )

    check_microphone()
    has_accessibility = check_accessibility()
    if not has_accessibility:
        print("  ⚠️  Accessibility not granted — auto-paste disabled")
        print("     Transcriptions will still be copied to clipboard")

    build_patterns()
    cleanup_stale_temps()

    print("\n  Starting audio recorder...")
    recorder = Recorder()

    handlers: list = []

    def _shutdown(signum=None, frame=None):
        print("\n👋 VoiceClip stopped.")
        for h in handlers:
            try:
                h.stop()
            except Exception:
                pass
        recorder.shutdown()
        cleanup_sounds()
        if config.HISTORY_ENABLED:
            try:
                from voiceclip.history import close as close_history
                close_history()
            except Exception:
                pass
        sys.exit(0)

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
        print("     Press again to stop & transcribe")
    else:
        print(f"  ⌨️  Hold {hotkey_name} to record")
        print("     Release to transcribe & copy to clipboard")

    if reflection_active:
        rname = config.hotkey_display_name(config.REFLECTION_HOTKEY)
        if config.REFLECTION_HOTKEY_MODE == "toggle":
            print(f"  💭  Press {rname} to capture a reflection (press again to stop)")
        else:
            print(f"  💭  Hold {rname} to capture a reflection (saved, not pasted)")

    if config.HISTORY_ENABLED:
        print("     History: voiceclip history")
    print("     Ctrl+C to quit")
    print()

    transcription_handler = HotkeyHandler(
        recorder,
        hotkey=config.resolve_hotkey(),
        mode=config.HOTKEY_MODE,
        profile="transcription",
        start_sound="Tink",
        done_sound="Glass",
    )
    transcription_handler.start()
    handlers.append(transcription_handler)

    if reflection_active:
        reflection_handler = HotkeyHandler(
            recorder,
            hotkey=reflection_key_obj,
            mode=config.REFLECTION_HOTKEY_MODE,
            profile="reflection",
            start_sound="Morse",
            done_sound="Submarine",
        )
        reflection_handler.start()
        handlers.append(reflection_handler)

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
    elif args.command == "view":
        _handle_view(args)
    elif args.command == "summarize":
        _handle_summarize(args)
    else:
        _run_voiceclip()


if __name__ == "__main__":
    main()
