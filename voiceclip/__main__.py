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
from voiceclip.macos import check_accessibility, check_microphone, cleanup_sounds


def setup_logging():
    """Configure logging with timestamps and levels."""
    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)


def _check_dependencies():
    """Verify critical dependencies are importable."""
    from voiceclip import config
    base_deps = ["sounddevice", "soundfile", "pynput", "numpy"]
    if config.ENGINE == "parakeet":
        base_deps.append("parakeet_mlx")
    else:
        base_deps.append("mlx_whisper")

    missing = []
    for mod in base_deps:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        if "parakeet_mlx" in missing:
            print("   Run: pip install parakeet-mlx")
        else:
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

    # doctor subcommand — system health check
    sub.add_parser("doctor", help="Run a health check on your VoiceClip setup")

    # export subcommand — full data dump
    exp = sub.add_parser("export", help="Export all history data (JSON or CSV)")
    exp.add_argument("--format", choices=["json", "csv"], default="json",
                     help="Output format (default: json)")
    exp.add_argument("--output", "-o", type=str, metavar="PATH",
                     help="Write to file instead of stdout")

    # import subcommand — restore from a JSON export
    imp = sub.add_parser("import", help="Import history from a JSON export file")
    imp.add_argument("file", type=str, help="Path to the JSON export file")
    imp.add_argument("--no-merge", action="store_true",
                     help="Fail on conflicts instead of skipping existing rows")

    # onboard subcommand — re-run the first-run walkthrough
    sub.add_parser("onboard", help="Run (or re-run) the first-launch walkthrough")

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
        clear_all,
        clear_kind,
        count,
        get_entry_full,
        init,
        promote_to_reflection,
        query_recent,
        query_search,
        query_today,
        query_yesterday,
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

    from voiceclip.consent import check_and_warn as _cloud_check
    from voiceclip.history import init
    from voiceclip.summarizer import cloud_provider_warning, summarize_day

    init()
    _cloud_check()

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


def _handle_export(args):
    """Handle the export subcommand — full data dump."""
    from voiceclip import config
    config.load()

    if not config.HISTORY_ENABLED:
        print("History is not enabled. Nothing to export.")
        sys.exit(0)

    from voiceclip.history import init
    init()

    from voiceclip.export import export_all
    data = export_all(fmt=args.format)

    if args.output:
        with open(args.output, "w") as f:
            f.write(data)
        print(f"  Exported to {args.output}")
    else:
        print(data)


def _handle_import(args):
    """Handle the import subcommand — restore from a JSON export."""
    from voiceclip import config
    config.load()

    # Import needs history enabled so the DB exists and is initialized.
    # If it's off, turn it on temporarily for the import — the user
    # clearly wants their data here.
    from voiceclip.history import init
    init()

    from voiceclip.export import import_from_file
    try:
        counts = import_from_file(args.file, merge=not args.no_merge)
    except RuntimeError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    total = sum(counts.values())
    if total == 0:
        print("  Nothing new to import (all rows already exist).")
    else:
        print(f"  ✅ Imported {total} rows:")
        if counts["entries"]:
            print(f"     {counts['entries']} entries")
        if counts["summaries"]:
            print(f"     {counts['summaries']} day summaries")
        if counts["timelines"]:
            print(f"     {counts['timelines']} day timelines")
        if counts["briefs"]:
            print(f"     {counts['briefs']} research briefs")


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

    from voiceclip import config
    config.load()
    _check_dependencies()
    config.validate()

    # First-run walkthrough, if the user hasn't been through it yet.
    # Safe no-op on non-TTY / repeat runs; reloads config after in case the
    # user opted into new features during the flow.
    from voiceclip.onboard import needs_onboarding
    from voiceclip.onboard import run as run_onboard
    if needs_onboarding():
        run_onboard()
        config.load()  # pick up any opt-ins the user just committed

    # Surface any cloud-provider config change before we start using it
    from voiceclip.consent import check_and_warn as _cloud_check
    _cloud_check()

    log = logging.getLogger("voiceclip")

    from voiceclip.formatter import build_patterns
    from voiceclip.hotkey import HotkeyHandler
    from voiceclip.recorder import Recorder, cleanup_stale_temps
    from voiceclip.transcriber import preload_model, start_keep_warm, stop_keep_warm

    print("=" * 50)
    print(f"  🎙️  VoiceClip v{__version__}")
    print("  Local voice → clipboard on Apple Silicon")
    print("=" * 50)
    print(f"\n  Engine:       {config.ENGINE}")
    if config.ENGINE == "parakeet":
        print(f"  Model:        {config.PARAKEET_MODEL}")
    else:
        print(f"  Model:        {config.MODEL}")
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
        from voiceclip.history import cleanup as cleanup_history
        from voiceclip.history import init as init_history
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

    # Best-effort workaround for a pyobjc lazy-import bug on Python 3.14
    # that makes `HIServices.AXIsProcessTrusted` raise KeyError when
    # pynput's listener thread tries to read it. Symptom: "Exception in
    # thread Thread-N" traceback during listener startup, then the
    # listener is dead and hotkeys do nothing.
    #
    # Touching HIServices here on the main thread does two things:
    #   (a) If the lazy import succeeds, the function object gets bound
    #       into the HIServices module namespace, so pynput's later
    #       access goes through the normal attribute lookup (not the
    #       broken lazy path) and doesn't crash.
    #   (b) If it fails HERE too — same bug in main thread — we at
    #       least log a debug line about it and proceed. The listener
    #       will still crash; _validate_hotkey_listener will spot the
    #       non-running listener and point the user at the logs.
    #
    # Known-good combos: pyobjc 12.x on Python 3.12. Known-bad: pyobjc
    # 12.1 on Python 3.14 (funcmap.pop KeyError). If you're on 3.14
    # and hitting this, consider falling back to 3.12 until pyobjc ships
    # a fix.
    try:
        import HIServices  # type: ignore[import-not-found]
        HIServices.AXIsProcessTrusted()
    except Exception as e:
        log.debug("pyobjc HIServices warmup skipped: %s", e)

    build_patterns()
    cleanup_stale_temps()

    print("\n  Starting audio recorder...")
    recorder = Recorder()
    handlers: list = []

    def _shutdown(signum=None, frame=None):
        print("\n👋 VoiceClip stopped.")
        stop_keep_warm()
        # Shut down whisper.cpp server if running
        if config.ENGINE == "whisper_cpp":
            try:
                from voiceclip.engine_whisper_cpp import shutdown as shutdown_whisper
                shutdown_whisper()
            except Exception:
                pass
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

    if config.ENGINE == "parakeet":
        print("\n  Preloading Parakeet model (first run downloads ~1-4 GB)...")
    else:
        print("\n  Preloading Whisper model (first run downloads ~3 GB)...")
    if preload_model():
        print("  ✅ Model ready")
    else:
        print("  ⚠️  Model not loaded (will retry on first use)")
    start_keep_warm()

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
        # Two different affordances users consistently miss according to the
        # product review — surface both here at daemon start, once per run.
        # Low-cost reminder that costs nothing if the user already knows
        # these commands.
        print("     History:   voiceclip history")
        print("     Browse:    voiceclip view    (opens the web journal)")
        print("     Diagnose:  voiceclip doctor  (if anything seems off)")
    else:
        print("     Diagnose:  voiceclip doctor  (if anything seems off)")
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

    # Polish hotkey — LLM-cleaned dictation. Requires a summaries provider.
    polish_active = False
    if config.POLISH_HOTKEY:
        used_keys = {config.HOTKEY.strip().lower()}
        if config.REFLECTION_HOTKEY:
            used_keys.add(config.REFLECTION_HOTKEY.strip().lower())
        if config.POLISH_HOTKEY.strip().lower() in used_keys:
            log.error(
                "polish_hotkey '%s' collides with another hotkey; disabled",
                config.POLISH_HOTKEY,
            )
        else:
            polish_key_obj = config.resolve_hotkey(config.POLISH_HOTKEY)
            polish_handler = HotkeyHandler(
                recorder,
                hotkey=polish_key_obj,
                mode=config.POLISH_HOTKEY_MODE,
                profile="polished",
                start_sound="Tink",
                done_sound="Hero",
                label="VoiceClip ✨",
            )
            polish_handler.start()
            handlers.append(polish_handler)
            polish_active = True
            pname = config.hotkey_display_name(config.POLISH_HOTKEY)
            print(f"  ✨  Hold {pname} to dictate with LLM polish ({config.POLISH_HOTKEY_MODE} mode)")

    # Validate the hotkey listener actually registered with macOS.
    # pynput's `Listener.start()` returns immediately but the underlying
    # CFRunLoop only checks AXIsProcessTrusted (Accessibility grant) on
    # its own thread shortly after. If permission is missing, the listener
    # stays "running" but silently receives no events — user presses the
    # key, nothing happens, no clue why. Wait a beat then check the
    # Darwin-specific IS_TRUSTED flag and log a loud warning if false.
    _validate_hotkey_listener(transcription_handler, "transcription")
    if reflection_active:
        _validate_hotkey_listener(reflection_handler, "reflection")
    if polish_active:
        _validate_hotkey_listener(polish_handler, "polished")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()


def _validate_hotkey_listener(handler, label: str):
    """Post-start check that the pynput listener is actually receiving
    events. The Darwin pynput backend surfaces an IS_TRUSTED flag that
    mirrors macOS Accessibility grant — if it's False the listener is
    a no-op even though `.running` is True.

    We give the listener 500ms to initialize on its CFRunLoop thread
    before checking.

    Three distinct failure modes, worth distinguishing because the fix
    for each is different:

      a) Listener object missing         -> pynput constructor failed
      b) Listener not running             -> _run() thread crashed before
                                             or during AXIsProcessTrusted
                                             check (library compat bug,
                                             not a permission issue)
      c) Listener running, IS_TRUSTED=False -> Accessibility is actually
                                               denied

    The (b) case matters: if the listener thread died (e.g. pyobjc's
    lazy-import raised KeyError on Python 3.14), IS_TRUSTED stays at
    its class-level default of False — which the old check misread as
    "permission denied" and shouted a misleading Accessibility error.
    Dictation might still work through a different code path (the child
    recorder process), which is maximally confusing for the user.

    Check `.running` BEFORE `IS_TRUSTED` so we don't diagnose a crashed
    listener as a permission problem.
    """
    log = logging.getLogger("voiceclip")
    listener = getattr(handler, "_listener", None)
    if listener is None:
        log.warning(
            "Hotkey %s listener did not start. Run `voiceclip doctor`.",
            label,
        )
        return

    time.sleep(0.5)  # let _run() set IS_TRUSTED on the listener thread

    # Case (b): listener thread crashed or never started.
    if not getattr(listener, "running", False):
        log.warning(
            "Hotkey %s listener is not running after startup. "
            "This usually means a library crashed during pynput init "
            "(check stderr for a Thread traceback above). "
            "Dictation may still work if it recovers on first key press. "
            "Run `voiceclip doctor` if the hotkey never responds.",
            label,
        )
        return

    # Case (c): listener is healthy, IS_TRUSTED is the trustworthy signal.
    is_trusted = getattr(listener, "IS_TRUSTED", None)
    if is_trusted is False:
        log.error(
            "⚠️  Hotkey %s is registered but macOS Accessibility is NOT granted. "
            "Key presses will be ignored until you grant it. "
            "System Settings → Privacy & Security → Accessibility → add your terminal. "
            "Then restart VoiceClip. Run `voiceclip doctor` to verify.",
            label,
        )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "history":
        _handle_history(args)
    elif args.command == "view":
        _handle_view(args)
    elif args.command == "summarize":
        _handle_summarize(args)
    elif args.command == "doctor":
        from voiceclip.doctor import run as run_doctor
        sys.exit(run_doctor())
    elif args.command == "export":
        _handle_export(args)
    elif args.command == "import":
        _handle_import(args)
    elif args.command == "onboard":
        from voiceclip.onboard import run as run_onboard
        run_onboard(force=True)
    else:
        _run_voiceclip()


if __name__ == "__main__":
    main()
