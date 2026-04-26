"""First-run onboarding flow for VoiceClip.

Runs once on first launch (no `~/.voiceclip/.onboarded` marker) or when
the user invokes `voiceclip onboard` explicitly. A 5-step keyboard-driven
walkthrough that:

  1. Introduces the tool in one screen
  2. Checks macOS permissions (Accessibility + Microphone)
  3. Invites a live test dictation
  4. Offers the optional features (reflections, toggle mode, history,
     summaries) one at a time with plain-language explanations
  5. Points at the next commands to try

Design principles:
  - Every step is skippable with Enter. Nothing is required.
  - Every opt-in writes to the real ~/.voiceclip/config.json via the
    shared patcher in voiceclip.config_io so the choice persists.
  - The onboarding itself captures nothing, sends nothing, and never errors
    the startup path — failures print one line and move on.
  - Zero new dependencies; pure stdlib + what VoiceClip already has.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from voiceclip import config
from voiceclip.config_io import write_config_patch

log = logging.getLogger(__name__)

# ANSI color helpers — silent on non-TTY so log files / piped output stay clean
_IS_TTY = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _IS_TTY else ""


_BOLD = _c("\x1b[1m")
_DIM = _c("\x1b[2m")
_GREEN = _c("\x1b[32m")
_YELLOW = _c("\x1b[33m")
_BLUE = _c("\x1b[34m")
_RESET = _c("\x1b[0m")


def _marker_path() -> Path:
    return Path(config.CONFIG_DIR) / ".onboarded"


def needs_onboarding() -> bool:
    """True if the user has never been through onboarding (and there's a TTY)."""
    if not _IS_TTY:
        return False  # piped / non-interactive — never prompt
    return not _marker_path().exists()


def mark_done():
    """Record that onboarding has finished (or been skipped)."""
    try:
        p = _marker_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Small UI helpers — pure stdlib, no curses
# ---------------------------------------------------------------------------

def _rule():
    print(f"\n{_DIM}───────────────────────────────────────────────────────────────{_RESET}\n")


def _header(step_num: int, total: int, title: str):
    print(f"\n{_BOLD}Step {step_num}/{total}:  {title}{_RESET}\n")


def _ask(prompt: str, *, default: str = "n") -> bool:
    """Yes/no prompt. Default is shown in caps. Enter accepts default."""
    suffix = " [Y/n]" if default.lower() == "y" else " [y/N]"
    try:
        raw = input(f"  {prompt}{suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not raw:
        return default.lower() == "y"
    return raw.startswith("y")


def _pause(msg: str = "Press Enter to continue, or 's' to skip this step"):
    try:
        raw = input(f"  {_DIM}{msg}{_RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    return raw != "s"


# ---------------------------------------------------------------------------
# Config writer — merge into ~/.voiceclip/config.json in-place
# ---------------------------------------------------------------------------

def _write_config_patch(patch: dict) -> bool:
    """Thin wrapper for backward compat with existing tests."""
    return write_config_patch(patch)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _step_welcome():
    _header(1, 5, "Welcome to VoiceClip")
    print(f"""  {_BOLD}Talk. It types.{_RESET}  Hold a key, speak, let go — text lands
  wherever your cursor is.

  This quick walkthrough (about a minute) sets up the basics and
  shows you a few features you might otherwise miss.
""")
    _pause("Press Enter to begin")


def _step_permissions():
    _header(2, 5, "macOS permissions")
    # Reuse the accessibility probe from doctor. Mic we can't really check
    # without opening a stream, so we just remind.
    print("  Checking permissions...\n")
    try:
        res = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to return name of first process'],
            capture_output=True, timeout=3,
        )
        if res.returncode == 0:
            print(f"  {_GREEN}✓{_RESET} Accessibility (type-into-apps) is granted")
        else:
            print(f"  {_YELLOW}!{_RESET} Accessibility is not granted yet.")
            print(f"     {_DIM}Open System Settings → Privacy & Security → Accessibility,{_RESET}")
            print(f"     {_DIM}and add the terminal app you're using (Terminal, iTerm, etc).{_RESET}")
    except Exception:
        print(f"  {_DIM}(couldn't check — you'll see a prompt later if it's missing){_RESET}")

    print(f"\n  {_DIM}Microphone permission is asked the first time you dictate.{_RESET}")
    print(f"  {_DIM}Grant it when macOS prompts — that's the only prompt you'll see.{_RESET}")
    _pause()


def _step_try_it():
    _header(3, 5, "Try dictating")
    print(f"""  Here's the core flow:

    1. {_BOLD}Hold Right Option (⌥){_RESET} — you hear a tink
    2. Say something — "hello world"
    3. Release — your words appear

  You don't need to do this right now, but it's how VoiceClip works
  in every app: Slack, Notes, email, code editor, browser, terminal.

  {_DIM}Skip for now; we'll finish setup and you can try it at the end.{_RESET}
""")
    _pause()


def _step_features() -> dict:
    """Walk the opt-in features. Returns the config patch to write."""
    _header(4, 5, "Optional features")
    print(f"""  {_BOLD}VoiceClip has a few extras that are off by default.{_RESET} They're
  off because they change what VoiceClip does — you should turn them
  on deliberately, not by accident.
""")
    patch: dict = {}

    # --- Reflections ---
    print(f"\n  {_BOLD}💭 Reflections hotkey{_RESET}")
    print(f"  {_DIM}A second key (F6 by default) for capturing private thoughts.{_RESET}")
    print(f"  {_DIM}Unlike normal dictation, reflections are saved to your journal{_RESET}")
    print(f"  {_DIM}only — they never paste anywhere or touch the clipboard.{_RESET}")
    if _ask("Enable reflections?", default="n"):
        patch["history"] = True
        patch["reflection_hotkey"] = "f6"
        print(f"    {_GREEN}✓{_RESET} Reflections on. Hold F6 to capture a reflection.")
        print(f"    {_DIM}You can change the key later in ~/.voiceclip/config.json{_RESET}")
    else:
        print(f"    {_DIM}Skipped. Enable later by setting reflection_hotkey in config.{_RESET}")

    # --- Toggle mode (accessibility) ---
    print(f"\n  {_BOLD}⌨️  Hotkey mode{_RESET}")
    print(f"  {_DIM}Default is 'hold-to-record' — natural, but taxing if you{_RESET}")
    print(f"  {_DIM}dictate for long spans or have any hand-use constraint.{_RESET}")
    print(f"  {_DIM}Toggle mode: press once to start, press again to stop.{_RESET}")
    if _ask("Use toggle mode instead?", default="n"):
        patch["hotkey_mode"] = "toggle"
        if "reflection_hotkey" in patch:
            patch["reflection_hotkey_mode"] = "toggle"
        print(f"    {_GREEN}✓{_RESET} Toggle mode on for all hotkeys.")
    else:
        print(f"    {_DIM}Sticking with hold mode. Change in config anytime.{_RESET}")

    # --- History / Journal ---
    if not patch.get("history"):
        print(f"\n  {_BOLD}📓 Journal / history{_RESET}")
        print(f"  {_DIM}Keep a searchable log of everything you dictate. Lives on{_RESET}")
        print(f"  {_DIM}your Mac, never synced. Enables the web viewer (`voiceclip view`).{_RESET}")
        if _ask("Enable the journal?", default="n"):
            patch["history"] = True
            print(f"    {_GREEN}✓{_RESET} Journal on. Browse with: voiceclip view")
        else:
            print(f"    {_DIM}Skipped. Dictation still works; nothing is saved.{_RESET}")

    # --- Daily summaries (only offer if history is on) ---
    if patch.get("history") or _has_history_enabled():
        print(f"\n  {_BOLD}📝 Daily summaries{_RESET}")
        print(f"  {_DIM}An LLM reads your day's entries and writes a 2-3 sentence{_RESET}")
        print(f"  {_DIM}summary. Two options:{_RESET}")
        print(f"     {_DIM}• Local (runs on your Mac, private, ~free, requires mlx-lm){_RESET}")
        print(f"     {_DIM}• Cloud (OpenAI/Anthropic, higher quality, data leaves your Mac){_RESET}")
        if _ask("Enable daily summaries (local)?", default="n"):
            patch["summaries"] = {"provider": "local"}
            print(f"    {_GREEN}✓{_RESET} Local summaries on. First run will download a ~4 GB model.")
            print(f"    {_DIM}Install mlx-lm: ~/.voiceclip/.venv/bin/pip install mlx-lm{_RESET}")
        else:
            print(f"    {_DIM}Skipped. Enable in config under summaries.provider.{_RESET}")

    return patch


def _has_history_enabled() -> bool:
    """Detect whether history is already on in the persisted config."""
    path = Path(config.CONFIG_PATH)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return bool(data.get("history"))
    except (OSError, json.JSONDecodeError):
        return False


def _step_whats_next(patch_applied: dict):
    _header(5, 5, "You're set up")
    print("  The three commands worth remembering:\n")
    print(f"    {_BLUE}voiceclip{_RESET}           {_DIM}Start dictation (run this once, keep it running){_RESET}")
    print(f"    {_BLUE}voiceclip view{_RESET}      {_DIM}Open the journal in your browser{_RESET}")
    print(f"    {_BLUE}voiceclip doctor{_RESET}    {_DIM}Check your setup if anything feels broken{_RESET}")

    if patch_applied:
        applied_keys = ", ".join(sorted(patch_applied))
        print(f"\n  {_DIM}Saved to config: {applied_keys}{_RESET}")
        print(f"  {_DIM}Edit anytime: ~/.voiceclip/config.json{_RESET}")

    print(f"\n  {_BOLD}Now go dictate something.{_RESET}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(force: bool = False) -> bool:
    """Run the onboarding flow. Returns True if completed, False if skipped
    or failed.

    `force=True` ignores the done marker so `voiceclip onboard` re-runs.
    """
    if not _IS_TTY:
        return False
    if not force and not needs_onboarding():
        return False

    # Make sure config is loaded (sets CONFIG_PATH etc.)
    config.load()

    try:
        _step_welcome()
        _rule()
        _step_permissions()
        _rule()
        _step_try_it()
        _rule()
        patch = _step_features()
        if patch:
            _write_config_patch(patch)
        _rule()
        _step_whats_next(patch)
        mark_done()
        return True
    except SystemExit:
        # User hit Ctrl+C or typed 's' at a hard stop — respect it
        mark_done()  # don't nag on next launch
        print(f"\n  {_DIM}Onboarding skipped. Run `voiceclip onboard` anytime to revisit.{_RESET}\n")
        return False
    except Exception as e:
        log.warning("Onboarding error: %s", e)
        # Never let onboarding break startup
        mark_done()
        return False
