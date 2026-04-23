"""macOS-specific utilities: clipboard, paste, notifications, sounds, permissions.

Performance notes:
- notify() and paste() use Popen (fire-and-forget, never block)
- Sound file paths are cached after first check
- PASTE_DELAY reduced to 50ms (sufficient for most apps)
"""

import logging
import os
import subprocess
import threading
import time

from voiceclip.config import PASTE_DELAY

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permission checks
# ---------------------------------------------------------------------------


def check_accessibility():
    """Check if the terminal has Accessibility permission.

    Returns True if granted, False otherwise. Logs a warning with
    instructions if not granted.
    """
    try:
        # osascript keystroke will fail silently without Accessibility,
        # but we can check via the system_profiler or a test keystroke.
        # The most reliable lightweight check: try to create an
        # AppleScript System Events reference.
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to return name of first process'],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            log.warning(
                "⚠️  Accessibility permission not granted. "
                "Auto-paste won't work.\n"
                "   Fix: System Settings → Privacy & Security → Accessibility "
                "→ add your terminal app"
            )
            return False
        return True
    except Exception:
        # Can't determine — assume it's fine
        return True


def check_microphone():
    """Check if microphone access is likely available.

    This is a best-effort check. The actual permission dialog is shown
    by macOS when sounddevice first opens the mic in the child process.
    We log a reminder so the user knows what to expect.
    """
    log.info(
        "ℹ️  If this is your first run, macOS will ask for Microphone "
        "permission. Grant it and restart VoiceClip if needed."
    )

# ---------------------------------------------------------------------------
# Clipboard — with save/restore to preserve user's clipboard
# ---------------------------------------------------------------------------


def _get_clipboard() -> str | None:
    """Read the current clipboard contents. Returns None on failure."""
    try:
        result = subprocess.run(
            ["pbpaste"], capture_output=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def copy_to_clipboard(text):
    """Copy text to the macOS clipboard via pbcopy."""
    try:
        proc = subprocess.Popen(
            ["pbcopy"], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=text.encode("utf-8"))
    except Exception as e:
        log.error("Clipboard error: %s", e)


def copy_paste_and_restore(text):
    """Copy text to clipboard, paste it, then restore the previous clipboard.

    This preserves whatever the user had copied before VoiceClip ran.
    The restore happens after a short delay to ensure the paste completes.
    """
    # Save what's currently on the clipboard
    previous = _get_clipboard()

    # Copy our text and paste it
    copy_to_clipboard(text)
    paste()

    # Restore the previous clipboard after paste completes
    if previous is not None:
        def _restore():
            time.sleep(0.5)  # Wait for paste to finish
            copy_to_clipboard(previous)
        threading.Thread(target=_restore, daemon=True).start()


def paste():
    """Simulate Cmd+V to paste into the active app. Non-blocking."""
    time.sleep(PASTE_DELAY)
    try:
        subprocess.Popen(
            ["osascript", "-e",
             'tell application "System Events" to keystroke "v" using command down'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def select_and_replace(old_text, new_text):
    """Replace previously pasted text with new text.

    Strategy: use Cmd+Z to undo the paste, then paste the new text.
    This is more reliable than trying to select exact character counts,
    which varies by app and font rendering.
    """
    try:
        # Undo the previous paste
        time.sleep(0.1)
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to keystroke "z" using command down'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=2,
        )
        time.sleep(0.1)

        # Copy new text and paste it
        copy_to_clipboard(new_text)
        time.sleep(PASTE_DELAY)
        subprocess.Popen(
            ["osascript", "-e",
             'tell application "System Events" to keystroke "v" using command down'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log.warning("select_and_replace failed: %s", e)


# ---------------------------------------------------------------------------
# Notifications — fire-and-forget
# ---------------------------------------------------------------------------


def notify(title, message):
    """Show a macOS notification. Non-blocking."""
    safe_msg = message.replace("\n", " ").replace("\r", " ")[:100]
    safe_msg = safe_msg.replace("\\", "\\\\").replace('"', '\\"')
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    try:
        subprocess.Popen(
            ["osascript", "-e",
             f'display notification "{safe_msg}" with title "{safe_title}"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# System sounds — cached paths, tracked for cleanup
# ---------------------------------------------------------------------------

_beep_lock = threading.Lock()
_beep_procs = []
_sound_cache = {}  # path string → bool (exists)


def beep(sound="Tink"):
    """Play a macOS system sound. Non-blocking, cached path lookup."""
    if sound not in _sound_cache:
        path = f"/System/Library/Sounds/{sound}.aiff"
        _sound_cache[sound] = path if os.path.exists(path) else None

    path = _sound_cache[sound]
    if not path:
        return

    try:
        with _beep_lock:
            _beep_procs[:] = [p for p in _beep_procs if p.poll() is None]
            proc = subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _beep_procs.append(proc)
    except OSError as e:
        log.warning("Could not play sound '%s': %s", sound, e)


def cleanup_sounds():
    """Terminate any still-playing sounds."""
    with _beep_lock:
        for p in _beep_procs:
            if p.poll() is None:
                p.terminate()
