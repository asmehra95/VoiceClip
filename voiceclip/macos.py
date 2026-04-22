"""macOS-specific utilities: clipboard, paste, notifications, sounds.

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
# Clipboard
# ---------------------------------------------------------------------------


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

    with _beep_lock:
        _beep_procs[:] = [p for p in _beep_procs if p.poll() is None]
        proc = subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _beep_procs.append(proc)


def cleanup_sounds():
    """Terminate any still-playing sounds."""
    with _beep_lock:
        for p in _beep_procs:
            if p.poll() is None:
                p.terminate()
