"""Global hotkey handler — hold Right Option (⌥) to record.

Threading model:
- pynput runs its own listener thread for key events
- _on_press and _on_release are called on that thread
- All blocking work (begin, end, transcribe) runs in background threads
  to avoid blocking the pynput listener
"""

import logging
import threading
import time

from pynput import keyboard

from voiceclip.config import MIN_HOLD_SECONDS
from voiceclip.recorder import Recorder
from voiceclip.transcriber import transcribe
from voiceclip.formatter import format_text
from voiceclip.polisher import is_available as polish_available, polish
from voiceclip.macos import (
    copy_to_clipboard, paste, select_and_replace, notify, beep,
)

log = logging.getLogger(__name__)


class HotkeyHandler:
    """Manages the hold-to-record hotkey lifecycle.

    Ensures only one record/transcribe cycle runs at a time and
    handles all error cases gracefully. All state flags are protected
    by a lock to prevent races between pynput and worker threads.
    """

    def __init__(self, recorder: Recorder):
        self._recorder = recorder
        self._lock = threading.Lock()
        self._active = False       # True while the key is held down
        self._press_time = 0.0
        self._busy = False         # True while a transcribe cycle is running
        self._listener = None

    def start(self):
        """Start listening for the hotkey."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        log.info("Hotkey listener started")

    def stop(self):
        """Stop listening."""
        if self._listener:
            self._listener.stop()

    def _on_press(self, key):
        if key != keyboard.Key.alt_r:
            return

        with self._lock:
            if self._active or self._busy:
                return
            self._active = True
            self._press_time = time.time()

        # Move begin() + beep() off the pynput thread so we never block
        # the keyboard listener (begin() can take up to 3s on timeout).
        threading.Thread(
            target=self._start_recording,
            daemon=True,
        ).start()

    def _start_recording(self):
        """Start the recorder and play the start sound. Runs in a worker thread."""
        try:
            self._recorder.begin()
        except RuntimeError as e:
            log.error("Failed to start recording: %s", e)
            with self._lock:
                self._active = False
            self._try_restart_recorder()
            return

        beep("Tink")
        log.info("Recording started")

    def _on_release(self, key):
        if key != keyboard.Key.alt_r:
            return

        with self._lock:
            if not self._active:
                return
            self._active = False
            hold_time = time.time() - self._press_time

        # Ignore accidental taps
        if hold_time < MIN_HOLD_SECONDS:
            log.info("Tap too short (%.1fs), ignoring", hold_time)
            self._discard_recording()
            return

        log.info("Recording stopped (%.1fs), transcribing...", hold_time)
        beep("Pop")

        with self._lock:
            self._busy = True

        threading.Thread(
            target=self._stop_and_transcribe,
            daemon=True,
        ).start()

    def _stop_and_transcribe(self):
        """Stop recording, transcribe, copy+paste. Runs in a background thread.

        If LLM polish is enabled (VOICECLIP_POLISH=true), uses the
        paste-first-polish-after pattern:
        1. Paste raw formatted text immediately
        2. Run LLM polish in background
        3. Undo + re-paste with polished text
        """
        try:
            path = self._recorder.end()
            if not path:
                log.warning("No audio captured")
                notify("VoiceClip", "No audio captured")
                return

            # Let the user know we're working on it
            notify("VoiceClip", "🔄 Transcribing...")

            t0 = time.time()
            text = transcribe(path)
            elapsed = time.time() - t0

            if text:
                # Apply regex formatting and dictionary substitutions
                text = format_text(text)

            if text:
                # Step 1: Paste immediately (raw formatted text)
                copy_to_clipboard(text)
                paste()
                beep("Glass")
                preview = text[:150] + ("..." if len(text) > 150 else "")
                log.info("Copied %d chars in %.1fs", len(text), elapsed)
                log.info('Text: "%s"', preview)

                # Step 2: If LLM polish is enabled, polish in background
                # and replace the pasted text
                if polish_available():
                    self._polish_and_replace(text)
                else:
                    notify("VoiceClip ✅", text[:100])
            else:
                log.warning("No speech detected")
                notify("VoiceClip", "No speech detected")

        except RuntimeError as e:
            log.error("Recorder error: %s", e)
            self._try_restart_recorder()
        except Exception as e:
            log.error("Unexpected error: %s", e)
        finally:
            with self._lock:
                self._busy = False

    def _polish_and_replace(self, original_text):
        """Run LLM polish and replace the pasted text if improved."""
        try:
            t0 = time.time()
            polished = polish(original_text)
            elapsed = time.time() - t0

            if polished and polished != original_text:
                select_and_replace(original_text, polished)
                beep("Morse")  # Subtle sound to indicate polish applied
                log.info("Polished in %.1fs: \"%s\"", elapsed, polished[:150])
                notify("VoiceClip ✨", polished[:100])
            else:
                # No improvement or polish failed — original already pasted
                log.info("Polish: no changes (%.1fs)", elapsed)
                notify("VoiceClip ✅", original_text[:100])
        except Exception as e:
            log.error("Polish failed: %s", e)
            notify("VoiceClip ✅", original_text[:100])

    def _discard_recording(self):
        """Clean up a too-short recording in the background."""
        def _do():
            try:
                self._recorder.end()
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def _try_restart_recorder(self):
        """Attempt to restart the recorder after a failure."""
        try:
            self._recorder.restart()
            log.info("Recorder restarted successfully")
        except Exception as e:
            log.error("Failed to restart recorder: %s", e)
            notify("VoiceClip ❌", "Recorder crashed. Restart VoiceClip.")
