"""Global hotkey handler — configurable key and mode.

Supports two modes:
- "hold": Hold key to record, release to stop and transcribe (default)
- "toggle": Press once to start recording, press again to stop and transcribe

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

from voiceclip.config import MIN_HOLD_SECONDS, HOTKEY_MODE, HISTORY_ENABLED, resolve_hotkey
from voiceclip.recorder import Recorder
from voiceclip.transcriber import transcribe
from voiceclip.formatter import format_text
from voiceclip.macos import (
    copy_to_clipboard, copy_paste_and_restore, paste,
    notify, beep,
)

log = logging.getLogger(__name__)


class HotkeyHandler:
    """Manages the hotkey lifecycle for both hold and toggle modes.

    Ensures only one record/transcribe cycle runs at a time and
    handles all error cases gracefully. All state flags are protected
    by a lock to prevent races between pynput and worker threads.
    """

    def __init__(self, recorder: Recorder):
        self._recorder = recorder
        self._lock = threading.Lock()
        self._active = False       # True while recording is active
        self._press_time = 0.0
        self._busy = False         # True while a transcribe cycle is running
        self._listener = None
        self._hotkey = resolve_hotkey()
        self._mode = HOTKEY_MODE   # "hold" or "toggle"

    def start(self):
        """Start listening for the hotkey."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        log.info("Hotkey listener started (mode=%s)", self._mode)

    def stop(self):
        """Stop listening."""
        if self._listener:
            self._listener.stop()

    def _key_matches(self, key) -> bool:
        """Check if the pressed/released key matches our configured hotkey."""
        return key == self._hotkey

    # ------------------------------------------------------------------
    # Press handler
    # ------------------------------------------------------------------

    def _on_press(self, key):
        if not self._key_matches(key):
            return

        if self._mode == "toggle":
            self._handle_toggle_press()
        else:
            self._handle_hold_press()

    def _handle_hold_press(self):
        """Hold mode: start recording on press."""
        with self._lock:
            if self._active or self._busy:
                return
            self._active = True
            self._press_time = time.time()

        threading.Thread(
            target=self._start_recording,
            daemon=True,
        ).start()

    def _handle_toggle_press(self):
        """Toggle mode: press once to start, press again to stop."""
        with self._lock:
            if self._busy:
                return

            if not self._active:
                # Start recording
                self._active = True
                self._press_time = time.time()
                threading.Thread(
                    target=self._start_recording,
                    daemon=True,
                ).start()
            else:
                # Stop recording
                self._active = False
                hold_time = time.time() - self._press_time

        # If we just stopped, process the recording
        if not self._active and hold_time > 0:
            log.info("Recording stopped (%.1fs), transcribing...", hold_time)
            beep("Pop")
            with self._lock:
                self._busy = True
            threading.Thread(
                target=self._stop_and_transcribe,
                daemon=True,
            ).start()

    # ------------------------------------------------------------------
    # Release handler
    # ------------------------------------------------------------------

    def _on_release(self, key):
        if not self._key_matches(key):
            return

        # Toggle mode handles everything in _on_press
        if self._mode == "toggle":
            return

        # Hold mode: stop recording on release
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

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

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

    def _stop_and_transcribe(self):
        """Stop recording, transcribe, copy+paste. Runs in a background thread."""
        try:
            path = self._recorder.end()
            if not path:
                log.warning("No audio captured")
                notify("VoiceClip", "No audio captured")
                return

            notify("VoiceClip", "🔄 Transcribing...")

            t0 = time.time()
            text = transcribe(path)
            elapsed = time.time() - t0

            raw_text = text  # Save before formatting

            if text:
                text = format_text(text)

            if text:
                # Save to history if enabled
                if HISTORY_ENABLED:
                    from voiceclip.history import save as save_history
                    save_history(raw_text or "", text, elapsed)

                copy_paste_and_restore(text)
                beep("Glass")
                preview = text[:150] + ("..." if len(text) > 150 else "")
                log.info("Copied %d chars in %.1fs", len(text), elapsed)
                log.info('Text: "%s"', preview)
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
