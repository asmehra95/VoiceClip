"""Global hotkey handler — hold Right Option (⌥) to record."""

import logging
import threading
import time

from pynput import keyboard

from voiceclip.config import MIN_HOLD_SECONDS
from voiceclip.recorder import Recorder
from voiceclip.transcriber import transcribe
from voiceclip.macos import copy_to_clipboard, paste, notify, beep

log = logging.getLogger(__name__)


class HotkeyHandler:
    """Manages the hold-to-record hotkey lifecycle.

    Ensures only one record/transcribe cycle runs at a time and
    handles all error cases gracefully.
    """

    def __init__(self, recorder: Recorder):
        self._recorder = recorder
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
        if key == keyboard.Key.alt_r and not self._active and not self._busy:
            self._active = True
            self._press_time = time.time()

            try:
                self._recorder.begin()
            except RuntimeError as e:
                log.error("Failed to start recording: %s", e)
                self._active = False
                self._try_restart_recorder()
                return

            beep("Tink")
            log.info("Recording started")

    def _on_release(self, key):
        if key != keyboard.Key.alt_r or not self._active:
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

        self._busy = True
        threading.Thread(
            target=self._stop_and_transcribe,
            daemon=True,
        ).start()

    def _stop_and_transcribe(self):
        """Stop recording, transcribe, copy+paste. Runs in a background thread."""
        try:
            path = self._recorder.end()
            if not path:
                log.warning("No audio captured")
                notify("VoiceClip", "No audio captured")
                return

            t0 = time.time()
            text = transcribe(path)
            elapsed = time.time() - t0

            if text:
                copy_to_clipboard(text)
                paste()
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
