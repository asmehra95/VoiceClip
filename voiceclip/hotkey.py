"""Global hotkey handler — configurable key, mode, and profile.

Supports two modes:
- "hold": Hold key to record, release to stop and transcribe
- "toggle": Press once to start recording, press again to stop

Supports two profiles:
- "transcription" (default): record → transcribe → paste + save to history
- "reflection": record → transcribe → save to history ONLY (no clipboard, no paste)

When both profiles are registered they share a module-level coordinator lock
so the single Recorder child process can only serve one cycle at a time.

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

from voiceclip.config import HISTORY_ENABLED, MIN_HOLD_SECONDS
from voiceclip.engine_cloud_stream import StreamingSession
from voiceclip.formatter import format_text
from voiceclip.macos import (
    beep,
    copy_paste_and_restore,
    get_active_app_name,
    notify,
)
from voiceclip.recorder import Recorder
from voiceclip.transcriber import TranscriptionError, transcribe

log = logging.getLogger(__name__)


# Module-level coordinator lock: shared across ALL HotkeyHandler instances.
# Ensures the single Recorder is never asked to serve two concurrent cycles.
_RECORDING_GATE = threading.Lock()


class _SharedListener:
    """One pynput ``keyboard.Listener`` shared by every HotkeyHandler.

    Why: each pynput listener spins up its own Darwin thread that calls
    into HIToolbox (``TISCreateCurrentKeyboardInputSourceRef``). With
    three concurrent listeners macOS aborts the process (SIGABRT in
    HIToolbox) during startup. One real listener fanning events out to
    every registered handler sidesteps the crash entirely — and is what
    pynput recommends anyway.

    Handlers register/unregister under a lock; the underlying listener
    is created on first register and stopped when the last handler
    leaves. A handler that raises inside its press/release callback is
    isolated so it can never kill dispatch for the others.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._handlers: list[HotkeyHandler] = []
        self._listener = None

    def register(self, handler: "HotkeyHandler"):
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)
            if self._listener is None:
                self._listener = keyboard.Listener(
                    on_press=self._on_press,
                    on_release=self._on_release,
                )
                self._listener.start()

    def unregister(self, handler: "HotkeyHandler"):
        with self._lock:
            try:
                self._handlers.remove(handler)
            except ValueError:
                pass
            if not self._handlers and self._listener is not None:
                listener, self._listener = self._listener, None
                listener.stop()

    @property
    def listener(self):
        """The live pynput listener (or None) — for health validation."""
        with self._lock:
            return self._listener

    def _snapshot(self):
        with self._lock:
            return list(self._handlers)

    def _on_press(self, key):
        for handler in self._snapshot():
            try:
                handler._on_press(key)
            except Exception:
                log.exception(
                    "Hotkey press handler failed (profile=%s)",
                    handler._profile,
                )

    def _on_release(self, key):
        for handler in self._snapshot():
            try:
                handler._on_release(key)
            except Exception:
                log.exception(
                    "Hotkey release handler failed (profile=%s)",
                    handler._profile,
                )


_SHARED_LISTENER = _SharedListener()


class HotkeyHandler:
    """Manages a single hotkey's lifecycle (hold or toggle mode, any profile).

    Multiple instances can coexist — they coordinate via the module-level
    `_RECORDING_GATE` lock so only one cycle runs at a time.
    """

    def __init__(
        self,
        recorder: Recorder,
        *,
        hotkey,
        mode: str = "hold",
        profile: str = "transcription",
        start_sound: str = "Tink",
        done_sound: str = "Glass",
        label: str = "VoiceClip",
    ):
        self._recorder = recorder
        self._hotkey = hotkey
        self._mode = mode
        self._profile = profile
        self._start_sound = start_sound
        self._done_sound = done_sound
        self._label = label

        self._lock = threading.Lock()
        self._active = False       # True while recording is active
        self._press_time = 0.0
        self._busy = False         # True while a transcribe cycle is running
        self._listener = None
        self._captured_app: str | None = None  # set at recording start
        # Streaming transcription state (one session per dictation).
        # _gen pairs a session with the dictation it was started for:
        # try_start() can block seconds on a slow connect, and a quick
        # dictation can be released before it returns — assigning the late
        # session unguarded would leak it AND let the NEXT dictation
        # consume a stale session.
        self._stream = None
        self._gen = 0
        self._pump_done = threading.Event()
        self._pump_stop = threading.Event()

    def start(self):
        """Start listening for the hotkey (via the shared listener)."""
        _SHARED_LISTENER.register(self)
        # Expose the (shared) listener so health validation can inspect
        # .running / IS_TRUSTED exactly as before.
        self._listener = _SHARED_LISTENER.listener
        log.info(
            "Hotkey registered (profile=%s, mode=%s)",
            self._profile, self._mode,
        )

    def stop(self):
        """Stop listening (shared listener stops when the last handler leaves)."""
        _SHARED_LISTENER.unregister(self)
        self._listener = None

    def _key_matches(self, key) -> bool:
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
        if self._profile == "assistant":
            from voiceclip.assistant import get_assistant
            if get_assistant().interrupt():
                log.info("Assistant interrupted")
                return
        with self._lock:
            if self._active or self._busy:
                return
            # Try to claim the shared recorder. If another handler is busy,
            # silently ignore the press.
            if not _RECORDING_GATE.acquire(blocking=False):
                log.info("Recorder busy with another profile; press ignored")
                return
            self._active = True
            self._press_time = time.time()

        threading.Thread(target=self._start_recording, daemon=True).start()

    def _handle_toggle_press(self):
        stopped = False
        hold_time = 0.0

        with self._lock:
            if self._busy:
                return

            if not self._active:
                if not _RECORDING_GATE.acquire(blocking=False):
                    log.info("Recorder busy with another profile; press ignored")
                    return
                self._active = True
                self._press_time = time.time()
                threading.Thread(target=self._start_recording, daemon=True).start()
            else:
                self._active = False
                hold_time = time.time() - self._press_time
                stopped = True

        if stopped and hold_time > 0:
            log.info("Recording stopped (%.1fs), transcribing...", hold_time)
            beep("Pop")
            with self._lock:
                self._busy = True
            threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    # ------------------------------------------------------------------
    # Release handler
    # ------------------------------------------------------------------

    def _on_release(self, key):
        if not self._key_matches(key):
            return
        if self._mode == "toggle":
            return

        with self._lock:
            if not self._active:
                return
            self._active = False
            hold_time = time.time() - self._press_time

        if hold_time < MIN_HOLD_SECONDS:
            log.info("Tap too short (%.1fs), ignoring", hold_time)
            self._discard_recording()
            # Release the coordinator gate since we never finish a cycle
            self._release_gate()
            return

        log.info("Recording stopped (%.1fs), transcribing...", hold_time)
        beep("Pop")

        with self._lock:
            self._busy = True

        threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def _start_recording(self):
        """Start the recorder, capture context, play start sound."""
        try:
            self._recorder.begin()
        except RuntimeError as e:
            log.error("Failed to start recording: %s", e)
            with self._lock:
                self._active = False
            self._release_gate()
            self._try_restart_recorder()
            return

        # Capture active-app context for history (best-effort, never blocks > 1s).
        # Only bothered when history is on — otherwise it's wasted work.
        if HISTORY_ENABLED:
            try:
                self._captured_app = get_active_app_name()
            except Exception:
                self._captured_app = None
        else:
            self._captured_app = None

        beep(self._start_sound)
        log.info("Recording started (profile=%s)", self._profile)

        # Streaming transcription (cloud engine only): open a realtime
        # session and pump live audio to it while the key is held, so the
        # transcript is nearly complete by release. Failure at any point
        # falls back to the batch path — the WAV is recorded regardless.
        with self._lock:
            self._stream = None
            gen = self._gen
        if self._recorder.tap is not None:
            self._recorder.drain_tap()
            # The pump ALWAYS runs while recording — even when no streaming
            # session materializes. The recorder child blocks writing tap
            # audio into a full pipe and then can't ack the stop command
            # ("Recorder timed out"), so the tap must be drained no matter
            # what. Chunks seen before the adoption decision are buffered
            # so a slow-connecting session still gets the head of speech.
            adoption = {"session": None}
            decided = threading.Event()
            self._pump_done.clear()
            self._pump_stop.clear()
            threading.Thread(
                target=self._pump_tap, args=(adoption, decided),
                daemon=True,
            ).start()
            session = StreamingSession.try_start()
            if session is not None:
                adopted = False
                with self._lock:
                    # Adopt only if this dictation is still current and
                    # still recording — otherwise the session arrived after
                    # release and must not leak to the next dictation.
                    if gen == self._gen and self._active:
                        self._stream = session
                        adopted = True
                if adopted:
                    adoption["session"] = session
                else:
                    log.debug("Streaming session arrived after release; aborted")
                    session.abort()
            decided.set()

    def _pump_tap(self, adoption, decided):
        """Drain live tap audio until the end-of-recording marker (or a
        stop/error). Runs for EVERY dictation, session or not: an unread
        tap pipe fills up and blocks the recorder child, which then can't
        ack the stop command ("Recorder timed out").

        Chunks arriving before the adoption decision (`decided` set,
        session in the `adoption` box or None) are buffered and replayed
        into the session once adopted, so a slow connect never clips the
        head of the speech. If no session is adopted the backlog is
        dropped and the pump keeps draining to nowhere."""
        tap = self._recorder.tap
        session = None
        backlog = []
        waiting = True
        try:
            while not self._pump_stop.is_set():
                if waiting and decided.is_set():
                    waiting = False
                    session = adoption["session"]
                    if session is not None:
                        for chunk in backlog:
                            session.feed(chunk)
                    backlog = []
                if not tap.poll(0.2):
                    continue
                data = tap.recv_bytes()
                if data == b"":
                    break  # end-of-recording marker
                if session is not None:
                    session.feed(data)
                elif waiting:
                    backlog.append(data)
        except (EOFError, OSError):
            pass
        finally:
            self._pump_done.set()

    def _stop_and_transcribe(self):
        """Stop recording, transcribe, deliver result.

        Three failure modes the user should know about:
          1. Recorder produced no audio (e.g. mic disconnected) — notified.
          2. Transcription raised TranscriptionError (timeout, model crash)
             — notified with the specific error + doctor pointer.
          3. Recorder raised RuntimeError (pipe broken, child died) —
             notified and restart attempted.

        "No speech detected" is distinct from the above — that's the empty
        None return, and we notify with neutral copy since it's usually
        user-initiated (quiet tap, mic muted, background noise only).
        """
        try:
            with self._lock:
                self._gen += 1
                stream, self._stream = self._stream, None
            path = self._recorder.end()
            if not path:
                log.warning("No audio captured")
                self._pump_stop.set()
                if stream is not None:
                    stream.abort()
                notify(
                    f"{self._label} ❌",
                    "No audio captured — mic disconnected? Run voiceclip doctor",
                )
                beep("Funk")
                return

            t0 = time.time()
            text = None
            if stream is not None:
                # Single time budget for the whole streaming stop-path so
                # the worst case stays bounded (pump wait + finish); batch
                # fallback has its own dispatcher timeout on top.
                budget = 8.0
                self._pump_done.wait(timeout=min(2.0, budget))
                self._pump_stop.set()
                remaining = max(1.0, budget - (time.time() - t0))
                text = stream.finish(timeout=remaining)
                if text:
                    log.info("Streamed transcription (%.2fs after release)",
                             time.time() - t0)
                else:
                    log.info("Streaming produced nothing; using batch path")
            else:
                # Drain-only pump: the end marker (sent by recorder.end()
                # above) lets it exit on its own, but be explicit so a
                # lost marker can't leave it running into the next cycle.
                self._pump_stop.set()
            if not text:
                try:
                    text = transcribe(path)
                except TranscriptionError as e:
                    # Whisper timed out or crashed. Surface the specific
                    # reason + the doctor pointer that's already baked into
                    # the error message.
                    log.error("Transcription failed: %s", e)
                    notify(f"{self._label} ❌", str(e)[:120])
                    beep("Funk")
                    return
            elapsed = time.time() - t0

            raw_text = text  # Save before formatting

            if text:
                text = format_text(text)

            if text:
                self._deliver(raw_text or "", text, elapsed)
            else:
                log.warning("No speech detected")
                notify(self._label, "No speech detected")

        except RuntimeError as e:
            log.error("Recorder error: %s", e)
            notify(
                f"{self._label} ❌",
                "Recorder error — attempting restart. Run voiceclip doctor if this persists.",
            )
            beep("Funk")
            self._try_restart_recorder()
        except Exception as e:
            log.error("Unexpected error: %s", e)
            notify(
                f"{self._label} ❌",
                "Unexpected error — run voiceclip doctor",
            )
            beep("Funk")
        finally:
            with self._lock:
                self._busy = False
            self._release_gate()

    def _deliver(self, raw_text: str, text: str, elapsed: float):
        """Profile-specific delivery: paste-and-save vs save-only vs polish-then-paste."""
        app_name = self._captured_app
        self._captured_app = None  # reset

        if self._profile == "reflection":
            # Save-only. No clipboard write, no paste, no clipboard restore.
            if HISTORY_ENABLED:
                from voiceclip.history import save as save_history
                save_history(
                    raw_text, text, elapsed,
                    kind="reflection",
                    app_name=app_name,
                )
            beep(self._done_sound)
            preview = text[:50] + ("..." if len(text) > 50 else "")
            log.info("Reflection saved (%d chars, %.1fs): %r", len(text), elapsed, preview)
            notify(f"{self._label} ✍️", "Reflection saved")
            return

        if self._profile == "assistant":
            # Ask the journal-aware companion; it speaks the reply. The
            # question and answer are both saved to history so the
            # conversation itself becomes part of the journal.
            from voiceclip.assistant import AssistantError, get_assistant
            try:
                reply = get_assistant().respond(text)
            except AssistantError as e:
                log.error("Assistant failed: %s", e)
                notify(f"{self._label} ❌", str(e)[:120])
                beep("Funk")
                return
            if HISTORY_ENABLED:
                from voiceclip.history import save as save_history
                save_history(raw_text, text, elapsed, kind="question",
                             app_name=app_name)
                save_history(reply, reply, 0.0, kind="assistant",
                             app_name=app_name)
            notify(self._label, reply[:200])
            log.info('Q: "%s"', text[:120])
            log.info('A: "%s"', reply[:160])
            return

        if self._profile == "polished":
            # Run through LLM for cleanup before pasting.
            from voiceclip.polisher import polish
            polished = polish(text)
            if HISTORY_ENABLED:
                from voiceclip.history import save as save_history
                save_history(
                    raw_text, polished, elapsed,
                    kind="transcription",
                    app_name=app_name,
                )
            copy_paste_and_restore(polished)
            beep(self._done_sound)
            preview = polished[:150] + ("..." if len(polished) > 150 else "")
            log.info("Polished %d→%d chars in %.1fs", len(text), len(polished), elapsed)
            log.info('Text: "%s"', preview)
            return

        # Default: transcription profile — paste raw formatted text.
        if HISTORY_ENABLED:
            from voiceclip.history import save as save_history
            save_history(
                raw_text, text, elapsed,
                kind="transcription",
                app_name=app_name,
            )
        copy_paste_and_restore(text)
        beep(self._done_sound)
        preview = text[:150] + ("..." if len(text) > 150 else "")
        log.info("Copied %d chars in %.1fs", len(text), elapsed)
        log.info('Text: "%s"', preview)

    def _discard_recording(self):
        """Clean up a too-short recording in the background."""
        with self._lock:
            self._gen += 1
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.abort()
        self._pump_stop.set()

        def _do():
            try:
                self._recorder.end()
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def _release_gate(self):
        """Release the shared recorder lock if we hold it."""
        try:
            _RECORDING_GATE.release()
        except RuntimeError:
            # Not held — safe to ignore
            pass

    def _try_restart_recorder(self):
        try:
            self._recorder.restart()
            log.info("Recorder restarted successfully")
        except Exception as e:
            log.error("Failed to restart recorder: %s", e)
            notify(
                f"{self._label} ❌",
                "Recorder crashed. Restart VoiceClip, then run voiceclip doctor.",
            )
