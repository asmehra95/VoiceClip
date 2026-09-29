"""Streaming cloud dictation — transcribe WHILE the hotkey is held.

Opens a WebSocket to the server's OpenAI-compatible `/v1/realtime` endpoint
(speaches behind LiteLLM) when recording starts, streams PCM16 audio chunks
as they arrive from the mic, and collects per-segment transcripts that the
server produces at each natural speech pause (server-side VAD). On release,
the ENTIRE remaining buffer is committed manually — the same "transcribe
everything captured" semantics that make the local engine immune to tail
clipping. (Never let VAD decide which audio to keep; it only finds pause
boundaries. See DESIGN.md.)

Session lifecycle (driven by hotkey.py):
    s = StreamingSession.try_start()     # on hotkey press; None → use batch
    s.feed(pcm_bytes)                    # 24kHz mono PCM16, from recorder
    text = s.finish(timeout=...)         # on release; None → fall back

The session is an explicit state machine (STREAMING → FINISHING → DONE /
FAILED). All mutable state is guarded by one lock and transitions are
driven by server events — an earlier counters-and-flags design produced
three separate race bugs (early exit between VAD events, hangs on
duplicate events, results discarded during cold model loads).

Any failure leaves the session FAILED and finish() reports None — the
caller then batch-transcribes the recorded WAV. Streaming is an
optimization, never a correctness dependency.

Requires the `websocket-client` package (pure Python). If it's missing,
try_start() returns None and dictation silently uses the batch path.
"""

import base64
import enum
import json
import logging
import ssl
import threading
import time
import urllib.parse

log = logging.getLogger(__name__)

# The realtime endpoint consumes 24kHz mono PCM16 (OpenAI convention).
SAMPLE_RATE = 24000

# Hard cap on finish(); actual budget is set by the caller. Bounded so a
# wedged server can't stall dictation past what batch fallback would cost.
_DEFAULT_FINISH_TIMEOUT = 8.0


class _State(enum.Enum):
    STREAMING = "streaming"   # feeding audio; VAD segments per pause
    FINISHING = "finishing"   # final manual commit sent; awaiting results
    DONE = "done"
    FAILED = "failed"


class StreamingSession:
    """One dictation's realtime WebSocket session. Not reusable."""

    def __init__(self, ws):
        self._ws = ws
        # _lock guards ALL mutable state below; _send_lock serializes
        # ws.send() between the pump thread (feed) and finish()'s commit —
        # websocket-client sends are not thread-safe.
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._state = _State.STREAMING
        self._segments: list[str] = []
        self._closed = False
        # Server-event counters (guarded by _lock)
        self._n_committed = 0
        self._n_completed = 0
        self._committed_at_finish = None   # committed count when commit sent
        self._empty_commit_ack = False     # "buffer too small" on final commit
        self._last_event_at = time.time()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def try_start(cls, _retry: bool = True):
        """Open a realtime session. Returns None (never raises) when
        streaming isn't possible — batch path handles the dictation."""
        from voiceclip import config

        if config.ENGINE != "cloud" or not getattr(config, "CLOUD_STREAMING", False):
            return None
        try:
            import websocket
        except ImportError:
            log.info("websocket-client not installed; streaming disabled")
            return None

        base = (config.CLOUD_BASE_URL or "").strip().rstrip("/")
        if not base:
            return None
        parsed = urllib.parse.urlparse(base)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        # Vocabulary biasing: the same initial prompt the batch path uses
        # (persona + custom_vocabulary) rides along as a query param, which
        # the server turns into Whisper's initial_prompt (names, jargon).
        # A query param — not session.update — because LiteLLM's realtime
        # passthrough drops speaches' input_audio_transcription block but
        # forwards the query string intact. Server-side patch required;
        # an unpatched server ignores the unknown param harmlessly.
        prompt = (getattr(config, "INITIAL_PROMPT", None) or "").strip()
        # The prompt rides INSIDE the model token ("<model>|prompt=<enc>"):
        # it's the one field LiteLLM's realtime passthrough forwards
        # verbatim (it rebuilds the query string from a whitelist and
        # rewrites session.update). Server decodes it; a LiteLLM wildcard
        # route ("Systran/*") accepts the suffixed name.
        model_token = config.CLOUD_REALTIME_MODEL
        if prompt:
            model_token += "|prompt=" + urllib.parse.quote(prompt, safe="")
        # Language rides the token too: LiteLLM drops ?language= (it
        # rebuilds the query string from a whitelist), which silently
        # re-enabled auto-detect — short utterances came back in the
        # wrong script entirely. The query param stays for gateway-less
        # (local dev) setups; the server prefers the token value.
        if config.ENGLISH_ONLY:
            model_token += "|lang=en"
        query = urllib.parse.urlencode({
            "model": model_token,
            "intent": "transcription",
            **({"language": "en"} if config.ENGLISH_ONLY else {}),
        })
        url = f"{ws_scheme}://{parsed.netloc}{parsed.path}/v1/realtime?{query}"

        headers = []
        key = (config.CLOUD_API_KEY or "").strip()
        if key:
            headers.append(f"Authorization: Bearer {key}")

        sslopt = None
        if ws_scheme == "wss":
            ca = (getattr(config, "CLOUD_CA_BUNDLE", "") or "").strip()
            if ca:
                import os as _os
                ctx = ssl.create_default_context(cafile=_os.path.expanduser(ca))
                sslopt = {"context": ctx}

        try:
            ws = websocket.create_connection(
                url, header=headers, timeout=5,
                **({"sslopt": sslopt} if sslopt else {}),
            )
            first = json.loads(ws.recv())
            if first.get("type") != "session.created":
                log.warning("Realtime session: unexpected first event %s",
                            first.get("type"))
                ws.close()
                return None
        except Exception as e:
            # The tunnel may have dropped (laptop sleep, instance restart).
            # Try to bring it back once, then retry the connection once.
            if _retry:
                try:
                    from voiceclip import tunnel
                    if tunnel.ensure():
                        return cls.try_start(_retry=False)
                except Exception:
                    pass
            log.info("Streaming unavailable (%s); using batch path", e)
            return None

        ws.settimeout(10)

        log.debug("Streaming session started")
        return cls(ws)

    # -- data path ----------------------------------------------------------

    def feed(self, pcm: bytes):
        """Send a chunk of 24kHz mono PCM16. Cheap; never raises."""
        with self._lock:
            if self._state in (_State.DONE, _State.FAILED) or not pcm:
                return
        try:
            payload = json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            })
            with self._send_lock:
                self._ws.send(payload)
        except Exception as e:
            log.warning("Streaming feed failed (%s); will fall back", e)
            self._fail()

    def finish(self, timeout: float = _DEFAULT_FINISH_TIMEOUT) -> str | None:
        """Commit the remaining buffer, wait for results, return transcript.

        Returns None when the session failed or produced nothing — the
        caller falls back to batch transcription of the recorded WAV.
        """
        with self._lock:
            if self._state is _State.FAILED:
                self._close_locked()
                return None
            if self._state is _State.STREAMING:
                self._state = _State.FINISHING
                self._committed_at_finish = self._n_committed

        # Manual commit of everything still buffered server-side. If VAD
        # already committed it all (user paused before release), the server
        # answers "buffer too small" — that's the ack, not a failure.
        try:
            with self._send_lock:
                self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as e:
            log.warning("Final commit failed (%s)", e)
            self._fail()
            with self._lock:
                self._close_locked()
                return None
        flushed_at = time.time()

        # quiet_line guards only the UNACKED phase (commit lost / server
        # wedged). Once acked, the transcription WILL arrive — bailing
        # early there discards the final segment (e.g. cold model load).
        quiet_line = 2.0
        deadline = time.time() + timeout
        with self._changed:
            while self._state is _State.FINISHING:
                self._recompute_done_locked()
                if self._state is not _State.FINISHING:
                    break
                now = time.time()
                if now >= deadline:
                    log.warning("finish(): deadline hit — taking %d segment(s)",
                                len(self._segments))
                    break
                if not self._commit_acked_locked() \
                        and now - max(self._last_event_at, flushed_at) >= quiet_line:
                    log.debug("finish(): commit never acknowledged — taking "
                              "%d segment(s)", len(self._segments))
                    break
                self._changed.wait(timeout=min(0.1, max(0.01, deadline - now)))
            self._close_locked()
            if self._state is _State.FAILED and not self._segments:
                return None
            text = " ".join(s.strip() for s in self._segments if s.strip())
        return text or None

    def abort(self):
        """Tear down without waiting (recording rejected, app quitting)."""
        self._fail()
        with self._lock:
            self._close_locked()

    # -- internals ----------------------------------------------------------

    def _commit_acked_locked(self) -> bool:
        return (self._empty_commit_ack
                or (self._committed_at_finish is not None
                    and self._n_committed > self._committed_at_finish))

    def _recompute_done_locked(self):
        """FINISHING → DONE once the final commit is acknowledged and every
        committed buffer has produced its transcription result."""
        if self._state is _State.FINISHING and self._commit_acked_locked() \
                and self._n_completed >= self._n_committed:
            self._state = _State.DONE

    def _fail(self):
        with self._changed:
            if self._state is not _State.DONE:
                self._state = _State.FAILED
            self._changed.notify_all()

    def _read_loop(self):
        while True:
            with self._lock:
                if self._closed:
                    return
            try:
                ev = json.loads(self._ws.recv())
            except Exception:
                with self._lock:
                    closed = self._closed
                if not closed:
                    self._fail()
                return
            etype = ev.get("type", "")
            with self._changed:
                self._last_event_at = time.time()
                if etype.endswith("input_audio_transcription.completed"):
                    self._segments.append(ev.get("transcript") or "")
                    self._n_completed += 1
                elif etype.endswith("input_audio_transcription.failed"):
                    # Count as done so finish() doesn't wait forever for a
                    # transcript that will never come.
                    self._n_completed += 1
                    log.warning("Realtime segment transcription failed")
                elif etype == "input_audio_buffer.committed":
                    self._n_committed += 1
                elif etype == "error":
                    msg = (ev.get("error") or {}).get("message", "")
                    # Benign: the final manual commit when everything was
                    # already committed yields "buffer too small" — that's
                    # the ack that nothing is pending, not a failure.
                    if "buffer too small" in msg:
                        self._empty_commit_ack = True
                    else:
                        log.warning("Realtime session error: %s", msg[:200])
                        if self._state is not _State.DONE:
                            self._state = _State.FAILED
                        self._changed.notify_all()
                        return
                self._recompute_done_locked()
                self._changed.notify_all()

    def _close_locked(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._ws.close()
        except Exception:
            pass
