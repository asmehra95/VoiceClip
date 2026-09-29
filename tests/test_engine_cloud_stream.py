"""StreamingSession state-machine tests.

Uses a fake WebSocket injected straight into the constructor — no network.
Event sequences replicate the REAL patterns observed in production logs
(duplicate VAD events, gaps between stopped/committed, slow completions
during cold model loads), because clean synthetic sequences are exactly
what masked three shipped bugs.
"""
import json
import queue
import threading
import time

import pytest

from voiceclip.engine_cloud_stream import StreamingSession, _State


class FakeWS:
    """recv() blocks on a queue the test feeds; send() records payloads."""

    def __init__(self):
        self.inbox = queue.Queue()
        self.sent = []
        self.closed = False

    def send(self, payload):
        if self.closed:
            raise OSError("send on closed ws")
        self.sent.append(json.loads(payload))

    def recv(self):
        item = self.inbox.get()
        if item is None:
            raise OSError("connection closed")
        return json.dumps(item)

    def close(self):
        self.closed = True
        self.inbox.put(None)   # unblock the reader

    def settimeout(self, t):
        pass

    # -- test helpers --
    def push(self, etype, **kw):
        self.inbox.put({"type": etype, **kw})

    def sent_types(self):
        return [m["type"] for m in self.sent]


@pytest.fixture
def ws():
    return FakeWS()


@pytest.fixture
def session(ws):
    s = StreamingSession(ws)
    yield s
    s.abort()


def _wait_for(cond, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def finish_async(session, timeout=5.0):
    """Run finish() on a thread; return a getter for its result."""
    box = {}
    def run():
        box["text"] = session.finish(timeout=timeout)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    def result(join_timeout=6.0):
        t.join(join_timeout)
        assert not t.is_alive(), "finish() did not return"
        return box["text"]
    return result


class TestHappyPaths:
    def test_segments_during_hold_then_final_commit(self, ws, session):
        # Two VAD segments arrive while "holding"
        ws.push("input_audio_buffer.speech_started")
        ws.push("input_audio_buffer.committed")
        ws.push("conversation.item.input_audio_transcription.completed",
                transcript="first segment.")
        assert _wait_for(lambda: session._n_completed == 1)

        session.feed(b"\x00\x00" * 240)
        get = finish_async(session)
        # commit ack + final result
        assert _wait_for(lambda: {"type": "input_audio_buffer.commit"}
                         in [{"type": t} for t in ws.sent_types()])
        ws.push("input_audio_buffer.committed")
        ws.push("conversation.item.input_audio_transcription.completed",
                transcript="second segment.")
        assert get() == "first segment. second segment."
        assert session._state is _State.DONE

    def test_pause_before_release_empty_commit(self, ws, session):
        ws.push("input_audio_buffer.speech_started")
        ws.push("input_audio_buffer.committed")
        ws.push("conversation.item.input_audio_transcription.completed",
                transcript="all done already.")
        assert _wait_for(lambda: session._n_completed == 1)
        get = finish_async(session)
        # server: nothing left to commit
        ws.push("error", error={"message": "Error committing input audio "
                                           "buffer: buffer too small. ..."})
        assert get() == "all done already."


class TestRealWorldEventPatterns:
    def test_slow_completion_after_ack_is_awaited(self, ws, session):
        """Cold model load: ack arrives, completion much later. An earlier
        version bailed on a quiet-line here and dropped the final segment."""
        get = finish_async(session)
        ws.push("input_audio_buffer.committed")           # ack
        time.sleep(2.6)                                    # > old quiet_line
        ws.push("conversation.item.input_audio_transcription.completed",
                transcript="late but complete.")
        assert get() == "late but complete."

    def test_duplicate_committed_events_do_not_hang(self, ws, session):
        """Real speech produces duplicate committed bursts; each committed
        must pair with one completion or finish() waits forever (the 10s
        dictation bug)."""
        ws.push("input_audio_buffer.speech_started")
        for _ in range(3):
            ws.push("input_audio_buffer.committed")
        for i in range(3):
            ws.push("conversation.item.input_audio_transcription.completed",
                    transcript=f"seg{i}.")
        assert _wait_for(lambda: session._n_completed == 3)
        get = finish_async(session)
        ws.push("error", error={"message": "buffer too small"})
        assert get() == "seg0. seg1. seg2."

    def test_quiet_line_only_before_ack(self, ws, session):
        """Commit lost entirely: no ack, no events. finish() must give up
        after ~quiet_line (2s), NOT the full deadline."""
        ws.push("input_audio_buffer.speech_started")
        ws.push("input_audio_buffer.committed")
        ws.push("conversation.item.input_audio_transcription.completed",
                transcript="only segment.")
        assert _wait_for(lambda: session._n_completed == 1)
        t0 = time.time()
        get = finish_async(session, timeout=8.0)
        text = get()
        elapsed = time.time() - t0
        assert text == "only segment."
        assert elapsed < 4.0, f"quiet-line escape took {elapsed:.1f}s"

    def test_failed_segment_counts_as_done(self, ws, session):
        get = finish_async(session)
        ws.push("input_audio_buffer.committed")
        ws.push("conversation.item.input_audio_transcription.failed")
        assert get() is None   # nothing transcribed → batch fallback


class TestFailureModes:
    def test_server_error_event_fails_session(self, ws, session):
        ws.push("error", error={"message": "Not Found"})
        assert _wait_for(lambda: session._state is _State.FAILED)
        assert session.finish(timeout=1.0) is None

    def test_socket_death_fails_session(self, ws, session):
        ws.inbox.put(None)   # reader sees closed connection
        assert _wait_for(lambda: session._state is _State.FAILED)
        assert session.finish(timeout=1.0) is None

    def test_feed_after_failure_is_noop(self, ws, session):
        ws.push("error", error={"message": "Not Found"})
        assert _wait_for(lambda: session._state is _State.FAILED)
        sent_before = len(ws.sent)
        session.feed(b"\x00\x00" * 100)
        assert len(ws.sent) == sent_before

    def test_abort_closes_ws(self, ws, session):
        session.abort()
        assert ws.closed
        assert session.finish(timeout=0.5) is None

    def test_concurrent_feed_and_finish_serialized(self, ws, session):
        """ws.send must never interleave between pump-feed and the final
        commit (websocket-client is not thread-safe)."""
        stop = threading.Event()
        def pump():
            while not stop.is_set():
                session.feed(b"\x00\x00" * 60)
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        get = finish_async(session, timeout=2.0)
        ws.push("input_audio_buffer.committed")
        ws.push("conversation.item.input_audio_transcription.completed",
                transcript="ok.")
        assert get() == "ok."
        stop.set(); t.join(1)
        # every recorded send is a complete, valid JSON message
        assert all("type" in m for m in ws.sent)


class TestTryStartPreconditions:
    def test_none_when_engine_not_cloud(self, isolated_config, monkeypatch):
        from voiceclip import config
        config.load()
        monkeypatch.setattr(config, "ENGINE", "whisper_cpp")
        monkeypatch.setattr(config, "CLOUD_STREAMING", True)
        assert StreamingSession.try_start() is None

    def test_none_when_streaming_disabled(self, isolated_config, monkeypatch):
        from voiceclip import config
        config.load()
        monkeypatch.setattr(config, "ENGINE", "cloud")
        monkeypatch.setattr(config, "CLOUD_STREAMING", False)
        assert StreamingSession.try_start() is None

    def test_none_when_no_base_url(self, isolated_config, monkeypatch):
        from voiceclip import config
        config.load()
        monkeypatch.setattr(config, "ENGINE", "cloud")
        monkeypatch.setattr(config, "CLOUD_STREAMING", True)
        monkeypatch.setattr(config, "CLOUD_BASE_URL", "")
        assert StreamingSession.try_start() is None

    def test_none_on_connect_refused(self, isolated_config, monkeypatch):
        from voiceclip import config, tunnel
        config.load()
        monkeypatch.setattr(config, "ENGINE", "cloud")
        monkeypatch.setattr(config, "CLOUD_STREAMING", True)
        monkeypatch.setattr(config, "CLOUD_BASE_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(tunnel, "ensure", lambda: False)
        assert StreamingSession.try_start() is None
