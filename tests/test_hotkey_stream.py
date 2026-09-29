"""Stop-path orchestration tests for streaming dictation in HotkeyHandler.

No pynput listener is started; handlers are driven by calling the
recording-lifecycle methods directly with fakes for the recorder and
StreamingSession.
"""
import threading
import time

import pytest

from voiceclip import hotkey as hotkey_mod
from voiceclip.hotkey import HotkeyHandler


class FakeTap:
    def __init__(self):
        self._chunks = []
        self._lock = threading.Lock()

    def poll(self, timeout=0):
        time.sleep(min(timeout, 0.01))
        with self._lock:
            return bool(self._chunks)

    def recv_bytes(self):
        with self._lock:
            return self._chunks.pop(0)

    def push(self, data):
        with self._lock:
            self._chunks.append(data)


class FakeRecorder:
    def __init__(self, wav_path="/tmp/fake.wav"):
        self.tap = FakeTap()
        self.wav_path = wav_path
        self.end_delay = 0.0

    def begin(self):
        pass

    def end(self):
        time.sleep(self.end_delay)
        self.tap.push(b"")   # recorder sends end-marker on stop
        return self.wav_path

    def drain_tap(self):
        pass


class FakeSession:
    def __init__(self, text="streamed text"):
        self.text = text
        self.fed = []
        self.aborted = False

    def feed(self, pcm):
        self.fed.append(pcm)

    def finish(self, timeout=None):
        return self.text

    def abort(self):
        self.aborted = True


@pytest.fixture
def handler(isolated_config, monkeypatch):
    from voiceclip import config
    config.load()
    h = HotkeyHandler(
        FakeRecorder(), hotkey="f13", mode="hold", profile="transcription",
    )
    # keep the test away from macOS APIs and real transcription
    monkeypatch.setattr(hotkey_mod, "beep", lambda *a, **k: None)
    monkeypatch.setattr(hotkey_mod, "notify", lambda *a, **k: None)
    monkeypatch.setattr(hotkey_mod, "get_active_app_name", lambda: None)
    monkeypatch.setattr(hotkey_mod, "format_text", lambda t: t)
    monkeypatch.setattr(hotkey_mod, "copy_paste_and_restore", lambda t: None)
    return h


def test_streamed_text_used_no_batch(handler, monkeypatch):
    session = FakeSession("hello from stream")
    monkeypatch.setattr(hotkey_mod.StreamingSession, "try_start",
                        classmethod(lambda cls, _retry=True: session))
    batch_calls = []
    monkeypatch.setattr(hotkey_mod, "transcribe",
                        lambda p: batch_calls.append(p) or "batch text")
    delivered = {}
    monkeypatch.setattr(handler, "_deliver",
                        lambda raw, text, el: delivered.update(text=text))
    with handler._lock:
        handler._active = True
    handler._start_recording()
    handler._stop_and_transcribe()
    assert delivered["text"] == "hello from stream"
    assert not batch_calls


def test_batch_fallback_when_stream_returns_none(handler, monkeypatch):
    session = FakeSession(text=None)
    monkeypatch.setattr(hotkey_mod.StreamingSession, "try_start",
                        classmethod(lambda cls, _retry=True: session))
    monkeypatch.setattr(hotkey_mod, "transcribe", lambda p: "batch text")
    delivered = {}
    monkeypatch.setattr(handler, "_deliver",
                        lambda raw, text, el: delivered.update(text=text))
    with handler._lock:
        handler._active = True
    handler._start_recording()
    handler._stop_and_transcribe()
    assert delivered["text"] == "batch text"


def test_late_session_after_release_is_aborted(handler, monkeypatch):
    """try_start returns AFTER the dictation was already released: the
    session must be aborted, never adopted by a later dictation."""
    session = FakeSession()
    release_now = threading.Event()

    def slow_try_start(cls, _retry=True):
        release_now.set()          # signal the test to 'release'
        time.sleep(0.3)            # connection slower than the dictation
        return session

    monkeypatch.setattr(hotkey_mod.StreamingSession, "try_start",
                        classmethod(slow_try_start))
    with handler._lock:
        handler._active = True
    t = threading.Thread(target=handler._start_recording, daemon=True)
    t.start()
    release_now.wait(1.0)
    # release happens while try_start is still connecting
    with handler._lock:
        handler._gen += 1
        handler._active = False
    t.join(2.0)
    assert session.aborted
    assert handler._stream is None


def test_discard_aborts_session_and_bumps_gen(handler, monkeypatch):
    session = FakeSession()
    handler._stream = session
    gen = handler._gen
    handler._discard_recording()
    assert session.aborted
    assert handler._gen == gen + 1


def test_tap_drained_when_no_session(handler, monkeypatch):
    """No streaming session (e.g. tunnel down): the pump must STILL drain
    the tap, or the recorder child blocks on the full pipe and can't ack
    the stop command ('Recorder timed out')."""
    monkeypatch.setattr(hotkey_mod.StreamingSession, "try_start",
                        classmethod(lambda cls, _retry=True: None))
    with handler._lock:
        handler._active = True
    tap = handler._recorder.tap
    for _ in range(5):
        tap.push(b"x" * 100)
    handler._start_recording()
    # live audio keeps arriving; the drain-only pump must consume it
    tap.push(b"y" * 100)
    deadline = time.time() + 2.0
    while time.time() < deadline and tap.poll(0):
        time.sleep(0.02)
    assert not tap.poll(0), "tap not drained without a session"
    tap.push(b"")  # end-of-recording marker
    assert handler._pump_done.wait(2.0)


def test_backlog_replayed_into_slow_session(handler, monkeypatch):
    """Audio captured while the session is still connecting must be
    buffered and fed to the session once adopted (no head clipping)."""
    session = FakeSession()

    def slow_try_start(cls, _retry=True):
        time.sleep(0.3)  # connection slower than the first chunks
        return session

    monkeypatch.setattr(hotkey_mod.StreamingSession, "try_start",
                        classmethod(slow_try_start))
    with handler._lock:
        handler._active = True
    tap = handler._recorder.tap
    tap.push(b"head1")
    tap.push(b"head2")
    handler._start_recording()   # blocks ~0.3s in try_start; pump drains
    tap.push(b"tail")
    tap.push(b"")                # end-of-recording marker
    assert handler._pump_done.wait(2.0)
    assert session.fed == [b"head1", b"head2", b"tail"]
