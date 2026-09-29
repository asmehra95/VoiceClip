"""Tests for the shared pynput listener dispatcher in voiceclip.hotkey.

No real keyboard.Listener is started; a fake listener class is patched
in so register/unregister and dispatch can be exercised deterministically.
The dispatcher exists because multiple concurrent pynput listeners abort
the process on macOS (HIToolbox SIGABRT) — every handler must ride ONE
listener, and one misbehaving handler must never break the others.
"""

import pytest

from voiceclip import hotkey as hotkey_mod
from voiceclip.hotkey import _SharedListener


class FakeListener:
    """Stands in for pynput's keyboard.Listener."""

    instances = []

    def __init__(self, on_press=None, on_release=None):
        self.on_press = on_press
        self.on_release = on_release
        self.started = False
        self.stopped = False
        FakeListener.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeHandler:
    """Minimal handler: records dispatched keys, optionally raises."""

    def __init__(self, raises=False):
        self._profile = "fake"
        self.presses = []
        self.releases = []
        self.raises = raises

    def _on_press(self, key):
        if self.raises:
            raise RuntimeError("boom")
        self.presses.append(key)

    def _on_release(self, key):
        if self.raises:
            raise RuntimeError("boom")
        self.releases.append(key)


@pytest.fixture
def shared(monkeypatch):
    FakeListener.instances = []
    monkeypatch.setattr(hotkey_mod.keyboard, "Listener", FakeListener)
    return _SharedListener()


def test_single_listener_for_many_handlers(shared):
    """Three handlers must share ONE underlying listener (the whole point)."""
    handlers = [FakeHandler() for _ in range(3)]
    for h in handlers:
        shared.register(h)
    assert len(FakeListener.instances) == 1
    assert FakeListener.instances[0].started
    assert shared.listener is FakeListener.instances[0]


def test_press_and_release_fan_out(shared):
    a, b = FakeHandler(), FakeHandler()
    shared.register(a)
    shared.register(b)
    shared._on_press("k1")
    shared._on_release("k1")
    assert a.presses == ["k1"] and b.presses == ["k1"]
    assert a.releases == ["k1"] and b.releases == ["k1"]


def test_exception_in_one_handler_does_not_break_others(shared):
    bad, good = FakeHandler(raises=True), FakeHandler()
    shared.register(bad)  # registered FIRST so its raise would shadow good
    shared.register(good)
    shared._on_press("k")
    shared._on_release("k")
    assert good.presses == ["k"]
    assert good.releases == ["k"]


def test_unregister_stops_listener_when_last_handler_leaves(shared):
    a, b = FakeHandler(), FakeHandler()
    shared.register(a)
    shared.register(b)
    listener = shared.listener
    shared.unregister(a)
    assert not listener.stopped          # b still registered
    shared.unregister(b)
    assert listener.stopped
    assert shared.listener is None


def test_reregister_after_full_stop_creates_new_listener(shared):
    a = FakeHandler()
    shared.register(a)
    shared.unregister(a)
    shared.register(a)
    assert len(FakeListener.instances) == 2
    assert FakeListener.instances[1].started
    assert not FakeListener.instances[1].stopped


def test_unregister_unknown_handler_is_harmless(shared):
    shared.unregister(FakeHandler())  # must not raise
    assert shared.listener is None


def test_double_register_dispatches_once(shared):
    a = FakeHandler()
    shared.register(a)
    shared.register(a)
    shared._on_press("k")
    assert a.presses == ["k"]


def test_handler_start_stop_use_shared_listener(shared, monkeypatch):
    """HotkeyHandler.start()/stop() ride the module-level shared listener
    and expose it via _listener for _validate_hotkey_listener."""
    monkeypatch.setattr(hotkey_mod, "_SHARED_LISTENER", shared)
    h1 = hotkey_mod.HotkeyHandler.__new__(hotkey_mod.HotkeyHandler)
    h2 = hotkey_mod.HotkeyHandler.__new__(hotkey_mod.HotkeyHandler)
    for h in (h1, h2):
        h._profile = "transcription"
        h._mode = "hold"
        h._listener = None
    h1.start()
    h2.start()
    assert len(FakeListener.instances) == 1
    assert h1._listener is FakeListener.instances[0]
    assert h2._listener is FakeListener.instances[0]
    h1.stop()
    assert h1._listener is None
    assert not FakeListener.instances[0].stopped
    h2.stop()
    assert FakeListener.instances[0].stopped
