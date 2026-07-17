"""Tests for the paste delivery path in voiceclip.macos.

The Cmd+V keystroke is posted in-process via pynput instead of spawning
osascript (~1ms vs ~185ms). These tests mock the keyboard controller and
clipboard helpers — no real events or pasteboard access.
"""

from unittest.mock import MagicMock, call

import pytest

from voiceclip import macos


@pytest.fixture
def fake_keyboard(monkeypatch):
    """Install a mock pynput controller and reset the lazy singleton."""
    kb = MagicMock()
    monkeypatch.setattr(macos, "_kb_controller", kb)
    return kb


class TestSendCmdV:
    def test_posts_v_with_cmd_held(self, fake_keyboard):
        from pynput.keyboard import Key

        assert macos._send_cmd_v() is True

        fake_keyboard.pressed.assert_called_once_with(Key.cmd)
        assert fake_keyboard.press.call_args == call("v")
        assert fake_keyboard.release.call_args == call("v")

    def test_failure_returns_false(self, fake_keyboard):
        fake_keyboard.pressed.side_effect = RuntimeError("no event tap")

        assert macos._send_cmd_v() is False  # must not raise

    def test_controller_created_lazily(self, monkeypatch):
        """Importing the module must not create a Controller; first use does."""
        monkeypatch.setattr(macos, "_kb_controller", None)
        created = []

        class FakeController:
            def __init__(self):
                created.append(self)

            def pressed(self, *a):
                return MagicMock(__enter__=MagicMock(), __exit__=MagicMock())

            def press(self, *a):
                pass

            def release(self, *a):
                pass

        import pynput.keyboard
        monkeypatch.setattr(pynput.keyboard, "Controller", FakeController)

        macos._send_cmd_v()
        macos._send_cmd_v()

        assert len(created) == 1  # reused, not recreated


class TestCopyPasteAndRestore:
    def test_sequence_and_restore(self, fake_keyboard, monkeypatch):
        """Saves clipboard, copies text, pastes, restores previous content."""
        events = []
        monkeypatch.setattr(macos, "_get_clipboard", lambda: "previous content")
        monkeypatch.setattr(
            macos, "copy_to_clipboard", lambda t: events.append(("copy", t))
        )
        monkeypatch.setattr(
            macos, "_send_cmd_v", lambda: events.append(("paste",)) or True
        )
        monkeypatch.setattr(macos.time, "sleep", lambda s: None)

        macos.copy_paste_and_restore("hello world")

        assert events == [
            ("copy", "hello world"),
            ("paste",),
            ("copy", "previous content"),
        ]

    def test_no_restore_when_clipboard_unreadable(self, fake_keyboard, monkeypatch):
        events = []
        monkeypatch.setattr(macos, "_get_clipboard", lambda: None)
        monkeypatch.setattr(
            macos, "copy_to_clipboard", lambda t: events.append(("copy", t))
        )
        monkeypatch.setattr(
            macos, "_send_cmd_v", lambda: events.append(("paste",)) or True
        )
        monkeypatch.setattr(macos.time, "sleep", lambda s: None)

        macos.copy_paste_and_restore("hello")

        assert events == [("copy", "hello"), ("paste",)]
