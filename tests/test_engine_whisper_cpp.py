"""Tests for the whisper_cpp engine's server lifecycle logic.

These cover the recovery/restart paths in _ensure_server() and
transcribe() — the exact fragilities that previously degraded the app to
per-call CLI model loads:

  - transient request failure trips _server_ready, recovery restores it
  - a dictation whose server request fails mid-flight retries via CLI
    instead of being dropped as "no speech"
  - a dead server triggers a rate-limited restart
  - blank audio (None with server healthy) is NOT retried via CLI

No real whisper-server is spawned; process/health/transport are all
monkeypatched module attributes.
"""

import os
import time

import pytest

from voiceclip import engine_whisper_cpp as eng


class FakeProc:
    """Stands in for subprocess.Popen — alive until kill()ed."""

    def __init__(self):
        self.pid = 12345
        self._rc = None

    def poll(self):
        return self._rc

    def kill(self):
        self._rc = -9

    def terminate(self):
        self._rc = -15

    def wait(self, timeout=None):
        return self._rc


@pytest.fixture
def engine_state(monkeypatch):
    """Reset module state to 'server up and healthy' with fakes."""
    proc = FakeProc()
    monkeypatch.setattr(eng, "_server_proc", proc)
    monkeypatch.setattr(eng, "_server_ready", True)
    monkeypatch.setattr(eng, "_last_restart_attempt", 0.0)
    monkeypatch.setattr(eng, "_server_health_check", lambda: True)
    monkeypatch.setattr(eng, "_resolve_model_path", lambda mid: f"/fake/{mid}.bin")
    yield proc


class TestEnsureServer:
    def test_healthy_server_untouched(self, engine_state, monkeypatch):
        started = []
        monkeypatch.setattr(eng, "_start_server", lambda *a, **k: started.append(1))

        eng._ensure_server("large-v3-turbo", timeout=5)

        assert eng._server_ready is True
        assert not started

    def test_transient_trip_recovers_without_restart(self, engine_state, monkeypatch):
        """Flag tripped but process alive+healthy → server mode restored."""
        monkeypatch.setattr(eng, "_server_ready", False)
        started = []
        monkeypatch.setattr(eng, "_start_server", lambda *a, **k: started.append(1))

        eng._ensure_server("large-v3-turbo", timeout=5)

        assert eng._server_ready is True
        assert not started

    def test_alive_but_unhealthy_does_not_restore(self, engine_state, monkeypatch):
        monkeypatch.setattr(eng, "_server_ready", False)
        monkeypatch.setattr(eng, "_server_health_check", lambda: False)

        eng._ensure_server("large-v3-turbo", timeout=5)

        assert eng._server_ready is False

    def test_dead_server_triggers_restart(self, engine_state, monkeypatch):
        engine_state.kill()
        calls = []
        monkeypatch.setattr(
            eng, "_start_server",
            lambda path, timeout=None: calls.append((path, timeout)) or True,
        )

        eng._ensure_server("large-v3-turbo", timeout=7)

        assert calls == [("/fake/large-v3-turbo.bin", 7)]

    def test_dead_server_clears_stale_ready_flag(self, engine_state, monkeypatch):
        engine_state.kill()
        monkeypatch.setattr(eng, "_start_server", lambda *a, **k: False)

        eng._ensure_server("large-v3-turbo", timeout=5)

        assert eng._server_ready is False

    def test_restart_rate_limited(self, engine_state, monkeypatch):
        """A second restart within the cooldown window is skipped."""
        engine_state.kill()
        calls = []
        monkeypatch.setattr(
            eng, "_start_server", lambda *a, **k: calls.append(1) or False,
        )

        eng._ensure_server("large-v3-turbo", timeout=5)
        eng._ensure_server("large-v3-turbo", timeout=5)

        assert len(calls) == 1

    def test_restart_allowed_after_cooldown(self, engine_state, monkeypatch):
        engine_state.kill()
        calls = []
        monkeypatch.setattr(
            eng, "_start_server", lambda *a, **k: calls.append(1) or False,
        )

        eng._ensure_server("large-v3-turbo", timeout=5)
        monkeypatch.setattr(
            eng, "_last_restart_attempt",
            time.time() - eng._RESTART_COOLDOWN - 1,
        )
        eng._ensure_server("large-v3-turbo", timeout=5)

        assert len(calls) == 2

    def test_restart_failure_is_contained(self, engine_state, monkeypatch):
        """_resolve_model_path raising must not propagate to the caller."""
        engine_state.kill()
        monkeypatch.setattr(
            eng, "_resolve_model_path",
            lambda mid: (_ for _ in ()).throw(FileNotFoundError("no model")),
        )

        eng._ensure_server("large-v3-turbo", timeout=5)  # must not raise

        assert eng._server_ready is False


class TestTranscribeFallback:
    def test_server_success_returns_text(self, engine_state, monkeypatch):
        monkeypatch.setattr(eng, "_transcribe_server", lambda p: "hello world")
        cli = []
        monkeypatch.setattr(
            eng, "_transcribe_cli", lambda p, m: cli.append(1) or "cli text",
        )

        assert eng.transcribe("/fake.wav", "large-v3-turbo") == "hello world"
        assert not cli

    def test_blank_audio_not_retried_via_cli(self, engine_state, monkeypatch):
        """None with the server still healthy means silence — no CLI retry."""
        monkeypatch.setattr(eng, "_transcribe_server", lambda p: None)
        cli = []
        monkeypatch.setattr(
            eng, "_transcribe_cli", lambda p, m: cli.append(1) or "cli text",
        )

        assert eng.transcribe("/fake.wav", "large-v3-turbo") is None
        assert not cli

    def test_midflight_failure_retries_via_cli(self, engine_state, monkeypatch):
        """Request failure (flag tripped, None returned) → CLI retry, not
        a silent 'no speech' drop."""
        def failing_server_request(path):
            eng._server_ready = False
            return None

        monkeypatch.setattr(eng, "_transcribe_server", failing_server_request)
        monkeypatch.setattr(eng, "_transcribe_cli", lambda p, m: "cli rescued it")

        assert eng.transcribe("/fake.wav", "large-v3-turbo") == "cli rescued it"

    def test_dead_server_uses_cli_when_restart_fails(self, engine_state, monkeypatch):
        engine_state.kill()
        monkeypatch.setattr(eng, "_start_server", lambda *a, **k: False)
        monkeypatch.setattr(eng, "_transcribe_cli", lambda p, m: "cli text")

        assert eng.transcribe("/fake.wav", "large-v3-turbo") == "cli text"

    def test_restart_uses_bounded_timeout(self, engine_state, monkeypatch):
        """In-line restarts must use SERVER_RESTART_TIMEOUT, not the
        generous startup timeout — the transcriber abandons the worker
        after 60s."""
        engine_state.kill()
        timeouts = []

        def fake_start(path, timeout=None):
            timeouts.append(timeout)
            return False

        monkeypatch.setattr(eng, "_start_server", fake_start)
        monkeypatch.setattr(eng, "_transcribe_cli", lambda p, m: None)

        eng.transcribe("/fake.wav", "large-v3-turbo")

        assert timeouts == [eng.SERVER_RESTART_TIMEOUT]
        assert eng.SERVER_RESTART_TIMEOUT < 60


class TestKeepWarmPing:
    def test_ping_restarts_dead_server(self, engine_state, monkeypatch):
        """The background ping is the preferred place to revive a dead
        server, keeping the restart off the dictation critical path."""
        engine_state.kill()
        calls = []

        def fake_start(path, timeout=None):
            calls.append(path)
            monkeypatch.setattr(eng, "_server_proc", FakeProc())
            monkeypatch.setattr(eng, "_server_ready", True)
            return True

        monkeypatch.setattr(eng, "_start_server", fake_start)
        pings = []
        monkeypatch.setattr(eng, "_transcribe_server", lambda p: pings.append(p))

        eng.keep_warm_ping("large-v3-turbo")

        assert calls == ["/fake/large-v3-turbo.bin"]
        assert len(pings) == 1

    def test_ping_skipped_when_server_unavailable(self, engine_state, monkeypatch):
        """Never ping via CLI — it would cold-load the model every interval."""
        engine_state.kill()
        monkeypatch.setattr(eng, "_start_server", lambda *a, **k: False)
        pings = []
        monkeypatch.setattr(eng, "_transcribe_server", lambda p: pings.append(p))

        eng.keep_warm_ping("large-v3-turbo")

        assert not pings


class TestFindBinary:
    """Binary resolution prefers installer-managed ~/.voiceclip/bin, then
    the from-source build, and returns the source path (even if missing)
    so error messages stay actionable."""

    def test_prefers_installed_bin(self, monkeypatch):
        installed = os.path.expanduser("~/.voiceclip/bin/whisper-server")
        monkeypatch.setattr(
            eng.os.path, "isfile", lambda p: p == installed,
        )
        assert eng._find_binary("whisper-server") == installed

    def test_falls_back_to_source_build(self, monkeypatch):
        source = os.path.expanduser("~/whisper.cpp/build/bin/whisper-server")
        monkeypatch.setattr(
            eng.os.path, "isfile", lambda p: p == source,
        )
        assert eng._find_binary("whisper-server") == source

    def test_returns_source_path_when_nothing_exists(self, monkeypatch):
        monkeypatch.setattr(eng.os.path, "isfile", lambda p: False)
        assert eng._find_binary("whisper-cli") == os.path.expanduser(
            "~/whisper.cpp/build/bin/whisper-cli"
        )
