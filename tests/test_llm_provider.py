"""Focused tests for voiceclip.llm_provider.

Scope intentionally narrow — we're not mocking the full SDK surface. Just
checking the thin wrappers: cache semantics, and that the SDK clients get a
timeout so a stalled cloud call can't wedge the viewer handler thread.
"""

import pytest

from voiceclip import llm_provider


@pytest.fixture(autouse=True)
def _reset_cache():
    """Drop any loaded local models between tests."""
    llm_provider.reset_mlx_cache()
    yield
    llm_provider.reset_mlx_cache()


class TestClientTimeouts:
    """The SDK clients must be constructed with a finite timeout.

    Without this, a stalled network request pins the handler thread for the
    SDK default (10 minutes on OpenAI, effectively unbounded on some
    Anthropic versions). 60 seconds is plenty for our short completions.

    We stub the SDK constructors so these tests run even when the real
    openai / anthropic packages aren't in the venv — the contract we care
    about is "timeout kwarg is passed", not "SDK behaves correctly".
    """

    def test_openai_client_passes_timeout(self, monkeypatch):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        # Inject a fake openai module so the lazy import inside
        # _openai_client picks it up instead of failing.
        import sys
        import types
        fake_mod = types.ModuleType("openai")
        fake_mod.OpenAI = FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_mod)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        llm_provider._openai_client()
        assert "timeout" in captured, "OpenAI client built without a timeout"
        assert 0 < float(captured["timeout"]) <= 120.0, \
            f"OpenAI timeout out of range: {captured['timeout']}"

    def test_openai_client_missing_key_raises_readable_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm_provider._openai_client()

    def test_anthropic_client_passes_timeout(self, monkeypatch):
        captured = {}

        class FakeAnthropicClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import sys
        import types
        fake_mod = types.ModuleType("anthropic")
        fake_mod.Anthropic = FakeAnthropicClient
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        llm_provider._anthropic_client()
        assert "timeout" in captured, "Anthropic client built without a timeout"
        assert 0 < float(captured["timeout"]) <= 120.0, \
            f"Anthropic timeout out of range: {captured['timeout']}"

    def test_anthropic_client_missing_key_raises_readable_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            llm_provider._anthropic_client()


class TestMlxCache:
    """The mlx-lm cache is the only reason local summaries don't re-pay the
    5-15s load on every click. These tests verify the cache works without
    requiring mlx-lm to actually be installed."""

    def test_reset_clears_cache(self):
        # Prime the cache with a fake entry; reset should wipe it.
        llm_provider._mlx_cache["fake-model"] = ("model-obj", "tok-obj")
        llm_provider.reset_mlx_cache()
        assert llm_provider._mlx_cache == {}

    def test_load_hits_cache_without_mlx_installed(self):
        """If the cache already has an entry, _mlx_load must return it
        without touching the mlx_lm import — which may not even be
        installed."""
        llm_provider._mlx_cache["cached-model"] = ("weights", "tokenizer")
        # Should return the cached tuple verbatim, no import path taken
        assert llm_provider._mlx_load("cached-model") == ("weights", "tokenizer")
