"""Focused tests for voiceclip.researcher.

The local branch must:
  - call complete_local (not a cloud path)
  - return empty sources and used_web_search=False
  - persist the resulting brief via history.save_brief with the right model id

We stub `llm_provider.complete_local` so tests run without mlx-lm installed.
"""

import pytest

from voiceclip import config, history, llm_provider, researcher


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Each test gets a fresh DB + config. Research is enabled in local mode."""
    monkeypatch.setattr(history, "DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
    config.load()
    history.init()
    # Force research into local mode for the duration of each test
    monkeypatch.setattr(config, "RESEARCH_PROVIDER", "local")
    monkeypatch.setattr(config, "RESEARCH_LOCAL_MODEL",
                        "mlx-community/TestStub-4bit")
    yield
    history.close()


class TestLocalResearchBranch:
    def test_local_path_calls_complete_local(self, monkeypatch):
        captured = {}

        def fake_local(*, system, user, model_id, max_tokens=400):
            captured["system"] = system
            captured["user"] = user
            captured["model_id"] = model_id
            captured["max_tokens"] = max_tokens
            return "A concise brief about the topic."

        monkeypatch.setattr(llm_provider, "complete_local", fake_local)

        tid = history.create_research_topic("How does Raft work")
        result = researcher.research_topic(tid)

        assert result is not None
        assert result["text"] == "A concise brief about the topic."
        # Local path never consults the web
        assert result["used_web_search"] is False
        assert result["sources"] == []
        assert result["provider"] == "local"
        assert result["model"] == "mlx-community/TestStub-4bit"
        # complete_local was called with the local prompt + wrapped user content
        assert "no web access" in captured["system"].lower()
        assert "<topic>How does Raft work</topic>" in captured["user"]
        assert captured["model_id"] == "mlx-community/TestStub-4bit"

    def test_local_path_ignores_cloud_functions(self, monkeypatch):
        """Cloud helpers must not be invoked when provider=local."""
        called = {"cloud": False}

        def spy_cloud(*args, **kwargs):
            called["cloud"] = True
            return ("", [], False)

        monkeypatch.setattr(llm_provider, "complete_openai_with_web_search", spy_cloud)
        monkeypatch.setattr(llm_provider, "complete_anthropic_with_web_search", spy_cloud)
        monkeypatch.setattr(llm_provider, "complete_local",
                            lambda **kw: "stub brief")

        tid = history.create_research_topic("Conceptual question")
        researcher.research_topic(tid)
        assert called["cloud"] is False

    def test_local_failure_persists_as_failed_brief(self, monkeypatch):
        """When mlx-lm isn't installed, the failure message should be
        actionable and the row should be stored with status=failed so the
        UI reflects the state."""
        def boom(**kwargs):
            raise RuntimeError(
                "mlx-lm is not installed. Install it with:\n"
                "    ~/.voiceclip/.venv/bin/pip install mlx-lm"
            )
        monkeypatch.setattr(llm_provider, "complete_local", boom)

        tid = history.create_research_topic("Some topic")
        with pytest.raises(RuntimeError, match="mlx-lm"):
            researcher.research_topic(tid)

        # The failed brief is recorded
        row = history._conn.execute(
            "SELECT status, error FROM research_briefs "
            "WHERE entry_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert row[0] == "failed"
        assert "mlx-lm" in row[1]

    def test_local_path_does_not_warn_about_network(self, monkeypatch):
        """network_warning() must not fire when provider=local — the whole
        point is it stays on-device."""
        monkeypatch.setattr(config, "RESEARCH_PROVIDER", "local")
        assert researcher.network_warning() is None

    def test_network_warning_fires_for_cloud(self, monkeypatch):
        monkeypatch.setattr(config, "RESEARCH_PROVIDER", "openai")
        warn = researcher.network_warning()
        assert warn is not None
        assert "openai" in warn.lower()


class TestHumanizeLocalErrors:
    def test_missing_mlx_lm_points_at_pip(self):
        err = RuntimeError("mlx-lm is not installed.")
        msg = researcher._humanize_provider_error(err, "local", "some-model")
        assert "pip install mlx-lm" in msg

    def test_bad_model_id_points_at_config_key(self):
        err = RuntimeError("404 not found for repository bogus-model")
        msg = researcher._humanize_provider_error(err, "local", "bogus-model")
        assert "research.local_model" in msg

    def test_out_of_memory_suggests_smaller_variant(self):
        err = RuntimeError("CUDA out of memory")  # OOM keyword triggers branch
        msg = researcher._humanize_provider_error(err, "local", "big-model")
        assert "smaller" in msg.lower()

    def test_generic_local_error_is_prefixed_clearly(self):
        err = RuntimeError("something went wrong")
        msg = researcher._humanize_provider_error(err, "local", "model-id")
        assert msg.startswith("Local research failed:")
