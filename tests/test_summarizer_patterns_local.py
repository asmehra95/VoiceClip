"""Local-path tests for summarizer and patterns.

Both modules should append REASONING_DIRECTIVE when calling complete_local
so reasoning-tuned models route their scratchpad into <think> tags we can
strip. Cloud paths must NOT get the directive — their reasoning is
server-side and they return clean answers.
"""

import pytest

from voiceclip import config, history, llm_provider, summarizer, patterns


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
    config.load()
    history.init()
    yield
    history.close()


class TestSummarizerLocalDirective:
    def test_local_summary_appends_reasoning_directive(self, monkeypatch):
        captured = {}

        def fake_local(*, system, user, model_id, max_tokens=400):
            captured["system"] = system
            return "You spent the day coding."

        monkeypatch.setattr(llm_provider, "complete_local", fake_local)
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "local")
        monkeypatch.setattr(config, "SUMMARIES_LOCAL_MODEL",
                            "mlx-community/Stub-4bit")

        # Need at least one entry for the day
        history.save("raw", "hello there", 1.0, kind="transcription",
                     app_name="Slack")
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        result = summarizer.summarize_day(today, force=True)
        assert result is not None
        assert "<think>" in captured["system"]
        # The underlying summary prompt (e.g. "2-4 sentences") is still there
        assert "sentences" in captured["system"]

    def test_openai_summary_does_not_get_directive(self, monkeypatch):
        captured = {}

        def fake_openai(*, system, user, model_id, **kwargs):
            captured["system"] = system
            return "A cloud summary."

        monkeypatch.setattr(llm_provider, "complete_openai", fake_openai)
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "openai")
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o-mini")

        history.save("raw", "a thing", 1.0, kind="transcription")
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        summarizer.summarize_day(today, force=True)
        # Cloud providers do their own reasoning server-side — they don't
        # need (or want) the local directive
        assert "<think>" not in captured["system"]


class TestPatternsLocalDirective:
    def test_local_patterns_appends_reasoning_directive(self, monkeypatch):
        captured = {}

        def fake_local(*, system, user, model_id, max_tokens=400):
            captured["system"] = system
            # Valid JSON so _parse_json_safely succeeds
            return '{"occupied_with": "stub", "themes": [], "suggestions": []}'

        monkeypatch.setattr(llm_provider, "complete_local", fake_local)
        monkeypatch.setattr(config, "PATTERNS_PROVIDER", "local")
        monkeypatch.setattr(config, "PATTERNS_LOCAL_MODEL",
                            "mlx-community/Stub-4bit")
        patterns.reset_cache()

        # Need at least one entry so patterns actually runs
        history.save("raw", "hello there", 1.0, kind="reflection",
                     app_name="Terminal")

        result = patterns.generate_patterns(window_days=7, force=True)
        assert result["ok"] is True
        assert "<think>" in captured["system"]
        # Patterns' own structural directive (the JSON schema) is still there
        assert "JSON" in captured["system"].upper() or "json" in captured["system"]

    def test_openai_patterns_does_not_get_directive(self, monkeypatch):
        captured = {}

        def fake_openai(*, system, user, model_id, **kwargs):
            captured["system"] = system
            return '{"occupied_with": "x", "themes": [], "suggestions": []}'

        monkeypatch.setattr(llm_provider, "complete_openai", fake_openai)
        monkeypatch.setattr(config, "PATTERNS_PROVIDER", "openai")
        monkeypatch.setattr(config, "PATTERNS_OPENAI_MODEL", "gpt-4o-mini")
        patterns.reset_cache()

        history.save("raw", "a thing", 1.0, kind="reflection")

        patterns.generate_patterns(window_days=7, force=True)
        assert "<think>" not in captured["system"]
