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



class TestTimelineGeneration:
    """generate_timeline is a parallel entry point to summarize_day.
    Uses the same config (summaries.*), persists to day_timelines
    (separate cache from day_summaries), and produces structurally
    different output via _SYSTEM_PROMPT_TIMELINE.
    """

    def test_timeline_calls_llm_with_timeline_prompt(self, monkeypatch):
        """The timeline system prompt must contain the word 'chronological'
        so reasoning models understand the task. Pins the contract without
        asserting the exact wording (which can evolve)."""
        from voiceclip.summarizer import generate_timeline, _SYSTEM_PROMPT_TIMELINE

        captured = {}

        def fake_local(*, system, user, model_id, max_tokens=400):
            captured["system"] = system
            captured["user"] = user
            return "Morning: you reviewed e-invoicing rules.\n\nAfternoon: you assigned onboarding tasks."

        monkeypatch.setattr(llm_provider, "complete_local", fake_local)
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "local")
        monkeypatch.setattr(config, "SUMMARIES_LOCAL_MODEL",
                            "mlx-community/Stub-4bit")

        history.save("raw", "hello there", 1.0, kind="transcription", app_name="Slack")
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        result = generate_timeline(today, force=True)
        assert result is not None
        assert "chronological" in captured["system"].lower()
        # REASONING_DIRECTIVE should also be appended (it's a local path)
        assert "<think>" in captured["system"]

    def test_timeline_persists_to_separate_cache(self, monkeypatch):
        """Timeline and summary are cached independently — regenerating
        one must not wipe the other."""
        from voiceclip.summarizer import generate_timeline, summarize_day

        monkeypatch.setattr(llm_provider, "complete_local",
                            lambda **kw: "First pass output")
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "local")
        monkeypatch.setattr(config, "SUMMARIES_LOCAL_MODEL",
                            "mlx-community/Stub-4bit")

        history.save("raw", "hi", 1.0, kind="transcription", app_name="Slack")
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        summarize_day(today, force=True)
        assert history.get_day_summary(today) is not None
        assert history.get_day_timeline(today) is None

        generate_timeline(today, force=True)
        # Both now cached, neither wiped
        assert history.get_day_summary(today) is not None
        assert history.get_day_timeline(today) is not None

    def test_timeline_respects_provider_none(self, monkeypatch):
        """Same guard as summarize_day — if provider is off, no call."""
        from voiceclip.summarizer import generate_timeline

        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "none")
        history.save("raw", "hi", 1.0, kind="transcription")
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        assert generate_timeline(today, force=True) is None

    def test_timeline_returns_none_for_day_with_no_entries(self, monkeypatch):
        from voiceclip.summarizer import generate_timeline

        monkeypatch.setattr(llm_provider, "complete_local",
                            lambda **kw: "whatever")
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "local")
        monkeypatch.setattr(config, "SUMMARIES_LOCAL_MODEL", "x")

        assert generate_timeline("2020-01-01", force=True) is None

    def test_timeline_reuses_cache_when_not_forced(self, monkeypatch):
        """If a cached timeline exists for a past day, don't re-call
        the LLM. Mirrors summarize_day's caching behavior."""
        from voiceclip.summarizer import generate_timeline

        call_count = {"n": 0}

        def counting_local(**kw):
            call_count["n"] += 1
            return "cached body"

        monkeypatch.setattr(llm_provider, "complete_local", counting_local)
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "local")
        monkeypatch.setattr(config, "SUMMARIES_LOCAL_MODEL", "x")

        # Save an entry on a past date
        history._conn.execute(
            "INSERT INTO transcriptions (timestamp, raw_text, formatted_text, "
            "duration_seconds, persona, model, word_count, kind, app_name, "
            "window_title, is_research_topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("2025-06-15T10:00:00", "raw", "Some thing.", 1.0, "d", "t", 2,
             "transcription", "Slack", None, 0),
        )
        history._conn.commit()

        generate_timeline("2025-06-15", force=True)
        assert call_count["n"] == 1

        # Second call without force: cache hit, no new call
        result = generate_timeline("2025-06-15", force=False)
        assert call_count["n"] == 1
        assert result["timeline"] == "cached body"
