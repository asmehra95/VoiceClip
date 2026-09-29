"""Voice-agent tool handlers — pure logic, no audio pipeline, no network."""

import pytest

from voiceclip import agent, history


class TestWebSearch:
    def test_formats_results(self, monkeypatch):
        class FakeDDGS:
            def text(self, query, max_results):
                return [{"title": "T1", "body": "B1", "href": "http://a"},
                        {"title": "T2", "body": "B2", "href": "http://b"}]
        import ddgs
        monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
        out = agent.web_search("anything")
        assert "T1: B1 (http://a)" in out and "T2" in out

    def test_no_results(self, monkeypatch):
        class FakeDDGS:
            def text(self, query, max_results):
                return []
        import ddgs
        monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
        assert agent.web_search("x") == "No results."

    def test_failure_is_a_message_not_a_raise(self, monkeypatch):
        class FakeDDGS:
            def text(self, query, max_results):
                raise RuntimeError("rate limited")
        import ddgs
        monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
        assert "Search failed" in agent.web_search("x")


class TestJournalTools:
    def test_save_note_and_search(self, live_history):
        assert agent.save_note("remember the belgium payment fix") \
            == "Saved to the journal."
        out = agent.journal_search("belgium")
        assert "belgium payment fix" in out
        assert "(reflection)" in out

    def test_search_no_match(self, live_history):
        assert agent.journal_search("xyzzy") == "Nothing in the journal matches."


class TestDayResolution:
    def test_today_yesterday_and_iso(self):
        from datetime import date, timedelta
        assert agent._resolve_day("today") == date.today().isoformat()
        assert agent._resolve_day("yesterday") == \
            (date.today() - timedelta(days=1)).isoformat()
        assert agent._resolve_day("2026-01-05") == "2026-01-05"


class TestCloudTools:
    def test_status(self, monkeypatch):
        monkeypatch.setattr("voiceclip.cloud_control.get_status",
                            lambda: {"state": "running", "type": "g6.2xlarge"})
        out = agent.cloud_status()
        assert "running" in out and "g6.2xlarge" in out

    def test_park(self, monkeypatch):
        calls = []
        monkeypatch.setattr("voiceclip.cloud_control.stop_instance",
                            lambda: calls.append(1) or {"state": "stopping"})
        out = agent.park_server()
        assert calls and "Parking" in out


class TestToolDispatchTable:
    def test_all_schema_tools_have_impls(self):
        schema = agent._tools_schema()
        names = {t.name for t in schema.standard_tools}
        assert names == set(agent._TOOL_IMPLS.keys())
