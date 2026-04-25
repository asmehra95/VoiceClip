"""Tests for voiceclip.history — SQLite transcription storage."""

import os
import pytest

from voiceclip import config
from voiceclip import history


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect history DB and config to temp directory."""
    cfg_dir = str(tmp_path)
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config, "CONFIG_PATH", os.path.join(cfg_dir, "config.json"))
    monkeypatch.setattr(history, "DB_PATH", os.path.join(cfg_dir, "history.db"))
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(config, "HISTORY_ENABLED", True)
    monkeypatch.setattr(config, "PERSONA", "default")
    monkeypatch.setattr(config, "MODEL", "large-v3-turbo")
    # Clear env vars
    for var in ["VOICECLIP_HISTORY"]:
        monkeypatch.delenv(var, raising=False)
    yield


class TestInit:
    def test_creates_db_file(self, tmp_path):
        history.init()
        assert os.path.exists(os.path.join(str(tmp_path), "history.db"))

    def test_db_permissions(self, tmp_path):
        history.init()
        db_path = os.path.join(str(tmp_path), "history.db")
        mode = oct(os.stat(db_path).st_mode)[-3:]
        assert mode == "600"


class TestSave:
    def test_save_and_count(self):
        history.init()
        assert history.count() == 0
        history.save("hello world", "Hello world.", 2.5)
        assert history.count() == 1

    def test_save_multiple(self):
        history.init()
        history.save("one", "One.", 1.0)
        history.save("two", "Two.", 1.5)
        history.save("three", "Three.", 2.0)
        assert history.count() == 3


class TestQuery:
    def test_query_recent(self):
        history.init()
        history.save("hello", "Hello.", 1.0)
        history.save("world", "World.", 1.0)
        result = history.query_recent(limit=5)
        assert "Hello." in result
        assert "World." in result

    def test_query_recent_limit(self):
        history.init()
        for i in range(20):
            history.save(f"msg {i}", f"Msg {i}.", 1.0)
        result = history.query_recent(limit=5)
        # Should have the 5 most recent
        assert "Msg 19." in result
        assert "Msg 15." in result
        assert "Msg 0." not in result

    def test_query_today(self):
        history.init()
        history.save("today msg", "Today msg.", 1.0)
        result = history.query_today()
        assert "Today msg." in result
        assert "1 transcription today" in result or "1 entry today" in result

    def test_query_search(self):
        history.init()
        history.save("meeting notes", "Meeting notes about the project.", 3.0)
        history.save("lunch plans", "Lunch at noon.", 1.0)
        result = history.query_search("meeting")
        assert "Meeting notes" in result
        assert "1 entry matching" in result or "1 result" in result

    def test_query_search_no_results(self):
        history.init()
        history.save("hello", "Hello.", 1.0)
        result = history.query_search("nonexistent")
        assert "0 entries" in result or "0 results" in result

    def test_get_by_id(self):
        history.init()
        history.save("first", "First.", 1.0)
        history.save("second", "Second.", 1.0)
        assert history.get_by_id(1) == "First."
        assert history.get_by_id(2) == "Second."
        assert history.get_by_id(999) is None


class TestClear:
    def test_clear_all(self):
        history.init()
        history.save("one", "One.", 1.0)
        history.save("two", "Two.", 1.0)
        assert history.count() == 2
        history.clear_all(force=True)
        assert history.count() == 0


class TestCleanup:
    def test_cleanup_old_entries(self):
        history.init()
        # Insert an entry with an old timestamp directly
        history._conn.execute(
            "INSERT INTO transcriptions "
            "(timestamp, raw_text, formatted_text, duration_seconds, persona, model, word_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2020-01-01T00:00:00", "old", "Old.", 1.0, "default", "tiny", 1),
        )
        history._conn.commit()
        history.save("new", "New.", 1.0)
        assert history.count() == 2

        history.cleanup(max_days=30)
        assert history.count() == 1  # Only the new one remains


class TestDisabled:
    def test_not_initialized_returns_gracefully(self, monkeypatch):
        monkeypatch.setattr(history, "_conn", None)
        # These should not crash
        history.save("test", "Test.", 1.0)
        assert history.count() == 0
        assert "not enabled" in history.query_recent()
        assert history.get_by_id(1) is None
