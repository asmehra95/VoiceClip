"""Tests for the meaningful-history additions to voiceclip.history.

Covers: schema migration, reflections, research topics, research briefs,
promotion, inline edit (update_text), kind-scoped queries, retention,
the patterns-support helpers.
"""

import os
import sqlite3
import tempfile
import pytest

from voiceclip import config, history


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Point history at a temp SQLite file for each test."""
    db = str(tmp_path / "history.db")
    monkeypatch.setattr(history, "DB_PATH", db)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
    config.load()  # sets PERSONA, MODEL, etc. needed by save()
    history.init()
    yield
    history.close()


class TestSchema:
    def test_fresh_db_has_all_columns(self):
        cols = {r[1] for r in history._conn.execute(
            "PRAGMA table_info(transcriptions)").fetchall()}
        for required in ("kind", "app_name", "window_title",
                         "edited_at", "is_research_topic"):
            assert required in cols, f"missing column: {required}"

    def test_fresh_db_has_auxiliary_tables(self):
        tables = {r[0] for r in history._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "day_summaries" in tables
        assert "research_briefs" in tables

    def test_init_is_idempotent(self):
        # Re-running init on an already-migrated DB must not raise.
        history.init()
        history.init()
        assert history._conn is not None

    def test_migration_from_pre_feature_schema(self, tmp_path, monkeypatch):
        """Seed a pre-feature schema, run init, verify backfill."""
        legacy_db = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(legacy_db)
        conn.execute("""
            CREATE TABLE transcriptions (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                formatted_text TEXT NOT NULL,
                duration_seconds REAL,
                persona TEXT,
                model TEXT,
                word_count INTEGER
            )
        """)
        conn.execute(
            "INSERT INTO transcriptions (timestamp, raw_text, formatted_text, "
            "duration_seconds, persona, model, word_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-01-01T10:00:00", "hi", "Hello.", 1.5, "default", "tiny", 1),
        )
        conn.commit()
        conn.close()

        # Swap history's DB to the legacy file and re-init
        history.close()
        monkeypatch.setattr(history, "DB_PATH", legacy_db)
        history.init()

        row = history._conn.execute(
            "SELECT kind, app_name, window_title, is_research_topic "
            "FROM transcriptions WHERE id = 1"
        ).fetchone()
        assert row == ("transcription", None, None, 0)


class TestSaveAndQuery:
    def test_save_returns_id(self):
        i = history.save("raw", "Hello.", 1.0, kind="transcription", app_name="Slack")
        assert isinstance(i, int)
        assert i > 0

    def test_kind_round_trip(self):
        history.save("a", "First.", 1.0, kind="transcription")
        history.save("b", "Second.", 1.0, kind="reflection")
        assert history.count(kind="transcription") == 1
        assert history.count(kind="reflection") == 1
        assert history.count() == 2

    def test_invalid_kind_falls_back(self):
        i = history.save("x", "X.", 1.0, kind="bogus")
        assert i is not None
        row = history._conn.execute(
            "SELECT kind FROM transcriptions WHERE id = ?", (i,)).fetchone()
        assert row[0] == "transcription"

    def test_research_topic_not_in_day_view(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        history.save("a", "A.", 1.0, kind="transcription", app_name="Slack")
        history.create_research_topic("A topic to research")
        entries = history.entries_for_day(today)
        assert len(entries) == 1
        assert entries[0]["kind"] == "transcription"


class TestInlineEdit:
    def test_update_text_sets_edited_at(self):
        i = history.save("a", "Original.", 1.0)
        r = history.update_text(i, "Edited!")
        assert r is not None
        assert r["text"] == "Edited!"
        assert r["edited_at"]  # ISO-8601 string, non-empty
        assert r["word_count"] == 1

    def test_update_text_missing_entry(self):
        assert history.update_text(9999, "anything") is None


class TestDelete:
    def test_delete_entry_returns_deleted(self):
        i = history.save("a", "Bye.", 1.0)
        r = history.delete_entry(i)
        assert r is not None
        assert r["id"] == i
        assert history.count() == 0

    def test_delete_missing_is_none(self):
        assert history.delete_entry(9999) is None


class TestPromote:
    def test_promote_last(self):
        i = history.save("a", "A.", 1.0, kind="transcription")
        r = history.promote_to_reflection(last=True)
        assert r is not None and r["id"] == i
        row = history._conn.execute(
            "SELECT kind FROM transcriptions WHERE id = ?", (i,)).fetchone()
        assert row[0] == "reflection"

    def test_promote_already_reflection_is_noop(self):
        i = history.save("a", "A.", 1.0, kind="reflection")
        assert history.promote_to_reflection(entry_id=i) is None

    def test_promote_missing(self):
        assert history.promote_to_reflection(entry_id=9999) is None
        assert history.promote_to_reflection(last=True) is None


class TestResearchTopics:
    def test_create_and_list(self):
        tid = history.create_research_topic("CRDTs vs OT")
        assert tid is not None
        topics = history.list_research_topics()
        assert len(topics) == 1
        assert topics[0]["status"] == "pending"

    def test_save_brief_transitions_status(self):
        tid = history.create_research_topic("Raft consensus")
        history.save_brief(tid, status="done", brief_text="brief body",
                           provider="openai", model="gpt-4o-mini")
        topics = history.list_research_topics()
        assert topics[0]["status"] == "ready"

    def test_latest_brief_fetches_most_recent(self):
        tid = history.create_research_topic("Topic X")
        history.save_brief(tid, status="done", brief_text="first",
                           provider="openai", model="gpt-4o-mini")
        history.save_brief(tid, status="done", brief_text="second",
                           provider="openai", model="gpt-4o-mini")
        b = history.latest_brief(tid)
        assert b is not None and b["text"] == "second"

    def test_find_research_topic_by_text(self):
        history.create_research_topic("Eventual consistency")
        match = history.find_research_topic_by_text("eventual CONSISTENCY")
        assert match is not None
        nope = history.find_research_topic_by_text("something else")
        assert nope is None


class TestCleanup:
    def test_cleanup_respects_kind(self):
        # Seed one old transcription + one old reflection
        history._conn.execute(
            "INSERT INTO transcriptions (timestamp, raw_text, formatted_text, "
            "duration_seconds, persona, model, word_count, kind, app_name, "
            "window_title, is_research_topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("2020-01-01T00:00:00", "old-t", "Old t.", 1.0, "d", "t", 1,
             "transcription", None, None, 0),
        )
        history._conn.execute(
            "INSERT INTO transcriptions (timestamp, raw_text, formatted_text, "
            "duration_seconds, persona, model, word_count, kind, app_name, "
            "window_title, is_research_topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("2020-01-01T00:00:00", "old-r", "Old r.", 1.0, "d", "t", 1,
             "reflection", None, None, 0),
        )
        history._conn.commit()
        history.cleanup(max_days=30, reflection_max_days=0)
        assert history.count(kind="transcription") == 0
        assert history.count(kind="reflection") == 1  # kept forever

    def test_cleanup_with_reflection_retention(self):
        history._conn.execute(
            "INSERT INTO transcriptions (timestamp, raw_text, formatted_text, "
            "duration_seconds, persona, model, word_count, kind, app_name, "
            "window_title, is_research_topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("2020-01-01T00:00:00", "old-r", "Old r.", 1.0, "d", "t", 1,
             "reflection", None, None, 0),
        )
        history._conn.commit()
        history.cleanup(max_days=30, reflection_max_days=7)
        assert history.count(kind="reflection") == 0
