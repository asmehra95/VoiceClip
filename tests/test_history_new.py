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



class TestSearch:
    def test_search_finds_simple_term(self):
        history.save("hello", "The quick brown fox.", 1.0)
        history.save("hello", "A slow turtle.", 1.0)
        results = history.search_entries("fox")
        assert len(results) == 1
        assert "fox" in results[0]["text"].lower()

    def test_search_multi_token_and(self):
        history.save("a", "the quick brown fox", 1.0)
        history.save("b", "brown sugar on toast", 1.0)
        results = history.search_entries("brown fox")
        assert len(results) == 1
        assert "fox" in results[0]["text"].lower()

    def test_search_prefix(self):
        history.save("a", "recording sounds nice", 1.0)
        results = history.search_entries("recor")
        assert len(results) == 1

    def test_search_excludes_research_topics(self):
        history.save("a", "A normal entry about cats.", 1.0)
        history.create_research_topic("cats and dogs research")
        results = history.search_entries("cats")
        assert len(results) == 1
        assert results[0]["kind"] == "transcription"

    def test_search_kind_filter(self):
        history.save("a", "Foxes are cool.", 1.0, kind="transcription")
        history.save("b", "Foxes are clever.", 1.0, kind="reflection")
        refl = history.search_entries("foxes", kind="reflection")
        assert len(refl) == 1
        assert refl[0]["kind"] == "reflection"

    def test_search_empty_query_returns_empty(self):
        history.save("a", "anything", 1.0)
        assert history.search_entries("") == []
        assert history.search_entries("   ") == []

    def test_search_fts_meta_chars_are_handled(self):
        # FTS5 treats ":" as a column filter and would otherwise fail
        history.save("a", "Python version 3.12 released", 1.0)
        results = history.search_entries('python: 3.12')
        assert len(results) >= 1

    def test_search_reflects_edits(self):
        # After an edit, the FTS index should update via trigger.
        i = history.save("raw", "Original about cats.", 1.0)
        history.update_text(i, "Edited text about dogs.")
        assert len(history.search_entries("cats")) == 0
        assert len(history.search_entries("dogs")) == 1

    def test_search_reflects_deletes(self):
        i = history.save("a", "Will be deleted soon.", 1.0)
        assert len(history.search_entries("deleted")) == 1
        history.delete_entry(i)
        assert len(history.search_entries("deleted")) == 0



class TestReconnect:
    """Auto-reconnect behavior on OperationalError.

    Verifies the fallback path in `_with_retry` — if the write fails once,
    the helper re-opens the connection and retries. Data should land.
    """

    def test_save_survives_transient_error(self, monkeypatch):
        import sqlite3

        real_conn = history._conn
        call_count = {"n": 0}

        class FlakyConn:
            """Wraps the real connection; raises on the first INSERT."""
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, params=()):
                call_count["n"] += 1
                if call_count["n"] == 1 and "INSERT" in sql.upper():
                    raise sqlite3.OperationalError("database is locked (simulated)")
                return self._inner.execute(sql, params)

            def commit(self):
                return self._inner.commit()

            def close(self):
                return self._inner.close()

        # Swap in the flaky wrapper. When _reconnect runs, it calls init()
        # which will replace history._conn with a fresh real connection,
        # so the retried write succeeds.
        history._conn = FlakyConn(real_conn)
        new_id = history.save("raw", "Survived.", 1.0)
        assert new_id is not None
        # Prove it actually wrote to the real DB
        row = history._conn.execute(
            "SELECT formatted_text FROM transcriptions WHERE id = ?", (new_id,)
        ).fetchone()
        assert row[0] == "Survived."

    def test_save_gives_up_cleanly_if_both_attempts_fail(self, monkeypatch):
        import sqlite3

        class AlwaysBroken:
            def execute(self, sql, params=()):
                raise sqlite3.OperationalError("permanently broken")

            def commit(self):
                pass

            def close(self):
                pass

        history._conn = AlwaysBroken()
        # Prevent _reconnect from healing the connection so we hit the
        # "both attempts failed" branch.
        monkeypatch.setattr(history, "_reconnect", lambda: False)
        result = history.save("raw", "Doomed.", 1.0)
        assert result is None



class TestUpdateBriefText:
    """Unit tests for the history.update_brief_text helper."""

    def test_updates_existing_brief(self):
        tid = history.create_research_topic("A topic")
        bid = history.save_brief(tid, status="done", brief_text="original",
                                 provider="openai", model="gpt-4o-mini")
        r = history.update_brief_text(bid, "edited")
        assert r is not None
        assert r["text"] == "edited"
        # And reading back confirms the write
        brief = history.latest_brief(tid)
        assert brief["text"] == "edited"

    def test_returns_none_for_missing_brief(self):
        assert history.update_brief_text(99999, "x") is None



class TestArchive:
    """Archive/unarchive flow for research topics.

    Archived topics stay in the DB — still visible to FTS, still in the
    `transcriptions` table — but drop out of `list_research_topics` and
    surface only through `list_archived_research_topics`. This matches
    the Queue's "Done" button UX: declutter the active view without
    losing the work.
    """

    def test_archive_hides_from_active_list(self):
        tid = history.create_research_topic("CRDTs")
        history.save_brief(tid, status="done", brief_text="b",
                           provider="openai", model="gpt-4o-mini")
        assert len(history.list_research_topics()) == 1
        r = history.archive_topic(tid)
        assert r is not None and r["id"] == tid
        assert r["archived_at"]
        assert history.list_research_topics() == []

    def test_archive_shows_in_archived_list(self):
        tid = history.create_research_topic("Raft")
        history.archive_topic(tid)
        archived = history.list_archived_research_topics()
        assert len(archived) == 1
        assert archived[0]["id"] == tid
        assert archived[0]["archived_at"]
        assert archived[0]["status"] == "archived"

    def test_unarchive_round_trip(self):
        tid = history.create_research_topic("Paxos")
        history.archive_topic(tid)
        assert len(history.list_research_topics()) == 0
        assert len(history.list_archived_research_topics()) == 1

        r = history.unarchive_topic(tid)
        assert r is not None and r["id"] == tid
        assert len(history.list_research_topics()) == 1
        assert len(history.list_archived_research_topics()) == 0

    def test_archive_preserves_row_and_brief(self):
        """Archive must not mutate anything but `archived_at` — the topic
        text and its briefs are still there when we unarchive."""
        tid = history.create_research_topic("Vector clocks")
        bid = history.save_brief(tid, status="done",
                                 brief_text="a brief on time",
                                 provider="openai", model="gpt-4o-mini")
        history.archive_topic(tid)
        history.unarchive_topic(tid)

        topics = history.list_research_topics()
        assert len(topics) == 1
        assert topics[0]["text"] == "Vector clocks"
        assert topics[0]["status"] == "ready"
        b = history.latest_brief(tid)
        assert b is not None and b["id"] == bid

    def test_archive_missing_id_returns_none(self):
        assert history.archive_topic(99999) is None

    def test_unarchive_missing_id_returns_none(self):
        assert history.unarchive_topic(99999) is None

    def test_archive_refuses_non_topic_entry(self):
        """A normal transcription isn't a research topic — archiving one
        should be a noop that surfaces the mistake rather than silently
        marking a random entry as archived."""
        eid = history.save("raw", "Just a regular dictation.", 1.0)
        assert history.archive_topic(eid) is None
        # And it must still be findable
        assert history.get_entry_full(eid) is not None

    def test_unarchive_refuses_non_topic_entry(self):
        eid = history.save("raw", "Another dictation.", 1.0)
        assert history.unarchive_topic(eid) is None

    def test_archived_topics_searchable_via_fts(self):
        """Done doesn't mean gone — FTS still finds the topic text.

        This is the whole point of archive instead of delete: you can come
        back via search months later.
        """
        tid = history.create_research_topic("obscureArchivalKeyword xyz testing")
        history.archive_topic(tid)
        # FTS search (search_entries) deliberately excludes research topics,
        # so we check the base FTS join directly to confirm the row is still
        # indexed. UI-level search still surfaces archived topics through the
        # list_archived path — this test just verifies no data went missing.
        row = history._conn.execute(
            "SELECT rowid FROM transcriptions_fts "
            "WHERE transcriptions_fts MATCH ?",
            ("obscureArchivalKeyword",),
        ).fetchone()
        assert row is not None
        assert row[0] == tid

    def test_archived_list_newest_first(self):
        """`list_archived_research_topics` must order by archived_at DESC
        so the most recently finished work sits at the top."""
        import time
        a = history.create_research_topic("first")
        b = history.create_research_topic("second")
        history.archive_topic(a)
        time.sleep(1.01)  # archived_at has second granularity
        history.archive_topic(b)
        archived = history.list_archived_research_topics()
        assert [t["text"] for t in archived] == ["second", "first"]
