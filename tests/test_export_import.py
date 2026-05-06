"""Stress tests for voiceclip export/import — 5 user scenarios.

Each scenario simulates a different real-world edge case that could
break the import flow. Tests use the live_history fixture (fresh DB
per test) so they're isolated.
"""

import json
import os
import tempfile

import pytest

from voiceclip import config, history
from voiceclip.export import export_all, import_from_file

# ---------------------------------------------------------------------------
# Scenario 1: Fresh device — import into a completely empty database
# ---------------------------------------------------------------------------
# User: Alice just got a new Mac. She installs VoiceClip, has zero entries,
# and imports her 500-entry backup from the old machine.

class TestScenario1_FreshDevice:
    def test_import_into_empty_db(self, live_history):
        """Import 100 entries + summaries + briefs into a fresh DB."""
        # Seed a fake export with 100 entries, 5 summaries, 3 briefs
        payload = _build_export(
            n_entries=100,
            n_summaries=5,
            n_timelines=3,
            n_briefs=10,
        )
        path = _write_tmp_json(payload)

        counts = import_from_file(path, merge=True)

        assert counts["entries"] == 100
        assert counts["summaries"] == 5
        assert counts["timelines"] == 3
        assert counts["briefs"] == 10
        # Verify they're actually in the DB
        assert history.count() == 100

    def test_import_preserves_entry_ids(self, live_history):
        """Original IDs from the export must be preserved — cross-table
        references (briefs → entry_id) depend on this."""
        payload = _build_export(n_entries=5, start_id=42)
        path = _write_tmp_json(payload)

        import_from_file(path)

        # Entry IDs should be 42, 43, 44, 45, 46
        for i in range(5):
            entry = history.get_by_id(42 + i)
            assert entry is not None, f"Entry id={42+i} not found"


# ---------------------------------------------------------------------------
# Scenario 2: Merge — both devices have data, some overlapping
# ---------------------------------------------------------------------------
# User: Bob uses VoiceClip on both his work Mac and personal Mac. He exports
# from work (IDs 1-50) and imports into personal (which already has IDs 1-20
# from its own use). The overlapping IDs 1-20 should be skipped; 21-50 added.

class TestScenario2_MergeOverlap:
    def test_overlapping_ids_are_skipped(self, live_history):
        """Existing entries with the same ID are not overwritten."""
        # Pre-populate local DB with entries 1-20
        for i in range(20):
            history.save("raw", f"local entry {i}", 1.0, kind="transcription")

        # Build an export with IDs 1-50 (overlaps 1-20)
        payload = _build_export(n_entries=50, start_id=1)
        path = _write_tmp_json(payload)

        counts = import_from_file(path, merge=True)

        # Only 30 new entries (IDs 21-50) should be imported
        assert counts["entries"] == 30
        # Total should be 50 (20 original + 30 imported)
        assert history.count() == 50
        # Original entries should be unchanged (not overwritten)
        entry_1 = history.get_by_id(1)
        assert "local entry 0" in entry_1  # original text preserved

    def test_no_merge_fails_on_conflict(self, live_history):
        """--no-merge mode raises on the first conflicting ID."""
        history.save("raw", "existing", 1.0)
        payload = _build_export(n_entries=5, start_id=1)
        path = _write_tmp_json(payload)

        with pytest.raises(RuntimeError, match="already exists"):
            import_from_file(path, merge=False)


# ---------------------------------------------------------------------------
# Scenario 3: Corrupt / malformed export file
# ---------------------------------------------------------------------------
# User: Charlie accidentally truncated the export file, or it's from a
# different app entirely, or it's an older format version.

class TestScenario3_CorruptFile:
    def test_empty_file(self, live_history):
        path = _write_tmp_string("")
        with pytest.raises(RuntimeError, match="Could not read"):
            import_from_file(path)

    def test_not_json(self, live_history):
        path = _write_tmp_string("this is not json at all")
        with pytest.raises(RuntimeError, match="Could not read"):
            import_from_file(path)

    def test_json_but_wrong_shape(self, live_history):
        """Valid JSON but missing the 'tables' key."""
        path = _write_tmp_json({"entries": [{"id": 1}]})
        with pytest.raises(RuntimeError, match="Invalid export file"):
            import_from_file(path)

    def test_missing_file(self, live_history):
        with pytest.raises(RuntimeError, match="Could not read"):
            import_from_file("/nonexistent/path/backup.json")

    def test_entries_with_missing_fields(self, live_history):
        """Entries that are missing optional fields should still import
        with sensible defaults (not crash)."""
        payload = {
            "exported_at": "2026-01-01T00:00:00",
            "format_version": 1,
            "tables": {
                "entries": {
                    "count": 2,
                    "rows": [
                        # Minimal entry — only required fields
                        {"id": 999, "timestamp": "2026-01-01T10:00:00",
                         "formatted_text": "hello"},
                        # Entry with all fields null/missing
                        {"id": 1000, "timestamp": "2026-01-01T11:00:00",
                         "formatted_text": "world", "kind": None,
                         "app_name": None, "duration_seconds": None},
                    ],
                },
                "day_summaries": {"count": 0, "rows": []},
                "day_timelines": {"count": 0, "rows": []},
                "research_briefs": {"count": 0, "rows": []},
            },
        }
        path = _write_tmp_json(payload)
        counts = import_from_file(path)
        assert counts["entries"] == 2
        # Verify defaults were applied
        entry = history.get_entry_full(999)
        assert entry is not None
        assert entry["kind"] == "transcription"  # default


# ---------------------------------------------------------------------------
# Scenario 4: Large export — 10,000 entries
# ---------------------------------------------------------------------------
# User: Diana has been using VoiceClip for a year. Her export has 10k entries.
# Import should complete in reasonable time without OOM or SQLite issues.

class TestScenario4_LargeExport:
    def test_10k_entries_import(self, live_history):
        """10,000 entries should import without error or timeout."""
        payload = _build_export(n_entries=10000)
        path = _write_tmp_json(payload)

        counts = import_from_file(path)

        assert counts["entries"] == 10000
        assert history.count() == 10000

    def test_large_export_round_trip(self, live_history):
        """Export 1000 entries, import into fresh DB, verify count matches."""
        # Seed 1000 entries
        for i in range(1000):
            history.save("raw", f"entry number {i}", 1.0, kind="transcription")

        # Export
        data = export_all(fmt="json")
        path = _write_tmp_string(data)

        # Clear and reimport
        history._conn.execute("DELETE FROM transcriptions")
        history._conn.commit()
        assert history.count() == 0

        counts = import_from_file(path)
        assert counts["entries"] == 1000
        assert history.count() == 1000


# ---------------------------------------------------------------------------
# Scenario 5: Cross-table references — briefs pointing at entry IDs
# ---------------------------------------------------------------------------
# User: Eve has research topics with completed briefs. The brief rows
# reference entry_id. If entries import with their original IDs, the
# briefs' foreign keys should still be valid.

class TestScenario5_CrossTableRefs:
    def test_briefs_reference_imported_entries(self, live_history):
        """Research briefs should correctly reference their parent entries
        after import — the entry_id FK must match."""
        payload = {
            "exported_at": "2026-01-01T00:00:00",
            "format_version": 1,
            "tables": {
                "entries": {
                    "count": 2,
                    "rows": [
                        {"id": 100, "timestamp": "2026-01-01T10:00:00",
                         "raw_text": "CRDTs", "formatted_text": "CRDTs",
                         "kind": "transcription", "is_research_topic": 1,
                         "duration_seconds": 0, "word_count": 1},
                        {"id": 101, "timestamp": "2026-01-01T11:00:00",
                         "raw_text": "Raft consensus", "formatted_text": "Raft consensus",
                         "kind": "transcription", "is_research_topic": 1,
                         "duration_seconds": 0, "word_count": 2},
                    ],
                },
                "day_summaries": {"count": 0, "rows": []},
                "day_timelines": {"count": 0, "rows": []},
                "research_briefs": {
                    "count": 2,
                    "rows": [
                        {"id": 50, "entry_id": 100, "status": "done",
                         "brief_text": "CRDTs are conflict-free...",
                         "sources": [{"title": "Wiki", "url": "https://en.wikipedia.org/wiki/CRDT"}],
                         "provider": "openai", "model": "gpt-4o-mini",
                         "used_web_search": True, "generated_at": "2026-01-01T12:00:00"},
                        {"id": 51, "entry_id": 101, "status": "done",
                         "brief_text": "Raft is a consensus algorithm...",
                         "sources": [], "provider": "local", "model": "stub",
                         "used_web_search": False, "generated_at": "2026-01-01T13:00:00"},
                    ],
                },
            },
        }
        path = _write_tmp_json(payload)
        counts = import_from_file(path)

        assert counts["entries"] == 2
        assert counts["briefs"] == 2

        # Verify the FK relationship holds
        brief = history.latest_brief(100)
        assert brief is not None
        assert "conflict-free" in brief["text"]
        assert brief["sources"][0]["url"] == "https://en.wikipedia.org/wiki/CRDT"

        brief2 = history.latest_brief(101)
        assert brief2 is not None
        assert "consensus" in brief2["text"]

    def test_summary_and_timeline_for_same_date(self, live_history):
        """A day can have both a summary and a timeline — both should import."""
        payload = {
            "exported_at": "2026-01-01T00:00:00",
            "format_version": 1,
            "tables": {
                "entries": {"count": 0, "rows": []},
                "day_summaries": {
                    "count": 1,
                    "rows": [{"date": "2026-04-28", "summary": "You coded all day.",
                              "provider": "local", "model": "stub",
                              "style": "descriptive", "generated_at": "2026-04-28T23:00:00",
                              "entry_count": 15}],
                },
                "day_timelines": {
                    "count": 1,
                    "rows": [{"date": "2026-04-28", "timeline": "9-12: coding. 2-5: meetings.",
                              "provider": "local", "model": "stub",
                              "generated_at": "2026-04-28T23:01:00",
                              "entry_count": 15}],
                },
                "research_briefs": {"count": 0, "rows": []},
            },
        }
        path = _write_tmp_json(payload)
        counts = import_from_file(path)

        assert counts["summaries"] == 1
        assert counts["timelines"] == 1

        s = history.get_day_summary("2026-04-28")
        assert s is not None
        assert "coded all day" in s["summary"]

        t = history.get_day_timeline("2026-04-28")
        assert t is not None
        assert "9-12" in t["timeline"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_export(
    n_entries: int = 10,
    n_summaries: int = 0,
    n_timelines: int = 0,
    n_briefs: int = 0,
    start_id: int = 1,
) -> dict:
    """Build a synthetic export payload for testing."""
    entries = []
    for i in range(n_entries):
        entries.append({
            "id": start_id + i,
            "timestamp": f"2026-04-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00",
            "raw_text": f"raw entry {i}",
            "formatted_text": f"formatted entry {i}",
            "duration_seconds": 2.5,
            "persona": "default",
            "model": "large-v3-turbo",
            "word_count": 3,
            "kind": "reflection" if i % 5 == 0 else "transcription",
            "app_name": "Slack" if i % 3 == 0 else "Notes",
            "window_title": None,
            "edited_at": None,
            "is_research_topic": 0,
            "archived_at": None,
        })

    summaries = []
    for i in range(n_summaries):
        summaries.append({
            "date": f"2026-04-{i + 1:02d}",
            "summary": f"Day {i+1} summary text.",
            "provider": "local",
            "model": "stub",
            "style": "descriptive",
            "generated_at": f"2026-04-{i + 1:02d}T23:00:00",
            "entry_count": 10,
        })

    timelines = []
    for i in range(n_timelines):
        timelines.append({
            "date": f"2026-04-{i + 1:02d}",
            "timeline": f"Morning: task A. Afternoon: task B. Day {i+1}.",
            "provider": "local",
            "model": "stub",
            "generated_at": f"2026-04-{i + 1:02d}T23:01:00",
            "entry_count": 10,
        })

    briefs = []
    for i in range(n_briefs):
        briefs.append({
            "id": 1000 + i,
            "entry_id": start_id + (i % n_entries) if n_entries > 0 else 1,
            "status": "done",
            "brief_text": f"Brief {i} content.",
            "sources": [],
            "provider": "local",
            "model": "stub",
            "used_web_search": False,
            "generated_at": f"2026-04-01T{i:02d}:00:00",
            "error": None,
        })

    return {
        "exported_at": "2026-05-01T00:00:00",
        "format_version": 1,
        "tables": {
            "entries": {"count": len(entries), "rows": entries},
            "day_summaries": {"count": len(summaries), "rows": summaries},
            "day_timelines": {"count": len(timelines), "rows": timelines},
            "research_briefs": {"count": len(briefs), "rows": briefs},
        },
    }


def _write_tmp_json(data: dict) -> str:
    """Write a dict as JSON to a temp file, return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f, ensure_ascii=False)
    f.close()
    return f.name


def _write_tmp_string(content: str) -> str:
    """Write a raw string to a temp file, return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    f.write(content)
    f.close()
    return f.name
