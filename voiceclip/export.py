"""Full data export — JSON or CSV dump of all history tables.

Produces a self-contained snapshot of everything in history.db:
  - transcriptions (entries, reflections, research topics)
  - day_summaries
  - day_timelines
  - research_briefs

The output is designed to be portable: no VoiceClip-specific encoding,
no binary blobs, just plain text in a standard format. A user can take
this to Obsidian, Notion, a spreadsheet, or another journaling tool.

Usage:
    voiceclip export                    # JSON to stdout
    voiceclip export --format csv -o ~/export.csv
    voiceclip export -o ~/backup.json   # JSON to file
"""

from __future__ import annotations

import csv
import io
import json
import logging

from voiceclip import history

log = logging.getLogger(__name__)


def _fetch_all_entries() -> list[dict]:
    """Return every row from the transcriptions table as a list of dicts."""
    if history._conn is None:
        return []
    rows = history._conn.execute(
        "SELECT id, timestamp, raw_text, formatted_text, duration_seconds, "
        "       persona, model, word_count, kind, app_name, window_title, "
        "       edited_at, is_research_topic, archived_at "
        "FROM transcriptions ORDER BY id ASC"
    ).fetchall()
    cols = [
        "id", "timestamp", "raw_text", "formatted_text", "duration_seconds",
        "persona", "model", "word_count", "kind", "app_name", "window_title",
        "edited_at", "is_research_topic", "archived_at",
    ]
    return [dict(zip(cols, row, strict=True)) for row in rows]


def _fetch_all_summaries() -> list[dict]:
    """Return every cached day summary."""
    if history._conn is None:
        return []
    rows = history._conn.execute(
        "SELECT date, summary, provider, model, style, generated_at, entry_count "
        "FROM day_summaries ORDER BY date ASC"
    ).fetchall()
    cols = ["date", "summary", "provider", "model", "style", "generated_at", "entry_count"]
    return [dict(zip(cols, row, strict=True)) for row in rows]


def _fetch_all_timelines() -> list[dict]:
    """Return every cached day timeline."""
    if history._conn is None:
        return []
    rows = history._conn.execute(
        "SELECT date, timeline, provider, model, generated_at, entry_count "
        "FROM day_timelines ORDER BY date ASC"
    ).fetchall()
    cols = ["date", "timeline", "provider", "model", "generated_at", "entry_count"]
    return [dict(zip(cols, row, strict=True)) for row in rows]


def _fetch_all_briefs() -> list[dict]:
    """Return every research brief."""
    if history._conn is None:
        return []
    rows = history._conn.execute(
        "SELECT id, entry_id, status, brief_text, sources_json, provider, "
        "       model, used_web_search, generated_at, error "
        "FROM research_briefs ORDER BY id ASC"
    ).fetchall()
    cols = [
        "id", "entry_id", "status", "brief_text", "sources_json", "provider",
        "model", "used_web_search", "generated_at", "error",
    ]
    results = []
    for row in rows:
        d = dict(zip(cols, row, strict=True))
        # Parse sources_json into a real list for the JSON export
        if d.get("sources_json"):
            try:
                d["sources"] = json.loads(d["sources_json"])
            except Exception:
                d["sources"] = []
        else:
            d["sources"] = []
        del d["sources_json"]
        d["used_web_search"] = bool(d["used_web_search"])
        results.append(d)
    return results


def export_all(fmt: str = "json") -> str:
    """Export all history data as a string in the requested format.

    Supported formats:
      - "json": a single JSON object with keys for each table
      - "csv": entries table only (flat format; summaries/timelines/briefs
        don't fit CSV well, so they're omitted with a note)

    Returns the formatted string. Caller decides whether to print or write.
    """
    entries = _fetch_all_entries()
    summaries = _fetch_all_summaries()
    timelines = _fetch_all_timelines()
    briefs = _fetch_all_briefs()

    if fmt == "csv":
        return _to_csv(entries)

    # JSON — full dump of all tables
    payload = {
        "exported_at": _now_iso(),
        "format_version": 1,
        "tables": {
            "entries": {
                "count": len(entries),
                "rows": entries,
            },
            "day_summaries": {
                "count": len(summaries),
                "rows": summaries,
            },
            "day_timelines": {
                "count": len(timelines),
                "rows": timelines,
            },
            "research_briefs": {
                "count": len(briefs),
                "rows": briefs,
            },
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _to_csv(entries: list[dict]) -> str:
    """Convert entries to CSV. Only the entries table — summaries/timelines
    don't fit a flat CSV shape well."""
    if not entries:
        return ""
    buf = io.StringIO()
    # Use a stable column order
    cols = [
        "id", "timestamp", "kind", "formatted_text", "duration_seconds",
        "word_count", "app_name", "persona", "model", "edited_at",
        "is_research_topic", "archived_at",
    ]
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for entry in entries:
        writer.writerow(entry)
    return buf.getvalue()


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Import — restore from a JSON export file
# ---------------------------------------------------------------------------

def import_from_file(path: str, *, merge: bool = True) -> dict:
    """Import a VoiceClip JSON export into the local database.

    Args:
        path: path to the JSON file (produced by `voiceclip export`)
        merge: if True (default), skip rows whose primary key already
               exists. If False, raise on any conflict.

    Returns a summary dict: {entries, summaries, timelines, briefs} with
    counts of rows imported per table.

    Raises RuntimeError on format errors or if history isn't initialized.
    """
    if history._conn is None:
        raise RuntimeError("History database not initialized. Run voiceclip first.")

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Could not read export file: {e}") from e

    if not isinstance(data, dict) or "tables" not in data:
        raise RuntimeError(
            "Invalid export file — expected a JSON object with a 'tables' key. "
            "Was this produced by `voiceclip export`?"
        )

    tables = data["tables"]
    counts = {"entries": 0, "summaries": 0, "timelines": 0, "briefs": 0}

    # --- Entries ---
    entries = (tables.get("entries") or {}).get("rows") or []
    for e in entries:
        if _entry_exists(e.get("id")):
            if merge:
                continue
            raise RuntimeError(f"Entry id={e['id']} already exists (use --merge to skip)")
        _insert_entry(e)
        counts["entries"] += 1

    # --- Day summaries ---
    summaries = (tables.get("day_summaries") or {}).get("rows") or []
    for s in summaries:
        if _summary_exists(s.get("date")):
            if merge:
                continue
            raise RuntimeError(f"Summary for {s['date']} already exists")
        _insert_summary(s)
        counts["summaries"] += 1

    # --- Day timelines ---
    timelines = (tables.get("day_timelines") or {}).get("rows") or []
    for t in timelines:
        if _timeline_exists(t.get("date")):
            if merge:
                continue
            raise RuntimeError(f"Timeline for {t['date']} already exists")
        _insert_timeline(t)
        counts["timelines"] += 1

    # --- Research briefs ---
    briefs = (tables.get("research_briefs") or {}).get("rows") or []
    for b in briefs:
        if _brief_exists(b.get("id")):
            if merge:
                continue
            raise RuntimeError(f"Brief id={b['id']} already exists")
        _insert_brief(b)
        counts["briefs"] += 1

    history._conn.commit()
    return counts


# ---------------------------------------------------------------------------
# Import helpers — low-level inserts
# ---------------------------------------------------------------------------

def _entry_exists(entry_id) -> bool:
    if entry_id is None or history._conn is None:
        return False
    row = history._conn.execute(
        "SELECT 1 FROM transcriptions WHERE id = ?", (entry_id,)
    ).fetchone()
    return row is not None


def _insert_entry(e: dict):
    history._conn.execute(
        "INSERT INTO transcriptions "
        "(id, timestamp, raw_text, formatted_text, duration_seconds, "
        " persona, model, word_count, kind, app_name, window_title, "
        " edited_at, is_research_topic, archived_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            e.get("id"),
            e.get("timestamp"),
            e.get("raw_text") or "",
            e.get("formatted_text") or "",
            e.get("duration_seconds") or 0,
            e.get("persona"),
            e.get("model"),
            e.get("word_count") or 0,
            e.get("kind") or "transcription",
            e.get("app_name"),
            e.get("window_title"),
            e.get("edited_at"),
            1 if e.get("is_research_topic") else 0,
            e.get("archived_at"),
        ),
    )


def _summary_exists(date) -> bool:
    if date is None or history._conn is None:
        return False
    row = history._conn.execute(
        "SELECT 1 FROM day_summaries WHERE date = ?", (date,)
    ).fetchone()
    return row is not None


def _insert_summary(s: dict):
    history._conn.execute(
        "INSERT INTO day_summaries "
        "(date, summary, provider, model, style, generated_at, entry_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            s.get("date"),
            s.get("summary", ""),
            s.get("provider", "unknown"),
            s.get("model", "unknown"),
            s.get("style", "descriptive"),
            s.get("generated_at", ""),
            s.get("entry_count", 0),
        ),
    )


def _timeline_exists(date) -> bool:
    if date is None or history._conn is None:
        return False
    row = history._conn.execute(
        "SELECT 1 FROM day_timelines WHERE date = ?", (date,)
    ).fetchone()
    return row is not None


def _insert_timeline(t: dict):
    history._conn.execute(
        "INSERT INTO day_timelines "
        "(date, timeline, provider, model, generated_at, entry_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            t.get("date"),
            t.get("timeline", ""),
            t.get("provider", "unknown"),
            t.get("model", "unknown"),
            t.get("generated_at", ""),
            t.get("entry_count", 0),
        ),
    )


def _brief_exists(brief_id) -> bool:
    if brief_id is None or history._conn is None:
        return False
    row = history._conn.execute(
        "SELECT 1 FROM research_briefs WHERE id = ?", (brief_id,)
    ).fetchone()
    return row is not None


def _insert_brief(b: dict):
    sources = b.get("sources") or []
    history._conn.execute(
        "INSERT INTO research_briefs "
        "(id, entry_id, status, brief_text, sources_json, provider, "
        " model, used_web_search, generated_at, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            b.get("id"),
            b.get("entry_id"),
            b.get("status", "done"),
            b.get("brief_text"),
            json.dumps(sources) if sources else None,
            b.get("provider"),
            b.get("model"),
            1 if b.get("used_web_search") else 0,
            b.get("generated_at"),
            b.get("error"),
        ),
    )
