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
