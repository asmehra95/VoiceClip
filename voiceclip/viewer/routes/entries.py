"""Journal entry endpoints — day views, search, edit, delete, promote.

GET  /api/days          — list distinct days that have entries
GET  /api/day           — full day payload (entries + stats + summary + timeline)
GET  /api/search        — full-text search across all history
POST /api/update        — edit an entry's text
POST /api/delete        — delete an entry
POST /api/promote       — promote a transcription to a reflection
"""

from __future__ import annotations

from datetime import datetime, timedelta

from voiceclip import config, history
from voiceclip.viewer.routes import register_get, register_post

# ---------------------------------------------------------------------------
# Helpers — kept local to the entries module because they're only used by
# the day payload construction below.
# ---------------------------------------------------------------------------

def _friendly_day_label(date_str: str) -> str:
    """Return 'Today' / 'Yesterday' / 'Monday' / full date for the page header."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return date_str
    today = datetime.now().date()
    if d == today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    if (today - d).days < 7:
        return d.strftime("%A")  # Monday, Tuesday, etc.
    return d.strftime("%B %-d, %Y")


def _adjacent_days(date_str: str) -> tuple[str | None, str | None]:
    """Return (previous_day_with_entries, next_day_with_entries)."""
    days = history.list_days(limit=365)
    if not days:
        return None, None
    if date_str not in days:
        return (days[0] if days else None), None
    idx = days.index(date_str)
    newer = days[idx - 1] if idx > 0 else None
    older = days[idx + 1] if idx + 1 < len(days) else None
    return older, newer


def _day_payload(date_str: str, include_summary: bool = True) -> dict:
    entries = history.entries_for_day(date_str)
    stats = history.day_stats(date_str)
    prev_day, next_day = _adjacent_days(date_str)

    summary_block: dict | None = None
    summary_error: str | None = None
    timeline_block: dict | None = None
    if include_summary and config.SUMMARIES_PROVIDER != "none" and entries:
        cached = history.get_day_summary(date_str)
        if cached:
            summary_block = cached
        cached_tl = history.get_day_timeline(date_str)
        if cached_tl:
            timeline_block = cached_tl

    # Resolve the currently-active model id for the configured provider.
    active_model = None
    if config.SUMMARIES_PROVIDER == "local":
        active_model = config.SUMMARIES_LOCAL_MODEL
    elif config.SUMMARIES_PROVIDER == "openai":
        active_model = config.SUMMARIES_OPENAI_MODEL
    elif config.SUMMARIES_PROVIDER == "anthropic":
        active_model = config.SUMMARIES_ANTHROPIC_MODEL

    return {
        "date": date_str,
        "day_label": _friendly_day_label(date_str),
        "stats": stats,
        "entries": entries,
        "prev_day": prev_day,
        "next_day": next_day,
        "summary": summary_block,
        "summary_enabled": config.SUMMARIES_PROVIDER != "none",
        "summary_provider": config.SUMMARIES_PROVIDER,
        "summary_model": active_model,
        "summary_error": summary_error,
        "timeline": timeline_block,
    }


# ---------------------------------------------------------------------------
# GET handlers
# ---------------------------------------------------------------------------

def _get_days(req, query):
    req._json({"days": history.list_days(limit=180)})


def _get_day(req, query):
    date = (query.get("date") or [datetime.now().strftime("%Y-%m-%d")])[0]
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        req._json({"error": "bad date"}, status=400)
        return
    req._json(_day_payload(date))


def _get_search(req, query):
    term = (query.get("q") or [""])[0]
    kind = (query.get("kind") or [None])[0]
    if kind not in (None, "transcription", "reflection"):
        kind = None
    if not term.strip():
        req._json({"entries": [], "query": ""})
        return
    entries = history.search_entries(term, limit=50, kind=kind)
    req._json({"entries": entries, "query": term})


# ---------------------------------------------------------------------------
# POST handlers
# ---------------------------------------------------------------------------

def _post_update(req, payload):
    entry_id = payload.get("id")
    text = payload.get("text", "")
    if not isinstance(entry_id, int) or not isinstance(text, str):
        req._json({"error": "bad payload"}, status=400)
        return
    text = text.strip()
    if not text:
        req._json({"error": "text is empty"}, status=400)
        return
    result = history.update_text(entry_id, text)
    if result is None:
        req._json({"error": "not found"}, status=404)
    else:
        req._json({"ok": True, "entry": result})


def _post_delete(req, payload):
    entry_id = payload.get("id")
    if not isinstance(entry_id, int):
        req._json({"error": "bad id"}, status=400)
        return
    result = history.delete_entry(entry_id)
    if result is None:
        req._json({"error": "not found"}, status=404)
    else:
        req._json({"ok": True, "entry": result})


def _post_promote(req, payload):
    entry_id = payload.get("id")
    if not isinstance(entry_id, int):
        req._json({"error": "bad id"}, status=400)
        return
    result = history.promote_to_reflection(entry_id=entry_id)
    if result is None:
        req._json({"error": "not found or already a reflection"}, status=404)
    else:
        req._json({"ok": True, "entry": result})


register_get("/api/days", _get_days)
register_get("/api/day", _get_day)
register_get("/api/search", _get_search)
register_post("/api/update", _post_update)
register_post("/api/delete", _post_delete)
register_post("/api/promote", _post_promote)
