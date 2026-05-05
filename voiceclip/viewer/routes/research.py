"""Research queue endpoints.

GET  /api/queue                  — list active topics + archived topics
POST /api/research/create        — add a new research topic
POST /api/research/update_brief  — edit a brief's text in place
POST /api/research/archive       — archive an active topic
POST /api/research/unarchive     — restore an archived topic
POST /api/research/run           — generate a brief for a topic via LLM
"""

from __future__ import annotations

import logging

from voiceclip import config, history
from voiceclip.viewer.routes import register_get, register_post

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GET /api/queue
# ---------------------------------------------------------------------------

def _get_queue(req, query):
    """Research queue listing. Ships both active and archived topics in one
    payload so the UI needs only one round-trip on tab switch."""
    status = (query.get("status") or [None])[0]
    topics = history.list_research_topics(status_filter=status)
    for t in topics:
        if t["brief_count"] > 0:
            t["brief"] = history.latest_brief(t["id"])
    archived = history.list_archived_research_topics()
    for a in archived:
        if a["brief_count"] > 0:
            a["brief"] = history.latest_brief(a["id"])
    req._json({
        "topics": topics,
        "archived": archived,
        "research_enabled": config.RESEARCH_PROVIDER != "none",
        "research_provider": config.RESEARCH_PROVIDER,
        "research_model": (
            config.RESEARCH_LOCAL_MODEL if config.RESEARCH_PROVIDER == "local"
            else config.RESEARCH_OPENAI_MODEL if config.RESEARCH_PROVIDER == "openai"
            else config.RESEARCH_ANTHROPIC_MODEL if config.RESEARCH_PROVIDER == "anthropic"
            else None
        ),
    })


# ---------------------------------------------------------------------------
# POST handlers
# ---------------------------------------------------------------------------

def _post_create(req, payload):
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        req._json({"error": "text is empty"}, status=400)
        return
    new_id = history.create_research_topic(text.strip(), app_name="Viewer")
    if new_id is None:
        req._json({"error": "failed to create topic"}, status=500)
    else:
        req._json({"ok": True, "id": new_id})


def _post_update_brief(req, payload):
    brief_id = payload.get("brief_id")
    text = payload.get("text", "")
    if not isinstance(brief_id, int) or not isinstance(text, str):
        req._json({"error": "bad payload"}, status=400)
        return
    text = text.strip()
    if not text:
        req._json({"error": "text is empty"}, status=400)
        return
    result = history.update_brief_text(brief_id, text)
    if result is None:
        req._json({"error": "brief not found"}, status=404)
    else:
        req._json({"ok": True, "brief": result})


def _post_archive(req, payload):
    entry_id = payload.get("id")
    if not isinstance(entry_id, int):
        req._json({"error": "bad id"}, status=400)
        return
    result = history.archive_topic(entry_id)
    if result is None:
        req._json({"error": "topic not found"}, status=404)
    else:
        req._json({"ok": True, "topic": result})


def _post_unarchive(req, payload):
    entry_id = payload.get("id")
    if not isinstance(entry_id, int):
        req._json({"error": "bad id"}, status=400)
        return
    result = history.unarchive_topic(entry_id)
    if result is None:
        req._json({"error": "topic not found"}, status=404)
    else:
        req._json({"ok": True, "topic": result})


def _post_run(req, payload):
    entry_id = payload.get("id")
    if not isinstance(entry_id, int):
        req._json({"error": "bad id"}, status=400)
        return
    if config.RESEARCH_PROVIDER == "none":
        req._json({
            "error": "Research is disabled. Set research.provider in ~/.voiceclip/config.json."
        }, status=400)
        return
    try:
        from voiceclip.researcher import research_topic
        brief = research_topic(entry_id)
        if brief is None:
            req._json({"error": "topic not found"}, status=404)
        else:
            req._json({"ok": True, "brief": brief})
    except RuntimeError as e:
        log.warning("Research failed: %s", e)
        req._json({"error": str(e)}, status=400)
    except Exception as e:
        log.exception("Research crashed")
        req._json({"error": f"internal error: {e}"}, status=500)


register_get("/api/queue", _get_queue)
register_post("/api/research/create", _post_create)
register_post("/api/research/update_brief", _post_update_brief)
register_post("/api/research/archive", _post_archive)
register_post("/api/research/unarchive", _post_unarchive)
register_post("/api/research/run", _post_run)
