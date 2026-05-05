"""Patterns endpoints — longitudinal coach across recent days.

GET  /api/patterns/config  — provider/model/window info for the empty state
POST /api/patterns/run     — generate a fresh patterns view (slow on local LLM)
POST /api/patterns/queue   — take a suggested topic and add it to research
"""

from __future__ import annotations

import logging

from voiceclip import config, history
from voiceclip.viewer.routes import register_get, register_post

log = logging.getLogger(__name__)


def _get_config(req, query):
    """Shape the Patterns tab needs to render empty-state or the
    Generate button without kicking off an LLM call."""
    active_model = None
    if config.PATTERNS_PROVIDER == "local":
        active_model = config.PATTERNS_LOCAL_MODEL
    elif config.PATTERNS_PROVIDER == "openai":
        active_model = config.PATTERNS_OPENAI_MODEL
    elif config.PATTERNS_PROVIDER == "anthropic":
        active_model = config.PATTERNS_ANTHROPIC_MODEL
    req._json({
        "enabled": config.PATTERNS_PROVIDER != "none",
        "provider": config.PATTERNS_PROVIDER,
        "model": active_model,
        "window_days": config.PATTERNS_WINDOW_DAYS,
    })


def _post_run(req, payload):
    if config.PATTERNS_PROVIDER == "none":
        req._json({
            "error": "Patterns are off. Set patterns.provider in ~/.voiceclip/config.json."
        }, status=400)
        return
    try:
        from voiceclip.patterns import generate_patterns
        window = payload.get("window_days")
        if not isinstance(window, int) or window < 1:
            window = None
        force = bool(payload.get("force", False))
        result = generate_patterns(window_days=window, force=force)
        req._json({"ok": True, "patterns": result})
    except RuntimeError as e:
        log.warning("Patterns failed: %s", e)
        req._json({"error": str(e)}, status=400)
    except Exception as e:
        log.exception("Patterns crashed")
        req._json({"error": f"internal error: {e}"}, status=500)


def _post_queue(req, payload):
    """Take a suggested topic from Patterns and add it to the research
    queue. Dedup: if the topic text already exists as a research topic,
    return a reference to the existing one."""
    topic_text = (payload.get("topic") or "").strip()
    reason = (payload.get("reason") or "").strip()
    quote = (payload.get("grounding_quote") or "").strip()
    if not topic_text:
        req._json({"error": "topic is empty"}, status=400)
        return

    existing = history.find_research_topic_by_text(topic_text)
    if existing:
        req._json({
            "ok": True, "duplicate": True,
            "id": existing["id"],
            "text": existing["text"],
        })
        return

    # Compose the stored text: topic + short provenance note.
    context_bits = []
    if reason:
        context_bits.append(reason)
    if quote:
        context_bits.append(f'"{quote}"')
    suffix = ""
    if context_bits:
        suffix = "\n\n(suggested from your reflections: " + " — ".join(context_bits) + ")"
    full_text = topic_text + suffix

    new_id = history.create_research_topic(full_text, app_name="Patterns")
    if new_id is None:
        req._json({"error": "failed to create topic"}, status=500)
    else:
        req._json({"ok": True, "id": new_id, "duplicate": False})


register_get("/api/patterns/config", _get_config)
register_post("/api/patterns/run", _post_run)
register_post("/api/patterns/queue", _post_queue)
