"""Daily summary + timeline endpoints.

POST /api/summarize  — generate / regenerate a day's summary
POST /api/timeline   — generate / regenerate a day's chronological timeline

Both share the same summaries.* config for provider/model — only the
output shape differs. A single `_run_dated_llm_feature` helper handles
the validation, error humanization, and response shaping; the two routes
just pass in a different feature_func and response key.
"""

from __future__ import annotations

import logging
from datetime import datetime

from voiceclip import config
from voiceclip.routes import register_post

log = logging.getLogger(__name__)


def _run_dated_llm_feature(
    req,
    payload: dict,
    *,
    feature_name: str,
    response_key: str,
    disabled_error: str,
    feature_func,
):
    """Shared implementation for both /api/summarize and /api/timeline.

    Contract:
      - Accepts {date, force} from the payload
      - Validates the date string
      - Refuses with 400 if SUMMARIES_PROVIDER is 'none'
      - Calls feature_func(date, force=force)
      - 404 if result is None (no entries for that day)
      - RuntimeError -> 400 with the humanized message
      - Any other Exception -> 500 with 'internal error: ...'
      - Success response is {"ok": True, <response_key>: result}
    """
    date = payload.get("date") or datetime.now().strftime("%Y-%m-%d")
    force = bool(payload.get("force", False))
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        req._json({"error": "bad date"}, status=400)
        return
    if config.SUMMARIES_PROVIDER == "none":
        req._json({"error": disabled_error}, status=400)
        return
    try:
        result = feature_func(date, force=force)
        if result is None:
            req._json({"error": "no entries for that day"}, status=404)
        else:
            req._json({"ok": True, response_key: result})
    except RuntimeError as e:
        log.warning("%s failed: %s", feature_name, e)
        req._json({"error": str(e)}, status=400)
    except Exception as e:
        log.exception("%s crashed", feature_name)
        req._json({"error": f"internal error: {e}"}, status=500)


def _post_summarize(req, payload):
    from voiceclip.summarizer import summarize_day
    _run_dated_llm_feature(
        req, payload,
        feature_name="Summarize",
        response_key="summary",
        disabled_error="summaries are disabled in config",
        feature_func=summarize_day,
    )


def _post_timeline(req, payload):
    from voiceclip.summarizer import generate_timeline
    _run_dated_llm_feature(
        req, payload,
        feature_name="Timeline",
        response_key="timeline",
        disabled_error="summaries are disabled in config",
        feature_func=generate_timeline,
    )


register_post("/api/summarize", _post_summarize)
register_post("/api/timeline", _post_timeline)
