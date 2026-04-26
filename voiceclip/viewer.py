"""Lightweight local web viewer for VoiceClip history.

Serves a single HTML page from localhost (never binds to 0.0.0.0).
Zero dependencies — uses only Python's stdlib http.server.

Views:
- Day-first dashboard: narrative summary (if enabled), app bar, reflections,
  collapsible transcriptions
- Day navigation: Previous / Today / Next + date jump
- Read-only (no edits, no deletes). Promote-to-reflection is the one write.

Start with:  voiceclip view
Stop with:   Ctrl+C
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import sys
import threading
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from voiceclip import config, history

log = logging.getLogger(__name__)

# Static assets live alongside this module. Extracted from the previously
# embedded `_PAGE_HTML` heredoc — no behavioural change, just maintainable.
_STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# API helpers
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
    # Days are newest-first. Find current position.
    if date_str not in days:
        # Pick the nearest day
        return (days[0] if days else None), None
    idx = days.index(date_str)
    newer = days[idx - 1] if idx > 0 else None
    older = days[idx + 1] if idx + 1 < len(days) else None
    return older, newer  # previous = older, next = newer


def _day_payload(date_str: str, include_summary: bool = True) -> dict:
    entries = history.entries_for_day(date_str)
    stats = history.day_stats(date_str)
    prev_day, next_day = _adjacent_days(date_str)

    summary_block: dict | None = None
    summary_error: str | None = None
    if include_summary and config.SUMMARIES_PROVIDER != "none" and entries:
        cached = history.get_day_summary(date_str)
        if cached:
            summary_block = cached

    # Resolve the currently-active model id for the configured provider.
    # Shown on the "Generate" card so the user knows what's about to run.
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
    }


# ---------------------------------------------------------------------------
# Settings (UI-editable config subset)
# ---------------------------------------------------------------------------
# A curated whitelist of config keys the viewer's Settings tab can read and
# write. Everything else (personas dictionary, model paths, retention
# policies) stays in config.json as power-user territory.
#
# `restart_required` marks fields that only take effect after the daemon
# reboots — the UI shows a banner when any of these change. Fields without
# it apply immediately (they're only read by the viewer or on-demand paths).

_SETTINGS_SCHEMA: dict[str, dict] = {
    # Dictation
    "hotkey": {
        "group": "Dictation",
        "type": "select",
        "restart_required": True,
        "choices": ["alt_r", "alt_l", "ctrl_r", "ctrl_l", "shift_r", "shift_l",
                    "cmd_r", "cmd_l", "caps_lock", "f1", "f2", "f3", "f4",
                    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                    "space", "esc"],
    },
    "hotkey_mode": {
        "group": "Dictation",
        "type": "select",
        "restart_required": True,
        "choices": ["hold", "toggle"],
    },
    "english_only": {
        "group": "Dictation",
        "type": "bool",
        "restart_required": True,
    },

    # Reflections
    "reflection_hotkey": {
        "group": "Reflections",
        "type": "select_or_none",
        "restart_required": True,
        "choices": ["alt_r", "alt_l", "ctrl_r", "ctrl_l", "shift_r", "shift_l",
                    "cmd_r", "cmd_l", "caps_lock", "f1", "f2", "f3", "f4",
                    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                    "space", "esc"],
    },
    "reflection_hotkey_mode": {
        "group": "Reflections",
        "type": "select",
        "restart_required": True,
        "choices": ["hold", "toggle"],
    },
    "window_title_capture": {
        "group": "Reflections",
        "type": "bool",
        "restart_required": True,
    },

    # Journal (history)
    "history": {
        "group": "Journal",
        "type": "bool",
        "restart_required": True,
    },

    # Summaries
    "summaries.provider": {
        "group": "Summaries",
        "type": "select",
        "cloud_providers": ["openai", "anthropic"],
        "choices": ["none", "local", "openai", "anthropic"],
    },
    "summaries.local_model": {
        "group": "Summaries",
        "type": "text",
        "placeholder": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    },
    "summaries.openai_model": {
        "group": "Summaries",
        "type": "text",
        "placeholder": "gpt-4o-mini",
    },
    "summaries.anthropic_model": {
        "group": "Summaries",
        "type": "text",
        "placeholder": "claude-haiku-4-5",
    },
    "summaries.style": {
        "group": "Summaries",
        "type": "select",
        "choices": ["descriptive", "reflective"],
    },

    # Research
    "research.provider": {
        "group": "Research",
        "type": "select",
        "cloud_providers": ["openai", "anthropic"],
        "choices": ["none", "openai", "anthropic"],
    },
    "research.openai_model": {
        "group": "Research",
        "type": "text",
        "placeholder": "gpt-4o-mini",
    },
    "research.anthropic_model": {
        "group": "Research",
        "type": "text",
        "placeholder": "claude-haiku-4-5",
    },

    # Patterns
    "patterns.provider": {
        "group": "Patterns",
        "type": "select",
        "cloud_providers": ["openai", "anthropic"],
        "choices": ["none", "local", "openai", "anthropic"],
    },
    "patterns.local_model": {
        "group": "Patterns",
        "type": "text",
        "placeholder": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    },
    "patterns.openai_model": {
        "group": "Patterns",
        "type": "text",
        "placeholder": "gpt-4o-mini",
    },
    "patterns.anthropic_model": {
        "group": "Patterns",
        "type": "text",
        "placeholder": "claude-haiku-4-5",
    },
    "patterns.window_days": {
        "group": "Patterns",
        "type": "int",
        "min": 1,
        "max": 90,
    },
}


def _get_nested(d: dict, key: str):
    """Resolve a dotted key ('summaries.provider') against a nested dict."""
    parts = key.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _runtime_value(key: str):
    """Read the current runtime value for a dotted config key."""
    mapping = {
        "hotkey": config.HOTKEY,
        "hotkey_mode": config.HOTKEY_MODE,
        "english_only": config.ENGLISH_ONLY,
        "reflection_hotkey": config.REFLECTION_HOTKEY,
        "reflection_hotkey_mode": config.REFLECTION_HOTKEY_MODE,
        "window_title_capture": False,  # not currently in runtime config; default
        "history": config.HISTORY_ENABLED,
        "summaries.provider": config.SUMMARIES_PROVIDER,
        "summaries.local_model": config.SUMMARIES_LOCAL_MODEL,
        "summaries.openai_model": config.SUMMARIES_OPENAI_MODEL,
        "summaries.anthropic_model": config.SUMMARIES_ANTHROPIC_MODEL,
        "summaries.style": config.SUMMARIES_STYLE,
        "research.provider": config.RESEARCH_PROVIDER,
        "research.openai_model": config.RESEARCH_OPENAI_MODEL,
        "research.anthropic_model": config.RESEARCH_ANTHROPIC_MODEL,
        "patterns.provider": config.PATTERNS_PROVIDER,
        "patterns.local_model": config.PATTERNS_LOCAL_MODEL,
        "patterns.openai_model": config.PATTERNS_OPENAI_MODEL,
        "patterns.anthropic_model": config.PATTERNS_ANTHROPIC_MODEL,
        "patterns.window_days": config.PATTERNS_WINDOW_DAYS,
    }
    return mapping.get(key)


def _settings_payload() -> dict:
    """Build the GET /api/settings response: schema + current values.

    The UI uses `schema` to render inputs (select vs bool vs text, groups,
    choices, restart_required flags, cloud-provider flag for the confirm
    modal) and `values` to pre-fill them.
    """
    values = {}
    for key in _SETTINGS_SCHEMA:
        values[key] = _runtime_value(key)
    return {
        "schema": _SETTINGS_SCHEMA,
        "values": values,
        # System info — a subset of `voiceclip doctor` that the Settings
        # tab shows in a collapsible at the bottom.
        "system": _system_info(),
    }


def _system_info() -> dict:
    """Snapshot of the same stuff `voiceclip doctor` surfaces."""
    import os as _os
    from pathlib import Path as _P
    db = _P(history.DB_PATH)
    cfg = _P(config.CONFIG_PATH)
    hf_cache = _P(_os.path.expanduser("~/.cache/huggingface"))

    def _size_gb(p):
        try:
            if not p.exists():
                return 0.0
            total = 0
            for f in p.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            return total / (1024 ** 3)
        except Exception:
            return 0.0

    return {
        "db_path": str(db),
        "db_size_mb": (db.stat().st_size / (1024 * 1024)) if db.exists() else 0,
        "config_path": str(cfg),
        "huggingface_cache_gb": round(_size_gb(hf_cache), 2),
        "history_enabled": config.HISTORY_ENABLED,
    }


def _apply_settings_patch(patch: dict) -> dict:
    """Validate a patch against the schema and write it.

    Returns `{"ok": True, "restart_required": bool, "applied": {...}}` on
    success, or `{"error": "..."}` on validation failure.

    `patch` is a flat dict of dotted keys to new values:
      {"hotkey": "f5", "summaries.provider": "local"}
    We translate dotted keys back to nested dicts before writing.
    """
    from voiceclip.config_io import write_config_patch

    if not isinstance(patch, dict) or not patch:
        return {"error": "empty or invalid patch"}

    # Validate every key against the schema
    normalized: dict = {}
    restart_required = False
    for key, new_value in patch.items():
        schema = _SETTINGS_SCHEMA.get(key)
        if schema is None:
            return {"error": f"unknown setting '{key}'"}
        typ = schema["type"]

        # Per-type validation
        if typ == "bool":
            if not isinstance(new_value, bool):
                return {"error": f"'{key}' must be true or false"}
        elif typ == "int":
            if not isinstance(new_value, int) or isinstance(new_value, bool):
                return {"error": f"'{key}' must be an integer"}
            if "min" in schema and new_value < schema["min"]:
                return {"error": f"'{key}' must be >= {schema['min']}"}
            if "max" in schema and new_value > schema["max"]:
                return {"error": f"'{key}' must be <= {schema['max']}"}
        elif typ == "select":
            if new_value not in schema["choices"]:
                return {"error": f"'{key}' must be one of {schema['choices']}"}
        elif typ == "select_or_none":
            if new_value is not None and new_value not in schema["choices"]:
                return {"error": f"'{key}' must be null or one of {schema['choices']}"}
        elif typ == "text":
            if new_value is not None and not isinstance(new_value, str):
                return {"error": f"'{key}' must be a string"}
        else:
            return {"error": f"unhandled type for '{key}'"}

        if schema.get("restart_required"):
            restart_required = True
        normalized[key] = new_value

    # Translate dotted keys into a nested-dict patch that the merging
    # writer will combine with existing config.
    nested: dict = {}
    for key, value in normalized.items():
        parts = key.split(".")
        target = nested
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value

    if not write_config_patch(nested):
        return {"error": "could not write config file"}

    # Reload runtime config so subsequent GET /api/settings reflects the
    # write. Fields that need daemon restart (hotkey, mode) won't actually
    # apply this session — the UI shows a banner to that effect.
    config.load()

    return {
        "ok": True,
        "restart_required": restart_required,
        "applied": normalized,
    }


# ---------------------------------------------------------------------------
# HTTP request handling
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    # Quiet the default noisy access log
    def log_message(self, format, *args):  # noqa: A002
        log.debug("%s - %s", self.address_string(), format % args)

    def _json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Defense in depth — localhost only, but still
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, rel_path: str):
        """Serve a file from voiceclip/static/ with path-traversal protection."""
        # Refuse anything with '..' or absolute-looking bits. Resolve the
        # request against _STATIC_DIR and verify the resolved path is still
        # inside it — belt and suspenders against traversal.
        if not rel_path or ".." in rel_path.split("/"):
            self._not_found()
            return
        target = (_STATIC_DIR / rel_path).resolve()
        try:
            target.relative_to(_STATIC_DIR.resolve())
        except ValueError:
            self._not_found()
            return
        if not target.is_file():
            self._not_found()
            return
        ctype, _ = mimetypes.guess_type(str(target))
        ctype = ctype or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"not found")

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        q = parse_qs(url.query)

        if path == "/" or path == "/index.html":
            self._static("index.html")
            return

        if path.startswith("/static/"):
            # Strip the prefix; refuse any path traversal
            rel = path[len("/static/"):]
            self._static(rel)
            return

        if path == "/api/days":
            self._json({"days": history.list_days(limit=180)})
            return

        if path == "/api/day":
            date = (q.get("date") or [datetime.now().strftime("%Y-%m-%d")])[0]
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                self._json({"error": "bad date"}, status=400)
                return
            self._json(_day_payload(date))
            return

        if path == "/api/queue":
            # Research queue listing
            status = (q.get("status") or [None])[0]
            topics = history.list_research_topics(status_filter=status)
            # Attach latest brief to each topic that has one
            for t in topics:
                if t["brief_count"] > 0:
                    t["brief"] = history.latest_brief(t["id"])
            self._json({
                "topics": topics,
                "research_enabled": config.RESEARCH_PROVIDER != "none",
                "research_provider": config.RESEARCH_PROVIDER,
                "research_model": (
                    config.RESEARCH_OPENAI_MODEL if config.RESEARCH_PROVIDER == "openai"
                    else config.RESEARCH_ANTHROPIC_MODEL if config.RESEARCH_PROVIDER == "anthropic"
                    else None
                ),
            })
            return

        if path == "/api/patterns/config":
            # Shape that the Patterns tab needs to render the empty state or
            # the "generate" button without kicking off an LLM call.
            active_model = None
            if config.PATTERNS_PROVIDER == "local":
                active_model = config.PATTERNS_LOCAL_MODEL
            elif config.PATTERNS_PROVIDER == "openai":
                active_model = config.PATTERNS_OPENAI_MODEL
            elif config.PATTERNS_PROVIDER == "anthropic":
                active_model = config.PATTERNS_ANTHROPIC_MODEL
            self._json({
                "enabled": config.PATTERNS_PROVIDER != "none",
                "provider": config.PATTERNS_PROVIDER,
                "model": active_model,
                "window_days": config.PATTERNS_WINDOW_DAYS,
            })
            return

        if path == "/api/search":
            # Full-text search across all history. Kind filter optional.
            term = (q.get("q") or [""])[0]
            kind = (q.get("kind") or [None])[0]
            if kind not in (None, "transcription", "reflection"):
                kind = None
            if not term.strip():
                self._json({"entries": [], "query": ""})
                return
            entries = history.search_entries(term, limit=50, kind=kind)
            self._json({"entries": entries, "query": term})
            return

        if path == "/api/settings":
            # Return the current value of every user-facing setting plus
            # metadata the UI uses to render inputs and restart banners.
            self._json(_settings_payload())
            return

        self._not_found()

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path

        # Read body (small — all our POSTs carry a tiny JSON blob)
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            payload = {}

        if path == "/api/promote":
            entry_id = payload.get("id")
            if not isinstance(entry_id, int):
                self._json({"error": "bad id"}, status=400)
                return
            result = history.promote_to_reflection(entry_id=entry_id)
            if result is None:
                self._json({"error": "not found or already a reflection"}, status=404)
            else:
                self._json({"ok": True, "entry": result})
            return

        if path == "/api/delete":
            entry_id = payload.get("id")
            if not isinstance(entry_id, int):
                self._json({"error": "bad id"}, status=400)
                return
            result = history.delete_entry(entry_id)
            if result is None:
                self._json({"error": "not found"}, status=404)
            else:
                self._json({"ok": True, "entry": result})
            return

        if path == "/api/update":
            entry_id = payload.get("id")
            text = payload.get("text", "")
            if not isinstance(entry_id, int) or not isinstance(text, str):
                self._json({"error": "bad payload"}, status=400)
                return
            text = text.strip()
            if not text:
                self._json({"error": "text is empty"}, status=400)
                return
            result = history.update_text(entry_id, text)
            if result is None:
                self._json({"error": "not found"}, status=404)
            else:
                self._json({"ok": True, "entry": result})
            return

        if path == "/api/research/create":
            text = payload.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._json({"error": "text is empty"}, status=400)
                return
            new_id = history.create_research_topic(text.strip(), app_name="Viewer")
            if new_id is None:
                self._json({"error": "failed to create topic"}, status=500)
            else:
                self._json({"ok": True, "id": new_id})
            return

        if path == "/api/research/run":
            entry_id = payload.get("id")
            if not isinstance(entry_id, int):
                self._json({"error": "bad id"}, status=400)
                return
            if config.RESEARCH_PROVIDER == "none":
                self._json({
                    "error": "Research is disabled. Set research.provider in ~/.voiceclip/config.json."
                }, status=400)
                return
            try:
                from voiceclip.researcher import research_topic
                brief = research_topic(entry_id)
                if brief is None:
                    self._json({"error": "topic not found"}, status=404)
                else:
                    self._json({"ok": True, "brief": brief})
            except RuntimeError as e:
                log.warning("Research failed: %s", e)
                self._json({"error": str(e)}, status=400)
            except Exception as e:
                log.exception("Research crashed")
                self._json({"error": f"internal error: {e}"}, status=500)
            return

        if path == "/api/patterns/run":
            # Generate a fresh patterns view. Can be slow (local LLM).
            if config.PATTERNS_PROVIDER == "none":
                self._json({
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
                self._json({"ok": True, "patterns": result})
            except RuntimeError as e:
                log.warning("Patterns failed: %s", e)
                self._json({"error": str(e)}, status=400)
            except Exception as e:
                log.exception("Patterns crashed")
                self._json({"error": f"internal error: {e}"}, status=500)
            return

        if path == "/api/patterns/queue":
            # Take a suggested topic from the Patterns view and add it to
            # the research queue. Dedup: if the topic text already exists as
            # a research topic, return a reference to the existing one.
            topic_text = (payload.get("topic") or "").strip()
            reason = (payload.get("reason") or "").strip()
            quote = (payload.get("grounding_quote") or "").strip()
            if not topic_text:
                self._json({"error": "topic is empty"}, status=400)
                return

            existing = history.find_research_topic_by_text(topic_text)
            if existing:
                self._json({
                    "ok": True, "duplicate": True,
                    "id": existing["id"],
                    "text": existing["text"],
                })
                return

            # Compose the stored text: topic + a short context note describing
            # where the suggestion came from. Keeps provenance visible later.
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
                self._json({"error": "failed to create topic"}, status=500)
            else:
                self._json({"ok": True, "id": new_id, "duplicate": False})
            return

        if path == "/api/settings/update":
            # Patch ~/.voiceclip/config.json with a partial update.
            # Validates keys against a whitelist so arbitrary fields can't
            # be shoved into config via the browser.
            result = _apply_settings_patch(payload)
            if "error" in result:
                self._json(result, status=400)
            else:
                self._json(result)
            return

        if path == "/api/summarize":
            date = payload.get("date") or datetime.now().strftime("%Y-%m-%d")
            force = bool(payload.get("force", False))
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                self._json({"error": "bad date"}, status=400)
                return
            if config.SUMMARIES_PROVIDER == "none":
                self._json({"error": "summaries are disabled in config"}, status=400)
                return
            try:
                from voiceclip.summarizer import summarize_day
                result = summarize_day(date, force=force)
                if result is None:
                    self._json({"error": "no entries for that day"}, status=404)
                else:
                    self._json({"ok": True, "summary": result})
            except RuntimeError as e:
                # Expected failures (missing deps, missing API keys, bad config).
                # Log a single warning line — no stack trace — and return the
                # message so the UI can show a friendly hint.
                log.warning("Summarize failed: %s", e)
                self._json({"error": str(e)}, status=400)
            except Exception as e:
                # Actually unexpected — full traceback is fair.
                log.exception("Summarize crashed")
                self._json({"error": f"internal error: {e}"}, status=500)
            return

        self._not_found()


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve(host: str = "127.0.0.1", port: int = 8723, open_browser: bool = True):
    """Start the viewer HTTP server. Blocks until Ctrl+C.

    Always binds to localhost only — never exposes your history over the network.
    """
    history.init()
    # Surface cloud-provider changes before the user starts clicking
    from voiceclip.consent import check_and_warn as _cloud_check
    _cloud_check()
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"\n  🌐 VoiceClip viewer running at {url}")
    print("     Ctrl+C to stop\n")

    if open_browser:
        # Open in a detached thread so a slow browser launch can't delay shutdown
        threading.Thread(
            target=lambda: webbrowser.open(url), daemon=True,
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  👋 Viewer stopped.")
    finally:
        server.server_close()
        history.close()
