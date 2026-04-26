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
