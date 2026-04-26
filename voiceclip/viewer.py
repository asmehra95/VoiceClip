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
import sys
import threading
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from voiceclip import config, history

log = logging.getLogger(__name__)


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
            self._html(_PAGE_HTML)
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
                result = generate_patterns(window_days=window)
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


# ---------------------------------------------------------------------------
# Single-page HTML (embedded — no external deps, no CDNs)
# ---------------------------------------------------------------------------

_PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoiceClip journal</title>
<style>
  :root {
    --bg: #fafaf8;
    --surface: #ffffff;
    --text: #1c1c1c;
    --muted: #6b6b6b;
    --border: #e6e4df;
    --accent: #2563eb;
    --reflection-bg: #fff7ed;
    --reflection-border: #f5d9b3;
    --bar: #d6d3cc;
    --bar-fill: #a2aab6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #141416;
      --surface: #1c1c1f;
      --text: #e9e7e3;
      --muted: #8a8a8a;
      --border: #2a2a2e;
      --accent: #7aa2ff;
      --reflection-bg: #26201a;
      --reflection-border: #4a3b28;
      --bar: #2d2d30;
      --bar-fill: #5b6370;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
  }
  .wrap {
    max-width: 720px;
    margin: 0 auto;
    padding: 32px 24px 96px;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }
  h1 {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: .2px;
    margin: 0;
    color: var(--muted);
  }
  .dayline {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 24px 0 8px;
  }
  h2 {
    font-size: 28px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .daynav { display: flex; gap: 8px; align-items: center; }
  .daynav button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 8px;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
  }
  .daynav button:hover { border-color: var(--muted); }
  .daynav button:disabled { opacity: 0.35; cursor: not-allowed; }
  .daynav input[type=date] {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 5px 8px;
    border-radius: 8px;
    font: inherit; font-size: 13px;
    color-scheme: light dark;
  }

  .stats {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 20px;
  }
  .apps {
    margin: 18px 0 28px;
  }
  .apps h3 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin: 0 0 10px;
    font-weight: 600;
  }
  .appbar {
    display: grid;
    grid-template-columns: 1fr 140px 28px;
    gap: 10px;
    align-items: center;
    font-size: 13px;
    margin: 4px 0;
  }
  .appbar .name { color: var(--text); }
  .appbar .bar {
    background: var(--bar);
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
  }
  .appbar .bar > span {
    display: block;
    height: 100%;
    background: var(--bar-fill);
    border-radius: 3px;
  }
  .appbar .count { color: var(--muted); text-align: right; }

  .summary {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    margin: 20px 0 28px;
    line-height: 1.65;
    font-size: 15.5px;
  }
  .summary .meta {
    color: var(--muted);
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .summary .body { white-space: pre-wrap; }
  .summary button.refresh {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    font-size: 11px; padding: 3px 8px;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: .5px;
  }
  .summary button.refresh:hover { color: var(--text); border-color: var(--muted); }

  .generate {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    margin: 20px 0 28px;
    color: var(--muted);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    font-size: 14px;
  }
  .generate button {
    background: var(--accent);
    color: white;
    border: 0;
    border-radius: 8px;
    padding: 8px 14px;
    font: inherit;
    font-size: 13px;
    cursor: pointer;
  }
  code.inline {
    background: var(--bar);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
  }
  .generate .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    margin-right: 8px;
    border: 2px solid var(--bar);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: -2px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  section.entries h3 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin: 32px 0 12px;
    font-weight: 600;
  }
  .entry {
    margin: 14px 0 22px;
    transition: opacity 0.35s ease, transform 0.35s ease, max-height 0.35s ease, margin 0.35s ease, padding 0.35s ease;
    overflow: hidden;
  }
  .entry.removing {
    opacity: 0;
    transform: translateX(-12px);
    max-height: 0;
    margin: 0;
    padding-top: 0;
    padding-bottom: 0;
  }
  .entry.reflection {
    background: var(--reflection-bg);
    border: 1px solid var(--reflection-border);
    border-radius: 12px;
    padding: 16px 20px;
  }
  .entry .text {
    font-size: 16px;
    line-height: 1.65;
    max-width: 62ch;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .entry.transcription .text { color: var(--muted); font-size: 14px; }
  .entry .meta {
    margin-top: 6px;
    font-size: 11.5px;
    color: var(--muted);
    display: flex;
    gap: 10px;
    align-items: center;
  }
  .entry .actions { display: flex; gap: 6px; margin-left: auto; }
  .entry .actions button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 2px 8px;
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .5px;
  }
  .entry .actions button:hover { color: var(--text); border-color: var(--muted); }
  .entry .actions .flash { color: var(--accent); border-color: var(--accent); }
  .entry .actions .danger:hover { color: #c44; border-color: #c44; }
  .entry .actions .armed {
    color: white !important;
    background: #c44;
    border-color: #c44;
  }
  .entry .text[contenteditable="true"]:focus {
    outline: 2px solid var(--accent);
    outline-offset: 4px;
    border-radius: 4px;
    background: var(--surface);
    padding: 6px 10px;
    margin: -6px -10px;
  }
  .entry .text .saved-pill {
    display: inline-block;
    font-size: 10px;
    background: var(--accent);
    color: white;
    padding: 1px 6px;
    border-radius: 3px;
    margin-left: 8px;
    vertical-align: 2px;
  }
  .entry .meta .edited {
    font-style: italic;
    opacity: 0.7;
  }

  details.transcriptions summary {
    cursor: pointer;
    color: var(--muted);
    font-size: 13px;
    margin: 16px 0 8px;
    user-select: none;
  }
  details.transcriptions summary::-webkit-details-marker { display: none; }
  details.transcriptions[open] summary { margin-bottom: 18px; }

  .empty {
    color: var(--muted);
    text-align: center;
    padding: 80px 0;
    font-size: 15px;
  }

  .warn {
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
  }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin: 24px 0 0;
  }
  .tab {
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    color: var(--muted);
    padding: 10px 14px 11px;
    font: inherit;
    font-size: 14px;
    cursor: pointer;
    margin-bottom: -1px;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--text); border-bottom-color: var(--accent); }

  /* Queue view */
  .queue-input {
    display: flex;
    gap: 8px;
    margin: 20px 0 8px;
  }
  .queue-input input {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 10px 14px;
    border-radius: 10px;
    font: inherit;
    font-size: 14px;
  }
  .queue-input input:focus {
    outline: 2px solid var(--accent);
    outline-offset: -1px;
    border-color: var(--accent);
  }
  .queue-input button {
    background: var(--accent);
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 0 16px;
    font: inherit;
    font-size: 13px;
    cursor: pointer;
  }
  .queue-hint {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 20px;
  }

  .topic {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    transition: opacity 0.35s ease, transform 0.35s ease;
  }
  .topic.pending { border-left: 3px solid var(--bar-fill); }
  .topic.ready { border-left: 3px solid var(--accent); }
  .topic.running { border-left: 3px solid #e1b345; }
  .topic.failed { border-left: 3px solid #c44; }
  .topic .topic-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
  }
  .topic .title {
    font-size: 15px;
    font-weight: 500;
    color: var(--text);
    flex: 1;
  }
  .topic .topic-meta {
    margin-top: 4px;
    color: var(--muted);
    font-size: 11.5px;
  }
  .topic .brief {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px dashed var(--border);
    font-size: 14.5px;
    line-height: 1.65;
    white-space: pre-wrap;
  }
  .topic .brief strong { color: var(--text); font-weight: 600; }
  .topic .sources {
    margin-top: 12px;
    font-size: 12px;
    color: var(--muted);
  }
  .topic .sources a {
    color: var(--accent);
    text-decoration: none;
    display: block;
    padding: 2px 0;
  }
  .topic .sources a:hover { text-decoration: underline; }
  .topic .actions {
    display: flex;
    gap: 6px;
  }
  .topic .actions button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 4px 10px;
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .5px;
  }
  .topic .actions button:hover { color: var(--text); border-color: var(--muted); }
  .topic .actions .primary {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .topic .actions .primary:hover { opacity: 0.9; }
  .topic .actions .danger:hover { color: #c44; border-color: #c44; }
  .topic.removing { opacity: 0; transform: translateX(-12px); }
  .topic .web-badge {
    font-size: 10px;
    background: var(--accent);
    color: white;
    padding: 1px 6px;
    border-radius: 3px;
    margin-left: 6px;
    vertical-align: 2px;
    text-transform: uppercase;
    letter-spacing: .5px;
  }

  /* Patterns view */
  .pattern-section {
    margin: 28px 0;
  }
  .pattern-section h3 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin: 0 0 12px;
    font-weight: 600;
  }
  .pattern-body {
    font-size: 15.5px;
    line-height: 1.65;
    color: var(--text);
    max-width: 62ch;
  }
  .theme {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--bar-fill);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
  }
  .theme .title {
    font-weight: 600;
    font-size: 14px;
  }
  .theme .count {
    color: var(--muted);
    font-size: 12px;
    margin-left: 8px;
    font-weight: 400;
  }
  .theme .quote {
    color: var(--muted);
    font-size: 14px;
    font-style: italic;
    margin-top: 6px;
    padding-left: 10px;
    border-left: 2px solid var(--border);
  }
  .suggestion {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    display: flex;
    gap: 14px;
    align-items: flex-start;
  }
  .suggestion .content { flex: 1; }
  .suggestion .topic-line {
    font-weight: 600;
    font-size: 14.5px;
    margin-bottom: 4px;
  }
  .suggestion .reason { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
  .suggestion .quote {
    color: var(--muted);
    font-size: 12.5px;
    font-style: italic;
    padding-left: 10px;
    border-left: 2px solid var(--border);
  }
  .suggestion .queue-btn {
    background: var(--accent);
    color: white;
    border: 0;
    border-radius: 6px;
    padding: 6px 12px;
    font: inherit;
    font-size: 12px;
    cursor: pointer;
    white-space: nowrap;
  }
  .suggestion .queue-btn:hover { opacity: 0.9; }
  .suggestion .queue-btn:disabled {
    background: var(--bar-fill);
    cursor: default;
  }
  .patterns-generate {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    text-align: center;
    color: var(--muted);
  }
  .patterns-generate button {
    background: var(--accent);
    color: white;
    border: 0;
    border-radius: 8px;
    padding: 10px 20px;
    font: inherit;
    font-size: 14px;
    cursor: pointer;
    margin-top: 14px;
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🎙️ VoiceClip journal</h1>
    <div class="daynav" id="journal_nav">
      <button id="prev">← Prev</button>
      <button id="today">Today</button>
      <button id="next">Next →</button>
      <input id="datepicker" type="date">
    </div>
  </header>

  <div class="tabs">
    <button class="tab active" data-view="journal">📓 Journal</button>
    <button class="tab" data-view="queue">📚 Queue</button>
    <button class="tab" data-view="patterns">📊 Patterns</button>
  </div>

  <div id="journal_view">
    <div class="dayline">
      <h2 id="daylabel">…</h2>
      <div class="stats" id="stats"></div>
    </div>

    <div id="summary_slot"></div>
    <div id="apps_slot"></div>
    <section class="entries" id="entries_slot"></section>
  </div>

  <div id="queue_view" style="display:none">
    <div class="dayline">
      <h2>Research queue</h2>
      <div class="stats" id="queue_stats"></div>
    </div>
    <div class="queue-input">
      <input id="topic_input" type="text"
             placeholder="What do you want researched? (e.g. 'CRDTs vs. operational transforms')">
      <button id="topic_add">Add</button>
    </div>
    <div class="queue-hint" id="queue_hint"></div>
    <section id="topics_slot"></section>
  </div>

  <div id="patterns_view" style="display:none">
    <div class="dayline">
      <h2>Patterns</h2>
      <div class="stats" id="patterns_stats"></div>
    </div>
    <div class="queue-hint" id="patterns_hint"></div>
    <section id="patterns_slot"></section>
  </div>
</div>

<script>
(function(){
  const state = { date: todayStr(), view: "journal" };

  function todayStr() {
    const d = new Date();
    const pad = n => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`;
  }

  function fmtTime(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    } catch(e) { return iso || ""; }
  }

  function fmtRelDate(iso) {
    try {
      const d = new Date(iso);
      const today = new Date();
      today.setHours(0,0,0,0);
      const dd = new Date(d); dd.setHours(0,0,0,0);
      const diffDays = Math.round((today - dd) / 86400000);
      if (diffDays === 0) return "today";
      if (diffDays === 1) return "yesterday";
      if (diffDays < 7) return d.toLocaleDateString([], { weekday: "long" }).toLowerCase();
      return d.toLocaleDateString([], { month: "short", day: "numeric" });
    } catch(e) { return iso || ""; }
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") node.className = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), attrs[k]);
      else node.setAttribute(k, attrs[k]);
    }
    if (children) for (const c of [].concat(children)) {
      if (c == null) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  // Very small markdown renderer — enough for **bold**, bullets, blank lines.
  // We use it only for research briefs (LLM-produced), not arbitrary input,
  // so the risk surface is limited.
  function renderMarkdown(src) {
    const escaped = String(src || "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    // Bold
    let s = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Bullets: lines starting with "- " become <li>; group consecutive into <ul>
    const lines = s.split("\n");
    const out = [];
    let inList = false;
    for (const ln of lines) {
      if (/^\s*-\s+/.test(ln)) {
        if (!inList) { out.push("<ul>"); inList = true; }
        out.push("<li>" + ln.replace(/^\s*-\s+/, "") + "</li>");
      } else {
        if (inList) { out.push("</ul>"); inList = false; }
        out.push(ln);
      }
    }
    if (inList) out.push("</ul>");
    return out.join("\n");
  }

  // ---------- Tab switching ----------
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => switchTab(tab.dataset.view));
  });

  function switchTab(view) {
    state.view = view;
    document.querySelectorAll(".tab").forEach(t => {
      t.classList.toggle("active", t.dataset.view === view);
    });
    document.getElementById("journal_view").style.display = view === "journal" ? "" : "none";
    document.getElementById("queue_view").style.display = view === "queue" ? "" : "none";
    document.getElementById("patterns_view").style.display = view === "patterns" ? "" : "none";
    document.getElementById("journal_nav").style.visibility = view === "journal" ? "" : "hidden";
    if (view === "queue") loadQueue();
    else if (view === "patterns") loadPatterns();
    else load(state.date);
  }

  // ---------- Journal (existing) ----------
  async function load(date) {
    state.date = date;
    document.getElementById("datepicker").value = date;
    const r = await fetch(`/api/day?date=${date}`);
    const data = await r.json();
    render(data);
  }

  function render(data) {
    document.getElementById("daylabel").textContent = data.day_label;
    const s = data.stats;
    document.getElementById("stats").textContent =
      `${s.transcriptions} transcription${s.transcriptions===1?"":"s"} · ${s.reflections} reflection${s.reflections===1?"":"s"}`;

    document.getElementById("prev").disabled = !data.prev_day;
    document.getElementById("prev").onclick = () => data.prev_day && load(data.prev_day);
    document.getElementById("next").disabled = !data.next_day;
    document.getElementById("next").onclick = () => data.next_day && load(data.next_day);
    document.getElementById("today").onclick = () => load(todayStr());
    document.getElementById("datepicker").onchange = (e) => {
      if (e.target.value) load(e.target.value);
    };

    const summarySlot = document.getElementById("summary_slot");
    summarySlot.innerHTML = "";
    if (data.entries.length > 0) {
      if (data.summary_enabled) {
        if (data.summary && data.summary.summary) {
          const meta = el("div", {class:"meta"},[
            `Summary · ${data.summary.provider} · ${shortModel(data.summary.model)}`,
            el("button", {class:"refresh", onclick: () => regen(data.date)}, "Refresh"),
          ]);
          const body = el("div", {class:"body"}, data.summary.summary);
          summarySlot.appendChild(el("div", {class:"summary"}, [meta, body]));
        } else {
          const label = data.summary_model
            ? `Generate a summary? · ${data.summary_provider} · ${shortModel(data.summary_model)}`
            : "Generate a summary for this day?";
          const msg = el("span", null, label);
          const btn = el("button", {onclick: () => regen(data.date)}, "Generate");
          summarySlot.appendChild(el("div", {class:"generate"}, [msg, btn]));
        }
      } else {
        const hint = el("div", {class:"generate"}, [
          el("span", null, [
            "💡 Summaries are off. Enable them by adding ",
            el("code", {class:"inline"}, '"summaries": {"provider": "local"}'),
            " to ~/.voiceclip/config.json — then restart ",
            el("code", {class:"inline"}, "voiceclip view"),
            ".",
          ]),
        ]);
        summarySlot.appendChild(hint);
      }
    }

    const appsSlot = document.getElementById("apps_slot");
    appsSlot.innerHTML = "";
    if (s.apps && s.apps.length) {
      const max = Math.max(...s.apps.map(a => a.count));
      const list = el("div", {class:"apps"}, [
        el("h3", null, "Where your voice went"),
        ...s.apps.map(a => {
          const pct = Math.round((a.count / max) * 100);
          return el("div", {class:"appbar"}, [
            el("div", {class:"name"}, a.name),
            el("div", {class:"bar"}, el("span", {style:`width:${pct}%`})),
            el("div", {class:"count"}, String(a.count)),
          ]);
        }),
      ]);
      appsSlot.appendChild(list);
    }

    const entriesSlot = document.getElementById("entries_slot");
    entriesSlot.innerHTML = "";
    if (!data.entries.length) {
      entriesSlot.appendChild(el("div", {class:"empty"}, "Nothing captured on this day."));
      return;
    }
    const reflections = data.entries.filter(e => e.kind === "reflection");
    const transcriptions = data.entries.filter(e => e.kind === "transcription");

    if (reflections.length) {
      entriesSlot.appendChild(el("h3", null, "Reflections"));
      reflections.forEach(e => entriesSlot.appendChild(renderEntry(e)));
    }
    if (transcriptions.length) {
      const details = el("details", {class:"transcriptions"});
      details.appendChild(el("summary", null,
        `Show ${transcriptions.length} transcription${transcriptions.length===1?"":"s"}`));
      transcriptions.forEach(e => details.appendChild(renderEntry(e)));
      entriesSlot.appendChild(details);
    }
  }

  function renderEntry(e) {
    const copyBtn = el("button", {onclick: ev => {
      const node = ev.target.closest(".entry").querySelector(".text");
      navigator.clipboard.writeText(node.textContent).then(() => flash(ev.target, "Copied"));
    }}, "Copy");
    const actions = [copyBtn];
    const wrap = el("div", {class:`entry ${e.kind}`});
    wrap.dataset.entryId = String(e.id);
    if (e.kind === "transcription") {
      const promoteBtn = el("button", {onclick: async ev => {
        const r = await fetch("/api/promote", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({id: e.id}),
        });
        if (r.ok) {
          flash(ev.target, "Promoted");
          setTimeout(() => fadeOutAndReconcile(wrap), 500);
        } else flash(ev.target, "Failed");
      }}, "💭 Keep");
      actions.push(promoteBtn);
    }
    const deleteBtn = el("button", {class: "danger", onclick: ev => handleDelete(ev.target, wrap, e)}, "Delete");
    actions.push(deleteBtn);

    const text = el("div", {class:"text", contenteditable: "true", spellcheck: "true"}, e.text);
    wireInlineEdit(text, e);

    const metaParts = [
      `${e.kind === "reflection" ? "💭" : "📝"} ${fmtTime(e.timestamp)}`,
      e.duration ? `· ${e.duration}s` : null,
      e.app_name ? `· ${e.app_name}` : null,
    ];
    const meta = el("div", {class:"meta"}, [
      ...metaParts,
      e.edited_at ? el("span", {class:"edited"}, "· edited") : null,
      el("div", {class:"actions"}, actions),
    ]);
    wrap.appendChild(text);
    wrap.appendChild(meta);
    return wrap;
  }

  // Inline editing — auto-save on blur, Esc to cancel, Cmd+Enter to save.
  function wireInlineEdit(node, entry) {
    node.dataset.original = entry.text;
    // Prevent pasted HTML — always insert as plain text.
    node.addEventListener("paste", (ev) => {
      ev.preventDefault();
      const text = (ev.clipboardData || window.clipboardData).getData("text/plain");
      document.execCommand("insertText", false, text);
    });
    node.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        node.textContent = node.dataset.original;
        node.blur();
        ev.preventDefault();
      } else if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        node.blur();
        ev.preventDefault();
      }
    });
    node.addEventListener("blur", async () => {
      const newText = node.textContent.trim();
      const oldText = node.dataset.original;
      if (!newText || newText === oldText) {
        if (!newText) node.textContent = oldText;  // don't allow empty
        return;
      }
      try {
        const r = await fetch("/api/update", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({id: entry.id, text: newText}),
        });
        if (!r.ok) {
          node.textContent = oldText;
          return;
        }
        node.dataset.original = newText;
        // Show the "edited" marker if the meta line doesn't have it yet.
        const metaLine = node.parentElement.querySelector(".meta");
        if (metaLine && !metaLine.querySelector(".edited")) {
          const actions = metaLine.querySelector(".actions");
          metaLine.insertBefore(
            el("span", {class:"edited"}, "· edited"),
            actions,
          );
        }
        // Little saved pill flash inside the text node.
        const pill = el("span", {class:"saved-pill"}, "saved");
        node.appendChild(pill);
        setTimeout(() => pill.remove(), 1200);
      } catch(e) {
        node.textContent = oldText;
      }
    });
  }

  function fadeOutAndReconcile(node) {
    node.classList.add("removing");
    setTimeout(() => {
      const parent = node.parentNode;
      node.remove();
      if (parent && parent.children && parent.children.length === 0 &&
          (parent.tagName === "SECTION" || parent.tagName === "DETAILS")) {
        parent.remove();
      }
      refreshCounts();
    }, 360);
  }

  async function refreshCounts() {
    try {
      const r = await fetch(`/api/day?date=${state.date}`);
      const data = await r.json();
      const s = data.stats;
      document.getElementById("stats").textContent =
        `${s.transcriptions} transcription${s.transcriptions===1?"":"s"} · ${s.reflections} reflection${s.reflections===1?"":"s"}`;
      const details = document.querySelector("details.transcriptions summary");
      if (details) {
        const remaining = document.querySelectorAll(".entry.transcription").length;
        details.textContent = `Show ${remaining} transcription${remaining===1?"":"s"}`;
      }
      if (data.entries.length === 0) load(state.date);
    } catch(e) { /* best effort */ }
  }

  function handleDelete(btn, node, entry) {
    if (btn.dataset.armed === "1") {
      btn.dataset.armed = "";
      fetch("/api/delete", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id: entry.id}),
      }).then(r => {
        if (r.ok) fadeOutAndReconcile(node);
        else flash(btn, "Failed");
      });
      return;
    }
    btn.dataset.armed = "1";
    const prev = btn.textContent;
    btn.textContent = "Really delete?";
    btn.classList.add("armed");
    setTimeout(() => {
      if (btn.dataset.armed === "1") {
        btn.dataset.armed = "";
        btn.textContent = prev;
        btn.classList.remove("armed");
      }
    }, 3000);
  }

  function flash(btn, label) {
    const prev = btn.textContent;
    btn.textContent = label;
    btn.classList.add("flash");
    setTimeout(() => { btn.textContent = prev; btn.classList.remove("flash"); }, 900);
  }

  async function regen(date) {
    const slot = document.getElementById("summary_slot");
    slot.innerHTML = `<div class="generate"><span><span class="spinner"></span>Thinking — local models take 15-60 seconds on first run…</span></div>`;
    try {
      const r = await fetch("/api/summarize", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({date, force: true}),
      });
      const data = await r.json();
      if (!r.ok) {
        slot.innerHTML = `<div class="generate"><span style="color:#c44; white-space:pre-line">${escapeHtml(data.error || "failed")}</span></div>`;
        return;
      }
      load(date);
    } catch(e) {
      slot.innerHTML = `<div class="generate"><span style="color:#c44">${escapeHtml(e.message)}</span></div>`;
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function shortModel(id) {
    if (!id) return "";
    const slash = id.lastIndexOf("/");
    return slash >= 0 ? id.slice(slash + 1) : id;
  }

  // ---------- Queue (new) ----------

  async function loadQueue() {
    const r = await fetch("/api/queue");
    const data = await r.json();
    renderQueue(data);
  }

  function renderQueue(data) {
    const topics = data.topics || [];
    const hint = document.getElementById("queue_hint");
    if (!data.research_enabled) {
      hint.innerHTML = `💡 Research is off. Enable it by adding <code class="inline">"research": {"provider": "openai"}</code> to ~/.voiceclip/config.json and exporting <code class="inline">OPENAI_API_KEY</code>.`;
    } else {
      hint.innerHTML = `Using ${data.research_provider} · ${shortModel(data.research_model)}. The model decides whether to search the web per topic.`;
    }
    document.getElementById("queue_stats").textContent =
      `${topics.length} topic${topics.length===1?"":"s"}`;

    const slot = document.getElementById("topics_slot");
    slot.innerHTML = "";
    if (!topics.length) {
      slot.appendChild(el("div", {class:"empty"}, "No research topics yet. Add one above."));
      return;
    }
    topics.forEach(t => slot.appendChild(renderTopic(t, data.research_enabled)));
  }

  function renderTopic(t, researchEnabled) {
    const wrap = el("div", {class:`topic ${t.status}`});
    wrap.dataset.topicId = String(t.id);

    const actions = [];
    if (t.status === "pending" || t.status === "failed") {
      const btn = el("button", {class:"primary",
        disabled: researchEnabled ? null : "disabled",
        onclick: ev => runResearch(wrap, t.id, ev.target)}, "Research");
      actions.push(btn);
    } else if (t.status === "running") {
      actions.push(el("button", {disabled: "disabled"}, "Running…"));
    } else if (t.status === "ready") {
      const btn = el("button", {
        onclick: ev => runResearch(wrap, t.id, ev.target, true)
      }, "Re-research");
      actions.push(btn);
    }
    const deleteBtn = el("button", {class:"danger", onclick: ev => handleTopicDelete(ev.target, wrap, t)}, "Delete");
    actions.push(deleteBtn);

    const head = el("div", {class:"topic-head"}, [
      el("div", {class:"title"}, t.text),
      el("div", {class:"actions"}, actions),
    ]);
    wrap.appendChild(head);

    const metaBits = [
      fmtRelDate(t.timestamp),
      `· ${t.status}`,
    ];
    wrap.appendChild(el("div", {class:"topic-meta"}, metaBits.join(" ")));

    if (t.brief) {
      wrap.appendChild(renderBrief(t.brief));
    }
    return wrap;
  }

  function renderBrief(brief) {
    const body = el("div", {class:"brief"});
    body.innerHTML = renderMarkdown(brief.text);
    if (brief.used_web_search) {
      const badge = el("span", {class:"web-badge"}, "web");
      // Append badge inline with the first <strong> heading, if any
      body.insertBefore(badge, body.firstChild);
    }
    const wrap = el("div", null, [body]);
    if (brief.sources && brief.sources.length) {
      const src = el("div", {class:"sources"});
      src.appendChild(el("div", null, `Sources (${brief.sources.length}):`));
      brief.sources.forEach(s => {
        src.appendChild(el("a", {href: s.url, target: "_blank", rel: "noopener"}, s.title));
      });
      wrap.appendChild(src);
    }
    return wrap;
  }

  async function runResearch(wrap, id, btn, isRerun) {
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Researching…";
    // Optimistically mark the topic running
    wrap.classList.remove("pending", "failed", "ready");
    wrap.classList.add("running");
    try {
      const r = await fetch("/api/research/run", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id}),
      });
      const data = await r.json();
      if (!r.ok) {
        wrap.classList.remove("running");
        wrap.classList.add("failed");
        btn.disabled = false;
        btn.textContent = prev;
        const err = el("div", {class:"brief",
          style:"color:#c44; white-space:pre-line"}, data.error || "failed");
        // Replace any existing brief with the error
        const existing = wrap.querySelector(".brief");
        if (existing) existing.parentNode.replaceChild(err, existing);
        else wrap.appendChild(err);
        return;
      }
      // Reload the whole queue so counts + status line refresh consistently
      loadQueue();
    } catch(e) {
      wrap.classList.remove("running");
      wrap.classList.add("failed");
      btn.disabled = false;
      btn.textContent = prev;
    }
  }

  function handleTopicDelete(btn, node, topic) {
    if (btn.dataset.armed === "1") {
      btn.dataset.armed = "";
      fetch("/api/delete", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id: topic.id}),
      }).then(r => {
        if (r.ok) {
          node.classList.add("removing");
          setTimeout(() => { node.remove(); loadQueue(); }, 360);
        } else flash(btn, "Failed");
      });
      return;
    }
    btn.dataset.armed = "1";
    const prev = btn.textContent;
    btn.textContent = "Really delete?";
    btn.classList.add("armed");
    setTimeout(() => {
      if (btn.dataset.armed === "1") {
        btn.dataset.armed = "";
        btn.textContent = prev;
        btn.classList.remove("armed");
      }
    }, 3000);
  }

  // Topic input
  document.getElementById("topic_add").addEventListener("click", addTopic);
  document.getElementById("topic_input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); addTopic(); }
  });

  async function addTopic() {
    const input = document.getElementById("topic_input");
    const text = input.value.trim();
    if (!text) return;
    try {
      const r = await fetch("/api/research/create", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({text}),
      });
      if (r.ok) {
        input.value = "";
        loadQueue();
      }
    } catch(e) {}
  }

  // Auto-refresh today's page every 20s while viewing today's journal
  setInterval(() => {
    if (state.view === "journal" && state.date === todayStr()) load(state.date);
  }, 20000);

  // ---------- Patterns (new) ----------

  async function loadPatterns() {
    const hint = document.getElementById("patterns_hint");
    const slot = document.getElementById("patterns_slot");
    const stats = document.getElementById("patterns_stats");
    const r = await fetch("/api/patterns/config");
    const cfg = await r.json();
    stats.textContent = `past ${cfg.window_days} days`;
    if (!cfg.enabled) {
      hint.innerHTML = `💡 Patterns are off. Enable them by adding <code class="inline">"patterns": {"provider": "local"}</code> to ~/.voiceclip/config.json — then restart <code class="inline">voiceclip view</code>.`;
      slot.innerHTML = "";
      return;
    }
    hint.textContent = `Using ${cfg.provider} · ${shortModel(cfg.model)}. This reads your last ${cfg.window_days} days — local keeps it private, cloud sends it to the provider.`;
    slot.innerHTML = "";
    const card = el("div", {class: "patterns-generate"}, [
      el("div", null, "Generate a patterns view to see what you've been occupied with, recurring themes, and suggested things to learn."),
      el("button", {onclick: () => runPatterns()}, "Generate"),
    ]);
    slot.appendChild(card);
  }

  async function runPatterns() {
    const slot = document.getElementById("patterns_slot");
    slot.innerHTML = `<div class="patterns-generate"><span class="spinner"></span> Reading your recent entries — this can take 30-60 seconds locally…</div>`;
    try {
      const r = await fetch("/api/patterns/run", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({}),
      });
      const data = await r.json();
      if (!r.ok) {
        slot.innerHTML = `<div class="patterns-generate" style="color:#c44; white-space:pre-line">${escapeHtml(data.error || "failed")}</div>`;
        return;
      }
      renderPatterns(data.patterns);
    } catch(e) {
      slot.innerHTML = `<div class="patterns-generate" style="color:#c44">${escapeHtml(e.message)}</div>`;
    }
  }

  function renderPatterns(p) {
    const slot = document.getElementById("patterns_slot");
    slot.innerHTML = "";

    if (p.empty) {
      slot.appendChild(el("div", {class:"empty"},
        "Nothing captured in this window. Dictate a few things and come back."));
      return;
    }

    const stats = p.stats || {};
    // Occupied with
    if (p.occupied_with) {
      const sec = el("div", {class:"pattern-section"}, [
        el("h3", null, "Occupied with"),
        el("div", {class:"pattern-body"}, p.occupied_with),
      ]);
      slot.appendChild(sec);
    }

    // Recurring themes
    if (p.themes && p.themes.length) {
      const themes = el("div", {class:"pattern-section"});
      themes.appendChild(el("h3", null, "Recurring themes"));
      p.themes.forEach(t => {
        const card = el("div", {class:"theme"}, [
          el("div", null, [
            el("span", {class:"title"}, t.title || ""),
            t.reflection_count
              ? el("span", {class:"count"}, `· ${t.reflection_count} reflection${t.reflection_count === 1 ? "" : "s"}`)
              : null,
          ]),
          t.quote ? el("div", {class:"quote"}, `"${t.quote}"`) : null,
        ]);
        themes.appendChild(card);
      });
      slot.appendChild(themes);
    }

    // Suggested to learn
    if (p.suggestions && p.suggestions.length) {
      const sug = el("div", {class:"pattern-section"});
      sug.appendChild(el("h3", null, "Suggested to learn"));
      p.suggestions.forEach(s => {
        const btn = el("button", {class:"queue-btn",
          onclick: (ev) => queueSuggestion(ev.target, s)}, "Queue it");
        const card = el("div", {class:"suggestion"}, [
          el("div", {class:"content"}, [
            el("div", {class:"topic-line"}, s.topic || ""),
            s.reason ? el("div", {class:"reason"}, s.reason) : null,
            s.grounding_quote
              ? el("div", {class:"quote"}, `"${s.grounding_quote}"`)
              : null,
          ]),
          btn,
        ]);
        sug.appendChild(card);
      });
      slot.appendChild(sug);
    }

    // Footer with window stats
    if (stats.transcription_count != null || stats.reflection_count != null) {
      slot.appendChild(el("div", {class:"queue-hint"},
        `Window: ${stats.start_date} to ${stats.end_date} · ` +
        `${stats.transcription_count || 0} transcriptions · ` +
        `${stats.reflection_count || 0} reflections · ` +
        `${p.provider || ""} ${shortModel(p.model || "")}`));
    }
    // Regen button
    slot.appendChild(el("button", {class:"queue-btn",
      onclick: () => runPatterns(),
      style: "margin-top:16px"}, "↻ Regenerate"));
  }

  async function queueSuggestion(btn, suggestion) {
    btn.disabled = true;
    btn.textContent = "Queuing…";
    try {
      const r = await fetch("/api/patterns/queue", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
          topic: suggestion.topic,
          reason: suggestion.reason,
          grounding_quote: suggestion.grounding_quote,
        }),
      });
      const data = await r.json();
      if (!r.ok) {
        btn.textContent = "Failed";
        setTimeout(() => { btn.disabled = false; btn.textContent = "Queue it"; }, 1500);
        return;
      }
      btn.textContent = data.duplicate ? "Already queued ✓" : "Queued ✓";
    } catch(e) {
      btn.textContent = "Failed";
      setTimeout(() => { btn.disabled = false; btn.textContent = "Queue it"; }, 1500);
    }
  }

  load(state.date);
})();
</script>
</body>
</html>
"""
