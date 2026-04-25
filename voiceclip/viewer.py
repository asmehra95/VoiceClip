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
            except Exception as e:
                log.exception("Summarize failed")
                self._json({"error": str(e)}, status=500)
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
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🎙️ VoiceClip journal</h1>
    <div class="daynav">
      <button id="prev">← Prev</button>
      <button id="today">Today</button>
      <button id="next">Next →</button>
      <input id="datepicker" type="date">
    </div>
  </header>

  <div class="dayline">
    <h2 id="daylabel">…</h2>
    <div class="stats" id="stats"></div>
  </div>

  <div id="summary_slot"></div>
  <div id="apps_slot"></div>
  <section class="entries" id="entries_slot"></section>
</div>

<script>
(function(){
  const state = { date: todayStr() };

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

  async function load(date) {
    state.date = date;
    document.getElementById("datepicker").value = date;
    const r = await fetch(`/api/day?date=${date}`);
    const data = await r.json();
    render(data);
  }

  function render(data) {
    // Day label + stats
    document.getElementById("daylabel").textContent = data.day_label;
    const s = data.stats;
    document.getElementById("stats").textContent =
      `${s.transcriptions} transcription${s.transcriptions===1?"":"s"} · ${s.reflections} reflection${s.reflections===1?"":"s"}`;

    // Nav buttons
    document.getElementById("prev").disabled = !data.prev_day;
    document.getElementById("prev").onclick = () => data.prev_day && load(data.prev_day);
    document.getElementById("next").disabled = !data.next_day;
    document.getElementById("next").onclick = () => data.next_day && load(data.next_day);
    document.getElementById("today").onclick = () => load(todayStr());
    document.getElementById("datepicker").onchange = (e) => {
      if (e.target.value) load(e.target.value);
    };

    // Summary
    const summarySlot = document.getElementById("summary_slot");
    summarySlot.innerHTML = "";
    if (data.summary_enabled && data.entries.length > 0) {
      if (data.summary && data.summary.summary) {
        const meta = el("div", {class:"meta"},[
          `Summary · ${data.summary.provider} · ${data.summary.model}`,
          el("button", {class:"refresh", onclick: () => regen(data.date)}, "Refresh"),
        ]);
        const body = el("div", {class:"body"}, data.summary.summary);
        summarySlot.appendChild(el("div", {class:"summary"}, [meta, body]));
      } else {
        const msg = el("span", null, "Generate a summary for this day?");
        const btn = el("button", {onclick: () => regen(data.date)}, "Generate");
        summarySlot.appendChild(el("div", {class:"generate"}, [msg, btn]));
      }
    }

    // Apps bar
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

    // Entries
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
      navigator.clipboard.writeText(e.text).then(() => flash(ev.target, "Copied"));
    }}, "Copy");
    const actions = [copyBtn];
    if (e.kind === "transcription") {
      const promoteBtn = el("button", {onclick: async ev => {
        const r = await fetch("/api/promote", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({id: e.id}),
        });
        if (r.ok) { flash(ev.target, "Promoted"); setTimeout(() => load(state.date), 400); }
        else flash(ev.target, "Failed");
      }}, "💭 Keep");
      actions.push(promoteBtn);
    }
    const text = el("div", {class:"text"}, e.text);
    const meta = el("div", {class:"meta"}, [
      `${e.kind === "reflection" ? "💭" : "📝"} ${fmtTime(e.timestamp)}`,
      e.duration ? `· ${e.duration}s` : null,
      e.app_name ? `· ${e.app_name}` : null,
      el("div", {class:"actions"}, actions),
    ]);
    return el("div", {class:`entry ${e.kind}`}, [text, meta]);
  }

  function flash(btn, label) {
    const prev = btn.textContent;
    btn.textContent = label;
    btn.classList.add("flash");
    setTimeout(() => { btn.textContent = prev; btn.classList.remove("flash"); }, 900);
  }

  async function regen(date) {
    const slot = document.getElementById("summary_slot");
    slot.innerHTML = `<div class="generate"><span>Thinking…</span></div>`;
    try {
      const r = await fetch("/api/summarize", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({date, force: true}),
      });
      const data = await r.json();
      if (!r.ok) {
        slot.innerHTML = `<div class="generate"><span style="color:#c44">${(data.error||"failed")}</span></div>`;
        return;
      }
      load(date);
    } catch(e) {
      slot.innerHTML = `<div class="generate"><span style="color:#c44">${e.message}</span></div>`;
    }
  }

  // Auto-refresh today's page every 20s while viewing today
  setInterval(() => { if (state.date === todayStr()) load(state.date); }, 20000);

  load(state.date);
})();
</script>
</body>
</html>
"""
