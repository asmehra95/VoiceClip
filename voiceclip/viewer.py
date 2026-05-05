"""Lightweight local web viewer for VoiceClip history.

Serves a single HTML page from localhost (never binds to 0.0.0.0).
Zero dependencies — uses only Python's stdlib http.server.

This module is deliberately thin. It contains:
  - The `Handler` class (HTTP plumbing, static file serving, dispatch)
  - The `serve()` entry point

Actual route logic lives in `voiceclip/routes/`, one file per feature
domain. Each route module registers its handlers into module-level
dicts via `routes.register_get()` / `routes.register_post()`. Handler's
`do_GET` and `do_POST` just look up by path and delegate.

Start with:  voiceclip view
Stop with:   Ctrl+C
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from voiceclip import history, routes

log = logging.getLogger(__name__)

# Static assets live alongside this module.
_STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# HTTP request handling
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    """Dispatches to route handlers registered in voiceclip.routes.

    Exposes a small set of response helpers (`_json`, `_html`, `_static`,
    `_not_found`) that route handlers call into. Handlers do not return
    values; they write responses via these helpers.
    """

    # Quiet the default noisy access log
    def log_message(self, format, *args):  # noqa: A002
        log.debug("%s - %s", self.address_string(), format % args)

    # ------------------------------------------------------------------
    # Response helpers — shared across route handlers
    # ------------------------------------------------------------------

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

        # For index.html, inject mtime-based cache-bust tokens onto the
        # asset URLs. Browsers love to cache /static/app.js even when we
        # send no-store (bfcache, service workers, aggressive tabs). By
        # changing the URL on every file change, any cache is forced to
        # treat the asset as a new resource. Solves the "I edited the JS
        # but the button still does nothing" class of bug.
        if rel_path == "index.html":
            try:
                js_mtime = int((_STATIC_DIR / "app.js").stat().st_mtime)
                css_mtime = int((_STATIC_DIR / "app.css").stat().st_mtime)
                text = data.decode("utf-8")
                text = text.replace(
                    "/static/app.js", f"/static/app.js?v={js_mtime}"
                ).replace(
                    "/static/app.css", f"/static/app.css?v={css_mtime}"
                )
                data = text.encode("utf-8")
            except Exception as e:
                log.debug("Could not version static assets: %s", e)

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

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        # Static content handled inline — not a "route" in the API sense.
        if path == "/" or path == "/index.html":
            self._static("index.html")
            return

        if path.startswith("/static/"):
            self._static(path[len("/static/"):])
            return

        handler = routes.get_handler(path)
        if handler is None:
            self._not_found()
            return
        query = parse_qs(url.query)
        handler(self, query)

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path

        # Read body (all our POSTs carry a small JSON blob)
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            payload = {}

        handler = routes.post_handler(path)
        if handler is None:
            self._not_found()
            return
        handler(self, payload)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve(host: str = "127.0.0.1", port: int = 8723, open_browser: bool = True):
    """Start the viewer HTTP server. Blocks until Ctrl+C.

    Always binds to localhost only — never exposes your history over the network.

    Cloud-provider consent is NOT auto-acked on viewer startup. Instead,
    /api/consent surfaces any unacked cloud features, the frontend renders
    a visible banner above the tabs, and the user dismisses it through
    POST /api/consent/ack. This closes the "user only uses the viewer,
    never sees the CLI banner" gap.
    """
    history.init()

    # Populate the route registry by importing every submodule in
    # voiceclip.routes. Doing this at server start (not at module import)
    # keeps the viewer importable from tests that don't need the full
    # routing surface.
    routes.load_all()

    # Log pending-consent state at startup so logs capture it, but do NOT
    # auto-ack — that's now a user action in the UI.
    from voiceclip.consent import pending_acks
    pending = pending_acks()
    if pending:
        log.info(
            "Cloud provider acknowledgment pending for: %s. "
            "Banner will appear in the viewer UI.",
            ", ".join(sorted(pending.keys())),
        )

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
# Test compatibility shim
# ---------------------------------------------------------------------------
# tests/conftest.py's `live_viewer` fixture starts ThreadingHTTPServer
# directly with `Handler` — no call to serve() — which means it skips
# routes.load_all(). Load routes at module import instead so the test
# harness has a working dispatch table without duplicating setup logic.
routes.load_all()
