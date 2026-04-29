"""Shared test fixtures and helpers for VoiceClip.

Three distinct isolation shapes emerged across the test suite:

  1. `isolated_config`   — point CONFIG_DIR/CONFIG_PATH at tmp_path, clear
                           every VOICECLIP_* env var. Use for tests that
                           only care about config loading/writing.

  2. `isolated_db`       — extend (1) with a fresh history DB pointer and
                           `_conn = None`. History is not auto-init'd so
                           tests that want a pre-init state keep it. Use
                           for history unit tests that call init()/save()
                           themselves.

  3. `live_history`      — extend (2) with config.load() + history.init()
                           so the DB schema is ready and config module
                           variables are populated. Use for tests that
                           want "library ready to go" without boilerplate.

  4. `live_viewer`       — extend (3) with a running ThreadingHTTPServer
                           bound to a free port. Yields the base URL.
                           Use for viewer endpoint tests that want to
                           hit real HTTP.

Each fixture is opt-in (no autouse) so existing per-file fixtures keep
working while new or migrated tests use the shared ones. HTTP request
helpers (_http_get, _http_post) live here too for test_settings.py and
test_viewer_endpoints.py.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from voiceclip import config, history


# Every VOICECLIP_* env var the test suite clears. Centralized so adding
# a new env override is a one-line addition and all tests benefit.
_ENV_VARS_TO_CLEAR = [
    "VOICECLIP_MODEL",
    "VOICECLIP_ENGLISH_ONLY",
    "VOICECLIP_PERSONA",
    "VOICECLIP_HOTKEY",
    "VOICECLIP_HOTKEY_MODE",
    "VOICECLIP_HISTORY",
    "VOICECLIP_REFLECTION_HOTKEY",
    "VOICECLIP_REFLECTION_HOTKEY_MODE",
    "VOICECLIP_REFLECTION_MAX_DAYS",
    "VOICECLIP_SUMMARIES_PROVIDER",
    "VOICECLIP_SUMMARIES_LOCAL_MODEL",
    "VOICECLIP_SUMMARIES_OPENAI_MODEL",
    "VOICECLIP_SUMMARIES_ANTHROPIC_MODEL",
    "VOICECLIP_RESEARCH_PROVIDER",
    "VOICECLIP_RESEARCH_LOCAL_MODEL",
    "VOICECLIP_RESEARCH_OPENAI_MODEL",
    "VOICECLIP_RESEARCH_ANTHROPIC_MODEL",
    "VOICECLIP_PATTERNS_PROVIDER",
    "VOICECLIP_PATTERNS_LOCAL_MODEL",
    "VOICECLIP_PATTERNS_OPENAI_MODEL",
    "VOICECLIP_PATTERNS_ANTHROPIC_MODEL",
]


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point config at tmp_path and clear VOICECLIP_* env vars.

    Does NOT call config.load() — callers that want a loaded runtime
    should use `live_history` instead, or call config.load() themselves.
    """
    cfg_dir = str(tmp_path)
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config, "CONFIG_PATH", os.path.join(cfg_dir, "config.json"))
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    yield tmp_path


@pytest.fixture
def isolated_db(isolated_config, monkeypatch):
    """Extend isolated_config with a fresh history DB pointer.

    Does NOT call history.init() — tests that want pre-init behaviour
    (e.g. migration tests, reconnect tests) can observe `_conn is None`.
    """
    tmp_path = isolated_config
    monkeypatch.setattr(history, "DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.setattr(history, "_conn", None)
    yield tmp_path


@pytest.fixture
def live_history(isolated_db):
    """Extend isolated_db with config.load() + history.init().

    Use when a test wants "library ready to use": config module variables
    populated, DB schema created, writes work via save()/update_text()/etc.
    """
    config.load()
    history.init()
    try:
        yield isolated_db
    finally:
        history.close()


@pytest.fixture
def live_viewer(live_history):
    """Extend live_history with a running ThreadingHTTPServer on a free
    port. Yields the base URL (e.g. 'http://127.0.0.1:54321'). Handles
    clean shutdown in teardown.
    """
    from voiceclip.viewer import Handler
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)  # let the server become ready
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# HTTP request helpers — used by viewer endpoint tests
# ---------------------------------------------------------------------------

def http_get(url: str):
    """GET a URL, return parsed JSON. Raises on non-200."""
    return json.loads(urllib.request.urlopen(url).read())


def http_post(url: str, body: dict) -> tuple[int, dict]:
    """POST JSON to a URL, return (status_code, parsed_response).

    Non-2xx responses don't raise — the caller needs the status code to
    assert correct error handling.
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req)
        return resp.getcode(), json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
