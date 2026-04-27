"""Integration tests for the viewer HTTP endpoints.

These spin up the real ThreadingHTTPServer against a temp DB and hit each
endpoint via urllib. LLM providers are disabled (provider='none') so no
network traffic; those endpoints are exercised for their disabled-guard
responses only.
"""

import json
import os
import tempfile
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime
from http.server import ThreadingHTTPServer

import pytest

from voiceclip import config, history
from voiceclip.viewer import Handler


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Start a viewer server on a free port pointed at a temp DB."""
    monkeypatch.setattr(history, "DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
    config.load()
    history.init()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()
        history.close()


def _get(url):
    return json.loads(urllib.request.urlopen(url).read())


def _post(url, body):
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


class TestGetEndpoints:
    def test_index_renders_html(self, server):
        body = urllib.request.urlopen(server).read().decode()
        assert "<title>VoiceClip journal</title>" in body

    def test_days_empty(self, server):
        r = _get(server + "/api/days")
        assert r == {"days": []}

    def test_day_empty(self, server):
        today = datetime.now().strftime("%Y-%m-%d")
        r = _get(f"{server}/api/day?date={today}")
        assert r["entries"] == []
        assert r["stats"]["transcriptions"] == 0

    def test_day_rejects_bad_date(self, server):
        try:
            urllib.request.urlopen(f"{server}/api/day?date=nope")
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_queue_listing_shape(self, server):
        r = _get(server + "/api/queue")
        assert "topics" in r
        assert "research_enabled" in r
        assert r["research_enabled"] is False

    def test_patterns_config_off_by_default(self, server):
        r = _get(server + "/api/patterns/config")
        assert r["enabled"] is False


class TestPostEndpoints:
    def test_update_roundtrip(self, server):
        i = history.save("raw", "Original text.", 1.0)
        code, r = _post(f"{server}/api/update", {"id": i, "text": "Edited text."})
        assert code == 200
        assert r["entry"]["text"] == "Edited text."

    def test_update_rejects_empty(self, server):
        i = history.save("raw", "Original.", 1.0)
        code, r = _post(f"{server}/api/update", {"id": i, "text": "   "})
        assert code == 400

    def test_delete_flow(self, server):
        i = history.save("raw", "Bye.", 1.0)
        code, r = _post(f"{server}/api/delete", {"id": i})
        assert code == 200
        assert r["entry"]["id"] == i

    def test_promote_flow(self, server):
        i = history.save("raw", "Promote me.", 1.0, kind="transcription")
        code, r = _post(f"{server}/api/promote", {"id": i})
        assert code == 200

    def test_research_create_and_dedup(self, server):
        code, r = _post(
            f"{server}/api/research/create",
            {"text": "CRDTs vs OT"},
        )
        assert code == 200
        first_id = r["id"]

        # Same topic via the patterns/queue path should dedup
        code, r2 = _post(
            f"{server}/api/patterns/queue",
            {"topic": "CRDTs vs OT", "reason": "from patterns"},
        )
        assert code == 200
        assert r2["duplicate"] is True
        assert r2["id"] == first_id

    def test_research_run_disabled(self, server):
        # Provider defaults to 'none'
        code, r = _post(f"{server}/api/research/run", {"id": 1})
        assert code == 400
        assert "disabled" in r["error"].lower()

    def test_patterns_run_disabled(self, server):
        code, r = _post(f"{server}/api/patterns/run", {})
        assert code == 400
        assert "off" in r["error"].lower() or "provider" in r["error"].lower()

    def test_summarize_run_disabled(self, server):
        code, r = _post(f"{server}/api/summarize", {})
        assert code == 400

    def test_bad_payload(self, server):
        code, r = _post(f"{server}/api/delete", {"id": "not-an-int"})
        assert code == 400



class TestBriefUpdateEndpoint:
    """POST /api/research/update_brief edits an existing brief's text."""

    def test_update_happy_path(self, server):
        tid = history.create_research_topic("CRDTs")
        bid = history.save_brief(tid, status="done", brief_text="before",
                                 provider="openai", model="gpt-4o-mini")
        code, r = _post(f"{server}/api/research/update_brief",
                        {"brief_id": bid, "text": "after"})
        assert code == 200
        assert r["brief"]["text"] == "after"

    def test_update_rejects_empty_text(self, server):
        tid = history.create_research_topic("X")
        bid = history.save_brief(tid, status="done", brief_text="whatever",
                                 provider="openai", model="gpt-4o-mini")
        code, r = _post(f"{server}/api/research/update_brief",
                        {"brief_id": bid, "text": "   "})
        assert code == 400

    def test_update_rejects_missing_brief(self, server):
        code, r = _post(f"{server}/api/research/update_brief",
                        {"brief_id": 99999, "text": "x"})
        assert code == 404

    def test_update_rejects_bad_id(self, server):
        code, r = _post(f"{server}/api/research/update_brief",
                        {"brief_id": "not-int", "text": "x"})
        assert code == 400


class TestArchiveEndpoints:
    """POST /api/research/archive and /api/research/unarchive.

    Also covers that GET /api/queue surfaces an `archived` array alongside
    the active topics so the UI only needs one round-trip on tab switch.
    """

    def test_archive_round_trip(self, server):
        tid = history.create_research_topic("Raft")
        code, r = _post(f"{server}/api/research/archive", {"id": tid})
        assert code == 200
        assert r["topic"]["id"] == tid
        assert r["topic"]["archived_at"]

        # Queue listing should exclude it from `topics` and include it in
        # `archived`
        data = _get(server + "/api/queue")
        assert all(t["id"] != tid for t in data["topics"])
        assert any(a["id"] == tid for a in data["archived"])

        code, r = _post(f"{server}/api/research/unarchive", {"id": tid})
        assert code == 200

        data = _get(server + "/api/queue")
        assert any(t["id"] == tid for t in data["topics"])
        assert all(a["id"] != tid for a in data["archived"])

    def test_archive_rejects_bad_id(self, server):
        code, r = _post(f"{server}/api/research/archive", {"id": "nope"})
        assert code == 400

    def test_archive_rejects_missing_topic(self, server):
        code, r = _post(f"{server}/api/research/archive", {"id": 99999})
        assert code == 404

    def test_archive_rejects_non_topic_entry(self, server):
        """A normal transcription isn't archivable — the endpoint should
        reject rather than silently mark something random as archived."""
        eid = history.save("raw", "Regular entry.", 1.0)
        code, r = _post(f"{server}/api/research/archive", {"id": eid})
        assert code == 404

    def test_unarchive_rejects_bad_id(self, server):
        code, r = _post(f"{server}/api/research/unarchive", {"id": "nope"})
        assert code == 400

    def test_unarchive_rejects_missing_topic(self, server):
        code, r = _post(f"{server}/api/research/unarchive", {"id": 99999})
        assert code == 404

    def test_queue_endpoint_shape_includes_archived(self, server):
        """Even when empty, the `archived` key must be present so the UI
        can rely on it without null-checks."""
        data = _get(server + "/api/queue")
        assert "archived" in data
        assert data["archived"] == []


class TestCustomVocabularySetting:
    """Settings endpoint accepts and validates the custom_vocabulary list."""

    def test_get_exposes_schema_and_value(self, server):
        data = _get(server + "/api/settings")
        assert "custom_vocabulary" in data["schema"]
        assert data["schema"]["custom_vocabulary"]["type"] == "text_list"
        assert data["values"]["custom_vocabulary"] == []

    def test_accepts_list(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": ["Kiro", "MeshClaw"]})
        assert code == 200
        # New value should round-trip through GET
        after = _get(server + "/api/settings")
        assert after["values"]["custom_vocabulary"] == ["Kiro", "MeshClaw"]

    def test_accepts_newline_string_from_textarea(self, server):
        """The UI sends a newline-joined string; server normalizes to list."""
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": "Kiro\nMeshClaw\n\n  AutoSDE  "})
        assert code == 200
        after = _get(server + "/api/settings")
        assert after["values"]["custom_vocabulary"] == ["Kiro", "MeshClaw", "AutoSDE"]

    def test_dedupes_case_insensitively(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": ["Kiro", "kiro", "KIRO"]})
        assert code == 200
        after = _get(server + "/api/settings")
        assert after["values"]["custom_vocabulary"] == ["Kiro"]

    def test_rejects_non_string_entries(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": ["ok", 42]})
        assert code == 400

    def test_rejects_overly_long_entry(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": ["x" * 200]})
        assert code == 400

    def test_rejects_too_many_entries(self, server):
        huge = [f"word{i}" for i in range(250)]
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": huge})
        assert code == 400

    def test_rejects_wrong_type(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": 42})
        assert code == 400


class TestResearchLocalProvider:
    """Settings schema exposes `local` as a valid research provider choice,
    and /api/queue surfaces the local model when the provider is set to it."""

    def test_schema_includes_local_choice(self, server):
        data = _get(server + "/api/settings")
        choices = data["schema"]["research.provider"]["choices"]
        assert "local" in choices
        # local is NOT cloud — cloud_providers stays narrow
        assert "local" not in data["schema"]["research.provider"]["cloud_providers"]

    def test_schema_exposes_local_model_field(self, server):
        data = _get(server + "/api/settings")
        assert "research.local_model" in data["schema"]
        assert data["schema"]["research.local_model"]["type"] == "text"

    def test_settings_update_accepts_local_provider(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"research.provider": "local"})
        assert code == 200

    def test_settings_update_accepts_local_model(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"research.local_model": "mlx-community/Test-4bit"})
        assert code == 200
        after = _get(server + "/api/settings")
        assert after["values"]["research.local_model"] == "mlx-community/Test-4bit"
