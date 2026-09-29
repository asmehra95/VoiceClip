"""Shared AI model (one model for every AI feature), the viewer's CSRF
guard, and the Agent tab's supervision endpoints."""

import json
import os
import urllib.error
import urllib.request

import pytest

from tests.conftest import http_get as http_get_json
from tests.conftest import http_post
from voiceclip import config

# ---------------------------------------------------------------------------
# Shared AI model in config
# ---------------------------------------------------------------------------

class TestSharedAIConfig:
    def _write(self, cfg):
        with open(config.CONFIG_PATH, "w") as f:
            json.dump(cfg, f)
        config.load()

    def test_ai_block_overrides_every_feature(self, isolated_config):
        self._write({"summaries": {"provider": "local"},
                     "ai": {"provider": "cloud", "model": "assistant"}})
        for feat in ("summaries", "research", "patterns"):
            assert getattr(config, f"{feat.upper()}_PROVIDER") == "cloud"
            assert config.model_id_for(feat) == "assistant"

    def test_no_ai_block_keeps_legacy_per_feature(self, isolated_config):
        self._write({"summaries": {"provider": "local"}})
        assert config.AI_PROVIDER == ""
        assert config.SUMMARIES_PROVIDER == "local"
        assert config.RESEARCH_PROVIDER == "none"

    def test_ai_off_turns_everything_off(self, isolated_config):
        self._write({"summaries": {"provider": "local"},
                     "ai": {"provider": "none", "model": ""}})
        assert config.SUMMARIES_PROVIDER == "none"
        assert config.PATTERNS_PROVIDER == "none"

    def test_invalid_provider_ignored(self, isolated_config):
        self._write({"ai": {"provider": "bogus", "model": "x"}})
        assert config.AI_PROVIDER == ""

    def test_assistant_and_agent_follow_cloud_model(self, isolated_config):
        from voiceclip import agent, assistant
        self._write({"ai": {"provider": "cloud", "model": "big-brain"}})
        assert assistant._llm_route() == "big-brain"
        assert agent._llm_route() == "big-brain"
        self._write({"ai": {"provider": "local", "model": "mlx-community/x"}})
        assert assistant._llm_route() == "assistant"


# ---------------------------------------------------------------------------
# Settings API: ai.model
# ---------------------------------------------------------------------------

class TestAIModelSetting:
    def test_write_and_read_back(self, live_viewer):
        status, body = http_post(f"{live_viewer}/api/settings/update",
                                 {"ai.model": "cloud:assistant"})
        assert status == 200, body
        with open(config.CONFIG_PATH) as f:
            saved = json.load(f)
        assert saved["ai"] == {"provider": "cloud", "model": "assistant"}
        values = http_get_json(f"{live_viewer}/api/settings")["values"]
        assert values["ai.model"] == "cloud:assistant"

    def test_off(self, live_viewer):
        status, _ = http_post(f"{live_viewer}/api/settings/update", {"ai.model": "none"})
        assert status == 200
        assert config.SUMMARIES_PROVIDER == "none"

    @pytest.mark.parametrize("bad", ["cloud", "bogus:model", "local:", 42])
    def test_rejects_malformed(self, live_viewer, bad):
        status, _ = http_post(f"{live_viewer}/api/settings/update", {"ai.model": bad})
        assert status == 400

    def test_per_feature_fields_hidden_but_still_accepted(self, live_viewer):
        schema = http_get_json(f"{live_viewer}/api/settings")["schema"]
        assert schema["summaries.provider"].get("hidden") is True
        status, _ = http_post(f"{live_viewer}/api/settings/update",
                              {"summaries.provider": "local"})
        assert status == 200

    def test_models_endpoint_lists_all_sources(self, live_viewer, monkeypatch):
        data = http_get_json(f"{live_viewer}/api/ai/models")
        providers = {m["provider"] for m in data["models"]}
        assert {"cloud", "local", "openai", "anthropic"} <= providers
        assert "current" in data


# ---------------------------------------------------------------------------
# CSRF guard on POST
# ---------------------------------------------------------------------------

class TestPostGuard:
    def _raw_post(self, url, body, headers):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            return urllib.request.urlopen(req).getcode()
        except urllib.error.HTTPError as e:
            return e.code

    def test_form_encoded_post_rejected(self, live_viewer):
        code = self._raw_post(f"{live_viewer}/api/settings/update",
                              b'{"history": false}',
                              {"Content-Type": "text/plain"})
        assert code == 403

    def test_foreign_origin_rejected(self, live_viewer):
        code = self._raw_post(f"{live_viewer}/api/settings/update",
                              b'{"history": true}',
                              {"Content-Type": "application/json",
                               "Origin": "https://evil.example"})
        assert code == 403

    def test_foreign_host_rejected(self, live_viewer):
        code = self._raw_post(f"{live_viewer}/api/agent/start", b"{}",
                              {"Content-Type": "application/json",
                               "Host": "rebind.evil.example:8723"})
        assert code == 403

    def test_same_origin_json_allowed(self, live_viewer):
        code = self._raw_post(f"{live_viewer}/api/settings/update",
                              b'{"history": true}',
                              {"Content-Type": "application/json",
                               "Origin": live_viewer})
        assert code == 200


# ---------------------------------------------------------------------------
# Agent event log + status endpoint
# ---------------------------------------------------------------------------

class TestAgentEvents:
    def test_event_log_and_status(self, live_viewer):
        from voiceclip import agent
        log = agent.EventLog(os.path.join(agent.agent_dir(), "events.jsonl"))
        log.emit("state", state="ready")
        log.emit("user", text="what's new")
        log.emit("tool", name="web_search")
        log.emit("assistant", text="Plenty.")
        s = http_get_json(f"{live_viewer}/api/agent/status?after=0")
        assert s["running"] is False
        assert [e["type"] for e in s["events"]] == ["state", "user", "tool", "assistant"]
        assert s["session"] is not None
        s2 = http_get_json(f"{live_viewer}/api/agent/status?after=2")
        assert [e["seq"] for e in s2["events"]] == [3, 4]

    def test_new_session_truncates(self, live_viewer):
        from voiceclip import agent
        path = os.path.join(agent.agent_dir(), "events.jsonl")
        agent.EventLog(path).emit("user", text="old")
        agent.EventLog(path).emit("user", text="new")
        s = http_get_json(f"{live_viewer}/api/agent/status?after=0")
        assert [e["text"] for e in s["events"]] == ["new"]

    def test_stop_when_not_running(self, live_viewer):
        status, body = http_post(f"{live_viewer}/api/agent/stop", {})
        assert status == 200 and body["was_running"] is False
