"""Tests for the viewer Settings endpoints + config_io merge helper.

Covers:
  - GET /api/settings shape (schema + values + system info)
  - POST /api/settings/update happy path + validation rejections
  - Restart-required flag propagates correctly
  - config_io.write_config_patch semantics (new file, merge, nested,
    corrupt-recovery, 0600 perms)
"""

import json
import os
import urllib.request
from pathlib import Path

import pytest

from tests.conftest import http_get as _get
from tests.conftest import http_post as _post
from voiceclip import config, config_io, history


@pytest.fixture
def server(live_viewer):
    """Back-compat alias: older tests here use `server` as the URL value."""
    return live_viewer


class TestConfigIo:
    """config_io.write_config_patch — the shared merge helper."""

    def test_write_to_fresh_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
        assert config_io.write_config_patch({"history": True}) is True
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        assert data == {"history": True}

    def test_merge_preserves_unrelated_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
        Path(config.CONFIG_PATH).write_text(json.dumps({"model": "tiny", "hotkey": "alt_r"}))
        config_io.write_config_patch({"history": True, "hotkey": "f5"})
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        assert data["model"] == "tiny"
        assert data["hotkey"] == "f5"
        assert data["history"] is True

    def test_shallow_merge_of_nested_dicts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
        Path(config.CONFIG_PATH).write_text(json.dumps({
            "summaries": {"provider": "openai", "openai_model": "gpt-4o-mini"},
        }))
        config_io.write_config_patch({"summaries": {"provider": "local"}})
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        assert data["summaries"]["provider"] == "local"
        # Unrelated nested key preserved
        assert data["summaries"]["openai_model"] == "gpt-4o-mini"

    def test_survives_corrupt_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
        Path(config.CONFIG_PATH).write_text("{ not json")
        assert config_io.write_config_patch({"history": True}) is True
        assert json.loads(Path(config.CONFIG_PATH).read_text()) == {"history": True}

    def test_permissions_after_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
        config_io.write_config_patch({"history": True})
        mode = os.stat(config.CONFIG_PATH).st_mode & 0o777
        assert mode == 0o600

    def test_read_config_returns_dict_even_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "nope.json"))
        assert config_io.read_config() == {}


class TestSettingsGet:
    def test_get_returns_schema_and_values(self, server):
        data = _get(server + "/api/settings")
        assert "schema" in data
        assert "values" in data
        assert "system" in data

    def test_schema_includes_expected_keys(self, server):
        data = _get(server + "/api/settings")
        for key in ("hotkey", "hotkey_mode", "history",
                    "summaries.provider", "research.provider",
                    "patterns.provider"):
            assert key in data["schema"], f"missing: {key}"

    def test_schema_marks_restart_required_keys(self, server):
        data = _get(server + "/api/settings")
        # Hotkey is hot-swap-hostile — daemon must restart to rebind
        assert data["schema"]["hotkey"].get("restart_required") is True
        # Patterns provider applies on next LLM call — no restart
        assert data["schema"]["patterns.provider"].get("restart_required") is not True

    def test_values_match_runtime(self, server):
        data = _get(server + "/api/settings")
        assert data["values"]["hotkey"] == config.HOTKEY
        assert data["values"]["summaries.provider"] == config.SUMMARIES_PROVIDER


class TestSettingsUpdate:
    def test_update_simple_field(self, server):
        code, r = _post(server + "/api/settings/update", {"hotkey": "f5"})
        assert code == 200
        assert r["ok"] is True
        assert r["restart_required"] is True
        # And the persisted file reflects it
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        assert data["hotkey"] == "f5"

    def test_update_nested_field(self, server):
        code, r = _post(server + "/api/settings/update",
                        {"summaries.provider": "local"})
        assert code == 200
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        assert data["summaries"]["provider"] == "local"

    def test_rejects_unknown_key(self, server):
        code, r = _post(server + "/api/settings/update",
                        {"malicious_key": "whatever"})
        assert code == 400
        assert "unknown setting" in r["error"]

    def test_rejects_bad_type(self, server):
        code, r = _post(server + "/api/settings/update",
                        {"history": "yes-please"})
        assert code == 400
        assert "true or false" in r["error"]

    def test_rejects_out_of_range_int(self, server):
        code, r = _post(server + "/api/settings/update",
                        {"patterns.window_days": 500})
        assert code == 400

    def test_rejects_invalid_select(self, server):
        code, r = _post(server + "/api/settings/update",
                        {"hotkey_mode": "doubletap"})
        assert code == 400

    def test_rejects_empty_patch(self, server):
        code, r = _post(server + "/api/settings/update", {})
        assert code == 400

    def test_update_preserves_other_keys(self, server):
        # Seed existing config that has unrelated values
        Path(config.CONFIG_PATH).write_text(json.dumps({
            "model": "tiny",
            "summaries": {"provider": "none", "openai_model": "gpt-4o-mini"},
        }))
        code, r = _post(server + "/api/settings/update",
                        {"summaries.provider": "local"})
        assert code == 200
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        assert data["model"] == "tiny"                              # unrelated key preserved
        assert data["summaries"]["openai_model"] == "gpt-4o-mini"   # unrelated nested key preserved
        assert data["summaries"]["provider"] == "local"             # target changed

    def test_no_restart_flag_for_provider_only_changes(self, server):
        code, r = _post(server + "/api/settings/update",
                        {"summaries.provider": "local"})
        assert code == 200
        assert r["restart_required"] is False

    def test_restart_flag_for_hotkey_change(self, server):
        code, r = _post(server + "/api/settings/update",
                        {"hotkey": "f5"})
        assert code == 200
        assert r["restart_required"] is True


class TestSystemInfo:
    def test_system_block_present(self, server):
        data = _get(server + "/api/settings")
        sys = data["system"]
        assert "db_path" in sys
        assert "config_path" in sys
        assert "history_enabled" in sys
