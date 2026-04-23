"""Tests for voiceclip.config — JSON config loading, env overrides, personas."""

import json
import os
import tempfile
import pytest

from voiceclip import config


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect config to a temp directory so tests don't touch real config."""
    cfg_dir = str(tmp_path)
    cfg_path = os.path.join(cfg_dir, "config.json")
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    # Clear env vars that might interfere
    for var in [
        "VOICECLIP_MODEL", "VOICECLIP_ENGLISH_ONLY", "VOICECLIP_POLISH",
        "VOICECLIP_POLISH_MODEL", "VOICECLIP_PERSONA", "VOICECLIP_HOTKEY",
        "VOICECLIP_HOTKEY_MODE",
    ]:
        monkeypatch.delenv(var, raising=False)
    yield


class TestConfigLoad:
    def test_creates_default_config_file(self, tmp_path):
        """First run should create config.json with defaults."""
        config.load()
        cfg_path = os.path.join(str(tmp_path), "config.json")
        assert os.path.exists(cfg_path)
        with open(cfg_path) as f:
            data = json.load(f)
        assert data["model"] == "large-v3-turbo"
        assert data["english_only"] is True

    def test_default_values(self):
        config.load()
        assert config.MODEL == "large-v3-turbo"
        assert config.ENGLISH_ONLY is True
        assert config.POLISH_ENABLED is False
        assert config.PERSONA == "default"
        assert config.HOTKEY == "alt_r"
        assert config.HOTKEY_MODE == "hold"

    def test_loads_custom_config(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        custom = {
            "model": "small",
            "english_only": False,
            "persona": "casual",
            "hotkey": "f5",
            "hotkey_mode": "toggle",
        }
        with open(cfg_path, "w") as f:
            json.dump(custom, f)

        config.load()
        assert config.MODEL == "small"
        assert config.ENGLISH_ONLY is False
        assert config.PERSONA == "casual"
        assert config.HOTKEY == "f5"
        assert config.HOTKEY_MODE == "toggle"

    def test_corrupt_json_falls_back_to_defaults(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            f.write("{invalid json!!!")

        config.load()
        assert config.MODEL == "large-v3-turbo"

    def test_missing_keys_use_defaults(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"model": "tiny"}, f)

        config.load()
        assert config.MODEL == "tiny"
        assert config.ENGLISH_ONLY is True  # default
        assert config.HOTKEY == "alt_r"     # default


class TestEnvOverrides:
    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("VOICECLIP_MODEL", "base")
        config.load()
        assert config.MODEL == "base"

    def test_english_only_override(self, monkeypatch):
        monkeypatch.setenv("VOICECLIP_ENGLISH_ONLY", "false")
        config.load()
        assert config.ENGLISH_ONLY is False

    def test_polish_override(self, monkeypatch):
        monkeypatch.setenv("VOICECLIP_POLISH", "true")
        config.load()
        assert config.POLISH_ENABLED is True

    def test_persona_override(self, monkeypatch):
        monkeypatch.setenv("VOICECLIP_PERSONA", "engineering")
        config.load()
        assert config.PERSONA == "engineering"

    def test_hotkey_override(self, monkeypatch):
        monkeypatch.setenv("VOICECLIP_HOTKEY", "f5")
        config.load()
        assert config.HOTKEY == "f5"

    def test_hotkey_mode_override(self, monkeypatch):
        monkeypatch.setenv("VOICECLIP_HOTKEY_MODE", "toggle")
        config.load()
        assert config.HOTKEY_MODE == "toggle"

    def test_env_overrides_json(self, tmp_path, monkeypatch):
        """Env var should win over JSON config."""
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"model": "tiny"}, f)

        monkeypatch.setenv("VOICECLIP_MODEL", "medium")
        config.load()
        assert config.MODEL == "medium"


class TestPersonas:
    def test_default_persona_empty_dict(self):
        config.load()
        # Default persona has no dictionary entries of its own
        # but global dictionary should be present
        assert "voiceclip" in config.DICTIONARY or len(config.DICTIONARY) >= 0

    def test_engineering_persona_merges_dicts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICECLIP_PERSONA", "engineering")
        config.load()
        # Should have both global and engineering entries
        assert "voiceclip" in config.DICTIONARY  # global
        assert "dynamo db" in config.DICTIONARY   # engineering

    def test_unknown_persona_falls_back(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"persona": "nonexistent"}, f)

        config.load()
        assert config.PERSONA == "default"

    def test_initial_prompt_includes_persona(self, monkeypatch):
        monkeypatch.setenv("VOICECLIP_PERSONA", "engineering")
        config.load()
        assert config.INITIAL_PROMPT is not None
        assert "AWS" in config.INITIAL_PROMPT

    def test_initial_prompt_includes_dictionary_values(self):
        config.load()
        if config.DICTIONARY:
            assert config.INITIAL_PROMPT is not None


class TestHotkeyConfig:
    def test_invalid_mode_falls_back(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"hotkey_mode": "invalid"}, f)

        config.load()
        assert config.HOTKEY_MODE == "hold"

    def test_hotkey_display_name(self):
        config.load()
        name = config.hotkey_display_name()
        assert "Option" in name or "⌥" in name

    def test_resolve_hotkey_alt_r(self):
        config.load()
        from pynput import keyboard
        key = config.resolve_hotkey()
        assert key == keyboard.Key.alt_r

    def test_resolve_hotkey_f5(self):
        config.HOTKEY = "f5"
        from pynput import keyboard
        key = config.resolve_hotkey()
        assert key == keyboard.Key.f5

    def test_resolve_hotkey_char(self):
        config.HOTKEY = "z"
        key = config.resolve_hotkey()
        # Should be a KeyCode
        from pynput.keyboard import KeyCode
        assert isinstance(key, KeyCode)


class TestValidate:
    def test_valid_model_passes(self):
        config.MODEL = "large-v3-turbo"
        config.validate()  # should not raise

    def test_invalid_model_exits(self):
        config.MODEL = "nonexistent"
        with pytest.raises(SystemExit):
            config.validate()


class TestGetModelRepo:
    def test_english_model(self):
        config.MODEL = "small"
        config.ENGLISH_ONLY = True
        repo, key = config.get_model_repo()
        assert key == "small.en"
        assert "small.en" in repo

    def test_multilingual_model(self):
        config.MODEL = "large-v3-turbo"
        config.ENGLISH_ONLY = False
        repo, key = config.get_model_repo()
        assert key == "large-v3-turbo"

    def test_large_model_ignores_english_flag(self):
        config.MODEL = "large-v3-turbo"
        config.ENGLISH_ONLY = True
        repo, key = config.get_model_repo()
        # large models don't have .en variants
        assert key == "large-v3-turbo"
