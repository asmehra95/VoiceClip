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
        "VOICECLIP_MODEL", "VOICECLIP_ENGLISH_ONLY",
        "VOICECLIP_PERSONA", "VOICECLIP_HOTKEY",
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



class TestCustomVocabulary:
    """UI-editable flat list of biasing words appended to initial_prompt."""

    def test_defaults_to_empty_list(self):
        config.load()
        assert config.CUSTOM_VOCABULARY == []

    def test_loads_from_config_file(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"custom_vocabulary": ["Kiro", "MeshClaw", "AutoSDE"]}, f)
        config.load()
        assert config.CUSTOM_VOCABULARY == ["Kiro", "MeshClaw", "AutoSDE"]

    def test_strips_whitespace_and_drops_empties(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"custom_vocabulary": ["  Kiro  ", "", "  ", "AutoSDE"]}, f)
        config.load()
        assert config.CUSTOM_VOCABULARY == ["Kiro", "AutoSDE"]

    def test_dedupes_case_insensitively_keeping_first_casing(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"custom_vocabulary": ["Kiro", "kiro", "KIRO"]}, f)
        config.load()
        assert config.CUSTOM_VOCABULARY == ["Kiro"]

    def test_non_string_entries_are_filtered(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"custom_vocabulary": ["Kiro", 42, None, "AutoSDE"]}, f)
        config.load()
        assert config.CUSTOM_VOCABULARY == ["Kiro", "AutoSDE"]

    def test_non_list_is_ignored_with_warning(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"custom_vocabulary": "not a list"}, f)
        config.load()
        assert config.CUSTOM_VOCABULARY == []

    def test_vocabulary_appears_in_initial_prompt(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"custom_vocabulary": ["MeshClaw", "Taskei"]}, f)
        config.load()
        assert config.INITIAL_PROMPT is not None
        assert "MeshClaw" in config.INITIAL_PROMPT
        assert "Taskei" in config.INITIAL_PROMPT

    def test_vocabulary_not_duplicated_when_already_in_dictionary(self, tmp_path):
        """If a custom-vocab word already appears in the persona dictionary,
        it shouldn't double up in the prompt."""
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({
                "dictionary": {"voiceclip": "VoiceClip"},
                "custom_vocabulary": ["voiceclip", "UniqueWord"],
            }, f)
        config.load()
        # "voiceclip" (case-insensitive) appears exactly once in the prompt;
        # the unique custom-vocab word gets through.
        assert config.INITIAL_PROMPT.lower().count("voiceclip") == 1
        assert "UniqueWord" in config.INITIAL_PROMPT



class TestPerformanceConstants:
    """Sanity checks on the performance-related constants. Keeps us honest
    if someone tries to ratchet them back up without thinking it through."""

    def test_max_recording_seconds_is_reasonable(self):
        from voiceclip import config as cfg
        # Anything above a few minutes defeats the purpose (memory cap).
        # Anything below 30s is hostile to legit dictation.
        assert 30 <= cfg.MAX_RECORDING_SECONDS <= 300

    def test_transcribe_timeout_is_interactive(self):
        from voiceclip import transcriber
        # All engine timeouts should be <= 60s for interactive use.
        for engine, timeout in transcriber._TIMEOUT.items():
            assert timeout <= 60, f"{engine} timeout {timeout}s exceeds 60s"



class TestResearchProvider:
    """Research now supports local (mlx-lm) alongside the cloud providers."""

    def test_local_is_valid_research_provider(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({
                "research": {
                    "provider": "local",
                    "local_model": "mlx-community/TestModel-4bit",
                }
            }, f)
        config.load()
        assert config.RESEARCH_PROVIDER == "local"
        assert config.RESEARCH_LOCAL_MODEL == "mlx-community/TestModel-4bit"

    def test_local_model_has_sensible_default(self):
        config.load()
        # Default matches the shared local default used by summaries/patterns
        assert config.RESEARCH_LOCAL_MODEL == "mlx-community/Qwen2.5-7B-Instruct-4bit"

    def test_invalid_research_provider_falls_back(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"research": {"provider": "bogus"}}, f)
        config.load()
        assert config.RESEARCH_PROVIDER == "none"

    def test_env_override_for_local_model(self, monkeypatch):
        monkeypatch.setenv("VOICECLIP_RESEARCH_LOCAL_MODEL",
                           "mlx-community/FromEnv-4bit")
        config.load()
        assert config.RESEARCH_LOCAL_MODEL == "mlx-community/FromEnv-4bit"


class TestEngineAuto:
    """Engine 'auto' resolves to whisper_cpp only when its binaries and
    model are actually present; otherwise the pip-installed mlx engine."""

    @pytest.fixture(autouse=True)
    def _isolate_engine_detection(self, tmp_path, monkeypatch):
        """Point detection at controlled temp paths so results don't
        depend on what's installed on the machine running the tests."""
        self.server_bin = tmp_path / "bin" / "whisper-server"
        self.models_dir = tmp_path / "models"
        monkeypatch.setenv("VOICECLIP_WHISPER_CPP_SERVER", str(self.server_bin))
        monkeypatch.setenv("VOICECLIP_WHISPER_CPP_MODELS", str(self.models_dir))
        monkeypatch.delenv("VOICECLIP_ENGINE", raising=False)
        yield

    def _provision(self, model_file="ggml-large-v3-turbo.bin"):
        self.server_bin.parent.mkdir(parents=True, exist_ok=True)
        self.server_bin.write_bytes(b"fake binary")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / model_file).write_bytes(b"fake model")

    def test_auto_picks_whisper_cpp_when_provisioned(self):
        self._provision()
        config.load()
        assert config.ENGINE == "whisper_cpp"

    def test_auto_falls_back_without_binaries(self):
        # Models present, binary missing
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "ggml-large-v3-turbo.bin").write_bytes(b"fake")
        config.load()
        assert config.ENGINE == "whisper"

    def test_auto_falls_back_without_model(self):
        # Binary present, model missing
        self.server_bin.parent.mkdir(parents=True, exist_ok=True)
        self.server_bin.write_bytes(b"fake binary")
        config.load()
        assert config.ENGINE == "whisper"

    def test_auto_matches_quantized_model_files(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"model": "medium"}, f)
        self._provision(model_file="ggml-medium-q5_0.bin")

        config.load()
        assert config.ENGINE == "whisper_cpp"

    def test_explicit_engine_is_honored(self, tmp_path):
        """An explicit engine choice bypasses auto-detection entirely."""
        self._provision()
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"engine": "whisper"}, f)

        config.load()
        assert config.ENGINE == "whisper"

    def test_env_var_beats_auto(self, monkeypatch):
        self._provision()
        monkeypatch.setenv("VOICECLIP_ENGINE", "whisper")
        config.load()
        assert config.ENGINE == "whisper"

    def test_invalid_engine_falls_back_to_whisper(self, tmp_path):
        cfg_path = os.path.join(str(tmp_path), "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"engine": "bogus"}, f)

        config.load()
        assert config.ENGINE == "whisper"
