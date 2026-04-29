"""Tests for the first-run onboarding flow.

The interactive walkthrough itself is gated behind `sys.stdout.isatty()`,
so we don't drive it end-to-end in tests. We cover:

  - Marker file behavior (needs_onboarding / mark_done)
  - Config patch writer (merges into existing config.json, survives missing
    file, merges nested dicts correctly)
  - TTY-gate: non-TTY runs are silent no-ops
"""

import json
import os
import pytest
from pathlib import Path

from voiceclip import config, onboard


@pytest.fixture(autouse=True)
def _isolated(isolated_config):
    """Use shared isolated_config — onboard tests only need config isolation."""
    yield


class TestMarker:
    def test_needs_onboarding_when_marker_missing(self, monkeypatch):
        monkeypatch.setattr(onboard, "_IS_TTY", True)
        assert onboard.needs_onboarding() is True

    def test_mark_done_creates_file(self):
        onboard.mark_done()
        marker = Path(config.CONFIG_DIR) / ".onboarded"
        assert marker.exists()

    def test_mark_done_then_needs_onboarding_false(self, monkeypatch):
        monkeypatch.setattr(onboard, "_IS_TTY", True)
        onboard.mark_done()
        assert onboard.needs_onboarding() is False

    def test_non_tty_never_needs_onboarding(self, monkeypatch):
        monkeypatch.setattr(onboard, "_IS_TTY", False)
        assert onboard.needs_onboarding() is False


class TestConfigPatch:
    def test_writes_new_file_when_none_exists(self):
        ok = onboard._write_config_patch({"history": True})
        assert ok is True
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        assert data["history"] is True

    def test_merges_into_existing_config(self):
        Path(config.CONFIG_PATH).write_text(json.dumps({
            "model": "tiny",
            "hotkey": "alt_r",
        }))
        onboard._write_config_patch({"history": True, "hotkey": "f5"})
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        # Unrelated key preserved
        assert data["model"] == "tiny"
        # Existing key overwritten
        assert data["hotkey"] == "f5"
        # New key added
        assert data["history"] is True

    def test_merges_nested_dicts(self):
        Path(config.CONFIG_PATH).write_text(json.dumps({
            "summaries": {
                "provider": "openai",
                "openai_model": "gpt-4o-mini",
            },
        }))
        onboard._write_config_patch({"summaries": {"provider": "local"}})
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        # Merged: provider changed, but openai_model preserved
        assert data["summaries"]["provider"] == "local"
        assert data["summaries"]["openai_model"] == "gpt-4o-mini"

    def test_patch_survives_corrupt_existing_config(self):
        Path(config.CONFIG_PATH).write_text("{ not valid json")
        ok = onboard._write_config_patch({"history": True})
        assert ok is True
        data = json.loads(Path(config.CONFIG_PATH).read_text())
        assert data == {"history": True}

    def test_patch_sets_file_permissions(self):
        onboard._write_config_patch({"history": True})
        mode = os.stat(config.CONFIG_PATH).st_mode & 0o777
        assert mode == 0o600


class TestRun:
    def test_run_non_tty_is_silent_noop(self, monkeypatch):
        monkeypatch.setattr(onboard, "_IS_TTY", False)
        assert onboard.run() is False
        # Did not create a marker file (gated on TTY check earlier)
        marker = Path(config.CONFIG_DIR) / ".onboarded"
        assert not marker.exists()

    def test_run_skipped_if_already_onboarded(self, monkeypatch, capsys):
        monkeypatch.setattr(onboard, "_IS_TTY", True)
        onboard.mark_done()
        assert onboard.run() is False
        # No prompts printed
        out = capsys.readouterr().out
        assert out == ""


class TestHistoryDetection:
    def test_has_history_enabled_reads_config(self):
        Path(config.CONFIG_PATH).write_text(json.dumps({"history": True}))
        assert onboard._has_history_enabled() is True

    def test_has_history_enabled_false_when_missing(self):
        assert onboard._has_history_enabled() is False

    def test_has_history_enabled_false_on_corrupt_config(self):
        Path(config.CONFIG_PATH).write_text("garbage")
        assert onboard._has_history_enabled() is False
