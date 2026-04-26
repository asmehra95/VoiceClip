"""Tests for the cloud-provider consent banner.

The banner should fire:
- Once when a feature first switches to a cloud provider
- Again if the provider or model changes
- Never when all providers are 'none'
- Never again on unchanged config (no nag on every launch)
"""

import json
import os
import pytest

from voiceclip import config, consent


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
    # Reset all provider configs to 'none' for a clean slate each test
    monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "none")
    monkeypatch.setattr(config, "RESEARCH_PROVIDER", "none")
    monkeypatch.setattr(config, "PATTERNS_PROVIDER", "none")
    yield


class TestConsent:
    def test_no_banner_when_all_none(self, capsys):
        result = consent.check_and_warn()
        assert result is None
        out = capsys.readouterr().out
        assert out.strip() == ""

    def test_banner_fires_on_first_cloud_provider(self, capsys, monkeypatch):
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "openai")
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o-mini")
        banner = consent.check_and_warn()
        assert banner is not None
        out = capsys.readouterr().out
        assert "Cloud provider change detected" in out
        assert "summaries" in out
        assert "gpt-4o-mini" in out

    def test_banner_does_not_fire_again_on_same_config(self, capsys, monkeypatch):
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "openai")
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o-mini")
        consent.check_and_warn()   # arm the ack file
        capsys.readouterr()         # clear the banner

        # Second call with unchanged config — should be silent
        result = consent.check_and_warn()
        assert result is None
        out = capsys.readouterr().out
        assert out.strip() == ""

    def test_banner_fires_again_when_provider_changes(self, capsys, monkeypatch):
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "openai")
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o-mini")
        consent.check_and_warn()
        capsys.readouterr()

        # User switches to Anthropic
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "anthropic")
        monkeypatch.setattr(config, "SUMMARIES_ANTHROPIC_MODEL", "claude-haiku-4-5")
        banner = consent.check_and_warn()
        assert banner is not None
        out = capsys.readouterr().out
        assert "anthropic" in out
        assert "claude-haiku-4-5" in out

    def test_banner_fires_again_when_model_upgrades(self, capsys, monkeypatch):
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "openai")
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o-mini")
        consent.check_and_warn()
        capsys.readouterr()

        # Same provider, bigger model
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o")
        banner = consent.check_and_warn()
        assert banner is not None

    def test_ack_file_has_correct_perms(self, monkeypatch):
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "openai")
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o-mini")
        consent.check_and_warn()
        p = os.path.join(config.CONFIG_DIR, "cloud_ack.json")
        assert os.path.exists(p)
        mode = os.stat(p).st_mode & 0o777
        assert mode == 0o600

    def test_banner_covers_multiple_features(self, capsys, monkeypatch):
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "openai")
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setattr(config, "RESEARCH_PROVIDER", "anthropic")
        monkeypatch.setattr(config, "RESEARCH_ANTHROPIC_MODEL", "claude-haiku-4-5")
        consent.check_and_warn()
        out = capsys.readouterr().out
        assert "summaries" in out
        assert "research" in out

    def test_ack_file_is_json(self, monkeypatch):
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "openai")
        monkeypatch.setattr(config, "SUMMARIES_OPENAI_MODEL", "gpt-4o-mini")
        consent.check_and_warn()
        p = os.path.join(config.CONFIG_DIR, "cloud_ack.json")
        data = json.loads(open(p).read())
        assert data["summaries"]["provider"] == "openai"
        assert data["summaries"]["model"] == "gpt-4o-mini"
