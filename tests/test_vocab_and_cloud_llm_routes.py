"""Vocabulary tab endpoints and the cloud-LLM switcher endpoint."""

import json

import pytest

from tests.conftest import http_get, http_post
from voiceclip import cloud_control, config, history, vocab_suggest


@pytest.fixture(autouse=True)
def small_dictionary(monkeypatch):
    monkeypatch.setattr(vocab_suggest, "_words_cache",
                        frozenset(["we", "met", "with", "and", "the", "file", "today", "send"]))


class TestVocab:
    def _seed(self):
        history.save("x", "We met with Priya and the SLA team today.", 1.0, kind="transcription")
        history.save("x", "Send the SLA file to Priya.", 1.0, kind="transcription")

    def test_suggestions_and_add(self, live_viewer):
        self._seed()
        d = http_get(f"{live_viewer}/api/vocab")
        terms = [s["term"] for s in d["suggestions"]]
        assert "Priya" in terms and "SLA" in terms
        status, _ = http_post(f"{live_viewer}/api/settings/update",
                              {"custom_vocabulary": ["Priya"]})
        assert status == 200
        d = http_get(f"{live_viewer}/api/vocab")
        assert d["terms"] == ["Priya"]
        assert "Priya" not in [s["term"] for s in d["suggestions"]]

    def test_ignore_and_unignore(self, live_viewer):
        self._seed()
        status, _ = http_post(f"{live_viewer}/api/vocab/ignore", {"term": "SLA"})
        assert status == 200
        d = http_get(f"{live_viewer}/api/vocab")
        assert "SLA" not in [s["term"] for s in d["suggestions"]]
        assert d["ignored"] == ["SLA"]
        http_post(f"{live_viewer}/api/vocab/unignore", {"term": "sla"})
        assert http_get(f"{live_viewer}/api/vocab")["ignored"] == []

    @pytest.mark.parametrize("bad", ["", "x" * 81])
    def test_ignore_validates(self, live_viewer, bad):
        status, _ = http_post(f"{live_viewer}/api/vocab/ignore", {"term": bad})
        assert status == 400


class TestCloudLLM:
    def test_choices_and_default(self, live_viewer):
        d = http_get(f"{live_viewer}/api/cloud/llm")
        assert d["current"] == cloud_control.DEFAULT_CLOUD_LLM
        assert any(c["id"] == d["current"] for c in d["choices"])

    @pytest.mark.parametrize("bad", ["x/y; rm -rf /", "a/$(id)", "noslash", "a/b/c", ""])
    def test_rejects_non_repo_ids(self, live_viewer, bad):
        status, _ = http_post(f"{live_viewer}/api/cloud/llm", {"model": bad})
        assert status == 400

    def test_apply_persists_choice(self, live_viewer, monkeypatch):
        sent = {}
        monkeypatch.setattr(config, "CLOUD_INSTANCE_ID", "i-123")
        monkeypatch.setattr(config, "CLOUD_REGION", "eu-central-1")
        monkeypatch.setattr(cloud_control, "set_llm_model",
                            lambda m: sent.update(model=m) or {"command_id": "c-1"})
        status, body = http_post(f"{live_viewer}/api/cloud/llm",
                                 {"model": "RedHatAI/Qwen3.5-4B-FP8-dynamic"})
        assert status == 200 and body["command_id"] == "c-1"
        assert sent["model"] == "RedHatAI/Qwen3.5-4B-FP8-dynamic"
        with open(config.CONFIG_PATH) as f:
            assert json.load(f)["cloud"]["llm_model"] == "RedHatAI/Qwen3.5-4B-FP8-dynamic"

    def test_set_llm_model_script_is_quoted_safely(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cloud_control, "_settings", lambda: ("i-1", "eu-central-1"))
        monkeypatch.setattr(cloud_control, "_aws",
                            lambda args, region: calls.append(args) or {"Command": {"CommandId": "c"}})
        cloud_control.set_llm_model("RedHatAI/Qwen3.5-4B-FP8-dynamic")
        params = json.loads(calls[0][calls[0].index("--parameters") + 1])
        script = params["commands"][0]
        assert "LLM_MODEL=RedHatAI/Qwen3.5-4B-FP8-dynamic" in script
        assert "docker compose --profile gpu up -d llm" in script
        with pytest.raises(cloud_control.CloudControlError):
            cloud_control.set_llm_model("a/b'; reboot; '")
