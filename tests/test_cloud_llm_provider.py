"""The "cloud" LLM provider — the user's own gateway as a first-class
provider alongside local/openai/anthropic.

Transport tests run against a real localhost HTTP stub (same approach as
test_assistant.py). Feature-dispatch tests monkeypatch complete_cloud and
assert each feature routes through it with its own budget.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voiceclip import config, llm_provider


class Stub(BaseHTTPRequestHandler):
    seen = []
    reply = "cloud says hi"
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, dict(self.headers), body))
        out = json.dumps(
            {"choices": [{"message": {"content": Stub.reply}}]}
        ).encode()
        self.send_response(Stub.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def gateway(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "CLOUD_BASE_URL",
                        f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "sk-test")
    Stub.seen.clear()
    Stub.reply = "cloud says hi"
    Stub.status = 200
    yield srv
    srv.shutdown()
    srv.server_close()


class TestCompleteCloud:
    def test_message_assembly_and_auth(self, gateway):
        out = llm_provider.complete_cloud(
            system="sys", user="hello",
            turns=[{"role": "user", "content": "a"},
                   {"role": "assistant", "content": "b"}],
            model_id="assistant", max_tokens=222, temperature=0.5,
        )
        assert out == "cloud says hi"
        path, headers, body = Stub.seen[0]
        assert path == "/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-test"
        assert body["model"] == "assistant"
        assert body["max_tokens"] == 222
        assert body["temperature"] == 0.5
        # system first, turns in the middle, current question last
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["system", "user", "assistant", "user"]
        assert body["messages"][-1]["content"] == "hello"
        # Qwen thinking disabled at the template level
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_json_mode_sets_response_format(self, gateway):
        llm_provider.complete_cloud(
            system="s", user="u", json_mode=True,
        )
        body = Stub.seen[0][2]
        assert body["response_format"] == {"type": "json_object"}

    def test_no_optional_params_by_default(self, gateway):
        llm_provider.complete_cloud(system="s", user="u")
        body = Stub.seen[0][2]
        assert "temperature" not in body
        assert "response_format" not in body

    def test_leaked_think_blocks_are_stripped(self, gateway):
        Stub.reply = "<think>hmm</think>The answer."
        assert llm_provider.complete_cloud(system="s", user="u") == "The answer."

    def test_http_error_raises_runtime_error(self, gateway):
        Stub.status = 500
        with pytest.raises(RuntimeError, match="HTTP 500"):
            llm_provider.complete_cloud(system="s", user="u")

    def test_unreachable_gateway_is_actionable(self, monkeypatch):
        monkeypatch.setattr(config, "CLOUD_BASE_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(config, "CLOUD_API_KEY", "k")
        with pytest.raises(RuntimeError, match="voiceclip cloud status"):
            llm_provider.complete_cloud(system="s", user="u", timeout=1.0)

    def test_missing_base_url_is_actionable(self, monkeypatch):
        monkeypatch.setattr(config, "CLOUD_BASE_URL", "")
        with pytest.raises(RuntimeError, match="cloud.base_url"):
            llm_provider.complete_cloud(system="s", user="u")


class TestConfigCloudProvider:
    def test_cloud_is_valid_provider(self, isolated_config):
        import os
        with open(config.CONFIG_PATH, "w") as f:
            json.dump({"summaries": {"provider": "cloud"}}, f)
        config.load()
        assert config.SUMMARIES_PROVIDER == "cloud"
        assert config.SUMMARIES_CLOUD_MODEL == "assistant"  # default
        assert config.model_id_for("summaries") == "assistant"

    def test_cloud_model_configurable(self, isolated_config):
        with open(config.CONFIG_PATH, "w") as f:
            json.dump({"patterns": {"provider": "cloud",
                                    "cloud_model": "assistant-v2"}}, f)
        config.load()
        assert config.model_id_for("patterns") == "assistant-v2"

    def test_env_override(self, isolated_config, monkeypatch):
        monkeypatch.setenv("VOICECLIP_RESEARCH_CLOUD_MODEL", "from-env")
        config.load()
        assert config.RESEARCH_CLOUD_MODEL == "from-env"


class TestFeatureDispatch:
    """Each feature routes provider='cloud' through complete_cloud."""

    @pytest.fixture
    def spy(self, monkeypatch):
        calls = []

        def fake_complete_cloud(**kw):
            calls.append(kw)
            return '{"themes": []}'  # patterns wants JSON; others don't care

        monkeypatch.setattr(llm_provider, "complete_cloud", fake_complete_cloud)
        return calls

    def test_polisher(self, spy, monkeypatch):
        from voiceclip import polisher
        monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "cloud")
        monkeypatch.setattr(config, "SUMMARIES_CLOUD_MODEL", "assistant")
        spy_result = polisher.polish("raw dictation")
        assert spy and spy[0]["model_id"] == "assistant"
        assert spy_result == '{"themes": []}'

    def test_summarizer_run(self, spy):
        from voiceclip import summarizer
        summarizer._run("cloud", "sys", "user", "assistant")
        assert spy[0]["max_tokens"] == 800

    def test_patterns_run_uses_json_mode(self, spy):
        from voiceclip import patterns
        patterns._run("cloud", "sys", "user", "assistant")
        assert spy[0]["json_mode"] is True

    def test_researcher_run_no_sources(self, spy):
        from voiceclip import researcher
        text, sources, used_web = researcher._run("cloud", "topic", "assistant")
        assert text == '{"themes": []}'
        assert sources == [] and used_web is False
