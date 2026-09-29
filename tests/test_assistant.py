"""Voice assistant: journal retrieval, LLM/TTS calls, persona hygiene,
interrupt handling. Network is stubbed at the http layer."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voiceclip import assistant, history


class Stub(BaseHTTPRequestHandler):
    seen = []
    reply = "You mentioned the Budget review twice this week."

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, body))
        if self.path.endswith("/chat/completions"):
            out = json.dumps({"choices": [{"message": {"content": Stub.reply}}]}).encode()
            ctype = "application/json"
        else:  # /audio/speech
            out = b"RIFF" + b"\x00" * 44
            ctype = "audio/wav"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def gateway(live_history, monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from voiceclip import config
    monkeypatch.setattr(config, "CLOUD_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "k")
    monkeypatch.setattr(config, "HISTORY_ENABLED", True)
    Stub.seen.clear()
    yield srv
    srv.shutdown()
    srv.server_close()


def _seed():
    history.save("budget review needs work", "Budget review needs work.", 1.0,
                 kind="transcription", app_name="Mail")
    history.save("submitted the GIS reviews", "Submitted the GIS reviews.", 1.0,
                 kind="transcription", app_name="Slack")


class TestJournalContext:
    def test_recent_and_fts_matches(self, gateway):
        _seed()
        ctx = assistant._journal_context("what did I say about budget payments")
        assert "Budget review" in ctx
        assert "Most recent journal entries" in ctx
        assert "matching the question" in ctx

    def test_empty_when_history_disabled(self, gateway, monkeypatch):
        from voiceclip import config
        monkeypatch.setattr(config, "HISTORY_ENABLED", False)
        assert assistant._journal_context("anything") == ""

    def test_stopwords_do_not_break_fts(self, gateway):
        _seed()
        # all stopwords -> no FTS query, but recent block still present
        ctx = assistant._journal_context("what did you say")
        assert "Most recent" in ctx


class TestLLM:
    def test_prompt_is_grounded_and_voice_shaped(self, gateway):
        _seed()
        reply = assistant.ask_llm("what about budget?", turns=[])
        assert reply == Stub.reply
        path, body = Stub.seen[-1]
        assert path.endswith("/v1/chat/completions")
        assert body["model"] == "assistant"
        system = body["messages"][0]["content"]
        assert "JOURNAL EXCERPTS" in system and "Budget review" in system
        assert "speaking out loud" in system          # voice persona
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_tags_and_markdown_stripped(self, gateway):
        Stub.reply = "<think>hmm</think>**Sure** — the *budget* one."
        try:
            assert assistant.ask_llm("q", turns=[]) == "Sure — the budget one."
        finally:
            Stub.reply = "You mentioned the Budget review twice this week."

    def test_turn_memory_is_sent_and_bounded(self, gateway):
        a = assistant.Assistant()
        a.player.play = lambda *args, **kw: None   # no afplay in tests
        for i in range(12):
            a.respond(f"question {i}")
        assert len(a.turns) == 2 * assistant._MAX_TURNS
        _, body = [x for x in Stub.seen if x[0].endswith("/chat/completions")][-1]
        # prior turns present between system and the new question
        roles = [m["role"] for m in body["messages"]]
        assert roles[0] == "system" and roles[-1] == "user" and "assistant" in roles

    def test_server_error_raises_assistant_error(self, gateway, monkeypatch):
        from voiceclip import config
        monkeypatch.setattr(config, "CLOUD_BASE_URL", "http://127.0.0.1:1")
        with pytest.raises(assistant.AssistantError):
            assistant.ask_llm("q", turns=[])


class TestPlayerAndInterrupt:
    def test_interrupt_when_idle_is_false(self):
        a = assistant.Assistant()
        assert a.interrupt() is False

    def test_speak_writes_wav(self, gateway):
        path = assistant.speak("hello")
        import os
        try:
            assert os.path.getsize(path) > 0
            _, body = Stub.seen[-1]
            assert body["input"] == "hello" and body["model"] == "tts"
        finally:
            os.unlink(path)


def test_history_accepts_assistant_kinds(live_history):
    history.save("q", "q", 0.0, kind="question")
    history.save("a", "a", 0.0, kind="assistant")
    assert history.count("question") == 1 and history.count("assistant") == 1
