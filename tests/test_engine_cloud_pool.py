"""Connection-reuse tests for the batch cloud engine.

A counting HTTP server verifies keep-alive semantics; failure injection
verifies the stale-retry and tunnel-re-ensure paths.
"""
import json
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voiceclip import engine_cloud


class Stub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    force_close = False

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.dumps({"text": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if Stub.force_close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


class CountingServer(ThreadingHTTPServer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.conn_count = 0

    def get_request(self):
        req = super().get_request()
        self.conn_count += 1
        return req


def make_wav():
    t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(t, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    t.close()
    return t.name


@pytest.fixture
def server(isolated_config, monkeypatch):
    srv = CountingServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from voiceclip import config
    config.load()
    monkeypatch.setattr(config, "ENGINE", "cloud")
    monkeypatch.setattr(config, "CLOUD_BASE_URL",
                        f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "k")
    # reset the module-level pool between tests
    engine_cloud._pooled_conn = None
    engine_cloud._pooled_sig = None
    Stub.force_close = False
    yield srv
    srv.shutdown()
    srv.server_close()


def test_three_requests_one_connection(server):
    for _ in range(3):
        assert engine_cloud.transcribe(make_wav(), "m") == "ok"
    assert server.conn_count == 1


def test_connection_close_not_pooled(server):
    Stub.force_close = True
    for _ in range(2):
        assert engine_cloud.transcribe(make_wav(), "m") == "ok"
    assert engine_cloud._pooled_conn is None


def test_stale_pooled_connection_retried(server, monkeypatch):
    assert engine_cloud.transcribe(make_wav(), "m") == "ok"   # pools conn
    port = server.server_address[1]
    server.shutdown(); server.server_close()                  # kill server
    srv2 = CountingServer(("127.0.0.1", port), Stub)          # same port
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    try:
        assert engine_cloud.transcribe(make_wav(), "m") == "ok"
    finally:
        srv2.shutdown(); srv2.server_close()


def test_unreachable_raises_after_tunnel_attempt(server, monkeypatch):
    from voiceclip import config, tunnel
    monkeypatch.setattr(config, "CLOUD_BASE_URL", "http://127.0.0.1:1")
    engine_cloud._pooled_conn = None
    calls = []
    monkeypatch.setattr(tunnel, "ensure", lambda: calls.append(1) and False)
    with pytest.raises(engine_cloud.CloudEngineError):
        engine_cloud.transcribe(make_wav(), "m")
    assert calls, "tunnel.ensure() was not attempted"


def test_config_change_invalidates_pool(server, monkeypatch):
    from voiceclip import config
    assert engine_cloud.transcribe(make_wav(), "m") == "ok"
    assert engine_cloud._pooled_conn is not None
    # New endpoint -> pooled conn must not be reused for it
    srv2 = CountingServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "CLOUD_BASE_URL",
                        f"http://127.0.0.1:{srv2.server_address[1]}")
    try:
        assert engine_cloud.transcribe(make_wav(), "m") == "ok"
        assert srv2.conn_count == 1
    finally:
        srv2.shutdown(); srv2.server_close()
