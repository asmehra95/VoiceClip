"""tunnel.py and cloud_control.py tests using a fake `aws` executable."""
import json
import os
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voiceclip import cloud_control, tunnel


@pytest.fixture
def fake_aws(tmp_path, monkeypatch):
    """Put a scriptable `aws` on PATH; behavior driven by AWS_FAKE_MODE."""
    script = tmp_path / "aws"
    script.write_text(
        "#!/bin/bash\n"
        'case "$AWS_FAKE_MODE" in\n'
        "  status) echo '{\"instance_id\":\"i-1\",\"region\":\"r\","
        "\"state\":\"running\",\"instance_type\":\"g6.2xlarge\"}';;\n"
        "  describe) echo '{\"State\":{\"Name\":\"running\"},"
        "\"InstanceType\":\"g6.2xlarge\"}';;\n"
        "  stop) echo '{\"StoppingInstances\":[{\"CurrentState\":"
        "{\"Name\":\"stopping\"}}]}';;\n"
        "  start) echo '{\"StartingInstances\":[{\"CurrentState\":"
        "{\"Name\":\"pending\"}}]}';;\n"
        "  fail) echo 'An error occurred (AuthFailure)' >&2; exit 254;;\n"
        "  badjson) echo 'not json';;\n"
        "  tunnel) sleep 30;;\n"
        "esac\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    return script


@pytest.fixture
def cloud_cfg(isolated_config, monkeypatch):
    from voiceclip import config
    config.load()
    monkeypatch.setattr(config, "ENGINE", "cloud")
    monkeypatch.setattr(config, "CLOUD_INSTANCE_ID", "i-1")
    monkeypatch.setattr(config, "CLOUD_REGION", "eu-central-1")
    monkeypatch.setattr(config, "CLOUD_AUTO_TUNNEL", True)
    return config


class TestCloudControl:
    def test_status(self, fake_aws, cloud_cfg, monkeypatch, capsys):
        monkeypatch.setenv("AWS_FAKE_MODE", "describe")
        assert cloud_control.run("status") == 0
        assert "running" in capsys.readouterr().out

    def test_stop(self, fake_aws, cloud_cfg, monkeypatch, capsys):
        monkeypatch.setenv("AWS_FAKE_MODE", "stop")
        assert cloud_control.run("stop") == 0
        assert "stopping" in capsys.readouterr().out

    def test_start(self, fake_aws, cloud_cfg, monkeypatch, capsys):
        monkeypatch.setenv("AWS_FAKE_MODE", "start")
        assert cloud_control.run("start") == 0
        assert "pending" in capsys.readouterr().out

    def test_aws_error_reported(self, fake_aws, cloud_cfg, monkeypatch, capsys):
        monkeypatch.setenv("AWS_FAKE_MODE", "fail")
        assert cloud_control.run("status") == 1
        assert "AuthFailure" in capsys.readouterr().out

    def test_missing_config(self, fake_aws, cloud_cfg, monkeypatch):
        monkeypatch.setattr(cloud_cfg, "CLOUD_INSTANCE_ID", "")
        assert cloud_control.run("status") == 1


class TestTunnel:
    def _gateway(self):
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")
            def log_message(self, *a):
                pass
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def test_ensure_noop_when_gateway_alive(self, fake_aws, cloud_cfg,
                                            monkeypatch):
        srv = self._gateway()
        try:
            monkeypatch.setattr(
                cloud_cfg, "CLOUD_BASE_URL",
                f"http://localhost:{srv.server_address[1]}")
            monkeypatch.setenv("AWS_FAKE_MODE", "tunnel")
            assert tunnel.ensure() is True
            assert tunnel._proc is None   # nothing spawned
        finally:
            srv.shutdown(); srv.server_close()

    def test_ensure_false_without_instance_id(self, fake_aws, cloud_cfg,
                                              monkeypatch):
        monkeypatch.setattr(cloud_cfg, "CLOUD_BASE_URL",
                            "http://localhost:1")
        monkeypatch.setattr(cloud_cfg, "CLOUD_INSTANCE_ID", "")
        assert tunnel.ensure() is False

    def test_ensure_false_for_remote_base_url(self, fake_aws, cloud_cfg,
                                              monkeypatch):
        monkeypatch.setattr(cloud_cfg, "CLOUD_BASE_URL",
                            "https://transcribe.example.com")
        assert tunnel.ensure() is False

    def test_shutdown_kills_process_group(self, fake_aws, cloud_cfg,
                                          monkeypatch):
        """The session's child (session-manager-plugin) holds the port;
        shutdown must kill the whole group, not just `aws`."""
        monkeypatch.setattr(cloud_cfg, "CLOUD_BASE_URL",
                            "http://localhost:1")
        proc = tunnel._spawn("i-1", "r", 1)
        with tunnel._lock:
            tunnel._proc = proc
            tunnel._running = True
        assert proc.poll() is None
        tunnel.shutdown()
        deadline = time.time() + 3
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert proc.poll() is not None, "tunnel process survived shutdown"
