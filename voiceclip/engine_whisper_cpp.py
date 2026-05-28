"""whisper.cpp engine — runs whisper-cli binary with Metal GPU acceleration.

Supports two modes:
  - CLI mode: spawns whisper-cli per transcription (stateless, no memory use)
  - Server mode: starts whisper-server once, sends audio via HTTP (fast, model
    stays loaded in GPU memory)

Server mode is preferred — eliminates model load time per transcription.
Falls back to CLI mode if the server can't start or dies mid-session.

Exposes the same interface as engine_whisper:
  load(model_id)              → start server (or verify CLI binary exists)
  transcribe(path, model_id)  → raw text or None
  keep_warm_ping(model_id)    → no-op
"""

import logging
import os
import subprocess
import time
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

# Paths — configurable via env vars for flexibility
WHISPER_CPP_BIN = os.environ.get(
    "VOICECLIP_WHISPER_CPP_BIN",
    os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"),
)
WHISPER_CPP_SERVER_BIN = os.environ.get(
    "VOICECLIP_WHISPER_CPP_SERVER",
    os.path.expanduser("~/whisper.cpp/build/bin/whisper-server"),
)
WHISPER_CPP_MODELS_DIR = os.environ.get(
    "VOICECLIP_WHISPER_CPP_MODELS",
    os.path.expanduser("~/.voiceclip/models"),
)
WHISPER_CPP_PORT = int(os.environ.get("VOICECLIP_WHISPER_CPP_PORT", "8178"))

# Map config model names → GGML filenames
_MODEL_FILES = {
    "large-v3": "ggml-large-v3.bin",
    "large-v3-q5": "ggml-large-v3-q5_0.bin",
    "large-v3-turbo": "ggml-large-v3-turbo-q5_0.bin",
    "medium": "ggml-medium-q5_0.bin",
    "medium.en": "ggml-medium.en-q5_0.bin",
    "small": "ggml-small-q5_0.bin",
    "small.en": "ggml-small.en-q5_0.bin",
    "base": "ggml-base-q5_0.bin",
    "base.en": "ggml-base.en-q5_0.bin",
    "tiny": "ggml-tiny-q5_0.bin",
    "tiny.en": "ggml-tiny.en-q5_0.bin",
}

# Server process state
_server_proc = None
_server_ready = False


def _resolve_model_path(model_id: str) -> str:
    """Resolve a model identifier to a GGML file path."""
    if os.path.isfile(model_id):
        return model_id
    if model_id in _MODEL_FILES:
        path = os.path.join(WHISPER_CPP_MODELS_DIR, _MODEL_FILES[model_id])
        if os.path.isfile(path):
            return path
    path = os.path.join(WHISPER_CPP_MODELS_DIR, model_id)
    if os.path.isfile(path):
        return path
    path = os.path.join(WHISPER_CPP_MODELS_DIR, f"ggml-{model_id}.bin")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"whisper.cpp model not found for '{model_id}'. "
        f"Expected at {WHISPER_CPP_MODELS_DIR}."
    )


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def _start_server(model_path: str) -> bool:
    """Start whisper-server as a background process. Returns True on success."""
    global _server_proc, _server_ready

    if not os.path.isfile(WHISPER_CPP_SERVER_BIN):
        log.warning("whisper-server binary not found at %s", WHISPER_CPP_SERVER_BIN)
        return False

    from voiceclip import config

    cmd = [
        WHISPER_CPP_SERVER_BIN,
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(WHISPER_CPP_PORT),
        "-t", "6",
        "-l", "en" if config.ENGLISH_ONLY else "auto",
        "--convert",  # use ffmpeg to convert incoming audio to proper format
    ]
    if config.INITIAL_PROMPT:
        cmd.extend(["--prompt", config.INITIAL_PROMPT])

    try:
        _server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log.error("Failed to start whisper-server: %s", e)
        return False

    # Wait for server to be ready (poll /health or /inference endpoint)
    for i in range(30):  # up to 15 seconds
        time.sleep(0.5)
        if _server_proc.poll() is not None:
            log.error("whisper-server exited early (rc=%d)", _server_proc.returncode)
            _server_proc = None
            return False
        if _server_health_check():
            _server_ready = True
            log.info("whisper-server ready on port %d (pid=%d)", WHISPER_CPP_PORT, _server_proc.pid)
            return True

    log.error("whisper-server did not become ready in 15s")
    _kill_server()
    return False


def _server_health_check() -> bool:
    """Check if the server is responding."""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{WHISPER_CPP_PORT}/",
            method="GET",
        )
        resp = urllib.request.urlopen(req, timeout=1)
        return resp.status == 200
    except Exception:
        return False


def _kill_server():
    """Terminate the server process."""
    global _server_proc, _server_ready
    _server_ready = False
    if _server_proc is not None:
        try:
            _server_proc.terminate()
            _server_proc.wait(timeout=5)
        except Exception:
            try:
                _server_proc.kill()
            except Exception:
                pass
        _server_proc = None


def _is_server_alive() -> bool:
    """Check if our server process is still running."""
    if _server_proc is None:
        return False
    return _server_proc.poll() is None


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------

def load(model_id: str):
    """Start the whisper-server with the given model."""
    model_path = _resolve_model_path(model_id)

    # Kill existing server if running with a different model
    if _server_proc is not None:
        _kill_server()

    if not _start_server(model_path):
        # Fall back to verifying CLI mode works
        if not os.path.isfile(WHISPER_CPP_BIN):
            raise FileNotFoundError(
                f"Neither whisper-server nor whisper-cli available. "
                f"Build whisper.cpp first."
            )
        log.warning("Server mode failed, will use CLI fallback")


def transcribe(audio_path: str, model_id: str) -> str | None:
    """Transcribe via server (preferred) or CLI fallback."""
    if _server_ready and _is_server_alive():
        return _transcribe_server(audio_path)
    else:
        return _transcribe_cli(audio_path, model_id)


def _transcribe_server(audio_path: str) -> str | None:
    """Send audio to the whisper-server via HTTP POST."""
    import json
    import http.client
    from email.mime.multipart import MIMEMultipart
    import mimetypes
    import uuid

    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Build proper multipart/form-data manually
        boundary = uuid.uuid4().hex
        filename = os.path.basename(audio_path)

        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: audio/wav\r\n"
            f"\r\n"
        ).encode() + audio_data + (
            f"\r\n"
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"response_format\"\r\n"
            f"\r\n"
            f"json\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        conn = http.client.HTTPConnection("127.0.0.1", WHISPER_CPP_PORT, timeout=60)
        conn.request(
            "POST",
            "/inference",
            body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        resp_data = resp.read().decode("utf-8")
        conn.close()

        if resp.status != 200:
            log.warning("Server returned %d: %s", resp.status, resp_data[:200])
            global _server_ready
            _server_ready = False
            return None

        result = json.loads(resp_data)
        text = result.get("text", "").strip()
        if not text or text == "[BLANK_AUDIO]":
            return None
        return text

    except Exception as e:
        log.warning("Server transcription error: %s, falling back to CLI", e)
        _server_ready = False
        return None


def _transcribe_cli(audio_path: str, model_id: str) -> str | None:
    """Fallback: run whisper-cli directly."""
    from voiceclip import config

    model_path = _resolve_model_path(model_id)

    cmd = [
        WHISPER_CPP_BIN,
        "-m", model_path,
        "-f", audio_path,
        "--no-timestamps",
        "--no-prints",
        "-t", "6",
    ]

    if config.ENGLISH_ONLY:
        cmd.extend(["-l", "en"])
    if config.INITIAL_PROMPT:
        cmd.extend(["--prompt", config.INITIAL_PROMPT])

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=60, text=True,
        )
    except subprocess.TimeoutExpired:
        log.error("whisper.cpp CLI timed out after 60s")
        return None
    except FileNotFoundError:
        log.error("whisper-cli not found at %s", WHISPER_CPP_BIN)
        return None

    if result.returncode != 0:
        log.error("whisper.cpp CLI failed (rc=%d)", result.returncode)
        return None

    text = result.stdout.strip()
    if not text or text == "[BLANK_AUDIO]":
        return None
    return text


def keep_warm_ping(model_id: str):
    """No-op — server keeps the model warm; CLI uses page cache."""
    pass


def shutdown():
    """Clean shutdown — kill the server. Called on VoiceClip exit."""
    _kill_server()


def model_label(model_id: str) -> str:
    """Return a descriptive label for analytics (includes quantization info)."""
    try:
        path = _resolve_model_path(model_id)
    except FileNotFoundError:
        return model_id

    filename = os.path.basename(path)
    name = filename.replace("ggml-", "").replace(".bin", "")
    if "q" not in name.split("-")[-1]:
        name = f"{name}-fp16"
    return name
