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
  keep_warm_ping(model_id)    → silence inference to keep weights resident
"""

import logging
import os
import subprocess
import threading
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

# How long to wait for whisper-server to become ready at startup (load()).
# Normal startup is 1-3s, but a Core ML encoder's first-ever load triggers
# an ANE compile that can take minutes. Giving up too early silently
# degrades every dictation to CLI mode, which reloads the model per call —
# much worse than a slow one-time startup.
SERVER_START_TIMEOUT = int(os.environ.get("VOICECLIP_WHISPER_CPP_START_TIMEOUT", "180"))

# Budget for automatic mid-session restarts. These happen inside
# transcribe() / keep_warm_ping(), whose caller (transcriber.py) abandons
# the worker thread after its own 60s timeout — so this must stay well
# under that. A warm restart takes 2-3s; the minutes-long ANE compile only
# happens once per machine and is paid at load() time.
SERVER_RESTART_TIMEOUT = int(os.environ.get("VOICECLIP_WHISPER_CPP_RESTART_TIMEOUT", "20"))

# Minimum seconds between automatic restart attempts, so a server that
# keeps crashing can't stall every dictation with a doomed restart.
_RESTART_COOLDOWN = 60

# Map config model names → GGML filenames
_MODEL_FILES = {
    "large-v3": "ggml-large-v3.bin",
    "large-v3-q5": "ggml-large-v3-q5_0.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
    "large-v3-turbo-q5": "ggml-large-v3-turbo-q5_0.bin",
    "medium": "ggml-medium-q5_0.bin",
    "medium.en": "ggml-medium.en-q5_0.bin",
    "small": "ggml-small-q5_0.bin",
    "small.en": "ggml-small.en-q5_0.bin",
    "base": "ggml-base-q5_0.bin",
    "base.en": "ggml-base.en-q5_0.bin",
    "tiny": "ggml-tiny-q5_0.bin",
    "tiny.en": "ggml-tiny.en-q5_0.bin",
}

# Server process state. All mutation of _server_proc/_server_ready during
# recovery or restart happens under _lifecycle_lock: transcriber.py's
# engine lock is released when a transcription worker is abandoned on
# timeout, so two threads *can* reach the lifecycle code concurrently —
# without the lock they could double-spawn servers or kill each other's.
_server_proc = None
_server_ready = False
_last_restart_attempt = 0.0  # rate-limits automatic server restarts
_lifecycle_lock = threading.Lock()


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

def _start_server(model_path: str, timeout: float | None = None) -> bool:
    """Start whisper-server as a background process. Returns True on success.

    `timeout` bounds the wait for readiness; defaults to
    SERVER_START_TIMEOUT (generous, for startup). Mid-session restarts
    pass SERVER_RESTART_TIMEOUT instead.
    """
    global _server_proc, _server_ready

    if timeout is None:
        timeout = SERVER_START_TIMEOUT

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
        "-bs", "5",  # beam search — whisper.cpp defaults to greedy (-1)
        "-l", "en" if config.ENGLISH_ONLY else "auto",
        # No --convert: the server decodes uploads in-process via miniaudio,
        # which resamples any WAV rate natively. Skipping the per-request
        # ffmpeg subprocess + temp-file round trip saves 60-90ms per
        # dictation. Requires whisper.cpp >= v1.9.0 — older servers had a
        # broken in-memory decode path and need --convert to work at all.
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

    # Wait for the server to become ready, as long as the process stays
    # alive. Keep polling well past the normal 1-3s startup — see
    # SERVER_START_TIMEOUT for why.
    started = time.time()
    slow_notice_shown = False
    while time.time() - started < timeout:
        time.sleep(0.5)
        if _server_proc.poll() is not None:
            log.error("whisper-server exited early (rc=%d)", _server_proc.returncode)
            _server_proc = None
            return False
        if _server_health_check():
            elapsed = time.time() - started
            _server_ready = True
            log.info("whisper-server ready on port %d (pid=%d) after %.1fs",
                     WHISPER_CPP_PORT, _server_proc.pid, elapsed)
            return True
        if not slow_notice_shown and time.time() - started > 15:
            slow_notice_shown = True
            print(
                "  ⏳ Speech model still loading — a first run with a new "
                "Core ML encoder can take a few minutes (one-time compile)...",
                flush=True,
            )

    log.error("whisper-server did not become ready in %ds", timeout)
    _kill_server()
    return False


def _ensure_server(model_id: str, timeout: float) -> None:
    """Recover or restart the server if it isn't serving. Thread-safe.

    Two repair paths, both under _lifecycle_lock:
      - Process alive but _server_ready tripped by a transient request
        failure (including keep-warm pings): restore server mode after a
        health check.
      - Process dead: restart it, rate-limited to one attempt per
        _RESTART_COOLDOWN. A restart (~2-3s) beats the CLI fallback,
        which reloads the model on every call (~40s with a Core ML
        encoder).
    """
    global _server_ready, _last_restart_attempt

    with _lifecycle_lock:
        if _server_ready and _is_server_alive():
            return

        if _is_server_alive():
            if _server_health_check():
                log.info("whisper-server recovered; resuming server mode")
                _server_ready = True
            return

        # Process is gone — the ready flag is meaningless now.
        _server_ready = False

        if time.time() - _last_restart_attempt <= _RESTART_COOLDOWN:
            return
        _last_restart_attempt = time.time()
        log.warning("whisper-server not running; attempting restart")
        try:
            _start_server(_resolve_model_path(model_id), timeout=timeout)
        except Exception as e:
            log.error("Server restart failed: %s", e)


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

    if not _start_server(model_path, timeout=SERVER_START_TIMEOUT):
        # Fall back to verifying CLI mode works
        if not os.path.isfile(WHISPER_CPP_BIN):
            raise FileNotFoundError(
                f"Neither whisper-server nor whisper-cli available. "
                f"Build whisper.cpp first."
            )
        log.warning("Server mode failed, will use CLI fallback")


def transcribe(audio_path: str, model_id: str) -> str | None:
    """Transcribe via server (preferred) or CLI fallback.

    Server mode is sticky in both directions:
      - A transient request failure (including a failed keep-warm ping)
        trips _server_ready, but if the server process is still alive and
        healthy we restore server mode instead of paying the CLI's
        per-call model load forever.
      - When a request fails mid-flight, that dictation is retried through
        the CLI rather than being silently dropped as "no speech".
    """
    _ensure_server(model_id, timeout=SERVER_RESTART_TIMEOUT)

    if _server_ready and _is_server_alive():
        text = _transcribe_server(audio_path)
        if _server_ready:
            return text
        # _transcribe_server tripped the flag — the request failed, not
        # the audio. Fall through and retry this dictation via CLI.
        log.warning("Server request failed; retrying this dictation via CLI")
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

        # whisper-server returns segment breaks as newlines — join into
        # a single line since our formatter handles paragraph structure.
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
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
        "-bs", "5",  # beam search — whisper.cpp defaults to greedy (-1)
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

    # CLI output also has segment newlines — join into single line
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return text


def keep_warm_ping(model_id: str):
    """Run a tiny silence inference through the server to keep weights resident.

    The server holds the model in its process, but macOS pages those
    gigabytes out after idle — the first transcription after a break then
    stalls on faulting them back in. A periodic inference touches the
    weights and keeps them warm.

    The ping runs off the user's critical path, which also makes it the
    ideal place to bring a dead server back — so the next dictation
    doesn't pay the restart cost. If the server can't be brought up
    (restart failed or rate-limited), the ping is skipped: pinging via
    whisper-cli would cold-load the model from disk every interval,
    which is worse than the problem it solves.
    """
    _ensure_server(model_id, timeout=SERVER_RESTART_TIMEOUT)
    if not (_server_ready and _is_server_alive()):
        return

    import tempfile
    import wave

    from voiceclip.config import TEMP_PREFIX
    from voiceclip.utils import safe_unlink

    tmp = tempfile.NamedTemporaryFile(
        prefix=TEMP_PREFIX, suffix=".wav", delete=False
    )
    try:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 8000)  # 0.5s of silence
        tmp.close()
        _transcribe_server(tmp.name)
    finally:
        safe_unlink(tmp.name)


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
