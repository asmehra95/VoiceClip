"""Cloud engine — OpenAI-compatible transcription over HTTP(S).

Sends the recorded WAV to a remote `/v1/audio/transcriptions` endpoint.
Works against any OpenAI-compatible server: vLLM, speaches, LiteLLM,
whisper.cpp server in OAI mode, or OpenAI itself.

Configuration (config.json "cloud" block, env vars win):
  base_url  VOICECLIP_CLOUD_BASE_URL   e.g. "https://stt.example.com"
  api_key   VOICECLIP_CLOUD_API_KEY    bearer token, empty for open servers
  model     VOICECLIP_CLOUD_MODEL      server-side model name

Exposes the same interface as the local engines:
  load(model_id)              → reachability check (warns, never blocks startup)
  transcribe(path, model_id)  → raw text or None (raises on network failure)
  keep_warm_ping(model_id)    → no-op (the server keeps its own model warm)

Connections are reused across dictations (HTTP keep-alive): TCP + TLS
setup costs 1-2 round trips, which against a remote region is a large
slice of total latency. A stale kept-alive connection (server or tunnel
dropped it between dictations) is detected and retried once on a fresh
connection before any error is surfaced.
"""

import http.client
import json
import logging
import os
import threading
import urllib.parse
import uuid

log = logging.getLogger(__name__)

# Read timeout for a single transcription request. Kept under
# transcriber.py's _TIMEOUT["cloud"] so the engine fails with a useful
# error before the dispatcher abandons the worker thread.
REQUEST_TIMEOUT = int(os.environ.get("VOICECLIP_CLOUD_TIMEOUT", "45"))

TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"


class CloudEngineError(RuntimeError):
    """Raised when the cloud endpoint is misconfigured or unreachable."""


def _endpoint():
    """Parse the configured base URL into (scheme, host, port, base_path).

    Raises CloudEngineError when the URL is missing or malformed so the
    dispatcher surfaces a clear message instead of a bare socket error.
    """
    from voiceclip import config

    base_url = (config.CLOUD_BASE_URL or "").strip().rstrip("/")
    if not base_url:
        raise CloudEngineError(
            "Cloud engine selected but cloud.base_url is not set. "
            "Add it to ~/.voiceclip/config.json or set "
            "VOICECLIP_CLOUD_BASE_URL."
        )

    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise CloudEngineError(
            f"cloud.base_url must be an http(s) URL, got: {base_url!r}"
        )

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname, port, parsed.path or ""


def _connect(scheme: str, host: str, port: int, timeout: float):
    from voiceclip import config

    if scheme == "https":
        # Default context verifies certificates and hostname. An optional
        # CA bundle (cloud.ca_bundle / VOICECLIP_CLOUD_CA_BUNDLE) supports
        # private CAs — e.g. Caddy's internal CA on a localhost dev stack —
        # without weakening verification.
        import ssl

        ca_bundle = (getattr(config, "CLOUD_CA_BUNDLE", "") or "").strip()
        context = (
            ssl.create_default_context(cafile=os.path.expanduser(ca_bundle))
            if ca_bundle else ssl.create_default_context()
        )
        return http.client.HTTPSConnection(
            host, port, timeout=timeout, context=context,
        )
    return http.client.HTTPConnection(host, port, timeout=timeout)


# ---------------------------------------------------------------------------
# Connection reuse
#
# A single kept-alive connection is cached between dictations. Checkout /
# checkin semantics rather than a lock held across the request: if a second
# thread needs a connection while one is checked out (e.g. the dispatcher
# abandoned a slow worker and started another), it simply opens a fresh one
# — nobody ever blocks on the pool.
# ---------------------------------------------------------------------------

_pool_lock = threading.Lock()
_pooled_conn = None       # the idle kept-alive connection, if any
_pooled_sig = None        # (scheme, host, port, ca_bundle) it was built for


def _conn_signature():
    from voiceclip import config

    scheme, host, port, _ = _endpoint()
    ca = (getattr(config, "CLOUD_CA_BUNDLE", "") or "").strip()
    return (scheme, host, port, ca)


def _checkout_conn(timeout: float):
    """Return (connection, reused_flag). Fresh if none pooled or config
    changed since the pooled one was opened."""
    global _pooled_conn, _pooled_sig

    sig = _conn_signature()
    with _pool_lock:
        conn, pooled_sig = _pooled_conn, _pooled_sig
        _pooled_conn = None
    if conn is not None and pooled_sig == sig:
        # Refresh the socket timeout for this request.
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        return conn, True
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    scheme, host, port, _ = sig[0], sig[1], sig[2], sig[3]
    return _connect(scheme, host, port, timeout=timeout), False


def _checkin_conn(conn):
    """Return a healthy connection to the pool (or close it if the slot
    is already occupied by a newer one)."""
    global _pooled_conn, _pooled_sig

    with _pool_lock:
        if _pooled_conn is None:
            _pooled_conn = conn
            _pooled_sig = _conn_signature()
            return
    try:
        conn.close()
    except Exception:
        pass


def _discard_conn(conn):
    try:
        conn.close()
    except Exception:
        pass


def _auth_headers() -> dict:
    from voiceclip import config

    key = (config.CLOUD_API_KEY or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _warn_if_plaintext(scheme: str, host: str):
    """Dictated audio is sensitive; flag unencrypted non-local transport."""
    if scheme == "http" and host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "cloud.base_url uses plain http to a non-local host — audio "
            "and API key travel unencrypted. Use https in production."
        )


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------

def load(model_id: str):
    """Best-effort reachability check at startup.

    Never raises on an unreachable server: the server may come up later,
    and preload_model() treats exceptions as a soft warning anyway. Does
    raise on outright misconfiguration (missing/malformed URL) so the
    user finds out at startup, not mid-dictation.
    """
    scheme, host, port, base_path = _endpoint()
    _warn_if_plaintext(scheme, host)

    try:
        conn = _connect(scheme, host, port, timeout=5)
        try:
            conn.request("GET", (base_path + "/v1/models") or "/v1/models",
                         headers=_auth_headers())
            resp = conn.getresponse()
            resp.read()
            if resp.status in (200, 401, 403):
                # 401/403 still proves the server is there; the actual
                # transcription call will surface auth problems loudly.
                log.info("Cloud endpoint reachable: %s://%s:%d (HTTP %d)",
                         scheme, host, port, resp.status)
                if resp.status in (401, 403):
                    log.warning(
                        "Cloud endpoint rejected the API key (HTTP %d) — "
                        "check cloud.api_key / VOICECLIP_CLOUD_API_KEY",
                        resp.status,
                    )
            else:
                log.warning("Cloud endpoint returned HTTP %d on /v1/models",
                            resp.status)
        finally:
            conn.close()
    except OSError as e:
        log.warning("Cloud endpoint not reachable yet (%s://%s:%d): %s",
                    scheme, host, port, e)
        return

    # Warm the server-side model off the critical path: after a container
    # restart (e.g. daily `voiceclip cloud start`) the model isn't loaded
    # until first use, making the first dictation pay a multi-second cold
    # load. A tiny silent clip fired now means the model is loading while
    # the app finishes starting.
    import threading

    def _warm():
        import io
        import wave as _wave

        buf = io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 4800)  # 0.3s silence
        buf.seek(0)
        tmp_path = None
        try:
            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as fh:
                fh.write(buf.read())
            transcribe(tmp_path, model_id)
            log.info("Cloud model warmed")
        except Exception as e:  # noqa: BLE001 — warmup is best-effort
            log.debug("Cloud model warmup skipped: %s", e)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    threading.Thread(target=_warm, daemon=True, name="cloud-warmup").start()


def transcribe(audio_path: str, model_id: str) -> str | None:
    """POST the WAV to /v1/audio/transcriptions. Returns text or None.

    Raises CloudEngineError on network/HTTP failure so the dispatcher
    reports a real error instead of silently dropping the dictation.
    """
    from voiceclip import config

    scheme, host, port, base_path = _endpoint()

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Multipart form fields, OpenAI transcription API shape.
    fields = {
        "model": model_id,
        "response_format": "json",
    }
    if config.ENGLISH_ONLY:
        fields["language"] = "en"
    if config.INITIAL_PROMPT:
        fields["prompt"] = config.INITIAL_PROMPT

    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n"
            ).encode()
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{os.path.basename(audio_path)}"\r\n'
            f"Content-Type: audio/wav\r\n"
            f"\r\n"
        ).encode()
        + audio_data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        **_auth_headers(),
    }

    # One attempt on the pooled keep-alive connection (if any), and if
    # that fails — the server or tunnel silently dropped it while idle —
    # one retry on a fresh connection. Errors on a fresh connection are
    # real and surface immediately.
    resp = resp_data = None
    for attempt in (1, 2):
        conn, reused = _checkout_conn(REQUEST_TIMEOUT)
        try:
            conn.request("POST", base_path + TRANSCRIPTIONS_PATH,
                         body=body, headers=headers)
            resp = conn.getresponse()
            resp_data = resp.read().decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException) as e:
            _discard_conn(conn)
            if reused and attempt == 1:
                log.debug("Pooled connection was stale (%s); retrying fresh", e)
                continue
            # Fresh-connection failure: the SSM tunnel may have dropped
            # (laptop sleep, instance restart). Try to bring it back once
            # and retry, instead of failing until the app is restarted.
            if attempt == 1:
                try:
                    from voiceclip import tunnel
                    if tunnel.ensure():
                        log.info("Tunnel re-established; retrying request")
                        continue
                except Exception:
                    pass
            raise CloudEngineError(
                f"Could not reach cloud transcription server at "
                f"{scheme}://{host}:{port} — {e}"
            ) from e
        # Keep the connection for the next dictation unless the server
        # asked to close it.
        if resp.will_close:
            _discard_conn(conn)
        else:
            _checkin_conn(conn)
        break

    if resp.status == 401 or resp.status == 403:
        raise CloudEngineError(
            f"Cloud server rejected the API key (HTTP {resp.status}). "
            "Check cloud.api_key / VOICECLIP_CLOUD_API_KEY."
        )
    if resp.status != 200:
        raise CloudEngineError(
            f"Cloud server returned HTTP {resp.status}: {resp_data[:200]}"
        )

    try:
        result = json.loads(resp_data)
    except ValueError as e:
        raise CloudEngineError(
            f"Cloud server returned invalid JSON: {resp_data[:200]}"
        ) from e

    text = (result.get("text") or "").strip()
    if not text:
        return None

    # Normalize segment newlines to a single line; the formatter owns
    # paragraph structure (same convention as the other engines).
    return " ".join(
        line.strip() for line in text.splitlines() if line.strip()
    ) or None


def keep_warm_ping(model_id: str):
    """No-op: keeping the model warm is the server's job, and burning
    audio-seconds of metered usage on silence pings would cost money."""
