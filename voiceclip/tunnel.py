"""Managed SSM tunnel — auto-started with VoiceClip for the cloud engine.

The cloud transcription server has no public endpoint; it's reached via an
AWS SSM port-forwarding session that maps the gateway to localhost. This
module owns that session for the lifetime of the VoiceClip process:

  - ensure()   → start the tunnel if the gateway isn't already reachable
                 (a manually-run tunnel or the local dev stack both answer
                 the health probe, in which case nothing is started)
  - shutdown() → terminate the managed session, if we started one

Resilience (same behavior as scripts/tunnel.sh in the server repo):
  - SSM kills sessions after ~20 minutes idle; a keepalive thread pings
    the gateway through the tunnel every 4 minutes to prevent that.
  - If the session dies anyway (laptop sleep, network change), a monitor
    thread restarts it within seconds.

Requires the AWS CLI + Session Manager plugin — the same dependencies as
`voiceclip cloud`. If they're missing, ensure() logs one warning and
returns; dictation then fails with the usual "could not reach server"
error, pointing the user at the manual setup.
"""

import http.client
import logging
import shutil
import subprocess
import threading
import time
import urllib.parse

log = logging.getLogger(__name__)

_KEEPALIVE_INTERVAL = 240   # seconds between health pings
_RECONNECT_DELAY = 3        # seconds before restarting a dead session
_STARTUP_WAIT = 12          # seconds to wait for the first connection
_WAKE_STARTUP_WAIT = 90     # same, but after waking a parked instance

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_running = False            # True while the monitor should keep the tunnel up


def _gateway_port() -> int:
    from voiceclip import config

    parsed = urllib.parse.urlparse((config.CLOUD_BASE_URL or "").strip())
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _gateway_is_local() -> bool:
    from voiceclip import config

    parsed = urllib.parse.urlparse((config.CLOUD_BASE_URL or "").strip())
    return parsed.hostname in ("localhost", "127.0.0.1", "::1")


def _gateway_alive(port: int, timeout: float = 2.0) -> bool:
    """True when something OpenAI-shaped answers on localhost:port."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            conn.request("GET", "/health/liveliness")
            return conn.getresponse().status < 500
        finally:
            conn.close()
    except OSError:
        return False


def _spawn(instance_id: str, region: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "aws", "ssm", "start-session",
            "--region", region,
            "--target", instance_id,
            "--document-name", "AWS-StartPortForwardingSession",
            "--parameters",
            f'{{"portNumber":["4000"],"localPortNumber":["{port}"]}}',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        # Own process group: `aws` runs session-manager-plugin as a child,
        # and that child is what holds the local port. Killing only the
        # parent would orphan the plugin and leave the tunnel open.
        start_new_session=True,
    )


def _terminate(proc: subprocess.Popen):
    """Terminate the session's whole process group (aws + plugin)."""
    import os
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            return
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _monitor(instance_id: str, region: str, port: int):
    """Keep the session alive: ping to defeat the idle timeout, restart
    the subprocess when it exits. Runs until shutdown() clears _running."""
    global _proc

    last_ping = time.time()
    while _running:
        time.sleep(1)
        with _lock:
            proc = _proc
        if not _running:
            break
        if proc is None or proc.poll() is not None:
            log.warning("SSM tunnel session ended; reconnecting...")
            time.sleep(_RECONNECT_DELAY)
            if not _running:
                break
            with _lock:
                _proc = _spawn(instance_id, region, port)
            last_ping = time.time()
            continue
        if time.time() - last_ping >= _KEEPALIVE_INTERVAL:
            _gateway_alive(port, timeout=5)   # ping through the tunnel
            last_ping = time.time()


def _wake_if_stopped(instance_id: str, region: str, wait_s: int = 180) -> bool:
    """Start a parked instance (the server stops itself after an idle hour
    — see infra/idle-stop.sh) and wait until it is running.

    Returns True when we actually woke it (caller should allow extra time
    for the SSM agent to register), False when it was already running or
    nothing sensible could be done. Never raises — downstream tunnel
    failure produces the usual "could not reach server" guidance.
    """
    from voiceclip import cloud_control
    try:
        state = cloud_control.get_status().get("state")
    except Exception as e:
        log.debug("wake: status check failed: %s", e)
        return False
    if state == "stopping":
        # Must reach full stop before EC2 accepts a start.
        log.info("Cloud instance is stopping — waiting to restart it...")
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if cloud_control.get_status().get("state") == "stopped":
                    state = "stopped"
                    break
            except Exception:
                pass
            time.sleep(5)
    if state != "stopped":
        return False
    log.info("Cloud instance was parked (idle auto-stop) — waking it "
             "(boot ~2 min; the assistant needs a few more to reload)")
    print("  ⏳ Cloud server was parked — waking it up (~2 min)...")
    try:
        cloud_control.start_instance()
    except Exception as e:
        log.warning("wake: start failed: %s — try `voiceclip cloud start`", e)
        return False
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            if cloud_control.get_status().get("state") == "running":
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def ensure() -> bool:
    """Start the managed tunnel when needed. Returns True when the gateway
    is reachable (whether or not we started anything). Never raises."""
    global _proc, _running

    from voiceclip import config

    if config.ENGINE != "cloud" or not getattr(config, "CLOUD_AUTO_TUNNEL", True):
        return False
    if not _gateway_is_local():
        return False   # remote base_url (public deployment) — no tunnel
    instance_id = (getattr(config, "CLOUD_INSTANCE_ID", "") or "").strip()
    region = (getattr(config, "CLOUD_REGION", "") or "").strip()
    port = _gateway_port()

    if _gateway_alive(port):
        log.info("Gateway already reachable on localhost:%d — "
                 "not starting a tunnel", port)
        return True
    if not instance_id or not region:
        log.info("cloud.instance_id/region not set; cannot auto-tunnel")
        return False
    if shutil.which("aws") is None:
        log.warning("AWS CLI not found — cannot start the SSM tunnel. "
                    "Install it (brew install awscli) or run the tunnel "
                    "manually.")
        return False

    woke = _wake_if_stopped(instance_id, region)
    with _lock:
        if _running:
            return True
        _running = True
        _proc = _spawn(instance_id, region, port)
    threading.Thread(
        target=_monitor, args=(instance_id, region, port),
        daemon=True, name="ssm-tunnel-monitor",
    ).start()

    # Wait briefly for the session to open so the first dictation works.
    # After a wake the SSM agent needs extra time to register, so wait
    # longer before giving up (the monitor keeps retrying regardless).
    deadline = time.time() + (_WAKE_STARTUP_WAIT if woke else _STARTUP_WAIT)
    while time.time() < deadline:
        if _gateway_alive(port):
            log.info("SSM tunnel up: localhost:%d → %s (%s)",
                     port, instance_id, region)
            return True
        time.sleep(0.5)

    log.warning(
        "SSM tunnel did not become ready in %ds — is the instance running? "
        "Check with: voiceclip cloud status", _STARTUP_WAIT,
    )
    return False


def shutdown():
    """Stop the managed tunnel (no-op when we never started one)."""
    global _proc, _running

    with _lock:
        _running = False
        proc, _proc = _proc, None
    if proc is not None and proc.poll() is None:
        _terminate(proc)
        log.info("SSM tunnel closed")
