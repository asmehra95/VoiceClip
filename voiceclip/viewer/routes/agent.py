"""Voice-agent controls for the viewer's Agent tab.

GET  /api/agent/status?after=N → {running, events: [...seq > N]}
POST /api/agent/start          → spawns `python -m voiceclip agent`
POST /api/agent/stop           → SIGINT (graceful pipecat shutdown), then SIGTERM

The agent runs as its own process (it owns the mic and speakers); this
module only supervises it and tails its event log. POSTs are protected by
the server's same-origin/JSON guard — starting a mic-listening process
must never be triggerable by another website.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time

from voiceclip.viewer.routes import register_get, register_post

log = logging.getLogger(__name__)

_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def _paths():
    from voiceclip.agent import agent_dir
    d = agent_dir()
    return (os.path.join(d, "agent.pid"), os.path.join(d, "events.jsonl"),
            os.path.join(d, "agent.log"))


def _running_pid() -> int | None:
    pid_path, _, _ = _paths()
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def _read_events(after: int, limit: int = 400) -> list[dict]:
    _, events_path, _ = _paths()
    out = []
    try:
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("seq", 0) > after:
                    out.append(rec)
    except OSError:
        return []
    return out[-limit:]


def _get_status(req, query):
    try:
        after = int((query.get("after") or ["0"])[0])
    except ValueError:
        after = 0
    running = _running_pid() is not None
    with _lock:
        starting = _proc is not None and _proc.poll() is None and not running
    events = _read_events(0, limit=100000)
    session = events[0]["ts"] if events else None      # new file per session
    level = None
    try:
        with open(os.path.join(os.path.dirname(_paths()[1]), "level.json")) as f:
            lv = json.load(f)
        if time.time() - lv.get("ts", 0) < 1.5:
            level = lv.get("level")
    except (OSError, ValueError):
        pass
    req._json({"running": running or starting, "session": session, "level": level,
               "events": [e for e in events if e.get("seq", 0) > after][-400:]})


def _post_start(req, payload):
    global _proc
    with _lock:
        if _running_pid() is not None or (_proc is not None and _proc.poll() is None):
            req._json({"ok": True, "already_running": True})
            return
        import voiceclip
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(voiceclip.__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        _, _, log_path = _paths()
        with open(log_path, "w") as logf:
            _proc = subprocess.Popen(
                [sys.executable, "-m", "voiceclip", "agent"],
                cwd=pkg_root, env=env, stdin=subprocess.DEVNULL,
                stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
            )
    req._json({"ok": True})


def _post_stop(req, payload):
    pid = _running_pid()
    with _lock:
        proc = _proc
    if pid is None and (proc is None or proc.poll() is not None):
        req._json({"ok": True, "was_running": False})
        return

    def _stop():
        target = pid or proc.pid
        try:
            os.kill(target, signal.SIGINT)        # pipecat handles SIGINT gracefully
        except OSError:
            return
        deadline = time.time() + 6
        while time.time() < deadline:
            try:
                os.kill(target, 0)
            except OSError:
                break
            time.sleep(0.2)
        else:
            try:
                os.kill(target, signal.SIGTERM)
            except OSError:
                pass
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except Exception:
                pass

    threading.Thread(target=_stop, daemon=True).start()
    req._json({"ok": True, "was_running": True})


register_get("/api/agent/status", _get_status)
register_post("/api/agent/start", _post_start)
register_post("/api/agent/stop", _post_stop)
