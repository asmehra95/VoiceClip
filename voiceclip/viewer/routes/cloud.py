"""Cloud server controls for the viewer's Settings tab.

GET  /api/cloud/status   → {configured, state, instance_type, instance_id, region}
POST /api/cloud/control  → {"action": "stop"|"start"} → resulting state

Thin wrappers over voiceclip.cloud_control (shared with the CLI). The
viewer binds to localhost only, so these carry the same trust level as
the settings API — anyone who can reach them already controls the config.
"""

from voiceclip import cloud_control, config
from voiceclip.viewer.routes import register_get, register_post


def _configured() -> bool:
    return bool((getattr(config, "CLOUD_INSTANCE_ID", "") or "").strip()
                and (getattr(config, "CLOUD_REGION", "") or "").strip())


def _tunnel_up() -> bool:
    """Is the gateway answering on its local tunnel port right now?"""
    try:
        from voiceclip import tunnel
        return tunnel._gateway_alive(tunnel._gateway_port(), timeout=1.0)
    except Exception:
        return False


def _get_status(req, query):
    if not _configured():
        req._json({"configured": False})
        return
    try:
        status = cloud_control.get_status()
    except cloud_control.CloudControlError as e:
        req._json({"configured": True, "error": str(e)}, status=502)
        return
    req._json({"configured": True, "tunnel": _tunnel_up(), **status})


def _post_control(req, payload):
    action = (payload or {}).get("action")
    if action not in ("stop", "start"):
        req._json({"error": "action must be 'stop' or 'start'"}, status=400)
        return
    if not _configured():
        req._json({"error": "cloud.instance_id / cloud.region not set"},
                  status=400)
        return
    try:
        fn = (cloud_control.stop_instance if action == "stop"
              else cloud_control.start_instance)
        result = fn()
    except cloud_control.CloudControlError as e:
        req._json({"error": str(e)}, status=502)
        return
    req._json({"ok": True, **result})


def _current_llm() -> str:
    raw = (getattr(config, "_raw", {}) or {}).get("cloud", {}) or {}
    return raw.get("llm_model") or cloud_control.DEFAULT_CLOUD_LLM


def _get_llm(req, query):
    """Choices + current cloud LLM; ?probe=1 also checks it answers."""
    out = {"choices": cloud_control.CLOUD_LLM_CHOICES, "current": _current_llm()}
    if (query.get("probe") or ["0"])[0] == "1":
        out["ready"] = cloud_control.llm_ready()
    req._json(out)


def _post_llm(req, payload):
    model = str((payload or {}).get("model") or "").strip()
    if not cloud_control.valid_llm_model_id(model):
        req._json({"error": "Use a Hugging Face repo id like 'Org/Model-Name'."}, status=400)
        return
    if not _configured():
        req._json({"error": "cloud.instance_id / cloud.region not set"}, status=400)
        return
    try:
        result = cloud_control.set_llm_model(model)
    except cloud_control.CloudControlError as e:
        req._json({"error": str(e)}, status=502)
        return
    # Remember locally so the UI shows the choice without an SSM round trip.
    from voiceclip.config_io import write_config_patch
    write_config_patch({"cloud": {"llm_model": model}})
    config.load()
    req._json({"ok": True, **result})


register_get("/api/cloud/status", _get_status)
register_get("/api/cloud/llm", _get_llm)
register_post("/api/cloud/llm", _post_llm)
register_post("/api/cloud/control", _post_control)
