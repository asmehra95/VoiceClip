"""Shared AI model inventory for the Settings picker.

GET /api/ai/models → {"models": [{value, label, provider, available, note}],
                      "current": "<provider>:<model>" | "none"}

Sources, in order of preference for a private setup:
  - cloud: chat routes on the user's own gateway (GET /v1/models, filtered
    to non-audio models). Offline gateway → the configured route is still
    listed, marked unavailable, so the current choice never disappears.
  - local: curated MLX models (models.json); downloaded ones flagged.
  - openai / anthropic: listed only as available when the API key env var
    is set in the viewer's environment.
"""

import json
import logging
import os

from voiceclip import config, llm_provider
from voiceclip.viewer.routes import register_get

log = logging.getLogger(__name__)

_AUDIO_MARKERS = ("whisper", "systran", "tts", "kokoro", "*")


def _cloud_models() -> list[dict]:
    configured = config.SUMMARIES_CLOUD_MODEL or "assistant"
    routes: list[str] = []
    if config.ENGINE == "cloud" and (config.CLOUD_BASE_URL or "").strip():
        import urllib.request

        base = config.CLOUD_BASE_URL.strip().rstrip("/")
        req = urllib.request.Request(
            base + "/v1/models",
            headers={"Authorization": f"Bearer {(config.CLOUD_API_KEY or '').strip()}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=2.5) as r:
                data = json.load(r)
            routes = [
                m["id"] for m in data.get("data", [])
                if not any(x in m.get("id", "").lower() for x in _AUDIO_MARKERS)
            ]
        except Exception as e:  # gateway down / tunnel closed — degrade quietly
            log.debug("gateway model list unavailable: %s", e)
    def label(route: str) -> str:
        # The "assistant" route serves whatever LLM the server runs; name it.
        if route == "assistant":
            from voiceclip import cloud_control
            current = ((getattr(config, "_raw", {}) or {}).get("cloud", {}) or {}).get(
                "llm_model") or cloud_control.DEFAULT_CLOUD_LLM
            pretty = next((c["label"].split(" — ")[0] for c in cloud_control.CLOUD_LLM_CHOICES
                           if c["id"] == current), current.split("/")[-1])
            return f"{pretty} · your cloud"
        return f"{route} · your cloud"

    out = [{"value": f"cloud:{r}", "label": label(r), "provider": "cloud",
            "available": True, "note": "Private — runs on your server"} for r in routes]
    if configured not in routes:
        out.append({"value": f"cloud:{configured}", "label": label(configured),
                    "provider": "cloud", "available": False,
                    "note": "Server offline right now"})
    return out


def _downloaded_repo_ids() -> set[str]:
    try:
        from huggingface_hub import scan_cache_dir

        return {r.repo_id for r in scan_cache_dir().repos if r.repo_type == "model"}
    except Exception:
        return set()


def _local_models() -> list[dict]:
    cached = _downloaded_repo_ids()
    out = []
    for m in llm_provider.list_recommended_models():
        downloaded = m["id"] in cached
        out.append({
            "value": f"local:{m['id']}",
            "label": f"{m.get('label') or m['id']} · this Mac",
            "provider": "local",
            "available": True,
            "note": "Downloaded" if downloaded else f"Downloads ~{m.get('size_gb', '?')} GB on first use",
        })
    return out


def _vendor_models() -> list[dict]:
    out = []
    for provider, env, model in (
        ("openai", "OPENAI_API_KEY", config.SUMMARIES_OPENAI_MODEL or "gpt-4o-mini"),
        ("anthropic", "ANTHROPIC_API_KEY", config.SUMMARIES_ANTHROPIC_MODEL or "claude-haiku-4-5"),
    ):
        has_key = bool(os.environ.get(env))
        out.append({
            "value": f"{provider}:{model}",
            "label": f"{model} · {provider.capitalize() if provider == 'openai' else 'Anthropic'}",
            "provider": provider,
            "available": has_key,
            "note": "Sends text to a third party" if has_key else f"Needs {env}",
        })
    return out


def _get_models(req, query):
    from voiceclip.viewer.routes.settings import _current_ai_model

    models = _cloud_models() + _local_models() + _vendor_models()
    current = _current_ai_model()
    if current != "none" and current not in {m["value"] for m in models}:
        prov, _, model = current.partition(":")
        models.insert(0, {"value": current, "label": f"{model} · {prov}", "provider": prov,
                          "available": True, "note": "Current (custom)"})
    req._json({"models": models, "current": current})


register_get("/api/ai/models", _get_models)
