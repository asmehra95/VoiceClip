"""Settings endpoints — the UI-editable subset of ~/.voiceclip/config.json.

GET  /api/settings          — schema + current values + system info
POST /api/settings/update   — validate and apply a partial config patch

Everything the Settings tab can show or change is governed by the
_SETTINGS_SCHEMA whitelist below. Anything not in the schema stays
power-user territory (edit config.json directly).
"""

from __future__ import annotations

import logging
import os as _os
from pathlib import Path as _P

from voiceclip import config, history
from voiceclip.viewer.routes import register_get, register_post

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# A curated whitelist of config keys the viewer's Settings tab can read and
# write. `restart_required` marks fields that only take effect after the
# daemon reboots — the UI shows a banner when any of these change. Fields
# without it apply immediately (they're only read by the viewer or on-demand
# paths).

_SETTINGS_SCHEMA: dict[str, dict] = {
    # Dictation
    "engine": {
        "group": "Dictation",
        "type": "select",
        "restart_required": True,
        # "auto" picks whisper_cpp when its binaries + model are present
        # (install.sh sets them up), mlx-whisper otherwise.
        "choices": ["auto", "whisper", "whisper_cpp", "parakeet", "cloud"],
    },
    # Cloud engine — OpenAI-compatible transcription server. The API key is
    # deliberately NOT exposed here: the viewer has no secret masking, and
    # keys belong in VOICECLIP_CLOUD_API_KEY (or config.json, chmod 600).
    "cloud.base_url": {
        "group": "Remote infra",
        "type": "text",
        "restart_required": True,
        "placeholder": "https://transcribe.example.com",
        "visible_when": {"engine": "cloud"},
        "advanced": True
    },
    "cloud.model": {
        "group": "Remote infra",
        "type": "text",
        "restart_required": True,
        "placeholder": "whisper-large-v3",
        "visible_when": {"engine": "cloud"},
        "advanced": True
    },
    # EC2 lifecycle control — powers the "Cloud server" panel and the
    # `voiceclip cloud` CLI. Applied immediately (read at call time).
    "cloud.instance_id": {
        "group": "Remote infra",
        "type": "text",
        "placeholder": "i-0123456789abcdef0",
        "visible_when": {"engine": "cloud"},
        "advanced": True
    },
    "cloud.region": {
        "group": "Remote infra",
        "type": "text",
        "placeholder": "eu-central-1",
        "visible_when": {"engine": "cloud"},
        "advanced": True
    },
    "cloud.fallback_engine": {
        "group": "Dictation",
        "type": "select",
        "restart_required": True,
        "choices": ["none", "whisper", "whisper_cpp", "parakeet"],
        "visible_when": {"engine": "cloud"},
    },
    "cloud.streaming": {
        "group": "Dictation",
        "type": "bool",
        "restart_required": True,
        "visible_when": {"engine": "cloud"},
    },
    "cloud.auto_tunnel": {
        "group": "Remote infra",
        "type": "bool",
        "restart_required": True,
        "visible_when": {"engine": "cloud"}
    },
    "model": {
        "group": "Dictation",
        "type": "select",
        "restart_required": True,
        "choices": ["tiny", "base", "small", "medium",
                    "large-v3-turbo", "large-v3"],
        "visible_when": {"engine": ["whisper", "whisper_cpp"]},
    },
    "parakeet_model": {
        "group": "Dictation",
        "type": "select",
        "restart_required": True,
        "choices": ["mlx-community/parakeet-tdt-0.6b-v3",
                    "mlx-community/parakeet-tdt-1.1b",
                    "mlx-community/parakeet-ctc-1.1b",
                    "mlx-community/parakeet-ctc-0.6b",
                    "mlx-community/parakeet-rnnt-1.1b"],
        "visible_when": {"engine": "parakeet"}
    },
    "hotkey": {
        "group": "Hotkeys",
        "type": "select",
        "restart_required": True,
        "choices": ["alt_r", "alt_l", "ctrl_r", "ctrl_l", "shift_r", "shift_l",
                    "cmd_r", "cmd_l", "caps_lock", "f1", "f2", "f3", "f4",
                    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                    "space", "esc"],
    },
    "hotkey_mode": {
        "group": "Hotkeys",
        "type": "select",
        "restart_required": True,
        "choices": ["hold", "toggle"],
    },
    "english_only": {
        "group": "Dictation",
        "type": "bool",
        "restart_required": True,
        # Also applies to the cloud engine (sent as the API `language` field).
        "visible_when": {"engine": ["whisper", "whisper_cpp", "cloud"]},
    },

    # Reflections
    "reflection_hotkey": {
        "group": "Hotkeys",
        "type": "select_or_none",
        "restart_required": True,
        "choices": ["alt_r", "alt_l", "ctrl_r", "ctrl_l", "shift_r", "shift_l",
                    "cmd_r", "cmd_l", "caps_lock", "f1", "f2", "f3", "f4",
                    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                    "space", "esc"],
    },
    "reflection_hotkey_mode": {
        "group": "Hotkeys",
        "type": "select",
        "restart_required": True,
        "choices": ["hold", "toggle"],
    },
    "window_title_capture": {
        "group": "Dictation",
        "type": "bool",
        "restart_required": True,
    },

    # Assistant (voice companion, cloud engine)
    "assistant_hotkey": {
        "group": "Hotkeys",
        "type": "select_or_none",
        "restart_required": True,
        "choices": ["alt_r", "alt_l", "ctrl_r", "ctrl_l", "shift_r", "shift_l",
                    "cmd_r", "cmd_l", "caps_lock", "f1", "f2", "f3", "f4",
                    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                    "space", "esc"],
    },
    "assistant_hotkey_mode": {
        "group": "Hotkeys",
        "type": "select",
        "restart_required": True,
        "choices": ["hold", "toggle"],
    },

    # Polish
    "polish_hotkey": {
        "group": "Hotkeys",
        "type": "select_or_none",
        "restart_required": True,
        "choices": ["alt_r", "alt_l", "ctrl_r", "ctrl_l", "shift_r", "shift_l",
                    "cmd_r", "cmd_l", "caps_lock", "f1", "f2", "f3", "f4",
                    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                    "space", "esc"],
    },
    "polish_hotkey_mode": {
        "group": "Hotkeys",
        "type": "select",
        "restart_required": True,
        "choices": ["hold", "toggle"],
    },
    # One model for every AI text feature (summaries, research, patterns,
    # polish, Ask). Value is "<provider>:<model>" or "none".
    "ai.model": {
        "group": "AI",
        "type": "ai_model",
        "cloud_providers": ["openai", "anthropic"],
    },
    "autocorrect": {
        "group": "AI",
        "type": "bool",
        "restart_required": True,
        "visible_when": {"ai.model": "!none"},
    },
    "polish_prompt": {
        "group": "AI",
        "type": "text",
        "placeholder": "Clean up this dictated text. Fix grammar...",
        "visible_when": {"polish_hotkey": "!null"}
    },

    # Journal (history)
    "history": {
        "group": "Dictation",
        "type": "bool",
        "restart_required": True,
    },

    # Summaries
    "summaries.provider": {
        "group": "AI",
        "type": "select",
        "cloud_providers": ["openai", "anthropic"],
        "choices": ["none", "local", "openai", "anthropic", "cloud"],
        "hidden": True
    },
    "summaries.local_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "visible_when": {"summaries.provider": "local"},
        "hidden": True
    },
    "summaries.openai_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "gpt-4o-mini",
        "visible_when": {"summaries.provider": "openai"},
        "hidden": True
    },
    "summaries.anthropic_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "claude-haiku-4-5",
        "visible_when": {"summaries.provider": "anthropic"},
        "hidden": True
    },
    "summaries.cloud_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "assistant",
        "visible_when": {"summaries.provider": "cloud"},
        "hidden": True
    },
    "summaries.style": {
        "group": "AI",
        "type": "select",
        "choices": ["descriptive", "reflective"],
        "visible_when": {"summaries.provider": "!none"},
        "hidden": True
    },

    # Research
    "research.provider": {
        "group": "AI",
        "type": "select",
        "cloud_providers": ["openai", "anthropic"],
        "choices": ["none", "local", "openai", "anthropic", "cloud"],
        "hidden": True
    },
    "research.local_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "visible_when": {"research.provider": "local"},
        "hidden": True
    },
    "research.openai_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "gpt-4o-mini",
        "visible_when": {"research.provider": "openai"},
        "hidden": True
    },
    "research.anthropic_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "claude-haiku-4-5",
        "visible_when": {"research.provider": "anthropic"},
        "hidden": True
    },
    "research.cloud_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "assistant",
        "visible_when": {"research.provider": "cloud"},
        "hidden": True
    },

    # Patterns
    "patterns.provider": {
        "group": "AI",
        "type": "select",
        "cloud_providers": ["openai", "anthropic"],
        "choices": ["none", "local", "openai", "anthropic", "cloud"],
        "hidden": True
    },
    "patterns.local_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "visible_when": {"patterns.provider": "local"},
        "hidden": True
    },
    "patterns.openai_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "gpt-4o-mini",
        "visible_when": {"patterns.provider": "openai"},
        "hidden": True
    },
    "patterns.anthropic_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "claude-haiku-4-5",
        "visible_when": {"patterns.provider": "anthropic"},
        "hidden": True
    },
    "patterns.cloud_model": {
        "group": "AI",
        "type": "text",
        "placeholder": "assistant",
        "visible_when": {"patterns.provider": "cloud"},
        "hidden": True
    },
    "patterns.window_days": {
        "group": "AI",
        "type": "int",
        "min": 1,
        "max": 90,
        "visible_when": {"patterns.provider": "!none"},
        "hidden": True
    },

    # Dictation vocabulary — flat list of words/phrases to bias Whisper.
    # Rendered as a textarea where each non-empty line is one entry.
    "custom_vocabulary": {
        "group": "Dictation",
        "type": "text_list",
        "restart_required": True,
        "placeholder": "One per line (names, jargon, acronyms)",
        # Also applies to the cloud engine (sent as the API `prompt` field).
        "visible_when": {"engine": ["whisper", "whisper_cpp", "cloud"]},
        "hidden": True
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_ai_model() -> str:
    """The shared AI model as "<provider>:<model>" (or "none").

    Falls back to the summaries block for configs written before the
    shared "ai" block existed, so the picker shows what's really in use.
    """
    provider = config.AI_PROVIDER or config.SUMMARIES_PROVIDER
    if not provider or provider == "none":
        return "none"
    model = config.AI_MODEL or config.model_id_for("summaries") or ""
    return f"{provider}:{model}" if model else "none"


_AI_PROVIDERS = ("local", "openai", "anthropic", "cloud")


def _runtime_value(key: str):
    """Read the current runtime value for a dotted config key."""
    mapping = {
        "engine": config.ENGINE,
        "model": config.MODEL,
        "parakeet_model": config.PARAKEET_MODEL,
        "cloud.base_url": config.CLOUD_BASE_URL,
        "cloud.model": config.CLOUD_MODEL,
        "cloud.instance_id": config.CLOUD_INSTANCE_ID,
        "cloud.region": config.CLOUD_REGION,
        "cloud.streaming": config.CLOUD_STREAMING,
        "cloud.fallback_engine": config.CLOUD_FALLBACK_ENGINE,
        "cloud.auto_tunnel": config.CLOUD_AUTO_TUNNEL,
        "hotkey": config.HOTKEY,
        "hotkey_mode": config.HOTKEY_MODE,
        "english_only": config.ENGLISH_ONLY,
        "reflection_hotkey": config.REFLECTION_HOTKEY,
        "reflection_hotkey_mode": config.REFLECTION_HOTKEY_MODE,
        "window_title_capture": False,  # not currently in runtime config; default
        "assistant_hotkey": config.ASSISTANT_HOTKEY,
        "assistant_hotkey_mode": config.ASSISTANT_HOTKEY_MODE,
        "polish_hotkey": config.POLISH_HOTKEY,
        "polish_hotkey_mode": config.POLISH_HOTKEY_MODE,
        "polish_prompt": config.POLISH_PROMPT,
        "history": config.HISTORY_ENABLED,
        "summaries.provider": config.SUMMARIES_PROVIDER,
        "summaries.local_model": config.SUMMARIES_LOCAL_MODEL,
        "summaries.openai_model": config.SUMMARIES_OPENAI_MODEL,
        "summaries.anthropic_model": config.SUMMARIES_ANTHROPIC_MODEL,
        "summaries.cloud_model": config.SUMMARIES_CLOUD_MODEL,
        "summaries.style": config.SUMMARIES_STYLE,
        "research.provider": config.RESEARCH_PROVIDER,
        "research.local_model": config.RESEARCH_LOCAL_MODEL,
        "research.openai_model": config.RESEARCH_OPENAI_MODEL,
        "research.anthropic_model": config.RESEARCH_ANTHROPIC_MODEL,
        "research.cloud_model": config.RESEARCH_CLOUD_MODEL,
        "patterns.provider": config.PATTERNS_PROVIDER,
        "patterns.local_model": config.PATTERNS_LOCAL_MODEL,
        "patterns.openai_model": config.PATTERNS_OPENAI_MODEL,
        "patterns.anthropic_model": config.PATTERNS_ANTHROPIC_MODEL,
        "patterns.cloud_model": config.PATTERNS_CLOUD_MODEL,
        "patterns.window_days": config.PATTERNS_WINDOW_DAYS,
        "custom_vocabulary": list(config.CUSTOM_VOCABULARY),
        "ai.model": _current_ai_model(),
        "autocorrect": config.AUTOCORRECT,
    }
    return mapping.get(key)


def _system_info() -> dict:
    """Snapshot of the same stuff `voiceclip doctor` surfaces."""
    db = _P(history.DB_PATH)
    cfg = _P(config.CONFIG_PATH)
    hf_cache = _P(_os.path.expanduser("~/.cache/huggingface"))

    def _size_gb(p):
        try:
            if not p.exists():
                return 0.0
            total = 0
            for f in p.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            return total / (1024 ** 3)
        except Exception:
            return 0.0

    return {
        "db_path": str(db),
        "db_size_mb": (db.stat().st_size / (1024 * 1024)) if db.exists() else 0,
        "config_path": str(cfg),
        "huggingface_cache_gb": round(_size_gb(hf_cache), 2),
        "history_enabled": config.HISTORY_ENABLED,
    }


def _settings_payload() -> dict:
    """Build the GET /api/settings response: schema + current values."""
    values = {}
    for key in _SETTINGS_SCHEMA:
        values[key] = _runtime_value(key)
    return {
        "schema": _SETTINGS_SCHEMA,
        "values": values,
        "system": _system_info(),
    }


def _apply_settings_patch(patch: dict) -> dict:
    """Validate a patch against the schema and write it.

    Returns `{"ok": True, "restart_required": bool, "applied": {...}}` on
    success, or `{"error": "..."}` on validation failure.
    """
    from voiceclip.config_io import write_config_patch

    if not isinstance(patch, dict) or not patch:
        return {"error": "empty or invalid patch"}

    normalized: dict = {}
    restart_required = False
    for key, new_value in patch.items():
        schema = _SETTINGS_SCHEMA.get(key)
        if schema is None:
            return {"error": f"unknown setting '{key}'"}
        typ = schema["type"]

        if typ == "bool":
            if not isinstance(new_value, bool):
                return {"error": f"'{key}' must be true or false"}
        elif typ == "int":
            if not isinstance(new_value, int) or isinstance(new_value, bool):
                return {"error": f"'{key}' must be an integer"}
            if "min" in schema and new_value < schema["min"]:
                return {"error": f"'{key}' must be >= {schema['min']}"}
            if "max" in schema and new_value > schema["max"]:
                return {"error": f"'{key}' must be <= {schema['max']}"}
        elif typ == "select":
            if new_value not in schema["choices"]:
                return {"error": f"'{key}' must be one of {schema['choices']}"}
        elif typ == "select_or_none":
            if new_value is not None and new_value not in schema["choices"]:
                return {"error": f"'{key}' must be null or one of {schema['choices']}"}
        elif typ == "text":
            if new_value is not None and not isinstance(new_value, str):
                return {"error": f"'{key}' must be a string"}
        elif typ == "ai_model":
            if not isinstance(new_value, str) or len(new_value) > 200:
                return {"error": f"'{key}' must be a string"}
            if new_value != "none":
                prov, sep, model = new_value.partition(":")
                if not sep or prov not in _AI_PROVIDERS or not model.strip():
                    return {"error": f"'{key}' must be 'none' or '<provider>:<model>'"}
        elif typ == "text_list":
            # Accept list of strings OR newline-separated string (textarea).
            # Normalize to cleaned list: trim, drop empties, dedupe preserving
            # first-seen casing.
            if isinstance(new_value, str):
                raw_items = new_value.splitlines()
            elif isinstance(new_value, list):
                raw_items = new_value
            else:
                return {"error": f"'{key}' must be a list or newline-separated string"}
            cleaned: list[str] = []
            seen: set[str] = set()
            for item in raw_items:
                if not isinstance(item, str):
                    return {"error": f"'{key}' entries must be strings"}
                s = item.strip()
                if not s:
                    continue
                if len(s) > 80:
                    return {"error": f"'{key}' entries must be 80 characters or shorter"}
                low = s.lower()
                if low in seen:
                    continue
                seen.add(low)
                cleaned.append(s)
            if len(cleaned) > 200:
                return {"error": f"'{key}' is limited to 200 entries"}
            new_value = cleaned
        else:
            return {"error": f"unhandled type for '{key}'"}

        if schema.get("restart_required"):
            restart_required = True
        normalized[key] = new_value

    # Translate dotted keys into a nested-dict patch that the merging
    # writer will combine with existing config.
    nested: dict = {}
    if "ai.model" in normalized:
        value = normalized.pop("ai.model")
        prov, _, model = value.partition(":")
        nested["ai"] = ({"provider": "none", "model": ""} if value == "none"
                        else {"provider": prov, "model": model.strip()})
        normalized_ai = value
    else:
        normalized_ai = None
    for key, value in normalized.items():
        parts = key.split(".")
        target = nested
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value

    if not write_config_patch(nested):
        return {"error": "could not write config file"}

    # Reload runtime config so subsequent GET /api/settings reflects the
    # write. Fields that need daemon restart won't actually apply this
    # session — the UI shows a banner to that effect.
    config.load()

    return {
        "ok": True,
        "restart_required": restart_required,
        "applied": ({**normalized, "ai.model": normalized_ai}
                    if normalized_ai is not None else normalized),
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _get_settings(req, query):
    req._json(_settings_payload())


def _post_update(req, payload):
    result = _apply_settings_patch(payload)
    if "error" in result:
        req._json(result, status=400)
    else:
        req._json(result)


register_get("/api/settings", _get_settings)
register_post("/api/settings/update", _post_update)
