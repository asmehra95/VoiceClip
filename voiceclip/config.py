"""Configuration — single JSON file with env var overrides.

All user config lives in ~/.voiceclip/config.json. Environment variables
override JSON values for quick one-off changes. Internal constants that
users shouldn't touch stay as Python constants here.

Config file is auto-created with sensible defaults on first run.
"""

import json
import logging
import os
import sys
from enum import Enum

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPC protocol (internal, not user-configurable)
# ---------------------------------------------------------------------------


class RecorderCmd(Enum):
    """Commands sent from the main process to the recorder child process."""
    START = "start"
    STOP = "stop"
    LIST_DEVICES = "list_devices"
    QUIT = "quit"


# ---------------------------------------------------------------------------
# Internal constants (not exposed in config.json)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000       # Whisper expects 16kHz
MIN_FILE_BYTES = 1000     # WAV files smaller than this are treated as empty
SILENCE_RMS_THRESHOLD = 0.003
MIN_AUDIO_DURATION = 0.3  # Seconds
TEMP_PREFIX = "voiceclip_"
MIN_HOLD_SECONDS = 0.3    # Taps shorter than this are ignored
PASTE_DELAY = 0.05        # Seconds between copy and simulated paste

# HuggingFace repos for each Whisper model variant
MODELS = {
    "tiny.en":        "mlx-community/whisper-tiny.en-mlx",
    "base.en":        "mlx-community/whisper-base.en-mlx",
    "small.en":       "mlx-community/whisper-small.en-mlx",
    "medium.en":      "mlx-community/whisper-medium.en-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
}

VALID_MODELS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")


# ---------------------------------------------------------------------------
# Config file path
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.expanduser("~/.voiceclip")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

# ---------------------------------------------------------------------------
# Default config (written on first run)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "model": "large-v3-turbo",
    "english_only": True,
    "polish": False,
    "polish_model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "persona": "default",
    "personas": {
        "default": {
            "prompt": "",
            "dictionary": {}
        },
        "engineering": {
            "prompt": "AWS, S3, DynamoDB, Lambda, EC2, CloudFormation, Kubernetes, Docker, API Gateway, microservices, latency, throughput, p99, SLA",
            "dictionary": {
                "dynamo db": "DynamoDB",
                "cloud formation": "CloudFormation",
                "s three": "S3",
                "e c two": "EC2",
                "kubernetes": "Kubernetes",
                "api": "API"
            }
        },
        "casual": {
            "prompt": "Hey, sounds good, let me know, no worries, thanks",
            "dictionary": {}
        },
        "medical": {
            "prompt": "diagnosis, prognosis, prescription, symptoms, patient, clinical, therapy, medication, dosage, referral",
            "dictionary": {}
        }
    },
    "dictionary": {
        "voiceclip": "VoiceClip",
        "macos": "macOS",
        "iphone": "iPhone"
    }
}


# ---------------------------------------------------------------------------
# Runtime config (populated by load())
# ---------------------------------------------------------------------------

# These are the "live" values used by all modules.
# Set by load() at startup, overridable by env vars.
MODEL = "large-v3-turbo"
ENGLISH_ONLY = True
POLISH_ENABLED = False
POLISH_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PERSONA = "default"

# Populated by load() — the merged dictionary (global + persona)
DICTIONARY: dict[str, str] = {}

# Populated by load() — the persona prompt + dictionary values for Whisper
INITIAL_PROMPT: str | None = None

# Full config dict for anything that needs raw access
_raw: dict = {}


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def _ensure_config_file():
    """Create config.json with defaults if it doesn't exist."""
    if os.path.exists(CONFIG_PATH):
        return
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=2)
        log.info("Created config at %s", CONFIG_PATH)
    except OSError as e:
        log.warning("Could not create config file: %s", e)


def load():
    """Load config from JSON file, apply env var overrides, resolve persona.

    Call this once at startup. Sets all module-level config variables.
    """
    global MODEL, ENGLISH_ONLY, POLISH_ENABLED, POLISH_MODEL
    global PERSONA, DICTIONARY, INITIAL_PROMPT, _raw

    _ensure_config_file()

    # Load JSON
    cfg = dict(_DEFAULT_CONFIG)  # start with defaults
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                # Deep merge personas
                if "personas" in user_cfg and "personas" in cfg:
                    cfg["personas"].update(user_cfg["personas"])
                    user_cfg_no_personas = {k: v for k, v in user_cfg.items() if k != "personas"}
                    cfg.update(user_cfg_no_personas)
                else:
                    cfg.update(user_cfg)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not load config (%s): %s", CONFIG_PATH, e)

    _raw = cfg

    # Apply env var overrides (env vars always win)
    MODEL = os.environ.get("VOICECLIP_MODEL", cfg.get("model", "large-v3-turbo"))
    ENGLISH_ONLY = os.environ.get(
        "VOICECLIP_ENGLISH_ONLY",
        str(cfg.get("english_only", True))
    ).lower() == "true"
    POLISH_ENABLED = os.environ.get(
        "VOICECLIP_POLISH",
        str(cfg.get("polish", False))
    ).lower() == "true"
    POLISH_MODEL = os.environ.get(
        "VOICECLIP_POLISH_MODEL",
        cfg.get("polish_model", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    )
    PERSONA = os.environ.get("VOICECLIP_PERSONA", cfg.get("persona", "default"))

    # Resolve persona
    personas = cfg.get("personas", {})
    active_persona = personas.get(PERSONA, {})
    if PERSONA != "default" and PERSONA not in personas:
        log.warning("Persona '%s' not found in config, using 'default'", PERSONA)
        PERSONA = "default"
        active_persona = personas.get("default", {})

    # Merge dictionaries: global + persona (persona wins on conflicts)
    global_dict = cfg.get("dictionary", {})
    persona_dict = active_persona.get("dictionary", {})
    DICTIONARY = {**global_dict, **persona_dict}

    # Build initial_prompt: persona prompt + all dictionary values
    prompt_parts = []
    persona_prompt = active_persona.get("prompt", "")
    if persona_prompt:
        prompt_parts.append(persona_prompt)

    # Add dictionary values as prompt hints
    dict_words = [v for v in DICTIONARY.values() if isinstance(v, str)]
    if dict_words:
        # Deduplicate
        seen = set()
        unique = []
        for w in dict_words:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        prompt_parts.append(", ".join(unique))

    INITIAL_PROMPT = ". ".join(prompt_parts)[:500] if prompt_parts else None

    log.info(
        "Config loaded: model=%s, english=%s, polish=%s, persona=%s, "
        "dict=%d entries, prompt=%s",
        MODEL, ENGLISH_ONLY, POLISH_ENABLED, PERSONA,
        len(DICTIONARY),
        repr(INITIAL_PROMPT[:80] + "...") if INITIAL_PROMPT and len(INITIAL_PROMPT) > 80 else repr(INITIAL_PROMPT),
    )


def validate():
    """Validate configuration at startup. Exits on error."""
    if MODEL not in VALID_MODELS:
        print(f"Unknown model: {MODEL}")
        print(f"Valid options: {', '.join(VALID_MODELS)}")
        sys.exit(1)


def get_model_repo():
    """Return the HuggingFace repo for the configured model."""
    if ENGLISH_ONLY and MODEL in ("tiny", "base", "small", "medium"):
        key = f"{MODEL}.en"
    else:
        key = MODEL
    return MODELS[key], key
