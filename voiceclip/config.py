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
    QUIT = "quit"


# ---------------------------------------------------------------------------
# Internal constants (not exposed in config.json)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000       # Whisper expects 16kHz
MIN_FILE_BYTES = 1000     # WAV files smaller than this are treated as empty
SILENCE_RMS_THRESHOLD = 0.003
MIN_AUDIO_DURATION = 0.3  # Seconds
# Hard cap on a single recording. Protects against stuck-key / forgotten-toggle
# scenarios that would otherwise grow the frame buffer linearly (~4 MB per
# minute at 16 kHz float32). Past the cap the callback drops new frames
# silently; the user still needs to release the key to finish the cycle, but
# memory stops growing and transcription only gets the first MAX_RECORDING_SECONDS
# of audio.
MAX_RECORDING_SECONDS = 120
TEMP_PREFIX = "voiceclip_"
MIN_HOLD_SECONDS = 0.3    # Taps shorter than this are ignored
PASTE_DELAY = 0.02        # Copy → paste grace period. pbcopy is synchronous,
                          # so this only exists for slow (Electron) apps to
                          # observe the pasteboard change before Cmd+V lands.

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

# Parakeet models via parakeet-mlx — mlx-community HuggingFace repos.
PARAKEET_MODELS = {
    "parakeet-tdt-0.6b-v3": "mlx-community/parakeet-tdt-0.6b-v3",
    "parakeet-tdt-1.1b": "mlx-community/parakeet-tdt-1.1b",
    "parakeet-ctc-1.1b": "mlx-community/parakeet-ctc-1.1b",
    "parakeet-ctc-0.6b": "mlx-community/parakeet-ctc-0.6b",
    "parakeet-rnnt-1.1b": "mlx-community/parakeet-rnnt-1.1b",
}

VALID_ENGINES = ("whisper", "whisper_cpp", "parakeet")


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
    "persona": "default",
    "hotkey": "alt_r",
    "hotkey_mode": "hold",
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
    },
    # Flat list of words/phrases that bias Whisper toward them. Easier
    # to edit from the UI than the persona/dictionary block — just one
    # string per entry, no key/value mapping required. Appended verbatim
    # to the Whisper initial_prompt.
    "custom_vocabulary": []
}


# ---------------------------------------------------------------------------
# Runtime config (populated by load())
# ---------------------------------------------------------------------------

# These are the "live" values used by all modules.
# Set by load() at startup, overridable by env vars.
ENGINE = "whisper"  # "whisper" or "parakeet"
MODEL = "large-v3-turbo"
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
ENGLISH_ONLY = True
PERSONA = "default"
HOTKEY = "alt_r"
HOTKEY_MODE = "hold"  # "hold" = hold-to-record, "toggle" = press-to-start/press-to-stop
HISTORY_ENABLED = False
HISTORY_MAX_DAYS = 30

# Reflections — second hotkey that saves a private note to history without pasting.
# Off unless REFLECTION_HOTKEY is set in config or env.
REFLECTION_HOTKEY: str | None = None
REFLECTION_HOTKEY_MODE = "hold"
REFLECTION_MAX_DAYS = 0  # 0 = never auto-delete reflections

# Polish — third hotkey that transcribes then runs the text through a local
# LLM for cleanup (grammar, structure, filler removal) before pasting.
# Off unless POLISH_HOTKEY is set in config or env.
POLISH_HOTKEY: str | None = None
POLISH_HOTKEY_MODE = "hold"
POLISH_PROMPT = (
    "Clean up this dictated text. Fix grammar, remove filler words (um, uh, like), "
    "add proper punctuation, and structure into clear sentences or paragraphs. "
    "Keep the original meaning and tone. Output only the cleaned text, nothing else."
)

# Summaries — LLM-generated daily recaps. Off ("none") by default.
# Provider: "none" | "local" | "openai" | "anthropic"
SUMMARIES_PROVIDER = "none"
SUMMARIES_LOCAL_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
SUMMARIES_OPENAI_MODEL = "gpt-4o-mini"
SUMMARIES_ANTHROPIC_MODEL = "claude-haiku-4-5"
SUMMARIES_STYLE = "descriptive"  # "descriptive" | "reflective"

# Research — queue-based research assistant. Off by default.
# Provider: "none" | "local" | "openai" | "anthropic"
# Local research answers from model knowledge only — no web search, no
# sources. Cloud providers can use their server-side web search tool.
RESEARCH_PROVIDER = "none"
RESEARCH_LOCAL_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
RESEARCH_OPENAI_MODEL = "gpt-4o-mini"
RESEARCH_ANTHROPIC_MODEL = "claude-haiku-4-5"

# Patterns — longitudinal coach looking across your recent history.
# Provider: "none" | "local" | "openai" | "anthropic"
PATTERNS_PROVIDER = "none"
PATTERNS_LOCAL_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
PATTERNS_OPENAI_MODEL = "gpt-4o-mini"
PATTERNS_ANTHROPIC_MODEL = "claude-haiku-4-5"
PATTERNS_WINDOW_DAYS = 7

# Populated by load() — the merged dictionary (global + persona)
DICTIONARY: dict[str, str] = {}

# User-maintained list of extra vocabulary (names, jargon, acronyms) that
# bias Whisper without needing a full persona-dictionary entry. Edited
# from the Settings tab as a textarea (one entry per line).
CUSTOM_VOCABULARY: list[str] = []

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
        os.chmod(CONFIG_PATH, 0o600)
        log.info("Created config at %s", CONFIG_PATH)
    except OSError as e:
        log.warning("Could not create config file: %s", e)


_VALID_PROVIDERS = ("none", "local", "openai", "anthropic")
_DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"


def _load_provider_block(
    cfg: dict, block_name: str, env_prefix: str,
) -> tuple[str, str, str, str]:
    """Load a feature's provider + three model-id fields from config + env.

    Each of summaries / research / patterns has the same shape:
      - A nested dict in config.json (e.g. cfg["summaries"])
      - A provider field validated against _VALID_PROVIDERS
      - Three model-id fields (local, openai, anthropic) with env overrides

    Returns (provider, local_model, openai_model, anthropic_model).
    """
    block = cfg.get(block_name, {})
    if not isinstance(block, dict):
        block = {}

    provider = os.environ.get(
        f"VOICECLIP_{env_prefix}_PROVIDER",
        block.get("provider", "none"),
    )
    if provider not in _VALID_PROVIDERS:
        log.warning("Invalid %s.provider '%s', using 'none'", block_name, provider)
        provider = "none"

    local_model = os.environ.get(
        f"VOICECLIP_{env_prefix}_LOCAL_MODEL",
        block.get("local_model", _DEFAULT_LOCAL_MODEL),
    )
    openai_model = os.environ.get(
        f"VOICECLIP_{env_prefix}_OPENAI_MODEL",
        block.get("openai_model", _DEFAULT_OPENAI_MODEL),
    )
    anthropic_model = os.environ.get(
        f"VOICECLIP_{env_prefix}_ANTHROPIC_MODEL",
        block.get("anthropic_model", _DEFAULT_ANTHROPIC_MODEL),
    )
    return provider, local_model, openai_model, anthropic_model


def load():
    """Load config from JSON file, apply env var overrides, resolve persona.

    Call this once at startup. Sets all module-level config variables.
    """
    global ENGINE, MODEL, PARAKEET_MODEL, ENGLISH_ONLY, PERSONA, DICTIONARY, INITIAL_PROMPT
    global CUSTOM_VOCABULARY
    global HOTKEY, HOTKEY_MODE, HISTORY_ENABLED, HISTORY_MAX_DAYS, _raw
    global REFLECTION_HOTKEY, REFLECTION_HOTKEY_MODE, REFLECTION_MAX_DAYS
    global POLISH_HOTKEY, POLISH_HOTKEY_MODE, POLISH_PROMPT
    global SUMMARIES_PROVIDER, SUMMARIES_LOCAL_MODEL
    global SUMMARIES_OPENAI_MODEL, SUMMARIES_ANTHROPIC_MODEL, SUMMARIES_STYLE
    global RESEARCH_PROVIDER, RESEARCH_LOCAL_MODEL
    global RESEARCH_OPENAI_MODEL, RESEARCH_ANTHROPIC_MODEL
    global PATTERNS_PROVIDER, PATTERNS_LOCAL_MODEL, PATTERNS_OPENAI_MODEL
    global PATTERNS_ANTHROPIC_MODEL, PATTERNS_WINDOW_DAYS

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
    ENGINE = os.environ.get("VOICECLIP_ENGINE", cfg.get("engine", "whisper"))
    if ENGINE not in VALID_ENGINES:
        log.warning("Invalid engine '%s', using 'whisper'", ENGINE)
        ENGINE = "whisper"

    MODEL = os.environ.get("VOICECLIP_MODEL", cfg.get("model", "large-v3-turbo"))

    PARAKEET_MODEL = os.environ.get(
        "VOICECLIP_PARAKEET_MODEL",
        cfg.get("parakeet_model", "mlx-community/parakeet-tdt-0.6b-v3"),
    )

    ENGLISH_ONLY = os.environ.get(
        "VOICECLIP_ENGLISH_ONLY",
        str(cfg.get("english_only", True))
    ).lower() == "true"
    PERSONA = os.environ.get("VOICECLIP_PERSONA", cfg.get("persona", "default"))
    HOTKEY = os.environ.get("VOICECLIP_HOTKEY", cfg.get("hotkey", "alt_r"))
    HOTKEY_MODE = os.environ.get("VOICECLIP_HOTKEY_MODE", cfg.get("hotkey_mode", "hold"))
    if HOTKEY_MODE not in ("hold", "toggle"):
        log.warning("Invalid hotkey_mode '%s', using 'hold'", HOTKEY_MODE)
        HOTKEY_MODE = "hold"

    HISTORY_ENABLED = os.environ.get(
        "VOICECLIP_HISTORY",
        str(cfg.get("history", False))
    ).lower() == "true"
    try:
        HISTORY_MAX_DAYS = int(cfg.get("history_max_days", 30))
    except (ValueError, TypeError):
        HISTORY_MAX_DAYS = 30
    if HISTORY_MAX_DAYS < 0:
        HISTORY_MAX_DAYS = 30

    # Reflection hotkey — optional, only active when set
    REFLECTION_HOTKEY = os.environ.get(
        "VOICECLIP_REFLECTION_HOTKEY",
        cfg.get("reflection_hotkey") or None,
    )
    if REFLECTION_HOTKEY is not None and not str(REFLECTION_HOTKEY).strip():
        REFLECTION_HOTKEY = None
    REFLECTION_HOTKEY_MODE = os.environ.get(
        "VOICECLIP_REFLECTION_HOTKEY_MODE",
        cfg.get("reflection_hotkey_mode", "hold"),
    )
    if REFLECTION_HOTKEY_MODE not in ("hold", "toggle"):
        log.warning("Invalid reflection_hotkey_mode '%s', using 'hold'", REFLECTION_HOTKEY_MODE)
        REFLECTION_HOTKEY_MODE = "hold"
    try:
        REFLECTION_MAX_DAYS = int(cfg.get("reflection_max_days", 0))
    except (ValueError, TypeError):
        REFLECTION_MAX_DAYS = 0
    if REFLECTION_MAX_DAYS < 0:
        REFLECTION_MAX_DAYS = 0

    # Polish hotkey — optional third hotkey for LLM-cleaned dictation
    POLISH_HOTKEY = os.environ.get(
        "VOICECLIP_POLISH_HOTKEY",
        cfg.get("polish_hotkey") or None,
    )
    if POLISH_HOTKEY is not None and not str(POLISH_HOTKEY).strip():
        POLISH_HOTKEY = None
    POLISH_HOTKEY_MODE = os.environ.get(
        "VOICECLIP_POLISH_HOTKEY_MODE",
        cfg.get("polish_hotkey_mode", "hold"),
    )
    if POLISH_HOTKEY_MODE not in ("hold", "toggle"):
        log.warning("Invalid polish_hotkey_mode '%s', using 'hold'", POLISH_HOTKEY_MODE)
        POLISH_HOTKEY_MODE = "hold"
    POLISH_PROMPT = cfg.get(
        "polish_prompt",
        POLISH_PROMPT,  # keep the module-level default if not in config
    )

    # Summaries / Research / Patterns — three features with identical
    # provider + model loading shape. Collapsed into a helper to avoid
    # 120 lines of three-way duplication.
    SUMMARIES_PROVIDER, SUMMARIES_LOCAL_MODEL, SUMMARIES_OPENAI_MODEL, \
        SUMMARIES_ANTHROPIC_MODEL = _load_provider_block(
            cfg, "summaries", "SUMMARIES")
    SUMMARIES_STYLE = cfg.get("summaries", {}).get("style", "descriptive") \
        if isinstance(cfg.get("summaries"), dict) else "descriptive"
    if SUMMARIES_STYLE not in ("descriptive", "reflective"):
        log.warning("Invalid summaries.style '%s', using 'descriptive'", SUMMARIES_STYLE)
        SUMMARIES_STYLE = "descriptive"

    RESEARCH_PROVIDER, RESEARCH_LOCAL_MODEL, RESEARCH_OPENAI_MODEL, \
        RESEARCH_ANTHROPIC_MODEL = _load_provider_block(
            cfg, "research", "RESEARCH")

    PATTERNS_PROVIDER, PATTERNS_LOCAL_MODEL, PATTERNS_OPENAI_MODEL, \
        PATTERNS_ANTHROPIC_MODEL = _load_provider_block(
            cfg, "patterns", "PATTERNS")
    patterns_cfg = cfg.get("patterns", {})
    if not isinstance(patterns_cfg, dict):
        patterns_cfg = {}
    try:
        PATTERNS_WINDOW_DAYS = int(patterns_cfg.get("window_days", 7))
    except (ValueError, TypeError):
        PATTERNS_WINDOW_DAYS = 7
    if PATTERNS_WINDOW_DAYS < 1:
        PATTERNS_WINDOW_DAYS = 7

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

    # Custom vocabulary — flat list of words/phrases, UI-editable.
    # Sanitize: keep strings only, trim whitespace, drop empties,
    # dedupe case-insensitively while preserving first-seen casing.
    raw_vocab = cfg.get("custom_vocabulary", [])
    if not isinstance(raw_vocab, list):
        log.warning("custom_vocabulary must be a list, got %s; ignoring",
                    type(raw_vocab).__name__)
        raw_vocab = []
    CUSTOM_VOCABULARY = []
    _seen_vocab: set[str] = set()
    for item in raw_vocab:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        key = s.lower()
        if key in _seen_vocab:
            continue
        _seen_vocab.add(key)
        CUSTOM_VOCABULARY.append(s)

    # Build initial_prompt: persona prompt + all dictionary values +
    # custom vocabulary (kept separate so users can add biasing words
    # without wading into the persona/dictionary map).
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

    # Append custom-vocab words last; same running dedupe so a word that
    # already appears in the persona dictionary isn't doubled up.
    if CUSTOM_VOCABULARY:
        fresh = [w for w in CUSTOM_VOCABULARY if w.lower() not in seen] \
                if dict_words else list(CUSTOM_VOCABULARY)
        if fresh:
            prompt_parts.append(", ".join(fresh))

    INITIAL_PROMPT = ". ".join(prompt_parts)[:900] if prompt_parts else None



def validate():
    """Validate configuration at startup. Exits on error."""
    if ENGINE == "whisper" and MODEL not in VALID_MODELS:
        print(f"Unknown Whisper model: {MODEL}")
        print(f"Valid options: {', '.join(VALID_MODELS)}")
        sys.exit(1)
    if ENGINE == "parakeet":
        known = set(PARAKEET_MODELS.values())
        if PARAKEET_MODEL not in known and not PARAKEET_MODEL.startswith("mlx-community/parakeet-"):
            valid_names = list(PARAKEET_MODELS.values())
            print(f"Unknown Parakeet model: {PARAKEET_MODEL}")
            print(f"Valid options: {', '.join(valid_names)}")
            sys.exit(1)


def get_model_repo():
    """Return the HuggingFace repo for the configured model."""
    if ENGLISH_ONLY and MODEL in ("tiny", "base", "small", "medium"):
        key = f"{MODEL}.en"
    else:
        key = MODEL
    return MODELS[key], key


# ---------------------------------------------------------------------------
# Cloud-provider warnings
# ---------------------------------------------------------------------------
# Each feature (summaries / research / patterns) needs to warn the user
# when its provider is set to a cloud service, so they know data will
# leave the machine. Centralized here so the message format is consistent
# and adding a new feature is one mapping entry, not a new function.

_CLOUD_WARNING_COPY = {
    "summaries": (
        "SUMMARIES_PROVIDER",
        "Summaries",
        "Your day's entries will be sent to that provider when a summary "
        "is generated.",
    ),
    "research": (
        "RESEARCH_PROVIDER",
        "Research",
        "When you research a topic, that topic is sent to the provider. "
        "If the model uses web search, the topic also goes to the search "
        "backend.",
    ),
    "patterns": (
        "PATTERNS_PROVIDER",
        "Patterns",
        "When you open the Patterns tab, your recent reflections and daily "
        "summaries are sent to the provider.",
    ),
}


def cloud_provider_warning(feature: str) -> str | None:
    """Return a user-facing warning string if the named feature is
    currently set to a cloud provider, else None.

    `feature` is one of: 'summaries', 'research', 'patterns'.
    Unknown features return None (no warning rather than a crash).
    """
    entry = _CLOUD_WARNING_COPY.get(feature)
    if entry is None:
        return None
    attr, label, body = entry
    provider = globals().get(attr, "none")
    if provider not in ("openai", "anthropic"):
        return None
    return f"{label}: cloud provider '{provider}' is enabled. {body}"


# ---------------------------------------------------------------------------
# Hotkey resolution
# ---------------------------------------------------------------------------

# Map of config string → pynput key. Supports both Key attributes and
# single characters for letter/number keys.
_KEY_MAP = {
    "alt_r": "Key.alt_r",
    "alt_l": "Key.alt_l",
    "ctrl_r": "Key.ctrl_r",
    "ctrl_l": "Key.ctrl_l",
    "shift_r": "Key.shift_r",
    "shift_l": "Key.shift_l",
    "cmd_r": "Key.cmd_r",
    "cmd_l": "Key.cmd_l",
    "caps_lock": "Key.caps_lock",
    "f1": "Key.f1", "f2": "Key.f2", "f3": "Key.f3", "f4": "Key.f4",
    "f5": "Key.f5", "f6": "Key.f6", "f7": "Key.f7", "f8": "Key.f8",
    "f9": "Key.f9", "f10": "Key.f10", "f11": "Key.f11", "f12": "Key.f12",
    "space": "Key.space",
    "esc": "Key.esc",
}


def resolve_hotkey(key_str: str | None = None):
    """Resolve a hotkey config string to a pynput key object.

    If key_str is None, uses the module-level HOTKEY (transcription hotkey).
    Returns a pynput Key enum member or a KeyCode for character keys.
    """
    from pynput import keyboard

    source = key_str if key_str is not None else HOTKEY
    ks = source.lower().strip()

    # Check named keys
    if ks in _KEY_MAP:
        attr_path = _KEY_MAP[ks]
        # e.g. "Key.alt_r" → keyboard.Key.alt_r
        parts = attr_path.split(".")
        obj = keyboard
        for part in parts:
            obj = getattr(obj, part)
        return obj

    # Single character key (e.g., "z", "x")
    if len(ks) == 1:
        return keyboard.KeyCode.from_char(ks)

    # Try as a pynput Key attribute directly
    try:
        return getattr(keyboard.Key, ks)
    except AttributeError:
        log.warning("Unknown hotkey '%s', falling back to Right Option", source)
        return keyboard.Key.alt_r


def hotkey_display_name(key_str: str | None = None) -> str:
    """Return a human-readable name for a configured hotkey."""
    source = key_str if key_str is not None else HOTKEY
    names = {
        "alt_r": "Right Option (⌥)",
        "alt_l": "Left Option (⌥)",
        "ctrl_r": "Right Control (⌃)",
        "ctrl_l": "Left Control (⌃)",
        "shift_r": "Right Shift (⇧)",
        "shift_l": "Left Shift (⇧)",
        "cmd_r": "Right Command (⌘)",
        "cmd_l": "Left Command (⌘)",
        "caps_lock": "Caps Lock",
        "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4",
        "f5": "F5", "f6": "F6", "f7": "F7", "f8": "F8",
        "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
        "space": "Space",
        "esc": "Escape",
    }
    return names.get(source.lower().strip(), source)


def model_id_for(feature: str) -> str | None:
    """Return the active model_id for a feature based on its configured provider.

    Resolves the if-local/openai/anthropic dispatch that every feature module
    repeats. Returns None if the provider is 'none' or unrecognized.

    Supported features: 'summaries', 'research', 'patterns'.
    """
    _FEATURE_MAP = {
        "summaries": (SUMMARIES_PROVIDER, SUMMARIES_LOCAL_MODEL,
                      SUMMARIES_OPENAI_MODEL, SUMMARIES_ANTHROPIC_MODEL),
        "research": (RESEARCH_PROVIDER, RESEARCH_LOCAL_MODEL,
                     RESEARCH_OPENAI_MODEL, RESEARCH_ANTHROPIC_MODEL),
        "patterns": (PATTERNS_PROVIDER, PATTERNS_LOCAL_MODEL,
                     PATTERNS_OPENAI_MODEL, PATTERNS_ANTHROPIC_MODEL),
    }
    entry = _FEATURE_MAP.get(feature)
    if entry is None:
        return None
    provider, local, openai, anthropic = entry
    if provider == "local":
        return local
    if provider == "openai":
        return openai
    if provider == "anthropic":
        return anthropic
    return None
