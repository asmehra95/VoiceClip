"""whisper.cpp engine — runs whisper-cli binary with Metal GPU acceleration.

Uses a pre-built whisper-cli binary and GGML quantized models for faster
inference than mlx_whisper, especially on large-v3 (32 decoder layers).

Exposes the same interface as engine_whisper:
  load(model_id)              → verify binary + model exist
  transcribe(path, model_id)  → raw text or None
  keep_warm_ping(model_id)    → no-op (binary is stateless)
"""

import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)

# Paths — configurable via env vars for flexibility
WHISPER_CPP_BIN = os.environ.get(
    "VOICECLIP_WHISPER_CPP_BIN",
    os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"),
)
WHISPER_CPP_MODELS_DIR = os.environ.get(
    "VOICECLIP_WHISPER_CPP_MODELS",
    os.path.expanduser("~/.voiceclip/models"),
)

# Map config model names → GGML filenames
_MODEL_FILES = {
    "large-v3": "ggml-large-v3-q5_0.bin",
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


def _resolve_model_path(model_id: str) -> str:
    """Resolve a model identifier to a GGML file path.

    Accepts:
      - A full path to a .bin file
      - A model name from _MODEL_FILES (e.g. "large-v3")
      - A filename in the models directory
    """
    # Direct path
    if os.path.isfile(model_id):
        return model_id

    # Known model name
    if model_id in _MODEL_FILES:
        path = os.path.join(WHISPER_CPP_MODELS_DIR, _MODEL_FILES[model_id])
        if os.path.isfile(path):
            return path

    # Try as filename in models dir
    path = os.path.join(WHISPER_CPP_MODELS_DIR, model_id)
    if os.path.isfile(path):
        return path

    # Try with .bin extension
    path = os.path.join(WHISPER_CPP_MODELS_DIR, f"ggml-{model_id}.bin")
    if os.path.isfile(path):
        return path

    raise FileNotFoundError(
        f"whisper.cpp model not found for '{model_id}'. "
        f"Expected at {WHISPER_CPP_MODELS_DIR}. "
        f"Download with: curl -L -o ~/.voiceclip/models/ggml-large-v3-q5_0.bin "
        f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin"
    )


def load(model_id: str):
    """Verify the whisper-cli binary and model file exist."""
    if not os.path.isfile(WHISPER_CPP_BIN):
        raise FileNotFoundError(
            f"whisper-cli binary not found at {WHISPER_CPP_BIN}. "
            "Build it: cd ~/whisper.cpp && cmake -B build -DGGML_METAL=ON && "
            "cmake --build build --config Release"
        )
    if not os.access(WHISPER_CPP_BIN, os.X_OK):
        raise PermissionError(f"whisper-cli is not executable: {WHISPER_CPP_BIN}")

    model_path = _resolve_model_path(model_id)
    log.info("whisper.cpp ready: binary=%s, model=%s", WHISPER_CPP_BIN, model_path)


def transcribe(audio_path: str, model_id: str) -> str | None:
    """Run whisper-cli on the audio file. Returns text or None."""
    from voiceclip import config

    model_path = _resolve_model_path(model_id)

    cmd = [
        WHISPER_CPP_BIN,
        "-m", model_path,
        "-f", audio_path,
        "--no-timestamps",
        "--no-prints",
        "-t", "4",  # threads
    ]

    # Language
    if config.ENGLISH_ONLY:
        cmd.extend(["-l", "en"])

    # Initial prompt (vocabulary biasing)
    if config.INITIAL_PROMPT:
        cmd.extend(["--prompt", config.INITIAL_PROMPT])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            text=True,
        )
    except subprocess.TimeoutExpired:
        log.error("whisper.cpp timed out after 60s")
        return None
    except FileNotFoundError:
        log.error("whisper-cli binary not found at %s", WHISPER_CPP_BIN)
        return None

    if result.returncode != 0:
        stderr = result.stderr.strip()[:200] if result.stderr else ""
        log.error("whisper.cpp failed (rc=%d): %s", result.returncode, stderr)
        return None

    text = result.stdout.strip()
    if not text or text == "[BLANK_AUDIO]":
        return None

    return text


def keep_warm_ping(model_id: str):
    """No-op — whisper.cpp is a stateless binary, no model to keep warm."""
    pass
