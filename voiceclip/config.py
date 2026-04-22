"""Configuration constants and environment variable parsing."""

import os
import sys
from enum import Enum


class RecorderCmd(Enum):
    """Commands sent from the main process to the recorder child process."""
    START = "start"
    STOP = "stop"
    LIST_DEVICES = "list_devices"
    QUIT = "quit"


# HuggingFace repos for each model variant
MODELS = {
    "tiny.en":        "mlx-community/whisper-tiny.en-mlx",
    "base.en":        "mlx-community/whisper-base.en-mlx",
    "small.en":       "mlx-community/whisper-small.en-mlx",
    "medium.en":      "mlx-community/whisper-medium.en-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
}

VALID_MODELS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")

# User-configurable via environment
MODEL = os.environ.get("VOICECLIP_MODEL", "large-v3-turbo")
ENGLISH_ONLY = os.environ.get("VOICECLIP_ENGLISH_ONLY", "true").lower() == "true"

# Timing
MIN_HOLD_SECONDS = 0.3    # Taps shorter than this are ignored
PASTE_DELAY = 0.05        # Seconds between copy and simulated paste

# Audio
SAMPLE_RATE = 16000       # Whisper expects 16kHz
MIN_FILE_BYTES = 1000     # WAV files smaller than this are treated as empty
SILENCE_RMS_THRESHOLD = 0.003
MIN_AUDIO_DURATION = 0.3  # Seconds

# Temp file prefix so cleanup only touches our files
TEMP_PREFIX = "voiceclip_"


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
