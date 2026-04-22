"""Whisper transcription via mlx-whisper on Apple Silicon GPU.

Performance notes:
- Model is preloaded at import time to eliminate first-transcription delay
- get_model_repo() is called once and cached
"""

import logging
import os

import mlx_whisper

from voiceclip.config import ENGLISH_ONLY, get_model_repo

log = logging.getLogger(__name__)

# Pre-resolve the model repo at import time (avoids per-call overhead)
_REPO, _MODEL_KEY = get_model_repo()


def preload_model():
    """Force-load the Whisper model into memory so the first transcription is fast.

    Call this at startup. mlx_whisper caches internally, so subsequent
    transcribe() calls reuse the loaded model.
    """
    log.info("Preloading model: %s (%s)...", _MODEL_KEY, _REPO)
    try:
        # Transcribe a tiny silent audio to trigger model loading
        # without doing real work. mlx_whisper will cache the model.
        import tempfile
        import numpy as np

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        # 0.5s of silence at 16kHz
        silence = np.zeros(8000, dtype=np.float32)
        import soundfile as sf
        sf.write(tmp.name, silence, 16000)
        tmp.close()

        mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=_REPO,
            language="en",
            no_speech_threshold=0.6,
        )
        os.unlink(tmp.name)
        log.info("Model preloaded successfully")
    except Exception as e:
        log.warning("Model preload failed (will load on first use): %s", e)


def transcribe(audio_path):
    """Transcribe a WAV file and return the text, or None.

    Deletes the audio file after transcription regardless of outcome.
    """
    if not audio_path:
        return None

    try:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=_REPO,
            language="en" if ENGLISH_ONLY else None,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
        )

        segments = result.get("segments", [])
        real = [s for s in segments if s.get("no_speech_prob", 0) < 0.7]

        if not real:
            return None

        text = " ".join(s["text"].strip() for s in real).strip()
        return text or None

    except Exception as e:
        log.error("Transcription error: %s", e)
        return None
    finally:
        _safe_unlink(audio_path)


def _safe_unlink(path):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
