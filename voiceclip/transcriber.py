"""Whisper transcription via mlx-whisper on Apple Silicon GPU.

Performance notes:
- Model is preloaded at import time to eliminate first-transcription delay
- get_model_repo() is called once and cached
- Transcription has a timeout watchdog to prevent permanent hangs
"""

import logging
import os
import tempfile
import threading

import mlx_whisper
import numpy as np
import soundfile as sf

from voiceclip.config import ENGLISH_ONLY, TEMP_PREFIX, get_model_repo
from voiceclip.utils import safe_unlink

log = logging.getLogger(__name__)

# Pre-resolve the model repo at import time (avoids per-call overhead)
_REPO, _MODEL_KEY = get_model_repo()

# Timeout for transcription calls (seconds). If mlx_whisper hangs beyond
# this, the watchdog thread will log an error. The call itself can't be
# forcefully killed (C extension), but _busy will be cleared so the user
# can keep recording.
TRANSCRIBE_TIMEOUT = 120

# Initial prompt built from user dictionary — biases Whisper toward
# correct spellings of custom words. Set by set_initial_prompt().
_initial_prompt: str | None = None


def set_initial_prompt(words: list[str]):
    """Build an initial_prompt from dictionary values.

    Whisper uses initial_prompt as a conditioning prefix, which biases
    the model toward producing these exact spellings. Call this after
    loading the user dictionary.
    """
    global _initial_prompt
    if words:
        # Deduplicate while preserving order, limit to ~200 chars
        seen = set()
        unique = []
        for w in words:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        _initial_prompt = ", ".join(unique)[:500]
        log.info("Whisper initial_prompt set (%d words)", len(unique))
    else:
        _initial_prompt = None


def preload_model():
    """Force-load the Whisper model into memory so the first transcription is fast.

    Call this at startup. mlx_whisper caches internally, so subsequent
    transcribe() calls reuse the loaded model.
    """
    log.info("Preloading model: %s (%s)...", _MODEL_KEY, _REPO)
    try:
        # Transcribe a tiny silent audio to trigger model loading
        # without doing real work. mlx_whisper will cache the model.
        tmp = tempfile.NamedTemporaryFile(
            prefix=TEMP_PREFIX, suffix=".wav", delete=False
        )
        # 0.5s of silence at 16kHz
        silence = np.zeros(8000, dtype=np.float32)
        sf.write(tmp.name, silence, 16000)
        tmp.close()

        mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=_REPO,
            language="en",
            no_speech_threshold=0.6,
        )
        safe_unlink(tmp.name)
        log.info("Model preloaded successfully")
    except Exception as e:
        log.warning("Model preload failed (will load on first use): %s", e)


def transcribe(audio_path):
    """Transcribe a WAV file and return the text, or None.

    Deletes the audio file after transcription regardless of outcome.
    Includes a timeout watchdog — if mlx_whisper hangs beyond
    TRANSCRIBE_TIMEOUT seconds, returns None so the hotkey handler
    can recover.
    """
    if not audio_path:
        return None

    result_box = [None]
    error_box = [None]

    def _do_transcribe():
        try:
            kwargs = dict(
                path_or_hf_repo=_REPO,
                language="en" if ENGLISH_ONLY else None,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
            )
            if _initial_prompt:
                kwargs["initial_prompt"] = _initial_prompt

            result_box[0] = mlx_whisper.transcribe(audio_path, **kwargs)
        except Exception as e:
            error_box[0] = e

    worker = threading.Thread(target=_do_transcribe, daemon=True)
    worker.start()
    worker.join(timeout=TRANSCRIBE_TIMEOUT)

    if worker.is_alive():
        log.error(
            "Transcription timed out after %ds — model may be hung. "
            "Returning None so recording can continue.",
            TRANSCRIBE_TIMEOUT,
        )
        # Can't kill the thread (C extension), but we return None
        # so the caller clears _busy and the user can keep recording.
        # The temp file will be cleaned up on next startup.
        return None

    # Clean up temp file now that transcription is done
    safe_unlink(audio_path)

    if error_box[0]:
        log.error("Transcription error: %s", error_box[0])
        return None

    result = result_box[0]
    if not result:
        return None

    segments = result.get("segments", [])
    real = [s for s in segments if s.get("no_speech_prob", 0) < 0.7]

    if not real:
        return None

    text = " ".join(s["text"].strip() for s in real).strip()
    return text or None
