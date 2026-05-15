"""Whisper engine — mlx_whisper transcription on Apple Silicon GPU.

Exposes three functions consumed by the transcriber dispatcher:
  load(model_id)           → force-load model into memory
  transcribe(path, model_id) → raw text or None
  keep_warm_ping(model_id)   → touch model weights to prevent swap-out
"""

import logging

log = logging.getLogger(__name__)


def load(model_id: str):
    """Force-load the Whisper model by transcribing silence."""
    import tempfile

    import mlx_whisper
    import numpy as np
    import soundfile as sf

    from voiceclip.config import TEMP_PREFIX
    from voiceclip.utils import safe_unlink

    tmp = tempfile.NamedTemporaryFile(
        prefix=TEMP_PREFIX, suffix=".wav", delete=False
    )
    silence = np.zeros(8000, dtype=np.float32)
    sf.write(tmp.name, silence, 16000)
    tmp.close()

    try:
        mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=model_id,
            language="en",
            no_speech_threshold=0.6,
        )
    finally:
        safe_unlink(tmp.name)


def transcribe(audio_path: str, model_id: str) -> str | None:
    """Run mlx_whisper on the audio file. Returns text or None (silence)."""
    import mlx_whisper

    from voiceclip import config

    kwargs = dict(
        path_or_hf_repo=model_id,
        language="en" if config.ENGLISH_ONLY else None,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
    )
    if config.INITIAL_PROMPT:
        kwargs["initial_prompt"] = config.INITIAL_PROMPT

    result = mlx_whisper.transcribe(audio_path, **kwargs)
    if not result:
        return None

    segments = result.get("segments", [])
    if not segments:
        return None

    # No post-filtering — the recorder already rejects silence via RMS
    # threshold, and the model's no_speech_threshold handles the rest.
    # Filtering here was dropping valid speech with background noise.
    text = " ".join(s["text"].strip() for s in segments).strip()
    return text or None


def keep_warm_ping(model_id: str):
    """Minimal transcription to keep model weights in page cache."""
    import tempfile

    import mlx_whisper
    import numpy as np
    import soundfile as sf

    from voiceclip.config import TEMP_PREFIX
    from voiceclip.utils import safe_unlink

    tmp = tempfile.NamedTemporaryFile(
        prefix=TEMP_PREFIX, suffix=".wav", delete=False
    )
    silence = np.zeros(8000, dtype=np.float32)
    sf.write(tmp.name, silence, 16000)
    tmp.close()

    try:
        mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=model_id,
            language="en",
            no_speech_threshold=0.6,
        )
    finally:
        safe_unlink(tmp.name)
