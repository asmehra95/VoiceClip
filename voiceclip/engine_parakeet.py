"""NVIDIA Parakeet ASR engine via parakeet-mlx on Apple Silicon GPU.

Uses the parakeet-mlx package (pip install parakeet-mlx) which runs
Parakeet CTC/RNNT/TDT models natively on Metal via MLX — same
acceleration approach as mlx_whisper.

Models hosted at mlx-community on HuggingFace:
  - mlx-community/parakeet-tdt-0.6b-v3  (best speed/quality)
  - mlx-community/parakeet-tdt-1.1b
  - mlx-community/parakeet-ctc-1.1b
  - mlx-community/parakeet-ctc-0.6b
  - mlx-community/parakeet-rnnt-1.1b

Requires: pip install parakeet-mlx

Exposes the same interface as engine_whisper:
  load(model_id)           → force-load model + warm JIT
  transcribe(path, model_id) → raw text or None
  keep_warm_ping(model_id)   → touch model weights
"""

import logging

log = logging.getLogger(__name__)

_model = None
_loaded_id: str | None = None


def load(model_id: str):
    """Load the Parakeet model and warm the MLX computation graph."""
    import tempfile

    import numpy as np
    import soundfile as sf

    from parakeet_mlx import from_pretrained

    from voiceclip.config import TEMP_PREFIX
    from voiceclip.utils import safe_unlink

    global _model, _loaded_id

    log.info("Loading Parakeet model: %s", model_id)
    _model = from_pretrained(model_id)
    _loaded_id = model_id

    # Run a silence transcription to warm MLX JIT — avoids a 2-5s
    # compilation hit on the first real transcription.
    tmp = tempfile.NamedTemporaryFile(
        prefix=TEMP_PREFIX, suffix=".wav", delete=False
    )
    silence = np.zeros(8000, dtype=np.float32)
    sf.write(tmp.name, silence, 16000)
    tmp.close()

    try:
        _model.transcribe(tmp.name)
    finally:
        safe_unlink(tmp.name)

    log.info("Parakeet model loaded and warmed")


def transcribe(audio_path: str, model_id: str) -> str | None:
    """Run Parakeet on the audio file. Returns text or None."""
    global _model, _loaded_id

    if _model is None or _loaded_id != model_id:
        load(model_id)

    result = _model.transcribe(audio_path)

    if not result or not getattr(result, "text", None):
        return None

    text = result.text.strip()
    return text or None


def keep_warm_ping(model_id: str):
    """Transcribe silence to keep model weights in page cache."""
    import tempfile

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
        transcribe(tmp.name, model_id)
    finally:
        safe_unlink(tmp.name)
