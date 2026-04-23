"""LLM-based text polisher — optional post-processing via a local MLX model.

Enabled by setting VOICECLIP_POLISH=true. Default off.

Uses the paste-first-polish-after pattern:
1. Raw formatted text is pasted immediately (no delay)
2. This module runs in the background to polish the text
3. The polished text replaces the original via select-all + paste

The LLM model is loaded lazily on first use and cached in memory.
Uses mlx-lm for Apple Silicon GPU inference.
"""

import logging
import os
import threading

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Toggle: set VOICECLIP_POLISH=true to enable
POLISH_ENABLED = os.environ.get("VOICECLIP_POLISH", "false").lower() == "true"

# Model to use for polishing — small and fast
POLISH_MODEL = os.environ.get(
    "VOICECLIP_POLISH_MODEL",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
)

# Max tokens to generate (polished text shouldn't be much longer than input)
POLISH_MAX_TOKENS = 512

# Timeout for polish operation (seconds)
POLISH_TIMEOUT = 15

# System prompt for the polisher
_SYSTEM_PROMPT = (
    "You are a text formatter. Fix punctuation, capitalization, and grammar "
    "in the following dictated text. Keep the original meaning and words. "
    "Do not add new content. Do not explain. Output only the corrected text."
)

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_load_lock = threading.Lock()
_available = None  # None = not checked, True/False = checked


def is_available() -> bool:
    """Check if mlx-lm is installed and polishing is enabled."""
    global _available
    if _available is not None:
        return _available

    if not POLISH_ENABLED:
        _available = False
        return False

    try:
        import mlx_lm  # noqa: F401
        _available = True
        log.info("LLM polish enabled (model: %s)", POLISH_MODEL)
    except ImportError:
        _available = False
        log.warning(
            "VOICECLIP_POLISH=true but mlx-lm is not installed. "
            "Install with: pip install mlx-lm"
        )
    return _available


def _ensure_model():
    """Load the model on first use. Thread-safe."""
    global _model, _tokenizer
    if _model is not None:
        return True

    with _load_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return True

        try:
            from mlx_lm import load
            log.info("Loading polish model: %s ...", POLISH_MODEL)
            _model, _tokenizer = load(POLISH_MODEL)
            log.info("Polish model loaded")
            return True
        except Exception as e:
            log.error("Failed to load polish model: %s", e)
            return False


def preload_polish_model():
    """Preload the polish model at startup (optional, called from __main__)."""
    if not is_available():
        return
    _ensure_model()


# ---------------------------------------------------------------------------
# Polish API
# ---------------------------------------------------------------------------

def polish(text: str) -> str | None:
    """Polish text using the local LLM.

    Returns the polished text, or None if polishing fails or times out.
    The caller should fall back to the original text on None.
    """
    if not is_available() or not text:
        return None

    if not _ensure_model():
        return None

    result_box = [None]

    def _do_polish():
        try:
            from mlx_lm import generate

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]

            prompt = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            polished = generate(
                _model,
                _tokenizer,
                prompt=prompt,
                max_tokens=POLISH_MAX_TOKENS,
                verbose=False,
            )

            # Clean up: strip whitespace and any quotes the model might add
            polished = polished.strip().strip('"').strip("'").strip()

            # Sanity check: if the model returned something wildly different
            # in length, it probably hallucinated — reject it
            if polished and 0.3 < len(polished) / max(len(text), 1) < 3.0:
                result_box[0] = polished
            else:
                log.warning(
                    "Polish result rejected (length ratio: %.1f)",
                    len(polished) / max(len(text), 1),
                )
        except Exception as e:
            log.error("Polish error: %s", e)

    worker = threading.Thread(target=_do_polish, daemon=True)
    worker.start()
    worker.join(timeout=POLISH_TIMEOUT)

    if worker.is_alive():
        log.warning("Polish timed out after %ds", POLISH_TIMEOUT)
        return None

    return result_box[0]
