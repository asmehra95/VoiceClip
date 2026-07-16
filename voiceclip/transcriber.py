"""Transcription dispatcher — routes to the active engine (whisper or parakeet).

Public API consumed by hotkey.py and __main__.py:
  preload_model()     → load the configured engine's model at startup
  transcribe(path)    → text or None (raises TranscriptionError on failure)
  start_keep_warm()   → background thread to keep model resident
  stop_keep_warm()    → clean shutdown
"""

import logging
import threading
import time

from voiceclip import config
from voiceclip.utils import safe_unlink

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine resolution — resolved once on first use, called uniformly everywhere.
# Serialized by _engine_lock so concurrent threads can't double-init or
# invoke the engine simultaneously (MLX is not thread-safe).
# ---------------------------------------------------------------------------

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        if config.ENGINE == "parakeet":
            from voiceclip import engine_parakeet as mod
        elif config.ENGINE == "whisper_cpp":
            from voiceclip import engine_whisper_cpp as mod
        else:
            from voiceclip import engine_whisper as mod
        _engine = mod
        return _engine


def _reset_engine():
    """Test hook — invalidate the cached engine module."""
    global _engine
    with _engine_lock:
        _engine = None


class TranscriptionError(RuntimeError):
    """Raised when transcription fails in a way the user should know about."""


_TIMEOUT = {"whisper": 30, "whisper_cpp": 60, "parakeet": 60}
# whisper_cpp: the server keeps the model in its process, but macOS still
# pages the weights out after idle — ping periodically to keep them resident.
_KEEP_WARM_INTERVAL = {"whisper": 300, "whisper_cpp": 300, "parakeet": 600}


def _engine_repo() -> str:
    """Return the model identifier for the current engine."""
    if config.ENGINE == "parakeet":
        return config.PARAKEET_MODEL
    elif config.ENGINE == "whisper_cpp":
        # For whisper_cpp, return the model name (e.g. "large-v3")
        # which engine_whisper_cpp resolves to a GGML file path
        return config.MODEL
    else:
        repo, _key = config.get_model_repo()
        return repo


def _is_model_cached() -> bool:
    """Check if the model is already downloaded in the HuggingFace cache."""
    if config.ENGINE == "whisper_cpp":
        # whisper_cpp uses local GGML files, not HuggingFace cache
        from voiceclip.engine_whisper_cpp import _resolve_model_path
        try:
            _resolve_model_path(config.MODEL)
            return True
        except FileNotFoundError:
            return False
    try:
        from huggingface_hub import try_to_load_from_cache
        repo = _engine_repo()
        result = try_to_load_from_cache(repo, "config.json")
        return result is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Preload
# ---------------------------------------------------------------------------

def preload_model():
    """Force-load the configured engine's model into memory."""
    repo = _engine_repo()
    log.info("Preloading engine=%s, model=%s", config.ENGINE, repo)

    cached = _is_model_cached()
    if not cached:
        print("  ⬇️  Downloading model (this only happens once)...")

    # Suppress httpx INFO logging that interleaves with download progress.
    httpx_logger = logging.getLogger("httpx")
    prev_level = httpx_logger.level
    httpx_logger.setLevel(logging.WARNING)

    try:
        if config.ENGINE == "parakeet":
            # Parakeet TDT models bind the MLX GPU stream to the calling
            # thread. Loading on the main thread prevents all worker threads
            # from running inference. Skip preload; the first transcribe()
            # call will lazy-load on its worker thread instead.
            log.info("Parakeet: skipping main-thread preload (lazy-load on first use)")
            return True
        _get_engine().load(repo)
        log.info("Model preloaded successfully")
        return True
    except Exception as e:
        log.warning("Model preload failed (will load on first use): %s", e)
        return False
    finally:
        httpx_logger.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Transcribe
# ---------------------------------------------------------------------------

def transcribe(audio_path):
    """Transcribe a WAV file and return the text, or None.

    Deletes the audio file after transcription. Raises TranscriptionError
    on timeout or engine failure.
    """
    if not audio_path:
        return None

    timeout = _TIMEOUT.get(config.ENGINE, 30)
    repo = _engine_repo()
    engine = _get_engine()

    result_box = [None]
    error_box = [None]

    def _do_transcribe():
        try:
            result_box[0] = engine.transcribe(audio_path, repo)
        except Exception as e:
            error_box[0] = e

    # Acquire the lock on THIS thread so it is always released — even on
    # timeout. The worker runs inference without holding the lock; the lock
    # just prevents concurrent engine use (keep-warm vs transcription).
    _engine_lock.acquire()
    try:
        worker = threading.Thread(target=_do_transcribe, daemon=True)
        worker.start()
        worker.join(timeout=timeout)
    finally:
        _engine_lock.release()

    if worker.is_alive():
        log.error("Transcription timed out after %ds", timeout)
        safe_unlink(audio_path)
        raise TranscriptionError(
            f"Transcription timed out after {timeout}s. "
            "The model may be stuck. Run `voiceclip doctor` to check "
            "your setup; if this keeps happening, restart VoiceClip."
        )

    safe_unlink(audio_path)

    global _last_model_use
    _last_model_use = time.time()

    if error_box[0]:
        log.error("Transcription error: %s", error_box[0])
        raise TranscriptionError(
            f"Transcription failed: {type(error_box[0]).__name__}: {error_box[0]}. "
            "Run `voiceclip doctor` to diagnose."
        ) from error_box[0]

    return result_box[0]


# ---------------------------------------------------------------------------
# Keep-warm
# ---------------------------------------------------------------------------

_keep_warm_thread: threading.Thread | None = None
_keep_warm_stop = threading.Event()
_last_model_use: float = 0.0


def _keep_warm_loop():
    """Background loop: ping the engine to keep model weights resident."""
    global _last_model_use
    interval = _KEEP_WARM_INTERVAL.get(config.ENGINE, 300)
    repo = _engine_repo()

    while not _keep_warm_stop.is_set():
        _keep_warm_stop.wait(interval)
        if _keep_warm_stop.is_set():
            return
        elapsed = time.time() - _last_model_use
        if elapsed < interval:
            log.debug("Keep-warm skipped (model used %.0fs ago)", elapsed)
            continue
        try:
            engine = _get_engine()
            with _engine_lock:
                engine.keep_warm_ping(repo)
            _last_model_use = time.time()
            log.debug("Keep-warm ping completed")
        except Exception:
            pass


def start_keep_warm():
    """Start the background keep-warm thread. Idempotent."""
    global _keep_warm_thread
    interval = _KEEP_WARM_INTERVAL.get(config.ENGINE, 300)
    if interval <= 0:
        log.info("Keep-warm disabled for engine=%s", config.ENGINE)
        return
    if _keep_warm_thread is not None and _keep_warm_thread.is_alive():
        return
    _keep_warm_stop.clear()
    _keep_warm_thread = threading.Thread(
        target=_keep_warm_loop,
        name="voiceclip-keep-warm",
        daemon=True,
    )
    _keep_warm_thread.start()
    log.info("Keep-warm started (engine=%s, interval=%ds)", config.ENGINE, interval)


def stop_keep_warm():
    """Stop the keep-warm thread."""
    global _keep_warm_thread
    _keep_warm_stop.set()
    if _keep_warm_thread is not None:
        _keep_warm_thread.join(timeout=2)
        _keep_warm_thread = None
