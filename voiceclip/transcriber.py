"""Whisper transcription via mlx-whisper on Apple Silicon GPU.

Performance notes:
- Model is preloaded at import time to eliminate first-transcription delay
- get_model_repo() is called once and cached
- Transcription has a timeout watchdog to prevent permanent hangs
"""

import logging
import tempfile
import threading

import mlx_whisper
import numpy as np
import soundfile as sf

from voiceclip.config import ENGLISH_ONLY, INITIAL_PROMPT, TEMP_PREFIX, get_model_repo
from voiceclip.utils import safe_unlink

log = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when transcription fails in a way the user should know about.

    Distinct from `transcribe(...)` returning None, which means "audio was
    processed but no speech was detected." An exception means "something
    actually broke" — timeout, model crash, missing file — and the caller
    should surface a notification with the error message.
    """


# Pre-resolve the model repo at import time (avoids per-call overhead)
_REPO, _MODEL_KEY = get_model_repo()

# Timeout for transcription calls (seconds). If mlx_whisper hangs beyond
# this, the watchdog thread logs an error and returns None so the hotkey
# coordinator frees up. The C-extension worker can't be killed, but 30s is
# a reasonable ceiling for interactive dictation — typical end-to-end on
# Apple Silicon for a 10s clip is 1-3s, and the 99th percentile is well
# under 15s. Long clips are already capped by MAX_RECORDING_SECONDS.
TRANSCRIBE_TIMEOUT = 30


def _is_model_cached() -> bool:
    """Check if the model is already downloaded in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check for a key file that indicates the model is downloaded
        result = try_to_load_from_cache(_REPO, "config.json")
        return result is not None
    except Exception:
        # Can't determine — assume not cached (will show spinner)
        return False


def preload_model():
    """Force-load the Whisper model into memory so the first transcription is fast.

    Call this at startup. mlx_whisper caches internally, so subsequent
    transcribe() calls reuse the loaded model.

    Shows a spinner in the terminal during loading so the user knows
    the app hasn't frozen (especially important during first-run download).
    """
    log.info("Preloading model: %s (%s)...", _MODEL_KEY, _REPO)

    cached = _is_model_cached()
    if not cached:
        print("  ⬇️  Downloading model (this only happens once)...")

    # Spinner runs in a background thread while preload happens
    stop_spinner = threading.Event()

    def _spin():
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        label = "Downloading & loading" if not cached else "Loading"
        while not stop_spinner.is_set():
            print(f"\r  {frames[i % len(frames)]} {label} model...", end="", flush=True)
            i += 1
            stop_spinner.wait(0.1)
        # Clear the spinner line
        print("\r" + " " * 50 + "\r", end="", flush=True)

    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()

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
    finally:
        stop_spinner.set()
        spinner.join(timeout=1)


def transcribe(audio_path):
    """Transcribe a WAV file and return the text, or None.

    Deletes the audio file after transcription regardless of outcome.
    Includes a timeout watchdog — if mlx_whisper hangs beyond
    TRANSCRIBE_TIMEOUT seconds, raises TranscriptionError so the hotkey
    handler can notify the user. Unexpected exceptions propagate the same
    way — the only silent-None path is "no speech detected" (legitimate
    empty output), which is why the caller distinguishes `None` (silence)
    from a raised exception (failure to recover from).
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
            if INITIAL_PROMPT:
                kwargs["initial_prompt"] = INITIAL_PROMPT

            result_box[0] = mlx_whisper.transcribe(audio_path, **kwargs)
        except Exception as e:
            error_box[0] = e

    worker = threading.Thread(target=_do_transcribe, daemon=True)
    worker.start()
    worker.join(timeout=TRANSCRIBE_TIMEOUT)

    if worker.is_alive():
        log.error(
            "Transcription timed out after %ds — model may be hung.",
            TRANSCRIBE_TIMEOUT,
        )
        # Can't kill the thread (C extension). The temp file will be
        # cleaned up on next startup. Raising — not returning None —
        # so the caller can tell "timed out" from "no speech detected".
        raise TranscriptionError(
            f"Transcription timed out after {TRANSCRIBE_TIMEOUT}s. "
            "The Whisper model may be stuck. Run `voiceclip doctor` to check "
            "your setup; if this keeps happening, restart VoiceClip."
        )

    # Clean up temp file now that transcription is done
    safe_unlink(audio_path)

    if error_box[0]:
        log.error("Transcription error: %s", error_box[0])
        raise TranscriptionError(
            f"Transcription failed: {type(error_box[0]).__name__}: {error_box[0]}. "
            "Run `voiceclip doctor` to diagnose."
        ) from error_box[0]

    result = result_box[0]
    if not result:
        return None

    segments = result.get("segments", [])
    real = [s for s in segments if s.get("no_speech_prob", 0) < 0.7]

    if not real:
        return None

    text = " ".join(s["text"].strip() for s in real).strip()
    return text or None


# ---------------------------------------------------------------------------
# Model keep-warm — prevents swap-out during idle periods
# ---------------------------------------------------------------------------
# macOS aggressively pages out memory that hasn't been touched recently.
# The Whisper model weights (~3GB for large-v3-turbo) get swapped to disk
# after a few minutes of inactivity, making the next dictation pay a 5-15s
# penalty to page everything back in.
#
# The fix: a background thread that runs a trivial transcription (0.5s of
# silence) every few minutes. This touches the model weights just enough
# to keep them in the OS page cache without producing any visible output
# or meaningful GPU load (~50ms per ping).
#
# The thread is daemon=True so it dies automatically when VoiceClip exits.
# Failures are swallowed silently — if a keep-warm ping fails, the worst
# case is the user gets one slow dictation, same as before this feature.

_KEEP_WARM_INTERVAL_SECONDS = 300  # 5 minutes

_keep_warm_thread: threading.Thread | None = None
_keep_warm_stop = threading.Event()

# Pre-generate the silent WAV once at module level rather than creating
# and deleting a temp file every 5 minutes.
_SILENCE_PATH: str | None = None


def _ensure_silence_file() -> str:
    """Create (once) a tiny silent WAV for keep-warm pings. Returns the path."""
    global _SILENCE_PATH
    if _SILENCE_PATH is not None:
        import os
        if os.path.exists(_SILENCE_PATH):
            return _SILENCE_PATH

    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        prefix="voiceclip_keepwarm_", suffix=".wav", delete=False,
    )
    silence = np.zeros(8000, dtype=np.float32)  # 0.5s at 16kHz
    sf.write(tmp.name, silence, 16000)
    tmp.close()
    _SILENCE_PATH = tmp.name
    return _SILENCE_PATH


def _keep_warm_loop():
    """Background loop: transcribe silence every N seconds to keep the
    model weights resident in memory."""
    while not _keep_warm_stop.is_set():
        _keep_warm_stop.wait(_KEEP_WARM_INTERVAL_SECONDS)
        if _keep_warm_stop.is_set():
            return
        try:
            path = _ensure_silence_file()
            mlx_whisper.transcribe(
                path,
                path_or_hf_repo=_REPO,
                language="en",
                no_speech_threshold=0.6,
            )
            log.debug("Keep-warm ping completed")
        except Exception as e:
            # Swallow — a failed ping just means one potentially slow
            # dictation, same as before this feature existed.
            log.debug("Keep-warm ping failed (harmless): %s", e)


def start_keep_warm():
    """Start the background keep-warm thread. Call once after preload_model().

    Idempotent — safe to call multiple times (only one thread runs).
    """
    global _keep_warm_thread
    if _keep_warm_thread is not None and _keep_warm_thread.is_alive():
        return
    _keep_warm_stop.clear()
    _keep_warm_thread = threading.Thread(
        target=_keep_warm_loop,
        name="voiceclip-keep-warm",
        daemon=True,
    )
    _keep_warm_thread.start()
    log.info(
        "Model keep-warm started (pings every %ds to prevent swap-out)",
        _KEEP_WARM_INTERVAL_SECONDS,
    )


def stop_keep_warm():
    """Stop the keep-warm thread. Called on shutdown for clean exit."""
    global _keep_warm_thread
    _keep_warm_stop.set()
    if _keep_warm_thread is not None:
        _keep_warm_thread.join(timeout=2)
        _keep_warm_thread = None
    # Clean up the silence file
    if _SILENCE_PATH:
        safe_unlink(_SILENCE_PATH)
