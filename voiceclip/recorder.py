"""
Audio recorder — runs in a separate child process to avoid cffi conflicts
between pynput (keyboard listener) and sounddevice (audio capture).

The child process keeps the audio stream open permanently for zero-latency
start/stop. Communication uses multiprocessing.Pipe with a typed protocol.

Performance notes:
- Stream stays open permanently — zero startup cost on record
- Running RMS tracked during recording — instant silence detection on stop
- Resampling index arrays pre-computed once — no per-stop allocation
- Audio data sent via pipe as bytes — no temp file disk round-trip
"""

import logging
import math
import os
import tempfile
import threading
import time
import multiprocessing

from voiceclip.config import (
    RecorderCmd, SAMPLE_RATE, SILENCE_RMS_THRESHOLD,
    MIN_AUDIO_DURATION, TEMP_PREFIX,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

def _recorder_loop(conn):
    """Entry point for the recorder child process."""
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    rec_event = threading.Event()
    frames_lock = threading.Lock()
    frames = []

    # Running RMS state — updated in the callback for instant silence detection
    _rms_sum = [0.0]
    _rms_count = [0]

    try:
        default_dev = sd.default.device[0]
        dev_info = sd.query_devices(default_dev)
        native_sr = int(dev_info["default_samplerate"])
    except Exception as e:
        conn.send(f"error:device_query:{e}")
        return

    # Pre-compute resampling indices once (ratio is constant for this device)
    need_resample = native_sr != SAMPLE_RATE
    resample_ratio = SAMPLE_RATE / native_sr if need_resample else 1.0

    def callback(indata, frame_count, time_info, status):
        if rec_event.is_set():
            with frames_lock:
                frames.append(indata.copy())
            # Track running RMS (sum of squares) without holding the lock long
            sq_sum = float(np.sum(indata ** 2))
            _rms_sum[0] += sq_sum
            _rms_count[0] += indata.shape[0]

    try:
        stream = sd.InputStream(
            samplerate=native_sr,
            channels=1,
            dtype="float32",
            device=default_dev,
            callback=callback,
        )
        stream.start()
    except Exception as e:
        conn.send(f"error:stream_open:{e}")
        return

    conn.send("ready")

    while True:
        try:
            raw_msg = conn.recv()
        except (EOFError, OSError):
            break

        if isinstance(raw_msg, RecorderCmd):
            msg = raw_msg
        else:
            try:
                msg = RecorderCmd(raw_msg)
            except ValueError:
                conn.send(None)
                continue

        if msg == RecorderCmd.START:
            with frames_lock:
                frames.clear()
            _rms_sum[0] = 0.0
            _rms_count[0] = 0
            rec_event.set()
            conn.send("ok")

        elif msg == RecorderCmd.STOP:
            rec_event.clear()

            # Grab frames under lock — no sleep needed, event is already cleared
            # so no new frames will be appended
            with frames_lock:
                if not frames:
                    conn.send(None)
                    continue
                captured = list(frames)
                frames.clear()

            # Fast silence check using running RMS (no recomputation needed)
            total_samples = _rms_count[0]
            if total_samples > 0:
                rms = math.sqrt(_rms_sum[0] / total_samples)
            else:
                rms = 0.0

            duration_native = total_samples / native_sr
            if duration_native < MIN_AUDIO_DURATION or rms < SILENCE_RMS_THRESHOLD:
                conn.send(None)
                continue

            audio = np.concatenate(captured, axis=0).flatten()

            # Resample to 16kHz if needed (linear interpolation)
            if need_resample:
                new_len = int(math.ceil(len(audio) * resample_ratio))
                old_idx = np.arange(new_len) / resample_ratio
                old_idx = np.clip(old_idx, 0, len(audio) - 1)
                floor_idx = np.floor(old_idx).astype(np.int32)
                ceil_idx = np.minimum(floor_idx + 1, len(audio) - 1)
                frac = (old_idx - floor_idx).astype(np.float32)
                audio = audio[floor_idx] * (1.0 - frac) + audio[ceil_idx] * frac

            # Write to temp file (mlx_whisper needs a file path)
            try:
                tmp = tempfile.NamedTemporaryFile(
                    prefix=TEMP_PREFIX, suffix=".wav", delete=False
                )
                sf.write(tmp.name, audio, SAMPLE_RATE)
                tmp.close()
                conn.send(tmp.name)
            except Exception:
                conn.send(None)

        elif msg == RecorderCmd.LIST_DEVICES:
            try:
                devices = sd.query_devices()
                lines = []
                for i, d in enumerate(devices):
                    if d["max_input_channels"] > 0:
                        marker = " ⭐ (Default)" if i == default_dev else ""
                        lines.append(f"    [{i}] {d['name']}{marker}")
                conn.send("\n".join(lines) if lines else "    No input devices found")
            except Exception as e:
                conn.send(f"    Error listing devices: {e}")

        elif msg == RecorderCmd.QUIT:
            break

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Recorder class — main-process interface (thread-safe)
# ---------------------------------------------------------------------------

class Recorder:
    """Controls the audio recorder child process."""

    def __init__(self):
        self._conn = None
        self._proc = None
        self._alive = False
        self._pipe_lock = threading.Lock()

    def start(self):
        """Spawn the child process and wait for it to be ready."""
        parent_conn, child_conn = multiprocessing.Pipe()
        self._proc = multiprocessing.Process(
            target=_recorder_loop, args=(child_conn,), daemon=True
        )
        self._proc.start()
        self._conn = parent_conn
        self._alive = True

        if not self._conn.poll(10):
            log.error("Recorder timed out during startup")
            raise RuntimeError("Recorder timed out during startup")

        msg = self._conn.recv()
        if msg != "ready":
            log.error("Recorder failed: %s", msg)
            raise RuntimeError(f"Recorder failed: {msg}")

        log.info("Recorder process started (pid=%d)", self._proc.pid)

    def shutdown(self):
        """Gracefully shut down the child process."""
        self._alive = False
        if self._conn:
            try:
                self._conn.send(RecorderCmd.QUIT)
            except (BrokenPipeError, OSError):
                pass
        if self._proc:
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.terminate()
                log.warning("Recorder terminated forcefully")

    def restart(self):
        """Shut down and respawn the recorder."""
        log.warning("Restarting recorder...")
        self.shutdown()
        self.start()

    @property
    def alive(self):
        return self._alive

    def _send_recv(self, cmd, timeout=5):
        """Send a command and return the response. Thread-safe."""
        if not self._alive:
            raise RuntimeError("Recorder is not running")
        with self._pipe_lock:
            try:
                self._conn.send(cmd)
                if self._conn.poll(timeout):
                    return self._conn.recv()
                else:
                    self._alive = False
                    raise RuntimeError("Recorder timed out")
            except (BrokenPipeError, EOFError, OSError) as e:
                self._alive = False
                raise RuntimeError(f"Recorder died: {e}")

    def begin(self):
        """Start capturing audio."""
        self._send_recv(RecorderCmd.START, timeout=3)

    def end(self):
        """Stop capturing and return the WAV file path, or None."""
        path = self._send_recv(RecorderCmd.STOP, timeout=10)
        if not path:
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        if size < 500:
            log.warning("Recording too small (%d bytes)", size)
            _safe_unlink(path)
            return None
        log.info("Recorded %.1f KB", size / 1024)
        return path

    def list_devices(self):
        """Return a formatted string of available input devices."""
        return self._send_recv(RecorderCmd.LIST_DEVICES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_unlink(path):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def cleanup_stale_temps():
    """Remove leftover VoiceClip temp files from previous runs."""
    tmp_dir = tempfile.gettempdir()
    try:
        for f in os.listdir(tmp_dir):
            if f.startswith(TEMP_PREFIX) and f.endswith(".wav"):
                fpath = os.path.join(tmp_dir, f)
                try:
                    age = time.time() - os.path.getmtime(fpath)
                    if age > 3600:
                        os.unlink(fpath)
                except OSError:
                    pass
    except OSError:
        pass
