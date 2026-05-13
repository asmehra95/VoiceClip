"""
Audio recorder — runs in a separate child process to avoid cffi conflicts
between pynput (keyboard listener) and sounddevice (audio capture).

The child process keeps the audio stream open permanently for zero-latency
start/stop. Communication uses multiprocessing.Pipe with a typed protocol.

Performance notes:
- Stream stays open permanently — zero startup cost on record
- Running RMS tracked during recording — instant silence detection on stop
- Resampling index arrays pre-computed once — no per-stop allocation
- Audio written to temp WAV file for mlx_whisper (requires file path)
"""

import logging
import math
import multiprocessing
import os
import tempfile
import threading
import time

from voiceclip.config import (
    MAX_RECORDING_SECONDS,
    MIN_AUDIO_DURATION,
    MIN_FILE_BYTES,
    SAMPLE_RATE,
    SILENCE_RMS_THRESHOLD,
    TEMP_PREFIX,
    RecorderCmd,
)
from voiceclip.utils import safe_unlink

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

def _recorder_loop(conn):
    """Entry point for the recorder child process."""
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    rec_event = threading.Event()
    frames_lock = threading.Lock()
    frames = []

    # Running RMS state — updated in the callback for instant silence detection
    # Using a lock to prevent data races between the callback and STOP handler
    _rms_lock = threading.Lock()
    _rms_sum = [0.0]
    _rms_count = [0]

    # Track the active device so we can detect mid-session switches.
    # macOS can silently change the default input device (e.g. AirPods
    # connect/disconnect, phone call starts) — when that happens the old
    # stream keeps running but receives silence. We check on every START
    # and reopen the stream if the default device changed.
    _active_device = [None]  # [device_index]
    _stream_box = [None]     # [sd.InputStream]
    _native_sr_box = [0]
    _need_resample_box = [False]
    _resample_ratio_box = [1.0]
    _max_samples_box = [0]

    # Timestamp of the last callback invocation. Used to detect streams
    # that are "open" but no longer receiving audio (e.g. after macOS
    # sleep/wake). If the callback hasn't fired in a while, we force a
    # stream reopen on the next START.
    import time as _time
    _last_callback_time = [_time.time()]
    _STREAM_STALE_SECONDS = 30  # if no callback in this window, stream is dead

    def _open_stream(force_device=None):
        """Open (or reopen) the input stream on the current default device.

        Returns True on success, False on failure. Updates all the
        mutable state boxes that the callback and STOP handler read.
        """
        nonlocal need_resample, resample_ratio, max_samples

        try:
            target_dev = force_device if force_device is not None else sd.default.device[0]
            dev_info = sd.query_devices(target_dev)
            native_sr = int(dev_info["default_samplerate"])
            dev_name = dev_info.get("name", "Unknown")
            print(
                f"[recorder] Device: {dev_name} (idx={target_dev}, "
                f"sr={native_sr}Hz, ch={dev_info['max_input_channels']})",
                flush=True,
            )
            if native_sr < 16000:
                print(
                    f"[recorder] ⚠️  Low sample rate ({native_sr}Hz). "
                    "Bluetooth HFP mode? Audio quality will be degraded. "
                    "Consider using the MacBook mic instead.",
                    flush=True,
                )
        except Exception as e:
            print(f"[recorder] ⚠️  Device query failed: {e}", flush=True)
            return False

        # Close old stream if any
        old_stream = _stream_box[0]
        if old_stream is not None:
            try:
                old_stream.stop()
                old_stream.close()
            except Exception:
                pass

        # Update resampling state
        need_resample = native_sr != SAMPLE_RATE
        resample_ratio = SAMPLE_RATE / native_sr if need_resample else 1.0
        max_samples = int(MAX_RECORDING_SECONDS * native_sr)

        _native_sr_box[0] = native_sr
        _need_resample_box[0] = need_resample
        _resample_ratio_box[0] = resample_ratio
        _max_samples_box[0] = max_samples
        _active_device[0] = target_dev

        try:
            new_stream = sd.InputStream(
                samplerate=native_sr,
                channels=1,
                dtype="float32",
                device=target_dev,
                callback=callback,
            )
            new_stream.start()
            _stream_box[0] = new_stream
            return True
        except Exception as e:
            print(f"[recorder] ⚠️  Stream open failed: {e}", flush=True)
            _stream_box[0] = None
            return False

    def _check_device_change() -> bool:
        """If the default input device changed, or the stream has gone
        stale (no callbacks in _STREAM_STALE_SECONDS), reopen.
        Called on every START command. Returns False if the stream is dead.
        """
        try:
            current_default = sd.default.device[0]
        except Exception:
            return _stream_box[0] is not None

        device_changed = current_default != _active_device[0]
        stream_stale = (
            _time.time() - _last_callback_time[0] > _STREAM_STALE_SECONDS
        )

        if device_changed:
            print(
                f"[recorder] ⚠️  Default input device changed "
                f"({_active_device[0]} → {current_default}). "
                "Reopening stream on new device...",
                flush=True,
            )
            if not _open_stream(force_device=current_default):
                return False
        elif stream_stale:
            print(
                "[recorder] ⚠️  Audio stream stale (no callbacks in "
                f"{_STREAM_STALE_SECONDS}s — likely sleep/wake). "
                "Reopening stream...",
                flush=True,
            )
            if not _open_stream(force_device=current_default):
                return False

        return True

    # Initialize mutable state used by callback before defining it
    need_resample = False
    resample_ratio = 1.0
    max_samples = int(MAX_RECORDING_SECONDS * SAMPLE_RATE)

    # Log a warning exactly once per recording when the cap trips.
    _cap_hit = [False]

    def callback(indata, frame_count, time_info, status):
        _last_callback_time[0] = _time.time()
        if status:
            # sounddevice reports input overflow, device disconnected, etc.
            print(f"[recorder] ⚠️  Stream status: {status}", flush=True)
        if rec_event.is_set():
            with _rms_lock:
                already = _rms_count[0]
            if already >= _max_samples_box[0]:
                if not _cap_hit[0]:
                    _cap_hit[0] = True
                    print(
                        f"[recorder] ⚠️  Recording hit {MAX_RECORDING_SECONDS}s cap; "
                        "dropping new audio. Release the hotkey to transcribe.",
                        flush=True,
                    )
                return
            with frames_lock:
                frames.append(indata.copy())
            # Track running RMS — dot product is allocation-free and ~2x faster
            flat = indata.flat
            sq_sum = float(np.dot(flat, flat))
            with _rms_lock:
                _rms_sum[0] += sq_sum
                _rms_count[0] += indata.shape[0]

    # Initial stream open
    try:
        default_dev = sd.default.device[0]
    except Exception as e:
        conn.send(f"error:device_query:{e}")
        return

    if not _open_stream(force_device=default_dev):
        conn.send(f"error:stream_open:could not open initial stream")
        return

    native_sr = _native_sr_box[0]

    conn.send("ready")

    while True:
        try:
            raw_msg = conn.recv()
        except (EOFError, OSError, KeyboardInterrupt):
            # Ctrl+C on macOS (and the Linux default) delivers SIGINT to the
            # whole foreground process group, so the child gets interrupted
            # mid-recv() before the parent can send QUIT. Treat KeyboardInterrupt
            # the same as "parent is going away" — fall through to the stream
            # teardown below. Without this, multiprocessing dumps a full
            # traceback every time the user hits Ctrl+C.
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
            # Check if the default device changed since last recording.
            # This catches AirPods connect/disconnect, Bluetooth HFP
            # switches, and other mid-session device changes that would
            # otherwise cause the stream to capture silence.
            if not _check_device_change():
                print("[recorder] ⚠️  No audio stream available", flush=True)
                conn.send("error:no_stream")
                continue

            with frames_lock:
                frames.clear()
            with _rms_lock:
                _rms_sum[0] = 0.0
                _rms_count[0] = 0
            _cap_hit[0] = False
            rec_event.set()
            conn.send("ok")

        elif msg == RecorderCmd.STOP:
            rec_event.clear()

            # Grab frames under lock — no sleep needed, event is already cleared
            # so no new frames will be appended
            with frames_lock:
                if not frames:
                    print("[recorder] STOP: no frames captured", flush=True)
                    conn.send(None)
                    continue
                captured = list(frames)
                frames.clear()

            # Fast silence check using running RMS (no recomputation needed)
            with _rms_lock:
                total_samples = _rms_count[0]
                rms_sum_val = _rms_sum[0]

            if total_samples > 0:
                rms = math.sqrt(rms_sum_val / total_samples)
            else:
                rms = 0.0

            # Use the current native_sr (may have changed if device switched)
            cur_native_sr = _native_sr_box[0]
            duration_native = total_samples / cur_native_sr if cur_native_sr > 0 else 0.0
            print(
                f"[recorder] STOP: {len(captured)} frames, "
                f"{duration_native:.2f}s, RMS={rms:.6f}, "
                f"threshold={SILENCE_RMS_THRESHOLD}",
                flush=True,
            )

            if duration_native < MIN_AUDIO_DURATION or rms < SILENCE_RMS_THRESHOLD:
                reason = "too short" if duration_native < MIN_AUDIO_DURATION else "silence"
                print(f"[recorder] Rejected: {reason}", flush=True)
                conn.send(None)
                continue

            audio = np.concatenate(captured, axis=0).flatten()

            # Resample to 16kHz if needed (linear interpolation)
            cur_need_resample = _need_resample_box[0]
            cur_resample_ratio = _resample_ratio_box[0]
            if cur_need_resample:
                new_len = int(math.ceil(len(audio) * cur_resample_ratio))
                old_idx = np.arange(new_len) / cur_resample_ratio
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

        elif msg == RecorderCmd.QUIT:
            break

    stream = _stream_box[0]
    if stream is not None:
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
        """Gracefully shut down the child process.

        Timing notes:
        - 5s join timeout. The child has to close the CoreAudio input
          stream, which can take 1-3s on Bluetooth / Continuity devices.
          The previous 2s timeout was racing that teardown, logging a
          scary "Recorder terminated forcefully" WARNING for a child
          that was milliseconds away from exiting cleanly on its own.
        - If the join still times out after 5s, something is genuinely
          wrong — terminate() and log at debug level. The user already
          saw "👋 VoiceClip stopped."; flooding the final line with
          WARNING is worse than silently cleaning up.
        """
        self._alive = False
        if self._conn:
            try:
                self._conn.send(RecorderCmd.QUIT)
            except (BrokenPipeError, OSError):
                pass
        if self._proc:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
                log.debug("Recorder did not exit within 5s; terminated")

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
                raise RuntimeError(f"Recorder died: {e}") from e

    def begin(self):
        """Start capturing audio. Raises RuntimeError if the stream is dead."""
        resp = self._send_recv(RecorderCmd.START, timeout=3)
        if isinstance(resp, str) and resp.startswith("error:"):
            raise RuntimeError(f"Recorder: {resp}")

    def end(self):
        """Stop capturing and return the WAV file path, or None."""
        path = self._send_recv(RecorderCmd.STOP, timeout=10)
        if not path:
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        if size < MIN_FILE_BYTES:
            log.warning("Recording too small (%d bytes)", size)
            safe_unlink(path)
            return None
        log.info("Recorded %.1f KB", size / 1024)
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
