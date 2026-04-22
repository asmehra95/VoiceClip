"""
VoiceClip — Local Voice-to-Clipboard for macOS

Hold the Right Option (⌥) key to record from your mic.
Release to transcribe with mlx-whisper and copy the text to your clipboard.

Usage:
    pip install mlx-whisper pynput
    python transcribe.py

Requirements:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - ffmpeg (brew install ffmpeg)
    - Accessibility permissions for your terminal app
      (System Settings → Privacy & Security → Accessibility)

Configuration (via environment variables):
    VOICECLIP_MODEL  — Whisper model to use (default: large-v3-turbo)
                       Options: tiny, base, small, medium, large-v3-turbo, large-v3
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

# ---------------------------------------------------------------------------
# Dependency checks — fail fast with helpful messages
# ---------------------------------------------------------------------------

_missing = []

try:
    import mlx_whisper
except ImportError:
    _missing.append("mlx-whisper")

try:
    from pynput import keyboard
except ImportError:
    _missing.append("pynput")

if _missing:
    print(f"Missing: {', '.join(_missing)}")
    print(f"Run: pip install {' '.join(_missing)}")
    sys.exit(1)

if not shutil.which("ffmpeg"):
    print("ffmpeg not found. Install with: brew install ffmpeg")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration — edit these or set env vars to customize
# ---------------------------------------------------------------------------

MODELS = {
    "tiny.en":        "mlx-community/whisper-tiny.en-mlx",
    "base.en":        "mlx-community/whisper-base.en-mlx",
    "small.en":       "mlx-community/whisper-small.en-mlx",
    "medium.en":      "mlx-community/whisper-medium.en-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
}

MODEL = os.environ.get("VOICECLIP_MODEL", "large-v3-turbo")
ENGLISH_ONLY = True
MIN_HOLD_SECONDS = 0.3   # Taps shorter than this are ignored
MIN_FILE_BYTES = 1000    # WAV files smaller than this are treated as empty
PASTE_DELAY = 0.15       # Seconds to wait between copy and paste

# Validate model at startup
if MODEL not in ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"):
    print(f"Unknown model: {MODEL}")
    print(f"Valid options: tiny, base, small, medium, large-v3-turbo, large-v3")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Recording — uses ffmpeg + AVFoundation to capture from the default mic.
# We use ffmpeg instead of sounddevice because pynput's cffi callbacks
# conflict with sounddevice's cffi callbacks, causing segfaults on macOS.
# ---------------------------------------------------------------------------

_ffmpeg_proc = None
_tmp_path = None
_recording = False
_rec_lock = threading.Lock()
_rec_ready = threading.Event()  # Signals that ffmpeg has been launched


def list_mics():
    """Print available AVFoundation audio input devices."""
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    audio_section = False
    for line in result.stderr.split("\n"):
        if "AVFoundation audio devices:" in line:
            audio_section = True
            continue
        if not audio_section:
            continue
        # Stop if we hit a non-device line (e.g. error or blank)
        if "] [" not in line:
            break
        parts = line.split("] ")
        if len(parts) >= 3:
            print(f"    [{parts[1].strip('[]')}] {parts[2].strip()}")


def start_recording():
    """Start recording from the default mic via ffmpeg."""
    global _ffmpeg_proc, _tmp_path, _recording

    _rec_ready.clear()

    with _rec_lock:
        if _recording:
            _rec_ready.set()
            return
        _recording = True

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            _ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg", "-y",
                    "-f", "avfoundation", "-i", ":default",
                    "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                    tmp_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _tmp_path = tmp_path
        except OSError as e:
            print(f"  ❌ Failed to start ffmpeg: {e}")
            _recording = False
            _tmp_path = None
            _safe_unlink(tmp_path)  # Fix #2: clean up leaked temp file

    # Small warmup for ffmpeg to initialize AVFoundation.
    # Most users naturally pause ~200-400ms between pressing the key
    # and starting to speak, which covers this.
    time.sleep(0.15)
    _rec_ready.set()


def stop_recording():
    """Stop ffmpeg and return the path to the recorded WAV, or None."""
    global _ffmpeg_proc, _recording, _tmp_path

    # Fix #1: wait for start_recording to finish launching ffmpeg
    _rec_ready.wait(timeout=3)

    # Grab references under lock, then release so we don't hold it during wait()
    with _rec_lock:
        _recording = False
        proc, path = _ffmpeg_proc, _tmp_path
        _ffmpeg_proc = None
        _tmp_path = None

    if proc:
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc.terminate()

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    time.sleep(0.1)  # Let the filesystem flush

    if path and os.path.exists(path):
        size = os.path.getsize(path)
        if size < MIN_FILE_BYTES:
            print(f"  ⚠️  Recording too small ({size} bytes), discarding")
            _safe_unlink(path)
            return None
        print(f"  📁 Recorded {size / 1024:.1f} KB")
        return path

    return None


# ---------------------------------------------------------------------------
# Transcription — mlx-whisper runs on the Apple Silicon GPU via Metal
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path):
    """Transcribe a WAV file and return the text, or None. Deletes the file after."""
    if not audio_path:
        return None

    try:
        # Pick the .en variant for English-only mode (faster, more accurate)
        if ENGLISH_ONLY and MODEL in ("tiny", "base", "small", "medium"):
            model_key = f"{MODEL}.en"
        else:
            model_key = MODEL

        repo = MODELS.get(model_key, MODELS[MODEL])

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=repo,
            language="en" if ENGLISH_ONLY else None,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )

        # Filter out segments Whisper hallucinated on silence
        segments = result.get("segments", [])
        real = [s for s in segments if s.get("no_speech_prob", 0) < 0.7]

        if not real:
            return None

        text = " ".join(s["text"].strip() for s in real).strip()
        return text or None

    except Exception as e:
        print(f"  ❌ Transcription error: {e}")
        return None
    finally:
        _safe_unlink(audio_path)


# ---------------------------------------------------------------------------
# macOS utilities — clipboard, paste, notifications, sounds
# ---------------------------------------------------------------------------

def copy_to_clipboard(text):
    """Copy text to the macOS clipboard via pbcopy."""
    try:
        proc = subprocess.Popen(
            ["pbcopy"], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=text.encode("utf-8"))
    except Exception as e:
        print(f"  ❌ Clipboard error: {e}")


def paste_from_clipboard():
    """Simulate Cmd+V to paste into the active app."""
    # Fix #3: small delay so the clipboard is ready in slow apps
    time.sleep(PASTE_DELAY)
    try:
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to keystroke "v" using command down'],
            capture_output=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def notify(title, message):
    """Show a macOS notification via osascript."""
    # Fix #8: strip newlines and escape for AppleScript
    safe_msg = (message
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("\\", "\\\\")
                .replace('"', '\\"')[:100])
    safe_title = (title
                  .replace("\\", "\\\\")
                  .replace('"', '\\"'))
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe_msg}" with title "{safe_title}"'],
            capture_output=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def _safe_unlink(path):
    """Delete a file if it exists, silently ignoring errors."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


_beep_procs = []  # Track afplay processes for cleanup


def beep(sound="Tink"):
    """Play a macOS system sound. Cleans up finished processes."""
    # Fix #7: track and clean up afplay processes
    global _beep_procs
    _beep_procs = [p for p in _beep_procs if p.poll() is None]

    path = f"/System/Library/Sounds/{sound}.aiff"
    if os.path.exists(path):
        proc = subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _beep_procs.append(proc)


# ---------------------------------------------------------------------------
# Hotkey — hold Right Option (⌥) to record, release to transcribe & copy.
# Uses a busy flag to prevent overlapping record/transcribe cycles.
# ---------------------------------------------------------------------------

_hotkey_active = False
_press_time = 0.0
_busy = False  # Fix #6: prevent overlapping cycles


def _on_press(key):
    global _hotkey_active, _press_time

    if key == keyboard.Key.alt_r and not _hotkey_active and not _busy:
        _hotkey_active = True
        _press_time = time.time()
        print("\n🔴 Recording... (release Right ⌥ to stop)")
        beep("Tink")  # Immediate feedback
        threading.Thread(target=start_recording, daemon=True).start()


def _on_release(key):
    global _hotkey_active, _busy

    if key != keyboard.Key.alt_r or not _hotkey_active:
        return

    _hotkey_active = False
    hold_time = time.time() - _press_time

    # Ignore accidental taps
    if hold_time < MIN_HOLD_SECONDS:
        print(f"  ⚠️  Too short ({hold_time:.1f}s) — hold longer to record")
        threading.Thread(target=stop_recording, daemon=True).start()
        return

    print(f"⏹️  Stopped ({hold_time:.1f}s). Transcribing...")
    beep("Pop")

    _busy = True  # Lock out new recordings until this cycle finishes

    def _stop_and_transcribe():
        global _busy
        try:
            path = stop_recording()
            if not path:
                print("  ⚠️  No audio captured")
                notify("VoiceClip", "No audio captured")
                return

            t0 = time.time()
            text = transcribe_audio(path)
            elapsed = time.time() - t0

            if text:
                copy_to_clipboard(text)
                paste_from_clipboard()
                beep("Glass")
                preview = text[:150] + ("..." if len(text) > 150 else "")
                print(f"  ✅ Copied! ({len(text)} chars, {elapsed:.1f}s)")
                print(f'  📋 "{preview}"')
                notify("VoiceClip ✅", text[:100])
            else:
                print("  ⚠️  No speech detected")
                notify("VoiceClip", "No speech detected")
        finally:
            _busy = False  # Unlock for next recording

    threading.Thread(target=_stop_and_transcribe, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  🎙️  VoiceClip")
    print("  Local voice → clipboard on Apple Silicon")
    print("=" * 50)
    print(f"\n  Model:        {MODEL}")
    print(f"  English only: {ENGLISH_ONLY}")
    print(f"\n  Microphones:")
    list_mics()
    print()
    print("  ⌨️  Hold Right Option (⌥) to record")
    print("     Release to transcribe & copy to clipboard")
    print("     Ctrl+C to quit")
    print()

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 VoiceClip stopped.")
        listener.stop()
        # Clean up any playing sounds
        for p in _beep_procs:
            if p.poll() is None:
                p.terminate()
        sys.exit(0)
