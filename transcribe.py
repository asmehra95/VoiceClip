"""
VoiceClip — Local Voice-to-Clipboard for macOS
Record from mic or upload a file, transcribe with mlx-whisper (Apple Silicon GPU),
copy to clipboard.

Usage:
    pip install mlx-whisper gradio sounddevice soundfile
    python transcribe.py
    # Opens browser at http://localhost:7860

Requires:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - ffmpeg (brew install ffmpeg)
"""

import sys
import os
import time
import tempfile
import threading

# Check dependencies early
_missing = []
try:
    import mlx_whisper
except ImportError:
    _missing.append("mlx-whisper")
try:
    import gradio as gr
except ImportError:
    _missing.append("gradio")
try:
    import sounddevice as sd
except ImportError:
    _missing.append("sounddevice")
try:
    import soundfile as sf
except ImportError:
    _missing.append("soundfile")

if _missing:
    print(f"Missing packages: {', '.join(_missing)}")
    print(f"Run: pip install {' '.join(_missing)}")
    sys.exit(1)

import numpy as np

# Model name -> HuggingFace repo
MODELS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
}

SAMPLE_RATE = 16000  # Whisper expects 16kHz

# Recording state
_recording = False
_audio_frames = []
_stream = None


def get_input_devices():
    """List available input devices, system default first."""
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    input_devices = {}

    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            label = d["name"]
            if i == default_input:
                label = f"⭐ {label} (System Default)"
            input_devices[label] = i

    return input_devices


def start_recording(mic_choice):
    """Start recording from the selected mic."""
    global _recording, _audio_frames, _stream

    if _recording:
        return "⚠️ Already recording"

    devices = get_input_devices()
    device_id = devices.get(mic_choice)

    _recording = True
    _audio_frames = []

    def callback(indata, frames, time_info, status):
        if _recording:
            _audio_frames.append(indata.copy())

    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device_id,
        callback=callback,
    )
    _stream.start()

    return "🔴 Recording... click Stop when done"


def stop_recording():
    """Stop recording and save to a temp WAV file."""
    global _recording, _stream

    _recording = False
    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None

    if not _audio_frames:
        return None, "⚠️ No audio recorded"

    audio_data = np.concatenate(_audio_frames, axis=0).flatten()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_data, SAMPLE_RATE)
    tmp.close()

    duration = len(audio_data) / SAMPLE_RATE
    return tmp.name, f"⏹️ Stopped — {duration:.1f}s recorded"


def transcribe(audio_path, model_size, english_only):
    """Transcribe audio using mlx-whisper on Apple Silicon GPU."""
    if audio_path is None:
        return "", "⚠️ No audio to transcribe"

    try:
        start_time = time.time()

        if english_only and model_size in ("tiny", "base", "small", "medium"):
            model_key = f"{model_size}.en"
        else:
            model_key = model_size

        repo = MODELS.get(model_key, MODELS["base.en"])

        kwargs = {"path_or_hf_repo": repo}
        if english_only:
            kwargs["language"] = "en"

        result = mlx_whisper.transcribe(audio_path, **kwargs)

        segments = result.get("segments", [])
        if not segments:
            return "", "No speech detected"

        plain = " ".join(seg["text"].strip() for seg in segments)
        elapsed = time.time() - start_time

        # Clean up temp file if it was from recording
        if audio_path.startswith(tempfile.gettempdir()):
            os.unlink(audio_path)

        return plain, f"✅ {len(plain)} chars in {elapsed:.1f}s ({model_key} on Metal GPU)"

    except Exception as e:
        return "", f"❌ Error: {e}"


def build_ui():
    devices = get_input_devices()
    device_names = list(devices.keys())
    # Put system default first
    default_name = next((n for n in device_names if "System Default" in n), device_names[0])

    with gr.Blocks(title="VoiceClip") as app:
        gr.Markdown("## 🎙️ VoiceClip")
        gr.Markdown("Record or upload audio → transcribe on Apple Silicon GPU → copy")

        # Hidden state to hold the recorded audio path
        recorded_path = gr.State(value=None)

        # --- Mic Recording Section ---
        gr.Markdown("### Record from Mic")

        mic_dropdown = gr.Dropdown(
            choices=device_names,
            value=default_name,
            label="Microphone",
        )

        with gr.Row():
            record_btn = gr.Button("⏺ Start Recording", variant="primary")
            stop_btn = gr.Button("⏹ Stop Recording", variant="stop")

        rec_status = gr.Markdown("")

        record_btn.click(
            fn=start_recording,
            inputs=[mic_dropdown],
            outputs=[rec_status],
        )
        stop_btn.click(
            fn=stop_recording,
            outputs=[recorded_path, rec_status],
        )

        # --- Upload Section ---
        gr.Markdown("### Or Upload a File")
        audio_upload = gr.Audio(
            label="Audio File",
            type="filepath",
            sources=["upload"],
        )

        # --- Transcription Settings ---
        gr.Markdown("### Transcribe")
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"],
                value="large-v3-turbo",
                label="Model",
                info="turbo ⭐ = best speed/quality, large-v3 = most accurate",
            )
            english_only = gr.Checkbox(
                label="English only",
                value=True,
                info="Faster & more accurate for English",
            )

        transcribe_rec_btn = gr.Button("🎙️ Transcribe Recording", variant="primary", size="lg")
        transcribe_file_btn = gr.Button("📁 Transcribe Uploaded File", size="lg")

        status = gr.Markdown("")
        plain_output = gr.Textbox(label="📋 Transcription", lines=6)

        copy_btn = gr.Button("📋 Copy to Clipboard")
        copy_btn.click(
            fn=None,
            inputs=[plain_output],
            js="(text) => { navigator.clipboard.writeText(text); }",
        )

        # Transcribe recording
        transcribe_rec_btn.click(
            fn=transcribe,
            inputs=[recorded_path, model_dropdown, english_only],
            outputs=[plain_output, status],
        )

        # Transcribe uploaded file
        transcribe_file_btn.click(
            fn=transcribe,
            inputs=[audio_upload, model_dropdown, english_only],
            outputs=[plain_output, status],
        )

    return app


if __name__ == "__main__":
    print("VoiceClip — mlx-whisper on Apple Silicon GPU")
    print("\nAvailable microphones:")
    for name in get_input_devices():
        print(f"  {name}")
    print("\nStarting...")
    app = build_ui()
    app.launch(theme=gr.themes.Soft())
