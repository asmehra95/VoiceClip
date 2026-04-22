"""
VoiceClip — Local Voice-to-Clipboard for macOS
Record from mic or upload a file, transcribe with mlx-whisper (Apple Silicon GPU),
copy to clipboard.

Usage:
    pip install mlx-whisper gradio
    python transcribe.py
    # Opens browser at http://localhost:7860

Requires:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - ffmpeg (brew install ffmpeg)
"""

import sys
import time

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

if _missing:
    print(f"Missing packages: {', '.join(_missing)}")
    print(f"Run: pip install {' '.join(_missing)}")
    sys.exit(1)

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

DEFAULT_MODEL = "base.en"


def transcribe(audio_path, model_size, english_only):
    """Transcribe audio using mlx-whisper on Apple Silicon GPU."""
    if audio_path is None:
        return "", "⚠️ No audio provided"

    try:
        start_time = time.time()

        # .en variants exist for tiny, base, small, medium
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

        return plain, f"✅ {len(plain)} chars in {elapsed:.1f}s ({model_key} on Metal GPU)"

    except Exception as e:
        return "", f"❌ Error: {e}"


def build_ui():
    with gr.Blocks(title="VoiceClip") as app:
        gr.Markdown("## 🎙️ VoiceClip")
        gr.Markdown("Record or upload audio → transcribe on Apple Silicon GPU → copy")

        with gr.Row():
            audio_input = gr.Audio(
                label="Audio",
                type="filepath",
                sources=["microphone", "upload"],
            )

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

        transcribe_btn = gr.Button("🎙️ Transcribe", variant="primary", size="lg")
        status = gr.Markdown("")

        plain_output = gr.Textbox(label="📋 Transcription", lines=6)

        copy_btn = gr.Button("📋 Copy to Clipboard")
        copy_btn.click(
            fn=None,
            inputs=[plain_output],
            js="(text) => { navigator.clipboard.writeText(text); }",
        )

        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, model_dropdown, english_only],
            outputs=[plain_output, status],
        )

    return app


if __name__ == "__main__":
    print("VoiceClip — mlx-whisper on Apple Silicon GPU")
    print("Starting...")
    app = build_ui()
    app.launch(theme=gr.themes.Soft())
