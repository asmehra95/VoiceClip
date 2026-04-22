# 🎙️ VoiceClip

**Local voice-to-clipboard for macOS.** Hold a key, speak, release — your words are transcribed and pasted instantly. Everything runs on-device using your Apple Silicon GPU. No cloud, no API keys, no subscriptions.

## How it works

1. **Hold Right Option (⌥)** — recording starts, you hear a "tink"
2. **Speak** — say whatever you want to type
3. **Release** — you hear a "pop", transcription begins
4. **Done** — text is copied to clipboard AND pasted into your active app, you hear a "glass" chime

That's it. The whole cycle takes 2-5 seconds depending on how long you spoke.

## Quick install

```bash
git clone <this-repo> && cd voiceclip
bash install.sh
```

The installer handles everything: Python venv, dependencies, ffmpeg, and creates a `voiceclip` command you can run from anywhere.

### Manual install

If you prefer to set things up yourself:

```bash
# Prerequisites
brew install ffmpeg

# Clone and set up
git clone <this-repo> && cd voiceclip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python transcribe.py
```

### macOS permissions (required)

VoiceClip needs two permissions the first time you run it:

1. **Accessibility** — for the global hotkey to work outside the terminal
   - System Settings → Privacy & Security → Accessibility → add your terminal app

2. **Microphone** — to record audio
   - System Settings → Privacy & Security → Microphone → enable your terminal app

You may need to restart your terminal after granting these.

## Usage

```bash
# Start VoiceClip
voiceclip

# Or if installed manually
python transcribe.py

# Or as a module
python -m voiceclip
```

Once running, you'll see:

```
==================================================
  🎙️  VoiceClip v0.4.0
  Local voice → clipboard on Apple Silicon
==================================================

  Model:        large-v3-turbo
  English only: True

  ⌨️  Hold Right Option (⌥) to record
     Release to transcribe & copy to clipboard
     Ctrl+C to quit
```

Just hold Right Option anywhere on your Mac — in Slack, email, a doc, your IDE — speak, and release. The text appears where your cursor is.

## Configuration

Set environment variables to customize behavior:

```bash
# Use a different model (smaller = faster, larger = more accurate)
VOICECLIP_MODEL=small voiceclip

# Disable English-only mode (for other languages)
VOICECLIP_ENGLISH_ONLY=false voiceclip
```

### Available models

| Model | Size | Speed (5s clip) | Best for |
|---|---|---|---|
| `tiny` | ~150MB | ~0.5s | Quick drafts, low accuracy is OK |
| `base` | ~300MB | ~1s | Casual use |
| `small` | ~950MB | ~2s | Good balance |
| `medium` | ~3GB | ~4s | High accuracy |
| `large-v3-turbo` | ~3GB | ~3s | **Best speed/quality ratio (default)** |
| `large-v3` | ~6GB | ~6s | Maximum accuracy |

Models are downloaded automatically on first use from HuggingFace and cached locally.

## Tips for best results

### Getting accurate transcriptions

- **Speak clearly and at a normal pace.** Whisper handles natural speech well — you don't need to speak slowly or robotically.
- **Pause briefly before speaking** after pressing the key. The "tink" sound is your cue that recording is active.
- **Keep recordings under 30 seconds** for best accuracy. For longer dictation, do multiple shorter clips.
- **Minimize background noise.** A headset mic gives much better results than the MacBook's built-in mic.
- **Use English-only mode** (the default) if you're speaking English. It's faster and more accurate than the multilingual mode.

### Choosing the right model

- Start with `large-v3-turbo` (the default). It's the best balance of speed and accuracy on Apple Silicon.
- If it feels slow, try `small` — noticeably faster with good accuracy.
- If accuracy matters more than speed (e.g., technical dictation), try `large-v3`.
- `tiny` and `base` are useful for testing but produce noticeably worse transcriptions.

### Handling misrecognized words

If Whisper consistently gets specific words wrong (names, jargon, acronyms), you can add them as a prompt hint. Edit `voiceclip/transcriber.py` and add an `initial_prompt` parameter:

```python
result = mlx_whisper.transcribe(
    audio_path,
    path_or_hf_repo=_REPO,
    initial_prompt="VoiceClip, Kiro, your-custom-terms-here",
    ...
)
```

This biases the model toward recognizing those specific words.

### Performance

- **First run is slower** — the model downloads (~3GB for turbo) and loads into GPU memory. Subsequent runs reuse the cached model.
- **VoiceClip preloads the model at startup** so your first transcription is fast.
- **Apple Silicon GPU** does all the heavy lifting. More unified memory = ability to run larger models.
- **16GB RAM** can comfortably run up to `large-v3-turbo`. **32GB+** can run `large-v3`.

## Project structure

```
voiceclip/
    __init__.py       # Version
    __main__.py       # Entry point, logging, startup sequence
    config.py         # Constants, env var parsing, model map
    recorder.py       # Audio capture (child process + IPC)
    transcriber.py    # mlx-whisper inference
    macos.py          # Clipboard, paste, notifications, sounds
    hotkey.py         # Global hotkey handler
transcribe.py         # Thin launcher (python transcribe.py)
install.sh            # One-command installer
requirements.txt      # Python dependencies
```

## Requirements

- macOS (Apple Silicon — M1, M2, M3, or M4)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

## Troubleshooting

**"This process is not trusted"**
→ Grant Accessibility permissions to your terminal app (see macOS permissions above). Restart the terminal after.

**No audio captured / silence detected**
→ Check that your mic is selected as the default input in System Settings → Sound → Input. Some USB/Bluetooth headsets need to be explicitly selected.

**First word gets clipped**
→ Wait for the "tink" sound before speaking. That's the signal that recording is active.

**"Thank you" or other hallucinated text**
→ This happens when the recording is mostly silence. VoiceClip filters these out, but if it persists, try speaking louder or moving closer to the mic.

**Slow transcription**
→ Try a smaller model: `VOICECLIP_MODEL=small voiceclip`. Also ensure no other heavy GPU tasks are running.

## License

Personal use. Built with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [pynput](https://github.com/moses-palmer/pynput), and [sounddevice](https://python-sounddevice.readthedocs.io/).
