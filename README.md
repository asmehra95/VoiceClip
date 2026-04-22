# 🎙️ VoiceClip

**Talk. It types.** One key, zero setup headaches.

A free, open-source alternative to WhisperSync and similar cloud transcription tools — except VoiceClip runs entirely on your Mac. No subscription, no cloud, no data leaving your laptop. Just your voice and your Apple Silicon GPU.

VoiceClip turns your voice into text anywhere on your Mac. Hold a key, say what you're thinking, let go. Your words appear wherever your cursor is.

```bash
bash install.sh && voiceclip
```

That's it. You're done.

---

## 30-second demo

1. Run `voiceclip`
2. Open Slack, an email, a doc — anywhere you type
3. **Hold Right Option (⌥)**  — you hear a *tink*
4. **Say something** — "Hey, running 10 minutes late to the standup"
5. **Let go** — you hear a *pop*, then a *chime*
6. The text is already typed out where your cursor was

No copy-paste. No switching apps. No waiting.

## Why VoiceClip

- **One command install** — `bash install.sh` handles Python, dependencies, everything
- **Works everywhere** — any app, any text field, system-wide hotkey
- **Runs 100% locally** — your voice never leaves your Mac. Period.
- **Apple Silicon native** — uses your M-chip GPU for fast transcription
- **Optimized for low latency** — model preloaded at startup, zero-delay mic capture, instant silence detection. From key release to text pasted in 2-3 seconds
- **No config needed** — sensible defaults, works out of the box

## Install

```bash
git clone https://github.com/asmehra95/VoiceClip.git && cd voiceclip
bash install.sh
```

The installer checks your system, sets up a Python environment, installs dependencies, and creates a `voiceclip` command. Takes about 2 minutes.

**Prerequisites** (the installer handles most of these):
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg — `brew install ffmpeg` (used by the transcription engine)

**One-time macOS permissions:**
- **Accessibility** — System Settings → Privacy & Security → Accessibility → add your terminal app
- **Microphone** — System Settings → Privacy & Security → Microphone → enable your terminal app

Then just:
```bash
voiceclip
```

### Manual install (if you prefer)

```bash
brew install ffmpeg
git clone <this-repo> && cd voiceclip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python transcribe.py
```

## Models

VoiceClip downloads the right model automatically on first run. Pick the one that fits your workflow:

| Model | Speed (5s clip) | Accuracy | RAM needed |
|---|---|---|---|
| `tiny` | ~0.5s | ★★☆☆☆ | 1 GB |
| `base` | ~1s | ★★★☆☆ | 1 GB |
| `small` | ~2s | ★★★★☆ | 2 GB |
| `medium` | ~4s | ★★★★☆ | 5 GB |
| **`large-v3-turbo`** | **~3s** | **★★★★★** | **3 GB** |
| `large-v3` | ~6s | ★★★★★ | 6 GB |

**Default is `large-v3-turbo`** — best balance of speed and accuracy. Change it with:

```bash
VOICECLIP_MODEL=small voiceclip
```

## Getting the best results

**Speak naturally.** Whisper handles conversational speech, pauses, and "um"s well. You don't need to talk like a robot.

**Wait for the tink.** The sound means recording is live. Speak after you hear it.

**Use a headset.** Built-in MacBook mics work, but a headset mic in a noisy room makes a big difference.

**Keep it under 30 seconds.** Whisper is optimized for short-to-medium clips. For longer dictation, do a few shorter recordings.

**Teach it your words.** If it keeps getting a name or term wrong, edit `voiceclip/transcriber.py` and add:
```python
initial_prompt="YourName, YourCompany, any-tricky-words",
```

## Configuration

All optional. VoiceClip works without any of these.

```bash
# Different model
VOICECLIP_MODEL=small voiceclip

# Multilingual mode (default is English-only)
VOICECLIP_ENGLISH_ONLY=false voiceclip
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "This process is not trusted" | Grant Accessibility permission to your terminal app, restart terminal |
| No audio / silence detected | Check System Settings → Sound → Input — make sure the right mic is selected |
| First word gets cut off | Wait for the "tink" sound before speaking |
| Says "Thank you" or random text | Recording was mostly silence — speak louder or check your mic |
| Slow transcription | Try `VOICECLIP_MODEL=small voiceclip` or close GPU-heavy apps |

## Project structure

```
voiceclip/
    __init__.py       # Version
    __main__.py       # Entry point
    config.py         # Settings and model map
    recorder.py       # Audio capture (separate process)
    transcriber.py    # Whisper inference on GPU
    macos.py          # Clipboard, paste, sounds
    hotkey.py         # Global hotkey handler
transcribe.py         # Launcher
install.sh            # One-command installer
requirements.txt      # Dependencies
```

## License

MIT License. See [LICENSE](LICENSE) for details.

Built with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [pynput](https://github.com/moses-palmer/pynput), and [sounddevice](https://python-sounddevice.readthedocs.io/).
