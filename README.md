# 🎙️ VoiceClip

**Talk. It types.** One key, zero setup headaches.

A simple live transcription tool that runs entirely on your Mac. No subscription, no cloud, no data leaving your laptop. Just your voice and your Apple Silicon GPU.

VoiceClip turns your voice into text anywhere on your Mac. Hold a key, say what you're thinking, let go. Your words appear wherever your cursor is.

That's it.

---

## How it works

1. Run `voiceclip`
2. Open Slack, an email, a doc — anywhere you type
3. **Hold Right Option (⌥)** — you hear a *tink*
4. **Say something** — "Hey, running 10 minutes late to the standup"
5. **Let go** — you hear a *pop*, then a *chime*
6. The text is already typed out where your cursor was

No copy-paste. No switching apps. No waiting. Your clipboard stays untouched.

## Why VoiceClip

- **One command install** — `bash install.sh` handles Python, dependencies, everything
- **Works everywhere** — any app, any text field, system-wide hotkey
- **Runs 100% locally** — your voice never leaves your Mac. Period.
- **Apple Silicon native** — uses your M-chip GPU for fast transcription
- **Optimized for low latency** — model preloaded at startup, zero-delay mic capture, instant silence detection. From key release to text pasted in 2-3 seconds
- **Smart formatting** — auto-capitalizes, converts spoken punctuation ("period", "comma", "new line"), strips Whisper hallucinations
- **Personas** — switch between engineering, casual, medical vocabulary with one config change
- **Clipboard preserved** — your previously copied text stays on the clipboard after dictation
- **No config needed** — sensible defaults, works out of the box

## Install

```bash
git clone ssh://git.amazon.com/pkg/VoiceClip && cd voiceclip
bash install.sh
```

The installer checks your system, sets up a Python environment, installs dependencies, creates a default config, and adds a `voiceclip` command. Takes about 2 minutes.

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

## Configuration

All config lives in `~/.voiceclip/config.json`. A default is created on first run. Edit it with any text editor.

```json
{
  "model": "large-v3-turbo",
  "english_only": true,
  "polish": false,
  "persona": "default",
  "hotkey": "alt_r",
  "hotkey_mode": "hold",
  "dictionary": {
    "voiceclip": "VoiceClip",
    "macos": "macOS"
  }
}
```

Environment variables override config for quick one-off changes:

```bash
VOICECLIP_MODEL=small voiceclip
VOICECLIP_PERSONA=engineering voiceclip
VOICECLIP_ENGLISH_ONLY=false voiceclip
VOICECLIP_HOTKEY=f5 voiceclip
VOICECLIP_HOTKEY_MODE=toggle voiceclip
```

See `config.default.json` in the repo for the full structure with all options.

## Personas

Personas bias Whisper toward domain-specific vocabulary and fix common misheard terms. Switch personas in config.json or via env var.

**Built-in personas:**

| Persona | What it does |
|---|---|
| `default` | Clean slate, no domain bias |
| `engineering` | AWS, Kubernetes, API terms. "s three" → "S3", "e c two" → "EC2" |
| `casual` | Conversational bias for chat messages |
| `medical` | Clinical terminology bias |

**Add your own:**

```json
"personas": {
  "amazon": {
    "prompt": "AWS, S3, DynamoDB, SIM, CR, code review, oncall, pip, Brazil",
    "dictionary": {
      "sim": "SIM",
      "c r": "CR",
      "pip": "pip"
    }
  }
}
```

The `prompt` field biases Whisper during transcription. The `dictionary` fixes words after transcription. Both are optional.

The global `dictionary` always applies on top of the active persona's dictionary.

## Hotkey

Default is **Right Option (⌥)** in **hold** mode (hold to record, release to stop). Both the key and mode are configurable.

**Change the key:**

```json
"hotkey": "f5"
```

Supported keys: `alt_r`, `alt_l`, `ctrl_r`, `ctrl_l`, `shift_r`, `shift_l`, `cmd_r`, `cmd_l`, `caps_lock`, `f1`–`f12`, `space`, `esc`, or any single letter.

**Toggle mode** (press once to start, press again to stop — useful for longer recordings or accessibility):

```json
"hotkey_mode": "toggle"
```

## Smart Formatting

VoiceClip automatically cleans up transcriptions with zero latency cost:

- **Capitalization** — first word, after periods, "i" → "I"
- **Spoken punctuation** — "period" → `.` "comma" → `,` "question mark" → `?`
- **Spoken formatting** — "new line" → actual line break, "bullet" → `• `
- **Hallucination filter** — strips phantom "thank you" / "thanks for watching"
- **Whitespace cleanup** — collapses extra spaces, fixes spacing around punctuation

Just speak naturally. Say "Hey comma running 10 minutes late period" and get: "Hey, running 10 minutes late."

## LLM Polish (opt-in)

For even cleaner output, enable local LLM polishing. This runs a small language model on your GPU to fix grammar and punctuation.

```json
{
  "polish": true,
  "polish_model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
}
```

**How it works:** Your text pastes immediately (no delay). The LLM polishes in the background and replaces the text ~1 second later. If it fails, the original stays.

**Requires:** `pip install mlx-lm` (only needed when polish is enabled)

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

**Default is `large-v3-turbo`** — best balance of speed and accuracy.

## Getting the best results

**Speak naturally.** Whisper handles conversational speech, pauses, and "um"s well. You don't need to talk like a robot.

**Wait for the tink.** The sound means recording is live. Speak after you hear it.

**Use a headset.** Built-in MacBook mics work, but a headset mic in a noisy room makes a big difference.

**Keep it under 30 seconds.** Whisper is optimized for short-to-medium clips. For longer dictation, do a few shorter recordings.

**Use personas.** If you're doing technical work, switch to the engineering persona. The vocabulary bias makes a real difference.

## Troubleshooting

| Problem | Fix |
|---|---|
| "This process is not trusted" | Grant Accessibility permission to your terminal app, restart terminal |
| No audio / silence detected | Check System Settings → Sound → Input — make sure the right mic is selected |
| First word gets cut off | Wait for the "tink" sound before speaking |
| Says "Thank you" or random text | Recording was mostly silence — speak louder or check your mic |
| Slow transcription | Try `VOICECLIP_MODEL=small voiceclip` or close GPU-heavy apps |
| App appears frozen on first run | Model is downloading (~3 GB). A spinner shows progress. |
| Bad accuracy with Bluetooth headset | Bluetooth mics use low-quality HFP mode (8kHz). Use MacBook mic for input and headset for output only, or use a USB headset |
| LLM polish says "not installed" | Run `pip install mlx-lm` in your VoiceClip venv |

## Project structure

```
voiceclip/
    __init__.py       # Version
    __main__.py       # Entry point, startup, signal handling
    config.py         # Unified config loader (JSON + env var overrides)
    recorder.py       # Audio capture (separate process for zero-latency)
    transcriber.py    # Whisper inference on GPU with timeout watchdog
    formatter.py      # Regex post-processing, spoken punctuation, dictionary
    polisher.py       # Optional LLM polish (paste-first-polish-after)
    hotkey.py         # Global hotkey handler with thread-safe state
    macos.py          # Clipboard, paste, sounds, permissions
    utils.py          # Shared utilities
transcribe.py         # Launcher
install.sh            # One-command installer
config.default.json   # Default configuration with example personas
requirements.txt      # Dependencies
tests/                # Unit tests (pytest)
```

## License

MIT License. See [LICENSE](LICENSE) for details.

Built with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [pynput](https://github.com/moses-palmer/pynput), and [sounddevice](https://python-sounddevice.readthedocs.io/).
