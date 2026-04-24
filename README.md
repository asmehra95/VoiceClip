# 🎙️ VoiceClip

**Talk. It types.** One key, zero setup headaches.

VoiceClip turns your voice into text anywhere on your Mac — Slack, email, docs, terminal, anywhere you can type. Hold a key, speak, let go.

Everything runs locally on your Apple Silicon GPU (any Mac from 2021 or newer). No cloud, no subscription, no data leaves your laptop. Audio is deleted immediately after transcription.

---

## ⚡ Quick Start

```bash
git clone ssh://git.amazon.com/pkg/VoiceClip && cd voiceclip
bash install.sh
voiceclip
```

That's it. Hold **Right Option (⌥)**, speak, release. Your words appear where your cursor is.

> **First time?** macOS will ask for Accessibility and Microphone permissions. Grant both, then restart your terminal. See [Troubleshooting](#troubleshooting) if you get stuck.

---

## How It Works

1. **Hold Right Option (⌥)** — you hear a *tink*
2. **Speak** — "Hey, running 10 minutes late to the standup"
3. **Release** — you hear a *pop*, then a *chime*
4. Text appears where your cursor was

Works in any app. Your clipboard stays untouched.

---

## What Makes It Different

| | VoiceClip | Cloud tools (Otter, etc.) |
|---|---|---|
| **Privacy** | 100% local — nothing leaves your Mac | Audio sent to servers |
| **Latency** | 2-3 seconds | 5-10 seconds |
| **Cost** | Free, forever | $10-20/month |
| **Works offline** | Yes | No |
| **Setup** | One command | Account, browser, extensions |

---

## Install

### One-command install (recommended)

```bash
git clone ssh://git.amazon.com/pkg/VoiceClip && cd voiceclip
bash install.sh
```

The installer:
- Checks your system (Apple Silicon, Python 3.10+, ffmpeg)
- Creates a Python environment with all dependencies
- Sets up the `voiceclip` command
- Takes about 2 minutes

### What you need

- **Apple Silicon Mac** (M1, M2, M3, or M4)
- **Python 3.10+** — most Macs have this. Check with `python3 --version`
- **ffmpeg** — the installer handles this, or `brew install ffmpeg`

### One-time macOS permissions

After installing, macOS needs two permissions:

1. **Accessibility** — lets VoiceClip type text into apps
   → System Settings → Privacy & Security → Accessibility → add your terminal

2. **Microphone** — lets VoiceClip hear you
   → System Settings → Privacy & Security → Microphone → enable your terminal

Then just run:
```bash
voiceclip
```

### Manual install

If you prefer to set things up yourself:

```bash
brew install ffmpeg
git clone ssh://git.amazon.com/pkg/VoiceClip && cd voiceclip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python transcribe.py
```

### Update

Pull the latest code and re-run the installer:

```bash
cd voiceclip && git pull && bash install.sh
```

### Uninstall

```bash
rm -rf ~/.voiceclip ~/.local/bin/voiceclip
```

---

## Configuration

**VoiceClip works out of the box with no configuration.** Everything below is optional.

All settings live in one file: `~/.voiceclip/config.json`. It's created automatically on first run. Edit it with any text editor.

```json
{
  "model": "large-v3-turbo",
  "english_only": true,
  "persona": "default",
  "hotkey": "alt_r",
  "hotkey_mode": "hold"
}
```

For quick one-off changes, use environment variables (they override the config file):

```bash
VOICECLIP_MODEL=small voiceclip          # Faster, less accurate
VOICECLIP_PERSONA=engineering voiceclip  # Technical vocabulary
VOICECLIP_HOTKEY=f5 voiceclip            # Different key
```

See [`config.default.json`](config.default.json) for the full config with all options and examples.

---

## Hotkey

**Default:** Hold **Right Option (⌥)** to record, release to stop.

### Change the key

```json
"hotkey": "f5"
```

Supported: `alt_r`, `alt_l`, `ctrl_r`, `ctrl_l`, `shift_r`, `shift_l`, `cmd_r`, `cmd_l`, `caps_lock`, `f1`–`f12`, `space`, `esc`, or any single letter.

### Toggle mode

If holding a key is uncomfortable (repetitive strain injury, accessibility needs, or longer recordings), switch to toggle mode — press once to start, press again to stop:

```json
"hotkey_mode": "toggle"
```

---

## Personas

Personas help VoiceClip understand your vocabulary. An engineering persona knows "S3" and "DynamoDB". A medical persona knows "prognosis" and "dosage".

### Built-in personas

| Persona | Best for | Example fixes |
|---|---|---|
| `default` | General use | No domain bias |
| `engineering` | Technical work | "s three" → "S3", "e c two" → "EC2" |
| `casual` | Chat, messaging | Conversational tone |
| `medical` | Clinical notes | Medical terminology |

Switch in config:
```json
"persona": "engineering"
```

Or on the fly:
```bash
VOICECLIP_PERSONA=engineering voiceclip
```

### Create your own

Add a persona to the `personas` section of your config:

```json
"personas": {
  "my-team": {
    "prompt": "SIM, CR, oncall, Brazil, pip, Avtar, ",
    "dictionary": {
      "sim": "SIM",
      "c r": "CR"
    }
  }
}
```

- **`prompt`** — words that bias the speech recognition model toward your vocabulary
- **`dictionary`** — find-and-replace rules applied after transcription (case-insensitive)

The global `dictionary` in your config always applies on top of the active persona.

---

## Voice Commands

VoiceClip converts spoken commands into formatting — no extra setup needed:

| Say this | Get this |
|---|---|
| "period" | `.` |
| "comma" | `,` |
| "question mark" | `?` |
| "exclamation mark" | `!` |
| "new line" | line break |
| "new paragraph" | blank line |
| "bullet" | `• ` |
| "colon" | `:` |
| "open quote" / "close quote" | `"` |

**Example:** Say *"Hey comma running 10 minutes late period I will join from my phone period"*

**Result:** Hey, running 10 minutes late. I will join from my phone.

VoiceClip also auto-capitalizes sentences and fixes standalone "i" → "I".

---

## Grammar Polish (opt-in)

For even cleaner output, enable the local AI grammar fixer. It runs a small language model on your GPU.

```json
{
  "polish": true
}
```

Your text pastes immediately. The grammar fixer runs in the background and silently replaces the text with a cleaner version about a second later. If it can't improve anything, the original stays.

**Requires one extra install:** `pip install mlx-lm`

---

## Models

VoiceClip downloads the right model on first run (~3 GB for the default). Choose based on your speed/accuracy preference:

| Model | Speed | Accuracy | RAM |
|---|---|---|---|
| `tiny` | Instant | Fair | 1 GB |
| `base` | ~1s | Good | 1 GB |
| `small` | ~2s | Great | 2 GB |
| **`large-v3-turbo`** ★ | **~3s** | **Excellent** | **3 GB** |
| `large-v3` | ~6s | Excellent | 6 GB |

★ Default. Best balance of speed and accuracy for most people.

Change with:
```json
"model": "small"
```

### Multilingual

VoiceClip defaults to English. To transcribe other languages, set:

```json
"english_only": false
```

Whisper auto-detects the language. Works best with `large-v3-turbo` or `large-v3` models.

---

## Tips for Best Results

- **Wait for the tink** before speaking — that's when recording starts
- **Speak naturally** — Whisper handles pauses, "um"s, and conversational speech well
- **Use the MacBook mic** for best quality — Bluetooth headset mics capture at lower quality
- **Keep clips under 30 seconds** — for longer dictation, do a few shorter recordings
- **Use personas** for technical work — the vocabulary bias makes a real difference

---

## Troubleshooting

**Most common issues:**

| Problem | Solution |
|---|---|
| "This process is not trusted" | Grant Accessibility permission to your terminal, then restart it |
| No audio / silence detected | System Settings → Sound → Input — check the right mic is selected |
| First word gets cut off | Wait for the "tink" sound before speaking |

**Less common:**

| Problem | Solution |
|---|---|
| Says "Thank you" randomly | Recording was mostly silence — speak louder or check mic |
| Slow transcription | Try `VOICECLIP_MODEL=small voiceclip` or close GPU-heavy apps |
| App frozen on first run | Model is downloading (~3 GB). A spinner shows progress |
| Bad accuracy with Bluetooth headset | Bluetooth mics use low-quality mode. Use MacBook mic for input, headset for output |
| LLM polish "not installed" | Run `pip install mlx-lm` in your VoiceClip venv |

---

## Project Structure

```
voiceclip/
    __init__.py       # Version
    __main__.py       # Entry point and startup
    config.py         # Config loader (JSON + env overrides)
    recorder.py       # Audio capture (separate process)
    transcriber.py    # Whisper inference on GPU
    formatter.py      # Text cleanup and voice commands
    polisher.py       # Optional AI grammar polish
    hotkey.py         # Hotkey handler (hold + toggle modes)
    macos.py          # Clipboard, paste, sounds, permissions
    utils.py          # Shared utilities
transcribe.py         # Launcher
install.sh            # Installer
config.default.json   # Default config with examples
requirements.txt      # Pinned dependencies
tests/                # Unit tests (95 tests, pytest)
```

---

## License

MIT — free and open source. See [LICENSE](LICENSE).

Built with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [pynput](https://github.com/moses-palmer/pynput), and [sounddevice](https://python-sounddevice.readthedocs.io/).
