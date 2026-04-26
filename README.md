# 🎙️ VoiceClip

**Talk. It types.** One key, zero setup headaches.

VoiceClip turns your voice into text anywhere on your Mac — Slack, email, docs, terminal, anywhere you can type. Hold a key, speak, let go.

**Dictation runs locally** on your Apple Silicon GPU (any Mac from 2021 or newer). Your audio is deleted immediately after transcription, and the text never leaves your Mac.

Optional AI features — daily summaries, research briefs, pattern detection — can be configured to run locally or via cloud APIs (OpenAI, Anthropic). All are off by default. See [Privacy & data flow](#privacy--data-flow) before turning any of them on.

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
| **Dictation privacy** | 100% local — audio and text stay on your Mac | Audio sent to servers |
| **Latency** | 2-3 seconds | 5-10 seconds |
| **Cost** | Free; cloud AI features are opt-in and pay-per-use | $10-20/month |
| **Works offline** | Dictation yes, cloud AI features no | No |
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
    "prompt": "SIM, CR, oncall, Brazil, pip, Avtar",
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

## Reflections (opt-in)

Want to capture a thought without pasting it anywhere? Set up a second hotkey for reflections. Same record → speak → release flow, but the text is saved privately to your local history and **never** touches the clipboard or pastes into any app.

### Enable it

Two things in `~/.voiceclip/config.json`:

```json
{
  "history": true,
  "reflection_hotkey": "f6"
}
```

That's it. Pick any key that's different from your main hotkey (see [Hotkey](#hotkey) for supported keys). Restart VoiceClip.

### How it feels

- **Hold your main key (⌥)** — normal dictation, pastes where your cursor is
- **Hold your reflection key (F6)** — different start chime, different done chime; your words are saved but nothing is pasted

The chime difference matters — it tells you at a glance which mode fired.

### Browse what you've captured

```bash
voiceclip history                           # Everything, most recent first
voiceclip history --reflections             # Only reflections
voiceclip history --transcriptions          # Only dictations
voiceclip history --reflections --today     # Today's reflections
voiceclip history --search "idea"           # Keyword search
voiceclip history --reflect-last            # Oh wait, that dictation was actually a reflection
```

### What gets stored

For every entry (dictation or reflection):
- Timestamp, duration, the text itself
- The app you were in (e.g. "Slack", "Xcode") — no permission needed for this, it's public info

That's it. No audio is ever kept, no window titles, no URLs, no screen contents. Database lives at `~/.voiceclip/history.db`, readable only by you (chmod 600).

### Retention

- Transcriptions age out after `history_max_days` (default 30)
- Reflections are **kept forever** unless you set `reflection_max_days` to a positive number
- Turning history off never deletes anything — your entries sit there until you run `voiceclip history --clear`

### Kind-scoped clearing

```bash
voiceclip history --clear --transcriptions  # Wipe dictations, keep reflections
voiceclip history --clear --reflections     # Wipe reflections, keep dictations
voiceclip history --clear                   # Wipe everything (with confirmation)
```

---

## Web viewer (opt-in)

Prefer to browse your day in a real UI? Turn on history, then:

```bash
voiceclip view
```

A lightweight local web page opens in your browser at `http://localhost:8723`. It binds to localhost only — nothing is ever exposed to your network.

### What you see

- **Today by default.** Big header, day navigation (← Yesterday · Today · Next →), date picker.
- **Where your voice went.** A small bar chart of which apps you dictated in today.
- **Reflections first.** Your 💭 entries get pride of place — generous type, warm background.
- **Transcriptions collapsed.** "Show N transcriptions" toggle — they're secondary, available when you want context.
- **One click to copy** any entry. One click to promote a past dictation to a reflection.
- **Dark mode follows your system.** No toggle, no setting.

### Daily summary (opt-in, costs-nothing edition)

At the end of a day, an LLM can read your dictations and reflections and write you a short summary — "today you spent most of your voice on Slack and Notes, kept circling back to the history feature, and had four reflections about start-small product decisions."

Off by default. Enable in config:

```json
{
  "summaries": {
    "provider": "local",
    "local_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "style": "descriptive"
  }
}
```

Three provider options:

| Provider | Where it runs | Needs | Best for |
|---|---|---|---|
| `local` | Your Mac (via mlx-lm) | `pip install mlx-lm`, ~4GB model download | Privacy, free, offline |
| `openai` | OpenAI API | `OPENAI_API_KEY` env, `pip install openai` | Quality |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY` env, `pip install anthropic` | Quality |

**Cloud providers send your day's entries to the provider.** That's the only feature in VoiceClip that touches the network, and only when you explicitly opt in. A warning appears at startup if a cloud provider is configured.

Recommended local models (bigger = slower but better — you're only running this once a day):

- `mlx-community/Qwen2.5-7B-Instruct-4bit` — default, ~4GB RAM, ~15s per day
- `mlx-community/Qwen2.5-14B-Instruct-4bit` — better, ~8GB RAM, ~30s per day
- `mlx-community/Llama-3.2-3B-Instruct-4bit` — leanest, ~2GB RAM

Generate from the CLI:

```bash
voiceclip summarize --day today
voiceclip summarize --day yesterday
voiceclip summarize --day 2026-04-23
voiceclip summarize --day today --force    # Regenerate even if cached
```

Or just open the viewer — today's page shows a "Generate" button that calls the same thing. Past days are cached forever; today regenerates when new entries arrive.

---

## Privacy & data flow

VoiceClip is built to be transparent about what crosses the network. Here is the complete picture:

### Always local (never leaves your Mac)
- **Audio capture and transcription.** Whisper runs on your Apple Silicon GPU. The audio WAV is deleted from `/tmp` immediately after transcription.
- **Clipboard and paste.** `pbcopy` + simulated Cmd-V are local.
- **History database.** SQLite file at `~/.voiceclip/history.db`, permissions `0600`, never synced, never transmitted.
- **Active-app context capture** (when you enable history). Uses `osascript` to read the frontmost app name. Nothing beyond the name is captured.

### Opt-in local (runs on your Mac if enabled)
- **Grammar polish** (`polish = true`) — uses `mlx-lm` locally.
- **Daily summaries** with `summaries.provider = "local"` — uses `mlx-lm` locally.
- **Patterns coach** with `patterns.provider = "local"` — uses `mlx-lm` locally. **This is the default for Patterns** because it reads the widest window of your history.

### Opt-in cloud (sends your data to a third-party API)

These features are all `provider = "none"` by default. If you change any of them to `openai` or `anthropic`, the following data will be sent off your Mac to that provider:

| Feature | What is sent |
|---|---|
| **Summaries** (`summaries.provider`) | Every entry (transcription + reflection) for the target day |
| **Research** (`research.provider`) | The research topic you typed or dictated (text only). If the model uses its web-search tool, that topic is also sent to a search index (Bing for OpenAI, Anthropic's integration for Anthropic). |
| **Patterns** (`patterns.provider`) | Up to 7 days of reflections and cached daily summaries in a single prompt |

API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are read from environment variables only. They are never stored in `config.json`, logs, or the database.

**Cloud provider retention:** OpenAI and Anthropic cache API requests for abuse detection (typically around 30 days). Your reflections, topics, or summaries may sit on their servers for that window before being deleted. Review each provider's data-use policy before enabling.

### What is NOT captured anywhere
- Window titles (off by default; requires explicit opt-in)
- Browser URLs
- Calendar events
- Screen contents
- Keystrokes outside the hotkey
- Telemetry or analytics of any kind — VoiceClip never phones home

### Uninstall
To remove everything:
```bash
rm -rf ~/.voiceclip ~/.local/bin/voiceclip
# Also clears cached AI models (can be 3-10 GB):
rm -rf ~/.cache/huggingface
```

---

## Troubleshooting

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
    formatter.py      # Text cleanup and dictionaryands
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
