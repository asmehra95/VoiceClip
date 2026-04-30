# 🎙️ VoiceClip

Push-to-talk dictation for macOS, with an optional private journal.

Hold a key, speak, release. Your words appear where your cursor was.
Audio never leaves your Mac — transcription runs on the local Apple Silicon GPU.

Optional features include a searchable journal, daily LLM summaries, a
chronological timeline view, and a longitudinal "patterns" coach. They're
all off by default, and can all run fully locally if you prefer. See the
[Privacy & data flow](#privacy--data-flow) section before enabling anything
that reaches the network.

---

## Quick Start

```bash
git clone https://github.com/asmehra95/VoiceClip.git && cd VoiceClip
bash install.sh
voiceclip
```

Hold **Right Option (⌥)**, speak, release. Your words paste at the cursor.

> **First time?** macOS will ask for Accessibility and Microphone permissions.
> Grant both, then restart your terminal. See [Troubleshooting](#troubleshooting)
> if anything feels off — or run `voiceclip doctor` for a one-shot health check.

---

## Is this for you?

Two kinds of people get the most out of VoiceClip:

**People who want fast local dictation.** The core loop is hold → speak →
release. No account, no cloud, no subscription, no menu-bar app to launch.
Works anywhere you can type: Slack, email, docs, terminal, IDE, browser.

**People who want to think out loud.** A second hotkey captures "reflections"
— spoken thoughts that get saved to a local database without touching your
clipboard. Browse them later in a simple web journal. If you enable it, a
local LLM can summarize your day, produce a chronological timeline, and
surface recurring themes across the past week. All optional, all private.

If you want a menu-bar GUI, meeting transcription, a mobile app, or team
sharing, other tools fit better. VoiceClip is terminal-launched, solo,
local-first, and deliberately small.

---

## How it works

1. Hold **Right Option (⌥)** — you hear a *tink*.
2. Speak: *"Running 10 minutes late to the standup"*.
3. Release — you hear a *pop*, then a *chime*.
4. Text appears where your cursor was. Your existing clipboard is preserved.

Works in any app. On Apple Silicon (any Mac from 2021 or newer), typical
end-to-end latency for a 5-10 second clip is **2-4 seconds** — model
warm-up + Whisper inference + paste. Cold-start on first run takes longer
because the model is downloading.

---

## Install

### Recommended: `install.sh`

```bash
git clone https://github.com/asmehra95/VoiceClip.git && cd VoiceClip
bash install.sh
```

The installer checks for Apple Silicon + Python 3.10-3.13 + ffmpeg, creates
a virtualenv at `~/.voiceclip/.venv`, installs pinned dependencies, and
drops a `voiceclip` symlink in `~/.local/bin`. Takes about 2 minutes on
first run. Re-running is safe — your config and history are preserved.

> **Python 3.14 note.** `pyobjc` has a known incompatibility with Python
> 3.14 that can break the hotkey listener. Use Python 3.10-3.13 for now.
> The app will run on 3.14 but the reflection hotkey may silently fail.

### What you need

- **Apple Silicon Mac** (M1 / M2 / M3 / M4)
- **Python 3.10-3.13** — check with `python3 --version`
- **ffmpeg** — `brew install ffmpeg`, or the installer prompts to install it
- **Terminal access** and comfort with one or two shell commands

### macOS permissions (one-time)

After install, macOS needs two permissions:

1. **Accessibility** — lets VoiceClip paste into other apps.
   *System Settings → Privacy & Security → Accessibility → add your terminal.*
2. **Microphone** — prompted automatically the first time you record.
   *System Settings → Privacy & Security → Microphone → your terminal.*

Then run `voiceclip` and try the hotkey.

### Manual install

```bash
git clone https://github.com/asmehra95/VoiceClip.git && cd VoiceClip
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python transcribe.py
```

### Update

Pull the latest code and re-run the installer:

```bash
git pull && bash install.sh
```

### Uninstall

```bash
rm -rf ~/.voiceclip ~/.local/bin/voiceclip
# Optional — also remove cached AI models (3-10 GB)
rm -rf ~/.cache/huggingface
```

---

## Configuration

VoiceClip works out of the box. Everything below is optional.

Settings live in `~/.voiceclip/config.json`, created on first run. Edit it
with any text editor, or use the Settings tab in the web viewer
(`voiceclip view`).

The most common tweaks:

```json
{
  "model": "large-v3-turbo",
  "english_only": true,
  "persona": "default",
  "hotkey": "alt_r",
  "hotkey_mode": "hold"
}
```

For quick one-off overrides, environment variables beat the config file:

```bash
VOICECLIP_MODEL=small voiceclip          # Faster, less accurate
VOICECLIP_PERSONA=engineering voiceclip  # Technical vocabulary
VOICECLIP_HOTKEY=f5 voiceclip            # Different key
```

Every top-level config key has a corresponding `VOICECLIP_<UPPER>` env var.
See [`config.default.json`](config.default.json) for the full schema with
defaults.

---

## Hotkey

Default: hold **Right Option (⌥)** to record, release to stop.

Supported keys: `alt_r`, `alt_l`, `ctrl_r`, `ctrl_l`, `shift_r`, `shift_l`,
`cmd_r`, `cmd_l`, `caps_lock`, `f1`–`f12`, `space`, `esc`, or any single
letter.

### Toggle mode — useful if you can't hold keys

If holding is uncomfortable — repetitive strain injury, motor disability,
long dictation sessions, or just preference — switch to toggle mode. Press
once to start, press again to stop:

```json
{ "hotkey_mode": "toggle" }
```

Toggle mode works exactly the same way for reflections (`reflection_hotkey_mode`).

---

## Personas

Personas bias Whisper toward your vocabulary. An engineering persona knows
"S3" and "DynamoDB". A medical persona knows "prognosis" and "dosage".

| Persona | Best for | Example fixes |
|---|---|---|
| `default` | General use | No domain bias |
| `engineering` | Technical work | "s three" → "S3", "e c two" → "EC2" |
| `casual` | Chat, messaging | Conversational tone |
| `medical` | Clinical notes | Medical terminology |

Switch in config:

```json
{ "persona": "engineering" }
```

Or define your own:

```json
{
  "personas": {
    "my-team": {
      "prompt": "CRDT, vector clock, eventual consistency, saga, idempotency",
      "dictionary": {
        "c r d t": "CRDT",
        "saga": "saga"
      }
    }
  }
}
```

- `prompt` biases the speech model toward your vocabulary.
- `dictionary` is a case-insensitive find-and-replace applied after transcription.
- The global `dictionary` in your config always applies on top of the active persona.

---

## Models

VoiceClip downloads the Whisper model on first run. Pick based on your
speed vs accuracy preference:

| Model | Latency* | Accuracy | RAM |
|---|---|---|---|
| `tiny` | <1s | Fair | 1 GB |
| `base` | ~1s | Good | 1 GB |
| `small` | ~2s | Great | 2 GB |
| **`large-v3-turbo`** ★ | **~3s** | **Excellent** | **3 GB** |
| `large-v3` | ~6s | Excellent | 6 GB |

\* Warm-cache end-to-end latency for a ~10 second clip on an M2 MacBook
Pro. First run of each model downloads weights; your hardware and clip
length will vary. ★ = default.

### Multilingual

VoiceClip defaults to English. For other languages:

```json
{ "english_only": false }
```

Whisper auto-detects the language. Works best with `large-v3-turbo` or
`large-v3`.

---

## Tips

- Wait for the **tink** before speaking — that's when recording starts.
- Speak naturally. Whisper handles pauses and filler words well.
- The MacBook mic beats Bluetooth headsets. Bluetooth audio switches to
  a low-quality mode for mic input; the MacBook mic stays high-quality.
- Keep clips under 30 seconds. For longer dictation, record in chunks.
- Use personas for domain work. The vocabulary bias is a real accuracy boost.

---

## Reflections — a private journal for thoughts

Dictation pastes into whatever app you're in. **Reflections** do the
opposite: you record, the text is saved to a local database, and nothing
is pasted or copied. Different hotkey, different start/done chimes, same
record-speak-release flow.

It's for the "I just want to remember this thought" moment, without
context-switching to a notes app.

### Enable it

Two keys in `~/.voiceclip/config.json`:

```json
{
  "history": true,
  "reflection_hotkey": "f6"
}
```

Pick any key that's different from your main hotkey. Restart VoiceClip.

Now:

- Hold **⌥** (main hotkey) — normal dictation, pastes at the cursor.
- Hold **F6** (reflection hotkey) — different chimes, text saved privately,
  nothing touches the clipboard.

### Browse your reflections

From the terminal:

```bash
voiceclip history                           # Everything, newest first
voiceclip history --reflections             # Only reflections
voiceclip history --reflections --today     # Today's reflections
voiceclip history --search "standup"        # Full-text search
voiceclip history --reflect-last            # Promote last dictation to a reflection
```

Or open the web viewer for a real UI:

```bash
voiceclip view
```

### What gets stored

For every entry (dictation or reflection):

- Timestamp, duration, the text itself
- The app you were in when you recorded ("Slack", "Xcode") — no permission
  needed, it's public info

That's the complete list. No audio retained, no window titles, no URLs, no
screen contents, no keystrokes outside the hotkey. The database lives at
`~/.voiceclip/history.db`, permissions `0600` (readable only by you).

### Retention

- Transcriptions age out after `history_max_days` days (default 30).
- Reflections are kept forever unless you set `reflection_max_days` to a
  positive number.
- Turning history off never deletes anything — your data sits there until
  you run `voiceclip history --clear`.

---

## Web viewer

```bash
voiceclip view
```

A local web page opens at `http://localhost:8723`. Localhost only — never
binds to your network.

The viewer gives you:

- **Today by default.** Day navigation, date picker, search across all days.
- **Reflections first.** Warm background, generous type. Dictations
  collapse under "Show N transcriptions."
- **Inline editing.** Click any entry to fix typos. Cmd+Enter saves.
- **One-click copy** on any entry. One-click promote a dictation to a
  reflection.
- **Dark mode follows your system** automatically.
- **Settings tab** for config changes that would otherwise require editing
  JSON — Whisper model, hotkeys, personas, custom vocabulary, AI providers.
- **Downloaded models** panel to see HuggingFace cache size and delete
  models you don't use.

---

## Daily summaries and timelines (opt-in)

If you enable history, an LLM can read your day's dictations and reflections
and produce two complementary views:

- **Summary** — a 2-4 sentence narrative recap.
- **Timeline** — a chronological walkthrough broken into morning / afternoon
  / evening / late night.

Off by default. To enable:

```json
{
  "summaries": {
    "provider": "local",
    "local_model": "mlx-community/Qwen2.5-7B-Instruct-4bit"
  }
}
```

Three provider options:

| Provider | Where it runs | Needs | Cost |
|---|---|---|---|
| `local` | Your Mac (mlx-lm) | `pip install mlx-lm`, ~4 GB model | Free |
| `openai` | OpenAI API | `OPENAI_API_KEY`, `pip install openai` | Pay-per-token |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY`, `pip install anthropic` | Pay-per-token |

Rough cost estimate for cloud summaries: a typical day of 20-50 entries is
~1-3k input tokens. At `gpt-4o-mini` rates (~$0.15/1M input, $0.60/1M
output) that's less than **a cent per day**. Your mileage will vary; the
Patterns feature can send up to 7 days of context in one call, so its cost
per run is higher.

Cloud providers cache API requests for ~30 days for abuse detection. Your
reflections, topics, or summaries may sit on their servers for that window.
Review each provider's data-use policy before enabling.

Generate from the CLI:

```bash
voiceclip summarize --day today
voiceclip summarize --day yesterday
voiceclip summarize --day 2026-04-23 --force   # Regenerate cached
```

Or just open the viewer — each day has a "Generate" button that calls the
same thing. Past days cache forever; today regenerates when new entries
arrive.

---

## Patterns — longitudinal view across recent days

The Patterns tab in the viewer reads the past 7 days of your reflections +
cached summaries + cached timelines and produces:

- **Occupied with** — a few sentences on what you spent time on.
- **Recurring themes** — short phrases, each grounded in a verbatim
  quote from a reflection.
- **Suggested topics to research** — things you kept circling back to,
  each with a grounding quote.

Every quote must be a direct substring of an actual reflection — the LLM
can't fabricate themes. If nothing qualifies, it returns an empty list,
which is the correct answer.

The Patterns feature is most useful when you've been generating daily
summaries alongside your reflections. The viewer footer shows how much
pre-digested input the model had access to (e.g. `3/7 daily summaries ·
1/7 daily timelines`) so you can tell whether the output is running on
full context or partial.

This is the most privacy-sensitive feature in VoiceClip — it reads a
week of your journal in a single prompt. The default provider is `local`
for that reason. Off entirely until you enable it.

---

## Privacy & data flow

VoiceClip is transparent about what crosses the network. Full picture:

### Always local (never leaves your Mac)

- **Audio capture and transcription.** Whisper runs on your Apple Silicon
  GPU. The audio WAV is deleted from `/tmp` immediately after transcription.
- **Clipboard and paste.** `pbcopy` + simulated Cmd-V, all local.
- **History database.** SQLite at `~/.voiceclip/history.db`, permissions
  `0600`, never synced, never transmitted.
- **Active-app context capture** (only when history is enabled). Uses
  `osascript` to read the frontmost app *name*. Nothing else about the app.

### Opt-in local (runs on your Mac if you enable it)

- **Daily summaries** with `summaries.provider = "local"`.
- **Timelines** with `summaries.provider = "local"` (shares the summaries config).
- **Patterns coach** with `patterns.provider = "local"`. Default is local
  because it reads the widest window of your journal.

### Opt-in cloud (sends data to a third-party API)

These features all default to `provider = "none"`. If you change any of
them to `openai` or `anthropic`, this is what goes off your Mac:

| Feature | What is sent |
|---|---|
| **Summaries** | Every entry for the target day. |
| **Timelines** | Same as summaries (they share the config). |
| **Research** | The research topic you typed or dictated. If the model uses its web-search tool, the topic also goes to the provider's search backend. Research has no local provider — it depends on live web search, which local models can't do reliably. |
| **Patterns** | Up to 7 days of reflections + cached daily summaries + cached timelines in one prompt. |

API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are read from environment
variables only. Never stored in `config.json`, logs, or the database.

**Cloud retention.** OpenAI and Anthropic cache API requests for about 30
days for abuse detection. Review their data-use policies before enabling.

### In-app consent

When you flip any provider to a cloud option, VoiceClip shows a consent
banner the next time you open the viewer — with the exact provider, model,
and a description of what will be sent. The banner stays up until you
dismiss it. Re-fires if you change provider or model again.

### What is NOT captured anywhere

- Window titles (off by default; requires explicit opt-in)
- Browser URLs
- Calendar events
- Screen contents
- Keystrokes outside the hotkey
- Telemetry or analytics — VoiceClip never phones home

---

## Troubleshooting

Run this first for any problem:

```bash
voiceclip doctor
```

It checks your system, permissions, audio device, database integrity,
schema version, optional providers, and model caches. One command, most
problems surfaced.

### Common issues

| Problem | Solution |
|---|---|
| "This process is not trusted" | Grant Accessibility permission to your terminal, then restart it. |
| No audio captured | System Settings → Sound → Input, pick the right mic. Or check the recorder's startup log line. |
| First word cut off | Wait for the *tink* before speaking. |
| Says "Thank you" randomly | Recording was mostly silence — speak louder or check mic levels. |
| Slow transcription | Try `VOICECLIP_MODEL=small voiceclip`, or close other GPU-heavy apps. |
| First run frozen | Whisper model is downloading (~3 GB). A spinner shows progress. |
| Bad Bluetooth headset accuracy | Bluetooth mics use a low-quality mode. Use the MacBook mic for input. |
| Reflection hotkey doesn't fire | Check `voiceclip doctor` output. On Python 3.14, pyobjc has a known compatibility bug — use 3.10-3.13 instead. |

---

## Development

This section is for anyone modifying the code.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Run the tests
pytest

# Lint
ruff check voiceclip/ tests/

# Type-check the frontend JS (optional, requires npx)
npx -p typescript tsc --noEmit --project voiceclip/static/jsconfig.json
```

The `[dev]` extra pulls in `ruff` and `pytest`. Cloud LLM providers and
multimodal models live behind `[cloud]` and `[vlm]` extras respectively —
install only what you need.

### Project structure

```
voiceclip/
  __init__.py       # Version
  __main__.py       # Entry point, startup, hotkey handlers
  config.py         # Config loader (JSON + env overrides)
  config_io.py      # Shared config read/merge/write helpers
  consent.py        # Cloud-provider consent banner logic
  doctor.py         # `voiceclip doctor` health check
  formatter.py      # Text cleanup and dictionary substitutions
  history.py        # SQLite storage, FTS5, day views, research queue
  hotkey.py         # Hotkey handler (hold + toggle modes, profiles)
  llm_provider.py   # Unified local/openai/anthropic transport
  macos.py          # Clipboard, paste, sounds, permissions
  models.json       # Curated LLM model list for the picker
  onboard.py        # First-run walkthrough
  patterns.py       # Longitudinal coach
  recorder.py       # Audio capture (separate child process)
  researcher.py     # Research brief generator
  summarizer.py     # Daily summary + timeline generator
  text_quality.py   # Garbage-transcription detector
  transcriber.py    # Whisper inference on GPU
  utils.py          # Shared utilities
  viewer.py         # Local web viewer + API endpoints
  static/           # Viewer CSS / JS / HTML + jsconfig.json
transcribe.py       # Launcher
install.sh          # Installer
pyproject.toml      # Package metadata + tool config (ruff, pytest)
config.default.json # Default config with examples
requirements.txt    # Core runtime deps (kept in sync with pyproject.toml)
tests/              # Unit tests — 367 passing, run via pytest
BACKLOG.md          # Known issues and deferred work
```

### Contributing

This project doesn't currently have a formal contribution process. If you
want to change something, the BACKLOG.md is a good place to see what's
already on the list. Small, focused PRs with clear commit messages work
best.

---

## License

MIT. See [LICENSE](LICENSE).

Built with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper),
[pynput](https://github.com/moses-palmer/pynput), and
[sounddevice](https://python-sounddevice.readthedocs.io/).
