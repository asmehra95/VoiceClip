#!/bin/bash
# VoiceClip installer for macOS (Apple Silicon)
# Usage: curl -sSL <url> | bash  OR  bash install.sh

set -e

# Refuse to run without a real HOME — installer writes to $HOME/.voiceclip
# and later does a `rm -rf "$INSTALL_DIR/voiceclip"`. An empty or unset
# $HOME would make that `rm -rf /.voiceclip/voiceclip`.
if [[ -z "${HOME:-}" || "$HOME" == "/" ]]; then
    echo "❌ \$HOME is empty or root. Refusing to install. Exiting."
    exit 1
fi

INSTALL_DIR="$HOME/.voiceclip"
VENV_DIR="$INSTALL_DIR/.venv"

echo "=================================================="
echo "  🎙️  VoiceClip Installer"
echo "  Local voice → clipboard on Apple Silicon"
echo "=================================================="

# --- Check prerequisites ---

# macOS only
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ VoiceClip only works on macOS. Exiting."
    exit 1
fi

# Apple Silicon only
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "❌ VoiceClip requires Apple Silicon (M1/M2/M3/M4). Exiting."
    exit 1
fi

# Python 3.10+ required
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found. Install with: brew install python@3.12"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    echo "❌ Python 3.10+ required (found $PY_VERSION). Install with: brew install python@3.12"
    exit 1
fi
echo "✅ Python $PY_VERSION"

# ffmpeg (needed by mlx-whisper for audio decoding)
if ! command -v ffmpeg &>/dev/null; then
    echo "📦 Installing ffmpeg (required by mlx-whisper)..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "❌ ffmpeg not found and Homebrew not available."
        echo "   Install Homebrew: https://brew.sh"
        echo "   Then run: brew install ffmpeg"
        exit 1
    fi
fi
echo "✅ ffmpeg"

# --- Install VoiceClip ---

echo ""
echo "📁 Installing to $INSTALL_DIR..."

# Clone or copy files
if [[ -d "$INSTALL_DIR" ]]; then
    echo "   Updating existing installation..."
else
    mkdir -p "$INSTALL_DIR"
fi

# Copy project files (works whether run from repo or standalone)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Remove old source files before copying to avoid nested directory issues
rm -rf "$INSTALL_DIR/voiceclip"
cp -r "$SCRIPT_DIR/voiceclip" "$INSTALL_DIR/voiceclip"
cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"

# Clean up any leftover transcribe.py from previous installs — older
# versions shipped this 5-line launcher, replaced by `python -m voiceclip`.
rm -f "$INSTALL_DIR/transcribe.py"

# Copy default config if user doesn't have one yet (preserve existing config)
if [[ ! -f "$INSTALL_DIR/config.json" ]]; then
    cp "$SCRIPT_DIR/config.default.json" "$INSTALL_DIR/config.json"
    echo "   Created default config.json"
else
    echo "   Keeping existing config.json"
fi

# Create virtual environment
echo "🐍 Setting up Python environment..."
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi

# Install dependencies
echo "📦 Installing dependencies (this may take a few minutes)..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# --- Fast speech engine (prebuilt whisper.cpp with Core ML) ---------------
# Downloads a small self-contained binary plus the turbo model so dictation
# runs on the GPU + Neural Engine with sub-second latency. Every step is
# optional: if anything fails (offline, release missing, checksum mismatch)
# VoiceClip silently falls back to the pip-installed mlx-whisper engine.
# Re-running is safe — existing files are kept.

echo ""
echo "⚡ Setting up the fast speech engine..."
FAST_ENGINE_OK=true
BIN_DIR="$INSTALL_DIR/bin"
MODELS_DIR="$INSTALL_DIR/models"
RELEASE_BASE="https://github.com/asmehra95/VoiceClip/releases/latest/download"
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
mkdir -p "$BIN_DIR" "$MODELS_DIR"

fetch_engine_binary() {
    # Download a release binary + its checksum, verify, install.
    local name=$1
    [[ -x "$BIN_DIR/$name" ]] && return 0
    curl -fsSL -o "$BIN_DIR/$name.tmp" "$RELEASE_BASE/$name-coreml-arm64" \
        || { rm -f "$BIN_DIR/$name.tmp"; return 1; }
    local expected actual
    expected=$(curl -fsSL "$RELEASE_BASE/$name-coreml-arm64.sha256") || expected=""
    actual=$(shasum -a 256 "$BIN_DIR/$name.tmp" | awk '{print $1}')
    if [[ -z "$expected" || "$expected" != "$actual" ]]; then
        echo "   ⚠️  Checksum mismatch for $name"
        rm -f "$BIN_DIR/$name.tmp"
        return 1
    fi
    chmod +x "$BIN_DIR/$name.tmp"
    xattr -c "$BIN_DIR/$name.tmp" 2>/dev/null || true
    mv "$BIN_DIR/$name.tmp" "$BIN_DIR/$name"
}

if ! fetch_engine_binary whisper-server || ! fetch_engine_binary whisper-cli; then
    FAST_ENGINE_OK=false
fi

if $FAST_ENGINE_OK && [[ ! -f "$MODELS_DIR/ggml-large-v3-turbo.bin" ]]; then
    echo "   ⬇️  Speech model (1.6 GB)..."
    if curl -fL --progress-bar -o "$MODELS_DIR/ggml-large-v3-turbo.bin.partial" \
        "$HF_BASE/ggml-large-v3-turbo.bin"; then
        mv "$MODELS_DIR/ggml-large-v3-turbo.bin.partial" \
           "$MODELS_DIR/ggml-large-v3-turbo.bin"
    else
        rm -f "$MODELS_DIR/ggml-large-v3-turbo.bin.partial"
        FAST_ENGINE_OK=false
    fi
fi

if $FAST_ENGINE_OK && [[ ! -d "$MODELS_DIR/ggml-large-v3-turbo-encoder.mlmodelc" ]]; then
    echo "   ⬇️  Neural Engine encoder (1.2 GB)..."
    if curl -fL --progress-bar -o "$MODELS_DIR/coreml-encoder.zip" \
        "$HF_BASE/ggml-large-v3-turbo-encoder.mlmodelc.zip" \
        && unzip -q -o "$MODELS_DIR/coreml-encoder.zip" -d "$MODELS_DIR"; then
        :
    else
        FAST_ENGINE_OK=false
    fi
    rm -rf "$MODELS_DIR/coreml-encoder.zip" "$MODELS_DIR/__MACOSX"
fi

if $FAST_ENGINE_OK; then
    # First launch of a new binary triggers a one-time Neural Engine
    # compile of the encoder that can take a few minutes and would make
    # the first dictation look frozen. Pay that cost here instead, with
    # an honest progress message.
    echo "   ⚙️  Optimizing the speech model for your Neural Engine"
    echo "      (one-time — can take a few minutes, safe to wait)..."
    "$BIN_DIR/whisper-server" -m "$MODELS_DIR/ggml-large-v3-turbo.bin" \
        --host 127.0.0.1 --port 8177 > /dev/null 2>&1 &
    WARM_PID=$!
    WARM_OK=false
    for _ in $(seq 1 900); do
        kill -0 "$WARM_PID" 2>/dev/null || break
        if curl -s -o /dev/null -w "%{http_code}" \
            http://127.0.0.1:8177/health 2>/dev/null | grep -q 200; then
            WARM_OK=true
            break
        fi
        sleep 1
    done
    kill "$WARM_PID" 2>/dev/null || true
    wait "$WARM_PID" 2>/dev/null || true
    if $WARM_OK; then
        echo "   ✅ Fast engine ready — dictation runs on the GPU + Neural Engine."
    else
        echo "   ⚠️  Warm-up didn't finish; it will complete on first launch."
    fi
else
    echo "   ⚠️  Fast engine unavailable — using the standard engine instead."
    echo "      (Re-run install.sh with a network connection to try again.)"
fi

# Create launcher script (named 'run' to avoid conflict with voiceclip/ directory)
cat > "$INSTALL_DIR/run" << 'LAUNCHER'
#!/bin/bash
INSTALL_DIR="$HOME/.voiceclip"
source "$INSTALL_DIR/.venv/bin/activate"
python -m voiceclip "$@"
LAUNCHER
chmod +x "$INSTALL_DIR/run"

# Add to PATH via symlink (user-facing command is still 'voiceclip')
LINK_DIR="$HOME/.local/bin"
mkdir -p "$LINK_DIR"
ln -sf "$INSTALL_DIR/run" "$LINK_DIR/voiceclip"

echo ""
echo "=================================================="
echo "  ✅ VoiceClip installed!"
echo "=================================================="
echo ""
echo "  To run:  voiceclip"
echo "           (or: $INSTALL_DIR/run)"
echo ""
echo "  ⚠️  First time setup:"
echo "  1. Add ~/.local/bin to your PATH if not already:"
echo "     echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
echo "     source ~/.zshrc"
echo ""
echo "  2. Grant Accessibility permissions:"
echo "     System Settings → Privacy & Security → Accessibility"
echo "     → Add your terminal app (Terminal, iTerm2, etc.)"
echo ""
echo "  3. Grant Microphone permissions:"
echo "     System Settings → Privacy & Security → Microphone"
echo "     → Enable your terminal app"
echo ""
echo "  Then just run: voiceclip"
echo "  Hold Right Option (⌥) to record, release to transcribe!"
echo ""
echo "  📝 Config: $INSTALL_DIR/config.json"
echo "     Edit to change model, add personas, or customize dictionary."
echo ""
