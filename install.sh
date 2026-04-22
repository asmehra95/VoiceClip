#!/bin/bash
# VoiceClip installer for macOS (Apple Silicon)
# Usage: curl -sSL <url> | bash  OR  bash install.sh

set -e

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
cp "$SCRIPT_DIR/transcribe.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"

# Create virtual environment
echo "🐍 Setting up Python environment..."
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi

# Install dependencies
echo "📦 Installing dependencies (this may take a few minutes)..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# Create launcher script
cat > "$INSTALL_DIR/voiceclip" << 'LAUNCHER'
#!/bin/bash
INSTALL_DIR="$HOME/.voiceclip"
source "$INSTALL_DIR/.venv/bin/activate"
python "$INSTALL_DIR/transcribe.py" "$@"
LAUNCHER
chmod +x "$INSTALL_DIR/voiceclip"

# Add to PATH via symlink
LINK_DIR="$HOME/.local/bin"
mkdir -p "$LINK_DIR"
ln -sf "$INSTALL_DIR/voiceclip" "$LINK_DIR/voiceclip"

echo ""
echo "=================================================="
echo "  ✅ VoiceClip installed!"
echo "=================================================="
echo ""
echo "  To run:  voiceclip"
echo "           (or: $INSTALL_DIR/voiceclip)"
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
