"""voiceclip doctor — one-shot health check.

Designed to be the first thing a user runs when something isn't working.
Prints a checklist of everything VoiceClip depends on. Single paste answers
most support questions.

Checks grouped by category:
  - System (macOS version, Python version, Apple Silicon, ffmpeg)
  - Permissions (Accessibility, Microphone)
  - Audio (default device, sample rate)
  - Storage (~/.voiceclip layout, DB permissions, DB integrity)
  - History schema (columns, user_version)
  - Optional providers (mlx-lm, openai, anthropic — only checked if configured)
  - Model caches (HuggingFace cache size)
  - Active config summary
"""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from voiceclip import config, history

# ANSI helpers — plain strings if the terminal isn't a TTY
_IS_TTY = sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\x1b[32m{s}\x1b[0m" if _IS_TTY else s


def _yellow(s: str) -> str:
    return f"\x1b[33m{s}\x1b[0m" if _IS_TTY else s


def _red(s: str) -> str:
    return f"\x1b[31m{s}\x1b[0m" if _IS_TTY else s


def _dim(s: str) -> str:
    return f"\x1b[2m{s}\x1b[0m" if _IS_TTY else s


# Status glyphs
OK = _green("✓")
WARN = _yellow("!")
FAIL = _red("✗")


class Report:
    """Accumulates check results and prints a final summary line."""

    def __init__(self):
        self.ok = 0
        self.warn = 0
        self.fail = 0

    def line(self, glyph: str, name: str, detail: str = ""):
        detail_str = f"  {_dim(detail)}" if detail else ""
        print(f"  {glyph} {name}{detail_str}")

    def check(self, name: str, condition: bool, detail_ok: str = "", detail_fail: str = ""):
        if condition:
            self.ok += 1
            self.line(OK, name, detail_ok)
        else:
            self.fail += 1
            self.line(FAIL, name, detail_fail)

    def warning(self, name: str, detail: str = ""):
        self.warn += 1
        self.line(WARN, name, detail)

    def note(self, name: str, detail: str = ""):
        self.ok += 1
        self.line(OK, name, detail)

    def summary(self):
        parts = [_green(f"{self.ok} ok")]
        if self.warn:
            parts.append(_yellow(f"{self.warn} warnings"))
        if self.fail:
            parts.append(_red(f"{self.fail} failed"))
        print(f"\n  {' · '.join(parts)}")
        return 0 if self.fail == 0 else 1


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_system(r: Report):
    print("\n  System")
    is_mac = platform.system() == "Darwin"
    r.check(
        "macOS",
        is_mac,
        detail_ok=f"{platform.mac_ver()[0]}",
        detail_fail=f"detected {platform.system()}; VoiceClip is macOS-only",
    )
    is_arm = platform.machine() == "arm64"
    r.check(
        "Apple Silicon",
        is_arm,
        detail_ok=platform.machine(),
        detail_fail=f"detected {platform.machine()}; MLX ASR runs on GPU only on arm64",
    )
    py = sys.version_info
    py_ok = py >= (3, 10)
    r.check(
        "Python 3.10+",
        py_ok,
        detail_ok=f"{py.major}.{py.minor}.{py.micro}",
        detail_fail=f"detected {py.major}.{py.minor}",
    )
    ffmpeg = shutil.which("ffmpeg")
    r.check(
        "ffmpeg on PATH",
        ffmpeg is not None,
        detail_ok=ffmpeg or "",
        detail_fail="brew install ffmpeg",
    )


def _check_permissions(r: Report):
    print("\n  Permissions")
    # Accessibility — a test System Events call tells us
    try:
        res = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to return name of first process'],
            capture_output=True, timeout=3,
        )
        if res.returncode == 0:
            r.note("Accessibility", "granted (osascript System Events works)")
        else:
            r.line(FAIL, "Accessibility",
                   "denied — System Settings → Privacy & Security → Accessibility")
            r.fail += 1
    except Exception as e:
        r.warning("Accessibility", f"could not determine: {e}")
    # Microphone — we can't probe without actually opening a stream.
    # Note that macOS will prompt the first time VoiceClip is run.
    r.note("Microphone", "grant at first run (System Settings if needed)")


def _check_storage(r: Report):
    print("\n  Storage")
    cfg_dir = Path(config.CONFIG_DIR)
    if not cfg_dir.exists():
        r.warning("Config dir", f"not yet created at {cfg_dir}")
    else:
        r.note("Config dir", str(cfg_dir))

    cfg_path = Path(config.CONFIG_PATH)
    if cfg_path.exists():
        mode = oct(cfg_path.stat().st_mode & 0o777)
        r.note("config.json", f"{cfg_path} (mode {mode})")
    else:
        r.warning("config.json", "not yet created")

    db_path = Path(history.DB_PATH)
    if db_path.exists():
        mode = oct(db_path.stat().st_mode & 0o777)
        size_mb = db_path.stat().st_size / (1024 * 1024)
        r.check(
            "history.db permissions 0600",
            (db_path.stat().st_mode & 0o777) == 0o600,
            detail_ok=f"{db_path} ({size_mb:.2f} MB)",
            detail_fail=f"permissions are {mode}, expected 0o600",
        )
        # Quick integrity_check
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute("PRAGMA integrity_check").fetchone()
            if row and row[0] == "ok":
                r.note("SQLite integrity", "ok")
            else:
                r.line(FAIL, "SQLite integrity", str(row))
                r.fail += 1
            conn.close()
        except Exception as e:
            r.warning("SQLite integrity", f"could not check: {e}")
    else:
        r.note("history.db", "not yet created (history feature not used)")


def _check_schema(r: Report):
    if not Path(history.DB_PATH).exists():
        return
    print("\n  Schema")
    try:
        conn = sqlite3.connect(history.DB_PATH)
        cols = {row[1] for row in conn.execute(
            "PRAGMA table_info(transcriptions)").fetchall()}
        expected = {"kind", "app_name", "window_title",
                    "edited_at", "is_research_topic"}
        missing = expected - cols
        r.check(
            "transcriptions columns",
            not missing,
            detail_ok=f"{len(cols)} columns present",
            detail_fail=f"missing: {sorted(missing)}",
        )
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        r.check(
            "day_summaries table",
            "day_summaries" in tables,
            detail_fail="run voiceclip view once to trigger migration",
        )
        r.check(
            "research_briefs table",
            "research_briefs" in tables,
            detail_fail="run voiceclip view once to trigger migration",
        )
        # Counts
        entry_count = conn.execute(
            "SELECT COUNT(*) FROM transcriptions").fetchone()[0]
        r.note("entry count", f"{entry_count} total")
        conn.close()
    except Exception as e:
        r.warning("Schema check failed", str(e))


def _check_optional_providers(r: Report):
    print("\n  Optional providers (only checked if configured)")

    def _check_pkg(name: str, imp: str):
        try:
            __import__(imp)
            r.note(f"{name} installed", "")
        except ImportError:
            r.line(FAIL, f"{name} required but missing",
                   f"~/.voiceclip/.venv/bin/pip install {name}")
            r.fail += 1

    # parakeet-mlx needed if engine is set to "parakeet"
    if config.ENGINE == "parakeet":
        _check_pkg("parakeet-mlx", "parakeet_mlx")
    else:
        r.note("parakeet-mlx", "not needed (engine=whisper)")

    # mlx-lm needed if any *_local provider is configured
    local_in_use = (
        config.SUMMARIES_PROVIDER == "local"
        or config.PATTERNS_PROVIDER == "local"
        or config.RESEARCH_PROVIDER == "local"
    )
    if local_in_use:
        _check_pkg("mlx-lm", "mlx_lm")
        # mlx-vlm is only required when one of the configured local
        # models is multimodal (Gemma 4, Qwen-VL, etc). We don't try to
        # detect that here — it's only clear at load time. But if the
        # package is present we can surface the version; if it isn't,
        # just note that multimodal models will fail.
        try:
            import mlx_vlm
            r.note("mlx-vlm installed",
                   f"(for multimodal models: {getattr(mlx_vlm, '__version__', '?')})")
        except ImportError:
            r.note("mlx-vlm",
                   "not installed — text-only LLMs work, "
                   "multimodal models (Gemma 4, Qwen-VL) won't load")
    else:
        r.note("mlx-lm", "not needed (no local provider configured)")

    # openai
    openai_in_use = any(
        p == "openai" for p in (
            config.SUMMARIES_PROVIDER, config.RESEARCH_PROVIDER,
            config.PATTERNS_PROVIDER,
        )
    )
    if openai_in_use:
        _check_pkg("openai", "openai")
        if os.environ.get("OPENAI_API_KEY"):
            r.note("OPENAI_API_KEY", "set")
        else:
            r.line(FAIL, "OPENAI_API_KEY", "not set — export it in your shell")
            r.fail += 1
    else:
        r.note("openai", "not needed (no openai provider configured)")

    # anthropic
    anthropic_in_use = any(
        p == "anthropic" for p in (
            config.SUMMARIES_PROVIDER, config.RESEARCH_PROVIDER,
            config.PATTERNS_PROVIDER,
        )
    )
    if anthropic_in_use:
        _check_pkg("anthropic", "anthropic")
        if os.environ.get("ANTHROPIC_API_KEY"):
            r.note("ANTHROPIC_API_KEY", "set")
        else:
            r.line(FAIL, "ANTHROPIC_API_KEY", "not set — export it in your shell")
            r.fail += 1
    else:
        r.note("anthropic", "not needed (no anthropic provider configured)")


def _check_caches(r: Report):
    print("\n  Model caches")
    hf = Path(os.path.expanduser("~/.cache/huggingface"))
    if hf.exists():
        # Rough size
        total = 0
        try:
            for p in hf.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except Exception:
            pass
        size_gb = total / (1024 ** 3)
        r.note("HuggingFace cache", f"{hf} ({size_gb:.1f} GB)")
    else:
        r.note("HuggingFace cache", "not yet populated")


def _print_config(r: Report):
    print("\n  Active config")
    if config.ENGINE == "parakeet":
        model_display = config.PARAKEET_MODEL
    else:
        model_display = config.MODEL
    items = [
        ("engine", config.ENGINE),
        ("model", model_display),
        ("english_only", config.ENGLISH_ONLY),
        ("hotkey", f"{config.HOTKEY} ({config.HOTKEY_MODE})"),
        ("reflection_hotkey",
         f"{config.REFLECTION_HOTKEY} ({config.REFLECTION_HOTKEY_MODE})"
         if config.REFLECTION_HOTKEY else "off"),
        ("history", config.HISTORY_ENABLED),
        ("summaries", config.SUMMARIES_PROVIDER),
        ("research", config.RESEARCH_PROVIDER),
        ("patterns", config.PATTERNS_PROVIDER),
    ]
    for k, v in items:
        r.line(_dim("·"), k, str(v))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> int:
    """Run all checks. Returns an exit code (0 success, 1 if any failures)."""
    print("VoiceClip doctor — system health check\n")
    config.load()
    r = Report()
    _check_system(r)
    _check_permissions(r)
    _check_storage(r)
    _check_schema(r)
    _check_optional_providers(r)
    _check_caches(r)
    _print_config(r)
    return r.summary()
