"""One-time cloud-provider consent banner.

When a user switches any `*.provider` in their config to `openai` or
`anthropic`, this module prints a loud visible warning the next time
VoiceClip starts. Once acknowledged (implicitly — by the fact that VoiceClip
started at all), we record an ack marker in ~/.voiceclip/cloud_ack.json so
subsequent launches don't nag.

Re-acks only trigger when the provider+model combination changes (e.g. user
flips from openai to anthropic, or upgrades model). This keeps the banner
rare enough to be respected.

Shape of ack file:
  {
    "summaries": {"provider": "openai", "model": "gpt-4o-mini"},
    "research":  {"provider": "anthropic", "model": "claude-haiku-4-5"}
  }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from voiceclip import config

log = logging.getLogger(__name__)


def _ack_path() -> Path:
    return Path(config.CONFIG_DIR) / "cloud_ack.json"


def _load_acks() -> dict:
    p = _ack_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_acks(acks: dict):
    p = _ack_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(acks, indent=2))
        os.chmod(p, 0o600)
    except OSError as e:
        log.warning("Could not persist cloud_ack.json: %s", e)


def _current_cloud_state() -> dict:
    """Which features are currently pointed at cloud providers, and with
    which model? Returns {} if none."""
    state = {}
    if config.SUMMARIES_PROVIDER in ("openai", "anthropic"):
        model = (
            config.SUMMARIES_OPENAI_MODEL if config.SUMMARIES_PROVIDER == "openai"
            else config.SUMMARIES_ANTHROPIC_MODEL
        )
        state["summaries"] = {"provider": config.SUMMARIES_PROVIDER, "model": model}
    if config.RESEARCH_PROVIDER in ("openai", "anthropic"):
        model = (
            config.RESEARCH_OPENAI_MODEL if config.RESEARCH_PROVIDER == "openai"
            else config.RESEARCH_ANTHROPIC_MODEL
        )
        state["research"] = {"provider": config.RESEARCH_PROVIDER, "model": model}
    if config.PATTERNS_PROVIDER in ("openai", "anthropic"):
        model = (
            config.PATTERNS_OPENAI_MODEL if config.PATTERNS_PROVIDER == "openai"
            else config.PATTERNS_ANTHROPIC_MODEL
        )
        state["patterns"] = {"provider": config.PATTERNS_PROVIDER, "model": model}
    return state


# Messaging text — kept in one place so it reads consistently wherever it
# fires (CLI startup banner, viewer startup log, viewer in-UI banner).
# Exposed without an underscore so the viewer API can return the same
# copy the CLI banner uses.
FEATURE_DATA_SENT = {
    "summaries": (
        "Your entries for the target day (every transcription and reflection "
        "in that day's window)"
    ),
    "research":  (
        "The research topic string you typed or dictated. If the model uses "
        "its web search tool, that topic also goes to the search backend"
    ),
    "patterns":  (
        "Up to a week of your reflections and daily summaries in a single prompt"
    ),
}


def _banner(new_features: dict) -> str:
    lines = [
        "",
        "  ┌─────────────────────────────────────────────────────────────",
        "  │  ⚠️  Cloud provider change detected",
        "  ├─────────────────────────────────────────────────────────────",
    ]
    for feature, info in sorted(new_features.items()):
        lines.append(f"  │  {feature:<10}  →  {info['provider']} · {info['model']}")
        lines.append(f"  │              {FEATURE_DATA_SENT.get(feature, '?')}")
    lines.extend([
        "  │",
        "  │  When these features run, data leaves your Mac and goes to",
        "  │  the chosen provider. Providers typically retain API data for",
        "  │  ~30 days for abuse detection.",
        "  │",
        "  │  Flip the provider back to \"none\" in ~/.voiceclip/config.json",
        "  │  if this was unintentional.",
        "  └─────────────────────────────────────────────────────────────",
        "",
    ])
    return "\n".join(lines)


def pending_acks() -> dict:
    """Return features whose current cloud config hasn't yet been acked.

    Pure read — does NOT mutate the ack file. Lets the viewer show a
    visible banner before auto-acking. The startup `check_and_warn()`
    auto-acks as a side effect of printing, which is fine for the CLI
    but invisible in the web UI.

    Shape matches `_current_cloud_state()` but only includes entries
    that differ from the stored ack file.
    """
    current = _current_cloud_state()
    if not current:
        return {}
    acks = _load_acks()
    pending = {}
    for feature, info in current.items():
        if acks.get(feature) != info:
            pending[feature] = info
    return pending


def record_acks(features: list[str] | None = None) -> dict:
    """Persist acks for the given features (or all pending if omitted).

    Returns the updated ack dict. Idempotent — safe to call multiple
    times. Used by the viewer's /api/consent/ack endpoint after the user
    dismisses the in-UI banner.
    """
    current = _current_cloud_state()
    acks = _load_acks()
    to_record = (
        {k: v for k, v in current.items() if features is None or k in features}
    )
    if not to_record:
        return acks
    acks.update(to_record)
    _save_acks(acks)
    return acks


def check_and_warn() -> str | None:
    """Compare current cloud config to the last-acknowledged set. If any
    feature has newly switched to cloud OR changed provider/model, print a
    banner and update the ack file. Returns the banner string (or None).

    Called at startup — by __main__._run_voiceclip (core daemon) and
    viewer.serve (web viewer).
    """
    current = _current_cloud_state()
    if not current:
        # All providers are 'none'. Nothing to warn about, nothing to record.
        return None

    acks = _load_acks()
    new_or_changed: dict = {}
    for feature, info in current.items():
        prior = acks.get(feature)
        if prior != info:
            new_or_changed[feature] = info

    if not new_or_changed:
        return None

    banner = _banner(new_or_changed)
    # Print immediately so the user sees it even if the caller doesn't
    print(banner)

    # Update the ack file to reflect the current state
    acks.update(current)
    _save_acks(acks)
    return banner
