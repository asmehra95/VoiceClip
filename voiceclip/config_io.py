"""Config file I/O — shared read/merge/write helpers.

Used by `onboard` (first-run opt-ins) and `viewer.py` (Settings tab).
One source of truth for:
  - Reading the persisted config.json
  - Writing a partial patch that preserves unrelated keys
  - Shallow-merging nested dicts (e.g. "summaries": {...}) so a patch
    that changes one provider setting doesn't clobber the rest
  - Enforcing 0600 on the config file after every write
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from voiceclip import config

log = logging.getLogger(__name__)


def read_config() -> dict:
    """Return the persisted config as a dict. Returns {} if missing or corrupt."""
    path = Path(config.CONFIG_PATH)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def write_config_patch(patch: dict) -> bool:
    """Merge `patch` into the persisted config. Returns True on success.

    Shallow merge — nested dicts (one level deep) are merged so a patch
    like `{"summaries": {"provider": "local"}}` does not wipe
    `summaries.openai_model` or `summaries.style`.
    """
    current = read_config()

    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(current.get(k), dict):
            current[k] = {**current[k], **v}
        else:
            current[k] = v

    path = Path(config.CONFIG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2))
        os.chmod(path, 0o600)
        return True
    except OSError as e:
        log.warning("Could not update config: %s", e)
        return False
