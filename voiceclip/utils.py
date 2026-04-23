"""Shared utilities for VoiceClip."""

import os


def safe_unlink(path):
    """Delete a file if it exists, ignoring errors."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
