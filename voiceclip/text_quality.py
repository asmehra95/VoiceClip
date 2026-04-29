"""Heuristics for classifying transcription text quality.

Voice dictation has failure modes that produce garbage entries:

  - Stuck hotkey in toggle mode: Whisper transcribes a long silence or
    ambient noise as a single word repeated dozens of times
    ("and and and and ..." or "machine and machine and machine ...").
  - Audio cable knocked loose: dropouts can make Whisper loop on a
    partial word ("recor recor recor ...").
  - Bluetooth glitches: produce character-level noise ("aaaaa..." or
    occasional punctuation loops).

The LLM summarizer and patterns features pay real cost for these
entries — they eat the context window, pull the model toward nonsense
topics, and occasionally become the dominant "theme" of a summary.

`is_garbage_text()` is a conservative filter. False negatives (garbage
that slips through) are fine — the model can tolerate some noise. False
positives (good entries flagged as garbage) would hide real content, so
the thresholds are tuned to only fire on clearly-degenerate input.

This module is pure — no I/O, no logging, safe to call on any string.
"""

from __future__ import annotations

import re
from collections import Counter


# Minimum length before any check fires. Short entries like "yes" or
# "okay thanks" are never garbage-filtered — they might be meaningful on
# their own.
_MIN_LEN_CHARS = 40

# Dominant-token threshold: if a single word accounts for this fraction
# or more of the total token count, we consider the entry degenerate.
# Tuned at 0.60 based on the "machine and machine..." sample:
# 3 real words out of ~200 tokens = ~1.5% real content — flagged easily.
# A legitimate sentence with a repeated emphasis word (e.g. "no no I
# think no") stays well under 60%.
_DOMINANT_TOKEN_RATIO = 0.60

# Minimum number of tokens before the dominant-token check kicks in.
# "yes yes yes" (3 tokens, 100% dominance) should not trip the filter
# because the user may have intended that. 12 tokens is roughly the
# length of two short sentences — below that we don't have enough
# signal.
_MIN_TOKENS_FOR_DOMINANCE = 12

# Character-repetition threshold: a 5-char-or-longer substring that
# repeats 10+ times consecutively. Covers "aaaaa..." with 10 a's (they
# count as a 1-char repeating, but a=1 char < 5 so that's separately
# caught below) and phrase-level loops like "ok ok ok ok..." (3 chars).
_CHAR_REPEAT_PATTERN = re.compile(r"(.{3,}?)\1{9,}", re.DOTALL)

# Single-character runs: 20+ of the same non-whitespace character in a
# row. Catches "aaaaaaaaaaaaaaaaaaaaaa" or "!!!!!!!!!!!!!!!!!!!!" kinds
# of noise that the phrase pattern above misses.
_CHAR_RUN_PATTERN = re.compile(r"(\S)\1{19,}")


def is_garbage_text(text: str) -> bool:
    """Return True if `text` looks like a transcription failure.

    Conservative: only fires on clearly-degenerate input. Valid text of
    any length and style passes through unchanged.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < _MIN_LEN_CHARS:
        return False

    # Fast path: raw character-level repetition
    if _CHAR_RUN_PATTERN.search(stripped):
        return True
    if _CHAR_REPEAT_PATTERN.search(stripped):
        return True

    # Token-level dominance
    tokens = stripped.lower().split()
    if len(tokens) < _MIN_TOKENS_FOR_DOMINANCE:
        return False
    counts = Counter(tokens)
    _, top_count = counts.most_common(1)[0]
    ratio = top_count / len(tokens)
    if ratio >= _DOMINANT_TOKEN_RATIO:
        return True

    return False


def filter_entries(entries: list[dict]) -> list[dict]:
    """Drop entries whose text looks like a transcription failure.

    Entries are shaped as `{"text": "...", ...}` — anything else about
    them is preserved. Kept as a list-in-list-out helper so callers
    (summarizer, patterns) have one obvious place to wire filtering in.
    """
    return [e for e in entries if not is_garbage_text(e.get("text") or "")]
