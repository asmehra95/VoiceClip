"""Post-processing formatter for Whisper output.

Applies regex-based cleanup and dictionary substitutions from config
to improve transcription quality with essentially zero latency cost.

Pipeline order:
1. Strip Whisper hallucinations (phantom "thank you" etc.)
2. Apply dictionary (global + persona word replacements)
3. Convert spoken punctuation to symbols
4. Convert spoken formatting (new line, bullet, etc.)
5. Fix capitalization
6. Clean up whitespace
"""

import logging
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dictionary patterns — built from config at startup
# ---------------------------------------------------------------------------

_dict_patterns: list[tuple[re.Pattern, str]] = []


def build_patterns():
    """Compile regex patterns from the config dictionary.

    Call this after config.load() so DICTIONARY is populated.
    """
    global _dict_patterns
    from voiceclip.config import DICTIONARY

    _dict_patterns = []
    for key, value in DICTIONARY.items():
        if not isinstance(value, str) or key.startswith("__"):
            continue
        try:
            pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
            _dict_patterns.append((pattern, value))
        except re.error:
            log.warning("Invalid dictionary key (skipped): %s", key)

    if _dict_patterns:
        log.info("Compiled %d dictionary patterns", len(_dict_patterns))


def _apply_dictionary(text: str) -> str:
    """Apply dictionary replacements."""
    for pattern, replacement in _dict_patterns:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Whisper hallucination cleanup
# ---------------------------------------------------------------------------

_HALLUCINATIONS = [
    r"^\s*thank you\.?\s*$",
    r"^\s*thanks for watching\.?\s*$",
    r"^\s*please subscribe\.?\s*$",
    r"^\s*bye\.?\s*$",
    r"^\s*you$",
    r"^\s*\.\s*$",
]
_HALLUCINATION_RE = re.compile(
    "|".join(_HALLUCINATIONS), re.IGNORECASE
)


def _strip_hallucinations(text: str) -> str | None:
    """Return None if the text is a known Whisper hallucination."""
    if _HALLUCINATION_RE.match(text.strip()):
        return None
    return text


# ---------------------------------------------------------------------------
# Spoken punctuation → symbols
# ---------------------------------------------------------------------------

_PUNCTUATION_MAP = [
    (r"\bquestion mark\b", "?"),
    (r"\bexclamation mark\b", "!"),
    (r"\bexclamation point\b", "!"),
    (r"\bopen paren\b", "("),
    (r"\bclose paren\b", ")"),
    (r"\bopen bracket\b", "["),
    (r"\bclose bracket\b", "]"),
    (r"\bopen quote\b", '"'),
    (r"\bclose quote\b", '"'),
    (r"\bellipsis\b", "..."),
    (r"\bperiod\b", "."),
    (r"\bcomma\b", ","),
    (r"\bcolon\b", ":"),
    (r"\bsemicolon\b", ";"),
    (r"\bdash\b", "—"),
    (r"\bhyphen\b", "-"),
]
_PUNCTUATION_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _PUNCTUATION_MAP
]


def _convert_spoken_punctuation(text: str) -> str:
    for pattern, symbol in _PUNCTUATION_PATTERNS:
        text = pattern.sub(symbol, text)
    return text


# ---------------------------------------------------------------------------
# Spoken formatting → actual formatting
# ---------------------------------------------------------------------------

_FORMATTING_MAP = [
    (r"\bnew paragraph\b", "\n\n"),
    (r"\bnew line\b", "\n"),
    (r"\bline break\b", "\n"),
    (r"(?:^|\n)\s*bullet point\b", "\n• "),
    (r"(?:^|\n)\s*bullet\b", "\n• "),
    (r"\btab\b", "\t"),
]
_FORMATTING_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _FORMATTING_MAP
]


def _convert_spoken_formatting(text: str) -> str:
    for pattern, repl in _FORMATTING_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Capitalization fixes
# ---------------------------------------------------------------------------

def _fix_capitalization(text: str) -> str:
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    text = re.sub(
        r'([.?!])\s+([a-z])',
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )
    text = re.sub(
        r'(\n\s*)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bi'", "I'", text)
    return text


# ---------------------------------------------------------------------------
# Whitespace cleanup
# ---------------------------------------------------------------------------

def _clean_whitespace(text: str) -> str:
    text = re.sub(r'\s+([.,;:?!])', r'\1', text)
    text = re.sub(r'([.,;:?!])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r' {2,}', ' ', text)
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_text(text: str) -> str | None:
    """Apply all formatting passes to transcribed text.

    Returns the formatted text, or None if the text was a hallucination.
    """
    if not text:
        return None

    text = _strip_hallucinations(text)
    if text is None:
        return None

    text = _apply_dictionary(text)
    text = _convert_spoken_punctuation(text)
    text = _convert_spoken_formatting(text)
    text = _fix_capitalization(text)
    text = _clean_whitespace(text)

    return text or None
