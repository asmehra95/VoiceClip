"""Post-processing formatter for Whisper output.

Applies regex-based cleanup and user dictionary substitutions to
improve transcription quality with essentially zero latency cost.

Pipeline order:
1. Strip Whisper hallucinations (phantom "thank you" etc.)
2. Apply user dictionary (custom word replacements)
3. Convert spoken punctuation to symbols
4. Convert spoken formatting (new line, bullet, etc.)
5. Fix capitalization
6. Clean up whitespace
"""

import json
import logging
import os
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User dictionary
# ---------------------------------------------------------------------------

_DICT_DIR = os.path.expanduser("~/.voiceclip")
_DICT_PATH = os.path.join(_DICT_DIR, "dictionary.json")

_EXAMPLE_DICT = {
    "__comment": "Add your custom word replacements below. Keys are case-insensitive.",
    "voiceclip": "VoiceClip",
    "macos": "macOS",
    "iphone": "iPhone",
}

_user_dict: dict[str, str] = {}
_user_dict_patterns: list[tuple[re.Pattern, str]] = []


def _ensure_dict_file():
    """Create the dictionary file with examples if it doesn't exist."""
    if os.path.exists(_DICT_PATH):
        return
    try:
        os.makedirs(_DICT_DIR, exist_ok=True)
        with open(_DICT_PATH, "w") as f:
            json.dump(_EXAMPLE_DICT, f, indent=2)
        log.info("Created dictionary at %s", _DICT_PATH)
    except OSError as e:
        log.warning("Could not create dictionary file: %s", e)


def load_dictionary():
    """Load the user dictionary from disk and compile regex patterns.

    Call this at startup. The dictionary is a simple JSON object where
    keys are words/phrases to find (case-insensitive) and values are
    their replacements.

    Also sets the Whisper initial_prompt from dictionary values so the
    model is biased toward producing correct spellings.
    """
    global _user_dict, _user_dict_patterns

    _ensure_dict_file()

    if not os.path.exists(_DICT_PATH):
        return

    try:
        with open(_DICT_PATH) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not load dictionary (%s): %s", _DICT_PATH, e)
        return

    if not isinstance(raw, dict):
        log.warning("Dictionary must be a JSON object, got %s", type(raw).__name__)
        return

    # Filter out comments and build patterns
    _user_dict = {}
    _user_dict_patterns = []
    prompt_words = []

    for key, value in raw.items():
        if key.startswith("__"):
            continue
        if not isinstance(value, str):
            continue
        _user_dict[key.lower()] = value
        prompt_words.append(value)
        # Word-boundary match, case-insensitive
        try:
            pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
            _user_dict_patterns.append((pattern, value))
        except re.error:
            log.warning("Invalid dictionary key (skipped): %s", key)

    count = len(_user_dict_patterns)
    if count:
        log.info("Loaded %d dictionary entries from %s", count, _DICT_PATH)

    # Feed dictionary words to Whisper as initial_prompt
    from voiceclip.transcriber import set_initial_prompt
    set_initial_prompt(prompt_words)


def _apply_dictionary(text: str) -> str:
    """Apply user dictionary replacements."""
    for pattern, replacement in _user_dict_patterns:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Whisper hallucination cleanup
# ---------------------------------------------------------------------------

# Common Whisper hallucinations when given silence or very short audio
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

# Order matters: longer phrases first to avoid partial matches
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
    """Replace spoken punctuation words with their symbols."""
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
    # "bullet" or "bullet point" at the start of a segment
    (r"(?:^|\n)\s*bullet point\b", "\n• "),
    (r"(?:^|\n)\s*bullet\b", "\n• "),
    (r"\btab\b", "\t"),
]
_FORMATTING_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _FORMATTING_MAP
]


def _convert_spoken_formatting(text: str) -> str:
    """Replace spoken formatting commands with actual formatting."""
    for pattern, repl in _FORMATTING_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Capitalization fixes
# ---------------------------------------------------------------------------

def _fix_capitalization(text: str) -> str:
    """Capitalize after sentence-ending punctuation and fix standalone 'i'."""
    # Capitalize first character
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Capitalize after . ? !
    text = re.sub(
        r'([.?!])\s+([a-z])',
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )

    # Capitalize after newlines
    text = re.sub(
        r'(\n\s*)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )

    # Fix standalone "i" → "I"
    text = re.sub(r"\bi\b", "I", text)

    # Fix "i'm" "i'll" "i've" "i'd"
    text = re.sub(r"\bi'", "I'", text)

    return text


# ---------------------------------------------------------------------------
# Whitespace cleanup
# ---------------------------------------------------------------------------

def _clean_whitespace(text: str) -> str:
    """Normalize whitespace: collapse runs, fix spacing around punctuation."""
    # Remove space before punctuation
    text = re.sub(r'\s+([.,;:?!])', r'\1', text)

    # Ensure space after punctuation (but not after newlines or before newlines)
    text = re.sub(r'([.,;:?!])([A-Za-z])', r'\1 \2', text)

    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    # Trim leading/trailing whitespace per line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Collapse 3+ newlines into 2
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

    # 1. Check for hallucinations
    text = _strip_hallucinations(text)
    if text is None:
        return None

    # 2. User dictionary
    text = _apply_dictionary(text)

    # 3. Spoken punctuation → symbols
    text = _convert_spoken_punctuation(text)

    # 4. Spoken formatting → actual formatting
    text = _convert_spoken_formatting(text)

    # 5. Capitalization
    text = _fix_capitalization(text)

    # 6. Whitespace cleanup
    text = _clean_whitespace(text)

    return text or None
