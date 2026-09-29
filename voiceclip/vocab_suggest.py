"""Suggest custom-vocabulary terms from what the user actually dictates.

Whisper spells common English fine; what it gets wrong are names, brands,
acronyms and team jargon. Those show up in the journal as:
  - acronyms              SLA, KPI, VAT
  - capitalised words mid-sentence that aren't dictionary words
                          Priya, Mateo, Oluwaseun
  - repeated capitalised phrases
                          Blue Harbor, Acme Cloud
Ranked by frequency. Pure functions over a list of texts, so the logic is
testable without a database.
"""

import re
from collections import Counter

_WORDLIST_PATHS = ("/usr/share/dict/words", "/usr/dict/words")
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9&]{1,6}$")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&'’-]*")
_COMMON_ACRONYMS = frozenset(["I", "OK", "AM", "PM", "US", "UK", "TV", "ID", "IDS", "OR", "SO", "NO", "IT", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "ME", "MY", "OF", "ON", "TO", "UP", "WE", "AI", "PC", "ETA", "FAQ", "CEO", "CTO", "CFO"])
_COMMON_CAPS = frozenset(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "English", "OK", "Okay", "Thanks", "Hello", "Hi", "Hey", "Yes", "No", "Also", "And", "But", "So", "Then", "Let", "Can", "Could", "Would", "Should", "Please", "Thank", "Mr", "Mrs", "Ms", "Dr"])

_words_cache: frozenset | None = None


def _dictionary() -> frozenset:
    global _words_cache
    if _words_cache is None:
        words = set()
        for path in _WORDLIST_PATHS:
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    words.update(w.strip().lower() for w in f if w.strip())
                break
            except OSError:
                continue
        _words_cache = frozenset(words)
    return _words_cache


def _is_dictionary_word(word: str, dictionary: frozenset) -> bool:
    w = word.lower().strip("'’-")
    if not dictionary:
        return False
    if w in dictionary:
        return True
    # The system wordlist has few inflections: try common suffix strips.
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"),
                         ("ing", ""), ("ing", "e"), ("ly", ""), ("er", ""), ("'s", "")):
        if (w.endswith(suffix) and len(w) - len(suffix) >= 3
                and w[: len(w) - len(suffix)] + repl in dictionary):
            return True
    return False


def _tokens_with_position(text: str):
    """Yield (token, is_sentence_start) for each word token."""
    prev_end = 0
    sentence_start = True
    for m in _TOKEN_RE.finditer(text):
        gap = text[prev_end:m.start()]
        if prev_end and re.search(r"[.!?…]", gap):
            sentence_start = True
        yield m.group(0).rstrip("'’-"), sentence_start
        sentence_start = False
        prev_end = m.end()


def suggest(texts, existing=(), ignored=(), limit: int = 40, min_count: int = 2):
    """Return [{"term", "count", "kind", "example"}] ranked by frequency."""
    dictionary = _dictionary()
    taken = {t.lower() for t in existing} | {t.lower() for t in ignored}
    counts: Counter = Counter()
    kinds: dict[str, str] = {}
    examples: dict[str, str] = {}

    def note(term, kind, text):
        if term.lower() in taken:
            return
        counts[term] += 1
        kinds.setdefault(term, kind)
        if term not in examples:
            i = text.find(term)
            lo, hi = max(0, i - 30), min(len(text), i + len(term) + 30)
            snippet = text[lo:hi].strip()
            examples[term] = ("…" if lo else "") + snippet + ("…" if hi < len(text) else "")

    for text in texts:
        if not text:
            continue
        toks = _tokens_with_position(text)
        run: list[str] = []
        for tok, at_start in toks:
            if "'" in tok or "’" in tok:        # contractions / possessives: I'm, Priya's
                base = tok.split("'")[0].split("’")[0]
                if not base or len(base) < 3 or not base[:1].isupper():
                    run = []
                    continue
                tok = base
            if _ACRONYM_RE.match(tok) and tok not in _COMMON_ACRONYMS:
                note(tok, "acronym", text)
            elif (tok[:1].isupper() and not at_start and len(tok) >= 3
                  and tok not in _COMMON_CAPS and not _is_dictionary_word(tok, dictionary)):
                note(tok, "name", text)
            # Capitalised phrases (2-3 words), first word not sentence-initial
            if tok[:1].isupper() and tok not in _COMMON_CAPS and (run or not at_start):
                run.append(tok)
            else:
                if 2 <= len(run) <= 3:
                    note(" ".join(run), "phrase", text)
                run = []
        if 2 <= len(run) <= 3:
            note(" ".join(run), "phrase", text)

    ranked = [t for t, c in counts.most_common() if c >= min_count]
    # A phrase subsumes its single words when they only appear inside it.
    phrases = [t for t in ranked if " " in t]
    out = []
    for t in ranked:
        if " " not in t and any(t in p.split() and counts[p] >= counts[t] for p in phrases):
            continue
        out.append({"term": t, "count": counts[t], "kind": kinds[t], "example": examples[t]})
        if len(out) >= limit:
            break
    return out


def classify(term: str) -> str:
    """Bucket an existing vocabulary term for display:
    'acronym' (SLA, KPI) · 'name' (Priya, Mateo) · 'phrase' (Blue Harbor,
    follow-up) · 'word' (ordinary words like 'side')."""
    t = (term or "").strip()
    if not t:
        return "word"
    if " " in t or "-" in t:
        return "phrase"
    if _ACRONYM_RE.match(t) or (t.isupper() and len(t) <= 8):
        return "acronym"
    # The user typed it capitalised on purpose, so it's a name even when the
    # system wordlist happens to contain it (Grace, Hope, Mark…).
    if t[:1].isupper() and t not in _COMMON_CAPS:
        return "name"
    return "word"
