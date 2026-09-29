"""Vocabulary tab: current custom vocabulary + suggestions mined from the
journal.

GET  /api/vocab         → {terms, suggestions: [{term,count,kind,example}], ignored}
POST /api/vocab/ignore  → {"term": "..."}  (never suggest it again)
POST /api/vocab/unignore→ {"term": "..."}

Adding/removing terms goes through the regular settings endpoint
(custom_vocabulary), so validation and persistence stay in one place.
"""

from voiceclip import config, history, vocab_suggest
from voiceclip.viewer.routes import register_get, register_post

_SCAN_LIMIT = 5000
_MAX_IGNORED = 500


def _ignored() -> list[str]:
    raw = (getattr(config, "_raw", {}) or {}).get("vocab_ignored", [])
    return [t for t in raw if isinstance(t, str)] if isinstance(raw, list) else []


def _get_vocab(req, query):
    texts = []
    if history._conn is not None:
        rows = history._conn.execute(
            "SELECT COALESCE(formatted_text, raw_text) FROM transcriptions "
            "WHERE archived_at IS NULL AND kind IN ('transcription', 'reflection') "
            "ORDER BY id DESC LIMIT ?", (_SCAN_LIMIT,)).fetchall()
        texts = [r[0] for r in rows]
    terms = list(config.CUSTOM_VOCABULARY)
    ignored = _ignored()
    req._json({
        "terms": terms,
        "kinds": {t: vocab_suggest.classify(t) for t in terms},
        "ignored": ignored,
        "scanned": len(texts),
        "suggestions": vocab_suggest.suggest(texts, existing=terms, ignored=ignored),
    })


def _set_ignored(terms: list[str]):
    from voiceclip.config_io import write_config_patch
    write_config_patch({"vocab_ignored": terms[-_MAX_IGNORED:]})
    config.load()


def _clean(payload) -> str | None:
    term = str((payload or {}).get("term") or "").strip()
    return term if 0 < len(term) <= 80 else None


def _post_ignore(req, payload):
    term = _clean(payload)
    if term is None:
        req._json({"error": "term must be 1-80 characters"}, status=400)
        return
    ignored = _ignored()
    if term.lower() not in {t.lower() for t in ignored}:
        ignored.append(term)
        _set_ignored(ignored)
    req._json({"ok": True})


def _post_unignore(req, payload):
    term = _clean(payload)
    if term is None:
        req._json({"error": "term must be 1-80 characters"}, status=400)
        return
    _set_ignored([t for t in _ignored() if t.lower() != term.lower()])
    req._json({"ok": True})


register_get("/api/vocab", _get_vocab)
register_post("/api/vocab/ignore", _post_ignore)
register_post("/api/vocab/unignore", _post_unignore)
