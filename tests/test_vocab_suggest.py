"""Custom-vocabulary suggestions from dictation history."""

import pytest

from voiceclip import vocab_suggest


@pytest.fixture(autouse=True)
def small_dictionary(monkeypatch):
    # Deterministic: don't depend on the host's /usr/share/dict/words.
    monkeypatch.setattr(vocab_suggest, "_words_cache", frozenset(
        ["the", "we", "send", "invoices", "to", "carrier", "carriers", "post", "today", "met", "with", "and", "about", "reply", "owe", "team", "review", "daily", "sync", "file"]))


def terms(out):
    return [s["term"] for s in out]


def test_acronyms_names_and_phrases():
    texts = [
        "We send the EDI file to Blue Harbor today.",
        "Met with Priya about the EDI file and Blue Harbor.",
        "I owe Priya a reply.",
    ]
    out = vocab_suggest.suggest(texts)
    t = terms(out)
    assert "EDI" in t
    assert "Priya" in t
    assert "Blue Harbor" in t
    assert "Harbor" not in t            # subsumed by the phrase
    edi = next(s for s in out if s["term"] == "EDI")
    assert edi["count"] == 2 and edi["kind"] == "acronym" and "EDI" in edi["example"]


def test_sentence_initial_capitals_are_not_names():
    out = vocab_suggest.suggest(["Carrier sync today.", "Carrier review today."])
    assert "Carrier" not in terms(out)


def test_existing_and_ignored_are_excluded_case_insensitively():
    texts = ["We met Priya and Mateo.", "Mateo and Priya again."]
    out = vocab_suggest.suggest(texts, existing=["priya"], ignored=["MATEO"])
    assert terms(out) == []


def test_min_count_and_common_words():
    out = vocab_suggest.suggest(["Thanks, we met Zorblax on Monday."])
    assert terms(out) == []            # single mention; Monday/Thanks are common


def test_inflections_count_as_dictionary_words():
    assert vocab_suggest._is_dictionary_word("Invoices", frozenset({"invoice"}))
    assert not vocab_suggest._is_dictionary_word("Oluwaseun", frozenset({"invoice"}))


def test_contractions_ignored_possessives_folded():
    out = vocab_suggest.suggest(["Well I'm sure I'll ask Priya's team.", "Then I'm asking Priya."])
    t = terms(out)
    assert "I'm" not in t and "I'll" not in t
    assert "Priya" in t


@pytest.mark.parametrize("term,kind", [
    ("EDI", "acronym"), ("INPUT", "acronym"), ("Priya", "name"),
    ("Blue Harbor", "phrase"), ("follow-up", "phrase"), ("reply", "word"),
    ("Grace", "name"), ("Hey", "word"),
])
def test_classify(term, kind):
    assert vocab_suggest.classify(term) == kind
