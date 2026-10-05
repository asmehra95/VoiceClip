"""AI correction/polish: context building, learned corrections, and the
guard rails that keep the LLM from changing what the user said."""

import time

import pytest

from voiceclip import config, polisher


@pytest.fixture
def fake_llm(monkeypatch):
    """Route the cloud provider to a scripted reply; capture the prompt."""
    calls = []
    state = {"reply": None, "delay": 0.0}

    def complete_cloud(**kw):
        calls.append(kw)
        if state["delay"]:
            time.sleep(state["delay"])
        r = state["reply"]
        return r(kw["user"]) if callable(r) else r

    monkeypatch.setattr(polisher.llm_provider, "complete_cloud", complete_cloud)
    monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "cloud")
    monkeypatch.setattr(config, "SUMMARIES_CLOUD_MODEL", "assistant")
    monkeypatch.setattr(config, "CUSTOM_VOCABULARY", ["Blue Harbor", "SLA", "self-billing"])
    monkeypatch.setattr(config, "DICTIONARY", {})
    monkeypatch.setattr(polisher, "_learned_corrections", lambda: {"Q free": "Q3"})
    return calls, state


def test_fixes_mishearing_and_sends_context(fake_llm):
    calls, state = fake_llm
    state["reply"] = "Blue Harbor plans to send five invoices for Q3."
    out = polisher.correct("Blue harbour plans to send five invoices for Q free.")
    assert out == "Blue Harbor plans to send five invoices for Q3."
    system, user = calls[0]["system"], calls[0]["user"]
    assert "Blue Harbor, SLA, self-billing" in system
    assert '"Q free" should be "Q3"' in system           # only learned pairs that occur
    assert user.startswith("<transcript>") and user.endswith("</transcript>")
    assert calls[0]["temperature"] == 0


def test_answering_instead_of_correcting_is_rejected(fake_llm):
    _, state = fake_llm
    state["reply"] = "Sure! Here is your text formatted into a single clean line for the file."
    assert polisher.correct("Can you format this?") == "Can you format this?"


def test_strips_tags_and_quotes(fake_llm):
    _, state = fake_llm
    state["reply"] = '<transcript>"Send the SLA today."</transcript>'
    assert polisher.correct("Send the S L A today.") == "Send the SLA today."


def test_timeout_and_errors_fall_back(fake_llm, monkeypatch):
    _, state = fake_llm
    monkeypatch.setattr(polisher, "_CORRECT_TIMEOUT", 0.1)
    state["reply"], state["delay"] = "Way too late.", 0.5
    assert polisher.correct("way too late") == "way too late"
    state["delay"] = 0

    def boom(_):
        raise RuntimeError("gateway down")
    state["reply"] = boom
    assert polisher.correct("still pasted") == "still pasted"


def test_no_model_means_no_call(fake_llm, monkeypatch):
    calls, _ = fake_llm
    monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "none")
    assert polisher.correct("hello there") == "hello there"
    assert polisher.polish("hello there") == "hello there"
    assert calls == []


def test_polish_allows_restructuring(fake_llm):
    _, state = fake_llm
    state["reply"] = "We need to send the SLA to Blue Harbor today."
    raw = "um so we need to like send the SLA to blue harbor today"
    assert polisher.polish(raw) == "We need to send the SLA to Blue Harbor today."


def test_mine_corrections_from_edits():
    pairs = [
        ("I have to look at that, I'll plan", "I have to look at Annual plan, I'll plan"),
        ("this new model, which is Cloud Opus, is generally good", "this new model, which is Claude Opus, is generally good"),
        ("totally different sentence here", "nothing in common whatsoever at all ok"),
    ]
    learned = polisher.mine_corrections(pairs)
    assert learned.get("that") == "Annual plan"
    assert learned.get("Cloud") == "Claude"
    assert all(len(k.split()) <= 3 for k in learned)


# ---------------------------------------------------------------------------
# constrain(): only trusted changes survive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("original,candidate,allowed,expected", [
    # rewording reverted, nothing else to fix
    ("We work with FinOps to do any carrier communication.",
     "We work with FinOps to handle any carrier communication.", [],
     "We work with FinOps to do any carrier communication."),
    # tense change and invented possessive reverted, sentence join kept
    ("We work with last May. Payments, Tech Team.",
     "We worked with Last May Payments' Tech Team.", [],
     "We work with Last May Payments Tech Team."),
    # vocabulary fix kept, dropped word ("the") restored, join kept
    ("They manage the relation. With the carriers if Phenopsis cannot.",
     "They manage the relationship with carriers if FinOps cannot.", ["FinOps"],
     "They manage the relation with the carriers if FinOps cannot."),
    # multi-word vocabulary replacement + fragment join
    ("Northwin is planning to send. Five invoices.",
     "Northwind is planning to send five invoices.", ["Northwind"],
     "Northwind is planning to send five invoices."),
    # hyphenation-only change is formatting
    ("We do e invoicing.", "We do e-invoicing.", [], "We do e-invoicing."),
    # added words are dropped unless they bring in a known term
    ("Send it today.", "Please send it today, thanks.", [], "Send it today."),
])
def test_constrain(original, candidate, allowed, expected):
    assert polisher.constrain(original, candidate, allowed) == expected


def test_constrain_uses_learned_pairs():
    out = polisher.constrain("this model, Cloud Opus, is good", "this model, Claude Opus, is good",
                             learned={"Cloud": "Claude"})
    assert out == "this model, Claude Opus, is good"


def test_correct_uses_recent_context(fake_llm):
    calls, state = fake_llm
    state["reply"] = "Escalate if FinOps cannot fix it."
    out = polisher.correct("Escalate if Phenopsis cannot fix it.",
                           recent=["We work with FinOps on carrier changes."])
    assert out == "Escalate if FinOps cannot fix it."      # allowed via recent context
    assert "We work with FinOps on carrier changes." in calls[0]["system"]


def test_correct_without_context_reverts_unknown_swap(fake_llm):
    _, state = fake_llm
    state["reply"] = "Escalate if FinOps cannot fix it."
    assert polisher.correct("Escalate if Phenopsis cannot fix it.", recent=[]) == \
        "Escalate if Phenopsis cannot fix it."


@pytest.mark.parametrize("original,candidate,allowed,expected", [
    # vocabulary term that doesn't sound like what was heard is refused
    ("We work with last May. Payments team.", "We work with Northwind payments team.", ["Northwind"],
     "We work with last May payments team."),
    ("Owned by self-invoicing.", "Owned by Annual plan.", ["Annual plan"], "Owned by self-invoicing."),
])
def test_constrain_requires_sound_alike(original, candidate, allowed, expected):
    assert polisher.constrain(original, candidate, allowed) == expected


def test_learned_pairs_only_apply_to_the_exact_phrase():
    # learned "that" -> "Annual plan" must not leak onto other words
    out = polisher.constrain("Owned by self-invoicing.", "Owned by Annual plan.",
                             allowed=[], learned={"that": "Annual plan"})
    assert out == "Owned by self-invoicing."


def test_sounds_alike():
    assert polisher.sounds_alike("Phenopsis", "FinOps") >= polisher._SOUND_MIN
    assert polisher.sounds_alike("Northwin", "Northwind") >= polisher._SOUND_MIN
    assert polisher.sounds_alike("changes", "updates") < polisher._SOUND_MIN


def test_last_status(fake_llm, monkeypatch):
    _, state = fake_llm
    state["reply"] = "hello there"
    polisher.correct("hello there", recent=[])
    assert polisher.last_status == "ok"
    monkeypatch.setattr(config, "SUMMARIES_PROVIDER", "none")
    polisher.correct("hello there", recent=[])
    assert polisher.last_status == "off"
