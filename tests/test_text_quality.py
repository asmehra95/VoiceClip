"""Unit tests for voiceclip.text_quality.

The filter is intentionally conservative — false negatives (garbage
slipping through) are fine; false positives (real content classified
as garbage) are bad. These tests lock that asymmetry in.
"""

from voiceclip.text_quality import filter_entries, is_garbage_text


class TestPassThroughShortText:
    """Anything under 40 chars is never flagged. Short dictations are
    often meaningful ('yes', 'okay done', 'note to self')."""

    def test_empty_is_not_garbage(self):
        assert not is_garbage_text("")

    def test_none_is_not_garbage(self):
        # Defensive: callers may pass None via .get() fallback
        assert not is_garbage_text(None)

    def test_short_repetition_passes(self):
        # "yes yes yes" is 3 tokens of 100% dominance but intentionally
        # allowed — user may have meant to emphasize.
        assert not is_garbage_text("yes yes yes")

    def test_short_phrase_passes(self):
        assert not is_garbage_text("okay thanks for the update")


class TestPassThroughRealContent:
    """Legitimate dictation of any length must not be flagged."""

    def test_long_real_sentence(self):
        text = (
            "We should probably set up a call with the compliance team "
            "next week to walk through the Belgium and Portugal e-invoicing "
            "rules before we commit to a delivery date."
        )
        assert not is_garbage_text(text)

    def test_technical_content_with_repeated_terminology(self):
        """Real tech discussion can repeat a term many times without
        being degenerate. 'DynamoDB' mentioned 6 times out of 50 words
        is well under the 60% threshold."""
        text = (
            "The DynamoDB partition key design matters here. If we hot-spot "
            "on a single DynamoDB partition, read capacity will drop. "
            "DynamoDB auto-scaling helps but adds latency, so the DynamoDB "
            "write path needs a better distribution. DynamoDB isn't the "
            "bottleneck — the DynamoDB request shape is."
        )
        assert not is_garbage_text(text)

    def test_reflection_with_thoughtful_repetition(self):
        text = (
            "I keep thinking about the annual plan. The annual plan is "
            "where I want to spend time this week. Without the annual "
            "plan being clear, everything else stays blurry."
        )
        assert not is_garbage_text(text)


class TestStuckHotkeyLoops:
    """The actual symptom from production: stuck toggle-mode hotkey
    transcribes long silence as one word repeated."""

    def test_machine_loop(self):
        """The real observed sample — 'machine and machine and machine...'"""
        text = " ".join(["and", "machine"] * 80)
        assert is_garbage_text(text)

    def test_the_the_the_loop(self):
        text = "the " * 50
        assert is_garbage_text(text)

    def test_dominant_token_just_under_threshold_passes(self):
        """50% dominance is legitimate (e.g. emphasis). Threshold is 60%."""
        # 15 tokens, 7 'foo' + 8 other = 46% — must pass
        tokens = ["foo"] * 7 + ["one", "two", "three", "four",
                                "five", "six", "seven", "eight"]
        text = " ".join(tokens)
        assert not is_garbage_text(text)

    def test_dominant_token_above_threshold_flagged(self):
        # 20 tokens, 14 'foo' + 6 other = 70% — flagged
        tokens = ["foo"] * 14 + ["one", "two", "three", "four", "five", "six"]
        text = " ".join(tokens)
        assert is_garbage_text(text)


class TestCharacterLevelNoise:
    """Bluetooth or driver glitches can produce single-char runs or
    short-phrase loops not caught by the token-level check."""

    def test_single_char_run(self):
        assert is_garbage_text("a" * 40)

    def test_punctuation_run(self):
        # Needs to exceed the 40-char minimum before any check fires.
        assert is_garbage_text("!" * 45)

    def test_short_phrase_repeating_loop(self):
        # "ok " (3 chars including space) repeated 15 times
        assert is_garbage_text("ok " * 15)

    def test_moderate_repetition_passes(self):
        # 5 'ok ' — under the 10x threshold
        assert not is_garbage_text("ok ok ok ok ok and a real thought here")


class TestFilterEntries:
    """filter_entries is a thin list-in-list-out helper — lock its
    contract so summarizer/patterns can rely on it."""

    def test_preserves_entry_shape(self):
        entries = [
            {"id": 1, "text": "Normal thought.", "kind": "reflection",
             "app_name": "Terminal"},
        ]
        out = filter_entries(entries)
        assert len(out) == 1
        assert out[0] == entries[0]

    def test_drops_only_the_garbage_entries(self):
        entries = [
            {"id": 1, "text": "A real sentence about the meeting."},
            {"id": 2, "text": "and " * 40},  # garbage
            {"id": 3, "text": "Another real entry."},
            {"id": 4, "text": "the the the the the the the the the the the the the the"},  # garbage
        ]
        out = filter_entries(entries)
        ids = [e["id"] for e in out]
        assert ids == [1, 3]

    def test_empty_list_is_empty(self):
        assert filter_entries([]) == []

    def test_missing_text_field_safe(self):
        """Defensive: entries without a text key are passed through
        (is_garbage_text handles None/empty)."""
        out = filter_entries([{"id": 1}])
        assert out == [{"id": 1}]
