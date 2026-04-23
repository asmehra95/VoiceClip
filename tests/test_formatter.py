"""Tests for voiceclip.formatter — regex post-processing pipeline."""

import pytest
from voiceclip.formatter import (
    format_text,
    _strip_hallucinations,
    _convert_spoken_punctuation,
    _convert_spoken_formatting,
    _fix_capitalization,
    _clean_whitespace,
    _apply_dictionary,
    build_patterns,
)


# ---------------------------------------------------------------------------
# Hallucination filtering
# ---------------------------------------------------------------------------

class TestHallucinations:
    @pytest.mark.parametrize("text", [
        "Thank you",
        "thank you.",
        "  Thank you  ",
        "Thanks for watching",
        "thanks for watching.",
        "Please subscribe",
        "Bye",
        "bye.",
        "You",
        ".",
        "  .  ",
    ])
    def test_known_hallucinations_return_none(self, text):
        assert _strip_hallucinations(text) is None

    @pytest.mark.parametrize("text", [
        "Thank you for the update",
        "I said bye to them",
        "You should try this",
        "Hello world",
        "The period was long",
    ])
    def test_real_text_passes_through(self, text):
        assert _strip_hallucinations(text) == text


# ---------------------------------------------------------------------------
# Spoken punctuation
# ---------------------------------------------------------------------------

class TestSpokenPunctuation:
    def test_period(self):
        assert _convert_spoken_punctuation("hello period") == "hello ."

    def test_comma(self):
        assert _convert_spoken_punctuation("hey comma how are you") == "hey , how are you"

    def test_question_mark(self):
        assert _convert_spoken_punctuation("how are you question mark") == "how are you ?"

    def test_exclamation(self):
        assert _convert_spoken_punctuation("wow exclamation mark") == "wow !"

    def test_exclamation_point(self):
        assert _convert_spoken_punctuation("wow exclamation point") == "wow !"

    def test_colon(self):
        assert _convert_spoken_punctuation("note colon") == "note :"

    def test_semicolon(self):
        assert _convert_spoken_punctuation("first semicolon second") == "first ; second"

    def test_ellipsis(self):
        assert _convert_spoken_punctuation("well ellipsis") == "well ..."

    def test_dash(self):
        assert _convert_spoken_punctuation("one dash two") == "one \u2014 two"

    def test_hyphen(self):
        assert _convert_spoken_punctuation("well hyphen known") == "well - known"

    def test_parens(self):
        result = _convert_spoken_punctuation("open paren note close paren")
        assert result == "( note )"

    def test_brackets(self):
        result = _convert_spoken_punctuation("open bracket 1 close bracket")
        assert result == "[ 1 ]"

    def test_quotes(self):
        result = _convert_spoken_punctuation("open quote hello close quote")
        assert result == '" hello "'

    def test_case_insensitive(self):
        assert _convert_spoken_punctuation("Hello PERIOD") == "Hello ."

    def test_multiple_punctuation(self):
        result = _convert_spoken_punctuation("hey comma how are you question mark")
        assert result == "hey , how are you ?"


# ---------------------------------------------------------------------------
# Spoken formatting
# ---------------------------------------------------------------------------

class TestSpokenFormatting:
    def test_new_line(self):
        assert _convert_spoken_formatting("first new line second") == "first \n second"

    def test_new_paragraph(self):
        assert _convert_spoken_formatting("first new paragraph second") == "first \n\n second"

    def test_line_break(self):
        assert _convert_spoken_formatting("first line break second") == "first \n second"

    def test_bullet(self):
        result = _convert_spoken_formatting("bullet buy milk")
        assert "•" in result
        assert "buy milk" in result

    def test_bullet_point(self):
        result = _convert_spoken_formatting("bullet point buy milk")
        assert "•" in result

    def test_tab(self):
        assert _convert_spoken_formatting("hello tab world") == "hello \t world"


# ---------------------------------------------------------------------------
# Capitalization
# ---------------------------------------------------------------------------

class TestCapitalization:
    def test_capitalize_first_char(self):
        assert _fix_capitalization("hello world") == "Hello world"

    def test_capitalize_after_period(self):
        assert _fix_capitalization("hello. world") == "Hello. World"

    def test_capitalize_after_question(self):
        assert _fix_capitalization("how? good") == "How? Good"

    def test_capitalize_after_exclamation(self):
        assert _fix_capitalization("wow! great") == "Wow! Great"

    def test_fix_standalone_i(self):
        assert _fix_capitalization("i think i will go") == "I think I will go"

    def test_fix_i_contractions(self):
        result = _fix_capitalization("i'm going and i'll be back")
        assert "I'm" in result
        assert "I'll" in result

    def test_capitalize_after_newline(self):
        assert _fix_capitalization("hello\nworld") == "Hello\nWorld"

    def test_already_capitalized(self):
        assert _fix_capitalization("Hello World") == "Hello World"


# ---------------------------------------------------------------------------
# Whitespace cleanup
# ---------------------------------------------------------------------------

class TestWhitespace:
    def test_collapse_spaces(self):
        assert _clean_whitespace("hello   world") == "hello world"

    def test_remove_space_before_punctuation(self):
        assert _clean_whitespace("hello .") == "hello."

    def test_add_space_after_punctuation(self):
        assert _clean_whitespace("hello.world") == "hello. world"

    def test_trim_lines(self):
        assert _clean_whitespace("  hello  \n  world  ") == "hello\nworld"

    def test_collapse_newlines(self):
        assert _clean_whitespace("hello\n\n\n\nworld") == "hello\n\nworld"

    def test_strip_outer(self):
        assert _clean_whitespace("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# Full pipeline (format_text)
# ---------------------------------------------------------------------------

class TestFormatText:
    def test_none_input(self):
        assert format_text(None) is None

    def test_empty_input(self):
        assert format_text("") is None

    def test_hallucination_returns_none(self):
        assert format_text("thank you") is None

    def test_basic_formatting(self):
        result = format_text("hello world")
        assert result == "Hello world"

    def test_spoken_punctuation_pipeline(self):
        result = format_text("hey period how are you question mark")
        assert result == "Hey. How are you?"

    def test_i_fix(self):
        result = format_text("i think i will go")
        assert result == "I think I will go"

    def test_comma_pipeline(self):
        result = format_text("hello comma world")
        assert result == "Hello, world"

    def test_new_line_pipeline(self):
        result = format_text("first new line second")
        assert "First" in result
        assert "\n" in result
        assert "Second" in result

    def test_idempotent(self):
        """Formatting the same text twice should give the same result."""
        text = "hello period how are you question mark"
        first = format_text(text)
        second = format_text(first)
        assert first == second

    def test_real_dictation(self):
        """Simulate a real dictation scenario."""
        result = format_text(
            "hey comma running 10 minutes late to the standup period "
            "i will join from my phone period"
        )
        assert "Hey," in result
        assert "standup." in result
        assert "I will" in result
        assert "phone." in result


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

class TestDictionary:
    def test_apply_dictionary_empty(self):
        """With no patterns loaded, text passes through."""
        assert _apply_dictionary("hello world") == "hello world"

    def test_build_and_apply(self):
        """Test dictionary loading and application via config."""
        from voiceclip import config
        config.DICTIONARY = {"voiceclip": "VoiceClip", "macos": "macOS"}
        build_patterns()

        result = _apply_dictionary("i love voiceclip on macos")
        assert "VoiceClip" in result
        assert "macOS" in result
