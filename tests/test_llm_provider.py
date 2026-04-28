"""Focused tests for voiceclip.llm_provider.

Scope intentionally narrow — we're not mocking the full SDK surface. Just
checking the thin wrappers: cache semantics, and that the SDK clients get a
timeout so a stalled cloud call can't wedge the viewer handler thread.
"""

import pytest

from voiceclip import llm_provider


@pytest.fixture(autouse=True)
def _reset_cache():
    """Drop any loaded local models between tests."""
    llm_provider.reset_mlx_cache()
    yield
    llm_provider.reset_mlx_cache()


class TestClientTimeouts:
    """The SDK clients must be constructed with a finite timeout.

    Without this, a stalled network request pins the handler thread for the
    SDK default (10 minutes on OpenAI, effectively unbounded on some
    Anthropic versions). 60 seconds is plenty for our short completions.

    We stub the SDK constructors so these tests run even when the real
    openai / anthropic packages aren't in the venv — the contract we care
    about is "timeout kwarg is passed", not "SDK behaves correctly".
    """

    def test_openai_client_passes_timeout(self, monkeypatch):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        # Inject a fake openai module so the lazy import inside
        # _openai_client picks it up instead of failing.
        import sys
        import types
        fake_mod = types.ModuleType("openai")
        fake_mod.OpenAI = FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_mod)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        llm_provider._openai_client()
        assert "timeout" in captured, "OpenAI client built without a timeout"
        assert 0 < float(captured["timeout"]) <= 120.0, \
            f"OpenAI timeout out of range: {captured['timeout']}"

    def test_openai_client_missing_key_raises_readable_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm_provider._openai_client()

    def test_anthropic_client_passes_timeout(self, monkeypatch):
        captured = {}

        class FakeAnthropicClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import sys
        import types
        fake_mod = types.ModuleType("anthropic")
        fake_mod.Anthropic = FakeAnthropicClient
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        llm_provider._anthropic_client()
        assert "timeout" in captured, "Anthropic client built without a timeout"
        assert 0 < float(captured["timeout"]) <= 120.0, \
            f"Anthropic timeout out of range: {captured['timeout']}"

    def test_anthropic_client_missing_key_raises_readable_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            llm_provider._anthropic_client()


class TestMlxCache:
    """The mlx-lm cache is the only reason local summaries don't re-pay the
    5-15s load on every click. These tests verify the cache works without
    requiring mlx-lm to actually be installed."""

    def test_reset_clears_cache(self):
        # Prime the cache with a fake entry; reset should wipe it.
        llm_provider._mlx_cache["fake-model"] = ("mlx_lm", "model-obj", "tok-obj")
        llm_provider.reset_mlx_cache()
        assert llm_provider._mlx_cache == {}

    def test_load_hits_cache_without_mlx_installed(self):
        """If the cache already has an entry, _mlx_load must return it
        without touching the mlx_lm import — which may not even be
        installed."""
        llm_provider._mlx_cache["cached-model"] = ("mlx_lm", "weights", "tokenizer")
        # Should return the cached tuple verbatim, no import path taken
        assert llm_provider._mlx_load("cached-model") == ("mlx_lm", "weights", "tokenizer")


class TestStripReasoning:
    """Reasoning-model models (Qwen3, DeepSeek-R1) emit a scratchpad before
    the real answer. The viewer should never show that to the user."""

    def test_passthrough_when_no_reasoning(self):
        assert llm_provider._strip_reasoning("Plain answer.") == "Plain answer."

    def test_empty_string_is_safe(self):
        assert llm_provider._strip_reasoning("") == ""
        assert llm_provider._strip_reasoning(None) is None

    def test_strips_think_block(self):
        raw = "<think>Let me analyze this step by step.</think>\n\nThe answer is 42."
        out = llm_provider._strip_reasoning(raw)
        assert "analyze" not in out.lower()
        assert "42" in out

    def test_strips_thinking_variant(self):
        raw = "<thinking>Reasoning here.</thinking>Final: done."
        out = llm_provider._strip_reasoning(raw)
        assert "Reasoning" not in out
        assert "Final: done." in out

    def test_strips_reasoning_variant(self):
        raw = "<reasoning>Step 1...</reasoning>Real output."
        out = llm_provider._strip_reasoning(raw)
        assert "Step 1" not in out
        assert "Real output." in out

    def test_strips_multiline_reasoning(self):
        raw = (
            "<think>\n"
            "1. Analyze the data\n"
            "2. Form a conclusion\n"
            "</think>\n"
            "Short summary for the user."
        )
        out = llm_provider._strip_reasoning(raw)
        assert "Analyze" not in out
        assert "Short summary for the user." in out.strip()

    def test_strips_multiple_blocks(self):
        raw = "<think>first</think>middle<think>second</think>end"
        out = llm_provider._strip_reasoning(raw)
        assert "first" not in out
        assert "second" not in out
        assert "middle" in out
        assert "end" in out

    def test_case_insensitive_tag(self):
        raw = "<THINK>thoughts</THINK>answer"
        out = llm_provider._strip_reasoning(raw)
        assert "thoughts" not in out
        assert "answer" in out

    def test_unclosed_think_at_start_is_dropped(self):
        """When a model hits max_tokens mid-reasoning, there's no usable
        answer — better to return empty than ship the scratchpad."""
        raw = "<think>Let me think about this for a really long time..."
        out = llm_provider._strip_reasoning(raw).strip()
        assert out == ""

    def test_preserves_content_before_unclosed_opener(self):
        """If prose precedes an unclosed <think>, the prose stays."""
        raw = "Quick answer.\n\n<think>and then I started second-guessing..."
        out = llm_provider._strip_reasoning(raw)
        assert "Quick answer." in out
        assert "second-guessing" not in out


class TestReasoningDirective:
    """The REASONING_DIRECTIVE is the contract we tell every local-model
    caller to append to their system prompt. It must mention <think> tags
    explicitly and instruct the model to put the answer after the closing
    tag — otherwise the extraction layer can't do its job."""

    def test_directive_mentions_think_tag(self):
        d = llm_provider.REASONING_DIRECTIVE
        assert "<think>" in d
        assert "</think>" in d

    def test_directive_instructs_answer_after_closer(self):
        d = llm_provider.REASONING_DIRECTIVE.lower()
        # The directive must make it clear the final answer comes AFTER
        # the closing tag — otherwise the model may put the answer inside.
        assert "after" in d
        assert "final answer" in d


class TestExtractAnswer:
    """Covers the DeepSeek-R1 channel-separator contract more directly
    than the legacy tests above. _extract_answer is the preferred name;
    _strip_reasoning is kept as an alias."""

    def test_canonical_channel_split(self):
        raw = "<think>I should summarize what happened.</think>\n\nYou spent the morning on research."
        out = llm_provider._extract_answer(raw).strip()
        # Reasoning is gone, answer remains verbatim
        assert out == "You spent the morning on research."

    def test_answer_only_passes_through(self):
        """Non-reasoning models produce plain answers. Must not touch them."""
        raw = "Plain summary with no tags."
        assert llm_provider._extract_answer(raw) == raw

    def test_alias_matches_extract_answer(self):
        """_strip_reasoning is a backward-compat alias and must behave
        identically to _extract_answer."""
        for sample in [
            "plain",
            "<think>x</think>answer",
            "<think>never closed",
            "",
        ]:
            assert (
                llm_provider._strip_reasoning(sample)
                == llm_provider._extract_answer(sample)
            )

    def test_unclosed_think_returns_prefix_only(self):
        """Model ran out of max_tokens mid-thought — no clean answer exists."""
        raw = "<think>Let me think about this for a very long..."
        assert llm_provider._extract_answer(raw).strip() == ""

    def test_prefix_before_unclosed_is_kept(self):
        """If the model wrote something useful before opening <think> and
        never closed it, keep what came before."""
        raw = "Quick note.\n\n<think>and now I keep reasoning..."
        out = llm_provider._extract_answer(raw)
        assert "Quick note." in out
        assert "reasoning" not in out

    def test_prose_reasoning_passes_through_unmodified(self):
        """Models that ignore the directive and emit prose reasoning
        ("Thinking Process: 1. ...") must surface visibly — don't silently
        chop them."""
        raw = (
            "Thinking Process:\n"
            "1. Analyze the request\n"
            "2. Draft the summary\n\n"
            "You spent the day coding."
        )
        out = llm_provider._extract_answer(raw)
        # The whole thing comes through — including the reasoning — so the
        # user sees that their model choice is emitting unstructured output
        # and can switch.
        assert out == raw


class TestCompleteLocalNoNoThink:
    """Regression: we used to append /no_think to the user content. That
    was unreliable (Qwen3-only) and pointless now that we use <think>-tag
    extraction. Verify it's gone."""

    def test_user_prompt_is_not_suffixed(self, monkeypatch):
        """complete_local must pass the user content through verbatim."""
        captured = {}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                return "PROMPT"

        def fake_generate(model, tokenizer, **kwargs):
            return kwargs.get("prompt", "PROMPT") + "some answer"

        llm_provider._mlx_cache["test-model"] = ("mlx_lm", "model-obj", FakeTokenizer())
        # Stub mlx_lm.generate so the test runs without mlx-lm installed.
        import sys
        import types
        fake_mlx = types.ModuleType("mlx_lm")
        fake_mlx.generate = fake_generate
        monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)

        llm_provider.complete_local(
            system="sys",
            user="the user content",
            model_id="test-model",
        )
        # The user message is exactly the user content — no /no_think suffix
        user_msg = next(m for m in captured["messages"] if m["role"] == "user")
        assert user_msg["content"] == "the user content"


class TestMlxVlmBackend:
    """The _mlx_load path must prefer mlx-vlm for known multimodal models
    (Gemma 4, Gemma 3n, *_vl). Everything else tries mlx-lm first and only
    falls back to mlx-vlm on the 'parameters not in model' signature."""

    def _stub_mlx_lm(self, monkeypatch, behavior):
        """Install a fake mlx_lm with a load() function that follows `behavior`.

        behavior("model_id") should either return (model, tokenizer) or raise.
        """
        import sys, types
        fake = types.ModuleType("mlx_lm")
        fake.load = behavior
        monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    def _stub_mlx_vlm(self, monkeypatch, behavior):
        """Install a fake mlx_vlm with a load() function."""
        import sys, types
        fake = types.ModuleType("mlx_vlm")
        fake.load = behavior
        monkeypatch.setitem(sys.modules, "mlx_vlm", fake)

    def _stub_preflight(self, monkeypatch, model_type: str | None):
        """Force the config.json preflight to return a specific model_type
        (or None to simulate preflight failure)."""
        monkeypatch.setattr(llm_provider, "_preflight_model_type",
                            lambda mid: model_type)

    def test_gemma4_skips_mlx_lm_and_uses_vlm(self, monkeypatch):
        """Known multimodal model type must go straight to mlx-vlm —
        no wasted mlx-lm attempt, no download churn."""
        lm_called = {"n": 0}
        vlm_called = {"n": 0}

        def lm_load(_):
            lm_called["n"] += 1
            return ("m", "t")

        def vlm_load(_):
            vlm_called["n"] += 1
            return ("vlm-model", "processor")

        self._stub_mlx_lm(monkeypatch, lm_load)
        self._stub_mlx_vlm(monkeypatch, vlm_load)
        self._stub_preflight(monkeypatch, "gemma4_text")

        backend, model, handle = llm_provider._mlx_load(
            "mlx-community/gemma-4-e4b-it-4bit")
        assert backend == "mlx_vlm"
        assert lm_called["n"] == 0
        assert vlm_called["n"] == 1

    def test_text_only_model_uses_mlx_lm(self, monkeypatch):
        """The common case — Qwen2.5, Llama, etc. — must use mlx-lm
        directly with no mlx-vlm involvement."""
        vlm_called = {"n": 0}

        def lm_load(_):
            return ("lm-model", "tokenizer")

        def vlm_load(_):
            vlm_called["n"] += 1
            return ("vlm-model", "processor")

        self._stub_mlx_lm(monkeypatch, lm_load)
        self._stub_mlx_vlm(monkeypatch, vlm_load)
        self._stub_preflight(monkeypatch, "qwen2")

        backend, model, handle = llm_provider._mlx_load(
            "mlx-community/Qwen2.5-7B-Instruct-4bit")
        assert backend == "mlx_lm"
        assert vlm_called["n"] == 0

    def test_fallback_on_parameters_not_in_model(self, monkeypatch):
        """When mlx-lm raises the multimodal ValueError for a model the
        preflight didn't flag, we still fall back to mlx-vlm."""
        lm_called = {"n": 0}

        def lm_load(_):
            lm_called["n"] += 1
            raise ValueError("Received 42 parameters not in model:\nweights...")

        def vlm_load(_):
            return ("vlm-model", "processor")

        self._stub_mlx_lm(monkeypatch, lm_load)
        self._stub_mlx_vlm(monkeypatch, vlm_load)
        self._stub_preflight(monkeypatch, "unknown_type")

        backend, model, handle = llm_provider._mlx_load("some/mystery-model")
        assert backend == "mlx_vlm"
        assert lm_called["n"] == 1

    def test_non_multimodal_value_error_is_classified(self, monkeypatch):
        """A ValueError that ISN'T the multimodal signature should NOT
        trigger mlx-vlm fallback — it's a genuine load failure that
        deserves a classified error message."""
        vlm_called = {"n": 0}

        def lm_load(_):
            raise ValueError("missing parameters from weights file")

        def vlm_load(_):
            vlm_called["n"] += 1
            return ("vlm-model", "processor")

        self._stub_mlx_lm(monkeypatch, lm_load)
        self._stub_mlx_vlm(monkeypatch, vlm_load)
        self._stub_preflight(monkeypatch, "qwen2")

        with pytest.raises(RuntimeError, match="missing weight files"):
            llm_provider._mlx_load("broken/download")
        assert vlm_called["n"] == 0

    def test_cached_backend_is_reused(self, monkeypatch):
        """Second call for the same model_id hits the cache without
        touching either loader."""
        lm_count = {"n": 0}

        def lm_load(_):
            lm_count["n"] += 1
            return ("m", "t")

        self._stub_mlx_lm(monkeypatch, lm_load)
        self._stub_preflight(monkeypatch, "qwen2")

        llm_provider._mlx_load("mlx/foo")
        llm_provider._mlx_load("mlx/foo")
        assert lm_count["n"] == 1

    def test_vlm_missing_gives_actionable_error(self, monkeypatch):
        """If a multimodal model is configured but mlx-vlm isn't installed,
        the error should tell the user exactly how to install it."""
        import sys
        self._stub_mlx_lm(monkeypatch,
                          lambda _: (_ for _ in ()).throw(
                              ValueError("Received 10 parameters not in model")))
        self._stub_preflight(monkeypatch, "qwen2")
        # Ensure mlx_vlm import fails
        monkeypatch.setitem(sys.modules, "mlx_vlm", None)

        with pytest.raises(RuntimeError, match="mlx-vlm"):
            llm_provider._mlx_load("gemma-4-test")
