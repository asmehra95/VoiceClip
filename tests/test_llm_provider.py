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

    def setup_method(self):
        # The MLX worker thread imports mlx_lm / mlx_vlm lazily. Tests
        # inject stubs via monkeypatch.setitem(sys.modules, ...) — those
        # stubs work on any thread, but a stale worker from a previous
        # test might have state we don't want. Start every test with a
        # fresh worker and empty model cache.
        llm_provider._reset_worker()
        llm_provider.reset_mlx_cache()

    def _stub_mlx_lm(self, monkeypatch, behavior):
        """Install a fake mlx_lm with a load() function that follows `behavior`.

        behavior("model_id") should either return (model, tokenizer) or raise.
        """
        import sys
        import types
        fake = types.ModuleType("mlx_lm")
        fake.load = behavior
        monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    def _stub_mlx_vlm(self, monkeypatch, behavior):
        """Install a fake mlx_vlm with a load() function."""
        import sys
        import types
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
        # The full multimodal Gemma 4 has model_type="gemma4" (without
        # _text). gemma4_text is a different, problematic case — see
        # test_optiq_style_broken_checkpoint_gets_readable_error.
        self._stub_preflight(monkeypatch, "gemma4")

        backend, model, handle = llm_provider._mlx_load(
            "mlx-community/gemma-4-e4b-it-4bit")
        assert backend == "mlx_vlm"
        assert lm_called["n"] == 0
        assert vlm_called["n"] == 1

    def test_optiq_style_broken_checkpoint_gets_readable_error(self, monkeypatch):
        """Mis-packaged text-only checkpoints (carry k_proj/k_norm in
        layers that should be KV-shared) can't load in either library.
        User should see a clear pointer to the standard variant rather
        than a 700-line traceback."""
        def lm_load(_):
            # Mimic the real mlx-lm error signature for OptiQ Gemma 4:
            # plain `model.*` prefix (no `language_model.`), includes
            # k_proj/k_norm tokens.
            raise ValueError(
                "Received 140 parameters not in model: \n"
                "model.layers.15.self_attn.k_norm.weight,\n"
                "model.layers.15.self_attn.k_proj.biases,\n"
                "model.layers.15.self_attn.k_proj.scales,\n..."
            )
        self._stub_mlx_lm(monkeypatch, lm_load)
        self._stub_mlx_vlm(monkeypatch, lambda _: ("vlm-model", "proc"))
        self._stub_preflight(monkeypatch, "gemma4_text")

        with pytest.raises(RuntimeError, match="mis-packaged"):
            llm_provider._mlx_load("mlx-community/gemma-4-e4b-it-OptiQ-4bit")

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
        """When mlx-lm raises the multimodal ValueError with language_model.*
        prefixes, we fall back to mlx-vlm — this is the full-Gemma-4 /
        Qwen-VL case."""
        lm_called = {"n": 0}

        def lm_load(_):
            lm_called["n"] += 1
            raise ValueError(
                "Received 42 parameters not in model:\n"
                "language_model.model.layers.0.mlp.weight,\n..."
            )

        def vlm_load(_):
            return ("vlm-model", "processor")

        self._stub_mlx_lm(monkeypatch, lm_load)
        self._stub_mlx_vlm(monkeypatch, vlm_load)
        self._stub_preflight(monkeypatch, "unknown_type")

        backend, model, handle = llm_provider._mlx_load("some/mystery-vlm")
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
                              ValueError(
                                  "Received 10 parameters not in model:\n"
                                  "language_model.foo")))
        self._stub_preflight(monkeypatch, "unknown_vlm_type")
        # Ensure mlx_vlm import fails
        monkeypatch.setitem(sys.modules, "mlx_vlm", None)

        with pytest.raises(RuntimeError, match="mlx-vlm"):
            llm_provider._mlx_load("gemma-4-test")


class TestRecommendedModels:
    """The curated model list in voiceclip/models.json drives the
    Settings picker dropdown. Tests cover:
      - parsing real models.json shipped with the package
      - per-feature filtering (good_for tag)
      - graceful degradation when the file is missing or malformed
    """

    def setup_method(self):
        # Each test starts with a fresh cache
        llm_provider._reset_recommended_models()

    def test_returns_list_with_expected_fields(self):
        models = llm_provider.list_recommended_models()
        assert len(models) > 0, "models.json ships empty — that's a bug"
        for m in models:
            assert "id" in m
            assert "label" in m
            assert "size_gb" in m
            assert "backend" in m
            assert "good_for" in m
            # At least one recommended model must actually be a real
            # HuggingFace repo id — mlx-community prefix is the canonical
            # one for MLX ports.
        ids = [m["id"] for m in models]
        assert any(i.startswith("mlx-community/") for i in ids)

    def test_filter_by_feature(self):
        research = llm_provider.list_recommended_models(feature="research")
        summaries = llm_provider.list_recommended_models(feature="summaries")
        assert len(research) > 0 and len(summaries) > 0
        # Everything returned for a feature must have that tag
        assert all("research" in m["good_for"] for m in research)
        assert all("summaries" in m["good_for"] for m in summaries)

    def test_unknown_feature_filters_to_empty(self):
        # Features that don't appear in any model's good_for tag
        assert llm_provider.list_recommended_models(feature="bogus") == []

    def test_cache_hits_on_second_call(self, monkeypatch):
        """Second call should not re-open the file — proves the cache works."""
        import builtins
        real_open = builtins.open
        open_calls = {"n": 0}

        def counting_open(*args, **kwargs):
            open_calls["n"] += 1
            return real_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting_open)
        llm_provider.list_recommended_models()
        llm_provider.list_recommended_models()
        llm_provider.list_recommended_models(feature="summaries")
        # Only the first call reads the file
        assert open_calls["n"] == 1

    def test_missing_file_returns_empty_not_crash(self, monkeypatch):
        """If models.json disappears, the Settings UI's dropdown just
        hides rather than breaking the whole tab."""
        monkeypatch.setattr(
            llm_provider, "_RECOMMENDED_MODELS_PATH", "/nonexistent/models.json",
        )
        assert llm_provider.list_recommended_models() == []

    def test_malformed_json_returns_empty(self, tmp_path, monkeypatch):
        bad = tmp_path / "broken.json"
        bad.write_text("{not valid json")
        monkeypatch.setattr(llm_provider, "_RECOMMENDED_MODELS_PATH", str(bad))
        assert llm_provider.list_recommended_models() == []

    def test_entries_without_id_are_dropped(self, tmp_path, monkeypatch):
        """Defensive against partially-edited models.json — skip invalid
        entries, don't fail the whole load."""
        import json as _json
        path = tmp_path / "mixed.json"
        path.write_text(_json.dumps({
            "models": [
                {"id": "good/model", "label": "OK", "size_gb": 1.0,
                 "backend": "mlx-lm", "good_for": ["summaries"]},
                {"label": "Missing id"},  # bad: no id
                "not a dict",              # bad: wrong type
            ],
        }))
        monkeypatch.setattr(llm_provider, "_RECOMMENDED_MODELS_PATH", str(path))
        models = llm_provider.list_recommended_models()
        assert len(models) == 1
        assert models[0]["id"] == "good/model"


class TestWorkerThread:
    """Regression guards for the MLX worker thread.

    The worker exists because mlx-vlm creates a non-thread-local
    `generation_stream` at import time, which then fails from any other
    thread. Running every MLX touch through a single dedicated thread
    bypasses the issue — these tests ensure that contract holds.
    """

    def setup_method(self):
        llm_provider._reset_worker()
        llm_provider.reset_mlx_cache()

    def test_submitted_work_runs_on_worker_not_caller(self):
        """Every call to _run_on_worker should execute on the dedicated
        worker thread, never on the calling thread."""
        import threading as _t
        caller_thread = _t.current_thread().ident
        seen = {"worker": None}

        def capture():
            seen["worker"] = _t.current_thread().ident
            return "ok"

        result = llm_provider._run_on_worker(capture)
        assert result == "ok"
        assert seen["worker"] is not None
        assert seen["worker"] != caller_thread

    def test_concurrent_submissions_are_serialized(self):
        """Two concurrent submissions must both succeed — the worker
        queue serializes them. Without serialization mlx-vlm's stream
        assumptions would break if two threads entered generate() at once."""
        import threading as _t
        import time

        running = {"count": 0, "max": 0}
        lock = _t.Lock()

        def stepwise_task():
            with lock:
                running["count"] += 1
                running["max"] = max(running["max"], running["count"])
            time.sleep(0.05)  # hold the worker briefly
            with lock:
                running["count"] -= 1
            return "done"

        threads = [
            _t.Thread(
                target=lambda: llm_provider._run_on_worker(stepwise_task),
            )
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()
        # At no point did more than one task run concurrently
        assert running["max"] == 1

    def test_exception_propagates_to_caller(self):
        """If the worker raises, the caller sees that exception, not a
        queue error or None."""
        def boom():
            raise ValueError("intentional test failure")

        with pytest.raises(ValueError, match="intentional test failure"):
            llm_provider._run_on_worker(boom)

    def test_worker_recovers_from_exception(self):
        """A bad submission shouldn't kill the worker — subsequent calls
        must still work."""
        def boom():
            raise RuntimeError("oops")

        try:
            llm_provider._run_on_worker(boom)
        except RuntimeError:
            pass

        # Worker is still alive and responsive
        assert llm_provider._run_on_worker(lambda: "ok") == "ok"

    def test_complete_local_dispatches_through_worker(self, monkeypatch):
        """End-to-end: complete_local submits work to the worker rather
        than calling MLX from the caller's thread."""
        import threading as _t
        caller_thread = _t.current_thread().ident
        seen = {"load_thread": None, "generate_thread": None}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "PROMPT"

        def fake_lm_load(model_id):
            seen["load_thread"] = _t.current_thread().ident
            return "model-obj", FakeTokenizer()

        def fake_generate(model, tokenizer, **kwargs):
            seen["generate_thread"] = _t.current_thread().ident
            return kwargs.get("prompt", "") + "the answer"

        import sys
        import types
        fake_lm = types.ModuleType("mlx_lm")
        fake_lm.load = fake_lm_load
        fake_lm.generate = fake_generate
        monkeypatch.setitem(sys.modules, "mlx_lm", fake_lm)
        # Stub mlx.core so the `with mx.stream(mx.gpu):` wrappers pass
        fake_mx_core = types.ModuleType("mlx.core")
        import contextlib
        fake_mx_core.stream = lambda _: contextlib.nullcontext()
        fake_mx_core.gpu = object()
        fake_mx = types.ModuleType("mlx")
        fake_mx.core = fake_mx_core
        monkeypatch.setitem(sys.modules, "mlx", fake_mx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)
        # Preflight should fall through so mlx-lm is attempted
        monkeypatch.setattr(llm_provider, "_preflight_model_type",
                            lambda _: "qwen2")

        result = llm_provider.complete_local(
            system="sys", user="u", model_id="mlx/fake", max_tokens=100,
        )
        assert result == "the answer"
        # Both load and generate ran on the worker — NOT on the caller thread
        assert seen["load_thread"] is not None
        assert seen["generate_thread"] is not None
        assert seen["load_thread"] != caller_thread
        assert seen["generate_thread"] != caller_thread
        # And they ran on the SAME thread — that's the whole point
        assert seen["load_thread"] == seen["generate_thread"]


class TestGgufDetection:
    """GGUF-format repos (llama.cpp native) can't load in MLX. Users
    routinely confuse them with MLX-compatible repos because both live
    on HuggingFace. We want a readable error, not a raw FileNotFoundError."""

    def test_gguf_error_names_the_format(self):
        err = FileNotFoundError(
            "No safetensors found in /Users/v/.cache/huggingface/hub/"
            "models--Jackrong--Qwen3.5-9B-GGUF/snapshots/abc"
        )
        msg = str(llm_provider._classify_mlx_load_error(
            "Jackrong/Qwen3.5-9B-GGUF", err,
        ))
        assert "GGUF" in msg
        assert "llama.cpp" in msg.lower()

    def test_gguf_error_points_at_mlx_community(self):
        """The recovery hint should actually help — point at the picker
        or mlx-community namespace, not just 'try a different model'."""
        err = FileNotFoundError("No safetensors found anywhere")
        msg = str(llm_provider._classify_mlx_load_error(
            "Some/Model-GGUF", err,
        ))
        assert "mlx-community" in msg.lower() or "quick pick" in msg.lower()

    def test_lowercase_gguf_suffix_also_detected(self):
        """Repo names use -GGUF or .gguf in practice; detection is
        case-insensitive."""
        err = FileNotFoundError("no safetensors")
        msg = str(llm_provider._classify_mlx_load_error(
            "author/some-model.gguf", err,
        ))
        assert "GGUF" in msg

    def test_non_gguf_safetensors_missing_gives_different_hint(self):
        """A safetensors-missing error on a non-GGUF repo is a different
        problem (interrupted download, typically). Don't misdiagnose."""
        err = FileNotFoundError(
            "No safetensors found in /some/mlx-community/Qwen-4bit"
        )
        msg = str(llm_provider._classify_mlx_load_error(
            "mlx-community/Qwen2.5-7B-Instruct-4bit", err,
        ))
        # Not the GGUF hint — falls through to the generic truncated-
        # error branch because the repo id doesn't signal GGUF.
        assert "GGUF" not in msg
        assert "Could not load local model" in msg

    def test_is_gguf_model_requires_both_signals(self):
        """Both the error AND the repo name have to signal GGUF — either
        alone is too weak to classify."""
        # Repo says GGUF, error doesn't mention safetensors — ambiguous
        assert not llm_provider._is_gguf_model(
            "author/model-GGUF", "some unrelated error",
        )
        # Error says safetensors, repo doesn't say GGUF — could be a
        # legit MLX-repo download interruption
        assert not llm_provider._is_gguf_model(
            "mlx-community/Qwen-4bit", "No safetensors found",
        )
        # Both signals present — confident GGUF
        assert llm_provider._is_gguf_model(
            "author/model-GGUF", "No safetensors found",
        )
