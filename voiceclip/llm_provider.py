"""Unified LLM provider abstraction.

Before this module existed, summarizer.py / researcher.py / patterns.py each
re-implemented the same local/openai/anthropic plumbing. Each feature now
declares a prompt + a small response parser, and delegates the transport to
the functions here.

Models are loaded lazily and cached in-process — a single `mlx_lm.load(...)`
call takes 5-15s, and we were paying it on every summary or patterns click.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-process model cache
# ---------------------------------------------------------------------------
# mlx-lm loads are expensive. Keep (model, tokenizer) pairs by model_id so
# subsequent calls in the same process reuse the loaded weights.
#
# Two concurrent requests with the same model_id will both miss the cache
# simultaneously without the lock and double-load ~4-8GB into memory.

_mlx_cache: dict[str, tuple[Any, Any]] = {}
_mlx_cache_lock = threading.Lock()


# Model types that go straight to mlx-vlm without trying mlx-lm first.
# These are full multimodal checkpoints (vision + audio towers alongside
# the language model) that mlx-lm genuinely can't load. Using the preflight
# saves ~5s of wasted load work per cold start for these.
#
# Note: do NOT list `gemma4_text` or `gemma3_text` here — those are
# mlx-lm model types. Some "text-only" checkpoints on HuggingFace are
# mis-packaged (include KV-shared weights they shouldn't); those fail
# in mlx-lm with a specific signature but aren't loadable by mlx-vlm
# either. The fallback path can't rescue them, so we surface a clear
# error with a pointer at the standard (non-OptiQ) variant.
_MLX_VLM_PREFERRED_TYPES: frozenset[str] = frozenset({
    "gemma4",
    "gemma3n",
})


def _preflight_model_type(model_id: str) -> str | None:
    """Peek at HuggingFace's config.json to learn the model_type field.

    Cached inside HF's own disk cache, so warm invocations are a single
    file read (~5ms). Cold invocations download a ~1KB file.

    Returns the model_type string (e.g. "gemma4_text", "qwen2_vl",
    "qwen2") or None on any failure. Failure is fine — the caller just
    skips the preflight shortcut and tries mlx-lm first.
    """
    try:
        from huggingface_hub import hf_hub_download
        import json as _json
        path = hf_hub_download(model_id, "config.json")
        with open(path, "r") as f:
            cfg = _json.loads(f.read())
        return cfg.get("model_type")
    except Exception as e:
        log.debug("config.json preflight failed for %s: %s", model_id, e)
        return None


# In-process cache of loaded models.
#
# Each entry is (backend, model, handle) where backend is one of:
#   "mlx_lm"  — handle is a tokenizer
#   "mlx_vlm" — handle is a processor (vlm-specific wrapper)
#
# Storing the backend tag lets complete_local() dispatch without re-probing.
# Two concurrent requests with the same model_id would double-load ~4-8GB
# into memory without the lock, so we serialize misses.

_mlx_cache: dict[str, tuple[str, Any, Any]] = {}
_mlx_cache_lock = threading.Lock()


def _mlx_load(model_id: str) -> tuple[str, Any, Any]:
    """Cached load of a local MLX model.

    Two-phase loader:

    1. Cheap preflight reads config.json and checks model_type. For known
       multimodal types (gemma4, gemma3n, *_vl) we go straight to mlx-vlm.
       This avoids wasting time on an mlx-lm load that would inevitably
       fail with "parameters not in model".

    2. Otherwise try mlx-lm first (the 95% case for text-only LLMs).
       On a ValueError with "parameters not in model", fall back to
       mlx-vlm — this catches multimodal checkpoints whose model_type
       we don't know about yet. Other ValueErrors get classified into
       readable hints.

    Returns (backend, model, handle) where handle is a tokenizer for
    mlx-lm or a processor for mlx-vlm. Callers dispatch on backend.
    """
    # Fast path: cache hit without holding the lock
    cached = _mlx_cache.get(model_id)
    if cached is not None:
        return cached

    with _mlx_cache_lock:
        # Re-check under the lock — another thread may have loaded it
        cached = _mlx_cache.get(model_id)
        if cached is not None:
            return cached

        log.info("Loading local model: %s (first use — may download)", model_id)

        # Phase 1: preflight — is this a known multimodal type?
        model_type = _preflight_model_type(model_id) or ""
        prefer_vlm = (
            model_type in _MLX_VLM_PREFERRED_TYPES
            or model_type.endswith("_vl")
            or "vision" in model_type
        )

        if prefer_vlm:
            log.info(
                "%s has model_type=%r; loading via mlx-vlm directly",
                model_id, model_type,
            )
            return _load_via_vlm(model_id)

        # Phase 2: try mlx-lm
        try:
            from mlx_lm import load as lm_load
        except ImportError:
            raise RuntimeError(
                "mlx-lm is not installed. Install it with:\n"
                "    ~/.voiceclip/.venv/bin/pip install mlx-lm"
            )

        try:
            model, tokenizer = lm_load(model_id)
            _mlx_cache[model_id] = ("mlx_lm", model, tokenizer)
            return _mlx_cache[model_id]
        except ValueError as e:
            msg = str(e)
            low = msg.lower()
            # Two very different "parameters not in model" failures:
            #
            # (a) Full multimodal checkpoint — extras are `language_model.*`,
            #     `vision_tower.*`, `audio_tower.*` prefixes. mlx-vlm can
            #     load this.
            #
            # (b) Mis-packaged text-only checkpoint (e.g. the OptiQ Gemma 4
            #     variants on mlx-community at time of writing). Extras are
            #     plain `model.layers.N.self_attn.k_proj` etc. in the
            #     KV-shared layer range. mlx-vlm can't load these either
            #     (it has no `gemma4_text` loader). Surface a specific
            #     error pointing at the standard variant.
            if "parameters not in model" in low:
                if (
                    "language_model." in msg
                    or "vision_tower" in low
                    or "audio_tower" in low
                ):
                    log.info(
                        "%s is a multimodal checkpoint; falling back to mlx-vlm",
                        model_id,
                    )
                    return _load_via_vlm(model_id)
                # Mis-packaged text-only checkpoint — mlx-lm sees extra KV
                # weights that shouldn't exist in a KV-shared layer.
                if "k_proj" in low or "k_norm" in low:
                    raise RuntimeError(
                        f"'{model_id}' appears to be a mis-packaged text-only "
                        "checkpoint — it carries KV-shared-layer weights "
                        "that shouldn't be in the file. mlx-lm and mlx-vlm "
                        "both reject it. Try the standard (non-OptiQ) "
                        "variant, e.g. mlx-community/gemma-4-e4b-it-4bit "
                        "or mlx-community/gemma-4-e2b-it-4bit."
                    ) from e
                # Generic "extras" — try vlm as a last resort
                log.info(
                    "%s has extra weights mlx-lm doesn't recognize; "
                    "falling back to mlx-vlm", model_id,
                )
                return _load_via_vlm(model_id)
            raise _classify_mlx_load_error(model_id, e) from e
        except Exception as e:
            raise RuntimeError(
                f"Could not load local model '{model_id}': "
                f"{type(e).__name__}: {str(e)[:300]}"
            ) from e


def _load_via_vlm(model_id: str) -> tuple[str, Any, Any]:
    """Load a model through mlx-vlm. Must only be called while holding
    _mlx_cache_lock. Installs the cache entry before returning."""
    try:
        from mlx_vlm import load as vlm_load
    except ImportError:
        raise RuntimeError(
            f"'{model_id}' is a multimodal model and needs mlx-vlm. "
            "Install it with:\n"
            "    ~/.voiceclip/.venv/bin/pip install mlx-vlm\n"
            "Then restart voiceclip view."
        )
    try:
        model, processor = vlm_load(model_id)
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{model_id}' with mlx-vlm: "
            f"{type(e).__name__}: {str(e)[:300]}"
        ) from e
    _mlx_cache[model_id] = ("mlx_vlm", model, processor)
    return _mlx_cache[model_id]


def _classify_mlx_load_error(model_id: str, err: Exception) -> RuntimeError:
    """Map mlx-lm weight-loading errors to readable hints.

    Called only on ValueErrors that weren't the multimodal signature
    (which has its own fallback path).
    """
    msg = str(err)
    low = msg.lower()
    if "missing parameters" in low or ("not found" in low and "weight" in low):
        return RuntimeError(
            f"'{model_id}' appears to be missing weight files. The download "
            "may have been interrupted. Try clearing it from the "
            "\"Downloaded models\" panel in Settings and loading again."
        )
    return RuntimeError(
        f"Could not load local model '{model_id}': {msg[:400]}"
    )


def reset_mlx_cache():
    """Test hook — drop any cached local models."""
    with _mlx_cache_lock:
        _mlx_cache.clear()


# ---------------------------------------------------------------------------
# Local (mlx-lm)
# ---------------------------------------------------------------------------

def complete_local(
    *,
    system: str,
    user: str,
    model_id: str,
    max_tokens: int = 400,
) -> str:
    """Run a single system+user completion via mlx-lm. Returns text only.

    Reasoning-tuned models (Qwen3, DeepSeek-R1, QwQ, R1 distills) emit a
    scratchpad before the real answer. The industry-standard convention
    (established by DeepSeek-R1 and adopted by vLLM, SGLang, and the major
    inference providers) is that the model wraps its chain-of-thought in
    <think>...</think> tags, and everything after the closing tag is the
    final answer — two distinct "channels" in one response.

    We rely on that convention and reinforce it via the system prompt:
    callers should append REASONING_DIRECTIVE so the model knows exactly
    where to put reasoning vs answer. Reasoning models comply trivially
    (it matches their training). Non-reasoning models just skip the tags
    and produce the answer directly — harmless.

    We deliberately do NOT send `/no_think` or `enable_thinking=False`.
    The whole point of a reasoning model is that it thinks — we just
    route the scratchpad into a channel we can strip.

    Supports both mlx-lm (text-only LLMs) and mlx-vlm (multimodal
    models used text-only) via the _mlx_load backend tag. The same
    REASONING_DIRECTIVE contract applies to both.
    """
    backend, model, handle = _mlx_load(model_id)
    if backend == "mlx_vlm":
        return _complete_local_vlm(model, handle, system, user, max_tokens)
    return _complete_local_lm(model, handle, system, user, max_tokens)


def _complete_local_lm(model, tokenizer, system: str, user: str, max_tokens: int) -> str:
    """mlx-lm generate path. Tokenizer produces the chat template;
    generate() returns a plain string."""
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"{system}\n\n{user}\n\n"
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    if out.startswith(prompt):
        out = out[len(prompt):]
    return _extract_answer(out).strip()


def _complete_local_vlm(model, processor, system: str, user: str, max_tokens: int) -> str:
    """mlx-vlm generate path, used for multimodal models in text-only
    mode. Processor handles the chat template via apply_chat_template;
    generate() returns a GenerationResult object we unwrap via .text.

    We pass num_images=0 and no image list — mlx-vlm then only runs the
    language tower, skipping the vision/audio encoders entirely. The
    vision weights sit in RAM unused but don't affect inference speed.
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    # mlx-vlm's apply_chat_template bakes the system prompt into the
    # user turn when the processor doesn't have a system slot. Preserve
    # the same structure we use for mlx-lm by concatenating — keeps
    # REASONING_DIRECTIVE effective and the output shape identical.
    combined = f"{system}\n\n{user}" if system else user
    try:
        prompt = apply_chat_template(
            processor, model.config, combined, num_images=0,
        )
    except Exception:
        prompt = combined

    result = generate(
        model, processor, prompt,
        max_tokens=max_tokens, verbose=False,
    )
    # mlx-vlm returns a GenerationResult dataclass with .text; older
    # versions returned a plain string. Handle both so we're resilient
    # to library bumps.
    text = getattr(result, "text", result) if result else ""
    if isinstance(text, str) and text.startswith(prompt):
        text = text[len(prompt):]
    return _extract_answer(text).strip()


# Shared directive appended to every local-model system prompt so reasoning
# models route their scratchpad into the <think> channel we can strip.
# Structural directives ("put X inside Y tags") are followed more reliably
# by instruction-tuned models than prose asks ("don't show reasoning").
# Non-reasoning models will simply skip the tags and produce the answer
# directly — harmless.
REASONING_DIRECTIVE = (
    "If you need to reason through this, put your reasoning inside "
    "<think>...</think> tags first. Your final answer must come after "
    "the closing </think> tag."
)


# Matches a closed <think>...</think> block (also covers <thinking> and
# <reasoning> variants some fine-tunes use). Non-greedy; DOTALL so it
# spans newlines; case-insensitive so <THINK> works.
_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Opener-only pattern (unclosed scratchpad — model hit max_tokens before
# finishing its reasoning).
_REASONING_OPEN_RE = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>",
    re.IGNORECASE,
)


def _extract_answer(text: str) -> str:
    """Pull the answer channel out of a reasoning-model completion.

    Contract (DeepSeek-R1 / industry standard):

        <think>...reasoning...</think> The real answer.

    Cases handled:

    1. Text contains one or more closed <think>...</think> blocks:
       strip them and return the remainder (the answer channel).
    2. Text contains an unclosed <think> with no closer (model ran out
       of max_tokens mid-reasoning): return whatever preceded the opener
       and log a warning. There's no reliable answer after an unclosed
       opener.
    3. No reasoning markers at all: pass through unchanged.

    This is deliberately NOT robust against prose-format reasoning like
    "Thinking Process: 1. Analyze...". A model that emits prose reasoning
    has ignored both its own fine-tune convention and the explicit
    REASONING_DIRECTIVE in the system prompt — surfacing that failure
    visibly is the right behaviour. Hiding it with heuristics would shift
    the problem from "this model doesn't work for me" to "sometimes the
    summary is mysteriously truncated", which is worse.
    """
    if not text:
        return text

    # Case 1: closed blocks — strip every one, keep the remainder
    cleaned, n_subs = _REASONING_BLOCK_RE.subn("", text)
    if n_subs > 0:
        return cleaned

    # Case 2: unclosed opener — keep only the prefix before it
    m = _REASONING_OPEN_RE.search(text)
    if m:
        log.warning(
            "Local model emitted an unclosed reasoning opener — ran out of "
            "max_tokens mid-thought. Answer channel is empty for this call."
        )
        return text[:m.start()]

    # Case 3: no markers — pass through unchanged
    return text


# Backward-compat alias so existing callers and tests don't break.
# _extract_answer is the preferred name going forward.
_strip_reasoning = _extract_answer


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell and restart voiceclip view."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "The 'openai' package is not installed. Install it with:\n"
            "    ~/.voiceclip/.venv/bin/pip install openai"
        )
    # 60s transport timeout so a stalled network can't wedge the viewer
    # handler thread for the SDK default (10 min). Readable TimeoutError
    # bubbles up to our error-humanizing path.
    return OpenAI(api_key=api_key, timeout=60.0)


def complete_openai(
    *,
    system: str,
    user: str,
    model_id: str,
    json_mode: bool = False,
) -> str:
    """Plain chat completion via OpenAI. Returns text only.

    Set json_mode=True to force JSON-structured output (patterns feature uses
    this; summarizer does not).

    We don't send `max_tokens` or `temperature` — newer reasoning models
    (GPT-5+, o-series) reject those parameters, and server defaults are
    fine for our short completions.
    """
    client = _openai_client()
    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def complete_openai_with_web_search(
    *,
    system: str,
    user: str,
    model_id: str,
) -> tuple[str, list[dict], bool]:
    """Use OpenAI's Responses API with the built-in web_search tool.

    Returns (text, sources, used_web_search). Falls back to a plain call if the
    model doesn't support the tool.
    """
    client = _openai_client()
    try:
        resp = client.responses.create(
            model=model_id,
            tools=[{"type": "web_search"}],
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        msg = str(e).lower()
        if "web_search" in msg or "tool" in msg or "not supported" in msg:
            log.info("web_search not supported by %s; falling back to plain call", model_id)
            resp = client.responses.create(
                model=model_id,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        else:
            raise

    text = getattr(resp, "output_text", None) or ""
    used_web = False
    sources: list[dict] = []
    output = getattr(resp, "output", None) or []
    for item in output:
        itype = getattr(item, "type", None)
        if itype and "web_search" in itype:
            used_web = True
        content = getattr(item, "content", None) or []
        for c in content:
            annotations = getattr(c, "annotations", None) or []
            for ann in annotations:
                atype = getattr(ann, "type", "")
                if "url_citation" in atype or "citation" in atype:
                    url = getattr(ann, "url", None)
                    title = getattr(ann, "title", None) or url
                    if url:
                        sources.append({"title": title or url, "url": url})
    return text.strip(), _dedupe_sources(sources), used_web


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell and restart voiceclip view."
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Install it with:\n"
            "    ~/.voiceclip/.venv/bin/pip install anthropic"
        )
    # 60s transport timeout — same rationale as _openai_client.
    return anthropic.Anthropic(api_key=api_key, timeout=60.0)


def complete_anthropic(
    *,
    system: str,
    user: str,
    model_id: str,
    max_tokens: int = 400,
) -> str:
    """Plain message completion via Anthropic. Returns concatenated text blocks."""
    client = _anthropic_client()
    resp = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in resp.content:
        t = getattr(block, "text", None)
        if t:
            parts.append(t)
    return "".join(parts).strip()


def complete_anthropic_with_web_search(
    *,
    system: str,
    user: str,
    model_id: str,
    max_tokens: int = 800,
) -> tuple[str, list[dict], bool]:
    """Use Anthropic's server-side web_search tool. Returns (text, sources, used_web)."""
    client = _anthropic_client()
    try:
        resp = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )
    except Exception as e:
        msg = str(e).lower()
        if "web_search" in msg or "tool" in msg or "unsupported" in msg:
            log.info("web_search not supported by %s; falling back to plain call", model_id)
            resp = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        else:
            raise

    text_parts: list[str] = []
    sources: list[dict] = []
    used_web = False
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            t = getattr(block, "text", "") or ""
            if t:
                text_parts.append(t)
            citations = getattr(block, "citations", None) or []
            for cit in citations:
                url = getattr(cit, "url", None)
                title = getattr(cit, "title", None) or url
                if url:
                    sources.append({"title": title or url, "url": url})
        elif btype == "server_tool_use" or "web_search" in (btype or ""):
            used_web = True
    return "".join(text_parts).strip(), _dedupe_sources(sources), used_web


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for s in sources:
        url = s.get("url")
        if url and url not in seen:
            seen.add(url)
            out.append(s)
    return out
