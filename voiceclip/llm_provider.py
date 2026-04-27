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


def _mlx_load(model_id: str):
    """Cached mlx-lm load. Returns (model, tokenizer)."""
    # Fast path: cache hit without holding the lock
    cached = _mlx_cache.get(model_id)
    if cached is not None:
        return cached
    try:
        from mlx_lm import load
    except ImportError:
        raise RuntimeError(
            "mlx-lm is not installed. Install it with:\n"
            "    ~/.voiceclip/.venv/bin/pip install mlx-lm"
        )
    with _mlx_cache_lock:
        # Re-check under the lock — another thread may have loaded it
        cached = _mlx_cache.get(model_id)
        if cached is not None:
            return cached
        log.info("Loading local model: %s (first use — may download)", model_id)
        _mlx_cache[model_id] = load(model_id)
        return _mlx_cache[model_id]


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

    Reasoning models (Qwen3, DeepSeek-R1, etc.) emit a scratchpad before
    the real answer. We handle this two ways:

    - Append `/no_think` to the user prompt. Qwen3 recognizes this and
      suppresses the reasoning block entirely. Models that don't recognize
      it just see extra characters and ignore them. No harm.
    - After generation, strip any `<think>...</think>` blocks that slipped
      through anyway — DeepSeek-R1 and some Qwen variants emit these even
      with the hint.

    The goal is a plain summary/brief/etc. with no chain-of-thought
    leaking into the UI.
    """
    from mlx_lm import generate

    model, tokenizer = _mlx_load(model_id)
    messages = [
        {"role": "system", "content": system},
        # /no_think is a Qwen3 convention; other models ignore the suffix.
        {"role": "user", "content": user + "\n/no_think"},
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
    return _strip_reasoning(out).strip()


# Matches <think>...</think>, <thinking>...</thinking>, <reasoning>...</reasoning>
# across newlines. Non-greedy so back-to-back blocks are handled individually.
_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_reasoning(text: str) -> str:
    """Remove reasoning-model scratchpad blocks from a completion.

    Handles the common cases:
      - <think>...</think> (Qwen3, DeepSeek-R1, QwQ)
      - <thinking>...</thinking> (some fine-tunes)
      - <reasoning>...</reasoning> (Claude-style when the model mimics it)
      - An unclosed <think>... that runs to the end (model hit max_tokens
        mid-reasoning; strip everything after the opener rather than ship
        a scratchpad with no answer).

    Conservative: if nothing matches, the original text comes back unchanged.
    """
    if not text:
        return text
    cleaned = _REASONING_BLOCK_RE.sub("", text)
    # Handle unclosed opener: if "<think>" appears with no matching close
    # after substitution, the model ran out of tokens mid-reasoning and
    # there's nothing useful after it anyway.
    low = cleaned.lower()
    open_idx = low.find("<think>")
    if open_idx >= 0 and "</think>" not in low[open_idx:]:
        cleaned = cleaned[:open_idx]
    return cleaned


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
