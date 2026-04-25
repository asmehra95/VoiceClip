"""Daily summary generator — pluggable LLM providers.

Off by default. Enable in config.json:

    "summaries": {
        "provider": "local",          // or "openai" or "anthropic"
        "local_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "openai_model": "gpt-4o-mini",
        "anthropic_model": "claude-haiku-4-5",
        "style": "descriptive"
    }

API keys come from environment variables only (OPENAI_API_KEY,
ANTHROPIC_API_KEY). They are never read from or written to the JSON config.

Only the LOCAL provider keeps your day on your Mac. The cloud providers
send your transcriptions and reflections over the wire — always an
explicit user choice.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from voiceclip import config, history

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_DESCRIPTIVE = """\
You are summarizing one person's day from their voice-dictation log.

Write 2 to 4 sentences, first-person (use "you"), in plain warm language.
No bullet points. No headers.

Describe what the person spent their day on, based on which apps they were
in and which topics recur. If any reflections (marked 💭) stand out,
quote one verbatim. If nothing reflective came up, do not fabricate any.

Do not interpret feelings. Describe, don't judge. If the day is light on
content, say so briefly.
"""

_SYSTEM_PROMPT_REFLECTIVE = """\
You are summarizing one person's day from their voice-dictation log.

Write 2 to 4 sentences, first-person (use "you"), in a warm reflective tone.
No bullet points. No headers.

Focus on what the person seemed to be thinking about, drawing mostly from
their reflections (marked 💭). Ground any reflective observation in a
direct quote from their own reflections. Do not speculate beyond what the
log shows.

If the day is light on content or reflections, say so briefly.
"""


def _build_prompt(date: str, entries: list[dict], style: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given day."""
    system = _SYSTEM_PROMPT_REFLECTIVE if style == "reflective" else _SYSTEM_PROMPT_DESCRIPTIVE
    lines = [f"Date: {date}", ""]
    for e in entries:
        ts = e.get("timestamp") or ""
        try:
            time_str = datetime.fromisoformat(ts).strftime("%I:%M %p").lstrip("0")
        except Exception:
            time_str = ts[11:16] if len(ts) >= 16 else ts
        marker = "💭" if e.get("kind") == "reflection" else "📝"
        app = e.get("app_name") or "unknown"
        text = (e.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{time_str}] {marker} ({app}): {text}")
    lines.append("")
    lines.append("Write the summary now.")
    user = "\n".join(lines)
    return system, user


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _summarize_local(system: str, user: str, model_id: str) -> str:
    """Use mlx-lm for on-device generation."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        raise RuntimeError(
            "Local summaries require mlx-lm. Install with: pip install mlx-lm"
        )

    log.info("Loading local model: %s (first run downloads the weights)", model_id)
    model, tokenizer = load(model_id)

    # Apply the chat template if the tokenizer supports one; otherwise fall back.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{system}\n\n{user}\n\nSummary:"

    out = generate(model, tokenizer, prompt=prompt, max_tokens=400, verbose=False)
    # Some mlx-lm versions echo the prompt; strip if they do.
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out.strip()


def _summarize_openai(system: str, user: str, model_id: str) -> str:
    """Use OpenAI's chat completions. Requires OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI summaries need OPENAI_API_KEY in your environment."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI summaries require the 'openai' package. "
            "Install with: pip install openai"
        )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


def _summarize_anthropic(system: str, user: str, model_id: str) -> str:
    """Use Anthropic's Messages API. Requires ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Anthropic summaries need ANTHROPIC_API_KEY in your environment."
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "Anthropic summaries require the 'anthropic' package. "
            "Install with: pip install anthropic"
        )
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model_id,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Anthropic returns a list of content blocks; concatenate the text ones.
    parts = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def summarize_day(date: str, *, force: bool = False) -> dict | None:
    """Generate (or return cached) summary for a YYYY-MM-DD day.

    Returns a dict with keys: summary, provider, model, style, entry_count.
    Returns None if the provider is 'none' or there are no entries.
    Raises RuntimeError on provider errors (missing API key, etc).
    """
    provider = config.SUMMARIES_PROVIDER
    if provider == "none":
        return None

    entries = history.entries_for_day(date)
    if not entries:
        return None

    cached = history.get_day_summary(date)
    if cached and not force:
        # For past days: always reuse cache.
        # For today: regenerate only if entries have grown since the cache.
        today = datetime.now().strftime("%Y-%m-%d")
        if date != today:
            return cached
        if cached.get("entry_count", 0) >= len(entries):
            return cached

    # Resolve model id for the chosen provider
    if provider == "local":
        model_id = config.SUMMARIES_LOCAL_MODEL
        fn = _summarize_local
    elif provider == "openai":
        model_id = config.SUMMARIES_OPENAI_MODEL
        fn = _summarize_openai
    elif provider == "anthropic":
        model_id = config.SUMMARIES_ANTHROPIC_MODEL
        fn = _summarize_anthropic
    else:
        return None

    style = config.SUMMARIES_STYLE
    system, user = _build_prompt(date, entries, style)
    summary = fn(system, user, model_id)

    if summary:
        history.save_day_summary(
            date, summary,
            provider=provider, model=model_id,
            style=style, entry_count=len(entries),
        )

    return {
        "date": date,
        "summary": summary,
        "provider": provider,
        "model": model_id,
        "style": style,
        "entry_count": len(entries),
    }


def cloud_provider_warning() -> str | None:
    """Return a human-readable warning if a cloud provider is configured."""
    if config.SUMMARIES_PROVIDER in ("openai", "anthropic"):
        return (
            f"Summaries: cloud provider '{config.SUMMARIES_PROVIDER}' is enabled. "
            "Your day's entries will be sent to that provider when a summary "
            "is generated. Nothing else about VoiceClip uses the network."
        )
    return None
