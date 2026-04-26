"""Research assistant — takes a topic you captured, produces a brief.

Uses cloud providers only (OpenAI or Anthropic). Both support a web search
tool the model decides whether to call based on the question. Conceptual
questions stay cheap; specific or current questions auto-upgrade to
web-searched answers with sources.

Off by default. Enable in config.json:

    "research": {
        "provider": "openai",              // or "anthropic"
        "openai_model": "gpt-4o-mini",
        "anthropic_model": "claude-haiku-4-5"
    }

API keys come from env only: OPENAI_API_KEY, ANTHROPIC_API_KEY.

This feature sends the topic string (and nothing else about your history)
to the chosen provider. It is the only feature in VoiceClip that uses the
network, and only when you explicitly trigger a research.
"""

from __future__ import annotations

import logging
import os

from voiceclip import config, history

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a personal research assistant producing short, scannable briefs
for a curious knowledge worker who is too busy to do the research themselves.

For the topic given, write a brief with these sections in markdown:

**What it is** — 2-3 sentences, plain language.
**Why it matters** — 2-3 sentences on who cares and why.
**Key tradeoffs / concepts** — 3-5 bullet points, concise.
**Things to think about** — 2-3 short prompts for reflection or further exploration.

Rules:
- If the topic is time-sensitive, specific to a product, or likely to
  require current information, USE THE WEB SEARCH TOOL and cite sources.
- If the topic is conceptual and well-established, answer from knowledge.
  No web call needed.
- Be direct. No preamble. No "great question". No "here is a brief".
- Total length: under 300 words.
- Use plain markdown — headings as **bold**, bullets as `- `, no fancy formatting.
"""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _research_openai(topic: str, model_id: str) -> tuple[str, list, bool]:
    """Return (brief_text, sources, used_web_search)."""
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

    client = OpenAI(api_key=api_key)

    # Use the Responses API with the web_search tool. The model decides
    # whether to call it. We prefer Responses because the tool surface is
    # cleaner than Chat Completions + tool_choice plumbing.
    try:
        resp = client.responses.create(
            model=model_id,
            tools=[{"type": "web_search"}],
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": topic},
            ],
        )
    except Exception as e:
        # Fall back: some OpenAI models don't support the web_search tool.
        # Drop the tool and retry.
        msg = str(e).lower()
        if "web_search" in msg or "tool" in msg or "not supported" in msg:
            log.info("web_search not supported by %s; falling back to plain call", model_id)
            resp = client.responses.create(
                model=model_id,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": topic},
                ],
            )
        else:
            raise

    # Extract final text + any citations / URLs.
    text = getattr(resp, "output_text", None) or ""
    used_web = False
    sources: list[dict] = []
    # Walk response items looking for web_search tool calls and URL citations.
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
    # Dedupe sources
    seen = set()
    deduped = []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            deduped.append(s)
    return text.strip(), deduped, used_web


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _research_anthropic(topic: str, model_id: str) -> tuple[str, list, bool]:
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

    client = anthropic.Anthropic(api_key=api_key)

    # Anthropic's server-side web search tool. Model decides whether to use it.
    try:
        resp = client.messages.create(
            model=model_id,
            max_tokens=800,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": topic}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )
    except Exception as e:
        msg = str(e).lower()
        # Fall back if the tool isn't available for this model/plan
        if "web_search" in msg or "tool" in msg or "unsupported" in msg:
            log.info("web_search not supported by %s; falling back to plain call", model_id)
            resp = client.messages.create(
                model=model_id,
                max_tokens=800,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": topic}],
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
            # Citations live on text blocks
            citations = getattr(block, "citations", None) or []
            for cit in citations:
                url = getattr(cit, "url", None)
                title = getattr(cit, "title", None) or url
                if url:
                    sources.append({"title": title or url, "url": url})
        elif btype == "server_tool_use" or "web_search" in (btype or ""):
            used_web = True
    seen = set()
    deduped = []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            deduped.append(s)
    return "".join(text_parts).strip(), deduped, used_web


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def research_topic(entry_id: int) -> dict | None:
    """Generate a research brief for a topic entry. Stores in `research_briefs`.

    Returns the brief dict on success. Raises RuntimeError on provider errors
    (missing key, missing package, etc).
    """
    provider = config.RESEARCH_PROVIDER
    if provider == "none":
        raise RuntimeError(
            "Research is disabled. Enable it in ~/.voiceclip/config.json "
            'under "research": {"provider": "openai"} (or "anthropic").'
        )

    topic = history.get_topic(entry_id)
    if topic is None:
        return None

    # Mark as running so the UI can reflect it.
    running_id = history.save_brief(entry_id, status="running", provider=provider)

    if provider == "openai":
        model_id = config.RESEARCH_OPENAI_MODEL
        fn = _research_openai
    elif provider == "anthropic":
        model_id = config.RESEARCH_ANTHROPIC_MODEL
        fn = _research_anthropic
    else:
        history.save_brief(entry_id, status="failed", provider=provider,
                           error=f"unknown provider {provider}")
        return None

    try:
        text, sources, used_web = fn(topic["text"], model_id)
    except RuntimeError:
        history.save_brief(entry_id, status="failed", provider=provider, model=model_id,
                           error="configuration error")
        raise
    except Exception as e:
        log.exception("Research call failed")
        history.save_brief(entry_id, status="failed", provider=provider, model=model_id,
                           error=str(e))
        raise RuntimeError(f"Research failed: {e}")

    history.save_brief(
        entry_id,
        status="done",
        brief_text=text,
        sources=sources,
        provider=provider,
        model=model_id,
        used_web_search=used_web,
    )
    return history.latest_brief(entry_id)


def network_warning() -> str | None:
    """Return a human-readable warning if a cloud research provider is configured."""
    if config.RESEARCH_PROVIDER in ("openai", "anthropic"):
        return (
            f"Research: cloud provider '{config.RESEARCH_PROVIDER}' is enabled. "
            "When you research a topic, that topic is sent to the provider."
        )
    return None
