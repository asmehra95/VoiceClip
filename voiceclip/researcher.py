"""Research assistant — produces a scannable brief for a queued topic.

Uses cloud providers only. Both OpenAI and Anthropic support a server-side
web search tool; the model decides per-topic whether to call it. Conceptual
questions stay cheap; specific or current questions auto-upgrade to searched
answers with sources.

Off by default. Enable in config.json:

    "research": {
        "provider": "openai",              // or "anthropic"
        "openai_model": "gpt-4o-mini",
        "anthropic_model": "claude-haiku-4-5"
    }

Transport is handled by voiceclip.llm_provider. This module is prompt +
persistence only.

This is the one feature that always uses the network when enabled. The
topic string is sent to the chosen provider; if the model uses web search,
that topic is also sent to the provider's search backend.
"""

from __future__ import annotations

import logging

from voiceclip import config, history, llm_provider

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
# Public entry point
# ---------------------------------------------------------------------------

def _run(provider: str, topic: str, model_id: str) -> tuple[str, list, bool]:
    if provider == "openai":
        return llm_provider.complete_openai_with_web_search(
            system=_SYSTEM_PROMPT, user=topic, model_id=model_id,
        )
    if provider == "anthropic":
        return llm_provider.complete_anthropic_with_web_search(
            system=_SYSTEM_PROMPT, user=topic, model_id=model_id,
        )
    raise RuntimeError(f"unknown research provider: {provider}")


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
    history.save_brief(entry_id, status="running", provider=provider)

    if provider == "openai":
        model_id = config.RESEARCH_OPENAI_MODEL
    elif provider == "anthropic":
        model_id = config.RESEARCH_ANTHROPIC_MODEL
    else:
        history.save_brief(entry_id, status="failed", provider=provider,
                           error=f"unknown provider {provider}")
        return None

    try:
        text, sources, used_web = _run(provider, topic["text"], model_id)
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
    if config.RESEARCH_PROVIDER in ("openai", "anthropic"):
        return (
            f"Research: cloud provider '{config.RESEARCH_PROVIDER}' is enabled. "
            "When you research a topic, that topic is sent to the provider."
        )
    return None
