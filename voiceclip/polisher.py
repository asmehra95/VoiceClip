"""Text polisher — runs raw dictation through a local LLM for cleanup.

Used by the "polish" hotkey profile. Takes raw Whisper output and
produces structured, grammatically correct prose. The prompt is
user-editable via the Settings tab.

Uses the summaries.* provider config for the LLM call — same model
that generates daily summaries handles the polish pass. This avoids
adding yet another provider/model config matrix.
"""

from __future__ import annotations

import logging

from voiceclip import config, llm_provider

log = logging.getLogger(__name__)


def polish(raw_text: str) -> str:
    """Run raw dictation text through the LLM for cleanup.

    Returns the polished text. If the LLM call fails or the provider
    is 'none', returns the original text unchanged (graceful degradation
    — the user still gets their dictation, just unpolished).
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    provider = config.SUMMARIES_PROVIDER
    if provider == "none":
        log.warning(
            "Polish hotkey fired but no LLM provider configured. "
            "Set summaries.provider in config to enable polishing. "
            "Pasting raw text instead."
        )
        return raw_text

    model_id = config.model_id_for("summaries")
    if model_id is None:
        return raw_text

    system = config.POLISH_PROMPT
    user = raw_text.strip()

    try:
        if provider == "local":
            # Don't append REASONING_DIRECTIVE — polish output should be
            # clean text only, no <think> tags. The prompt already asks
            # for "output only the cleaned text."
            result = llm_provider.complete_local(
                system=system, user=user,
                model_id=model_id, max_tokens=2000,
            )
        elif provider == "openai":
            result = llm_provider.complete_openai(
                system=system, user=user, model_id=model_id,
            )
        elif provider == "anthropic":
            result = llm_provider.complete_anthropic(
                system=system, user=user,
                model_id=model_id, max_tokens=2000,
            )
        else:
            return raw_text

        # If the LLM returned something useful, use it. Otherwise fall
        # back to the raw text.
        if result and result.strip():
            return result.strip()
        return raw_text

    except Exception as e:
        log.warning("Polish LLM call failed (%s); pasting raw text", e)
        return raw_text
