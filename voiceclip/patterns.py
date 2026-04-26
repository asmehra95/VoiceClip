"""Patterns — longitudinal view of your recent history.

Reads the last N days of entries (default 7) + their cached daily summaries
and asks an LLM to produce:

  - "occupied_with": 2-3 sentences on what the user spent time on
  - "themes": recurring threads, each with a quoted reflection
  - "suggestions": learning topics, each grounded in a direct quote

Off by default. Enable in config.json:

    "patterns": {
        "provider": "local",          // or "openai" / "anthropic"
        "local_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "window_days": 7
    }

This is the most privacy-sensitive feature in VoiceClip — it reads a week
of your journal in one prompt. Default provider is local for that reason.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from voiceclip import config, history

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a thoughtful observer reading one person's voice-dictation log from
the past several days. You are NOT a coach. You describe what you see; you
suggest only when a suggestion is obviously grounded in the log itself.

You will receive:
  - A list of daily summaries (1 per day)
  - The full text of every REFLECTION (marked 💭) across the window
  - A distribution of which apps they dictated in

Produce a JSON object with exactly these keys:

{
  "occupied_with": "2 to 3 sentences, first-person ('you'), plain language.",

  "themes": [
    {
      "title": "short noun phrase (2-5 words)",
      "reflection_count": 3,
      "quote": "a single short reflection quoted verbatim from the log"
    }
  ],

  "suggestions": [
    {
      "topic": "a concrete thing to research, 3-8 words",
      "reason": "one sentence, quoting or referencing the log",
      "grounding_quote": "a short quote from their own reflection that supports this"
    }
  ]
}

Rules:
  - Return STRICT JSON. No prose before or after. No markdown fencing.
  - Every "quote" and "grounding_quote" must be a direct substring of a
    reflection in the log. If you cannot find a real quote to support a
    theme, omit that theme.
  - "suggestions" must be grounded: only include topics the user explicitly
    expressed interest in learning, or questions they asked multiple times
    without researching. If you cannot find at least one well-grounded
    suggestion, return "suggestions": []. An empty list is correct; made-up
    suggestions are wrong.
  - Maximum 4 themes, maximum 3 suggestions.
  - Keep "occupied_with" under 350 characters.
"""


def _build_user_message(window_days: int) -> tuple[str, dict]:
    """Build the user-facing prompt for the patterns call. Returns
    (user_prompt, stats_dict_for_response)."""
    end = datetime.now()
    start = end - timedelta(days=window_days)
    end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = start.strftime("%Y-%m-%d")

    # Gather all reflections in the window (direct — these are the "signal")
    reflections = history.entries_for_window(start_str, end_str, kind="reflection")
    # Daily summaries (cached from the summarizer)
    daily_summaries = []
    for i in range(window_days):
        d = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        s = history.get_day_summary(d)
        if s and s.get("summary"):
            daily_summaries.append({"date": d, "summary": s["summary"]})
    # App distribution
    apps = history.app_distribution_for_window(start_str, end_str, top_n=8)
    # Total counts
    all_entries = history.entries_for_window(start_str, end_str)
    n_transcriptions = sum(1 for e in all_entries if e["kind"] == "transcription")
    n_reflections = len(reflections)

    lines = [
        f"Window: {start_str} to {end_str} ({window_days} days)",
        f"Counts: {n_transcriptions} transcriptions, {n_reflections} reflections",
        "",
        "Apps (top, by count):",
    ]
    for a in apps:
        lines.append(f"  - {a['name']}: {a['count']}")
    lines.append("")
    lines.append("Daily summaries (from cache, newest first):")
    if daily_summaries:
        for s in daily_summaries:
            lines.append(f"  [{s['date']}] {s['summary']}")
    else:
        lines.append("  (no cached summaries)")
    lines.append("")
    lines.append("Reflections (verbatim):")
    if reflections:
        for r in reflections:
            try:
                when = datetime.fromisoformat(r["timestamp"]).strftime("%a %I:%M %p")
            except Exception:
                when = r["timestamp"]
            text = r["text"].replace("\n", " ").strip()
            lines.append(f'  [{when}] 💭 "{text}"')
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Return the JSON object now.")

    stats = {
        "start_date": start_str,
        "end_date": end.strftime("%Y-%m-%d"),
        "transcription_count": n_transcriptions,
        "reflection_count": n_reflections,
        "apps": apps,
        "cached_summaries": len(daily_summaries),
    }
    return "\n".join(lines), stats


def _parse_json_safely(raw: str) -> dict:
    """Extract the first JSON object in the response, tolerate minor noise."""
    s = (raw or "").strip()
    # Strip common code-fence wrappers even though the prompt forbids them
    if s.startswith("```"):
        # Drop leading ```json / ``` and the trailing ```
        first_nl = s.find("\n")
        if first_nl >= 0:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # If the model added prose before or after, try to pull the first {...} span
    try:
        return json.loads(s)
    except Exception:
        pass
    # Best-effort: find outermost {...}
    lo = s.find("{")
    hi = s.rfind("}")
    if lo >= 0 and hi > lo:
        candidate = s[lo:hi + 1]
        return json.loads(candidate)
    raise ValueError("model did not return valid JSON")


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _run_local(system: str, user: str, model_id: str) -> str:
    try:
        from mlx_lm import load, generate
    except ImportError:
        raise RuntimeError(
            "mlx-lm is not installed. Install it with:\n"
            "    ~/.voiceclip/.venv/bin/pip install mlx-lm"
        )
    log.info("Loading local model: %s", model_id)
    model, tokenizer = load(model_id)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{system}\n\n{user}\n\n"
    out = generate(model, tokenizer, prompt=prompt, max_tokens=900, verbose=False)
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out.strip()


def _run_openai(system: str, user: str, model_id: str) -> str:
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
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


def _run_anthropic(system: str, user: str, model_id: str) -> str:
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
    resp = client.messages.create(
        model=model_id,
        max_tokens=900,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in resp.content:
        t = getattr(block, "text", None)
        if t:
            parts.append(t)
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_patterns(window_days: int | None = None) -> dict:
    """Produce a patterns view for the current window. Returns a dict with
    the LLM output plus metadata."""
    provider = config.PATTERNS_PROVIDER
    if provider == "none":
        raise RuntimeError(
            "Patterns are off. Set patterns.provider in ~/.voiceclip/config.json."
        )
    if window_days is None:
        window_days = config.PATTERNS_WINDOW_DAYS

    user_prompt, stats = _build_user_message(window_days)

    # If we have zero signal, don't spend a model call on it.
    if stats["transcription_count"] == 0 and stats["reflection_count"] == 0:
        return {
            "ok": True,
            "empty": True,
            "stats": stats,
            "occupied_with": "",
            "themes": [],
            "suggestions": [],
        }

    if provider == "local":
        model_id = config.PATTERNS_LOCAL_MODEL
        fn = _run_local
    elif provider == "openai":
        model_id = config.PATTERNS_OPENAI_MODEL
        fn = _run_openai
    elif provider == "anthropic":
        model_id = config.PATTERNS_ANTHROPIC_MODEL
        fn = _run_anthropic
    else:
        raise RuntimeError(f"unknown patterns provider: {provider}")

    raw = fn(_SYSTEM_PROMPT, user_prompt, model_id)
    try:
        parsed = _parse_json_safely(raw)
    except Exception as e:
        log.warning("Patterns: model returned unparseable output: %s", e)
        raise RuntimeError(
            "The model returned output I couldn't parse. Try refreshing; "
            "larger models tend to produce cleaner JSON."
        )

    themes = parsed.get("themes") or []
    suggestions = parsed.get("suggestions") or []
    # Sanity-bound
    themes = themes[:4]
    suggestions = suggestions[:3]

    return {
        "ok": True,
        "empty": False,
        "provider": provider,
        "model": model_id,
        "stats": stats,
        "occupied_with": (parsed.get("occupied_with") or "").strip(),
        "themes": themes,
        "suggestions": suggestions,
    }


def cloud_provider_warning() -> str | None:
    if config.PATTERNS_PROVIDER in ("openai", "anthropic"):
        return (
            f"Patterns: cloud provider '{config.PATTERNS_PROVIDER}' is enabled. "
            "When you open the Patterns tab, your recent reflections and daily "
            "summaries are sent to the provider."
        )
    return None
