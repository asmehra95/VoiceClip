"""Achievements page: words dictated, time saved, AI fixes, streaks, badges.

GET /api/stats → one JSON blob for the Stats tab.

Time saved is an estimate: typing the words at TYPING_WPM versus saying
them at SPEAKING_WPM. (The per-entry duration column records how long
transcription took, not how long the user spoke, so it can't be used.)
"""

import json
from collections import Counter
from datetime import date, timedelta

from voiceclip import config, history
from voiceclip.viewer.routes import register_get

TYPING_WPM = 40      # average typist
SPEAKING_WPM = 150   # conversational speech

_DICTATED = ("transcription", "reflection")

# (id, emoji, title, description, metric, target)
BADGES = [
    ("first_words", "🎙️", "First words", "Dictate for the first time", "dictations", 1),
    ("hundred", "💯", "Centurion", "100 dictations", "dictations", 100),
    ("k1", "📝", "Page one", "1,000 words dictated", "words", 1_000),
    ("k10", "📚", "Short story", "10,000 words dictated", "words", 10_000),
    ("k50", "📖", "Novella", "50,000 words dictated", "words", 50_000),
    ("k100", "🏛️", "Novelist", "100,000 words dictated", "words", 100_000),
    ("hour", "⏱️", "Hour back", "Save an hour of typing", "minutes_saved", 60),
    ("day", "🌅", "Day back", "Save 8 hours of typing", "minutes_saved", 480),
    ("streak7", "🔥", "On a roll", "Dictate 7 days in a row", "best_streak", 7),
    ("streak30", "🌋", "Habit", "Dictate 30 days in a row", "best_streak", 30),
    ("ai1", "🪄", "Good catch", "First AI fix of a misheard word", "ai_fixes", 1),
    ("ai50", "🧠", "Sharp ears", "50 misheard words fixed by AI", "ai_fixes", 50),
    ("vocab25", "🔤", "Wordsmith", "25 words in your vocabulary", "vocab", 25),
    ("ask10", "🗣️", "Curious", "Ask the assistant 10 questions", "questions", 10),
]


def _streaks(days: set[str]) -> tuple[int, int]:
    """(current, best) runs of consecutive days. The current streak still
    counts if today has no entries yet but yesterday does."""
    if not days:
        return 0, 0
    parsed = sorted(date.fromisoformat(d) for d in days)
    best = run = 1
    for prev, cur in zip(parsed, parsed[1:], strict=False):
        run = run + 1 if (cur - prev).days == 1 else 1
        best = max(best, run)
    today = date.today()
    start = today if today in parsed else today - timedelta(days=1)
    current = 0
    day_set = set(parsed)
    while start in day_set:
        current += 1
        start -= timedelta(days=1)
    return current, best


def minutes_saved(words: int) -> float:
    return max(0.0, words / TYPING_WPM - words / SPEAKING_WPM)


def compute() -> dict:
    conn = history._conn
    if conn is None:
        return {"error": "journal is off"}
    ph = ",".join("?" * len(_DICTATED))
    rows = conn.execute(
        f"SELECT substr(timestamp, 1, 10), word_count, app_name, ai_fixes, ai_pass FROM transcriptions "
        f"WHERE kind IN ({ph}) AND is_research_topic = 0", _DICTATED).fetchall()
    questions = conn.execute(
        "SELECT COUNT(*) FROM transcriptions WHERE kind = 'question'").fetchone()[0]

    words_by_day: Counter = Counter()
    apps: Counter = Counter()
    fixes: list[list[str]] = []
    fixed_entries = checked = polished = polish_changes = 0
    for day, wc, app, ai, ai_pass in rows:
        if ai_pass == "checked":
            checked += 1
        elif ai_pass == "polished":
            polished += 1
        words_by_day[day] += wc or 0
        if app:
            apps[app] += wc or 0
        if ai:
            try:
                pairs = json.loads(ai)
            except ValueError:
                pairs = []
            if pairs and ai_pass == "polished":
                polish_changes += len(pairs)        # rewrites, not fixes
            elif pairs:
                fixed_entries += 1
                fixes.extend(pairs)

    words = sum(words_by_day.values())
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    words_today = words_by_day.get(today.isoformat(), 0)
    words_week = sum(w for d, w in words_by_day.items() if date.fromisoformat(d) >= week_start)
    current, best = _streaks(set(words_by_day))
    best_day = max(words_by_day.items(), key=lambda kv: kv[1]) if words_by_day else None

    metrics = {
        "dictations": len(rows),
        "words": words,
        "minutes_saved": minutes_saved(words),
        "best_streak": best,
        "ai_fixes": len(fixes),
        "vocab": len(config.CUSTOM_VOCABULARY),
        "questions": questions,
    }
    badges = [{
        "id": bid, "emoji": emoji, "title": title, "description": desc,
        "earned": metrics[metric] >= target,
        "progress": min(1.0, metrics[metric] / target) if target else 1.0,
    } for bid, emoji, title, desc, metric, target in BADGES]

    last14 = [(today - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
    return {
        **metrics,
        "words_today": words_today,
        "words_week": words_week,
        "minutes_saved_week": minutes_saved(words_week),
        "current_streak": current,
        "days_active": len(words_by_day),
        "best_day": {"date": best_day[0], "words": best_day[1]} if best_day else None,
        "top_apps": [{"name": n, "words": w} for n, w in apps.most_common(5)],
        "ai_fixed_entries": fixed_entries,
        "ai_checked": checked,
        "ai_polished": polished,
        "ai_polish_changes": polish_changes,
        "ai_recent": fixes[-6:][::-1],
        "autocorrect_on": bool(getattr(config, "AUTOCORRECT", False)),
        "daily": [{"date": d, "words": words_by_day.get(d, 0)} for d in last14],
        "badges": badges,
        "assumptions": {"typing_wpm": TYPING_WPM, "speaking_wpm": SPEAKING_WPM},
    }


def _get_stats(req, query):
    req._json(compute())


register_get("/api/stats", _get_stats)
