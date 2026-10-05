"""Stats tab: streaks, time-saved estimate, AI-fix recording, endpoint."""

import json
from datetime import date, timedelta

from tests.conftest import http_get
from voiceclip import history, polisher
from voiceclip.viewer.routes import stats


def _d(days_ago):
    return (date.today() - timedelta(days=days_ago)).isoformat()


def test_streaks():
    assert stats._streaks(set()) == (0, 0)
    assert stats._streaks({_d(0), _d(1), _d(2), _d(5), _d(6)}) == (3, 3)
    assert stats._streaks({_d(1), _d(2)}) == (2, 2)      # today not dictated yet
    assert stats._streaks({_d(3), _d(4), _d(5), _d(6)}) == (0, 4)


def test_minutes_saved():
    # 1,200 words: typing 30 min at 40 wpm, saying 8 min at 150 wpm
    assert abs(stats.minutes_saved(1200) - 22.0) < 1e-9
    assert stats.minutes_saved(0) == 0


def test_word_changes_ignores_formatting():
    assert polisher.word_changes(
        "if Phenopsis is not able. To do so", "if FinOps is not able to do so") == [["Phenopsis", "FinOps"]]
    assert polisher.word_changes("we do e invoicing", "We do e-invoicing.") == []


def test_ai_fixes_saved_and_counted(live_viewer):
    history.save("x", "Escalate if FinOps cannot.", 1.0, kind="transcription",
                 ai_fixes=[["Phenopsis", "FinOps"]])
    history.save("y", "Plain one two three.", 1.0, kind="transcription")
    history.save("q", "what's up", 0.5, kind="question")
    row = history._conn.execute(
        "SELECT ai_fixes FROM transcriptions WHERE formatted_text LIKE 'Escalate%'").fetchone()
    assert json.loads(row[0]) == [["Phenopsis", "FinOps"]]
    d = http_get(f"{live_viewer}/api/stats")
    assert d["dictations"] == 2 and d["words"] == 8      # questions aren't dictations
    assert d["ai_fixes"] == 1 and d["ai_recent"] == [["Phenopsis", "FinOps"]]
    assert d["questions"] == 1 and d["current_streak"] == 1
    assert len(d["daily"]) == 14 and d["daily"][-1]["words"] == 8
    earned = {b["id"] for b in d["badges"] if b["earned"]}
    assert {"first_words", "ai1"} <= earned and "k1" not in earned


def test_checked_and_polished_counted_separately(live_viewer):
    history.save("a", "Escalate if FinOps cannot.", 1.0, kind="transcription",
                 ai_fixes=[["Phenopsis", "FinOps"]], ai_pass="checked")
    history.save("b", "Nothing to fix here.", 1.0, kind="transcription", ai_pass="checked")
    history.save("c", "We can set up time to understand it.", 1.0, kind="transcription",
                 ai_fixes=[["and", "to"]], ai_pass="polished")
    d = http_get(f"{live_viewer}/api/stats")
    assert d["ai_checked"] == 2 and d["ai_polished"] == 1
    assert d["ai_fixes"] == 1                     # polish rewrites aren't "fixes"
    assert d["ai_polish_changes"] == 1
    assert d["ai_recent"] == [["Phenopsis", "FinOps"]]
