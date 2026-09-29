"""Voice assistant — talk to a journal-aware companion, hear it answer.

Flow (driven by hotkey.py on the assistant hotkey):
    hold  → record question (existing recorder; streaming STT if available)
    release → transcribe → retrieve relevant journal entries (FTS5 + recency)
            → LLM (Qwen3.5 via the gateway) → TTS (Kokoro) → play audio
    press again while speaking → interrupt (stop playback)

Everything goes through the LiteLLM gateway (auth + metering), over the
same tunnel as dictation, via the shared llm_provider cloud transport. The assistant is GROUNDED in the journal: the
system prompt carries the user's recent + matching dictations, so "what
did I say about the Q3 budget review?" gets a real answer, and it is
told to say so when the journal has nothing relevant rather than invent.

Deliberately push-to-talk: no wake word, no always-on mic, no barge-in
model — those are the hard, privacy-sensitive parts of voice assistants
and none are needed for a useful v1.
"""

import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime

log = logging.getLogger(__name__)

# How much journal to show the model: recent entries for "what was I
# doing" questions + FTS matches for specific topics. Bounded so the
# prompt stays fast on a 9B model.
_RECENT_ENTRIES = 12
_MATCH_ENTRIES = 12
_MAX_CONTEXT_CHARS = 6000

# Rolling conversation memory within a session (turns), so follow-ups
# ("and what about the second one?") work.
_MAX_TURNS = 8

PERSONA = """You are the user's voice companion inside VoiceClip, a dictation app. You have \
read their dictation journal (excerpts below) and you talk WITH them, not AT them.

Voice and manner:
- You're speaking out loud, so keep it short: one to three sentences unless asked for detail. \
No lists, no markdown, no headers — this is speech.
- Warm, quick, a little playful. Dry humor is welcome; sycophancy is not. Never say \
"Great question" or "As an AI".
- Be direct. Answer first, then a beat of color if it earns its place.
- When the journal has the answer, cite it naturally ("On Tuesday you said..." / "you mentioned \
the budget review twice this week").
- When the journal has NOTHING relevant, say so plainly and, if useful, answer from general \
knowledge — but never invent things the user supposedly said.
- Names and acronyms in the journal are correct as written (they came from the user's own \
vocabulary list); use them as-is.
- If the question is really a dictation ("write an email to..."), draft it briefly and offer to \
paste it.

Today is {today}."""


class AssistantError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Journal retrieval
# ---------------------------------------------------------------------------

def _journal_context(question: str) -> str:
    """Recent entries + FTS matches for the question, formatted for the prompt.
    Returns "" if history is disabled or empty."""
    try:
        from voiceclip import config, history
        if not config.HISTORY_ENABLED:
            return ""
        conn = history._conn
        if conn is None:
            history.init()
            conn = history._conn
        if conn is None:
            return ""
    except Exception as e:
        log.debug("journal unavailable: %s", e)
        return ""

    lines = []

    def add(rows, label):
        # No cross-block dedup on purpose: an entry that is BOTH recent and a
        # topic match should appear under "matching" too — that's the signal
        # telling the model which recent entries the question is about.
        block = []
        for _rid, ts, kind, text in rows:
            if not text:
                continue
            when = ts[:16].replace("T", " ") if ts else "?"
            block.append(f"- [{when}] ({kind}) {text.strip()}")
        if block:
            lines.append(f"{label}:")
            lines.extend(block)

    try:
        recent = conn.execute(
            "SELECT id, timestamp, kind, COALESCE(formatted_text, raw_text) "
            "FROM transcriptions WHERE archived_at IS NULL "
            "ORDER BY id DESC LIMIT ?", (_RECENT_ENTRIES,)
        ).fetchall()
        add(recent, "Most recent journal entries (newest first)")

        # FTS: OR the meaningful words so partial topic matches still hit.
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", question)
                 if w.lower() not in _STOP]
        if words:
            match = " OR ".join(f'"{w}"' for w in words[:12])
            rows = conn.execute(
                "SELECT t.id, t.timestamp, t.kind, COALESCE(t.formatted_text, t.raw_text) "
                "FROM transcriptions_fts f JOIN transcriptions t ON t.id = f.rowid "
                "WHERE transcriptions_fts MATCH ? AND t.archived_at IS NULL "
                "ORDER BY bm25(transcriptions_fts) LIMIT ?", (match, _MATCH_ENTRIES)
            ).fetchall()
            add(rows, "Journal entries matching the question")
    except Exception as e:
        log.debug("journal query failed: %s", e)

    text = "\n".join(lines)
    return text[:_MAX_CONTEXT_CHARS]


_STOP = frozenset(w for w in """the a an and or but if then what when where which who whom
why how did do does is are was were be been being have has had about tell me you your yours my
mine i we they it this that these those there here of in on at to for from with by as into over
under again please can could would should just like say said mention mentioned last week today
yesterday recently earlier anything something everything nothing thing things""".split())  # noqa: SIM905


# ---------------------------------------------------------------------------
# Gateway calls (LLM + TTS) — via the shared llm_provider "cloud" transport
# ---------------------------------------------------------------------------


def _llm_route() -> str:
    """Shared AI model when it's on the user's cloud; else 'assistant'."""
    from voiceclip import config
    if getattr(config, "AI_PROVIDER", "") == "cloud" and getattr(config, "AI_MODEL", ""):
        return config.AI_MODEL
    return "assistant"


def ask_llm(question: str, turns: list[dict]) -> str:
    """One chat completion, grounded in the journal. Returns reply text.

    Rides llm_provider.complete_cloud — the same transport summaries /
    polish / patterns use when their provider is "cloud". The assistant is
    just another consumer of the gateway's "assistant" route.
    """
    from voiceclip import llm_provider

    system = PERSONA.format(today=datetime.now().strftime("%A, %B %d %Y"))
    ctx = _journal_context(question)
    if ctx:
        system += "\n\n=== JOURNAL EXCERPTS ===\n" + ctx
    else:
        system += "\n\n(The journal is empty or unavailable right now.)"
    try:
        text = llm_provider.complete_cloud(
            system=system, user=question, turns=turns,
            model_id=_llm_route(), max_tokens=220, temperature=0.7,
        )
    except RuntimeError as e:
        raise AssistantError(str(e)) from e
    # Strip markdown emphasis — it's going to TTS. (<think> blocks are
    # already stripped by the provider layer.)
    return re.sub(r"[*_#`]+", "", text).strip()


def speak(text: str) -> str:
    """Synthesize `text` with Kokoro via the gateway. Returns a temp WAV path."""
    from voiceclip import llm_provider

    try:
        data = llm_provider.cloud_request("/v1/audio/speech", {
            "model": "tts",  # gateway route name -> speaches Kokoro
            "input": text[:1500],
            "voice": "af_heart",
            "response_format": "wav",
            "speed": 1.05,
        }, timeout=60)
    except RuntimeError as e:
        raise AssistantError(str(e)) from e
    fd, path = tempfile.mkstemp(prefix="voiceclip-tts-", suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------------
# Playback (macOS afplay — no new dependencies; interruptible)
# ---------------------------------------------------------------------------

class Player:
    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    def play(self, wav_path: str, block: bool = True):
        self.stop()
        with self._lock:
            self._proc = subprocess.Popen(
                ["afplay", wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            proc = self._proc
        if block:
            proc.wait()
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    def stop(self):
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None


# ---------------------------------------------------------------------------
# Session: conversation memory + orchestration
# ---------------------------------------------------------------------------

class Assistant:
    """Holds rolling conversation turns and the audio player."""

    def __init__(self):
        self.turns: list[dict] = []
        self.player = Player()
        self._lock = threading.Lock()

    def interrupt(self) -> bool:
        """Stop speaking. Returns True if it was speaking."""
        was = self.player.speaking
        self.player.stop()
        return was

    def respond(self, question: str) -> str:
        """Full turn: LLM → TTS → play. Returns the reply text."""
        t0 = time.time()
        reply = ask_llm(question, self.turns)
        t1 = time.time()
        if not reply:
            reply = "I didn't come up with anything useful there — try me again?"
        with self._lock:
            self.turns.extend([{"role": "user", "content": question},
                               {"role": "assistant", "content": reply}])
            self.turns = self.turns[-2 * _MAX_TURNS:]
        wav = speak(reply)
        t2 = time.time()
        log.info("Assistant: llm %.1fs, tts %.1fs — %r", t1 - t0, t2 - t1, reply[:80])
        threading.Thread(target=self.player.play, args=(wav,), daemon=True).start()
        return reply


_instance: Assistant | None = None
_instance_lock = threading.Lock()


def get_assistant() -> Assistant:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Assistant()
        return _instance
