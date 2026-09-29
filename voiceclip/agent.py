"""Always-on duplex voice agent — talk naturally, it answers and acts.

`voiceclip agent` starts a Pipecat pipeline wired entirely to the user's
own cloud stack through the SSM tunnel:

    mic → Silero VAD → Whisper (gateway) → Qwen (tools) → Kokoro → speakers

Unlike the push-to-talk hotkeys, this is a continuous conversation with
built-in barge-in: start talking and it stops. Dictation-to-paste stays on
the hotkeys — an always-on channel can't reliably distinguish "type this"
from "do this", so the agent acts through explicit tools instead:

    web_search      — internet search (ddgs, client-side; SearXNG later)
    journal_search  — FTS over the dictation journal
    save_note       — append a note/reminder to the journal
    daily_summary   — read back a day's summary (needs summaries provider)
    cloud_status    — instance state + burn rate
    park_server     — stop the instance (ends the conversation, on purpose)

Tool handlers are plain functions (unit-testable without audio); pipecat
plumbing lives in run() so importing this module stays light.
"""

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime

log = logging.getLogger(__name__)

_SEARCH_RESULTS = 5
_VAD_MIN_VOLUME = 0.3
_JOURNAL_RESULTS = 10

AGENT_EXTRAS = """

You can DO things, not just talk, via tools:
- web_search: for anything current or outside the journal/your knowledge.
- journal_search: exact lookups in the user's dictation journal.
- save_note: when asked to remember/note/remind, save it and confirm briefly.
- daily_summary: to recap a day.
- cloud_status / park_server: the transcription server this very
  conversation runs on. park_server stops it — say goodbye first, the
  conversation ends when it parks.
Call a tool when it clearly helps; answer directly when it doesn't. Never
read URLs out loud — summarize what matters."""


# ---------------------------------------------------------------------------
# Session event log — read by the viewer's Agent tab
# ---------------------------------------------------------------------------
# One JSON object per line in ~/.voiceclip/agent/events.jsonl, truncated at
# each session start:  {"seq", "ts", "type", ...}
#   type=state      state: starting|ready|listening|thinking|speaking|stopped|error
#   type=user       text: what the user said (final transcription)
#   type=assistant  text: the full reply
#   type=tool       name: tool being called
# agent.pid holds the running process id while a session is live.

def agent_dir() -> str:
    from voiceclip import config
    d = os.path.join(config.CONFIG_DIR, "agent")
    os.makedirs(d, exist_ok=True)
    return d


class LevelMeter:
    """Writes the current mic level (0..1) to level.json ~4x/second."""

    def __init__(self, path: str):
        self._path = path
        self._tmp = path + ".tmp"
        self._sum = 0.0
        self._n = 0
        self._last = 0.0

    def feed(self, pcm: bytes):
        import numpy as np
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size:
            self._sum += float(np.mean(x * x)) * x.size
            self._n += x.size
        now = time.time()
        if now - self._last >= 0.25 and self._n:
            rms = (self._sum / self._n) ** 0.5
            self._sum, self._n, self._last = 0.0, 0, now
            # Map RMS to a friendly 0..1 meter (log scale: 0.0005 → 0, 0.05 → 1)
            import math
            level = max(0.0, min(1.0, (math.log10(max(rms, 1e-6)) + 3.3) / 2.0))
            try:
                with open(self._tmp, "w") as f:
                    json.dump({"ts": now, "level": round(level, 3), "rms": round(rms, 5)}, f)
                os.replace(self._tmp, self._path)
            except OSError:
                pass


_meter: "LevelMeter | None" = None


class EventLog:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._seq = 0
        with open(path, "w"):
            pass  # truncate: one file per session

    def emit(self, type_: str, **fields):
        with self._lock:
            self._seq += 1
            rec = {"seq": self._seq, "ts": time.time(), "type": type_, **fields}
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_events: EventLog | None = None


def _emit(type_: str, **fields):
    if _events is not None:
        try:
            _events.emit(type_, **fields)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Tool implementations (sync, testable; run in threads from handlers)
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """Search the web via DuckDuckGo (ddgs). Returns a compact digest."""
    try:
        from ddgs import DDGS

        rows = DDGS().text(query, max_results=_SEARCH_RESULTS)
    except Exception as e:
        return f"Search failed: {e}"
    if not rows:
        return "No results."
    lines = []
    for r in rows:
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        href = r.get("href") or ""
        lines.append(f"- {title}: {body} ({href})")
    return "\n".join(lines)


def journal_search(query: str) -> str:
    """FTS over the journal — same index the F2 assistant uses."""
    from voiceclip import history

    try:
        if history._conn is None:
            history.init()
        rows = history._conn.execute(
            "SELECT t.timestamp, t.kind, COALESCE(t.formatted_text, t.raw_text) "
            "FROM transcriptions_fts f JOIN transcriptions t ON t.id = f.rowid "
            "WHERE transcriptions_fts MATCH ? AND t.archived_at IS NULL "
            "ORDER BY bm25(transcriptions_fts) LIMIT ?",
            (query, _JOURNAL_RESULTS),
        ).fetchall()
    except Exception as e:
        return f"Journal search failed: {e}"
    if not rows:
        return "Nothing in the journal matches."
    return "\n".join(
        f"- [{(ts or '?')[:16]}] ({kind}) {text.strip()}"
        for ts, kind, text in rows if text
    )


def save_note(text: str) -> str:
    """Append a note to the journal (kind=reflection, app 'voice agent')."""
    from voiceclip import history

    try:
        history.save(text, text, 0.0, kind="reflection", app_name="voice agent")
        return "Saved to the journal."
    except Exception as e:
        return f"Could not save the note: {e}"


def daily_summary(day: str = "today") -> str:
    """Generate (or fetch cached) LLM summary for a day."""
    from voiceclip import summarizer

    try:
        result = summarizer.summarize_day(_resolve_day(day))
    except Exception as e:
        return f"Summary failed: {e}"
    if not result:
        return ("No summary available — either no entries that day or no "
                "summaries provider configured in Settings.")
    return result.get("summary") or "Summary came back empty."


def _resolve_day(day: str) -> str:
    from datetime import date, timedelta

    d = (day or "today").strip().lower()
    if d == "today":
        return date.today().isoformat()
    if d == "yesterday":
        return (date.today() - timedelta(days=1)).isoformat()
    return d  # trust YYYY-MM-DD; summarizer validates


def cloud_status() -> str:
    from voiceclip import cloud_control

    try:
        s = cloud_control.get_status()
        return (f"Instance {s.get('state', 'unknown')}"
                + (f", type {s['type']}" if s.get("type") else "")
                + ". Running costs about a dollar an hour; parked is nearly free.")
    except Exception as e:
        return f"Could not check: {e}"


def park_server() -> str:
    from voiceclip import cloud_control

    try:
        cloud_control.stop_instance()
        return ("Parking now — this conversation will end in a few seconds. "
                "Dictating later wakes it automatically.")
    except Exception as e:
        return f"Could not park the server: {e}"


# ---------------------------------------------------------------------------
# Pipecat wiring
# ---------------------------------------------------------------------------

def _runtime_info(base_url: str) -> dict:
    """Where every stage runs — shown on the Agent screen."""
    from voiceclip import cloud_control, config
    raw_cloud = ((getattr(config, "_raw", {}) or {}).get("cloud", {}) or {})
    llm_id = raw_cloud.get("llm_model") or cloud_control.DEFAULT_CLOUD_LLM
    route = _llm_route()
    llm = next((c["label"].split(" — ")[0] for c in cloud_control.CLOUD_LLM_CHOICES
                if c["id"] == llm_id), llm_id) if route == "assistant" else route
    return {
        "gateway": base_url,
        "server": " · ".join(x for x in (config.CLOUD_INSTANCE_ID, config.CLOUD_REGION) if x),
        "stt": "Whisper large-v3",
        "llm": llm,
        "tts": "Kokoro",
    }


def _llm_route() -> str:
    """Gateway route for the agent's LLM: the shared AI model when it lives
    on the user's cloud, else the default 'assistant' route (the agent is
    cloud-native — it needs the gateway for STT/TTS anyway)."""
    from voiceclip import config
    if getattr(config, "AI_PROVIDER", "") == "cloud" and getattr(config, "AI_MODEL", ""):
        return config.AI_MODEL
    return "assistant"


def _system_prompt() -> str:
    from voiceclip.assistant import PERSONA, _journal_context

    prompt = PERSONA.format(today=datetime.now().strftime("%A, %B %d %Y"))
    ctx = _journal_context("")  # recent entries only at session start
    if ctx:
        prompt += "\n\n=== JOURNAL EXCERPTS (session start) ===\n" + ctx
    return prompt + AGENT_EXTRAS


def _tools_schema():
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    def f(name, desc, props=None, required=None):
        return FunctionSchema(name=name, description=desc,
                              properties=props or {}, required=required or [])

    q = {"query": {"type": "string", "description": "The search query"}}
    return ToolsSchema(standard_tools=[
        f("web_search", "Search the internet for current information.",
          q, ["query"]),
        f("journal_search", "Search the user's dictation journal (FTS).",
          q, ["query"]),
        f("save_note", "Save a note or reminder into the journal.",
          {"text": {"type": "string", "description": "The note text"}},
          ["text"]),
        f("daily_summary", "Summary of a day's journal entries.",
          {"day": {"type": "string",
                   "description": "'today', 'yesterday', or YYYY-MM-DD"}}),
        f("cloud_status", "State of the user's cloud transcription server."),
        f("park_server", "Stop (park) the cloud server to save money. "
                         "Ends this conversation."),
    ])


_TOOL_IMPLS = {
    "web_search": lambda a: web_search(a.get("query", "")),
    "journal_search": lambda a: journal_search(a.get("query", "")),
    "save_note": lambda a: save_note(a.get("text", "")),
    "daily_summary": lambda a: daily_summary(a.get("day", "today")),
    "cloud_status": lambda a: cloud_status(),
    "park_server": lambda a: park_server(),
}


def _kokoro_tts_class():
    """OpenAITTSService minus its OpenAI-only voice allowlist.

    Pipecat rejects any voice not in OpenAI's catalogue ('alloy', 'nova'…)
    before making a request, so Kokoro's 'af_heart' never reached the
    gateway and the agent was silent. Same request, voice passed through.
    """
    from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
    from pipecat.services.openai.tts import OpenAITTSService

    class KokoroTTSService(OpenAITTSService):
        async def run_tts(self, text: str, context_id: str):
            create_params = {
                "input": text,
                "model": self._settings.model,
                "voice": self._settings.voice,
                "response_format": "pcm",
            }
            try:
                async with self._client.audio.speech.with_streaming_response.create(
                    **create_params
                ) as r:
                    if r.status_code != 200:
                        yield ErrorFrame(error=f"TTS failed (HTTP {r.status_code}): {await r.text()}")
                        return
                    await self.start_tts_usage_metrics(text)
                    async for chunk in r.iter_bytes(self.chunk_size):
                        if chunk:
                            await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
            except Exception as e:
                yield ErrorFrame(error=f"TTS failed: {e}")

    return KokoroTTSService


def _make_observer():
    """Pipecat observer → EventLog. Frames are pushed hop by hop through
    the pipeline, so the same frame is seen several times; dedupe by id."""
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        FunctionCallInProgressFrame,
        InputAudioRawFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.observers.base_observer import BaseObserver

    class _Observer(BaseObserver):
        def __init__(self):
            super().__init__()
            self._seen: set[int] = set()
            self._reply: list[str] = []
            self._state = None

        def _state_to(self, s):
            if s != self._state:
                self._state = s
                _emit("state", state=s)

        async def on_push_frame(self, data):
            frame = data.frame
            fid = getattr(frame, "id", None)
            if isinstance(frame, InputAudioRawFrame):
                # Meter only at the source hop (the transport's first push).
                if _meter is not None and fid not in self._seen:
                    self._seen.add(fid)
                    _meter.feed(frame.audio)
                return
            if fid in self._seen:
                return
            if isinstance(frame, (TranscriptionFrame, LLMTextFrame,
                                  FunctionCallInProgressFrame,
                                  LLMFullResponseStartFrame, LLMFullResponseEndFrame,
                                  UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                                  BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                self._seen.add(fid)
                if len(self._seen) > 5000:
                    self._seen.clear()
            else:
                return
            if isinstance(frame, UserStartedSpeakingFrame):
                self._state_to("listening")
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._state_to("thinking")
            elif isinstance(frame, TranscriptionFrame):
                if (frame.text or "").strip():
                    _emit("user", text=frame.text.strip())
            elif isinstance(frame, FunctionCallInProgressFrame):
                _emit("tool", name=frame.function_name)
            elif isinstance(frame, LLMFullResponseStartFrame):
                self._reply = []
            elif isinstance(frame, LLMTextFrame):
                self._reply.append(frame.text or "")
            elif isinstance(frame, LLMFullResponseEndFrame):
                text = "".join(self._reply).strip()
                self._reply = []
                if text:
                    _emit("assistant", text=text)
            elif isinstance(frame, BotStartedSpeakingFrame):
                self._state_to("speaking")
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._state_to("ready")

    return _Observer()


async def _run_pipeline(transport=None, observers_extra=()):
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.openai.stt import OpenAISTTService

    from voiceclip import config
    from voiceclip.agent_transport import (
        SoundDeviceTransport,
        SoundDeviceTransportParams,
    )

    base_url = (config.CLOUD_BASE_URL or "").strip().rstrip("/") + "/v1"
    api_key = (config.CLOUD_API_KEY or "").strip()

    if transport is None:  # tests inject a file-driven transport
        transport = SoundDeviceTransport(SoundDeviceTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ))
    stt = OpenAISTTService(
        settings=OpenAISTTService.Settings(model="whisper-large-v3"),
        api_key=api_key, base_url=base_url,
    )
    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(
            model=_llm_route(),
            system_instruction=_system_prompt(),
            temperature=0.7,
            max_tokens=300,
            # Qwen3.5 thinks out loud unless told not to — the F2 assistant
            # already sends this; without it the agent spoke its reasoning.
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
        api_key=api_key, base_url=base_url,
    )
    KokoroTTS = _kokoro_tts_class()
    tts = KokoroTTS(
        settings=KokoroTTS.Settings(model="tts", voice="af_heart"),
        api_key=api_key, base_url=base_url,
    )

    async def _dispatch(params):
        impl = _TOOL_IMPLS.get(params.function_name)
        if impl is None:
            await params.result_callback("Unknown tool.")
            return
        result = await asyncio.to_thread(impl, params.arguments or {})
        await params.result_callback(result)

    llm.register_function(None, _dispatch)

    context = LLMContext(tools=_tools_schema())
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(
            # Pipecat's default min_volume=0.6 drops quiet-but-clear speech
            # (headset mics in call mode measure ~0.58). Silero's speech
            # confidence already rejects noise, so the volume gate only
            # needs to exclude near-silence.
            min_volume=_VAD_MIN_VOLUME,
        ))),
        stt,
        aggregators.user(),
        llm,
        tts,
        transport.output(),
        aggregators.assistant(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True),
                        observers=[_make_observer(), *observers_extra])
    print("  🗣️  Voice agent live — just talk. Barge in to interrupt. Ctrl+C to quit.")
    _emit("runtime", **_runtime_info(base_url))
    _emit("state", state="ready")
    await PipelineRunner(handle_sigint=True).run(task)


def run() -> int:
    """Entry point for `voiceclip agent` (CLI or the viewer's Agent tab)."""
    global _events, _meter
    from voiceclip import config, tunnel

    d = agent_dir()
    _events = EventLog(os.path.join(d, "events.jsonl"))
    _meter = LevelMeter(os.path.join(d, "level.json"))
    pid_path = os.path.join(d, "agent.pid")
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))
    _emit("state", state="starting")
    try:
        return _run_session(config, tunnel)
    finally:
        _emit("state", state="stopped")
        try:
            os.unlink(pid_path)
        except OSError:
            pass


def _run_session(config, tunnel) -> int:

    if config.ENGINE != "cloud":
        print("The voice agent needs the cloud engine (set engine=cloud).")
        _emit("state", state="error", message="The agent needs the cloud engine.")
        return 1
    print("  Checking cloud gateway...")
    if not tunnel.ensure():
        print("  ❌ Cloud gateway unreachable — check `voiceclip cloud status`.")
        _emit("state", state="error",
              message="Can't reach your cloud server. Is it parked? Wake it in Settings.")
        return 1
    print("  ✅ Gateway up")
    try:
        from voiceclip import engine_cloud
        engine_cloud.load(config.CLOUD_MODEL)  # warms Whisper off-path
    except Exception as e:
        log.debug("warmup skipped: %s", e)
    try:
        asyncio.run(_run_pipeline())
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.shutdown()
    print("👋 Voice agent stopped.")
    return 0
