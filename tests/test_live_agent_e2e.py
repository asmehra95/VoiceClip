"""Live end-to-end voice-agent test: spoken question in, spoken answer out.

Drives the REAL pipeline (Silero VAD → cloud Whisper → cloud LLM → cloud
Kokoro) with a file-backed transport instead of mic/speakers, then checks
that the agent transcribed the question, replied without leaking its
reasoning, and produced actual audio. Needs the cloud stack reachable:
    VOICECLIP_LIVE_TESTS=1 pytest tests/test_live_agent_e2e.py -s
"""

import asyncio
import os
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VOICECLIP_LIVE_TESTS") != "1",
    reason="live cloud tests disabled (set VOICECLIP_LIVE_TESTS=1)",
)


def _spoken_pcm16(text: str, rate: int = 16000) -> bytes:
    d = tempfile.mkdtemp()
    aiff, wav = os.path.join(d, "q.aiff"), os.path.join(d, "q.wav")
    subprocess.run(["say", "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1", aiff, wav], check=True)
    import soundfile as sf
    data, _ = sf.read(wav, dtype="int16")
    return data.tobytes()


def _file_transport(question_pcm: bytes, captured: bytearray):
    from pipecat.frames.frames import InputAudioRawFrame, StartFrame
    from pipecat.transports.base_input import BaseInputTransport
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import BaseTransport, TransportParams

    params = TransportParams(audio_in_enabled=True, audio_out_enabled=True)

    class FileIn(BaseInputTransport):
        async def start(self, frame: StartFrame):
            await super().start(frame)
            await self.set_transport_ready(frame)
            self._feeder = asyncio.create_task(self._feed())

        async def _feed(self):
            rate = self.sample_rate
            block = int(rate * 0.02) * 2          # 20 ms of int16
            await asyncio.sleep(1.0)              # let the pipeline settle
            audio = question_pcm + b"\x00\x00" * rate * 60   # then a minute of silence
            for i in range(0, len(audio), block):
                await self.push_audio_frame(InputAudioRawFrame(
                    audio=audio[i:i + block], sample_rate=rate, num_channels=1))
                await asyncio.sleep(0.02)

        async def cleanup(self):
            await super().cleanup()
            t = getattr(self, "_feeder", None)
            if t:
                t.cancel()

    class FileOut(BaseOutputTransport):
        async def start(self, frame: StartFrame):
            await super().start(frame)
            await self.set_transport_ready(frame)

        async def write_audio_frame(self, frame) -> bool:
            captured.extend(frame.audio)
            return True

    class FileTransport(BaseTransport):
        def __init__(self):
            super().__init__()
            self._in, self._out = FileIn(params), FileOut(params)

        def input(self):
            return self._in

        def output(self):
            return self._out

    return FileTransport()


def test_agent_hears_thinks_and_speaks(tmp_path, monkeypatch):
    from voiceclip import agent, config

    config.load()
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))  # isolate event log
    agent._events = agent.EventLog(os.path.join(agent.agent_dir(), "events.jsonl"))
    events = []
    orig_emit = agent.EventLog.emit

    def spy(self, type_, **fields):
        events.append((type_, fields))
        orig_emit(self, type_, **fields)

    monkeypatch.setattr(agent.EventLog, "emit", spy)
    captured = bytearray()
    transport = _file_transport(_spoken_pcm16("What is the capital of France?"), captured)

    async def run():
        task = asyncio.create_task(agent._run_pipeline(transport=transport))
        for _ in range(90):                       # up to 90 s for a full turn
            await asyncio.sleep(1)
            if any(t == "assistant" for t, _ in events) and len(captured) > 24000 * 2:
                await asyncio.sleep(2)            # let the tail of speech flush
                break
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    asyncio.run(run())
    user = [f["text"] for t, f in events if t == "user"]
    replies = [f["text"] for t, f in events if t == "assistant"]
    print("\n  heard:", user, "\n  replied:", replies, f"\n  audio out: {len(captured) / 2 / 24000:.1f}s")
    assert user and "france" in " ".join(user).lower()
    assert replies, "agent never replied"
    assert "paris" in replies[0].lower()
    assert "the user is asking" not in replies[0].lower(), "reasoning leaked into speech"
    assert len(captured) > 24000 * 2, "no spoken audio came back"   # > 1 s at 24 kHz
