"""Local duplex audio transport for the voice agent, on sounddevice.

Pipecat ships a LocalAudioTransport built on pyaudio, but pyaudio needs a
C build against system PortAudio (broken CommandLineTools SDKs make that a
lottery — see the agent design notes). VoiceClip already depends on
sounddevice, which bundles PortAudio as a prebuilt wheel, so this is a
1:1 port of pipecat's transport onto it: 20ms int16 capture blocks pushed
into the pipeline, blocking playback writes on a single-thread executor.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams


class SoundDeviceTransportParams(TransportParams):
    """Transport params + optional explicit device indices."""

    input_device_index: int | None = None
    output_device_index: int | None = None


class SoundDeviceInputTransport(BaseInputTransport):
    """Microphone capture → InputAudioRawFrame (20ms int16 blocks)."""

    _params: SoundDeviceTransportParams

    def __init__(self, params: SoundDeviceTransportParams):
        super().__init__(params)
        self._stream = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._stream is None:
            import sounddevice as sd

            loop = self.get_event_loop()
            channels = self._params.audio_in_channels
            sample_rate = self.sample_rate
            blocksize = int(sample_rate / 100) * 2  # 20ms, mirrors pipecat

            def _callback(indata, frames, time_info, status):
                audio = InputAudioRawFrame(
                    audio=bytes(indata),
                    sample_rate=sample_rate,
                    num_channels=channels,
                )
                asyncio.run_coroutine_threadsafe(self.push_audio_frame(audio), loop)

            self._stream = sd.RawInputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                blocksize=blocksize,
                device=self._params.input_device_index,
                callback=_callback,
            )
            self._stream.start()
        await self.set_transport_ready(frame)

    async def cleanup(self):
        await super().cleanup()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SoundDeviceOutputTransport(BaseOutputTransport):
    """OutputAudioRawFrame → speakers, blocking writes off the event loop."""

    _params: SoundDeviceTransportParams

    def __init__(self, params: SoundDeviceTransportParams):
        super().__init__(params)
        self._stream = None
        # Single writer thread — frames arrive from one task, in order.
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._stream is None:
            import sounddevice as sd

            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self._params.audio_out_channels,
                dtype="int16",
                device=self._params.output_device_index,
            )
            self._stream.start()
        await self.set_transport_ready(frame)

    async def cleanup(self):
        try:
            await super().cleanup()
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        finally:
            self._executor.shutdown(wait=False)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._stream is not None:
            await self.get_event_loop().run_in_executor(
                self._executor, self._stream.write, frame.audio
            )
            return True
        return False


class SoundDeviceTransport(BaseTransport):
    """Unified local audio transport (mic in, speakers out)."""

    def __init__(self, params: SoundDeviceTransportParams):
        super().__init__()
        self._params = params
        self._input: SoundDeviceInputTransport | None = None
        self._output: SoundDeviceOutputTransport | None = None

    def input(self) -> SoundDeviceInputTransport:
        if self._input is None:
            self._input = SoundDeviceInputTransport(self._params)
        return self._input

    def output(self) -> SoundDeviceOutputTransport:
        if self._output is None:
            self._output = SoundDeviceOutputTransport(self._params)
        return self._output
