"""Live end-to-end streaming regressions — the scenarios that actually
shipped as bugs. Requires the cloud stack + tunnel; opt in with:

    VOICECLIP_LIVE_TESTS=1 pytest tests/test_live_streaming.py
"""
import os
import subprocess
import time
import wave

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("VOICECLIP_LIVE_TESTS"),
    reason="live cloud tests disabled (set VOICECLIP_LIVE_TESTS=1)",
)

SENT = "the belgium one i think we had some work on the payment side as well can you check that"


def _audio(quiet=True, fade_in_ms=400, fade_out_ms=0):
    subprocess.run(["say", "-v", "Samantha", "-o", "/tmp/_lt.aiff", SENT], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1",
                    "/tmp/_lt.aiff", "/tmp/_lt.wav"], check=True)
    w = wave.open("/tmp/_lt.wav", "rb")
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    if quiet:
        rms = np.sqrt(np.mean((pcm / 32768.0) ** 2))
        pcm *= 0.006 / rms
    if fade_in_ms:
        n = int(24000 * fade_in_ms / 1000)
        pcm[:n] *= np.linspace(0.15, 1.0, n)
    if fade_out_ms:
        n = int(24000 * fade_out_ms / 1000)
        pcm[-n:] *= np.linspace(1.0, 0.10, n)
    return pcm.astype(np.int16).tobytes()


def _dictate(audio):
    from voiceclip import config, tunnel
    config.load()
    # The autouse _hermetic_engine_detection fixture deletes VOICECLIP_ENGINE,
    # so force the cloud engine explicitly for this live test.
    config.ENGINE = "cloud"
    config.CLOUD_STREAMING = True
    assert tunnel.ensure(), "tunnel could not be established"
    from voiceclip.engine_cloud_stream import StreamingSession
    s = StreamingSession.try_start()
    assert s is not None
    chunk = 24000 * 2 // 10
    t0 = time.time()
    for n, i in enumerate(range(0, len(audio), chunk)):
        s.feed(audio[i:i + chunk])
        time.sleep(max(0, (n + 1) * 0.1 - (time.time() - t0)))
    rel = time.time()
    return s.finish(), time.time() - rel


@pytest.mark.parametrize("case,kw", [
    ("quiet_fade_in_release_at_end", {}),
    ("vad_killer_fading_tail", {"fade_out_ms": 900}),
])
def test_head_and_tail_survive(case, kw):
    text, dt = _dictate(_audio(**kw))
    assert text, f"{case}: no transcript"
    low = text.lower().lstrip()
    assert low.startswith("the belgium"), f"{case}: head clipped: {text!r}"
    assert "check that" in low, f"{case}: tail clipped: {text!r}"
    assert dt < 4.0, f"{case}: too slow: {dt:.2f}s"
