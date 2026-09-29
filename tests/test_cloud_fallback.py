"""Cloud → local fallback in the transcription dispatcher.

Engines are fake modules; no models, no network. The audio "file" is a
real temp file so the delete-once contract can be asserted.
"""

import os
import tempfile
import types

import pytest

from voiceclip import config, transcriber


def _fake_engine(behaviour):
    """behaviour: callable(audio_path, repo) -> text, or raises."""
    mod = types.SimpleNamespace()
    mod.calls = []

    def transcribe(audio_path, repo):
        mod.calls.append((audio_path, repo))
        return behaviour(audio_path, repo)

    mod.transcribe = transcribe
    mod.keep_warm_ping = lambda repo: None
    mod.load = lambda repo: None
    return mod


@pytest.fixture
def wav():
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def cloud_setup(monkeypatch):
    """engine=cloud with injectable cloud + fallback engines."""
    transcriber._reset_engine()
    transcriber._reset_fallback()
    monkeypatch.setattr(config, "ENGINE", "cloud")
    monkeypatch.setattr(config, "CLOUD_MODEL", "whisper-large-v3")
    monkeypatch.setattr(config, "MODEL", "large-v3-turbo")
    monkeypatch.setattr(config, "get_model_repo",
                        lambda: ("mlx-community/whisper-large-v3-turbo", "turbo"))
    notes = []
    monkeypatch.setattr("voiceclip.macos.notify",
                        lambda title, msg: notes.append((title, msg)))

    def install(cloud, fallback=None, fallback_name="whisper"):
        monkeypatch.setattr(config, "CLOUD_FALLBACK_ENGINE",
                            fallback_name if fallback else "none")
        transcriber._engine = cloud
        transcriber._fallback_mod = fallback
        return notes

    yield install
    transcriber._reset_engine()
    transcriber._reset_fallback()


def test_cloud_ok_never_touches_fallback(cloud_setup, wav):
    cloud = _fake_engine(lambda p, r: "from the cloud")
    local = _fake_engine(lambda p, r: "from local")
    cloud_setup(cloud, local)
    assert transcriber.transcribe(wav) == "from the cloud"
    assert local.calls == []
    assert not os.path.exists(wav)


def test_cloud_failure_falls_back_and_notifies(cloud_setup, wav):
    def boom(p, r):
        raise ConnectionError("tunnel down")
    cloud = _fake_engine(boom)
    local = _fake_engine(lambda p, r: "from local")
    notes = cloud_setup(cloud, local)
    assert transcriber.transcribe(wav) == "from local"
    # local engine got the SAME file (not deleted after the cloud attempt)
    assert local.calls[0][0] == wav
    assert local.calls[0][1] == "mlx-community/whisper-large-v3-turbo"
    assert not os.path.exists(wav)  # deleted exactly once, at the end
    assert notes and "locally" in notes[0][1]


def test_notification_is_rate_limited(cloud_setup, wav, tmp_path):
    def boom(p, r):
        raise ConnectionError("down")
    cloud = _fake_engine(boom)
    local = _fake_engine(lambda p, r: "ok")
    notes = cloud_setup(cloud, local)
    transcriber.transcribe(wav)
    second = tmp_path / "b.wav"
    second.write_bytes(b"")
    transcriber.transcribe(str(second))
    assert len(notes) == 1


def test_no_fallback_configured_raises(cloud_setup, wav):
    def boom(p, r):
        raise ConnectionError("down")
    cloud_setup(_fake_engine(boom), fallback=None)
    with pytest.raises(transcriber.TranscriptionError):
        transcriber.transcribe(wav)
    assert not os.path.exists(wav)


def test_fallback_failure_surfaces(cloud_setup, wav):
    def boom(p, r):
        raise ConnectionError("down")
    def also_boom(p, r):
        raise RuntimeError("no mlx")
    cloud_setup(_fake_engine(boom), _fake_engine(also_boom))
    with pytest.raises(transcriber.TranscriptionError, match="no mlx"):
        transcriber.transcribe(wav)


def test_first_fallback_gets_load_headroom(cloud_setup):
    cloud_setup(_fake_engine(lambda p, r: ""), _fake_engine(lambda p, r: ""))
    first = transcriber._fallback_timeout()
    second = transcriber._fallback_timeout()
    assert first == transcriber._TIMEOUT["whisper"] + 120
    assert second == transcriber._TIMEOUT["whisper"]


def test_config_validates_fallback(isolated_config):
    import json
    with open(config.CONFIG_PATH, "w") as f:
        json.dump({"cloud": {"fallback_engine": "bogus"}}, f)
    config.load()
    assert config.CLOUD_FALLBACK_ENGINE == "none"
    with open(config.CONFIG_PATH, "w") as f:
        json.dump({"cloud": {"fallback_engine": "whisper"}}, f)
    config.load()
    assert config.CLOUD_FALLBACK_ENGINE == "whisper"
