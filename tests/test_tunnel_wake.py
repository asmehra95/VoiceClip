"""Wake-on-use: tunnel.ensure() starts a parked instance before opening
the SSM session. cloud_control is faked; no AWS calls."""

import pytest

from voiceclip import tunnel


class FakeCloudControl:
    def __init__(self, states, start_raises=False):
        self.states = list(states)   # consumed one per get_status call
        self.start_calls = 0
        self.start_raises = start_raises

    def get_status(self):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {"state": state}

    def start_instance(self):
        self.start_calls += 1
        if self.start_raises:
            raise RuntimeError("boom")
        return {"state": "pending"}


@pytest.fixture
def fake_cc(monkeypatch):
    def install(states, **kw):
        cc = FakeCloudControl(states, **kw)
        monkeypatch.setattr("voiceclip.cloud_control.get_status", cc.get_status)
        monkeypatch.setattr("voiceclip.cloud_control.start_instance", cc.start_instance)
        return cc
    return install


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    monkeypatch.setattr(tunnel.time, "sleep", lambda s: None)


def test_running_instance_is_not_touched(fake_cc):
    cc = fake_cc(["running"])
    assert tunnel._wake_if_stopped("i-1", "eu-central-1") is False
    assert cc.start_calls == 0


def test_stopped_instance_is_started_and_waited_on(fake_cc):
    cc = fake_cc(["stopped", "pending", "running"])
    assert tunnel._wake_if_stopped("i-1", "eu-central-1") is True
    assert cc.start_calls == 1


def test_start_failure_is_swallowed(fake_cc):
    cc = fake_cc(["stopped"], start_raises=True)
    assert tunnel._wake_if_stopped("i-1", "eu-central-1") is False
    assert cc.start_calls == 1


def test_status_failure_is_swallowed(monkeypatch):
    def boom():
        raise RuntimeError("no credentials")
    monkeypatch.setattr("voiceclip.cloud_control.get_status", boom)
    assert tunnel._wake_if_stopped("i-1", "eu-central-1") is False


def test_stopping_waits_for_stop_then_starts(fake_cc):
    cc = fake_cc(["stopping", "stopped", "pending", "running"])
    assert tunnel._wake_if_stopped("i-1", "eu-central-1") is True
    assert cc.start_calls == 1
