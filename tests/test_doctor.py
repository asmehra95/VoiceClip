"""Smoke tests for `voiceclip doctor`.

Runs the end-to-end health check against a temp config + temp DB and
captures stdout. Verifies: runs without raising, produces a known set of
section headers, returns an exit code (0 or 1).
"""

import os
import pytest
from voiceclip import config, history


@pytest.fixture
def _isolated(isolated_db):
    """Use shared isolated_db fixture (config + history DB pointer set,
    but no init). doctor.run() exercises the pre-init path separately."""
    yield


class TestDoctor:
    def test_runs_without_raising(self, _isolated, capsys):
        from voiceclip.doctor import run
        rc = run()
        assert rc in (0, 1)
        out = capsys.readouterr().out
        assert "VoiceClip doctor" in out

    def test_prints_expected_sections(self, _isolated, capsys):
        from voiceclip.doctor import run
        run()
        out = capsys.readouterr().out
        for section in ("System", "Permissions", "Storage",
                        "Optional providers", "Active config"):
            assert section in out, f"missing section: {section}"

    def test_reports_on_seeded_db(self, _isolated, capsys):
        # Seed a minimal DB so the 'Schema' section has something to check
        config.load()
        history.init()
        history.save("hi", "Hi.", 1.0)
        history.close()

        from voiceclip.doctor import run
        rc = run()
        out = capsys.readouterr().out
        assert "Schema" in out
        assert "entry count" in out
        # Exit code 0 if everything passed, 1 if accessibility isn't granted
        # in the test env — either is fine. We just want to verify it's not 2+.
        assert rc in (0, 1)
