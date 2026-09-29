"""`voiceclip private-cloud-deploy` — CLI wrapper around the service
repo's deploy.sh. No real deploys: subprocess is faked."""

import argparse

import pytest

from voiceclip import __main__ as cli


def _args(region=None, az=None, repo=None):
    return argparse.Namespace(region=region, az=az, repo=repo)


@pytest.fixture
def fake_repo(tmp_path):
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "deploy.sh"
    script.write_text("#!/bin/bash\n")
    return tmp_path


@pytest.fixture
def run_spy(monkeypatch):
    calls = []

    class Result:
        returncode = 0

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    return calls


def test_parser_accepts_command():
    p = cli._build_parser()
    a = p.parse_args(["private-cloud-deploy", "eu-west-1", "--az", "1"])
    assert (a.command, a.region, a.az) == ("private-cloud-deploy", "eu-west-1", 1)


def test_missing_script_is_actionable(isolated_config, tmp_path, capsys):
    rc = cli._handle_private_cloud_deploy(_args(repo=str(tmp_path)))
    assert rc == 1
    err = capsys.readouterr().err
    assert "deploy.sh not found" in err
    assert "--repo" in err


def test_runs_script_from_repo(isolated_config, fake_repo, run_spy):
    rc = cli._handle_private_cloud_deploy(_args(repo=str(fake_repo)))
    assert rc == 0
    cmd, kw = run_spy[0]
    assert cmd == [str(fake_repo / "scripts" / "deploy.sh")]
    assert kw["cwd"] == str(fake_repo)


def test_region_and_az_forwarded(isolated_config, fake_repo, run_spy):
    cli._handle_private_cloud_deploy(_args(region="eu-west-1", az=0, repo=str(fake_repo)))
    cmd, _ = run_spy[0]
    assert cmd[1:] == ["eu-west-1", "--az", "0"]


def test_az_without_region_inserts_default(isolated_config, fake_repo, run_spy):
    """deploy.sh parses --az positionally after the region, so the region
    slot must be filled when only --az is given."""
    cli._handle_private_cloud_deploy(_args(az=1, repo=str(fake_repo)))
    cmd, _ = run_spy[0]
    assert cmd[1:] == ["eu-central-1", "--az", "1"]


def test_repo_from_config(isolated_config, fake_repo, run_spy, monkeypatch):
    import json

    from voiceclip import config
    with open(config.CONFIG_PATH, "w") as f:
        json.dump({"cloud": {"deploy_repo": str(fake_repo)}}, f)
    rc = cli._handle_private_cloud_deploy(_args())
    assert rc == 0
    assert run_spy[0][1]["cwd"] == str(fake_repo)
