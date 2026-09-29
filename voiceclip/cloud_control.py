"""Cloud server lifecycle — start/stop/status for the EC2 instance.

Lets you park the GPU instance when you're done dictating (it bills by the
hour while running; stopped it costs almost nothing and keeps all data).

    voiceclip cloud status   → instance state
    voiceclip cloud stop     → stop it (data persists, billing stops)
    voiceclip cloud start    → start it back up

The same operations are exposed in the web viewer's Settings tab (via
/api/cloud/*), so both surfaces share the data-level functions below.

Uses the AWS CLI (same dependency as the SSM tunnel) so VoiceClip gains no
Python dependencies. Requires config:

    "cloud": {
        ...,
        "instance_id": "i-0123456789abcdef0",
        "region": "eu-central-1"
    }

Env overrides: VOICECLIP_CLOUD_INSTANCE_ID, VOICECLIP_CLOUD_REGION.
"""

import json
import re
import shutil
import subprocess

_TIMEOUT = 30


class CloudControlError(RuntimeError):
    """Configuration or AWS failure with a user-facing message."""


def _settings() -> tuple[str, str]:
    """Return (instance_id, region) or raise with a helpful message."""
    from voiceclip import config

    instance_id = (getattr(config, "CLOUD_INSTANCE_ID", "") or "").strip()
    region = (getattr(config, "CLOUD_REGION", "") or "").strip()

    if shutil.which("aws") is None:
        raise CloudControlError(
            "AWS CLI not found. Install it: brew install awscli")
    if not instance_id:
        raise CloudControlError(
            "cloud.instance_id is not set in ~/.voiceclip/config.json")
    if not region:
        raise CloudControlError('cloud.region is not set (e.g. "eu-central-1")')
    return instance_id, region


def _aws(args: list, region: str):
    """Run an AWS CLI command, returning parsed JSON. Raises on failure."""
    cmd = ["aws", *args, "--region", region, "--output", "json"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise CloudControlError(
            "AWS CLI timed out — check your network / credentials.") from None
    except FileNotFoundError:
        raise CloudControlError(
            "AWS CLI not found. Install it: brew install awscli") from None

    if proc.returncode != 0:
        # First line of stderr carries the AWS error; keep it short.
        err = (proc.stderr or "unknown error").strip().splitlines()[0]
        raise CloudControlError(err)
    if not proc.stdout.strip():
        return {}
    try:
        return json.loads(proc.stdout)
    except ValueError:
        raise CloudControlError("Unexpected non-JSON output from AWS CLI") from None


# ---------------------------------------------------------------------------
# Data-level operations (shared by CLI and viewer API)
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """Return {instance_id, region, state, instance_type}."""
    instance_id, region = _settings()
    inst = _aws(
        ["ec2", "describe-instances", "--instance-ids", instance_id,
         "--query", "Reservations[0].Instances[0]"],
        region,
    ) or {}
    return {
        "instance_id": instance_id,
        "region": region,
        "state": inst.get("State", {}).get("Name", "unknown"),
        "instance_type": inst.get("InstanceType", "?"),
    }


def stop_instance() -> dict:
    """Stop the instance. Returns {instance_id, region, state}."""
    instance_id, region = _settings()
    data = _aws(
        ["ec2", "stop-instances", "--instance-ids", instance_id], region,
    )
    state = (data or {}).get("StoppingInstances", [{}])[0] \
        .get("CurrentState", {}).get("Name", "stopping")
    return {"instance_id": instance_id, "region": region, "state": state}


def start_instance() -> dict:
    """Start the instance. Returns {instance_id, region, state}."""
    instance_id, region = _settings()
    data = _aws(
        ["ec2", "start-instances", "--instance-ids", instance_id], region,
    )
    state = (data or {}).get("StartingInstances", [{}])[0] \
        .get("CurrentState", {}).get("Name", "pending")
    return {"instance_id": instance_id, "region": region, "state": state}


# ---------------------------------------------------------------------------
# Cloud LLM selection (which model the gateway's "assistant" route serves)
# ---------------------------------------------------------------------------
# Curated for the default g6.2xlarge (one 23 GB L4 shared with Whisper and
# Kokoro — vLLM gets ~18 GB). Anything bigger fails to load on that box.
CLOUD_LLM_CHOICES = [
    {"id": "RedHatAI/Qwen3.5-9B-FP8-dynamic", "label": "Qwen 3.5 9B (FP8) — balanced, default"},
    {"id": "RedHatAI/Qwen3.5-9B-quantized.w4a16", "label": "Qwen 3.5 9B (4-bit) — lighter, more headroom"},
    {"id": "RedHatAI/Qwen3.5-4B-FP8-dynamic", "label": "Qwen 3.5 4B (FP8) — fastest replies"},
]
DEFAULT_CLOUD_LLM = CLOUD_LLM_CHOICES[0]["id"]

# A Hugging Face repo id and nothing else. This string is interpolated into
# a shell script run as root on the instance, so it must never carry shell
# metacharacters — the allowlist regex is the injection guard.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def valid_llm_model_id(model: str) -> bool:
    return bool(_HF_REPO_RE.match(model or ""))


def set_llm_model(model: str) -> dict:
    """Point the server's LLM at `model` (a Hugging Face repo id) and
    restart just the vLLM container. Persists in the instance's .env, so
    later update-server.sh runs keep the choice. Returns {command_id}.

    Loading a new model takes a few minutes (download + GPU load); poll
    llm_ready() to know when it answers again.
    """
    if not valid_llm_model_id(model):
        raise CloudControlError(
            "Model must be a Hugging Face repo id like 'Org/Model-Name'.")
    instance_id, region = _settings()
    script = (
        "set -e; cd /opt/transcription-service; "
        f"if grep -q '^LLM_MODEL=' .env; then sed -i 's|^LLM_MODEL=.*|LLM_MODEL={model}|' .env; "
        f"else echo 'LLM_MODEL={model}' >> .env; fi; "
        "docker compose --profile gpu up -d llm"
    )
    data = _aws(
        ["ssm", "send-command", "--instance-ids", instance_id,
         "--document-name", "AWS-RunShellScript",
         "--comment", "VoiceClip: switch cloud LLM",
         "--parameters", json.dumps({"commands": [script]})],
        region,
    ) or {}
    return {"command_id": (data.get("Command") or {}).get("CommandId")}


def llm_ready(timeout: float = 6.0) -> bool:
    """Does the gateway's assistant route answer a 1-token completion?"""
    from voiceclip import llm_provider
    try:
        llm_provider.cloud_request("/v1/chat/completions", {
            "model": "assistant",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }, timeout=timeout)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run(action: str) -> int:
    """Entry point for `voiceclip cloud <action>`. Returns an exit code."""
    try:
        if action == "status":
            s = get_status()
            icon = {"running": "🟢", "stopped": "⚪", "stopping": "🟡",
                    "pending": "🟡"}.get(s["state"], "❓")
            print(f"  {icon} {s['instance_id']} ({s['instance_type']}, "
                  f"{s['region']}): {s['state']}")
            if s["state"] == "running":
                print("     Costs accrue while running — "
                      "`voiceclip cloud stop` when done.")
            elif s["state"] == "stopped":
                print("     Start it with: voiceclip cloud start")
            return 0

        elif action == "stop":
            s = stop_instance()
            print(f"  ⚪ {s['instance_id']}: {s['state']}")
            print("     Billing stops; models, keys, and usage data persist.")
            print("     Bring it back with: voiceclip cloud start")
            return 0

        elif action == "start":
            s = start_instance()
            print(f"  🟡 {s['instance_id']}: {s['state']}")
            print("     Ready in ~2 minutes. Then open the tunnel and dictate:")
            print(f"     scripts/tunnel.sh {s['instance_id']} {s['region']}")
            return 0

        print(f"  ✗ Unknown action: {action}")
        return 1

    except CloudControlError as e:
        print(f"  ✗ {e}")
        return 1
