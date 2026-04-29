"""Integration tests for the viewer HTTP endpoints.

These spin up the real ThreadingHTTPServer against a temp DB and hit each
endpoint via urllib. LLM providers are disabled (provider='none') so no
network traffic; those endpoints are exercised for their disabled-guard
responses only.
"""

import urllib.request
from datetime import datetime

import pytest

from tests.conftest import http_get as _get
from tests.conftest import http_post as _post
from voiceclip import config, history


@pytest.fixture
def server(live_viewer):
    """Back-compat alias: tests here use `server` as the URL value."""
    return live_viewer


class TestGetEndpoints:
    def test_index_renders_html(self, server):
        body = urllib.request.urlopen(server).read().decode()
        assert "<title>VoiceClip journal</title>" in body

    def test_days_empty(self, server):
        r = _get(server + "/api/days")
        assert r == {"days": []}

    def test_day_empty(self, server):
        today = datetime.now().strftime("%Y-%m-%d")
        r = _get(f"{server}/api/day?date={today}")
        assert r["entries"] == []
        assert r["stats"]["transcriptions"] == 0

    def test_day_rejects_bad_date(self, server):
        try:
            urllib.request.urlopen(f"{server}/api/day?date=nope")
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_queue_listing_shape(self, server):
        r = _get(server + "/api/queue")
        assert "topics" in r
        assert "research_enabled" in r
        assert r["research_enabled"] is False

    def test_patterns_config_off_by_default(self, server):
        r = _get(server + "/api/patterns/config")
        assert r["enabled"] is False


class TestPostEndpoints:
    def test_update_roundtrip(self, server):
        i = history.save("raw", "Original text.", 1.0)
        code, r = _post(f"{server}/api/update", {"id": i, "text": "Edited text."})
        assert code == 200
        assert r["entry"]["text"] == "Edited text."

    def test_update_rejects_empty(self, server):
        i = history.save("raw", "Original.", 1.0)
        code, r = _post(f"{server}/api/update", {"id": i, "text": "   "})
        assert code == 400

    def test_delete_flow(self, server):
        i = history.save("raw", "Bye.", 1.0)
        code, r = _post(f"{server}/api/delete", {"id": i})
        assert code == 200
        assert r["entry"]["id"] == i

    def test_promote_flow(self, server):
        i = history.save("raw", "Promote me.", 1.0, kind="transcription")
        code, r = _post(f"{server}/api/promote", {"id": i})
        assert code == 200

    def test_research_create_and_dedup(self, server):
        code, r = _post(
            f"{server}/api/research/create",
            {"text": "CRDTs vs OT"},
        )
        assert code == 200
        first_id = r["id"]

        # Same topic via the patterns/queue path should dedup
        code, r2 = _post(
            f"{server}/api/patterns/queue",
            {"topic": "CRDTs vs OT", "reason": "from patterns"},
        )
        assert code == 200
        assert r2["duplicate"] is True
        assert r2["id"] == first_id

    def test_research_run_disabled(self, server):
        # Provider defaults to 'none'
        code, r = _post(f"{server}/api/research/run", {"id": 1})
        assert code == 400
        assert "disabled" in r["error"].lower()

    def test_patterns_run_disabled(self, server):
        code, r = _post(f"{server}/api/patterns/run", {})
        assert code == 400
        assert "off" in r["error"].lower() or "provider" in r["error"].lower()

    def test_summarize_run_disabled(self, server):
        code, r = _post(f"{server}/api/summarize", {})
        assert code == 400

    def test_bad_payload(self, server):
        code, r = _post(f"{server}/api/delete", {"id": "not-an-int"})
        assert code == 400



class TestBriefUpdateEndpoint:
    """POST /api/research/update_brief edits an existing brief's text."""

    def test_update_happy_path(self, server):
        tid = history.create_research_topic("CRDTs")
        bid = history.save_brief(tid, status="done", brief_text="before",
                                 provider="openai", model="gpt-4o-mini")
        code, r = _post(f"{server}/api/research/update_brief",
                        {"brief_id": bid, "text": "after"})
        assert code == 200
        assert r["brief"]["text"] == "after"

    def test_update_rejects_empty_text(self, server):
        tid = history.create_research_topic("X")
        bid = history.save_brief(tid, status="done", brief_text="whatever",
                                 provider="openai", model="gpt-4o-mini")
        code, r = _post(f"{server}/api/research/update_brief",
                        {"brief_id": bid, "text": "   "})
        assert code == 400

    def test_update_rejects_missing_brief(self, server):
        code, r = _post(f"{server}/api/research/update_brief",
                        {"brief_id": 99999, "text": "x"})
        assert code == 404

    def test_update_rejects_bad_id(self, server):
        code, r = _post(f"{server}/api/research/update_brief",
                        {"brief_id": "not-int", "text": "x"})
        assert code == 400


class TestArchiveEndpoints:
    """POST /api/research/archive and /api/research/unarchive.

    Also covers that GET /api/queue surfaces an `archived` array alongside
    the active topics so the UI only needs one round-trip on tab switch.
    """

    def test_archive_round_trip(self, server):
        tid = history.create_research_topic("Raft")
        code, r = _post(f"{server}/api/research/archive", {"id": tid})
        assert code == 200
        assert r["topic"]["id"] == tid
        assert r["topic"]["archived_at"]

        # Queue listing should exclude it from `topics` and include it in
        # `archived`
        data = _get(server + "/api/queue")
        assert all(t["id"] != tid for t in data["topics"])
        assert any(a["id"] == tid for a in data["archived"])

        code, r = _post(f"{server}/api/research/unarchive", {"id": tid})
        assert code == 200

        data = _get(server + "/api/queue")
        assert any(t["id"] == tid for t in data["topics"])
        assert all(a["id"] != tid for a in data["archived"])

    def test_archive_rejects_bad_id(self, server):
        code, r = _post(f"{server}/api/research/archive", {"id": "nope"})
        assert code == 400

    def test_archive_rejects_missing_topic(self, server):
        code, r = _post(f"{server}/api/research/archive", {"id": 99999})
        assert code == 404

    def test_archive_rejects_non_topic_entry(self, server):
        """A normal transcription isn't archivable — the endpoint should
        reject rather than silently mark something random as archived."""
        eid = history.save("raw", "Regular entry.", 1.0)
        code, r = _post(f"{server}/api/research/archive", {"id": eid})
        assert code == 404

    def test_unarchive_rejects_bad_id(self, server):
        code, r = _post(f"{server}/api/research/unarchive", {"id": "nope"})
        assert code == 400

    def test_unarchive_rejects_missing_topic(self, server):
        code, r = _post(f"{server}/api/research/unarchive", {"id": 99999})
        assert code == 404

    def test_queue_endpoint_shape_includes_archived(self, server):
        """Even when empty, the `archived` key must be present so the UI
        can rely on it without null-checks."""
        data = _get(server + "/api/queue")
        assert "archived" in data
        assert data["archived"] == []


class TestCustomVocabularySetting:
    """Settings endpoint accepts and validates the custom_vocabulary list."""

    def test_get_exposes_schema_and_value(self, server):
        data = _get(server + "/api/settings")
        assert "custom_vocabulary" in data["schema"]
        assert data["schema"]["custom_vocabulary"]["type"] == "text_list"
        assert data["values"]["custom_vocabulary"] == []

    def test_accepts_list(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": ["Kiro", "MeshClaw"]})
        assert code == 200
        # New value should round-trip through GET
        after = _get(server + "/api/settings")
        assert after["values"]["custom_vocabulary"] == ["Kiro", "MeshClaw"]

    def test_accepts_newline_string_from_textarea(self, server):
        """The UI sends a newline-joined string; server normalizes to list."""
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": "Kiro\nMeshClaw\n\n  AutoSDE  "})
        assert code == 200
        after = _get(server + "/api/settings")
        assert after["values"]["custom_vocabulary"] == ["Kiro", "MeshClaw", "AutoSDE"]

    def test_dedupes_case_insensitively(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": ["Kiro", "kiro", "KIRO"]})
        assert code == 200
        after = _get(server + "/api/settings")
        assert after["values"]["custom_vocabulary"] == ["Kiro"]

    def test_rejects_non_string_entries(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": ["ok", 42]})
        assert code == 400

    def test_rejects_overly_long_entry(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": ["x" * 200]})
        assert code == 400

    def test_rejects_too_many_entries(self, server):
        huge = [f"word{i}" for i in range(250)]
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": huge})
        assert code == 400

    def test_rejects_wrong_type(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"custom_vocabulary": 42})
        assert code == 400


class TestResearchLocalProvider:
    """Settings schema exposes `local` as a valid research provider choice,
    and /api/queue surfaces the local model when the provider is set to it."""

    def test_schema_includes_local_choice(self, server):
        data = _get(server + "/api/settings")
        choices = data["schema"]["research.provider"]["choices"]
        assert "local" in choices
        # local is NOT cloud — cloud_providers stays narrow
        assert "local" not in data["schema"]["research.provider"]["cloud_providers"]

    def test_schema_exposes_local_model_field(self, server):
        data = _get(server + "/api/settings")
        assert "research.local_model" in data["schema"]
        assert data["schema"]["research.local_model"]["type"] == "text"

    def test_settings_update_accepts_local_provider(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"research.provider": "local"})
        assert code == 200

    def test_settings_update_accepts_local_model(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"research.local_model": "mlx-community/Test-4bit"})
        assert code == 200
        after = _get(server + "/api/settings")
        assert after["values"]["research.local_model"] == "mlx-community/Test-4bit"


class TestModelsEndpoints:
    """GET /api/models lists cached HF models; POST /api/models/delete
    removes one. Both paths stub huggingface_hub.scan_cache_dir so tests
    don't depend on what's in the user's actual cache."""

    def _stub_scan_cache_dir(self, monkeypatch, repos):
        """Install a fake scan_cache_dir that returns the given repo stubs."""
        import sys
        import types

        class FakeRevision:
            def __init__(self, commit_hash, size, size_str):
                self.commit_hash = commit_hash
                self.size_on_disk = size
                self.size_on_disk_str = size_str

        class FakeRepo:
            def __init__(self, repo_id, size, size_str, last, last_str,
                         nb_files, repo_type="model"):
                self.repo_id = repo_id
                self.repo_type = repo_type
                self.size_on_disk = size
                self.size_on_disk_str = size_str
                self.last_accessed = last
                self.last_accessed_str = last_str
                self.nb_files = nb_files
                # Single revision per repo is the typical shape
                self.revisions = [
                    FakeRevision(f"hash_{repo_id}", size, size_str)
                ]

        fake_repos = [FakeRepo(**r) for r in repos]

        class FakeInfo:
            size_on_disk = sum(r["size"] for r in repos)
            def __init__(self): self.repos = fake_repos
            def delete_revisions(self, *hashes):
                # Return an object with .expected_freed_size and .execute()
                class Strategy:
                    expected_freed_size = sum(
                        r.size_on_disk for r in fake_repos
                        if any(f"hash_{r.repo_id}" in hashes for _ in [0])
                    )
                    def execute(self_): return None
                return Strategy()

        def fake_scan():
            return FakeInfo()

        # Both the viewer module's usage sites import lazily
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "scan_cache_dir", fake_scan,
                            raising=False)

    def test_list_models_happy_path(self, server, monkeypatch):
        self._stub_scan_cache_dir(monkeypatch, [
            {"repo_id": "mlx-community/Whisper-Small", "size": 500_000_000,
             "size_str": "500M", "last": 1000.0, "last_str": "today",
             "nb_files": 3},
            {"repo_id": "mlx-community/Qwen-4bit", "size": 4_000_000_000,
             "size_str": "4G", "last": 900.0, "last_str": "2 days ago",
             "nb_files": 5},
        ])
        data = _get(server + "/api/models")
        assert data["error"] is None
        ids = [m["repo_id"] for m in data["models"]]
        assert "mlx-community/Whisper-Small" in ids
        assert "mlx-community/Qwen-4bit" in ids
        # Sorted by size desc — Qwen (4G) should be first
        assert data["models"][0]["repo_id"] == "mlx-community/Qwen-4bit"

    def test_list_models_flags_in_use(self, server, monkeypatch):
        self._stub_scan_cache_dir(monkeypatch, [
            {"repo_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
             "size": 4_000_000_000, "size_str": "4G",
             "last": 1.0, "last_str": "now", "nb_files": 5},
        ])
        # Default config points summaries at Qwen2.5-7B-Instruct-4bit when
        # provider is local. Force that state.
        from voiceclip import config as cfg
        monkeypatch.setattr(cfg, "SUMMARIES_PROVIDER", "local")
        monkeypatch.setattr(cfg, "SUMMARIES_LOCAL_MODEL",
                            "mlx-community/Qwen2.5-7B-Instruct-4bit")

        data = _get(server + "/api/models")
        assert data["models"][0]["in_use_for"] == "summaries"

    def test_list_models_scan_failure_returns_error(self, server, monkeypatch):
        import huggingface_hub
        def boom():
            raise RuntimeError("fake scan failure")
        monkeypatch.setattr(huggingface_hub, "scan_cache_dir", boom,
                            raising=False)
        data = _get(server + "/api/models")
        assert data["error"] is not None
        assert "fake scan failure" in data["error"]
        assert data["models"] == []

    def test_delete_model_happy_path(self, server, monkeypatch):
        self._stub_scan_cache_dir(monkeypatch, [
            {"repo_id": "mlx-community/Something-4bit",
             "size": 2_000_000_000, "size_str": "2G",
             "last": 100.0, "last_str": "6 hours ago", "nb_files": 4},
        ])
        code, r = _post(f"{server}/api/models/delete",
                        {"repo_id": "mlx-community/Something-4bit"})
        assert code == 200
        assert r["repo_id"] == "mlx-community/Something-4bit"

    def test_delete_refuses_dictation_model(self, server, monkeypatch):
        """The currently-loaded Whisper model cannot be deleted — refusal
        must come with status 409 and an actionable message."""
        from voiceclip import config as cfg

        # get_model_repo reads MODEL + ENGLISH_ONLY. Default is the turbo
        # model; grab that and feed it to our stub.
        from voiceclip.config import get_model_repo
        active_repo, _ = get_model_repo()
        self._stub_scan_cache_dir(monkeypatch, [
            {"repo_id": active_repo, "size": 1_500_000_000,
             "size_str": "1.5G", "last": 0.0, "last_str": "just now",
             "nb_files": 10},
        ])
        code, r = _post(f"{server}/api/models/delete", {"repo_id": active_repo})
        assert code == 409
        assert "dictation model currently in use" in r["error"]

    def test_delete_rejects_missing_repo(self, server, monkeypatch):
        self._stub_scan_cache_dir(monkeypatch, [])
        code, r = _post(f"{server}/api/models/delete",
                        {"repo_id": "nonexistent/model"})
        assert code == 404

    def test_delete_rejects_empty_repo_id(self, server):
        code, r = _post(f"{server}/api/models/delete", {"repo_id": ""})
        assert code == 400

    def test_delete_rejects_wrong_type(self, server):
        code, r = _post(f"{server}/api/models/delete", {"repo_id": 42})
        assert code == 400


class TestRecommendedModelsEndpoint:
    """GET /api/models/recommended drives the Settings Quick-pick dropdown."""

    def test_returns_full_list_without_feature(self, server):
        data = _get(server + "/api/models/recommended")
        assert "models" in data
        assert isinstance(data["models"], list)
        # The shipped models.json always has at least one entry
        assert len(data["models"]) > 0
        m = data["models"][0]
        assert "id" in m and "label" in m and "backend" in m

    def test_filters_by_feature(self, server):
        data = _get(server + "/api/models/recommended?feature=research")
        assert all("research" in m["good_for"] for m in data["models"])

    def test_ignores_unknown_feature(self, server):
        """Bad feature values silently fall back to the full list."""
        data = _get(server + "/api/models/recommended?feature=bogus")
        # Falls through to no filter (not an error) — UI won't pass bogus
        # values, but if it did, better to show everything than nothing.
        assert isinstance(data["models"], list)


class TestTimelineEndpoint:
    """POST /api/timeline generates a chronological walkthrough for the
    day, stored separately from the summary. Mirrors /api/summarize."""

    def test_timeline_disabled_when_provider_none(self, server):
        code, r = _post(f"{server}/api/timeline", {})
        assert code == 400
        assert "disabled" in r["error"].lower()

    def test_timeline_rejects_bad_date(self, server):
        from voiceclip import config
        # Force provider != none so we hit the date validation path
        config.SUMMARIES_PROVIDER = "local"
        try:
            code, r = _post(f"{server}/api/timeline", {"date": "not-a-date"})
            assert code == 400
        finally:
            config.SUMMARIES_PROVIDER = "none"

    def test_day_payload_includes_timeline_key_when_empty(self, server):
        """The /api/day response shape must always include a `timeline`
        key (null when not cached) so the frontend can rely on it."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        data = _get(f"{server}/api/day?date={today}")
        assert "timeline" in data
        assert data["timeline"] is None  # fresh DB, never generated

    def test_day_payload_surfaces_cached_timeline(self, server):
        """After a timeline is saved, /api/day should return it."""
        from datetime import datetime

        from voiceclip import config
        today = datetime.now().strftime("%Y-%m-%d")

        # _day_payload gates timeline on SUMMARIES_PROVIDER != "none",
        # same as summary. Flip it for the assertion; restore after.
        saved = config.SUMMARIES_PROVIDER
        config.SUMMARIES_PROVIDER = "local"
        try:
            # Need an entry on the day so stats aren't empty
            history.save("raw", "A real entry.", 1.0, kind="transcription",
                         app_name="Slack")
            history.save_day_timeline(
                today, "Morning: you did a thing.",
                provider="openai", model="gpt-4o-mini", entry_count=1,
            )
            data = _get(f"{server}/api/day?date={today}")
            assert data["timeline"] is not None
            assert data["timeline"]["timeline"] == "Morning: you did a thing."
            assert data["timeline"]["model"] == "gpt-4o-mini"
        finally:
            config.SUMMARIES_PROVIDER = saved


class TestDictationModelSetting:
    """The Whisper model selection lives in the Settings tab's Dictation
    group. Picker dropdown, restart-required, and validation all wired
    against the existing settings schema infrastructure."""

    def test_schema_exposes_model_field(self, server):
        data = _get(server + "/api/settings")
        schema = data["schema"]
        assert "model" in schema
        assert schema["model"]["group"] == "Dictation"
        assert schema["model"]["type"] == "select"
        assert schema["model"]["restart_required"] is True

    def test_choices_match_valid_models(self, server):
        """The Settings dropdown must list every model that config.validate()
        would accept. Keeps the UI in sync with the runtime constraint."""
        from voiceclip import config as cfg
        data = _get(server + "/api/settings")
        choices = data["schema"]["model"]["choices"]
        # Every dropdown choice must be a real VALID_MODELS entry
        for c in choices:
            assert c in cfg.VALID_MODELS, f"dropdown has invalid model: {c}"
        # And no valid model missing from the UI
        for m in cfg.VALID_MODELS:
            assert m in choices, f"valid model missing from UI: {m}"

    def test_current_model_surfaced_as_value(self, server):
        from voiceclip import config as cfg
        data = _get(server + "/api/settings")
        assert data["values"]["model"] == cfg.MODEL

    def test_update_accepts_valid_model(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"model": "small"})
        assert code == 200
        # Restart-required flag should bubble up in the response so the
        # UI knows to show the banner.
        assert r.get("restart_required") is True

    def test_update_rejects_unknown_model(self, server):
        code, r = _post(f"{server}/api/settings/update",
                        {"model": "bogus-3000"})
        assert code == 400
        assert "must be one of" in r["error"]
