"""Model management endpoints — HuggingFace cache inspection and cleanup.

GET  /api/models              — list cached models with size + in-use markers
GET  /api/models/recommended  — curated list of known-good models for the picker
POST /api/models/delete       — delete a cached repo (refuses the dictation model)
"""

from __future__ import annotations

import logging

from voiceclip import config
from voiceclip.viewer.routes import register_get, register_post

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configured_model_ids() -> dict[str, str]:
    """Return {repo_id: feature-label} for every model currently wired up
    in config. Used to flag which cached repos are "in use" so the UI can
    warn before the user deletes one out from under an active feature.

    Feature label is what we show the user — "dictation", "summaries",
    "research", "patterns". Multiple features can share a model id; the
    first one wins (setdefault).
    """
    from voiceclip.config import get_model_repo
    configured: dict[str, str] = {}
    try:
        repo, _ = get_model_repo()
        configured.setdefault(repo, "dictation")
    except Exception:
        pass
    # Each of these only matters if the provider is local — cloud model
    # ids aren't on the HuggingFace cache.
    if config.SUMMARIES_PROVIDER == "local" and config.SUMMARIES_LOCAL_MODEL:
        configured.setdefault(config.SUMMARIES_LOCAL_MODEL, "summaries")
    if config.RESEARCH_PROVIDER == "local" and config.RESEARCH_LOCAL_MODEL:
        configured.setdefault(config.RESEARCH_LOCAL_MODEL, "research")
    if config.PATTERNS_PROVIDER == "local" and config.PATTERNS_LOCAL_MODEL:
        configured.setdefault(config.PATTERNS_LOCAL_MODEL, "patterns")
    return configured


def _list_cached_models() -> dict:
    """Scan the HuggingFace cache and return cached repos with metadata.

    Returns `{"models": [...], "total_size_gb": N, "error": str | None}`.
    Any scan failure is caught and surfaced as `error` so the Settings tab
    can render a friendly fallback instead of a 500.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return {
            "models": [],
            "total_size_gb": 0.0,
            "error": "huggingface_hub is not installed",
        }
    try:
        info = scan_cache_dir()
    except Exception as e:
        log.warning("scan_cache_dir failed: %s", e)
        return {
            "models": [],
            "total_size_gb": 0.0,
            "error": f"could not scan cache: {e}",
        }

    configured = _configured_model_ids()
    models = []
    for repo in info.repos:
        # Filter to model repos — skip datasets and spaces which wouldn't
        # be VoiceClip's.
        if repo.repo_type != "model":
            continue
        in_use_for = configured.get(repo.repo_id)
        models.append({
            "repo_id": repo.repo_id,
            "size_on_disk": repo.size_on_disk,
            "size_on_disk_str": repo.size_on_disk_str,
            "last_accessed": repo.last_accessed,
            "last_accessed_str": repo.last_accessed_str,
            "nb_files": repo.nb_files,
            "in_use_for": in_use_for,
            "revisions": [rev.commit_hash for rev in repo.revisions],
        })
    # Order by size descending so the biggest footguns sit at the top.
    models.sort(key=lambda m: m["size_on_disk"], reverse=True)
    return {
        "models": models,
        "total_size_gb": round(info.size_on_disk / (1024 ** 3), 2),
        "error": None,
    }


def _delete_cached_model(repo_id: str) -> dict:
    """Delete every revision of a cached model repo.

    Refuses to delete the Whisper model currently in use by the running
    daemon — deletion while loaded is a latent crash waiting to happen the
    next time the user tries to dictate. Deletion of an LLM model that
    belongs to a configured feature is allowed (feature will just
    redownload or fail on next use), but the UI shows a warning in the
    confirm modal.

    Returns `{"ok": True, "freed_bytes": N}` on success or
    `{"error": "..."}` on refusal / failure. Never raises.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return {"error": "huggingface_hub is not installed"}

    try:
        info = scan_cache_dir()
    except Exception as e:
        return {"error": f"could not scan cache: {e}"}

    # Guard against the dictation model — the running daemon has it memory-
    # mapped and deleting it can crash the recorder on the next start.
    from voiceclip.config import get_model_repo
    try:
        dictation_repo, _ = get_model_repo()
    except Exception:
        dictation_repo = None
    if repo_id == dictation_repo:
        return {
            "error": (
                f"'{repo_id}' is the dictation model currently in use. "
                "Switch to a different model in Settings (restart required) "
                "before deleting this one."
            )
        }

    target = None
    for r in info.repos:
        if r.repo_id == repo_id and r.repo_type == "model":
            target = r
            break
    if target is None:
        return {"error": f"'{repo_id}' is not in the cache"}

    commits = [rev.commit_hash for rev in target.revisions]
    if not commits:
        return {"error": "nothing to delete (no revisions)"}

    try:
        strategy = info.delete_revisions(*commits)
        freed = strategy.expected_freed_size
        strategy.execute()
        log.info("Deleted cached model %s (freed ~%s bytes)", repo_id, freed)
        return {"ok": True, "repo_id": repo_id, "freed_bytes": freed}
    except Exception as e:
        log.exception("Failed to delete cached model %s", repo_id)
        return {"error": f"delete failed: {e}"}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _get_models(req, query):
    req._json(_list_cached_models())


def _get_recommended(req, query):
    """Curated list of known-good local models for the picker dropdown.
    Optional `?feature=summaries` filter limits to models flagged as good
    for that feature."""
    from voiceclip.llm_provider import list_recommended_models
    feat = (query.get("feature") or [None])[0]
    if feat not in (None, "summaries", "research", "patterns"):
        feat = None
    req._json({"models": list_recommended_models(feature=feat)})


def _post_delete(req, payload):
    repo_id = payload.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        req._json({"error": "repo_id is required"}, status=400)
        return
    result = _delete_cached_model(repo_id.strip())
    if "error" in result:
        # 409 for in-use dictation model; 404 for missing; 400 otherwise.
        status = 400
        err = result["error"]
        if "not in the cache" in err:
            status = 404
        elif "dictation model currently in use" in err:
            status = 409
        req._json(result, status=status)
    else:
        req._json(result)


register_get("/api/models", _get_models)
register_get("/api/models/recommended", _get_recommended)
register_post("/api/models/delete", _post_delete)
