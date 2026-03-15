"""yak_core.github_persistence -- Push feedback data back to the GitHub repo.

After calibration_feedback or edge_feedback writes JSON to disk, this module
commits those files back to the repo so they survive Streamlit Cloud
redeploys / cold-starts.

Requires a GitHub personal-access token with repo write access, stored in
Streamlit secrets as ``GITHUB_TOKEN``, or as an environment variable.

Usage
-----
    from yak_core.github_persistence import sync_feedback_to_github
    sync_feedback_to_github()  # pushes all feedback JSON files
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from typing import Dict, List, Optional

import requests

from yak_core.config import YAKOS_ROOT

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────

_OWNER = "reace-ricky"
_REPO = "YakOS"
_BRANCH = "main"
_API = f"https://api.github.com/repos/{_OWNER}/{_REPO}"

# Files we want to persist (relative to repo root)
_FEEDBACK_FILES = [
    "data/calibration_feedback/nba/slate_errors.json",
    "data/calibration_feedback/nba/correction_factors.json",
    "data/calibration_feedback/pga/slate_errors.json",
    "data/calibration_feedback/pga/correction_factors.json",
    "data/edge_feedback/signal_history.json",
    "data/edge_feedback/signal_weights.json",
    "data/contest_results/history.json",
    "data/sim_sandbox/breakout_profile.json",
    "data/outcome_log/nba/outcomes.parquet",
    "data/outcome_log/pga/outcomes.parquet",
    "data/calibration/active_config.json",
    "data/calibration/config_history.json",
    "data/calibration/optimizer_overrides.json",
]

# Binary file extensions that need base64 encoding directly from bytes
_BINARY_EXTENSIONS = {".parquet", ".pkl", ".joblib", ".npy"}

# Serialize all pushes so concurrent calls don't race on the ref
_push_lock = threading.Lock()

# Debounce: batch rapid-fire sync requests into one commit
_DEBOUNCE_SECONDS = 3
_pending_lock = threading.Lock()
_pending_files: set = set()
_pending_timer: Optional[threading.Timer] = None


def _get_token() -> Optional[str]:
    """Resolve GitHub token from Streamlit secrets or env var."""
    # Try Streamlit secrets first
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN")
        if token:
            return str(token)
    except Exception:
        pass

    # Fallback to env var
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token

    # Last resort: local dev file
    for path in ["/tmp/gh_token.txt", os.path.expanduser("~/.gh_token")]:
        if os.path.isfile(path):
            return open(path).read().strip()

    return None


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def sync_feedback_to_github(
    files: Optional[List[str]] = None,
    commit_message: Optional[str] = None,
) -> Dict[str, str]:
    """Push feedback JSON files to the GitHub repo.

    Thread-safe with automatic retry on 422 (stale ref from concurrent
    pushes).

    Parameters
    ----------
    files : list[str], optional
        Specific repo-relative paths to push. Defaults to all feedback files.
    commit_message : str, optional
        Custom commit message.

    Returns
    -------
    dict
        {"status": "ok", "sha": "..."} on success,
        {"status": "skipped", "reason": "..."} or
        {"status": "error", "reason": "..."} on failure.
    """
    token = _get_token()
    if not token:
        return {"status": "skipped", "reason": "No GitHub token available"}

    targets = files or _FEEDBACK_FILES
    hdrs = _headers(token)

    # Collect files that actually exist on disk
    blobs_to_push: List[Dict] = []
    for rel_path in targets:
        abs_path = os.path.join(YAKOS_ROOT, rel_path)
        if not os.path.isfile(abs_path):
            continue
        # Detect binary files by extension
        _, ext = os.path.splitext(rel_path)
        if ext.lower() in _BINARY_EXTENSIONS:
            with open(abs_path, "rb") as f:
                raw_bytes = f.read()
            blobs_to_push.append({
                "path": rel_path,
                "b64": base64.b64encode(raw_bytes).decode(),
            })
        else:
            with open(abs_path, "r") as f:
                content = f.read()
            blobs_to_push.append({"path": rel_path, "content": content})

    if not blobs_to_push:
        return {"status": "skipped", "reason": "No feedback files to push"}

    msg = commit_message or f"Auto-sync feedback data ({len(blobs_to_push)} file(s))"

    # Serialize pushes to prevent 422 race conditions
    with _push_lock:
        for attempt in range(3):
            try:
                # 1. Fresh ref each attempt
                ref_resp = requests.get(
                    f"{_API}/git/refs/heads/{_BRANCH}", headers=hdrs, timeout=10
                )
                ref_resp.raise_for_status()
                current_sha = ref_resp.json()["object"]["sha"]

                # 2. Base tree
                commit_resp = requests.get(
                    f"{_API}/git/commits/{current_sha}", headers=hdrs, timeout=10
                )
                commit_resp.raise_for_status()
                base_tree = commit_resp.json()["tree"]["sha"]

                # 3. Create blobs
                tree_items = []
                for item in blobs_to_push:
                    # Binary files already have b64, text files need encoding
                    if "b64" in item:
                        blob_b64 = item["b64"]
                    else:
                        blob_b64 = base64.b64encode(
                            item["content"].encode()
                        ).decode()
                    blob_resp = requests.post(
                        f"{_API}/git/blobs",
                        headers=hdrs,
                        json={
                            "content": blob_b64,
                            "encoding": "base64",
                        },
                        timeout=10,
                    )
                    blob_resp.raise_for_status()
                    tree_items.append({
                        "path": item["path"],
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_resp.json()["sha"],
                    })

                # 4. Create tree
                tree_resp = requests.post(
                    f"{_API}/git/trees",
                    headers=hdrs,
                    json={"base_tree": base_tree, "tree": tree_items},
                    timeout=10,
                )
                tree_resp.raise_for_status()

                # 5. Create commit
                new_commit_resp = requests.post(
                    f"{_API}/git/commits",
                    headers=hdrs,
                    json={
                        "message": msg,
                        "tree": tree_resp.json()["sha"],
                        "parents": [current_sha],
                    },
                    timeout=10,
                )
                new_commit_resp.raise_for_status()
                new_sha = new_commit_resp.json()["sha"]

                # 6. Update ref
                update_resp = requests.patch(
                    f"{_API}/git/refs/heads/{_BRANCH}",
                    headers=hdrs,
                    json={"sha": new_sha},
                    timeout=10,
                )

                if update_resp.status_code == 422 and attempt < 2:
                    logger.info(f"Ref 422 on attempt {attempt + 1}, retrying...")
                    time.sleep(1)
                    continue

                update_resp.raise_for_status()
                logger.info(f"Feedback synced to GitHub: {new_sha[:12]}")
                return {"status": "ok", "sha": new_sha}

            except requests.HTTPError as e:
                if (
                    e.response is not None
                    and e.response.status_code == 422
                    and attempt < 2
                ):
                    time.sleep(1)
                    continue
                logger.warning(f"GitHub sync failed: {e}")
                return {"status": "error", "reason": str(e)}
            except Exception as e:
                logger.warning(f"GitHub sync failed: {e}")
                return {"status": "error", "reason": str(e)}

    return {"status": "error", "reason": "Max retries exceeded"}


def _flush_pending() -> None:
    """Called by the debounce timer — grabs pending files and pushes."""
    global _pending_files

    with _pending_lock:
        files = list(_pending_files)
        _pending_files = set()

    if files:
        sync_feedback_to_github(
            files=files,
            commit_message=f"Auto-sync feedback data ({len(files)} file(s))",
        )


def sync_feedback_async(
    files: Optional[List[str]] = None,
    commit_message: Optional[str] = None,
) -> None:
    """Debounced fire-and-forget background sync.

    Batches calls within a 3-second window into a single commit so that
    calibration + edge feedback (which fire back-to-back) don't create
    two separate commits or race each other.
    """
    global _pending_timer

    new_files = set(files or _FEEDBACK_FILES)

    with _pending_lock:
        _pending_files.update(new_files)

        # Reset the debounce timer
        if _pending_timer is not None:
            _pending_timer.cancel()

        _pending_timer = threading.Timer(_DEBOUNCE_SECONDS, _flush_pending)
        _pending_timer.daemon = True
        _pending_timer.start()


def delete_files_from_github(
    rel_paths: List[str],
    commit_message: Optional[str] = None,
) -> Dict[str, str]:
    """Delete files from the GitHub repo.

    Uses the Git tree API to create a commit that removes the specified
    files.  Thread-safe with automatic retry on 422.

    Parameters
    ----------
    rel_paths : list[str]
        Repo-relative paths to delete (e.g. ``data/published/nba/gpp_main_lineups.parquet``).
    commit_message : str, optional
        Custom commit message.

    Returns
    -------
    dict
        ``{"status": "ok", "sha": "..."}`` on success,
        ``{"status": "skipped", ...}`` or ``{"status": "error", ...}`` on failure.
    """
    token = _get_token()
    if not token:
        return {"status": "skipped", "reason": "No GitHub token available"}

    if not rel_paths:
        return {"status": "skipped", "reason": "No paths to delete"}

    hdrs = _headers(token)
    msg = commit_message or f"Delete {len(rel_paths)} published file(s)"

    with _push_lock:
        for attempt in range(3):
            try:
                # 1. Get current ref
                ref_resp = requests.get(
                    f"{_API}/git/refs/heads/{_BRANCH}", headers=hdrs, timeout=10
                )
                ref_resp.raise_for_status()
                current_sha = ref_resp.json()["object"]["sha"]

                # 2. Get base tree
                commit_resp = requests.get(
                    f"{_API}/git/commits/{current_sha}", headers=hdrs, timeout=10
                )
                commit_resp.raise_for_status()
                base_tree = commit_resp.json()["tree"]["sha"]

                # 3. Create tree with deletions (sha=None removes the entry)
                tree_items = []
                for rp in rel_paths:
                    tree_items.append({
                        "path": rp,
                        "mode": "100644",
                        "type": "blob",
                        "sha": None,
                    })

                tree_resp = requests.post(
                    f"{_API}/git/trees",
                    headers=hdrs,
                    json={"base_tree": base_tree, "tree": tree_items},
                    timeout=10,
                )
                tree_resp.raise_for_status()

                # 4. Create commit
                new_commit_resp = requests.post(
                    f"{_API}/git/commits",
                    headers=hdrs,
                    json={
                        "message": msg,
                        "tree": tree_resp.json()["sha"],
                        "parents": [current_sha],
                    },
                    timeout=10,
                )
                new_commit_resp.raise_for_status()
                new_sha = new_commit_resp.json()["sha"]

                # 5. Update ref
                update_resp = requests.patch(
                    f"{_API}/git/refs/heads/{_BRANCH}",
                    headers=hdrs,
                    json={"sha": new_sha},
                    timeout=10,
                )

                if update_resp.status_code == 422 and attempt < 2:
                    logger.info(f"Ref 422 on attempt {attempt + 1}, retrying...")
                    time.sleep(1)
                    continue

                update_resp.raise_for_status()
                logger.info(f"Deleted {len(rel_paths)} file(s) from GitHub: {new_sha[:12]}")
                return {"status": "ok", "sha": new_sha}

            except requests.HTTPError as e:
                if (
                    e.response is not None
                    and e.response.status_code == 422
                    and attempt < 2
                ):
                    time.sleep(1)
                    continue
                logger.warning(f"GitHub delete failed: {e}")
                return {"status": "error", "reason": str(e)}
            except Exception as e:
                logger.warning(f"GitHub delete failed: {e}")
                return {"status": "error", "reason": str(e)}

    return {"status": "error", "reason": "Max retries exceeded"}
