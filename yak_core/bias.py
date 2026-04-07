"""yak_core.bias -- Ricky's per-player projection and exposure overrides.

Stores manual adjustments (proj delta, max exposure cap) that persist
across sessions via data/ricky_bias.json and sync to GitHub.

Used by the optimizer (pre-prepare_pool injection) and The Board
(auto-fade writes, manual adjustment panel).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from .config import YAKOS_ROOT

BIAS_PATH = Path(YAKOS_ROOT) / "data" / "ricky_bias.json"


def load_bias() -> Dict[str, Dict[str, Any]]:
    """Load persisted bias overrides from disk.

    Returns dict of {player_name: {"proj_adj": float, "max_exposure": float|None}}.
    """
    if BIAS_PATH.exists():
        try:
            return json.loads(BIAS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_bias(bias: Dict[str, Dict[str, Any]]) -> None:
    """Persist bias overrides to disk and sync to GitHub.

    Uses an atomic write (temp file + rename) so a mid-write crash cannot
    leave the file in a partially-written or empty state.
    """
    BIAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bias, indent=2, default=str)

    # Write to a temp file in the same directory, then atomically rename.
    # os.replace() is POSIX-atomic when src and dst are on the same filesystem.
    fd, tmp_path = tempfile.mkstemp(dir=BIAS_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp_path, BIAS_PATH)
    except Exception:
        # Clean up orphaned temp file before re-raising so callers get the
        # real error rather than a follow-up PermissionError on the next run.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    try:
        from yak_core.github_persistence import sync_feedback_async
        sync_feedback_async(
            files=["data/ricky_bias.json"],
            commit_message="Auto-sync Ricky bias overrides",
        )
    except Exception:
        pass
