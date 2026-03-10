"""File-based persistence for published lineup sets.

Published lineups are saved as parquet + JSON metadata in
``data/published/``.  This allows lineup data to survive Streamlit
session resets (page refresh, new tab, etc.) until explicitly cleared.

Usage
-----
- ``save_published_set(label, data)`` — called automatically by
  ``LineupSetState.publish()``.
- ``load_all_published()`` — called by ``get_lineup_state()`` when the
  session is fresh (no in-memory published sets).
- ``clear_published(label)`` — called by the UI "Clear Published" button.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISH_DIR = _REPO_ROOT / "data" / "published"


def _safe_label(contest_label: str) -> str:
    """Convert a contest label to a filesystem-safe slug."""
    return contest_label.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_published_set(contest_label: str, published_data: Dict[str, Any]) -> None:
    """Persist a single published lineup set to disk.

    Parameters
    ----------
    contest_label : str
        e.g. ``"GPP Main"``, ``"PGA GPP"``
    published_data : dict
        The dict stored in ``LineupSetState.published_sets[label]``.
        Expected keys: ``lineups_df``, ``config``, ``published_at``,
        ``boom_bust_df`` (optional), ``exposure_df`` (optional).
    """
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    slug = _safe_label(contest_label)

    # Lineups DataFrame (required)
    lineups_df = published_data.get("lineups_df")
    if lineups_df is not None and not lineups_df.empty:
        lineups_df.to_parquet(PUBLISH_DIR / f"{slug}_lineups.parquet", index=False)

    # Boom/bust DataFrame (optional)
    bb_df = published_data.get("boom_bust_df")
    if bb_df is not None and isinstance(bb_df, pd.DataFrame) and not bb_df.empty:
        bb_df.to_parquet(PUBLISH_DIR / f"{slug}_boom_bust.parquet", index=False)

    # Exposure DataFrame (optional)
    expo_df = published_data.get("exposure_df")
    if expo_df is not None and isinstance(expo_df, pd.DataFrame) and not expo_df.empty:
        expo_df.to_parquet(PUBLISH_DIR / f"{slug}_exposure.parquet", index=False)

    # Metadata JSON (config, timestamps, original label)
    meta = {
        "contest_label": contest_label,
        "published_at": published_data.get("published_at", ""),
        "config": published_data.get("config", {}),
    }
    (PUBLISH_DIR / f"{slug}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all_published() -> Dict[str, Dict[str, Any]]:
    """Load every published set from disk.

    Returns a dict matching the shape of
    ``LineupSetState.published_sets``:
    ``{contest_label: {"lineups_df": ..., "config": ..., ...}}``.
    """
    result: Dict[str, Dict[str, Any]] = {}
    if not PUBLISH_DIR.exists():
        return result

    for meta_path in sorted(PUBLISH_DIR.glob("*_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        contest_label = meta.get("contest_label", "")
        if not contest_label:
            continue

        slug = _safe_label(contest_label)

        # Lineups
        lu_path = PUBLISH_DIR / f"{slug}_lineups.parquet"
        if not lu_path.exists():
            continue
        try:
            lineups_df = pd.read_parquet(lu_path)
        except Exception:
            continue

        # Boom/bust (optional)
        bb_df = None
        bb_path = PUBLISH_DIR / f"{slug}_boom_bust.parquet"
        if bb_path.exists():
            try:
                bb_df = pd.read_parquet(bb_path)
            except Exception:
                pass

        # Exposure (optional)
        expo_df = None
        expo_path = PUBLISH_DIR / f"{slug}_exposure.parquet"
        if expo_path.exists():
            try:
                expo_df = pd.read_parquet(expo_path)
            except Exception:
                pass

        result[contest_label] = {
            "lineups_df": lineups_df,
            "config": meta.get("config", {}),
            "published_at": meta.get("published_at", ""),
            "boom_bust_df": bb_df,
            "exposure_df": expo_df,
        }

    return result


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def clear_published(contest_label: Optional[str] = None) -> None:
    """Delete persisted published data.

    Parameters
    ----------
    contest_label : str or None
        If provided, only delete files for that contest.
        If ``None``, delete *all* published files.
    """
    if not PUBLISH_DIR.exists():
        return

    if contest_label is not None:
        slug = _safe_label(contest_label)
        for f in PUBLISH_DIR.glob(f"{slug}_*"):
            f.unlink(missing_ok=True)
    else:
        for f in PUBLISH_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
