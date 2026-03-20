"""Lineup archive helpers — append full lineup populations to a persistent store.

All lineups (the full 40+) are archived with a ``ricky_selected`` flag so
calibration runs can filter to Ricky-picked lineups or analyze the full pool.

Storage: ``data/lineup_archive/all_lineups.parquet``
(Parquet preferred over CSV for DFS DataFrames with many numeric columns.)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "lineup_archive"
ARCHIVE_PATH = _ARCHIVE_DIR / "all_lineups.parquet"


def append_to_archive(df: pd.DataFrame) -> Path:
    """Append rows to the lineup archive (parquet).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``lineup_index`` and ``ricky_selected`` columns.
        Additional metadata columns (``contest_type``, ``archived_at``, etc.)
        should be set by the caller before passing in.

    Returns
    -------
    Path
        The archive file path.
    """
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        existing = pd.read_parquet(ARCHIVE_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    combined.to_parquet(str(ARCHIVE_PATH), index=False)

    # Sync to GitHub so the archive survives Streamlit Cloud restarts
    try:
        from yak_core.github_persistence import sync_feedback_async

        rel = os.path.relpath(str(ARCHIVE_PATH), str(_ARCHIVE_DIR.parent.parent))
        sync_feedback_async(
            files=[rel],
            commit_message=f"Archive {len(df)} lineups ({df['ricky_selected'].sum()} Ricky picks)",
        )
    except Exception as exc:
        print(f"[archive] GitHub sync failed: {exc}")

    return ARCHIVE_PATH
