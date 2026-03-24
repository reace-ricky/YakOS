"""Sidecar parquet storage for heuristic own_proj estimates.

Stores per-(sport, site, slate_id, contest_bucket) ownership projections as
a lightweight parquet archive so they survive across sessions.

On Streamlit Cloud (ephemeral filesystem) this module must be wired into
github_persistence._FEEDBACK_FILES to survive cold restarts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from yak_core.config import YAKOS_ROOT

# ---------------------------------------------------------------------------
# Storage path
# ---------------------------------------------------------------------------

_STORE_DIR = Path(YAKOS_ROOT) / "data" / "field_ownership"


def _store_path(sport: str, site: str, slate_id: str, contest_bucket: str) -> Path:
    """Return the parquet path for this (sport, site, slate_id, contest_bucket)."""
    sport = sport.lower()
    site = site.lower()
    contest_bucket = contest_bucket.lower().replace(" ", "_")
    # slate_id may contain slashes (dates); replace with underscores
    safe_slate = slate_id.replace("/", "-").replace("\\", "-")
    fname = f"{sport}_{site}_{safe_slate}_{contest_bucket}_own.parquet"
    return _STORE_DIR / fname


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_own_proj_to_archive(
    sport: str,
    site: str,
    slate_id: str,
    contest_bucket: str,
    ownership_df: pd.DataFrame,
) -> None:
    """Persist own_proj for a slate to a sidecar parquet file.

    Parameters
    ----------
    sport : str
        e.g. "nba" or "pga".
    site : str
        e.g. "dk".
    slate_id : str
        e.g. "2026-03-24" or a DraftKings draftGroupId.
    contest_bucket : str
        e.g. "gpp_main", "cash", "showdown_gpp".
    ownership_df : pd.DataFrame
        Must contain columns: player_id, own_proj.
    """
    if ownership_df.empty:
        return

    required = {"player_id", "own_proj"}
    missing = required - set(ownership_df.columns)
    if missing:
        raise ValueError(f"ownership_df missing required columns: {missing}")

    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = _store_path(sport, site, slate_id, contest_bucket)

    # Add key columns for identification
    df = ownership_df[["player_id", "own_proj"]].copy()
    df["sport"] = sport.lower()
    df["site"] = site.lower()
    df["slate_id"] = slate_id
    df["contest_bucket"] = contest_bucket.lower()

    df.to_parquet(path, index=False)
    print(f"[ownership_store] Saved {len(df)} own_proj rows → {path}")

    # Persist to GitHub on Streamlit Cloud so data survives cold restarts
    try:
        rel_path = str(path.relative_to(Path(YAKOS_ROOT)))
        from yak_core.github_persistence import sync_feedback_async
        sync_feedback_async(files=[rel_path], commit_message="Auto-sync field ownership")
    except (ValueError, Exception):
        pass  # Non-fatal — file is already on disk


def load_own_proj_for_slate(
    sport: str,
    site: str,
    slate_id: str,
    contest_bucket: str,
) -> pd.DataFrame:
    """Load previously saved own_proj for a slate.

    Parameters
    ----------
    sport, site, slate_id, contest_bucket : str
        Same key used in write_own_proj_to_archive.

    Returns
    -------
    pd.DataFrame with columns [player_id, own_proj] or empty DataFrame.
    """
    path = _store_path(sport, site, slate_id, contest_bucket)
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "own_proj"])
    try:
        df = pd.read_parquet(path)
        return df[["player_id", "own_proj"]].copy()
    except Exception as exc:
        print(f"[ownership_store] Failed to load {path}: {exc}")
        return pd.DataFrame(columns=["player_id", "own_proj"])


def attach_own_proj_to_pool(
    pool: pd.DataFrame,
    sport: str,
    site: str,
    slate_id: str,
    contest_bucket: str,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Join stored own_proj onto a player pool DataFrame.

    Looks up the sidecar parquet and left-joins on the player identifier
    column. If no sidecar exists, the pool is returned unchanged.

    Parameters
    ----------
    pool : pd.DataFrame
        The player pool.  Must have a player identifier column.
    sport, site, slate_id, contest_bucket : str
        Storage key.
    id_col : str, optional
        Override for the player identifier column name.  Auto-detected if None.

    Returns
    -------
    pd.DataFrame
        pool with an own_proj column added (NaN where no match).
    """
    own_df = load_own_proj_for_slate(sport, site, slate_id, contest_bucket)
    if own_df.empty:
        return pool

    # Determine player id column
    if id_col is None:
        for col in ("player_id", "dk_player_id", "player_name"):
            if col in pool.columns:
                id_col = col
                break
    if id_col is None:
        return pool

    # Remove existing own_proj to avoid duplicate columns
    merged = pool.copy()
    if "own_proj" in merged.columns:
        merged = merged.drop(columns=["own_proj"])

    merged = merged.merge(
        own_df.rename(columns={"player_id": id_col}),
        on=id_col,
        how="left",
    )
    return merged
