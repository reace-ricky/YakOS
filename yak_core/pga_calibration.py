"""yak_core.pga_calibration -- PGA calibration helpers.

Fetch historical actuals from DataGolf, merge into a player pool, and
record projection errors via the sport-keyed calibration_feedback system.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import pandas as pd

from yak_core.config import YAKOS_ROOT

_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")


def get_pga_event_list(dg: Any) -> pd.DataFrame:
    """Get PGA events available for calibration.

    Wraps ``dg.get_dfs_event_list()`` and filters to PGA events with
    DraftKings salary data.  Returns sorted by date descending.

    Parameters
    ----------
    dg : DataGolfClient
        Authenticated DataGolf API client.

    Returns
    -------
    pd.DataFrame
        Columns include: event_id, event_name, calendar_year, date, tour,
        dk_salaries, dk_ownerships.
    """
    event_df = dg.get_dfs_event_list()
    if event_df.empty:
        return event_df

    mask = (event_df["tour"] == "pga") & (event_df["dk_salaries"] == "yes")
    filtered = event_df[mask].copy()
    if "date" in filtered.columns:
        filtered = filtered.sort_values("date", ascending=False)
    return filtered.reset_index(drop=True)


def fetch_pga_actuals(
    dg: Any,
    event_id: int,
    year: int,
    tour: str = "pga",
    site: str = "draftkings",
) -> pd.DataFrame:
    """Fetch actual DFS fantasy points for a completed PGA event.

    Parameters
    ----------
    dg : DataGolfClient
        Authenticated DataGolf API client.
    event_id : int
        DataGolf event ID.
    year : int
        Calendar year of the event.

    Returns
    -------
    pd.DataFrame
        Columns: player_name, dg_id, salary, actual_fp, ownership, fin_text, pos.
        Empty DataFrame if data unavailable (e.g. no premium subscription).
    """
    for attempt in range(3):
        try:
            dfs_df = dg.get_historical_dfs_points(
                event_id=event_id, year=year, tour=tour, site=site,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(30)
            else:
                raise
    else:
        return pd.DataFrame()

    if dfs_df.empty:
        return dfs_df

    # Map to YakOS column names
    rename = {}
    if "total_pts" in dfs_df.columns:
        rename["total_pts"] = "actual_fp"
    dfs_df = dfs_df.rename(columns=rename)

    # Ensure required columns
    if "actual_fp" not in dfs_df.columns:
        return pd.DataFrame()

    # All PGA players get pos="G"
    dfs_df["pos"] = "G"

    # Keep relevant columns
    keep = ["player_name", "dg_id", "salary", "actual_fp", "ownership", "fin_text", "pos"]
    keep = [c for c in keep if c in dfs_df.columns]
    return dfs_df[keep].copy()


def calibrate_pga_event(
    dg: Any,
    event_id: int,
    year: int,
    pool_df: Optional[pd.DataFrame] = None,
    slate_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Full calibration flow for one PGA event.

    Steps:
    1. If pool_df not provided, try to load from slate archive
    2. Fetch actuals via DataGolf historical DFS points
    3. Merge actual_fp into pool (match on dg_id or player_name)
    4. Call record_slate_errors(sport="PGA")
    5. Return calibration result summary

    Parameters
    ----------
    dg : DataGolfClient
        Authenticated DataGolf API client.
    event_id : int
        DataGolf event ID.
    year : int
        Calendar year of the event.
    pool_df : pd.DataFrame, optional
        Pre-existing pool with projections.  If None, builds from actuals
        (uses actual salary as proxy and DataGolf pre-tournament predictions
        for proj if available).
    slate_date : str, optional
        ISO date string for the event.  Used as the key in calibration history.

    Returns
    -------
    dict
        Calibration result from ``record_slate_errors``, or error dict.
    """
    from yak_core.calibration_feedback import record_slate_errors

    # Fetch actuals
    actuals = fetch_pga_actuals(dg, event_id, year)
    if actuals.empty:
        return {"error": "No actuals available (may require DataGolf premium)"}

    # Determine slate_date
    if not slate_date:
        slate_date = f"{year}-{event_id:03d}"  # Fallback identifier

    # If no pool provided, try to build one from archive or preds
    if pool_df is None:
        pool_df = _try_load_archived_pool(event_id, slate_date)

    if pool_df is not None and not pool_df.empty:
        # Merge actuals into existing pool
        merged = _merge_actuals_into_pool(pool_df, actuals)
    else:
        # No pre-existing pool — use actuals + pre-tournament predictions as proj
        merged = _build_pool_from_actuals_and_preds(dg, actuals, event_id, year)

    if merged.empty or "actual_fp" not in merged.columns:
        return {"error": "Could not build calibration pool"}

    # Ensure required columns for record_slate_errors
    if "pos" not in merged.columns:
        merged["pos"] = "G"
    if "proj" not in merged.columns:
        return {"error": "No projection data available for calibration"}

    # Filter to players with valid data
    merged = merged[
        (merged["actual_fp"] > 0) & merged["proj"].notna() & merged["salary"].notna()
    ].copy()

    if merged.empty:
        return {"error": "No valid proj/actual pairs after filtering"}

    # Record calibration errors
    result = record_slate_errors(slate_date, merged, sport="PGA")
    result["n_players_calibrated"] = len(merged)
    result["event_id"] = event_id
    result["year"] = year

    # Backfill actual_fp into any matching PGA archive parquets so the
    # Sim Sandbox can score PGA slates.
    _backfill_actuals_into_archive(slate_date, actuals)

    return result


def _backfill_actuals_into_archive(
    slate_date: str,
    actuals: pd.DataFrame,
) -> None:
    """Write actual_fp into PGA archive parquets matching *slate_date*.

    This enables the Sim Sandbox to score PGA slates against real results.
    Matches on player_name.  Only overwrites the parquet if new data was merged.
    """
    import glob as _glob

    pattern = os.path.join(_ARCHIVE_DIR, f"{slate_date}_pga_*.parquet")
    matches = _glob.glob(pattern)
    if not matches:
        # Also try the *_pga_gpp.parquet naming convention
        pattern = os.path.join(_ARCHIVE_DIR, f"*{slate_date}*pga*.parquet")
        matches = _glob.glob(pattern)

    if not actuals.empty and "actual_fp" in actuals.columns and "player_name" in actuals.columns:
        act_map = actuals.set_index("player_name")["actual_fp"].to_dict()
    else:
        return

    for path in matches:
        try:
            df = pd.read_parquet(path)
            if df.empty or "player_name" not in df.columns:
                continue
            if "actual_fp" in df.columns and df["actual_fp"].notna().any():
                continue  # Already has actuals, skip
            df["actual_fp"] = df["player_name"].map(act_map)
            if df["actual_fp"].notna().sum() > 0:
                df.to_parquet(path, index=False)
        except Exception:
            pass


def _try_load_archived_pool(event_id: int, slate_date: str) -> Optional[pd.DataFrame]:
    """Try to load a previously archived pool for this event."""
    import glob

    pattern = os.path.join(_ARCHIVE_DIR, f"pga_{slate_date}_{event_id}_*.parquet")
    matches = glob.glob(pattern)
    if not matches:
        # Try broader pattern
        pattern = os.path.join(_ARCHIVE_DIR, f"pga_*_{event_id}_*.parquet")
        matches = glob.glob(pattern)

    if matches:
        try:
            return pd.read_parquet(matches[0])
        except Exception:
            pass
    return None


def _merge_actuals_into_pool(
    pool_df: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Merge actual fantasy points into an existing pool."""
    pool = pool_df.copy()

    # Try merging on dg_id first
    if "dg_id" in pool.columns and "dg_id" in actuals.columns:
        act_map = actuals.set_index("dg_id")["actual_fp"].to_dict()
        pool["actual_fp"] = pool["dg_id"].map(act_map)
        matched = pool["actual_fp"].notna().sum()
        if matched > 0:
            return pool

    # Fall back to player_name
    if "player_name" in pool.columns and "player_name" in actuals.columns:
        act_map = actuals.set_index("player_name")["actual_fp"].to_dict()
        pool["actual_fp"] = pool["player_name"].map(act_map)

    return pool


def _build_pool_from_actuals_and_preds(
    dg: Any,
    actuals: pd.DataFrame,
    event_id: int,
    year: int,
) -> pd.DataFrame:
    """Build a calibration pool from actuals + pre-tournament predictions.

    When no archived pool exists, we use historical pre-tournament predictions
    as the 'proj' column and actuals for 'actual_fp'.
    """
    # Try to fetch historical pre-tournament predictions
    preds_df = pd.DataFrame()
    for attempt in range(3):
        try:
            preds_df = dg.get_historical_pre_tournament(
                event_id=event_id, year=year,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(30)
            else:
                break  # Non-retryable error

    if preds_df.empty:
        # No predictions available — can't calibrate without proj
        return pd.DataFrame()

    # Derive proj from pre-tournament probabilities (same model as pga_archiver)
    from yak_core.pga_archiver import _derive_projections

    for col in ("win", "top_5", "top_10", "top_20", "make_cut"):
        if col not in preds_df.columns:
            return pd.DataFrame()

    proj_df = _derive_projections(preds_df)

    # Merge with actuals on dg_id
    if "dg_id" in proj_df.columns and "dg_id" in actuals.columns:
        merged = proj_df.merge(
            actuals[["dg_id", "actual_fp", "salary"]].drop_duplicates("dg_id"),
            on="dg_id",
            how="inner",
        )
    elif "player_name" in proj_df.columns and "player_name" in actuals.columns:
        merged = proj_df.merge(
            actuals[["player_name", "actual_fp", "salary"]].drop_duplicates("player_name"),
            on="player_name",
            how="inner",
        )
    else:
        return pd.DataFrame()

    merged["pos"] = "G"
    return merged
