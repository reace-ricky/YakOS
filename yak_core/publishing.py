"""yak_core.publishing – Ricky lineup builder and Edge Share publisher.

This module provides two core functions for publishing YakOS analysis:

1. ``build_ricky_lineups`` – builds lineups for Ricky from an edge_df,
   incorporating calibration state and contest-type rules.

2. ``publish_edge_and_lineups`` – packages the current SlateState (metadata,
   edge sections, and selected lineups) into a payload dict ready for
   Edge Share persistence.

Usage
-----
    from yak_core.publishing import build_ricky_lineups, publish_edge_and_lineups

    lineups_df = build_ricky_lineups(edge_df, "GPP_20", calibration_state)
    payload = publish_edge_and_lineups(slate_state, lineups_df)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Contest-type → optimizer config overrides
_CONTEST_BUILD_PRESETS: Dict[str, Dict[str, Any]] = {
    "GPP_150": {
        "NUM_LINEUPS": 20,
        "MAX_EXPOSURE": 0.25,
        "MIN_SALARY_USED": 46000,
        "STACK_WEIGHT": 0.15,
        "VALUE_WEIGHT": 0.05,
    },
    "GPP_20": {
        "NUM_LINEUPS": 5,
        "MAX_EXPOSURE": 0.35,
        "MIN_SALARY_USED": 46000,
        "STACK_WEIGHT": 0.10,
        "VALUE_WEIGHT": 0.05,
    },
    "SE_3MAX": {
        "NUM_LINEUPS": 3,
        "MAX_EXPOSURE": 1.0,
        "MIN_SALARY_USED": 45000,
        "STACK_WEIGHT": 0.05,
        "VALUE_WEIGHT": 0.10,
    },
    "CASH": {
        "NUM_LINEUPS": 1,
        "MAX_EXPOSURE": 1.0,
        "MIN_SALARY_USED": 47000,
        "STACK_WEIGHT": 0.0,
        "VALUE_WEIGHT": 0.15,
    },
    "SHOWDOWN": {
        "NUM_LINEUPS": 3,
        "MAX_EXPOSURE": 0.60,
        "MIN_SALARY_USED": 46000,
        "STACK_WEIGHT": 0.0,
        "VALUE_WEIGHT": 0.05,
    },
}
_DEFAULT_BUILD_PRESET: Dict[str, Any] = {
    "NUM_LINEUPS": 5,
    "MAX_EXPOSURE": 0.35,
    "MIN_SALARY_USED": 46000,
    "STACK_WEIGHT": 0.05,
    "VALUE_WEIGHT": 0.05,
}

# Edge section classification thresholds
_SMASH_THRESH = 0.20     # smash_prob ≥ this → "Core"
_VALUE_THRESH = 0.10     # smash_prob ≥ this (and < Core) → "Value"
_LEVERAGE_THRESH = 1.3   # leverage ≥ this → "Leverage"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ricky_lineups(
    edge_df: pd.DataFrame,
    contest_type: str,
    calibration_state: Optional[Dict[str, Any]] = None,
    salary_cap: int = 50000,
    num_lineups: Optional[int] = None,
    lock: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build DK lineups for Ricky from the current edge_df.

    Uses ``yak_core.lineups.build_multiple_lineups_with_exposure`` with
    contest-type presets merged with any caller overrides.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Player edge metrics table (output of ``compute_edge_metrics``).
        Required columns: ``player_name``, ``salary``, ``proj``, ``pos``.
        Optional: ``ownership``, ``own_pct``, ``leverage``, ``floor``, ``ceil``.
    contest_type : str
        One of ``"GPP_150"``, ``"GPP_20"``, ``"SE_3MAX"``, ``"CASH"``.
    calibration_state : dict, optional
        Active calibration adjustments; currently stored in
        ``SlateState.calibration_state``.  Unused at build time (already
        applied upstream in edge_df), but recorded in lineups metadata.
    salary_cap : int
        DK salary cap (default 50 000).
    num_lineups : int, optional
        Override the contest preset's ``NUM_LINEUPS`` value.
    lock : list of str, optional
        Player names to lock in every lineup.
    exclude : list of str, optional
        Player names to exclude from all lineups.

    Returns
    -------
    pd.DataFrame
        Long-format lineup DataFrame (one player per row) with columns:
        ``lineup_index``, ``slot``, ``player_name``, ``team``, ``pos``,
        ``salary``, ``proj``.  Returns an empty DataFrame on failure.
    """
    from yak_core.lineups import build_multiple_lineups_with_exposure  # noqa: PLC0415
    from yak_core.lineups import build_showdown_lineups  # noqa: PLC0415
    from yak_core.config import SALARY_CAP  # noqa: PLC0415

    if edge_df is None or edge_df.empty:
        return pd.DataFrame()

    preset = dict(_CONTEST_BUILD_PRESETS.get(contest_type.upper(), _DEFAULT_BUILD_PRESET))
    preset["SALARY_CAP"] = salary_cap or SALARY_CAP
    preset["CONTEST_TYPE"] = contest_type

    if num_lineups is not None:
        preset["NUM_LINEUPS"] = int(num_lineups)
    if lock:
        preset["LOCK"] = lock
    if exclude:
        preset["EXCLUDE"] = exclude

    # Normalise own_pct → ownership for the optimizer
    pool = edge_df.copy()
    if "own_pct" in pool.columns and "ownership" not in pool.columns:
        pool["ownership"] = pool["own_pct"]

    # Ensure player_id exists (optimizer requires it)
    if "player_id" not in pool.columns:
        pool["player_id"] = pool.get("player_name", pd.Series(range(len(pool)), dtype=str))

    # Ensure proj > 0 to satisfy build_player_pool's filter
    proj_col = "proj"
    if proj_col not in pool.columns or (pd.to_numeric(pool[proj_col], errors="coerce").fillna(0) <= 0).all():
        return pd.DataFrame()

    try:
        # Route to Showdown Captain optimizer for Showdown contest type
        if contest_type.upper() == "SHOWDOWN":
            lineups_df = build_showdown_lineups(
                pool,
                num_lineups=preset.get("NUM_LINEUPS", 3),
                lock=lock,
                exclude=exclude,
            )
        else:
            lineups_df, _ = build_multiple_lineups_with_exposure(pool, preset)
    except Exception:
        return pd.DataFrame()

    return lineups_df


def publish_edge_and_lineups(
    slate_state: Any,
    lineups: pd.DataFrame,
) -> Dict[str, Any]:
    """Package the current SlateState and selected lineups into an Edge Share payload.

    The returned dict is the canonical payload written to Edge Share
    (e.g. persisted to ``data/published/`` or sent to a Streamlit data store).

    Parameters
    ----------
    slate_state : SlateState
        The current shared slate state object.
    lineups : pd.DataFrame
        Lineup DataFrame (output of ``build_ricky_lineups``).

    Returns
    -------
    dict
        Payload containing:
        - ``"slate_meta"`` : dict of slate metadata fields.
        - ``"edge_sections"`` : dict with ``"core"``, ``"value"``,
          ``"leverage"``, ``"notes"`` edge classification lists.
        - ``"lineups"`` : list of lineup dicts (one per lineup_index).
        - ``"published_at"`` : ISO timestamp.
    """
    ts = datetime.now(timezone.utc).isoformat()

    # Slate metadata
    slate_meta: Dict[str, Any] = {
        "sport": getattr(slate_state, "sport", ""),
        "site": getattr(slate_state, "site", ""),
        "slate_date": getattr(slate_state, "slate_date", ""),
        "draft_group_id": getattr(slate_state, "draft_group_id", None),
        "contest_name": getattr(slate_state, "contest_name", ""),
        "contest_type": getattr(slate_state, "contest_type", ""),
        "salary_cap": getattr(slate_state, "salary_cap", 50000),
        "active_layers": list(getattr(slate_state, "active_layers", ["Base"])),
    }

    # Edge sections from edge_df
    edge_sections: Dict[str, Any] = {
        "core": [],
        "value": [],
        "leverage": [],
        "notes": "",
    }

    edge_df = getattr(slate_state, "edge_df", None)
    if edge_df is not None and not edge_df.empty:
        smash = pd.to_numeric(edge_df.get("smash_prob", 0), errors="coerce").fillna(0)
        lev = pd.to_numeric(edge_df.get("leverage", 0), errors="coerce").fillna(0)

        core_mask = smash >= _SMASH_THRESH
        value_mask = (smash >= _VALUE_THRESH) & ~core_mask
        leverage_mask = lev >= _LEVERAGE_THRESH

        edge_sections["core"] = edge_df[core_mask]["player_name"].tolist()
        edge_sections["value"] = edge_df[value_mask]["player_name"].tolist()
        edge_sections["leverage"] = edge_df[leverage_mask]["player_name"].tolist()

    # Package lineups
    lineups_list: List[Dict[str, Any]] = []
    if lineups is not None and not lineups.empty and "lineup_index" in lineups.columns:
        for idx, grp in lineups.groupby("lineup_index"):
            lineups_list.append({
                "lineup_index": int(idx),
                "players": grp[["player_name", "slot", "salary", "proj"]].to_dict("records"),
                "total_salary": int(pd.to_numeric(grp.get("salary", 0), errors="coerce").fillna(0).sum()),
                "total_proj": float(pd.to_numeric(grp.get("proj", 0), errors="coerce").fillna(0).sum()),
            })

    return {
        "slate_meta": slate_meta,
        "edge_sections": edge_sections,
        "lineups": lineups_list,
        "published_at": ts,
    }
