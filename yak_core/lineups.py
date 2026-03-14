"""YakOS Core - data loading, player pool building, and PuLP optimizer."""
import os
from typing import Dict, Any, Tuple
from datetime import date

import numpy as np
import pandas as pd
import pulp

from .projections import apply_projections
from .ownership import apply_ownership, compute_leverage
from .live import fetch_live_opt_pool
from .config import (
    YAKOS_ROOT,
    DEFAULT_CONFIG,
    DK_LINEUP_SIZE,
    DK_POS_SLOTS,
    DK_PGA_LINEUP_SIZE,
    DK_PGA_POS_SLOTS,
    DK_SHOWDOWN_LINEUP_SIZE,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_CAPTAIN_MULTIPLIER,
    merge_config,
)

# ------------------------------------------------------------------
# Temporary compatibility shim: old code expects load_opt_pool_from_config
# ------------------------------------------------------------------

def load_opt_pool_from_config(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Deprecated compatibility wrapper.

    yak_core.__init__ still imports this name. Real callers should use
    run_lineups_from_config(cfg) instead; this exists only to satisfy
    that import.
    """
    raise RuntimeError(
        "load_opt_pool_from_config is deprecated; "
        "call run_lineups_from_config(cfg) instead."
    )


# --------------------------------------------------------------------
# Internal loader for historical parquet pools
# --------------------------------------------------------------------
def _load_opt_pool_from_parquet(slate_date: str,
                                yakos_root: str = YAKOS_ROOT) -> pd.DataFrame:
    """
    Load a historical opt_pool parquet for the given slate_date.

    Prefers the *_reproj.parquet version if present; otherwise falls
    back to the base parquet. Applies minimal cleanup (salary > 0,
    ensures player_id).
    """
    date_key = slate_date.replace("-", "")
    base_name = f"tank_opt_pool_{date_key}"
    reproj_path = os.path.join(yakos_root, f"{base_name}_reproj.parquet")
    base_path = os.path.join(yakos_root, f"{base_name}.parquet")

    if os.path.exists(reproj_path):
        path = reproj_path
    elif os.path.exists(base_path):
        path = base_path
    else:
        raise FileNotFoundError(
            f"No opt pool parquet for {slate_date}: {reproj_path} or {base_path}"
        )

    opt_pool = pd.read_parquet(path)

    if "salary" in opt_pool.columns:
        opt_pool = opt_pool[opt_pool["salary"].notna() & (opt_pool["salary"] > 0)]

    if "player_id" not in opt_pool.columns:
        if "player_name" in opt_pool.columns:
            opt_pool["player_id"] = opt_pool["player_name"]
        elif "player" in opt_pool.columns:
            opt_pool["player_id"] = opt_pool["player"]
        else:
            opt_pool["player_id"] = opt_pool.index.astype(str)

    return opt_pool


# --------------------------------------------------------------------
# Player pool construction
# --------------------------------------------------------------------
def build_player_pool(opt_pool: pd.DataFrame,
                      cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Take a raw opt_pool and enforce basic sanity filters + standard columns
    for the optimizer (player_id, player_name, team, opponent, pos, salary, proj).
    """
    df = opt_pool.copy()

    # Ensure player_id exists (pool from The Lab may only have player_name / dk_player_id)
    if "player_id" not in df.columns:
        if "player_name" in df.columns:
            df["player_id"] = df["player_name"]
        elif "dk_player_id" in df.columns:
            df["player_id"] = df["dk_player_id"]
        elif "player" in df.columns:
            df["player_id"] = df["player"]
        else:
            df["player_id"] = df.index.astype(str)

    proj_col = cfg.get("PROJ_COL", "proj")
    if proj_col not in df.columns:
        raise ValueError(
            f"Projection column '{proj_col}' not found in pool columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["player_id", "salary", proj_col])
    df = df[df["salary"] > 0]
    df = df[df[proj_col] > 0]

    # normalize to 'proj' for downstream optimizer
    df["proj"] = df[proj_col]

    if "player_name" not in df.columns:
        if "name" in df.columns:
            df["player_name"] = df["name"].astype(str)
        elif "player" in df.columns:
            df["player_name"] = df["player"].astype(str)
        else:
            df["player_name"] = df["player_id"].astype(str)

    if "opponent" not in df.columns and "opp" in df.columns:
        df["opponent"] = df["opp"]

    df["slate_date"] = cfg.get("SLATE_DATE", "")
    df["site"] = cfg.get("SITE", "DK")
    df["sport"] = cfg.get("SPORT", "NBA")
    df["contest_type"] = cfg.get("CONTEST_TYPE", "gpp")

    # --- Filter OUT / IR / WD players BEFORE column select ---
    # Must run here while 'status' column still exists in the DataFrame.
    _REMOVE_STATUSES = {"OUT", "IR", "SUSPENDED", "WD"}
    if "status" in df.columns:
        _before = len(df)
        df = df[
            ~df["status"].fillna("").str.strip().str.upper().isin(_REMOVE_STATUSES)
        ].reset_index(drop=True)
        _removed = _before - len(df)
        if _removed:
            print(f"[build_player_pool] Filtered {_removed} OUT/IR/WD player(s)")

    base_cols = [
        "player_id",
        "player_name",
        "team",
        "opponent",
        "pos",
        "salary",
        "proj",
        "slate_date",
        "site",
        "sport",
        "contest_type",
        "status",
    ]
    if "ownership" in df.columns:
        base_cols.append("ownership")
    if "own_proj" in df.columns:
        base_cols.append("own_proj")
    if "actual_fp" in df.columns:
        base_cols.append("actual_fp")
    if "leverage" in df.columns:
        base_cols.append("leverage")

    cols = [c for c in base_cols if c in df.columns]
    df = df[cols].reset_index(drop=True)

    # --- EXCLUDE: remove players by name ---
    exclude_list = [n.strip() for n in cfg.get("EXCLUDE", [])]
    if exclude_list:
        before = len(df)
        df = df[~df["player_name"].isin(exclude_list)].reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"[build_player_pool] Excluded {removed} player(s) by name: {exclude_list}")

    # --- BUMP: multiply projections by user factor ---
    bump_map = cfg.get("BUMP", {})
    if bump_map and "proj" in df.columns:
        for pname, mult in bump_map.items():
            mask = df["player_name"] == pname
            if mask.any():
                df.loc[mask, "proj"] = df.loc[mask, "proj"] * mult

    return df


def build_slate_pool(
    opt_pool: pd.DataFrame,
    slate_players: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Build a player pool restricted to players actually on the DK slate.

    This is the **correct** function to use for live / pre-slate sims.  It
    performs an **inner join** of *opt_pool* (which may contain projections for
    every rostered player) with *slate_players* (the official DK draftables for
    the selected draft group).  Players that are not on the slate are silently
    dropped; this prevents off-day players (e.g. Shai on a rest night) from
    appearing in projections or lineups.

    An assertion verifies the invariant: no ``player_id`` in the returned pool
    is absent from *slate_players*.

    The function intentionally does **not** filter by ``proj_minutes`` or any
    minutes threshold — that is the responsibility of
    :func:`yak_core.sims.compute_sim_eligible` (used only in post-slate
    analysis contexts).

    Parameters
    ----------
    opt_pool : pd.DataFrame
        Raw projection pool.  Must have a ``player_id`` column.
    slate_players : pd.DataFrame
        DK draftables for the target draft group (from
        ``fetch_dk_draftables``).  Must have a ``player_id`` column.
    cfg : dict
        Optimizer config dict (same shape as expected by :func:`build_player_pool`).

    Returns
    -------
    pd.DataFrame
        Filtered, normalised player pool ready for the optimizer.

    Raises
    ------
    ValueError
        If *opt_pool* or *slate_players* is missing a ``player_id`` column.
    AssertionError
        If any ``player_id`` in the returned pool is not in *slate_players*
        (should never happen — indicates a bug in the join logic).
    """
    if "player_id" not in opt_pool.columns:
        raise ValueError(
            "build_slate_pool: opt_pool must have a 'player_id' column."
        )
    if "player_id" not in slate_players.columns:
        raise ValueError(
            "build_slate_pool: slate_players must have a 'player_id' column."
        )

    slate_ids = set(slate_players["player_id"].astype(str).str.strip())

    # Inner join: only keep players whose player_id appears in slate_players
    merged = opt_pool[opt_pool["player_id"].astype(str).str.strip().isin(slate_ids)].copy()

    # Assertion: every player_id in the pool must be a slate player
    bad_ids = set(merged["player_id"].astype(str)) - slate_ids
    assert not bad_ids, (
        f"build_slate_pool: {len(bad_ids)} player_id(s) in pool not found in "
        f"slate_players after join: {bad_ids}"
    )

    # Delegate remaining normalisation to build_player_pool
    return build_player_pool(merged, cfg)
