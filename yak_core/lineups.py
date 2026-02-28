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
    ]
    if "ownership" in df.columns:
        base_cols.append("ownership")
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


# --------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------
def _eligible_slots(pos_str: str) -> Tuple[str, ...]:
    if not isinstance(pos_str, str):
        return ("UTIL",)
    parts = [p.strip().upper() for p in pos_str.split("/")]
    slots = set()
    for p in parts:
        if p in ["PG", "SG"]:
            slots.add(p)
            slots.add("G")
        elif p in ["SF", "PF"]:
            slots.add(p)
            slots.add("F")
        elif p == "C":
            slots.add("C")
    slots.add("UTIL")
    return tuple(sorted(slots))


def build_multiple_lineups_with_exposure(
    player_pool: pd.DataFrame,
    cfg: Dict[str, Any],
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_lineups = int(cfg.get("NUM_LINEUPS", 20))
    salary_cap = int(cfg.get("SALARY_CAP", 50000))
    max_exposure = float(cfg.get("MAX_EXPOSURE", 0.35))
    min_salary = int(cfg.get("MIN_SALARY_USED", 46000))
    own_weight = float(cfg.get("OWN_WEIGHT", 0.0))
    solver_time_limit = int(cfg.get("SOLVER_TIME_LIMIT", 30))
    max_appearances = max(1, int(num_lineups * max_exposure))
    pos_caps = cfg.get("POS_CAPS", {})
    lock_names = [n.strip() for n in cfg.get("LOCK", [])]

    players = player_pool.to_dict("records")
    n = len(players)
    if n < DK_LINEUP_SIZE:
        raise ValueError(
            f"Pool has only {n} players, need at least {DK_LINEUP_SIZE} to build a lineup"
        )

    for i, p in enumerate(players):
        p["_idx"] = i
        p["_slots"] = _eligible_slots(p.get("pos", ""))

    appearance_count = [0] * n
    all_lineups = []
    cancel_reasons: list[tuple[int, str]] = []

    for lu_num in range(num_lineups):
        prob = pulp.LpProblem(f"lineup_{lu_num}", pulp.LpMaximize)
        x = {}

        for i in range(n):
            for s in DK_POS_SLOTS:
                x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}_{lu_num}", cat="Binary")

        # Objective: blend projection + ownership leverage
        if own_weight > 0 and any("leverage" in p for p in players):
            prob += pulp.lpSum(
                (
                    (1 - own_weight) * players[i]["proj"]
                    + own_weight
                    * players[i]["proj"]
                    * players[i].get("leverage", 0.5)
                )
                * x[(i, s)]
                for i in range(n)
                for s in DK_POS_SLOTS
            )
        else:
            prob += pulp.lpSum(
                players[i]["proj"] * x[(i, s)]
                for i in range(n)
                for s in DK_POS_SLOTS
            )

        # Exactly one player per slot
        for s in DK_POS_SLOTS:
            prob += pulp.lpSum(x[(i, s)] for i in range(n)) == 1

        # Each player at most in one slot
        for i in range(n):
            prob += pulp.lpSum(x[(i, s)] for s in DK_POS_SLOTS) <= 1

        # Position eligibility
        for i in range(n):
            for s in DK_POS_SLOTS:
                if s not in players[i]["_slots"]:
                    prob += x[(i, s)] == 0

        # Salary band
        prob += pulp.lpSum(
            players[i]["salary"] * x[(i, s)]
            for i in range(n)
            for s in DK_POS_SLOTS
        ) <= salary_cap
        prob += pulp.lpSum(
            players[i]["salary"] * x[(i, s)]
            for i in range(n)
            for s in DK_POS_SLOTS
        ) >= min_salary

        # Per-position caps
        if pos_caps:
            for nat_pos, cap_val in pos_caps.items():
                eligible_players = [
                    j
                    for j in range(n)
                    if nat_pos
                    in players[j].get("pos", "").upper().split("/")
                ]
                if eligible_players:
                    prob += pulp.lpSum(
                        x[(j, s)]
                        for j in eligible_players
                        for s in DK_POS_SLOTS
                    ) <= cap_val

        # LOCK list
        if lock_names:
            for j in range(n):
                if players[j].get("player_name", "") in lock_names:
                    prob += pulp.lpSum(
                        x[(j, s)] for s in DK_POS_SLOTS
                    ) == 1

        # Exposure cap across lineups
        for i in range(n):
            if appearance_count[i] >= max_appearances:
                for s in DK_POS_SLOTS:
                    prob += x[(i, s)] == 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit))
        if prob.status != 1:
            reason = pulp.LpStatus.get(prob.status, f"status={prob.status}")
            cancel_reasons.append((lu_num, reason))
            print(f"[optimizer] Lineup {lu_num} cancelled: {reason}")
            if progress_callback is not None:
                progress_callback(lu_num + 1, num_lineups)
            continue

        for i in range(n):
            for s in DK_POS_SLOTS:
                if pulp.value(x[(i, s)]) and pulp.value(x[(i, s)]) > 0.5:
                    row = dict(players[i])
                    row["slot"] = s
                    row["lineup_index"] = lu_num
                    all_lineups.append(row)
                    appearance_count[i] += 1

        if progress_callback is not None:
            progress_callback(lu_num + 1, num_lineups)

    if not all_lineups:
        cancellation_summary = (
            "; ".join(f"lineup {lu}: {r}" for lu, r in cancel_reasons)
            if cancel_reasons
            else "unknown"
        )
        raise RuntimeError(
            f"Optimizer produced 0 feasible lineups out of {num_lineups} requested. "
            f"Pool had {n} players. Cancellation reasons: {cancellation_summary}"
        )

    if cancel_reasons:
        reason_counts: dict[str, int] = {}
        for _, r in cancel_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        summary = ", ".join(f"{r} ×{c}" for r, c in reason_counts.items())
        print(
            f"[optimizer] {len(cancel_reasons)} of {num_lineups} lineup(s) cancelled — {summary}"
        )

    lineups_df = pd.DataFrame(all_lineups)
    if "_idx" in lineups_df.columns:
        lineups_df.drop(columns=["_idx", "_slots"], inplace=True, errors="ignore")

    exposures_df = (
        lineups_df.groupby("player_id")["lineup_index"]
        .nunique()
        .reset_index()
        .rename(columns={"lineup_index": "num_lineups"})
    )
    exposures_df["exposure"] = exposures_df["num_lineups"] / float(num_lineups)

    return lineups_df, exposures_df


# --------------------------------------------------------------------
# Public engine entrypoint
# --------------------------------------------------------------------
def run_lineups_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single public engine entrypoint:
        load pool (historical parquet or live API) ->
        build player pool -> apply projections/ownership/leverage ->
        optimize -> validate -> return results.
    """
    merged = merge_config(cfg)

    # Default slate_date for live runs: today if not provided
    if merged.get("DATA_MODE") == "live" and "SLATE_DATE" not in merged:
        merged["SLATE_DATE"] = date.today().isoformat()

    slate_date = merged["SLATE_DATE"]
    num_lineups = int(merged["NUM_LINEUPS"])
    data_mode = merged.get("DATA_MODE", "historical")

    # 1) Load opt pool
    if data_mode == "live":
        opt_pool = fetch_live_opt_pool(slate_date, cfg=merged)
    else:
        try:
            opt_pool = _load_opt_pool_from_parquet(
                slate_date=slate_date,
                yakos_root=YAKOS_ROOT,
            )
        except FileNotFoundError:
            # Optional: block fallback if caller demands pure historical
            if (
                merged.get("DATA_MODE") == "historical"
                and not merged.get("ALLOW_LIVE_FALLBACK", True)
            ):
                raise

            # Fallback to live from Tank01
            data_mode = "live"
            merged["DATA_MODE"] = "live"
            opt_pool = fetch_live_opt_pool(slate_date, cfg=merged)

    # Empty pool guard
    if opt_pool.empty:
        return {
            "pool_df": opt_pool,
            "lineups": [],
            "slate_date": slate_date,
            "error": "No players available for " + slate_date,
        }

    # Live-mode: ensure we have projections before pool filter
    if data_mode == "live" and "salary" in opt_pool.columns:
        has_real_proj = (
            "proj" in opt_pool.columns
            and opt_pool["proj"].notna().any()
            and (opt_pool["proj"] > 0).any()
        )
        if not has_real_proj:
            opt_pool = apply_projections(opt_pool, merged)
            opt_pool = apply_ownership(opt_pool)
            opt_pool = compute_leverage(opt_pool)

    # 2) Build player pool
    pool_df = build_player_pool(opt_pool=opt_pool, cfg=merged)

    # Historical mode: separate actual_fp from forward-looking proj,
    # then apply projection/ownership models.
    if data_mode != "live":
        if "actual_fp" not in pool_df.columns and "proj" in pool_df.columns:
            pool_df["actual_fp"] = pool_df["proj"].copy()

        pool_df = apply_projections(pool_df, merged)
        pool_df = apply_ownership(pool_df)
        pool_df = compute_leverage(pool_df)

    # 3) Optimize
    lineups_df, exposures_df = build_multiple_lineups_with_exposure(
        player_pool=pool_df,
        cfg=merged,
    )

    # 4) Metadata + validation
    meta = {
        "slate_date": slate_date,
        "contest_type": merged.get("CONTEST_TYPE"),
        "num_lineups": num_lineups,
        "config_id": merged.get("CONFIG_ID"),
        "sport": merged.get("SPORT"),
        "site": merged.get("SITE"),
        "data_mode": merged.get("DATA_MODE"),
        "salary_cap": merged.get("SALARY_CAP"),
        "max_exposure": merged.get("MAX_EXPOSURE"),
        "logic_profile": merged.get("LOGIC_PROFILE"),
        "band": merged.get("BAND"),
        "min_salary_used": merged.get("MIN_SALARY_USED"),
        "yakos_root": YAKOS_ROOT,
        "num_players": int(len(pool_df)),
        "config": merged,
    }

    salary_cap = int(merged.get("SALARY_CAP", 50000))
    lineup_size = DK_LINEUP_SIZE
    validation_errors = []
    for lu_idx in lineups_df["lineup_index"].unique():
        lu = lineups_df[lineups_df["lineup_index"] == lu_idx]
        if len(lu) != lineup_size:
            validation_errors.append(
                "Lineup %d: expected %d players, got %d"
                % (lu_idx, lineup_size, len(lu))
            )
        lu_salary = lu["salary"].sum()
        if lu_salary > salary_cap:
            validation_errors.append(
                "Lineup %d: salary %d exceeds cap %d"
                % (lu_idx, lu_salary, salary_cap)
            )

    if validation_errors:
        raise ValueError("Validation failed: %d error(s)" % len(validation_errors))

    return {
        "lineups_df": lineups_df,
        "exposures_df": exposures_df,
        "pool_df": pool_df,
        "meta": meta,
    }
