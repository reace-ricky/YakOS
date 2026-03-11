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


# --------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------
def _eligible_slots(pos_str: str, pos_slots: tuple = None) -> Tuple[str, ...]:
    if pos_slots is None:
        pos_slots = tuple(DK_POS_SLOTS)

    # PGA / generic "G"-only roster: every player is eligible for every slot
    _unique_slots = set(pos_slots)
    if _unique_slots == {"G"} or _unique_slots == {"UTIL"}:
        return tuple(sorted(_unique_slots))

    if not isinstance(pos_str, str):
        return ("UTIL",) if "UTIL" in _unique_slots else tuple(sorted(_unique_slots))[:1]
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
        elif p == "G" and "G" in _unique_slots:
            slots.add("G")
    if "UTIL" in _unique_slots:
        slots.add("UTIL")
    # Only return slots that are actually in the roster
    slots = slots & _unique_slots
    if not slots:
        # Fallback: allow UTIL or first slot
        slots = {"UTIL"} if "UTIL" in _unique_slots else {pos_slots[0]}
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
    # Per-player exposure overrides: {player_name: max_exposure_float}
    player_max_exp = cfg.get("PLAYER_MAX_EXPOSURE", {})
    pos_caps = cfg.get("POS_CAPS", {})
    lock_names = [n.strip() for n in cfg.get("LOCK", [])]
    max_pair_appearances = int(cfg.get("MAX_PAIR_APPEARANCES", 0))
    # NOT_WITH: list of [player_a, player_b] pairs that must not appear together
    not_with_raw = cfg.get("NOT_WITH", [])
    not_with_pairs: list[tuple[str, str]] = [
        (str(pair[0]).strip(), str(pair[1]).strip())
        for pair in not_with_raw
        if isinstance(pair, (list, tuple)) and len(pair) >= 2
    ]
    # TIER_CONSTRAINTS: enforce min/max players from edge tiers per lineup.
    # Calibrated from 21-slate backtest: neutrals outperform 51.5%,
    # fades ($8K+ chalk) over-project by +2.54 FP.
    # Format: {"tier_player_names": {tier: [names]},
    #          "tier_min_players": {group: count},
    #          "tier_max_players": {tier: count}}
    tier_constraints = cfg.get("TIER_CONSTRAINTS", {})
    tier_player_names: Dict[str, list] = tier_constraints.get("tier_player_names", {})
    tier_min_players: Dict[str, int] = tier_constraints.get("tier_min_players", {})
    tier_max_players: Dict[str, int] = tier_constraints.get("tier_max_players", {})

    # ── Sport-aware roster shape ────────────────────────────────────
    # PGA uses 6 "G" slots instead of NBA's 8 positional slots.
    # Read from cfg (populated from contest preset), fall back to NBA defaults.
    _pos_slots = tuple(cfg.get("POS_SLOTS", cfg.get("pos_slots", DK_POS_SLOTS)))
    _lineup_size = int(cfg.get("LINEUP_SIZE", cfg.get("lineup_size", DK_LINEUP_SIZE)))

    # ── Contest-type detection ──────────────────────────────────────
    contest_type = cfg.get("CONTEST_TYPE", "gpp").lower()
    is_gpp = contest_type == "gpp"
    is_cash = contest_type == "cash"

    # GPP constraint knobs — overridable via cfg
    gpp_max_punt_players = int(cfg.get("GPP_MAX_PUNT_PLAYERS", 1))       # max players < $4 000
    gpp_min_mid_players  = int(cfg.get("GPP_MIN_MID_PLAYERS", 5))        # min players $4 000-$7 000
    gpp_own_cap          = float(cfg.get("GPP_OWN_CAP", 4.8))            # max total lineup ownership
    gpp_min_low_own      = int(cfg.get("GPP_MIN_LOW_OWN_PLAYERS", 3))    # min players below threshold
    gpp_low_own_thresh   = float(cfg.get("GPP_LOW_OWN_THRESHOLD", 0.45)) # ownership threshold
    gpp_force_game_stack = bool(cfg.get("GPP_FORCE_GAME_STACK", True))   # require 3+ from one game

    players = player_pool.to_dict("records")
    n = len(players)
    if n < _lineup_size:
        raise ValueError(
            f"Pool has only {n} players, need at least {_lineup_size} to build a lineup"
        )

    for i, p in enumerate(players):
        p["_idx"] = i
        p["_slots"] = _eligible_slots(p.get("pos", ""), _pos_slots)

    # ── Pre-compute per-player scores by contest type ──────────────
    if is_gpp:
        # GPP formula (v6 — backtested on 13 GPP slates 2026-02-02 → 2026-03-08):
        # Ceiling-dominant with light ownership penalty.
        # v5/H5_ForcedDiverse over-penalized ownership (-own*15) causing
        # infeasibility on small slates and systematic avoidance of smash
        # candidates who happened to be popular.
        # gpp_score = proj*0.35 + ceil*0.55 - own*5
        for p in players:
            proj  = float(p.get("proj", 0))
            ceil_ = float(p.get("ceil", p.get("proj", 0)))
            own   = float(p.get("ownership", p.get("own_proj", 0.5)))
            p["_gpp_score"] = (
                proj * 0.35
                + ceil_ * 0.55
                - own * 5.0
            )
    elif is_cash:
        # Cash formula (blueprint): cash_score = floor*w1 + proj*w2
        # Floor is king — consistency beats upside in 50/50 & double-ups.
        cash_floor_w = float(cfg.get("CASH_FLOOR_WEIGHT", 0.6))
        cash_proj_w  = float(cfg.get("CASH_PROJ_WEIGHT", 0.4))
        for p in players:
            proj_  = float(p.get("proj", 0))
            floor_ = float(p.get("floor", proj_ * 0.6))
            p["_cash_score"] = floor_ * cash_floor_w + proj_ * cash_proj_w

    appearance_count = [0] * n
    # pair_appearances[(i, j)] = number of lineups where both player i and j appear
    pair_appearances: dict[tuple[int, int], int] = {}
    all_lineups = []
    _prev_lineups: list[list[int]] = []  # player-index sets for uniqueness constraints
    cancel_reasons: list[tuple[int, str]] = []

    for lu_num in range(num_lineups):
        prob = pulp.LpProblem(f"lineup_{lu_num}", pulp.LpMaximize)
        x = {}

        for i in range(n):
            for s in _pos_slots:
                x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}_{lu_num}", cat="Binary")

        # ── Objective function ───────────────────────────────────────
        if is_gpp:
            # GPP: maximize H5_ForcedDiverse gpp_score
            prob += pulp.lpSum(
                players[i]["_gpp_score"] * x[(i, s)]
                for i in range(n)
                for s in _pos_slots
            )
        elif is_cash:
            # Cash: maximize floor-weighted score (consistency is king)
            prob += pulp.lpSum(
                players[i]["_cash_score"] * x[(i, s)]
                for i in range(n)
                for s in _pos_slots
            )
        else:
            # Fallback: blend projection + ownership leverage + edge scores.
            stack_weight = float(cfg.get("STACK_WEIGHT", 0.0))
            value_weight = float(cfg.get("VALUE_WEIGHT", 0.0))
            has_stack_scores = any("stack_score" in p for p in players)
            has_value_scores = any("value_score" in p for p in players)

            def _effective_proj(p):
                base = float(p.get("proj", 0))
                if own_weight > 0 and "leverage" in p:
                    base = (1 - own_weight) * base + own_weight * base * p.get("leverage", 0.5)
                if stack_weight > 0 and has_stack_scores:
                    ss = float(p.get("stack_score", 50.0) or 50.0) / 100.0
                    base = base * (1 + stack_weight * (ss - 0.5))
                if value_weight > 0 and has_value_scores:
                    vs = float(p.get("value_score", 50.0) or 50.0) / 100.0
                    base = base * (1 + value_weight * (vs - 0.5))
                return base

            prob += pulp.lpSum(
                _effective_proj(players[i]) * x[(i, s)]
                for i in range(n)
                for s in _pos_slots
            )

        # Exactly one player per slot
        for s in _pos_slots:
            prob += pulp.lpSum(x[(i, s)] for i in range(n)) == 1

        # Each player at most in one slot
        for i in range(n):
            prob += pulp.lpSum(x[(i, s)] for s in _pos_slots) <= 1

        # Position eligibility
        for i in range(n):
            for s in _pos_slots:
                if s not in players[i]["_slots"]:
                    prob += x[(i, s)] == 0

        # Salary band
        prob += pulp.lpSum(
            players[i]["salary"] * x[(i, s)]
            for i in range(n)
            for s in _pos_slots
        ) <= salary_cap
        prob += pulp.lpSum(
            players[i]["salary"] * x[(i, s)]
            for i in range(n)
            for s in _pos_slots
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
                        for s in _pos_slots
                    ) <= cap_val

        # LOCK list
        if lock_names:
            for j in range(n):
                if players[j].get("player_name", "") in lock_names:
                    prob += pulp.lpSum(
                        x[(j, s)] for s in _pos_slots
                    ) == 1

        # NOT_WITH: pairs of players that must never appear in the same lineup.
        # Enforced by constraining the sum of slot-assignment variables for both
        # players to at most 1 (so only one of the pair can be selected).
        if not_with_pairs:
            name_to_idx = {players[i]["player_name"]: i for i in range(n)}
            for pa, pb in not_with_pairs:
                ia = name_to_idx.get(pa)
                ib = name_to_idx.get(pb)
                if ia is not None and ib is not None:
                    prob += (
                        pulp.lpSum(x[(ia, s)] for s in _pos_slots)
                        + pulp.lpSum(x[(ib, s)] for s in _pos_slots)
                        <= 1
                    )

        # Exposure cap across lineups (supports per-player overrides)
        for i in range(n):
            pname = players[i].get("player_name", "")
            p_max_exp = player_max_exp.get(pname)
            p_max_app = max(1, int(num_lineups * p_max_exp)) if p_max_exp is not None else max_appearances
            if appearance_count[i] >= p_max_app:
                for s in _pos_slots:
                    prob += x[(i, s)] == 0

        # Pair-fade diversity: prevent overused player pairs from appearing together
        if max_pair_appearances > 0:
            for (pi, pj), count in pair_appearances.items():
                if count >= max_pair_appearances:
                    prob += (
                        pulp.lpSum(x[(pi, s)] for s in _pos_slots)
                        + pulp.lpSum(x[(pj, s)] for s in _pos_slots)
                        <= 1
                    )

        # Tier composition constraints: enforce min/max players from edge tiers.
        # This biases lineup construction toward undervalued tiers (core, neutral)
        # and limits expensive chalk (fade) without touching projections.
        if tier_player_names:
            name_to_idx_tier = {players[i].get("player_name", ""): i for i in range(n)}
            # Min constraints: e.g., "core_or_neutral" >= 2
            for group_key, min_count in tier_min_players.items():
                # Parse group: "core_or_neutral" → ["core", "neutral"]
                tier_list = [t.strip() for t in group_key.replace("_or_", ",").split(",")]
                eligible_idx = []
                for t in tier_list:
                    for pname in tier_player_names.get(t, []):
                        idx_val = name_to_idx_tier.get(pname)
                        if idx_val is not None:
                            eligible_idx.append(idx_val)
                if eligible_idx:
                    prob += pulp.lpSum(
                        x[(j, s)] for j in set(eligible_idx) for s in _pos_slots
                    ) >= min_count

            # Max constraints: e.g., "fade" <= 3
            for tier_key, max_count in tier_max_players.items():
                tier_idx = [
                    name_to_idx_tier.get(pname)
                    for pname in tier_player_names.get(tier_key, [])
                    if name_to_idx_tier.get(pname) is not None
                ]
                if tier_idx:
                    prob += pulp.lpSum(
                        x[(j, s)] for j in set(tier_idx) for s in _pos_slots
                    ) <= max_count

        # Uniqueness: each new lineup must differ by at least 3 players
        # from every previously built lineup.
        _MIN_DIFF = 3
        for prev_indices in _prev_lineups:
            # sum of slots assigned to prev players <= LINEUP_SIZE - _MIN_DIFF
            prob += pulp.lpSum(
                x[(pi, s)] for pi in prev_indices for s in _pos_slots
            ) <= _lineup_size - _MIN_DIFF

        # ── GPP-specific constraints (H5_ForcedDiverse) ──────────────
        # Backtested on 3/8 slate: moved from 0 lineups above 280 to
        # multiple 280+ lineups.  Best single lineup hit 302.5.
        if is_gpp:
            # 1. Max punt players (salary < $4000) — avoid dead-weight filler
            _punt_idx = [i for i in range(n) if players[i].get("salary", 0) < 4000]
            if _punt_idx:
                prob += pulp.lpSum(
                    x[(i, s)] for i in _punt_idx for s in _pos_slots
                ) <= gpp_max_punt_players

            # 2. Min mid-salary players ($4000-$7000) — the smash sweet spot
            _mid_idx = [i for i in range(n)
                        if 4000 <= players[i].get("salary", 0) <= 7000]
            if _mid_idx and gpp_min_mid_players > 0:
                prob += pulp.lpSum(
                    x[(i, s)] for i in _mid_idx for s in _pos_slots
                ) >= gpp_min_mid_players

            # 3. Total lineup ownership cap — force diversification
            prob += pulp.lpSum(
                float(players[i].get("ownership", players[i].get("own_proj", 0.5)))
                * x[(i, s)]
                for i in range(n) for s in _pos_slots
            ) <= gpp_own_cap

            # 4. Min low-owned players — ensure contrarian exposure
            if gpp_min_low_own > 0:
                _low_own_idx = [
                    i for i in range(n)
                    if float(players[i].get("ownership",
                             players[i].get("own_proj", 1.0))) < gpp_low_own_thresh
                ]
                if _low_own_idx:
                    prob += pulp.lpSum(
                        x[(i, s)] for i in _low_own_idx for s in _pos_slots
                    ) >= gpp_min_low_own

            # 5. Game stacking — correlated upside (3+ players from one game)
            #    Only enforced when opponent data is clean (no blank opponents).
            #    Archived slates sometimes have missing opponent for some teams,
            #    which creates degenerate game-key groups and infeasibility.
            if gpp_force_game_stack:
                _games: Dict[tuple, list] = {}
                _has_blank_opp = False
                for i in range(n):
                    opp = players[i].get("opponent", players[i].get("opp", ""))
                    if not opp or not str(opp).strip():
                        _has_blank_opp = True
                    gk = tuple(sorted([
                        players[i].get("team", ""),
                        str(opp or ""),
                    ]))
                    _games.setdefault(gk, []).append(i)
                # Skip game stack if any player has blank opponent data
                if not _has_blank_opp:
                    _gs_vars = {}
                    for gk, indices in _games.items():
                        if len(indices) >= 3:
                            gv = pulp.LpVariable(
                                f"gs_{gk[0]}_{gk[1]}_{lu_num}", cat="Binary"
                            )
                            _gs_vars[gk] = gv
                            prob += pulp.lpSum(
                                x[(i, s)] for i in indices for s in _pos_slots
                            ) >= 3 * gv
                    if _gs_vars:
                        prob += pulp.lpSum(_gs_vars.values()) >= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit))
        if prob.status != 1:
            reason = pulp.LpStatus.get(prob.status, f"status={prob.status}")
            # Fallback: retry without exposure caps to diagnose / recover from
            # cap-exhaustion infeasibility (tight MAX_EXPOSURE settings).
            capped_player_indices = [i for i in range(n) if appearance_count[i] >= max_appearances]
            if capped_player_indices:
                prob2 = pulp.LpProblem(f"lineup_{lu_num}_fallback", pulp.LpMaximize)
                x2 = {}
                for i in range(n):
                    for s in _pos_slots:
                        x2[(i, s)] = pulp.LpVariable(f"x2_{i}_{s}_{lu_num}", cat="Binary")
                if own_weight > 0 and any("leverage" in p for p in players):
                    prob2 += pulp.lpSum(
                        (
                            (1 - own_weight) * players[i]["proj"]
                            + own_weight * players[i]["proj"] * players[i].get("leverage", 0.5)
                        )
                        * x2[(i, s)]
                        for i in range(n)
                        for s in _pos_slots
                    )
                else:
                    prob2 += pulp.lpSum(
                        players[i]["proj"] * x2[(i, s)]
                        for i in range(n)
                        for s in _pos_slots
                    )
                for s in _pos_slots:
                    prob2 += pulp.lpSum(x2[(i, s)] for i in range(n)) == 1
                for i in range(n):
                    prob2 += pulp.lpSum(x2[(i, s)] for s in _pos_slots) <= 1
                for i in range(n):
                    for s in _pos_slots:
                        if s not in players[i]["_slots"]:
                            prob2 += x2[(i, s)] == 0
                salary_sum2 = pulp.lpSum(
                    players[i]["salary"] * x2[(i, s)]
                    for i in range(n)
                    for s in _pos_slots
                )
                prob2 += salary_sum2 <= salary_cap
                prob2 += salary_sum2 >= min_salary
                if pos_caps:
                    for nat_pos, cap_val in pos_caps.items():
                        eligible_players = [
                            j for j in range(n)
                            if nat_pos in players[j].get("pos", "").upper().split("/")
                        ]
                        if eligible_players:
                            prob2 += pulp.lpSum(
                                x2[(j, s)] for j in eligible_players for s in _pos_slots
                            ) <= cap_val
                if lock_names:
                    for j in range(n):
                        if players[j].get("player_name", "") in lock_names:
                            prob2 += pulp.lpSum(x2[(j, s)] for s in _pos_slots) == 1
                # Tier constraints in fallback too
                if tier_player_names:
                    _nit = {players[i].get("player_name", ""): i for i in range(n)}
                    for group_key, min_count in tier_min_players.items():
                        tl = [t.strip() for t in group_key.replace("_or_", ",").split(",")]
                        eidx = [_nit[p] for t in tl for p in tier_player_names.get(t, []) if p in _nit]
                        if eidx:
                            prob2 += pulp.lpSum(x2[(j, s)] for j in set(eidx) for s in _pos_slots) >= min_count
                    for tier_key, max_count in tier_max_players.items():
                        tidx = [_nit[p] for p in tier_player_names.get(tier_key, []) if p in _nit]
                        if tidx:
                            prob2 += pulp.lpSum(x2[(j, s)] for j in set(tidx) for s in _pos_slots) <= max_count
                prob2.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit))
                if prob2.status == 1:
                    print(
                        f"[optimizer] Lineup {lu_num}: exposure cap relaxed for "
                        f"{len(capped_player_indices)} player(s) — consider raising MAX_EXPOSURE"
                    )
                    selected_in_lu = []
                    for i in range(n):
                        for s in _pos_slots:
                            if pulp.value(x2[(i, s)]) and pulp.value(x2[(i, s)]) > 0.5:
                                row = dict(players[i])
                                row["slot"] = s
                                row["lineup_index"] = lu_num
                                all_lineups.append(row)
                                appearance_count[i] += 1
                                selected_in_lu.append(i)
                    if max_pair_appearances > 0:
                        for a in range(len(selected_in_lu)):
                            for b in range(a + 1, len(selected_in_lu)):
                                key = (selected_in_lu[a], selected_in_lu[b])
                                pair_appearances[key] = pair_appearances.get(key, 0) + 1
                    _prev_lineups.append(selected_in_lu)
                    if progress_callback is not None:
                        progress_callback(lu_num + 1, num_lineups)
                    continue
            cancel_reasons.append((lu_num, reason))
            print(f"[optimizer] Lineup {lu_num} cancelled: {reason}")
            if progress_callback is not None:
                progress_callback(lu_num + 1, num_lineups)
            continue

        selected_in_lu = []
        for i in range(n):
            for s in _pos_slots:
                if pulp.value(x[(i, s)]) and pulp.value(x[(i, s)]) > 0.5:
                    row = dict(players[i])
                    row["slot"] = s
                    row["lineup_index"] = lu_num
                    all_lineups.append(row)
                    appearance_count[i] += 1
                    selected_in_lu.append(i)

        if max_pair_appearances > 0:
            for a in range(len(selected_in_lu)):
                for b in range(a + 1, len(selected_in_lu)):
                    key = (selected_in_lu[a], selected_in_lu[b])
                    pair_appearances[key] = pair_appearances.get(key, 0) + 1

        _prev_lineups.append(selected_in_lu)

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
# DraftKings upload format export
# --------------------------------------------------------------------

def to_dk_upload_format(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format lineups to DraftKings bulk upload format.

    DraftKings expects one row per lineup in their bulk-entry upload CSV.
    Each roster slot occupies its own column with the player formatted as
    ``"Name (TEAM)"``.  The four header columns (Entry ID, Contest Name,
    Contest ID, Entry Fee) are left blank so the user can fill them in
    before uploading.

    Parameters
    ----------
    lineups_df : pd.DataFrame
        Long-format DataFrame produced by
        ``build_multiple_lineups_with_exposure``, containing at minimum
        ``lineup_index``, ``slot``, ``player_name``, and ``team`` columns.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with one row per lineup and columns:
        ``Entry ID``, ``Contest Name``, ``Contest ID``, ``Entry Fee``,
        ``PG``, ``SG``, ``SF``, ``PF``, ``C``, ``G``, ``F``, ``UTIL``.
    """
    meta_cols = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
    # Derive slot columns from the actual lineup data; fall back to NBA default
    if (
        not lineups_df.empty
        and "slot" in lineups_df.columns
        and "lineup_index" in lineups_df.columns
    ):
        _slot_list = list(lineups_df.groupby("lineup_index")["slot"].apply(list).iloc[0])
    else:
        _slot_list = list(DK_POS_SLOTS)
    all_cols = meta_cols + _slot_list

    if lineups_df.empty or "lineup_index" not in lineups_df.columns:
        return pd.DataFrame(columns=all_cols)

    rows = []
    for lu_id in sorted(lineups_df["lineup_index"].unique()):
        lu = lineups_df[lineups_df["lineup_index"] == lu_id]
        row: Dict[str, Any] = {c: "" for c in all_cols}
        for slot in _slot_list:
            slot_players = lu[lu["slot"] == slot]
            if not slot_players.empty:
                p = slot_players.iloc[0]
                name = str(p.get("player_name", ""))
                team = str(p.get("team", ""))
                row[slot] = f"{name} ({team})" if team and team != "nan" else name
        rows.append(row)

    return pd.DataFrame(rows, columns=all_cols)


def to_dk_showdown_upload_format(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format Showdown lineups to DraftKings Showdown bulk upload format.

    DraftKings Showdown format has one CPT column and five FLEX columns.
    Players are formatted as ``"Name (TEAM)"``.

    Parameters
    ----------
    lineups_df : pd.DataFrame
        Long-format DataFrame produced by ``build_showdown_lineups``,
        containing ``lineup_index``, ``slot``, ``player_name``, ``team`` columns.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns:
        ``Entry ID``, ``Contest Name``, ``Contest ID``, ``Entry Fee``,
        ``CPT``, ``FLEX``, ``FLEX``, ``FLEX``, ``FLEX``, ``FLEX``.
    """
    meta_cols = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
    # DK Showdown upload uses CPT + five FLEX columns
    slot_cols = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
    all_cols = meta_cols + slot_cols

    if lineups_df.empty or "lineup_index" not in lineups_df.columns:
        return pd.DataFrame(columns=all_cols)

    rows = []
    for lu_id in sorted(lineups_df["lineup_index"].unique()):
        lu = lineups_df[lineups_df["lineup_index"] == lu_id]
        row: Dict[str, Any] = {c: "" for c in meta_cols}
        # Collect CPT and FLEX players
        cpt_players = lu[lu["slot"] == "CPT"]
        flex_players = lu[lu["slot"] == "FLEX"]

        def _fmt(p_row) -> str:
            name = str(p_row.get("player_name", ""))
            team = str(p_row.get("team", ""))
            return f"{name} ({team})" if team and team != "nan" else name

        row["CPT"] = _fmt(cpt_players.iloc[0]) if not cpt_players.empty else ""
        for idx_flex in range(5):
            if idx_flex < len(flex_players):
                row[f"_flex_{idx_flex}"] = _fmt(flex_players.iloc[idx_flex])
            else:
                row[f"_flex_{idx_flex}"] = ""

        # Build final row with duplicate FLEX column names handled via list
        final_row = [
            row.get("Entry ID", ""),
            row.get("Contest Name", ""),
            row.get("Contest ID", ""),
            row.get("Entry Fee", ""),
            row.get("CPT", ""),
            row.get("_flex_0", ""),
            row.get("_flex_1", ""),
            row.get("_flex_2", ""),
            row.get("_flex_3", ""),
            row.get("_flex_4", ""),
        ]
        rows.append(final_row)

    return pd.DataFrame(rows, columns=all_cols)


# --------------------------------------------------------------------
# Showdown Captain optimizer
# --------------------------------------------------------------------

def build_showdown_lineups(
    player_pool: pd.DataFrame,
    cfg: Dict[str, Any],
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build DraftKings Showdown Captain lineups.

    DK Showdown uses a 6-player roster: 1 Captain (CPT) + 5 FLEX.
    The Captain slot costs 1.5× the player's listed salary and scores
    1.5× their actual fantasy points.  Any player can fill any slot.

    The optimizer models this by duplicating each player:
    - CPT version: salary × 1.5, proj × 1.5, eligible only for CPT slot.
    - FLEX version: normal salary + proj, eligible only for FLEX slots.

    A player can appear as CPT *or* FLEX but not both in the same lineup.

    Parameters
    ----------
    player_pool : pd.DataFrame
        Classic player pool (filtered to the two Showdown teams).
    cfg : dict
        Optimizer config.  Same keys as ``build_multiple_lineups_with_exposure``
        plus ``SALARY_CAP`` (default 50 000).

    Returns
    -------
    (lineups_df, exposures_df) in the same long format as the Classic optimizer,
    with ``slot`` values of ``"CPT"`` or ``"FLEX"``.
    """
    num_lineups = int(cfg.get("NUM_LINEUPS", 20))
    salary_cap = int(cfg.get("SALARY_CAP", 50000))
    min_salary = int(cfg.get("MIN_SALARY_USED", 45000))
    max_exposure = float(cfg.get("MAX_EXPOSURE", 0.35))
    solver_time_limit = int(cfg.get("SOLVER_TIME_LIMIT", 30))
    lock_names = [n.strip() for n in cfg.get("LOCK", [])]
    max_appearances = max(1, int(num_lineups * max_exposure))
    max_pair_appearances = int(cfg.get("MAX_PAIR_APPEARANCES", 0))

    base_players = player_pool.to_dict("records")
    m = len(base_players)
    if m < DK_SHOWDOWN_LINEUP_SIZE:
        raise ValueError(
            f"Showdown pool has only {m} players; need at least {DK_SHOWDOWN_LINEUP_SIZE}"
        )

    # Build two entries per player: CPT variant (index 0..m-1) and FLEX (m..2m-1)
    # CPT entry: salary × 1.5, proj × 1.5
    cpt_players = []
    flex_players = []
    for p in base_players:
        cpt_entry = dict(p)
        cpt_entry["salary"] = round(p["salary"] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER)
        cpt_entry["proj"] = p.get("proj", 0) * DK_SHOWDOWN_CAPTAIN_MULTIPLIER
        cpt_entry["_is_cpt"] = True
        cpt_players.append(cpt_entry)

        flex_entry = dict(p)
        flex_entry["_is_cpt"] = False
        flex_players.append(flex_entry)

    # i in [0, m) → CPT variant; i in [m, 2m) → FLEX variant
    players = cpt_players + flex_players
    n = len(players)  # 2 * m

    appearance_count = [0] * m  # track per original player (both variants share the slot)
    pair_appearances: dict[tuple[int, int], int] = {}
    _prev_lineups: list[list[int]] = []  # previous lineups as lists of original-player indices
    all_lineups = []
    cancel_reasons: list[tuple[int, str]] = []

    for lu_num in range(num_lineups):
        prob = pulp.LpProblem(f"showdown_{lu_num}", pulp.LpMaximize)

        # Binary vars: y[i] = 1 if player i (CPT or FLEX entry) is selected
        y = {i: pulp.LpVariable(f"y_{i}_{lu_num}", cat="Binary") for i in range(n)}

        # Objective: maximise total projected points with captain leverage
        # Captain selection is biased toward low-owned, high-ceiling players.
        # sd_score for CPT = proj*1.5 + ceil_bonus - own_penalty
        # sd_score for FLEX = proj (standard)
        sd_own_penalty = float(cfg.get("SD_CAPTAIN_OWN_PENALTY", 10.0))
        sd_ceil_bonus  = float(cfg.get("SD_CAPTAIN_CEIL_BONUS", 0.2))

        # Detect ownership scale: if max > 1.0, data is in percentage form
        _max_own = max(
            float(p.get("ownership", p.get("own_proj", 0))) for p in players
        ) if players else 1.0
        _own_divisor = 100.0 if _max_own > 1.0 else 1.0  # normalize to 0-1

        sd_obj_coeffs = []
        for i in range(n):
            p = players[i]
            base_proj = float(p.get("proj", 0))
            if i < m:  # CPT variant
                own = float(p.get("ownership", p.get("own_proj", 0.5))) / _own_divisor
                ceil_ = float(p.get("ceil", base_proj / DK_SHOWDOWN_CAPTAIN_MULTIPLIER))
                # Captain obj = proj (already 1.5x) + ceil_bonus*ceil - own_penalty*own
                sd_obj_coeffs.append(
                    base_proj + sd_ceil_bonus * ceil_ - sd_own_penalty * own
                )
            else:  # FLEX variant
                sd_obj_coeffs.append(base_proj)

        prob += pulp.lpSum(sd_obj_coeffs[i] * y[i] for i in range(n))

        # Exactly 1 CPT (from CPT variants, indices 0..m-1)
        prob += pulp.lpSum(y[i] for i in range(m)) == 1

        # Exactly 5 FLEX (from FLEX variants, indices m..2m-1)
        prob += pulp.lpSum(y[m + j] for j in range(m)) == 5

        # Each original player used at most once (CPT or FLEX, not both)
        for j in range(m):
            prob += y[j] + y[m + j] <= 1

        # Salary cap and floor
        prob += pulp.lpSum(players[i]["salary"] * y[i] for i in range(n)) <= salary_cap
        prob += pulp.lpSum(players[i]["salary"] * y[i] for i in range(n)) >= min_salary

        # LOCK: locked players must appear (either as CPT or FLEX)
        if lock_names:
            for j in range(m):
                if base_players[j].get("player_name", "") in lock_names:
                    prob += y[j] + y[m + j] == 1

        # Exposure cap per original player
        for j in range(m):
            if appearance_count[j] >= max_appearances:
                prob += y[j] == 0
                prob += y[m + j] == 0

        # Pair-fade diversity
        if max_pair_appearances > 0:
            for (pi, pj), count in pair_appearances.items():
                if count >= max_pair_appearances:
                    # selected_pi + selected_pj <= 1
                    prob += (
                        (y[pi] + y[m + pi]) + (y[pj] + y[m + pj]) <= 1
                    )

        # Lineup uniqueness: each new lineup must differ from every
        # previous lineup by at least 2 original players (out of 6).
        _SD_MIN_DIFF = 2
        for prev_indices in _prev_lineups:
            prob += pulp.lpSum(
                y[j] + y[m + j] for j in prev_indices
            ) <= DK_SHOWDOWN_LINEUP_SIZE - _SD_MIN_DIFF

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_time_limit))

        if prob.status != 1:
            reason = pulp.LpStatus.get(prob.status, f"status={prob.status}")
            cancel_reasons.append((lu_num, reason))
            print(f"[showdown] Lineup {lu_num} cancelled: {reason}")
            if progress_callback is not None:
                progress_callback(lu_num + 1, num_lineups)
            continue

        selected_original = []
        for j in range(m):
            if pulp.value(y[j]) and pulp.value(y[j]) > 0.5:
                row = dict(base_players[j])
                row["slot"] = "CPT"
                row["lineup_index"] = lu_num
                # Store the 1.5× salary and proj so display reflects captain pricing
                row["salary"] = cpt_players[j]["salary"]
                row["proj"] = cpt_players[j]["proj"]
                all_lineups.append(row)
                appearance_count[j] += 1
                selected_original.append(j)
            elif pulp.value(y[m + j]) and pulp.value(y[m + j]) > 0.5:
                row = dict(base_players[j])
                row["slot"] = "FLEX"
                row["lineup_index"] = lu_num
                all_lineups.append(row)
                appearance_count[j] += 1
                selected_original.append(j)

        if max_pair_appearances > 0:
            for a in range(len(selected_original)):
                for b in range(a + 1, len(selected_original)):
                    key = (selected_original[a], selected_original[b])
                    pair_appearances[key] = pair_appearances.get(key, 0) + 1

        _prev_lineups.append(selected_original)

        if progress_callback is not None:
            progress_callback(lu_num + 1, num_lineups)

    if not all_lineups:
        cancellation_summary = (
            "; ".join(f"lineup {lu}: {r}" for lu, r in cancel_reasons)
            if cancel_reasons
            else "unknown"
        )
        raise RuntimeError(
            f"Showdown optimizer produced 0 feasible lineups out of {num_lineups} requested. "
            f"Pool had {m} players. Cancellation reasons: {cancellation_summary}"
        )

    if cancel_reasons:
        reason_counts: dict[str, int] = {}
        for _, r in cancel_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        summary = ", ".join(f"{r} ×{c}" for r, c in reason_counts.items())
        print(f"[showdown] {len(cancel_reasons)} of {num_lineups} lineup(s) cancelled — {summary}")

    lineups_df = pd.DataFrame(all_lineups)

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
    lineup_size = int(merged.get("LINEUP_SIZE", merged.get("lineup_size", DK_LINEUP_SIZE)))
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
