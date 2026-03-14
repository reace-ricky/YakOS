"""YakOS Core - data loading, player pool building, and PuLP optimizer."""
import os
from typing import Dict, Any, Tuple
from datetime import date

import numpy as np
import pandas as pd
import pulp

from yak_core.config import (
    DEFAULT_CONFIG,
    YAKOS_ROOT,
    SALARY_CAP,
    DK_LINEUP_SIZE,
    DK_POS_SLOTS,
    DK_SHOWDOWN_LINEUP_SIZE,
    DK_SHOWDOWN_SLOTS,
    DK_SHOWDOWN_CAPTAIN_MULTIPLIER,
)

# ── Silence PuLP's default stdout banner ─────────────────────────────────────
pulp.LpSolverDefault.msg = False  # type: ignore[attr-defined]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_player_pool(
    sport: str = "NBA",
    data_mode: str = "historical",
    slate_date: str | None = None,
    yakos_root: str | None = None,
) -> pd.DataFrame:
    """Load the player pool for a given slate.

    Parameters
    ----------
    sport : str
        One of "NBA" or "PGA" (case-insensitive).
    data_mode : str
        ``"historical"`` to read from the local parquet cache;
        ``"live"`` to fetch from Tank01 RapidAPI.
    slate_date : str or None
        ISO 8601 date string (``"YYYY-MM-DD"``) used in ``historical`` mode
        to select the correct parquet partition.  Defaults to today.
    yakos_root : str or None
        Override for the repository root directory.  Defaults to
        ``YAKOS_ROOT`` from ``yak_core.config``.

    Returns
    -------
    pd.DataFrame
        Player pool with at minimum: ``player_name``, ``team``,
        ``salary``, ``position``.
    """
    root = yakos_root or YAKOS_ROOT
    sport = sport.upper()

    if data_mode == "live":
        from yak_core.live_pool import fetch_live_pool  # type: ignore[import]
        return fetch_live_pool(sport=sport, root=root)

    # --- historical: read local parquet ---
    if slate_date is None:
        slate_date = str(date.today())

    # Try sport-specific subdirectory first, then legacy flat layout
    parquet_candidates = [
        os.path.join(root, "data", "pools", sport, f"{slate_date}.parquet"),
        os.path.join(root, "data", "pools", f"{slate_date}.parquet"),
        os.path.join(root, "data", "pools", "latest.parquet"),
    ]
    for path in parquet_candidates:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Normalise column names to snake_case
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            return df

    raise FileNotFoundError(
        f"No pool parquet found for {sport}/{slate_date}. "
        f"Tried: {parquet_candidates}"
    )


# =============================================================================
# SCORING / PROJECTION HELPERS
# =============================================================================

def _salary_implied_projection(
    salary: float,
    fp_per_k: float = 4.0,
) -> float:
    """Compute a naïve salary-implied projection: salary / 1000 * fp_per_k."""
    return (salary / 1_000.0) * fp_per_k


def _add_projections(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Ensure the pool has a ``proj`` column.

    Strategy selected by ``cfg['PROJ_SOURCE']``:
    - ``'parquet'``       : use existing ``proj`` column (no-op if present)
    - ``'salary_implied'``: fp_per_k * salary / 1000
    - ``'regression'``   : placeholder, falls back to salary_implied
    - ``'blend'``        : weighted average of parquet proj and salary_implied
    """
    source = cfg.get("PROJ_SOURCE", "salary_implied")
    fp_per_k = float(cfg.get("FP_PER_K", 4.0))
    noise_frac = float(cfg.get("PROJ_NOISE", 0.05))
    blend_w = float(cfg.get("PROJ_BLEND_WEIGHT", 0.7))  # weight on parquet in blend

    si_proj = df["salary"].apply(lambda s: _salary_implied_projection(s, fp_per_k))

    if source == "parquet":
        if "proj" not in df.columns:
            df["proj"] = si_proj
    elif source == "blend":
        if "proj" in df.columns:
            parquet_proj = df["proj"].fillna(si_proj)
            df["proj"] = blend_w * parquet_proj + (1.0 - blend_w) * si_proj
        else:
            df["proj"] = si_proj
    else:  # salary_implied (default) or regression fallback
        df["proj"] = si_proj

    # Add small noise for lineup differentiation
    if noise_frac > 0:
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, noise_frac * df["proj"].mean(), size=len(df))
        df["proj"] = (df["proj"] + noise).clip(lower=0)

    return df


def _add_ownership(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Ensure the pool has an ``own_pct`` column (0-1 scale)."""
    own_source = cfg.get("OWN_SOURCE", "auto")

    if "own_pct" in df.columns and own_source != "salary_rank":
        df["own_pct"] = pd.to_numeric(df["own_pct"], errors="coerce").fillna(0.0)
        # Convert from percentage to fraction if values look like percentages
        if df["own_pct"].max() > 1.5:
            df["own_pct"] = df["own_pct"] / 100.0
        return df

    # Generate salary-rank-based ownership proxy
    n = len(df)
    rank = df["salary"].rank(ascending=False, method="first")
    # Linear decay: rank 1 => ~0.45, rank n => ~0.05
    df["own_pct"] = 0.45 - 0.40 * (rank - 1) / max(n - 1, 1)
    df["own_pct"] = df["own_pct"].clip(0.02, 0.60)
    return df


def _add_scores(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Attach gpp_score, cash_score, value_score, stack_score."""
    own_weight   = float(cfg.get("OWN_WEIGHT",   0.0))
    stack_weight = float(cfg.get("STACK_WEIGHT", 0.0))
    value_weight = float(cfg.get("VALUE_WEIGHT", 0.0))

    # ── gpp_score = proj*GPP_PROJ_WEIGHT + ceil*GPP_CEIL_WEIGHT + own*GPP_OWN_WEIGHT
    # v7: ceil=2.0x proj, so ceiling term now adds real differentiation.
    # Shifted to 25/65 split to chase ceilings harder in GPP.
    gpp_proj_weight = float(cfg.get("GPP_PROJ_WEIGHT", 0.25))
    gpp_ceil_weight = float(cfg.get("GPP_CEIL_WEIGHT", 0.65))
    gpp_own_weight  = float(cfg.get("GPP_OWN_WEIGHT", -3.0))

    ceil_col = df["ceil"] if "ceil" in df.columns else df["proj"]
    own_col  = df["own_pct"]
    df["gpp_score"] = (
        df["proj"] * gpp_proj_weight
        + ceil_col * gpp_ceil_weight
        + own_col * gpp_own_weight
    )

    # ── cash_score = floor-weighted ─────────────────────────────────────────────
    floor_w = float(cfg.get("CASH_FLOOR_WEIGHT", 0.6))
    proj_w  = float(cfg.get("CASH_PROJ_WEIGHT",  0.4))
    if "floor" in df.columns:
        df["cash_score"] = floor_w * df["floor"] + proj_w * df["proj"]
    else:
        df["cash_score"] = df["proj"]

    # ── value_score ──────────────────────────────────────────────────────────────
    if value_weight > 0:
        df["value_score"] = df["proj"] / (df["salary"] / 1_000.0 + 1e-9)
    else:
        df["value_score"] = 0.0

    # ── stack_score (game-level correlation bonus) ───────────────────────────────
    if stack_weight > 0 and "game_id" in df.columns:
        game_proj = df.groupby("game_id")["proj"].transform("sum")
        df["stack_score"] = stack_weight * (game_proj / game_proj.max())
    else:
        df["stack_score"] = 0.0

    return df


def prepare_pool(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Clean, project, and score the player pool.

    Runs:
    1. Column normalisation (lowercase, rename aliases)
    2. Salary / projection cleaning
    3. Ownership proxy
    4. Score columns (gpp_score, cash_score, value_score, stack_score)

    Parameters
    ----------
    df : pd.DataFrame
        Raw player pool.
    cfg : Dict[str, Any]
        Merged config dict (from ``yak_core.config.merge_config``).

    Returns
    -------
    pd.DataFrame
        Enriched pool, index reset.
    """
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # ---- Rename common DK export columns ----
    rename_map = {
        "name + id": "player_name",
        "name+id": "player_name",
        "name": "player_name",
        "id": "player_id",
        "pos": "position",
        "position": "position",
        "game info": "game_info",
        "teamabbrev": "team",
        "avgpointspergame": "proj",
        "salary": "salary",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ---- Drop duplicate columns (can occur when API returns both 'pos' and 'position') ----
    if df.columns.duplicated().any():
        _dup_names = list(df.columns[df.columns.duplicated(keep=False)].unique())
        print(f"[prepare_pool] Dropping duplicate columns: {_dup_names}")
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # ---- Guard required columns ----
    for col in ("player_name", "salary"):
        if col not in df.columns:
            raise ValueError(f"Player pool missing required column: '{col}'")

    # ---- Filter OUT / IR / WD / Suspended players ----
    # Must run early while 'status' column still exists in the DataFrame.
    _REMOVE_STATUSES = {"OUT", "IR", "SUSPENDED", "WD"}
    if "status" in df.columns:
        _before = len(df)
        df = df[
            ~df["status"].fillna("").str.strip().str.upper().isin(_REMOVE_STATUSES)
        ].reset_index(drop=True)
        _removed = _before - len(df)
        if _removed:
            print(f"[prepare_pool] Filtered {_removed} OUT/IR/WD/Suspended player(s)")

    # ---- Salary cleaning ----
    df["salary"] = (
        df["salary"]
        .astype(str)
        .str.replace("[$,]", "", regex=True)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df = df[df["salary"] > 0].copy()

    # ---- Position normalisation ----
    if "position" not in df.columns:
        df["position"] = "UTIL"
    df["position"] = df["position"].astype(str).str.upper().str.strip()

    # ---- Team normalisation ----
    if "team" not in df.columns:
        df["team"] = "UNK"
    df["team"] = df["team"].astype(str).str.upper().str.strip()

    # ---- Projections ----
    df = _add_projections(df, cfg)

    # ---- Ownership proxy ----
    df = _add_ownership(df, cfg)

    # ---- Derived scores ----
    df = _add_scores(df, cfg)

    return df.reset_index(drop=True)


# =============================================================================
# PuLP OPTIMISER — CLASSIC
# =============================================================================

def _build_one_lineup(
    players: list,
    pos_slots: list,
    salary_cap: int,
    min_salary: int,
    max_appearances: dict,  # {player_idx: max remaining appearances}
    score_col: str = "gpp_score",
    stack_weight: float = 0.0,
    value_weight: float = 0.0,
    pos_caps: dict | None = None,
    solver_time_limit: int = 30,
    gpp_constraints: dict | None = None,
    tier_constraints: dict | None = None,
    not_with_pairs: list | None = None,
    forced_players: list | None = None,
    excluded_players: list | None = None,
) -> list | None:
    """Solve one LP and return a list of (player_dict, slot_name) tuples.

    Returns ``None`` if the LP is infeasible.
    """
    n = len(players)
    slots = pos_slots
    k = len(slots)

    # Slot eligibility: can player i fill slot s?
    def _eligible(player: dict, slot: str) -> bool:
        nat_pos = str(player.get("position", "")).upper()
        # UTIL and G/F flex slots accept any position
        if slot == "UTIL":
            return True
        if slot == "G":
            return nat_pos in ("PG", "SG", "G")
        if slot == "F":
            return nat_pos in ("SF", "PF", "F")
        # Exact match for PG, SG, SF, PF, C
        return nat_pos == slot

    prob = pulp.LpProblem("dfs_classic", pulp.LpMaximize)

    # Use slot INDEX (j) instead of slot NAME as dict key to avoid
    # key collision when slots has duplicates (e.g. PGA ["G"]*6).
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
         for i in range(n) for j in range(k)}

    # Objective: maximise total score
    score_key = score_col
    prob += pulp.lpSum(
        players[i].get(score_key, players[i].get("proj", 0)) * x[(i, j)]
        for i in range(n)
        for j in range(k)
    )

    # Each slot must be filled by exactly one player
    for j in range(k):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if _eligible(players[i], slots[j])) == 1

    # Each player may appear at most once across all slots
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(k)) <= 1

    # Salary cap
    prob += (
        pulp.lpSum(
            players[i]["salary"] * x[(i, j)]
            for i in range(n)
            for j in range(k)
        ) <= salary_cap
    )

    # Salary floor
    prob += (
        pulp.lpSum(
            players[i]["salary"] * x[(i, j)]
            for i in range(n)
            for j in range(k)
        ) >= min_salary
    )

    # Per-position caps
    if pos_caps:
        for nat_pos, cap_val in pos_caps.items():
            eligible_for_pos = [
                i for i in range(n)
                if str(players[i].get("position", "")).upper() == nat_pos
            ]
            if eligible_for_pos:
                prob += (
                    pulp.lpSum(x[(i, j)] for i in eligible_for_pos for j in range(k))
                    <= cap_val
                )

    # Exposure: each player may appear at most max_appearances[i] more times
    for i, max_app in max_appearances.items():
        prob += pulp.lpSum(x[(i, j)] for j in range(k)) <= max_app

    # Forced players (locks)
    if forced_players:
        for fp_idx in forced_players:
            prob += pulp.lpSum(x[(fp_idx, j)] for j in range(k)) == 1

    # Excluded players
    if excluded_players:
        for ep_idx in excluded_players:
            prob += pulp.lpSum(x[(ep_idx, j)] for j in range(k)) == 0

    # NOT_WITH pairs
    if not_with_pairs:
        for (idx_a, idx_b) in not_with_pairs:
            prob += (
                pulp.lpSum(x[(idx_a, j)] for j in range(k))
                + pulp.lpSum(x[(idx_b, j)] for j in range(k))
                <= 1
            )

    # GPP constraints
    gc = gpp_constraints or {}
    if gc:
        # Max punt players (salary < 4000)
        max_punt = gc.get("max_punt_players")
        if max_punt is not None:
            punt_idxs = [i for i in range(n) if players[i]["salary"] < 4000]
            if punt_idxs:
                prob += (
                    pulp.lpSum(x[(i, j)] for i in punt_idxs for j in range(k))
                    <= max_punt
                )

        # Min mid-salary players (4000-7000)
        min_mid = gc.get("min_mid_players")
        if min_mid is not None:
            mid_idxs = [i for i in range(n) if 4000 <= players[i]["salary"] <= 7000]
            if mid_idxs:
                prob += (
                    pulp.lpSum(x[(i, j)] for i in mid_idxs for j in range(k))
                    >= min_mid
                )

        # Max total ownership cap
        own_cap = gc.get("own_cap")
        if own_cap is not None and own_cap > 0:
            own_vals = [float(p.get("own_pct", 0)) for p in players]
            if max(own_vals) > 0:
                prob += (
                    pulp.lpSum(
                        own_vals[i] * x[(i, j)]
                        for i in range(n)
                        for j in range(k)
                    ) <= own_cap
                )

        # Min low-ownership players
        min_low_own = gc.get("min_low_own_players")
        low_own_thresh = gc.get("low_own_threshold", 0.40)
        if min_low_own is not None and min_low_own > 0:
            low_own_idxs = [
                i for i in range(n)
                if float(players[i].get("own_pct", 0)) < low_own_thresh
            ]
            if low_own_idxs:
                prob += (
                    pulp.lpSum(x[(i, j)] for i in low_own_idxs for j in range(k))
                    >= min_low_own
                )

        # Game stack: require min_game_stack players from same game
        force_game_stack = gc.get("force_game_stack", False)
        min_game_stack = gc.get("min_game_stack", 3)
        if force_game_stack and "game_id" in players[0]:
            game_ids = list({p["game_id"] for p in players if p.get("game_id")})
            if game_ids:
                # Binary indicator: g_var[gid] = 1 if this game is the stacked game
                g_vars = {gid: pulp.LpVariable(f"g_{gid}", cat="Binary") for gid in game_ids}
                prob += pulp.lpSum(g_vars[gid] for gid in game_ids) == 1
                for gid in game_ids:
                    game_idxs = [i for i in range(n) if players[i].get("game_id") == gid]
                    prob += (
                        pulp.lpSum(x[(i, j)] for i in game_idxs for j in range(k))
                        >= min_game_stack * g_vars[gid]
                    )

        # Team stack: require min_team_stack players from same team
        min_team_stack = gc.get("min_team_stack", 0)
        if min_team_stack > 0:
            teams = list({p["team"] for p in players if p.get("team")})
            if teams:
                t_vars = {t: pulp.LpVariable(f"t_{t}", cat="Binary") for t in teams}
                prob += pulp.lpSum(t_vars[t] for t in teams) >= 1
                for t in teams:
                    team_idxs = [i for i in range(n) if players[i].get("team") == t]
                    prob += (
                        pulp.lpSum(x[(i, j)] for i in team_idxs for j in range(k))
                        >= min_team_stack * t_vars[t]
                    )

        # Bring-back: if we stack 2+ players from one team in a game,
        # require at least 1 from the opposing team (game correlation).
        force_bring_back = gc.get("force_bring_back", False)
        if force_bring_back and "game_id" in players[0] and "team" in players[0]:
            game_ids = list({p["game_id"] for p in players if p.get("game_id")})
            for gid in game_ids:
                game_players = [i for i in range(n) if players[i].get("game_id") == gid]
                if len(game_players) >= 2:
                    teams_in_game = list({players[i]["team"] for i in game_players})
                    if len(teams_in_game) == 2:
                        team_a, team_b = teams_in_game
                        a_idxs = [i for i in game_players if players[i]["team"] == team_a]
                        b_idxs = [i for i in game_players if players[i]["team"] == team_b]
                        # Binary: bb_a = 1 if >=2 from team_a selected
                        bb_a = pulp.LpVariable(f"bb_a_{gid}", cat="Binary")
                        bb_b = pulp.LpVariable(f"bb_b_{gid}", cat="Binary")
                        a_count = pulp.lpSum(x[(i, j)] for i in a_idxs for j in range(k))
                        b_count = pulp.lpSum(x[(i, j)] for i in b_idxs for j in range(k))
                        # bb_a=1 when a_count >= 2  (big-M: M=8)
                        prob += a_count >= 2 * bb_a
                        prob += a_count <= 1 + 7 * bb_a
                        # bb_b=1 when b_count >= 2
                        prob += b_count >= 2 * bb_b
                        prob += b_count <= 1 + 7 * bb_b
                        # If stacking team_a (2+), require >=1 from team_b
                        prob += b_count >= bb_a
                        # If stacking team_b (2+), require >=1 from team_a
                        prob += a_count >= bb_b

    # Tier constraints from edge state
    tc = tier_constraints or {}
    if tc:
        tier_player_names = tc.get("tier_player_names", {})
        tier_min = tc.get("tier_min_players", {})
        tier_max = tc.get("tier_max_players", {})
        name_to_idx = {p["player_name"]: i for i, p in enumerate(players)}

        for tier, names in tier_player_names.items():
            idxs = [name_to_idx[nm] for nm in names if nm in name_to_idx]
            if not idxs:
                continue
            if tier in tier_min:
                prob += (
                    pulp.lpSum(x[(i, j)] for i in idxs for j in range(k))
                    >= tier_min[tier]
                )
            if tier in tier_max:
                prob += (
                    pulp.lpSum(x[(i, j)] for i in idxs for j in range(k))
                    <= tier_max[tier]
                )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    result = []
    for j in range(k):
        for i in range(n):
            if _eligible(players[i], slots[j]) and pulp.value(x[(i, j)]) > 0.5:
                result.append((players[i], slots[j]))
                break
    return result if len(result) == k else None


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

    # ── Contest-type detection ──────────────────────────────────────────────────────
    contest_type = cfg.get("CONTEST_TYPE", "gpp").lower()
    is_showdown_classic = contest_type in ("showdown",)
    is_gpp = contest_type in ("gpp", "mme", "sd", "captain")
    is_cash = contest_type in ("cash", "50/50", "double-up")

    # GPP constraint knobs — overridable via cfg; defaults come from DEFAULT_CONFIG
    # (single source of truth in config.py — do NOT add numeric literals here)
    gpp_max_punt_players = int(cfg.get("GPP_MAX_PUNT_PLAYERS", DEFAULT_CONFIG["GPP_MAX_PUNT_PLAYERS"]))
    gpp_min_mid_players  = int(cfg.get("GPP_MIN_MID_PLAYERS",  DEFAULT_CONFIG["GPP_MIN_MID_PLAYERS"]))
    gpp_own_cap          = float(cfg.get("GPP_OWN_CAP",         DEFAULT_CONFIG["GPP_OWN_CAP"]))
    gpp_min_low_own      = int(cfg.get("GPP_MIN_LOW_OWN_PLAYERS", DEFAULT_CONFIG["GPP_MIN_LOW_OWN_PLAYERS"]))
    gpp_low_own_thresh   = float(cfg.get("GPP_LOW_OWN_THRESHOLD", DEFAULT_CONFIG["GPP_LOW_OWN_THRESHOLD"]))
    gpp_force_game_stack = bool(cfg.get("GPP_FORCE_GAME_STACK",  DEFAULT_CONFIG["GPP_FORCE_GAME_STACK"]))
    gpp_min_game_stack   = int(cfg.get("GPP_MIN_GAME_STACK",     DEFAULT_CONFIG["GPP_MIN_GAME_STACK"]))
    gpp_min_team_stack   = int(cfg.get("GPP_MIN_TEAM_STACK",     DEFAULT_CONFIG["GPP_MIN_TEAM_STACK"]))
    gpp_force_bring_back = bool(cfg.get("GPP_FORCE_BRING_BACK",  DEFAULT_CONFIG["GPP_FORCE_BRING_BACK"]))

    pos_slots = cfg.get("POS_SLOTS", DK_POS_SLOTS)
    players = player_pool.to_dict("records")
    n = len(players)

    score_col = "cash_score" if is_cash else "gpp_score"

    # Build name-to-index maps
    name_to_idx = {p["player_name"]: i for i, p in enumerate(players)}
    lock_indices = [name_to_idx[nm] for nm in lock_names if nm in name_to_idx]
    exclude_names = [nm.strip() for nm in cfg.get("EXCLUDE", [])]
    exclude_indices = [name_to_idx[nm] for nm in exclude_names if nm in name_to_idx]
    not_with_idx_pairs = [
        (name_to_idx[a], name_to_idx[b])
        for (a, b) in not_with_pairs
        if a in name_to_idx and b in name_to_idx
    ]

    # Build per-player appearance budget
    remaining = {}
    for i, p in enumerate(players):
        if i in exclude_indices:
            remaining[i] = 0
        elif i in lock_indices:
            remaining[i] = num_lineups  # locks appear in every lineup
        else:
            pname = p["player_name"]
            # Per-player override if provided
            if pname in player_max_exp:
                cap = max(1, int(num_lineups * float(player_max_exp[pname])))
            else:
                cap = max_appearances
            remaining[i] = cap

    # Build tier constraint structures
    tier_dict: Dict[str, list] = {}
    tier_min_d: Dict[str, int] = {}
    tier_max_d: Dict[str, int] = {}
    if tier_constraints:
        for tier, names in tier_player_names.items():
            tier_dict[tier] = [name_to_idx[nm] for nm in names if nm in name_to_idx]
        tier_min_d = dict(tier_min_players)
        tier_max_d = dict(tier_max_players)

    # Pair-appearances tracking matrix
    pair_appearances: Dict[tuple, int] = {}

    # ── Per-solve projection randomization for GPP/MME ──────────────────────
    # Re-seed projections each solve so the optimizer explores different
    # player combos.  Without this, exposure limits are the ONLY source of
    # lineup diversity and all lineups converge to the same core.
    rng = np.random.default_rng(seed=42)

    lineups = []
    for lineup_num in range(num_lineups):
        if progress_callback:
            progress_callback(lineup_num, num_lineups)

        # Randomize scores for GPP/MME — each solve sees a different surface
        if is_gpp and num_lineups > 1:
            noise = rng.normal(1.0, 0.10, size=n)  # ±10% noise
            for i in range(n):
                base_score = players[i].get(score_col, players[i].get("proj", 0))
                players[i]["_solve_score"] = base_score * noise[i]
            _solve_score_col = "_solve_score"
        else:
            _solve_score_col = score_col

        gpp_constraints_d = None
        if is_gpp:
            gpp_constraints_d = {
                "max_punt_players":   gpp_max_punt_players,
                "min_mid_players":    gpp_min_mid_players,
                "own_cap":            gpp_own_cap,
                "min_low_own_players": gpp_min_low_own,
                "low_own_threshold": gpp_low_own_thresh,
                "force_game_stack":   gpp_force_game_stack,
                "min_game_stack":     gpp_min_game_stack,
                "min_team_stack":     gpp_min_team_stack,
                "force_bring_back":   gpp_force_bring_back,
            }

        tier_constraints_d = None
        if tier_dict:
            tier_constraints_d = {
                "tier_player_names": tier_dict,
                "tier_min_players": tier_min_d,
                "tier_max_players": tier_max_d,
            }

        # Build pair-appearance constraints from tracking
        extra_not_with = list(not_with_idx_pairs)
        if max_pair_appearances > 0:
            over_pairs = [
                (a, b) for (a, b), cnt in pair_appearances.items()
                if cnt >= max_pair_appearances
            ]
            extra_not_with.extend(over_pairs)

        result = _build_one_lineup(
            players=players,
            pos_slots=pos_slots,
            salary_cap=salary_cap,
            min_salary=min_salary,
            max_appearances=remaining,
            score_col=_solve_score_col,
            pos_caps=pos_caps,
            solver_time_limit=solver_time_limit,
            gpp_constraints=gpp_constraints_d,
            tier_constraints=tier_constraints_d,
            not_with_pairs=extra_not_with,
            forced_players=lock_indices,
            excluded_players=exclude_indices,
        )

        if result is None:
            # Retry without GPP constraints (fallback)
            result = _build_one_lineup(
                players=players,
                pos_slots=pos_slots,
                salary_cap=salary_cap,
                min_salary=min_salary,
                max_appearances=remaining,
                score_col=_solve_score_col,
                pos_caps=pos_caps,
                solver_time_limit=solver_time_limit,
                gpp_constraints=None,
                tier_constraints=None,
                not_with_pairs=not_with_idx_pairs,
                forced_players=lock_indices,
                excluded_players=exclude_indices,
            )

        if result is None:
            break

        # Record lineup
        lineup_rows = []
        selected_indices = []
        for player, slot in result:
            selected_indices.append(name_to_idx[player["player_name"]])
            lineup_rows.append({
                "lineup_index": lineup_num,
                "slot": slot,
                "player_name": player["player_name"],
                "team": player.get("team", ""),
                "position": player.get("position", ""),
                "salary": player["salary"],
                "proj": player.get("proj", 0),
                "own_pct": player.get("own_pct", 0),
                "gpp_score": player.get("gpp_score", 0),
                "cash_score": player.get("cash_score", 0),
            })

        lineups.append(lineup_rows)

        # Update remaining appearances
        for idx in selected_indices:
            if idx not in lock_indices:
                remaining[idx] = max(0, remaining[idx] - 1)

        # Update pair appearances
        if max_pair_appearances > 0:
            for a in selected_indices:
                for b in selected_indices:
                    if a < b:
                        pair_appearances[(a, b)] = pair_appearances.get((a, b), 0) + 1

    # Flatten lineups into a DataFrame
    if not lineups:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = [row for lu in lineups for row in lu]
    lineups_df = pd.DataFrame(all_rows)

    # Exposure report
    if not lineups_df.empty:
        n_built = lineups_df["lineup_index"].nunique()
        exp_rows = []
        for pname, idx in name_to_idx.items():
            times = sum(
                1 for lu in lineups
                if any(r["player_name"] == pname for r in lu)
            )
            if times > 0:
                exp_rows.append({
                    "player_name": pname,
                    "team": players[idx].get("team", ""),
                    "salary": players[idx]["salary"],
                    "proj": players[idx].get("proj", 0),
                    "own_pct": players[idx].get("own_pct", 0),
                    "lineups": times,
                    "exposure": times / max(n_built, 1),
                })
        exposures_df = pd.DataFrame(exp_rows).sort_values("exposure", ascending=False)
    else:
        exposures_df = pd.DataFrame()

    return lineups_df, exposures_df


# =============================================================================
# STACKING HELPERS (game correlation)
# =============================================================================

def _build_one_lineup_with_stacks(
    players: list,
    pos_slots: list,
    salary_cap: int,
    min_salary: int,
    max_appearances: dict,
    score_col: str = "gpp_score",
    pos_caps: dict | None = None,
    solver_time_limit: int = 30,
    stack_rules: dict | None = None,
    not_with_pairs: list | None = None,
    forced_players: list | None = None,
    excluded_players: list | None = None,
) -> list | None:
    """Extend _build_one_lineup with game/team stacking constraints."""
    # Delegate to the core solver; stacking is implemented inside gpp_constraints.
    return _build_one_lineup(
        players=players,
        pos_slots=pos_slots,
        salary_cap=salary_cap,
        min_salary=min_salary,
        max_appearances=max_appearances,
        score_col=score_col,
        pos_caps=pos_caps,
        solver_time_limit=solver_time_limit,
        gpp_constraints=stack_rules,
        not_with_pairs=not_with_pairs,
        forced_players=forced_players,
        excluded_players=excluded_players,
    )


def _build_one_lineup_pass2(
    players: list,
    pos_slots: list,
    salary_cap: int,
    min_salary: int,
    max_appearances: dict,
    score_col: str = "gpp_score",
    pos_caps: dict | None = None,
    solver_time_limit: int = 30,
    gpp_constraints: dict | None = None,
    tier_constraints: dict | None = None,
    not_with_pairs: list | None = None,
    forced_players: list | None = None,
    excluded_players: list | None = None,
    already_built: list | None = None,
    min_unique_players: int = 1,
) -> list | None:
    """Build one lineup that differs from all already-built lineups by at
    least ``min_unique_players`` players.

    Adds a uniqueness constraint: for each existing lineup, the sum of
    indicator variables for players in that lineup must be ≤
    (lineup_size − min_unique_players).
    """
    n = len(players)
    k = len(pos_slots)

    def _eligible(player: dict, slot: str) -> bool:
        nat_pos = str(player.get("position", "")).upper()
        if slot == "UTIL":
            return True
        if slot == "G":
            return nat_pos in ("PG", "SG", "G")
        if slot == "F":
            return nat_pos in ("SF", "PF", "F")
        return nat_pos == slot

    prob2 = pulp.LpProblem("dfs_classic_pass2", pulp.LpMaximize)

    # Use slot INDEX (j) instead of slot NAME to avoid key collision
    # when pos_slots has duplicates (e.g. PGA ["G"]*6).
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in range(n)
        for j in range(k)
    }

    score_key = score_col
    prob2 += pulp.lpSum(
        players[i].get(score_key, players[i].get("proj", 0)) * x[(i, j)]
        for i in range(n)
        for j in range(k)
    )

    for j in range(k):
        prob2 += pulp.lpSum(
            x[(i, j)] for i in range(n) if _eligible(players[i], pos_slots[j])
        ) == 1

    for i in range(n):
        prob2 += pulp.lpSum(x[(i, j)] for j in range(k)) <= 1

    salary_sum2 = pulp.lpSum(
        players[i]["salary"] * x[(i, j)]
        for i in range(n)
        for j in range(k)
    )
    prob2 += salary_sum2 <= salary_cap
    prob2 += salary_sum2 >= min_salary
    if pos_caps:
        for nat_pos, cap_val in pos_caps.items():
            eligible_players = [
                i for i in range(n)
                if str(players[i].get("position", "")).upper() == nat_pos
            ]
            if eligible_players:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in eligible_players for j in range(k))
                    <= cap_val
                )

    for i, max_app in max_appearances.items():
        prob2 += pulp.lpSum(x[(i, j)] for j in range(k)) <= max_app

    if forced_players:
        for fp_idx in forced_players:
            prob2 += pulp.lpSum(x[(fp_idx, j)] for j in range(k)) == 1

    if excluded_players:
        for ep_idx in excluded_players:
            prob2 += pulp.lpSum(x[(ep_idx, j)] for j in range(k)) == 0

    if not_with_pairs:
        for (idx_a, idx_b) in not_with_pairs:
            prob2 += (
                pulp.lpSum(x[(idx_a, j)] for j in range(k))
                + pulp.lpSum(x[(idx_b, j)] for j in range(k))
                <= 1
            )

    gc = gpp_constraints or {}
    if gc:
        max_punt = gc.get("max_punt_players")
        if max_punt is not None:
            punt_idxs = [i for i in range(n) if players[i]["salary"] < 4000]
            if punt_idxs:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in punt_idxs for j in range(k))
                    <= max_punt
                )
        min_mid = gc.get("min_mid_players")
        if min_mid is not None:
            mid_idxs = [i for i in range(n) if 4000 <= players[i]["salary"] <= 7000]
            if mid_idxs:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in mid_idxs for j in range(k))
                    >= min_mid
                )
        own_cap = gc.get("own_cap")
        if own_cap is not None and own_cap > 0:
            own_vals = [float(p.get("own_pct", 0)) for p in players]
            if max(own_vals) > 0:
                prob2 += (
                    pulp.lpSum(
                        own_vals[i] * x[(i, j)]
                        for i in range(n)
                        for j in range(k)
                    ) <= own_cap
                )
        min_low_own = gc.get("min_low_own_players")
        low_own_thresh = gc.get("low_own_threshold", 0.40)
        if min_low_own is not None and min_low_own > 0:
            low_own_idxs = [
                i for i in range(n)
                if float(players[i].get("own_pct", 0)) < low_own_thresh
            ]
            if low_own_idxs:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in low_own_idxs for j in range(k))
                    >= min_low_own
                )

    tc = tier_constraints or {}
    if tc:
        tier_player_names2 = tc.get("tier_player_names", {})
        tier_min2 = tc.get("tier_min_players", {})
        tier_max2 = tc.get("tier_max_players", {})
        name_to_idx2 = {p["player_name"]: i for i, p in enumerate(players)}
        for tier, names in tier_player_names2.items():
            idxs = [name_to_idx2[nm] for nm in names if nm in name_to_idx2]
            if not idxs:
                continue
            if tier in tier_min2:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in idxs for j in range(k))
                    >= tier_min2[tier]
                )
            if tier in tier_max2:
                prob2 += (
                    pulp.lpSum(x[(i, j)] for i in idxs for j in range(k))
                    <= tier_max2[tier]
                )

    # Uniqueness constraints
    if already_built:
        for existing_lineup in already_built:
            # existing_lineup is a list of (player_dict, slot) tuples
            existing_idxs = set()
            name_to_idx_local = {p["player_name"]: i for i, p in enumerate(players)}
            for (p_dict, _slot) in existing_lineup:
                nm = p_dict["player_name"]
                if nm in name_to_idx_local:
                    existing_idxs.add(name_to_idx_local[nm])
            if existing_idxs:
                prob2 += (
                    pulp.lpSum(
                        x[(i, j)]
                        for i in existing_idxs
                        for j in range(k)
                        if i < n
                    ) <= k - min_unique_players
                )

    solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
    prob2.solve(solver2)

    if pulp.LpStatus[prob2.status] != "Optimal":
        return None

    result = []
    for j in range(k):
        for i in range(n):
            if _eligible(players[i], pos_slots[j]) and pulp.value(x[(i, j)]) > 0.5:
                result.append((players[i], pos_slots[j]))
                break
    return result if len(result) == k else None


# =============================================================================
# PuLP OPTIMISER — SHOWDOWN (Captain Mode)
# =============================================================================

def build_showdown_lineups(
    player_pool: pd.DataFrame,
    cfg: Dict[str, Any],
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build Showdown Captain-mode lineups (1 CPT + 5 FLEX).

    The Captain slot:
    - Earns 1.5× fantasy points
    - Costs 1.5× salary
    - May be filled by any position

    Parameters
    ----------
    player_pool : pd.DataFrame
        Pool already filtered to the 2-team matchup.
    cfg : Dict[str, Any]
        Config dict.  Key fields used:
        - ``NUM_LINEUPS``        : number of lineups to build (default 20)
        - ``SALARY_CAP``         : max total salary (default 50000)
        - ``MIN_SALARY_USED``    : min total salary (default 0 for Showdown)
        - ``MAX_EXPOSURE``       : max fraction of lineups any player appears in
        - ``SOLVER_TIME_LIMIT``  : seconds per LP solve (default 30)
        - ``LOCK``               : list of player names forced into every lineup
        - ``SD_CAPTAIN_OWN_PENALTY``: penalise high-owned captains (default 10.0)
        - ``SD_CAPTAIN_CEIL_BONUS`` : bonus weight on ceiling for CPT slot

    Returns
    -------
    (lineups_df, exposures_df) in the same long format as the Classic optimizer,
    with ``slot`` values of ``"CPT"`` or ``"FLEX"``.
    """
    num_lineups = int(cfg.get("NUM_LINEUPS", 20))
    salary_cap = int(cfg.get("SALARY_CAP", 50000))
    min_salary = int(cfg.get("MIN_SALARY_USED", 0))  # Showdown has no salary floor
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

    # Build augmented player list: each base player appears twice:
    #   index i      -> FLEX version (salary = base, score = base)
    #   index i + m  -> CPT version  (salary = 1.5x, score = 1.5x)
    cpt_mult = DK_SHOWDOWN_CAPTAIN_MULTIPLIER

    players: list[dict] = []
    for p in base_players:
        # FLEX copy
        players.append({
            **p,
            "_role": "FLEX",
            "_base_idx": len(players),  # will be overwritten below
        })
    flex_count = len(players)  # == m
    for p in base_players:
        # CPT copy
        cpt_salary = int(round(p["salary"] * cpt_mult))
        cpt_score  = p.get("gpp_score", p.get("proj", 0)) * cpt_mult
        # Adjust captain score: bonus for ceiling, penalty for high ownership
        ceil_bonus    = float(cfg.get("SD_CAPTAIN_CEIL_BONUS", 0.2))
        own_penalty   = float(cfg.get("SD_CAPTAIN_OWN_PENALTY", 10.0))
        if "ceil" in p:
            cpt_score += ceil_bonus * float(p["ceil"])
        cpt_score -= own_penalty * float(p.get("own_pct", 0))
        players.append({
            **p,
            "salary": cpt_salary,
            "gpp_score": cpt_score,
            "_role": "CPT",
            "_base_idx": len(players) - flex_count,
        })

    # Fix _base_idx for FLEX copies
    for i in range(flex_count):
        players[i]["_base_idx"] = i

    n = len(players)  # == 2 * m
    pos_slots = DK_SHOWDOWN_SLOTS  # ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
    k = len(pos_slots)

    # Name-to-index maps (FLEX indices = 0..m-1, CPT indices = m..2m-1)
    name_to_flex = {p["player_name"]: i for i, p in enumerate(players[:flex_count])}
    name_to_cpt  = {p["player_name"]: i for i, p in enumerate(players[flex_count:], start=flex_count)}

    lock_flex_indices = [name_to_flex[nm] for nm in lock_names if nm in name_to_flex]

    # Appearance budgets (apply to base-player level, not CPT/FLEX separately)
    base_remaining = {i: max_appearances for i in range(m)}
    for i in range(m):
        pname = base_players[i]["player_name"]
        if pname in [base_players[li]["player_name"] for li in lock_flex_indices]:
            base_remaining[i] = num_lineups

    pair_appearances: Dict[tuple, int] = {}
    lineups = []

    for lineup_num in range(num_lineups):
        if progress_callback:
            progress_callback(lineup_num, num_lineups)

        # Build extra not-with pairs from pair appearance tracking
        extra_not_with_base: list[tuple[int, int]] = []
        if max_pair_appearances > 0:
            extra_not_with_base = [
                (a, b) for (a, b), cnt in pair_appearances.items()
                if cnt >= max_pair_appearances
            ]

        prob = pulp.LpProblem(f"sd_{lineup_num}", pulp.LpMaximize)

        # y[i] = 1 if player i (in augmented list) is selected
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        # Objective: maximise total adjusted score
        prob += pulp.lpSum(
            players[i].get("gpp_score", players[i].get("proj", 0)) * y[i]
            for i in range(n)
        )

        # Exactly 1 CPT slot
        prob += pulp.lpSum(y[i] for i in range(flex_count, n)) == 1

        # Exactly 5 FLEX slots
        prob += pulp.lpSum(y[i] for i in range(flex_count)) == 5

        # A base player cannot be both CPT and FLEX
        for base_i in range(m):
            prob += y[base_i] + y[base_i + flex_count] <= 1

        # Salary cap and floor
        prob += pulp.lpSum(players[i]["salary"] * y[i] for i in range(n)) <= salary_cap
        prob += pulp.lpSum(players[i]["salary"] * y[i] for i in range(n)) >= min_salary

        # LOCK: locked players must appear (either as CPT or FLEX)
        if lock_names:
            for nm in lock_names:
                if nm in name_to_flex:
                    flex_i = name_to_flex[nm]
                    cpt_i  = name_to_cpt[nm]
                    prob += y[flex_i] + y[cpt_i] >= 1

        # Exposure: each base player can appear at most base_remaining[base_i] more times
        for base_i in range(m):
            budget = base_remaining[base_i]
            prob += y[base_i] + y[base_i + flex_count] <= budget

        # NOT_WITH at base-player level
        if extra_not_with_base:
            for (bi_a, bi_b) in extra_not_with_base:
                prob += (
                    y[bi_a] + y[bi_a + flex_count]
                    + y[bi_b] + y[bi_b + flex_count]
                    <= 1
                )

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        # Decode solution
        lineup_rows = []
        selected_base_idxs = []

        # CPT
        for i in range(flex_count, n):
            if pulp.value(y[i]) > 0.5:
                p = players[i]
                base_i = i - flex_count
                selected_base_idxs.append(base_i)
                lineup_rows.append({
                    "lineup_index": lineup_num,
                    "slot": "CPT",
                    "player_name": p["player_name"],
                    "team": p.get("team", ""),
                    "position": p.get("position", ""),
                    "salary": p["salary"],  # already 1.5x
                    "proj": p.get("proj", 0) * cpt_mult,
                    "own_pct": p.get("own_pct", 0),
                    "gpp_score": p.get("gpp_score", 0),
                    "cash_score": p.get("cash_score", 0),
                })
                break

        # FLEX
        for i in range(flex_count):
            if pulp.value(y[i]) > 0.5:
                p = players[i]
                selected_base_idxs.append(i)
                lineup_rows.append({
                    "lineup_index": lineup_num,
                    "slot": "FLEX",
                    "player_name": p["player_name"],
                    "team": p.get("team", ""),
                    "position": p.get("position", ""),
                    "salary": p["salary"],
                    "proj": p.get("proj", 0),
                    "own_pct": p.get("own_pct", 0),
                    "gpp_score": p.get("gpp_score", 0),
                    "cash_score": p.get("cash_score", 0),
                })

        lineups.append(lineup_rows)

        # Update appearance budgets
        for base_i in selected_base_idxs:
            pname = base_players[base_i]["player_name"]
            is_locked = pname in lock_names
            if not is_locked:
                base_remaining[base_i] = max(0, base_remaining[base_i] - 1)

        # Update pair appearances
        if max_pair_appearances > 0:
            for a in selected_base_idxs:
                for b in selected_base_idxs:
                    if a < b:
                        pair_appearances[(a, b)] = pair_appearances.get((a, b), 0) + 1

    if not lineups:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = [row for lu in lineups for row in lu]
    lineups_df = pd.DataFrame(all_rows)

    # Exposure
    n_built = lineups_df["lineup_index"].nunique() if not lineups_df.empty else 0
    exp_rows = []
    for base_i, p in enumerate(base_players):
        times = sum(
            1 for lu in lineups
            if any(r["player_name"] == p["player_name"] for r in lu)
        )
        if times > 0:
            exp_rows.append({
                "player_name": p["player_name"],
                "team": p.get("team", ""),
                "salary": p["salary"],
                "proj": p.get("proj", 0),
                "own_pct": p.get("own_pct", 0),
                "lineups": times,
                "exposure": times / max(n_built, 1),
            })
    exposures_df = pd.DataFrame(exp_rows).sort_values("exposure", ascending=False)

    return lineups_df, exposures_df


# =============================================================================
# RUN CONFIGURATION BUILDER
# =============================================================================

def build_run_config(
    player_pool: pd.DataFrame,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Merge user overrides into DEFAULT_CONFIG and return enriched run dict.

    Parameters
    ----------
    player_pool : pd.DataFrame
        The player pool that will be optimised (used for metadata).
    overrides : dict or None
        Keys to override in DEFAULT_CONFIG.  Accepts both canonical
        UPPER-case keys and lowercase aliases.

    Returns
    -------
    dict
        Merged config with extra keys: ``num_players``, ``config``.
    """
    from yak_core.config import merge_config
    merged = merge_config(overrides or {})
    return {
        "num_lineups": merged.get("NUM_LINEUPS"),
        "salary_cap": merged.get("SALARY_CAP"),
        "max_exposure": merged.get("MAX_EXPOSURE"),
        "logic_profile": merged.get("LOGIC_PROFILE"),
        "band": merged.get("BAND"),
        "min_salary_used": merged.get("MIN_SALARY_USED"),
        "yakos_root": YAKOS_ROOT,
        "num_players": int(len(player_pool)),
        "config": merged,
    }


# =============================================================================
# BACKWARD-COMPATIBILITY SHIMS
# =============================================================================
# __init__.py and older callers may still import these names.
# Keep them around as thin wrappers / deprecation stubs.

def load_opt_pool_from_config(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Deprecated. Use load_player_pool() + prepare_pool() instead."""
    raise RuntimeError(
        "load_opt_pool_from_config is deprecated; "
        "use load_player_pool() + prepare_pool() instead."
    )


def build_player_pool(opt_pool: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Deprecated wrapper — delegates to prepare_pool()."""
    import warnings
    warnings.warn(
        "build_player_pool() is deprecated; use prepare_pool() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return prepare_pool(opt_pool, cfg)


def build_slate_pool(*args, **kwargs) -> pd.DataFrame:
    """Deprecated stub. Slate building now lives in scripts/load_pool.py."""
    raise RuntimeError(
        "build_slate_pool() was removed in the lineups rewrite. "
        "Use scripts/load_pool.py to build the slate pool."
    )


def run_lineups_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Deprecated stub. Use build_run_config() + build_multiple_lineups_with_exposure() directly."""
    raise RuntimeError(
        "run_lineups_from_config() was removed in the lineups rewrite. "
        "Use build_run_config() + the new optimizer functions instead."
    )


def to_dk_upload_format(lineups_df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated stub. DK upload formatting now lives in yak_core.publishing."""
    raise RuntimeError(
        "to_dk_upload_format() was removed in the lineups rewrite. "
        "Use yak_core.publishing instead."
    )
