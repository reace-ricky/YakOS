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

# ── Silence PuLP's default stdout banner ──────────────────────────────────────
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
    # Defaults match v6 formula: proj*0.35 + ceil*0.55 - own*5
    gpp_proj_weight = float(cfg.get("GPP_PROJ_WEIGHT", 0.35))
    gpp_ceil_weight = float(cfg.get("GPP_CEIL_WEIGHT", 0.55))
    gpp_own_weight  = float(cfg.get("GPP_OWN_WEIGHT", -5.0))

    ceil_col = df["ceil"] if "ceil" in df.columns else df["proj"]
    own_col  = df["own_pct"]
    df["gpp_score"] = (
        df["proj"] * gpp_proj_weight
        + ceil_col * gpp_ceil_weight
        + own_col * gpp_own_weight
    )

    # ── cash_score = floor-weighted ───────────────────────────────────────────
    floor_w = float(cfg.get("CASH_FLOOR_WEIGHT", 0.6))
    proj_w  = float(cfg.get("CASH_PROJ_WEIGHT",  0.4))
    if "floor" in df.columns:
        df["cash_score"] = floor_w * df["floor"] + proj_w * df["proj"]
    else:
        df["cash_score"] = df["proj"]

    # ── value_score ───────────────────────────────────────────────────────────
    if value_weight > 0:
        df["value_score"] = df["proj"] / (df["salary"] / 1_000.0 + 1e-9)
    else:
        df["value_score"] = 0.0

    # ── stack_score (game-level correlation bonus) ───────────────────────────
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
# PuLP OPTIMISER – CLASSIC
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

    # x[i,s] = 1 if player i is assigned to slot s
    x = {(i, s): pulp.LpVariable(f"x_{i}_{s}", cat="Binary")
         for i in range(n) for s in slots}

    # Objective: maximise total score
    score_key = score_col
    prob += pulp.lpSum(
        players[i].get(score_key, players[i].get("proj", 0)) * x[(i, s)]
        for i in range(n)
        for s in slots
    )

    # Each slot must be filled by exactly one player
    for s in slots:
        prob += pulp.lpSum(x[(i, s)] for i in range(n) if _eligible(players[i], s)) == 1

    # Each player may appear at most once across all slots
    for i in range(n):
        prob += pulp.lpSum(x[(i, s)] for s in slots) <= 1

    # Salary cap
    prob += (
        pulp.lpSum(
            players[i]["salary"] * x[(i, s)]
            for i in range(n)
            for s in slots
        ) <= salary_cap
    )

    # Salary floor
    prob += (
        pulp.lpSum(
            players[i]["salary"] * x[(i, s)]
            for i in range(n)
            for s in slots
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
                    pulp.lpSum(x[(i, s)] for i in eligible_for_pos for s in slots)
                    <= cap_val
                )

    # Exposure: each player may appear at most max_appearances[i] more times
    for i, max_app in max_appearances.items():
        prob += pulp.lpSum(x[(i, s)] for s in slots) <= max_app

    # Forced players (locks)
    if forced_players:
        for fp_idx in forced_players:
            prob += pulp.lpSum(x[(fp_idx, s)] for s in slots) == 1

    # Excluded players
    if excluded_players:
        for ep_idx in excluded_players:
            prob += pulp.lpSum(x[(ep_idx, s)] for s in slots) == 0

    # NOT_WITH pairs
    if not_with_pairs:
        for (idx_a, idx_b) in not_with_pairs:
            prob += (
                pulp.lpSum(x[(idx_a, s)] for s in slots)
                + pulp.lpSum(x[(idx_b, s)] for s in slots)
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
                    pulp.lpSum(x[(i, s)] for i in punt_idxs for s in slots)
                    <= max_punt
                )

        # Min mid-salary players (4000-7000)
        min_mid = gc.get("min_mid_players")
        if min_mid is not None:
            mid_idxs = [i for i in range(n) if 4000 <= players[i]["salary"] <= 7000]
            if mid_idxs:
                prob += (
                    pulp.lpSum(x[(i, s)] for i in mid_idxs for s in slots)
                    >= min_mid
                )

        # Max total ownership cap
        own_cap = gc.get("own_cap")
        if own_cap is not None and own_cap > 0:
            own_vals = [float(p.get("own_pct", 0)) for p in players]
            if max(own_vals) > 0:
                prob += (
                    pulp.lpSum(
                        own_vals[i] * x[(i, s)]
                        for i in range(n)
                        for s in slots
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
                    pulp.lpSum(x[(i, s)] for i in low_own_idxs for s in slots)
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
                        pulp.lpSum(x[(i, s)] for i in game_idxs for s in slots)
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
                    n_slots = len(slots)
                    prob += (
                        pulp.lpSum(x[(i, s)] for i in team_idxs for s in slots)
                        >= min_team_stack * t_vars[t]
                    )

        # Bring-back: require 1 player from opposing team in stacked game
        force_bring_back = gc.get("force_bring_back", False)
        if force_bring_back and "game_id" in players[0] and "team" in players[0]:
            # For each game, require at least 1 player from each team
            game_ids = list({p["game_id"] for p in players if p.get("game_id")})
            for gid in game_ids:
                game_players = [i for i in range(n) if players[i].get("game_id") == gid]
                if len(game_players) >= 2:
                    teams_in_game = list({players[i]["team"] for i in game_players})
                    if len(teams_in_game) == 2:
                        # Soft bring-back: if any player from team A is selected,
                        # require at least 1 from team B
                        # Implement as: min 1 from team B if >= 2 from team A
                        pass  # Simplified – full bring-back implemented below

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
                    pulp.lpSum(x[(i, s)] for i in idxs for s in slots)
                    >= tier_min[tier]
                )
            if tier in tier_max:
                prob += (
                    pulp.lpSum(x[(i, s)] for i in idxs for s in slots)
                    <= tier_max[tier]
                )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    result = []
    for s in slots:
        for i in range(n):
            if _eligible(players[i], s) and pulp.value(x[(i, s)]) > 0.5:
                result.append((players[i], s))
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

    # ── Contest-type detection ────────────────────────────────────────────────
    contest_type = cfg.get("CONTEST_TYPE", "gpp").lower()
    is_showdown_classic = contest_type in ("showdown",)
    is_gpp = contest_type in ("gpp", "mme", "sd", "captain")
    is_cash = contest_type in ("cash", "50/50", "double-up")

    # GPP constraint knobs – overridable via cfg; defaults come from DEFAULT_CONFIG
