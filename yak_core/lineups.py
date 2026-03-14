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
        Player pool with standardised columns (varies by sport).
    """
    sport = sport.upper()
    if yakos_root is None:
        yakos_root = YAKOS_ROOT

    if data_mode == "historical":
        return _load_historical(sport, slate_date, yakos_root)
    elif data_mode == "live":
        return _load_live(sport, slate_date)
    else:
        raise ValueError(f"Unknown data_mode '{data_mode}'. Use 'historical' or 'live'.")


def _load_historical(
    sport: str,
    slate_date: str | None,
    yakos_root: str,
) -> pd.DataFrame:
    """Read the player pool from the local parquet cache."""
    if slate_date is None:
        slate_date = str(date.today())

    base = os.path.join(yakos_root, "data", sport.lower(), "slates")
    parquet_path = os.path.join(base, f"{slate_date}.parquet")
    csv_path = os.path.join(base, f"{slate_date}.csv")

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No slate file found for {sport} on {slate_date}. "
            f"Checked:\n  {parquet_path}\n  {csv_path}"
        )


def _load_live(sport: str, slate_date: str | None) -> pd.DataFrame:
    """Fetch the player pool from the Tank01 RapidAPI."""
    if sport == "NBA":
        return _load_live_nba(slate_date)
    elif sport == "PGA":
        return _load_live_pga(slate_date)
    else:
        raise ValueError(f"Live data not supported for sport '{sport}'.")


def _load_live_nba(slate_date: str | None) -> pd.DataFrame:
    """Fetch NBA DFS projections from Tank01."""
    import requests  # lazy import – only needed in live mode

    api_key = os.environ.get("TANK01_API_KEY", "")
    api_host = os.environ.get("TANK01_API_HOST", "tank01-fantasy-stats.p.rapidapi.com")

    if not api_key:
        raise EnvironmentError(
            "TANK01_API_KEY environment variable is not set. "
            "Set it to your RapidAPI key for Tank01."
        )

    if slate_date is None:
        slate_date = str(date.today())

    url = f"https://{api_host}/getNBADFSSlate"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host,
    }
    params = {"date": slate_date.replace("-", "")}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    players: list[dict] = []
    for slate in data.get("body", []):
        for player in slate.get("players", []):
            players.append(player)

    if not players:
        raise ValueError(
            f"Tank01 returned no players for NBA on {slate_date}. "
            "Check the date or your API subscription."
        )

    df = pd.DataFrame(players)
    return df


def _load_live_pga(slate_date: str | None) -> pd.DataFrame:
    """Fetch PGA DFS projections from Tank01."""
    import requests  # lazy import

    api_key = os.environ.get("TANK01_API_KEY", "")
    api_host = os.environ.get("TANK01_API_HOST", "tank01-fantasy-stats.p.rapidapi.com")

    if not api_key:
        raise EnvironmentError(
            "TANK01_API_KEY environment variable is not set."
        )

    if slate_date is None:
        slate_date = str(date.today())

    url = f"https://{api_host}/getPGADFSSlate"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host,
    }
    params = {"date": slate_date.replace("-", "")}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    players: list[dict] = []
    for slate in data.get("body", []):
        for player in slate.get("players", []):
            players.append(player)

    if not players:
        raise ValueError(
            f"Tank01 returned no players for PGA on {slate_date}. "
            "Check the date or your API subscription."
        )

    df = pd.DataFrame(players)
    return df


# =============================================================================
# PLAYER POOL PREPARATION
# =============================================================================

def prepare_pool(
    df: pd.DataFrame,
    sport: str = "NBA",
    contest_type: str = "gpp",
    slate_type: str = "main",
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Clean and enrich a raw player-pool DataFrame for lineup optimisation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw player pool (from ``load_player_pool``).
    sport : str
        ``"NBA"`` or ``"PGA"``.
    contest_type : str
        ``"gpp"`` (tournament) or ``"cash"`` (50/50, double-up).
    slate_type : str
        ``"main"`` or ``"showdown"``.
    config : dict or None
        Override for any keys in ``DEFAULT_CONFIG``.

    Returns
    -------
    pd.DataFrame
        Cleaned pool ready for the LP optimiser.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    sport = sport.upper()

    if sport == "NBA":
        return _prepare_nba(df, contest_type, slate_type, cfg)
    elif sport == "PGA":
        return _prepare_pga(df, contest_type, cfg)
    else:
        raise ValueError(f"Unsupported sport '{sport}'.")


def _prepare_nba(
    df: pd.DataFrame,
    contest_type: str,
    slate_type: str,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """NBA-specific pool preparation."""
    df = df.copy()

    # ── 1. Rename Tank01 columns to internal standard names ──────────────────
    rename_map = {
        "playerName": "name",
        "team": "team",
        "pos": "position",          # <-- Tank01 uses 'pos'; internal name is 'position'
        "salary": "salary",
        "projPts": "projection",
        "minutes": "minutes",
        "usage": "usage",
        "value": "value",
        "status": "status",
        "gameInfo": "game_info",
        "teamAbv": "team",           # fallback if 'team' absent
    }
    # Only rename columns that are actually present
    present_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=present_renames)

    # ── 2. Ensure required columns exist ────────────────────────────────────
    required = ["name", "position", "salary", "projection"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Player pool is missing required columns after rename: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # ── 3. Coerce numeric types ──────────────────────────────────────────────
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0.0)

    for col in ["minutes", "usage", "value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ── 4. Normalise position strings ────────────────────────────────────────
    df["position"] = df["position"].astype(str).str.upper().str.strip()

    # ── 5. Expand multi-position eligibility ─────────────────────────────────
    #   DraftKings lists some players as "PG/SG" etc.  Explode into one row
    #   per position so the LP solver can slot them anywhere they qualify.
    df["position"] = df["position"].str.split("/")
    df = df.explode("position").reset_index(drop=True)
    df["position"] = df["position"].str.strip()

    # ── 6. Filter to valid DK positions ──────────────────────────────────────
    if slate_type == "showdown":
        valid_positions = {"CPT", "FLEX"}
    else:
        valid_positions = {"PG", "SG", "SF", "PF", "C", "UTIL"}
    df = df[df["position"].isin(valid_positions)].reset_index(drop=True)

    # ── 7. Salary filter ─────────────────────────────────────────────────────
    min_salary = cfg.get("min_salary", 3000)
    df = df[df["salary"] >= min_salary].reset_index(drop=True)

    # ── 8. Projection floor ──────────────────────────────────────────────────
    min_proj = cfg.get("min_projection", 0.0)
    df = df[df["projection"] >= min_proj].reset_index(drop=True)

    # ── 9. Exclude injured / out players ─────────────────────────────────────
    if "status" in df.columns:
        exclude_statuses = {"OUT", "INJ", "INACTIVE", "DOUBTFUL"}
        df = df[
            ~df["status"].astype(str).str.upper().isin(exclude_statuses)
        ].reset_index(drop=True)

    # ── 10. Value column ─────────────────────────────────────────────────────
    df["value"] = df["projection"] / (df["salary"] / 1000).replace(0, np.nan)
    df["value"] = df["value"].fillna(0.0)

    # ── 11. GPP-specific adjustments ─────────────────────────────────────────
    if contest_type == "gpp":
        ownership_col = cfg.get("ownership_col", "ownership")
        if ownership_col in df.columns:
            df[ownership_col] = pd.to_numeric(
                df[ownership_col], errors="coerce"
            ).fillna(cfg.get("default_ownership", 20.0))

    return df


def _prepare_pga(
    df: pd.DataFrame,
    contest_type: str,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """PGA-specific pool preparation."""
    df = df.copy()

    rename_map = {
        "playerName": "name",
        "pos": "position",
        "salary": "salary",
        "projPts": "projection",
        "worldRank": "world_rank",
        "teamAbv": "team",
    }
    present_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=present_renames)

    required = ["name", "salary", "projection"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"PGA pool missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0.0)
    df["position"] = "G"  # PGA classic: all golfers

    min_salary = cfg.get("min_salary", 6000)
    df = df[df["salary"] >= min_salary].reset_index(drop=True)

    df["value"] = df["projection"] / (df["salary"] / 1000).replace(0, np.nan)
    df["value"] = df["value"].fillna(0.0)

    return df


# =============================================================================
# LP OPTIMIZER
# =============================================================================

def build_lineups(
    pool: pd.DataFrame,
    sport: str = "NBA",
    n_lineups: int = 20,
    contest_type: str = "gpp",
    slate_type: str = "main",
    config: Dict[str, Any] | None = None,
) -> list[pd.DataFrame]:
    """Generate ``n_lineups`` optimal DraftKings lineups via LP.

    Parameters
    ----------
    pool : pd.DataFrame
        Prepared player pool (output of ``prepare_pool``).
    sport : str
        ``"NBA"`` or ``"PGA"``.
    n_lineups : int
        Number of distinct lineups to produce.
    contest_type : str
        ``"gpp"`` or ``"cash"``.
    slate_type : str
        ``"main"`` or ``"showdown"``.
    config : dict or None
        Override for any keys in ``DEFAULT_CONFIG``.

    Returns
    -------
    list of pd.DataFrame
        Each element is a single lineup (DataFrame with the selected players).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    sport = sport.upper()

    if sport == "NBA":
        if slate_type == "showdown":
            solver = _solve_showdown
        else:
            solver = _solve_classic
    elif sport == "PGA":
        solver = _solve_pga
    else:
        raise ValueError(f"Unsupported sport '{sport}'.")

    lineups: list[pd.DataFrame] = []
    excluded_sets: list[frozenset] = []

    for _ in range(n_lineups):
        lineup = solver(pool, cfg, excluded_sets)
        if lineup is None:
            break  # no more feasible lineups
        lineups.append(lineup)
        excluded_sets.append(frozenset(lineup.index))

    return lineups


# ---------------------------------------------------------------------------
# Classic NBA solver (8-man: PG, SG, SF, PF, C, PG/SG, SF/PF, UTIL)
# ---------------------------------------------------------------------------

def _solve_classic(
    pool: pd.DataFrame,
    cfg: Dict[str, Any],
    excluded_sets: list[frozenset],
) -> pd.DataFrame | None:
    """Solve one classic NBA DK lineup.  Returns None if infeasible."""

    prob = pulp.LpProblem("DK_NBA_Classic", pulp.LpMaximize)

    # Binary decision variable per player row
    x = [
        pulp.LpVariable(f"x_{i}", cat="Binary")
        for i in range(len(pool))
    ]

    # ── Objective: maximise total projection ────────────────────────────────
    prob += pulp.lpSum(pool["projection"].iloc[i] * x[i] for i in range(len(pool)))

    # ── Salary cap ──────────────────────────────────────────────────────────
    prob += pulp.lpSum(pool["salary"].iloc[i] * x[i] for i in range(len(pool))) <= SALARY_CAP

    # ── Roster size ─────────────────────────────────────────────────────────
    prob += pulp.lpSum(x) == DK_LINEUP_SIZE

    # ── Position slots ───────────────────────────────────────────────────────
    #   DK_POS_SLOTS = {"PG": 2, "SG": 2, "SF": 2, "PF": 2, "C": 1, "UTIL": 1}
    #   A player can fill their own position slot *or* the UTIL slot.
    for pos, count in DK_POS_SLOTS.items():
        if pos == "UTIL":
            # UTIL can be filled by any non-C player
            eligible = [
                i for i in range(len(pool))
                if pool["position"].iloc[i] != "C"
            ]
            prob += pulp.lpSum(x[i] for i in eligible) >= count
        else:
            eligible = [
                i for i in range(len(pool))
                if pool["position"].iloc[i] == pos
            ]
            prob += pulp.lpSum(x[i] for i in eligible) >= count

    # ── Team stacking (GPP) ──────────────────────────────────────────────────
    if cfg.get("enable_stacking", True):
        min_stack = cfg.get("min_stack_size", 2)
        teams = pool["team"].unique() if "team" in pool.columns else []
        if len(teams) > 0:
            # At least one team must contribute ≥ min_stack players
            team_vars = {
                t: pulp.LpVariable(f"stack_{t}", cat="Binary") for t in teams
            }
            for t in teams:
                team_idx = [
                    i for i in range(len(pool)) if pool["team"].iloc[i] == t
                ]
                prob += (
                    pulp.lpSum(x[i] for i in team_idx)
                    >= min_stack * team_vars[t]
                )
            prob += pulp.lpSum(team_vars[t] for t in teams) >= 1

    # ── Uniqueness constraints ────────────────────────────────────────────────
    for prev in excluded_sets:
        prob += pulp.lpSum(x[i] for i in prev) <= len(prev) - 1

    # ── Max players per team ─────────────────────────────────────────────────
    max_per_team = cfg.get("max_players_per_team", 8)
    if "team" in pool.columns:
        for t in pool["team"].unique():
            team_idx = [i for i in range(len(pool)) if pool["team"].iloc[i] == t]
            prob += pulp.lpSum(x[i] for i in team_idx) <= max_per_team

    # ── Ownership exposure cap (GPP) ─────────────────────────────────────────
    ownership_col = cfg.get("ownership_col", "ownership")
    if ownership_col in pool.columns:
        max_own = cfg.get("max_avg_ownership", 30.0)
        total_own = pulp.lpSum(
            pool[ownership_col].iloc[i] * x[i] for i in range(len(pool))
        )
        prob += total_own <= max_own * DK_LINEUP_SIZE

    # ── Solve ────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    selected = [i for i in range(len(pool)) if pulp.value(x[i]) == 1]
    return pool.iloc[selected].copy()


# ---------------------------------------------------------------------------
# Showdown NBA solver (6-man: 1 CPT + 5 FLEX)
# ---------------------------------------------------------------------------

def _solve_showdown(
    pool: pd.DataFrame,
    cfg: Dict[str, Any],
    excluded_sets: list[frozenset],
) -> pd.DataFrame | None:
    """Solve one DK NBA Showdown lineup.  Returns None if infeasible."""

    cpt_pool = pool[pool["position"] == "CPT"].reset_index(drop=True)
    flex_pool = pool[pool["position"] == "FLEX"].reset_index(drop=True)

    prob = pulp.LpProblem("DK_NBA_Showdown", pulp.LpMaximize)

    xc = [pulp.LpVariable(f"xc_{i}", cat="Binary") for i in range(len(cpt_pool))]
    xf = [pulp.LpVariable(f"xf_{i}", cat="Binary") for i in range(len(flex_pool))]

    # Objective
    prob += (
        pulp.lpSum(
            cpt_pool["projection"].iloc[i] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER * xc[i]
            for i in range(len(cpt_pool))
        )
        + pulp.lpSum(
            flex_pool["projection"].iloc[i] * xf[i]
            for i in range(len(flex_pool))
        )
    )

    # Salary cap (captain salary is 1.5× in DK showdown)
    prob += (
        pulp.lpSum(
            cpt_pool["salary"].iloc[i] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER * xc[i]
            for i in range(len(cpt_pool))
        )
        + pulp.lpSum(
            flex_pool["salary"].iloc[i] * xf[i]
            for i in range(len(flex_pool))
        )
        <= SALARY_CAP
    )

    # Exactly 1 CPT, exactly 5 FLEX
    prob += pulp.lpSum(xc) == DK_SHOWDOWN_SLOTS["CPT"]
    prob += pulp.lpSum(xf) == DK_SHOWDOWN_SLOTS["FLEX"]

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    sel_cpt = cpt_pool.iloc[[i for i in range(len(cpt_pool)) if pulp.value(xc[i]) == 1]].copy()
    sel_flex = flex_pool.iloc[[i for i in range(len(flex_pool)) if pulp.value(xf[i]) == 1]].copy()
    return pd.concat([sel_cpt, sel_flex], ignore_index=True)


# ---------------------------------------------------------------------------
# PGA Classic solver (6-man roster)
# ---------------------------------------------------------------------------

def _solve_pga(
    pool: pd.DataFrame,
    cfg: Dict[str, Any],
    excluded_sets: list[frozenset],
) -> pd.DataFrame | None:
    """Solve one DK PGA Classic lineup.  Returns None if infeasible."""

    prob = pulp.LpProblem("DK_PGA_Classic", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(pool))]

    prob += pulp.lpSum(pool["projection"].iloc[i] * x[i] for i in range(len(pool)))
    prob += pulp.lpSum(pool["salary"].iloc[i] * x[i] for i in range(len(pool))) <= SALARY_CAP
    prob += pulp.lpSum(x) == 6  # DK PGA: 6 golfers

    for prev in excluded_sets:
        prob += pulp.lpSum(x[i] for i in prev) <= len(prev) - 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    selected = [i for i in range(len(pool)) if pulp.value(x[i]) == 1]
    return pool.iloc[selected].copy()


# =============================================================================
# SCORING & ANALYSIS
# =============================================================================

def score_lineup(
    lineup: pd.DataFrame,
    actuals_col: str = "actual_pts",
) -> float:
    """Return the total actual fantasy points for a lineup.

    Parameters
    ----------
    lineup : pd.DataFrame
        A single lineup (output of one element from ``build_lineups``).
    actuals_col : str
        Column containing realised fantasy points.  Defaults to
        ``"actual_pts"``.

    Returns
    -------
    float
        Sum of actual points across all players in the lineup.
    """
    if actuals_col not in lineup.columns:
        raise ValueError(
            f"Column '{actuals_col}' not found in lineup. "
            f"Available: {list(lineup.columns)}"
        )
    return float(lineup[actuals_col].sum())


def score_all(
    lineups: list[pd.DataFrame],
    actuals_col: str = "actual_pts",
) -> pd.Series:
    """Score every lineup and return results as a Series.

    Parameters
    ----------
    lineups : list of pd.DataFrame
    actuals_col : str

    Returns
    -------
    pd.Series
        Index = lineup number (0-based), values = total actual points.
    """
    scores = [score_lineup(lu, actuals_col) for lu in lineups]
    return pd.Series(scores, name="score")


def summarise_lineups(
    lineups: list[pd.DataFrame],
    actuals_col: str = "actual_pts",
) -> pd.DataFrame:
    """Build a summary table of lineup scores and top performers.

    Returns
    -------
    pd.DataFrame
        Columns: ``lineup_idx``, ``score``, ``top_scorer``,
        ``top_scorer_pts``, ``n_players``.
    """
    rows = []
    for i, lu in enumerate(lineups):
        if actuals_col in lu.columns:
            score = float(lu[actuals_col].sum())
            top_row = lu.loc[lu[actuals_col].idxmax()]
            top_name = top_row.get("name", top_row.name)
            top_pts = float(top_row[actuals_col])
        else:
            score = float(lu["projection"].sum())
            top_row = lu.loc[lu["projection"].idxmax()]
            top_name = top_row.get("name", top_row.name)
            top_pts = float(top_row["projection"])
        rows.append(
            {
                "lineup_idx": i,
                "score": score,
                "top_scorer": top_name,
                "top_scorer_pts": top_pts,
                "n_players": len(lu),
            }
        )
    return pd.DataFrame(rows)


def lineup_exposure(
    lineups: list[pd.DataFrame],
) -> pd.DataFrame:
    """Calculate per-player exposure (% of lineups they appear in).

    Returns
    -------
    pd.DataFrame
        Columns: ``name``, ``appearances``, ``exposure_pct``.
        Sorted by ``exposure_pct`` descending.
    """
    from collections import Counter

    counter: Counter = Counter()
    for lu in lineups:
        if "name" in lu.columns:
            counter.update(lu["name"].tolist())

    n = len(lineups) if lineups else 1
    rows = [
        {"name": k, "appearances": v, "exposure_pct": round(v / n * 100, 1)}
        for k, v in counter.most_common()
    ]
    return pd.DataFrame(rows)


def top_value_players(
    pool: pd.DataFrame,
    n: int = 10,
    value_col: str = "value",
) -> pd.DataFrame:
    """Return the top-n players by value (pts per $1k salary).

    Parameters
    ----------
    pool : pd.DataFrame
    n : int
    value_col : str

    Returns
    -------
    pd.DataFrame
    """
    if value_col not in pool.columns:
        raise ValueError(f"Column '{value_col}' not found. Available: {list(pool.columns)}")
    return pool.nlargest(n, value_col).reset_index(drop=True)


# =============================================================================
# BACK-TESTING
# =============================================================================

def backtest(
    sport: str = "NBA",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    n_lineups: int = 20,
    contest_type: str = "gpp",
    slate_type: str = "main",
    config: Dict[str, Any] | None = None,
    yakos_root: str | None = None,
) -> pd.DataFrame:
    """Run a multi-date back-test, returning per-slate score summaries.

    Parameters
    ----------
    sport : str
    start_date, end_date : str
        ISO 8601 date strings (inclusive).
    n_lineups : int
    contest_type : str
    slate_type : str
    config : dict or None
    yakos_root : str or None

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``lineup_idx``, ``score``, ``top_scorer``,
        ``top_scorer_pts``, ``n_players``.
    """
    from datetime import timedelta

    cfg = {**DEFAULT_CONFIG, **(config or {})}

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    current = start

    all_rows: list[pd.DataFrame] = []

    while current <= end:
        slate_str = str(current)
        try:
            raw = load_player_pool(
                sport=sport,
                data_mode="historical",
                slate_date=slate_str,
                yakos_root=yakos_root,
            )
            pool = prepare_pool(
                raw,
                sport=sport,
                contest_type=contest_type,
                slate_type=slate_type,
                config=cfg,
            )
            lineups = build_lineups(
                pool,
                sport=sport,
                n_lineups=n_lineups,
                contest_type=contest_type,
                slate_type=slate_type,
                config=cfg,
            )
            summary = summarise_lineups(lineups)
            summary.insert(0, "date", slate_str)
            all_rows.append(summary)
        except FileNotFoundError:
            pass  # No slate for this date – skip silently
        except Exception as exc:
            # Surface unexpected errors without aborting the whole run
            import warnings
            warnings.warn(f"Back-test error on {slate_str}: {exc}")

        current += timedelta(days=1)

    if not all_rows:
        return pd.DataFrame(
            columns=["date", "lineup_idx", "score", "top_scorer", "top_scorer_pts", "n_players"]
        )
    return pd.concat(all_rows, ignore_index=True)


# =============================================================================
# FORMATTING / EXPORT
# =============================================================================

def format_lineups_for_dk(
    lineups: list[pd.DataFrame],
    sport: str = "NBA",
    slate_type: str = "main",
) -> pd.DataFrame:
    """Format lineups for DraftKings CSV upload.

    Returns a DataFrame where each row is one lineup with player names
    in the columns expected by DK's bulk-upload template.

    Parameters
    ----------
    lineups : list of pd.DataFrame
    sport : str
    slate_type : str

    Returns
    -------
    pd.DataFrame
    """
    sport = sport.upper()

    if sport == "NBA" and slate_type == "main":
        return _format_nba_classic(lineups)
    elif sport == "NBA" and slate_type == "showdown":
        return _format_nba_showdown(lineups)
    elif sport == "PGA":
        return _format_pga(lineups)
    else:
        raise ValueError(f"format_lineups_for_dk: unknown sport/slate combo '{sport}/{slate_type}'.")


def _format_nba_classic(lineups: list[pd.DataFrame]) -> pd.DataFrame:
    """Classic NBA: PG, SG, SF, PF, C, PG, SF, UTIL columns."""
    slot_order = ["PG", "SG", "SF", "PF", "C", "PG", "SF", "UTIL"]
    rows = []

    for lu in lineups:
        slot_counts = {p: 0 for p in ["PG", "SG", "SF", "PF", "C", "UTIL"]}
        row: dict[str, str] = {}
        assigned: set[int] = set()

        for slot in slot_order:
            candidates = lu[
                (lu["position"] == slot) & (~lu.index.isin(assigned))
            ]
            if candidates.empty and slot in ("PG", "SF"):
                # flex slot – pick from UTIL-eligible
                candidates = lu[
                    (lu["position"].isin(["PG", "SG", "SF", "PF"]))
                    & (~lu.index.isin(assigned))
                ]
            if candidates.empty:
                # UTIL fallback
                candidates = lu[~lu.index.isin(assigned)]

            if candidates.empty:
                row[slot] = ""
            else:
                best = candidates.iloc[0]
                row[slot] = str(best.get("name", ""))
                assigned.add(best.name)

        rows.append(row)

    return pd.DataFrame(rows, columns=slot_order)


def _format_nba_showdown(lineups: list[pd.DataFrame]) -> pd.DataFrame:
    """Showdown NBA: CPT + FLEX×5 columns."""
    rows = []
    for lu in lineups:
        cpt = lu[lu["position"] == "CPT"]
        flex = lu[lu["position"] == "FLEX"]
        row = {"CPT": cpt.iloc[0].get("name", "") if not cpt.empty else ""}
        for j, (_, p) in enumerate(flex.iterrows(), 1):
            row[f"FLEX{j}"] = str(p.get("name", ""))
        rows.append(row)
    cols = ["CPT"] + [f"FLEX{j}" for j in range(1, 6)]
    return pd.DataFrame(rows, columns=cols)


def _format_pga(lineups: list[pd.DataFrame]) -> pd.DataFrame:
    """PGA Classic: G1–G6 columns."""
    rows = []
    for lu in lineups:
        row = {f"G{j+1}": str(lu.iloc[j].get("name", "")) for j in range(min(6, len(lu)))}
        rows.append(row)
    return pd.DataFrame(rows, columns=[f"G{j}" for j in range(1, 7)])


# =============================================================================
# CONVENIENCE PIPELINE
# =============================================================================

def run_pipeline(
    sport: str = "NBA",
    data_mode: str = "historical",
    slate_date: str | None = None,
    n_lineups: int = 20,
    contest_type: str = "gpp",
    slate_type: str = "main",
    config: Dict[str, Any] | None = None,
    yakos_root: str | None = None,
) -> Tuple[list[pd.DataFrame], pd.DataFrame]:
    """End-to-end pipeline: load → prepare → optimise → format.

    Parameters
    ----------
    sport : str
    data_mode : str
    slate_date : str or None
    n_lineups : int
    contest_type : str
    slate_type : str
    config : dict or None
    yakos_root : str or None

    Returns
    -------
    tuple
        ``(lineups, dk_upload_df)`` where *lineups* is the raw list of
        lineup DataFrames and *dk_upload_df* is the DK-formatted upload
        table.
    """
    raw = load_player_pool(
        sport=sport,
        data_mode=data_mode,
        slate_date=slate_date,
        yakos_root=yakos_root,
    )
    pool = prepare_pool(
        raw,
        sport=sport,
        contest_type=contest_type,
        slate_type=slate_type,
        config=config,
    )
    lineups = build_lineups(
        pool,
        sport=sport,
        n_lineups=n_lineups,
        contest_type=contest_type,
        slate_type=slate_type,
        config=config,
    )
    dk_df = format_lineups_for_dk(lineups, sport=sport, slate_type=slate_type)
    return lineups, dk_df


# =============================================================================
# DEPRECATED SHIMS (kept for backward compatibility)
# =============================================================================

def load_dk_salaries(*args, **kwargs):
    """Deprecated.  Use ``load_player_pool`` instead."""
    raise NotImplementedError(
        "load_dk_salaries() was removed in the lineups rewrite. "
        "Use yak_core.loading instead."
    )


def export_lineups(*args, **kwargs):
    """Deprecated.  Use ``format_lineups_for_dk`` instead."""
    raise NotImplementedError(
        "export_lineups() was removed in the lineups rewrite. "
        "Use yak_core.publishing instead."
    )


def load_format(*args, **kwargs):
    """Deprecated.  Use ``run_pipeline`` instead."""
    raise NotImplementedError(
        "load_format() was removed in the lineups rewrite. "
        "Use yak_core.publishing instead."
    )
