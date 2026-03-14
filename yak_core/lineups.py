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
        Raw player pool with at least the columns expected by
        :func:`prepare_pool`.
    """
    sport = sport.upper()
    yakos_root = yakos_root or YAKOS_ROOT

    if data_mode == "live":
        return _load_live(sport)
    return _load_historical(sport, slate_date, yakos_root)


def _load_historical(
    sport: str,
    slate_date: str | None,
    yakos_root: str,
) -> pd.DataFrame:
    """Read the local parquet cache for *sport* on *slate_date*."""
    target_date = slate_date or str(date.today())

    if sport == "NBA":
        data_dir = os.path.join(yakos_root, "data", "nba")
        pattern = os.path.join(data_dir, f"players_{target_date}.parquet")
        import glob as _glob
        files = sorted(_glob.glob(pattern))
        if not files:
            # Fall back to the most-recent file available.
            fallback = sorted(
                _glob.glob(os.path.join(data_dir, "players_*.parquet"))
            )
            if not fallback:
                raise FileNotFoundError(
                    f"No NBA player-pool parquet found under {data_dir}"
                )
            files = [fallback[-1]]
        return pd.read_parquet(files[0])

    if sport == "PGA":
        data_dir = os.path.join(yakos_root, "data", "pga")
        pattern = os.path.join(data_dir, f"players_{target_date}.parquet")
        import glob as _glob
        files = sorted(_glob.glob(pattern))
        if not files:
            fallback = sorted(
                _glob.glob(os.path.join(data_dir, "players_*.parquet"))
            )
            if not fallback:
                raise FileNotFoundError(
                    f"No PGA player-pool parquet found under {data_dir}"
                )
            files = [fallback[-1]]
        return pd.read_parquet(files[0])

    raise ValueError(f"Unsupported sport: {sport!r}")


def _load_live(sport: str) -> pd.DataFrame:
    """Fetch today's player pool from Tank01 RapidAPI."""
    if sport == "NBA":
        from yak_core.tank01 import fetch_nba_player_pool
        return fetch_nba_player_pool()
    if sport == "PGA":
        from yak_core.tank01 import fetch_pga_player_pool
        return fetch_pga_player_pool()
    raise ValueError(f"Unsupported sport for live data: {sport!r}")


# =============================================================================
# POOL PREPARATION
# =============================================================================

# Injury statuses that disqualify a player from being included in any lineup.
_INJURY_OUT_STATUSES: frozenset[str] = frozenset({
    "OUT",
    "IR",
    "WD",       # PGA withdrawal
    "Suspended",
})


def prepare_pool(
    df: pd.DataFrame,
    sport: str = "NBA",
    config: Dict[str, Any] | None = None,
    *,
    filter_injury: bool = True,
) -> pd.DataFrame:
    """Clean and enrich the raw player pool.

    Parameters
    ----------
    df : pd.DataFrame
        Raw player pool returned by :func:`load_player_pool`.
    sport : str
        ``"NBA"`` or ``"PGA"`` (case-insensitive).
    config : dict or None
        YakOS run-config dict.  Falls back to ``DEFAULT_CONFIG`` when ``None``.
    filter_injury : bool
        When ``True`` (default) players whose ``injury_status`` column matches
        one of the statuses in ``_INJURY_OUT_STATUSES`` are dropped before any
        further processing.  Set to ``False`` only in testing or when the
        upstream data source has already applied its own injury filter.

    Returns
    -------
    pd.DataFrame
        Enriched, filtered player pool ready to be passed to the optimizer.
    """
    cfg = config or DEFAULT_CONFIG
    sport = sport.upper()

    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Injury filter (new in this merge)
    # ------------------------------------------------------------------
    if filter_injury and "injury_status" in df.columns:
        before = len(df)
        df = df[
            ~df["injury_status"]
            .fillna("")
            .str.strip()
            .isin(_INJURY_OUT_STATUSES)
        ].copy()
        dropped = before - len(df)
        if dropped:
            import logging
            logging.getLogger(__name__).info(
                "prepare_pool: dropped %d player(s) with OUT/IR/WD/Suspended status",
                dropped,
            )

    # ------------------------------------------------------------------
    # 2. Sport-specific normalisation
    # ------------------------------------------------------------------
    if sport == "NBA":
        return _prepare_nba(df, cfg)
    if sport == "PGA":
        return _prepare_pga(df, cfg)
    raise ValueError(f"Unsupported sport: {sport!r}")


def _prepare_nba(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """NBA-specific pool preparation."""
    required = {"name", "position", "salary", "projection"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Player pool missing columns: {missing}")

    # Normalise position strings (e.g. "PG/SG" -> "PG")
    df["position"] = df["position"].str.split("/").str[0].str.strip().str.upper()

    # Salary must be numeric
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df = df.dropna(subset=["salary"])
    df["salary"] = df["salary"].astype(int)

    # Projection must be numeric; drop rows where we have no projection
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce")
    df = df.dropna(subset=["projection"])

    # Apply minimum-salary filter
    min_sal = cfg.get("min_salary", 3000)
    df = df[df["salary"] >= min_sal]

    # Value column (pts per $1 000 of salary)
    df["value"] = df["projection"] / (df["salary"] / 1_000)

    return df.reset_index(drop=True)


def _prepare_pga(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """PGA-specific pool preparation."""
    required = {"name", "salary", "projection"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Player pool missing columns: {missing}")

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df = df.dropna(subset=["salary"])
    df["salary"] = df["salary"].astype(int)

    df["projection"] = pd.to_numeric(df["projection"], errors="coerce")
    df = df.dropna(subset=["projection"])

    min_sal = cfg.get("min_salary", 6000)
    df = df[df["salary"] >= min_sal]

    df["value"] = df["projection"] / (df["salary"] / 1_000)

    return df.reset_index(drop=True)


# =============================================================================
# OPTIMIZER  –  CLASSIC (NBA)
# =============================================================================

def build_lineup(
    pool: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
) -> pd.DataFrame:
    """Build a single DFS lineup using PuLP (classic NBA contest).

    Parameters
    ----------
    pool : pd.DataFrame
        Prepared player pool from :func:`prepare_pool`.
    config : dict or None
        YakOS run-config.  Falls back to ``DEFAULT_CONFIG``.
    locked : list[str] or None
        Player names that *must* appear in the lineup.
    excluded : list[str] or None
        Player names that *must not* appear in the lineup.

    Returns
    -------
    pd.DataFrame
        Single-row-per-player lineup DataFrame.
    """
    cfg = config or DEFAULT_CONFIG
    locked = locked or []
    excluded = excluded or []

    prob = pulp.LpProblem("DFS_Classic", pulp.LpMaximize)
    n = len(pool)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective
    prob += pulp.lpSum(pool["projection"].iloc[i] * x[i] for i in range(n))

    # Salary cap
    prob += (
        pulp.lpSum(pool["salary"].iloc[i] * x[i] for i in range(n)) <= SALARY_CAP
    )

    # Lineup size
    prob += pulp.lpSum(x) == DK_LINEUP_SIZE

    # Position constraints
    for pos, (mn, mx) in DK_POS_SLOTS.items():
        idx = [i for i in range(n) if pool["position"].iloc[i] == pos]
        prob += pulp.lpSum(x[i] for i in idx) >= mn
        prob += pulp.lpSum(x[i] for i in idx) <= mx

    # Locked / excluded
    for name in locked:
        idx = pool.index[pool["name"] == name].tolist()
        for i in idx:
            prob += x[pool.index.get_loc(i)] == 1

    for name in excluded:
        idx = pool.index[pool["name"] == name].tolist()
        for i in idx:
            prob += x[pool.index.get_loc(i)] == 0

    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(
            f"Optimizer did not find an optimal solution: {pulp.LpStatus[prob.status]}"
        )

    selected = [i for i in range(n) if pulp.value(x[i]) == 1]
    return pool.iloc[selected].reset_index(drop=True)


def build_lineups_with_exposure(
    pool: pd.DataFrame,
    n_lineups: int = 20,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
    max_exposure: float = 0.6,
) -> pd.DataFrame:
    """Generate *n_lineups* unique DFS lineups with exposure capping.

    Parameters
    ----------
    pool : pd.DataFrame
        Prepared player pool.
    n_lineups : int
        Number of lineups to generate.
    config : dict or None
        YakOS run-config.
    locked : list[str] or None
        Players locked into every lineup.
    excluded : list[str] or None
        Players excluded from every lineup.
    max_exposure : float
        Maximum fraction of lineups any single player may appear in
        (default 0.60 = 60 %).

    Returns
    -------
    pd.DataFrame
        All lineups stacked, with an added ``lineup_id`` column.
    """
    cfg = config or DEFAULT_CONFIG
    locked = locked or []
    excluded = excluded or []

    exposure: Dict[str, int] = {}
    lineups: list[pd.DataFrame] = []

    for lineup_num in range(n_lineups):
        # Temporarily exclude over-exposed players
        over_exposed = [
            name
            for name, count in exposure.items()
            if count / (lineup_num + 1) > max_exposure
        ]
        current_excluded = excluded + over_exposed

        try:
            lineup = build_lineup(pool, cfg, locked, current_excluded)
        except RuntimeError:
            # Relaxation: drop exposure exclusions if infeasible
            lineup = build_lineup(pool, cfg, locked, excluded)

        lineup["lineup_id"] = lineup_num + 1
        lineups.append(lineup)

        for name in lineup["name"]:
            exposure[name] = exposure.get(name, 0) + 1

    return pd.concat(lineups, ignore_index=True)


# =============================================================================
# OPTIMIZER  –  SHOWDOWN (NBA)
# =============================================================================

def build_showdown_lineup(
    pool: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
) -> pd.DataFrame:
    """Build a single DraftKings Showdown lineup.

    In Showdown contests there is one Captain slot (1.5× salary & projection)
    and five FLEX slots.  The same player can fill the Captain role.

    Parameters
    ----------
    pool : pd.DataFrame
        Prepared player pool (salary-floor already applied).
    config : dict or None
        YakOS run-config.  Falls back to ``DEFAULT_CONFIG``.
    locked : list[str] or None
        Player names that must appear (in any slot).
    excluded : list[str] or None
        Player names that must not appear.

    Returns
    -------
    pd.DataFrame
        Six-row DataFrame: one captain + five FLEX players, with a
        ``slot`` column (``"CPT"`` or ``"FLEX"``).  The captain row
        reflects the 1.5× salary / projection.
    """
    cfg = config or DEFAULT_CONFIG
    locked = locked or []
    excluded = excluded or []

    n = len(pool)

    # Decision variables: c_i = 1 if player i is Captain, f_i = 1 if FLEX
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]
    f = [pulp.LpVariable(f"f_{i}", cat="Binary") for i in range(n)]

    prob = pulp.LpProblem("DFS_Showdown", pulp.LpMaximize)

    # Objective: CPT gets 1.5× projection multiplier
    prob += pulp.lpSum(
        pool["projection"].iloc[i] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER * c[i]
        + pool["projection"].iloc[i] * f[i]
        for i in range(n)
    )

    # Salary cap: CPT salary is also 1.5×
    prob += (
        pulp.lpSum(
            pool["salary"].iloc[i] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER * c[i]
            + pool["salary"].iloc[i] * f[i]
            for i in range(n)
        )
        <= SALARY_CAP
    )

    # Exactly one captain
    prob += pulp.lpSum(c) == DK_SHOWDOWN_SLOTS["CPT"]
    # Exactly five FLEX
    prob += pulp.lpSum(f) == DK_SHOWDOWN_SLOTS["FLEX"]

    # A player can appear at most once total (CPT or FLEX, not both)
    for i in range(n):
        prob += c[i] + f[i] <= 1

    # Locked players must appear somewhere
    for name in locked:
        matches = [i for i in range(n) if pool["name"].iloc[i] == name]
        for i in matches:
            prob += c[i] + f[i] >= 1

    # Excluded players must not appear
    for name in excluded:
        matches = [i for i in range(n) if pool["name"].iloc[i] == name]
        for i in matches:
            prob += c[i] + f[i] == 0

    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(
            f"Showdown optimizer failed: {pulp.LpStatus[prob.status]}"
        )

    rows: list[dict] = []
    for i in range(n):
        if pulp.value(c[i]) == 1:
            row = pool.iloc[i].to_dict()
            row["slot"] = "CPT"
            row["salary"] = int(row["salary"] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER)
            row["projection"] = row["projection"] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER
            rows.append(row)
        elif pulp.value(f[i]) == 1:
            row = pool.iloc[i].to_dict()
            row["slot"] = "FLEX"
            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def build_showdown_lineups_with_exposure(
    pool: pd.DataFrame,
    n_lineups: int = 20,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
    max_exposure: float = 0.6,
) -> pd.DataFrame:
    """Generate multiple Showdown lineups with exposure capping.

    Parameters
    ----------
    pool : pd.DataFrame
        Prepared player pool.
    n_lineups : int
        Number of lineups to generate.
    config : dict or None
        YakOS run-config.
    locked : list[str] or None
        Players locked into every lineup.
    excluded : list[str] or None
        Players excluded from every lineup.
    max_exposure : float
        Max fraction of lineups any player may appear in (0–1).

    Returns
    -------
    pd.DataFrame
        All Showdown lineups stacked with a ``lineup_id`` column.
    """
    cfg = config or DEFAULT_CONFIG
    locked = locked or []
    excluded = excluded or []

    exposure: Dict[str, int] = {}
    lineups: list[pd.DataFrame] = []

    for lineup_num in range(n_lineups):
        over_exposed = [
            name
            for name, count in exposure.items()
            if count / (lineup_num + 1) > max_exposure
        ]
        current_excluded = excluded + over_exposed

        try:
            lineup = build_showdown_lineup(pool, cfg, locked, current_excluded)
        except RuntimeError:
            lineup = build_showdown_lineup(pool, cfg, locked, excluded)

        lineup["lineup_id"] = lineup_num + 1
        lineups.append(lineup)

        for name in lineup["name"]:
            exposure[name] = exposure.get(name, 0) + 1

    return pd.concat(lineups, ignore_index=True)


# =============================================================================
# SHOWDOWN SALARY-FLOOR HELPER
# =============================================================================

def _showdown_min_salary(config: Dict[str, Any] | None = None) -> int:
    """Return the effective per-player salary floor for Showdown slates.

    The floor is read from ``config["showdown_min_salary"]`` when present;
    otherwise it falls back to half of the classic ``min_salary`` value,
    with a hard minimum of 3 000.

    Parameters
    ----------
    config : dict or None
        YakOS run-config.  Uses ``DEFAULT_CONFIG`` when ``None``.

    Returns
    -------
    int
        Minimum salary (in DraftKings salary units).
    """
    cfg = config or DEFAULT_CONFIG
    if "showdown_min_salary" in cfg:
        return int(cfg["showdown_min_salary"])
    fallback = max(3_000, int(cfg.get("min_salary", 3_000)) // 2)
    return fallback


def prepare_showdown_pool(
    df: pd.DataFrame,
    sport: str = "NBA",
    config: Dict[str, Any] | None = None,
    *,
    filter_injury: bool = True,
) -> pd.DataFrame:
    """Prepare a player pool specifically for Showdown contests.

    Applies the same injury filter and numeric normalisation as
    :func:`prepare_pool`, but uses the Showdown-specific salary floor
    (typically lower than the classic floor so that low-salary
    game-script plays remain eligible for FLEX slots).

    Parameters
    ----------
    df : pd.DataFrame
        Raw player pool from :func:`load_player_pool`.
    sport : str
        ``"NBA"`` or ``"PGA"``.
    config : dict or None
        YakOS run-config.
    filter_injury : bool
        Forward to :func:`prepare_pool`.

    Returns
    -------
    pd.DataFrame
        Showdown-ready player pool.
    """
    cfg = config or DEFAULT_CONFIG
    # Inject the Showdown-specific salary floor before delegating.
    showdown_cfg = dict(cfg)
    showdown_cfg["min_salary"] = _showdown_min_salary(cfg)
    return prepare_pool(df, sport=sport, config=showdown_cfg, filter_injury=filter_injury)


# =============================================================================
# MULTI-SLATE HELPERS
# =============================================================================

def load_and_prepare(
    sport: str = "NBA",
    data_mode: str = "historical",
    slate_date: str | None = None,
    config: Dict[str, Any] | None = None,
    yakos_root: str | None = None,
    *,
    filter_injury: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: load + prepare in one call.

    Parameters
    ----------
    sport, data_mode, slate_date, yakos_root
        Forwarded to :func:`load_player_pool`.
    config
        Forwarded to :func:`prepare_pool`.
    filter_injury
        Forwarded to :func:`prepare_pool`.

    Returns
    -------
    pd.DataFrame
        Prepared player pool.
    """
    raw = load_player_pool(
        sport=sport,
        data_mode=data_mode,
        slate_date=slate_date,
        yakos_root=yakos_root,
    )
    return prepare_pool(raw, sport=sport, config=config, filter_injury=filter_injury)


def load_and_prepare_showdown(
    sport: str = "NBA",
    data_mode: str = "historical",
    slate_date: str | None = None,
    config: Dict[str, Any] | None = None,
    yakos_root: str | None = None,
    *,
    filter_injury: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: load + Showdown-prepare in one call."""
    raw = load_player_pool(
        sport=sport,
        data_mode=data_mode,
        slate_date=slate_date,
        yakos_root=yakos_root,
    )
    return prepare_showdown_pool(
        raw, sport=sport, config=config, filter_injury=filter_injury
    )


# =============================================================================
# BATCH LINEUP GENERATION
# =============================================================================

def generate_classic_lineups(
    pool: pd.DataFrame,
    n_lineups: int = 20,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
    max_exposure: float = 0.6,
) -> pd.DataFrame:
    """Thin alias for :func:`build_lineups_with_exposure`.

    Provided so call-sites can import a semantically clear name without
    caring about the internal function naming.
    """
    return build_lineups_with_exposure(
        pool,
        n_lineups=n_lineups,
        config=config,
        locked=locked,
        excluded=excluded,
        max_exposure=max_exposure,
    )


def generate_showdown_lineups(
    pool: pd.DataFrame,
    n_lineups: int = 20,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
    max_exposure: float = 0.6,
) -> pd.DataFrame:
    """Thin alias for :func:`build_showdown_lineups_with_exposure`."""
    return build_showdown_lineups_with_exposure(
        pool,
        n_lineups=n_lineups,
        config=config,
        locked=locked,
        excluded=excluded,
        max_exposure=max_exposure,
    )


# =============================================================================
# LINEUP DIVERSIFICATION  –  CORRELATION / STACK HELPERS
# =============================================================================

def _player_team_map(pool: pd.DataFrame) -> Dict[str, str]:
    """Return a ``{player_name: team}`` mapping from the pool."""
    if "team" not in pool.columns:
        return {}
    return dict(zip(pool["name"], pool["team"]))


def build_stacked_lineup(
    pool: pd.DataFrame,
    stack_team: str,
    stack_size: int = 3,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
) -> pd.DataFrame:
    """Build a single lineup with a mandatory team stack.

    Parameters
    ----------
    pool : pd.DataFrame
        Prepared player pool; must contain a ``team`` column.
    stack_team : str
        Abbreviation of the team to stack (e.g. ``"BOS"``).
    stack_size : int
        Minimum number of players from *stack_team* in the lineup.
    config, locked, excluded
        Forwarded to :func:`build_lineup`.

    Returns
    -------
    pd.DataFrame
        Lineup with at least *stack_size* players from *stack_team*.
    """
    cfg = config or DEFAULT_CONFIG
    locked = list(locked or [])

    if "team" not in pool.columns:
        raise ValueError("Pool must contain a 'team' column for stacking.")

    team_pool = pool[pool["team"] == stack_team]
    if len(team_pool) < stack_size:
        raise ValueError(
            f"Only {len(team_pool)} players available for team {stack_team!r}; "
            f"cannot form a stack of {stack_size}."
        )

    # Lock the top-projected players from the target team
    top_team = (
        team_pool.sort_values("projection", ascending=False)
        .head(stack_size)["name"]
        .tolist()
    )
    locked = list(set(locked) | set(top_team))

    return build_lineup(pool, cfg, locked, excluded)


def build_stacked_lineups(
    pool: pd.DataFrame,
    stack_team: str,
    n_lineups: int = 20,
    stack_size: int = 3,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
    max_exposure: float = 0.6,
) -> pd.DataFrame:
    """Generate multiple stacked lineups.

    Parameters
    ----------
    pool : pd.DataFrame
        Prepared player pool.
    stack_team : str
        Team abbreviation to stack.
    n_lineups : int
        Number of lineups.
    stack_size : int
        Minimum stack depth.
    config, locked, excluded, max_exposure
        Forwarded to :func:`build_lineups_with_exposure` (exposure capping
        is applied globally, not per-stack).

    Returns
    -------
    pd.DataFrame
        Stacked lineups with ``lineup_id`` column.
    """
    cfg = config or DEFAULT_CONFIG
    locked = list(locked or [])
    excluded = list(excluded or [])

    if "team" not in pool.columns:
        raise ValueError("Pool must contain a 'team' column for stacking.")

    team_pool = pool[pool["team"] == stack_team]
    if len(team_pool) < stack_size:
        raise ValueError(
            f"Only {len(team_pool)} players available for team {stack_team!r}."
        )

    top_team = (
        team_pool.sort_values("projection", ascending=False)
        .head(stack_size)["name"]
        .tolist()
    )
    locked = list(set(locked) | set(top_team))

    return build_lineups_with_exposure(
        pool,
        n_lineups=n_lineups,
        config=cfg,
        locked=locked,
        excluded=excluded,
        max_exposure=max_exposure,
    )


# =============================================================================
# SIMULATION / MONTE-CARLO
# =============================================================================

def simulate_lineup_scores(
    lineup: pd.DataFrame,
    n_sims: int = 10_000,
    noise_std: float = 5.0,
    seed: int | None = None,
) -> np.ndarray:
    """Monte-Carlo score simulation for a single lineup.

    Adds Gaussian noise to each player's projection for *n_sims* trials
    and returns the simulated total lineup scores.

    Parameters
    ----------
    lineup : pd.DataFrame
        Lineup DataFrame with a ``projection`` column.
    n_sims : int
        Number of simulation iterations.
    noise_std : float
        Standard deviation of the per-player Gaussian noise.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_sims,)
        Simulated total lineup scores.
    """
    rng = np.random.default_rng(seed)
    projs = lineup["projection"].values  # shape (k,)
    noise = rng.normal(0, noise_std, size=(n_sims, len(projs)))
    scores = (projs + noise).sum(axis=1)
    return scores


def simulate_portfolio(
    lineups: pd.DataFrame,
    n_sims: int = 10_000,
    noise_std: float = 5.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Monte-Carlo simulation across a portfolio of lineups.

    Parameters
    ----------
    lineups : pd.DataFrame
        Stacked lineup DataFrame with a ``lineup_id`` column.
    n_sims, noise_std, seed
        Forwarded to :func:`simulate_lineup_scores`.

    Returns
    -------
    pd.DataFrame
        Summary stats per ``lineup_id``: mean, std, p10, p50, p90 scores.
    """
    results = []
    for lid, grp in lineups.groupby("lineup_id"):
        scores = simulate_lineup_scores(grp, n_sims=n_sims, noise_std=noise_std, seed=seed)
        results.append(
            {
                "lineup_id": lid,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "p10": float(np.percentile(scores, 10)),
                "p50": float(np.percentile(scores, 50)),
                "p90": float(np.percentile(scores, 90)),
            }
        )
    return pd.DataFrame(results)


# =============================================================================
# CONTEST / SCORING UTILITIES
# =============================================================================

def score_lineup(
    lineup: pd.DataFrame,
    actuals: pd.DataFrame,
    score_col: str = "actual_points",
) -> float:
    """Score a lineup against actual player results.

    Parameters
    ----------
    lineup : pd.DataFrame
        Lineup DataFrame with a ``name`` column.
    actuals : pd.DataFrame
        Actual results DataFrame with ``name`` and *score_col* columns.
    score_col : str
        Column in *actuals* containing the actual fantasy points.

    Returns
    -------
    float
        Total actual fantasy points for the lineup.
    """
    merged = lineup[["name"]].merge(actuals[["name", score_col]], on="name", how="left")
    return float(merged[score_col].fillna(0).sum())


def score_portfolio(
    lineups: pd.DataFrame,
    actuals: pd.DataFrame,
    score_col: str = "actual_points",
) -> pd.DataFrame:
    """Score every lineup in a portfolio against actual results.

    Parameters
    ----------
    lineups : pd.DataFrame
        Stacked lineup DataFrame with ``lineup_id`` and ``name`` columns.
    actuals : pd.DataFrame
        Actual results with ``name`` and *score_col* columns.
    score_col : str
        Column in *actuals* for actual fantasy points.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``lineup_id`` and ``total_score`` columns, sorted
        descending by score.
    """
    rows = []
    for lid, grp in lineups.groupby("lineup_id"):
        rows.append(
            {"lineup_id": lid, "total_score": score_lineup(grp, actuals, score_col)}
        )
    return (
        pd.DataFrame(rows)
        .sort_values("total_score", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# EXPORT / SERIALIZATION
# =============================================================================

def lineups_to_csv(
    lineups: pd.DataFrame,
    output_path: str,
    *,
    index: bool = False,
) -> str:
    """Write *lineups* to a CSV file.

    Parameters
    ----------
    lineups : pd.DataFrame
        Lineup DataFrame to export.
    output_path : str
        Destination file path (will be created or overwritten).
    index : bool
        Whether to write the DataFrame index.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    lineups.to_csv(output_path, index=index)
    return os.path.abspath(output_path)


def lineups_to_json(
    lineups: pd.DataFrame,
    output_path: str,
    orient: str = "records",
) -> str:
    """Write *lineups* to a JSON file.

    Parameters
    ----------
    lineups : pd.DataFrame
    output_path : str
    orient : str
        pandas ``to_json`` orient string (default ``"records"``).

    Returns
    -------
    str
        Absolute path of the written file.
    """
    lineups.to_json(output_path, orient=orient, indent=2)
    return os.path.abspath(output_path)


def lineups_to_parquet(
    lineups: pd.DataFrame,
    output_path: str,
) -> str:
    """Write *lineups* to a Parquet file."""
    lineups.to_parquet(output_path, index=False)
    return os.path.abspath(output_path)


# =============================================================================
# RUN-CONFIG BUILDER
# =============================================================================

def build_run_config(
    sport: str = "NBA",
    n_lineups: int = 20,
    max_exposure: float = 0.6,
    salary_cap: int | None = None,
    min_salary: int = 3_000,
    showdown_min_salary: int | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
    stack_team: str | None = None,
    stack_size: int = 3,
    data_mode: str = "historical",
    slate_date: str | None = None,
    output_format: str = "csv",
    output_path: str = "lineups_output.csv",
    **extra,
) -> Dict[str, Any]:
    """Construct a validated YakOS run-config dictionary.

    This is the canonical way to build a config for all optimizer
    entry-points.  Unknown keyword arguments are accepted and stored
    verbatim so that caller code can attach ad-hoc metadata without
    breaking validation.

    Parameters
    ----------
    sport : str
    n_lineups : int
    max_exposure : float  0–1
    salary_cap : int or None   override for ``SALARY_CAP``
    min_salary : int
    showdown_min_salary : int or None
    locked, excluded : list[str]
    stack_team : str or None
    stack_size : int
    data_mode : str  ``"historical"`` | ``"live"``
    slate_date : str or None
    output_format : str  ``"csv"`` | ``"json"`` | ``"parquet"``
    output_path : str
    **extra : any
        Additional key-value pairs stored verbatim.

    Returns
    -------
    dict
        Merged config ready for all ``build_*`` / ``generate_*`` functions.
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "sport": sport.upper(),
            "n_lineups": n_lineups,
            "max_exposure": max_exposure,
            "min_salary": min_salary,
            "locked": list(locked or []),
            "excluded": list(excluded or []),
            "stack_team": stack_team,
            "stack_size": stack_size,
            "data_mode": data_mode,
            "slate_date": slate_date,
            "output_format": output_format,
            "output_path": output_path,
        }
    )
    if salary_cap is not None:
        cfg["salary_cap"] = salary_cap
    if showdown_min_salary is not None:
        cfg["showdown_min_salary"] = showdown_min_salary
    cfg.update(extra)
    return cfg


# =============================================================================
# TOP-LEVEL ENTRY POINTS
# =============================================================================

def run_classic(
    config: Dict[str, Any] | None = None,
    *,
    pool: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full classic-contest pipeline.

    Loads (or accepts) a player pool, prepares it, generates lineups,
    and writes the output to the path specified in *config*.

    Parameters
    ----------
    config : dict or None
        YakOS run-config built by :func:`build_run_config`.
    pool : pd.DataFrame or None
        Pre-loaded + prepared pool (skips load/prepare if supplied).

    Returns
    -------
    (lineups_df, summary_df)
        The raw stacked lineups and a per-lineup summary.
    """
    cfg = config or DEFAULT_CONFIG

    if pool is None:
        pool = load_and_prepare(
            sport=cfg.get("sport", "NBA"),
            data_mode=cfg.get("data_mode", "historical"),
            slate_date=cfg.get("slate_date"),
            config=cfg,
        )

    lineups = generate_classic_lineups(
        pool,
        n_lineups=cfg.get("n_lineups", 20),
        config=cfg,
        locked=cfg.get("locked"),
        excluded=cfg.get("excluded"),
        max_exposure=cfg.get("max_exposure", 0.6),
    )

    summary = score_portfolio(lineups, pool)  # "actuals" = projections for pre-contest

    fmt = cfg.get("output_format", "csv")
    path = cfg.get("output_path", "lineups_output.csv")
    if fmt == "csv":
        lineups_to_csv(lineups, path)
    elif fmt == "json":
        lineups_to_json(lineups, path)
    elif fmt == "parquet":
        lineups_to_parquet(lineups, path)

    return lineups, summary


def run_showdown(
    config: Dict[str, Any] | None = None,
    *,
    pool: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full Showdown-contest pipeline.

    Parameters
    ----------
    config : dict or None
        YakOS run-config.  ``showdown_min_salary`` is honoured if present.
    pool : pd.DataFrame or None
        Pre-loaded + Showdown-prepared pool (skips load/prepare if supplied).

    Returns
    -------
    (lineups_df, summary_df)
    """
    cfg = config or DEFAULT_CONFIG

    if pool is None:
        pool = load_and_prepare_showdown(
            sport=cfg.get("sport", "NBA"),
            data_mode=cfg.get("data_mode", "historical"),
            slate_date=cfg.get("slate_date"),
            config=cfg,
        )

    lineups = generate_showdown_lineups(
        pool,
        n_lineups=cfg.get("n_lineups", 20),
        config=cfg,
        locked=cfg.get("locked"),
        excluded=cfg.get("excluded"),
        max_exposure=cfg.get("max_exposure", 0.6),
    )

    summary = score_portfolio(lineups, pool)

    fmt = cfg.get("output_format", "csv")
    path = cfg.get("output_path", "lineups_output.csv")
    if fmt == "csv":
        lineups_to_csv(lineups, path)
    elif fmt == "json":
        lineups_to_json(lineups, path)
    elif fmt == "parquet":
        lineups_to_parquet(lineups, path)

    return lineups, summary


# =============================================================================
# PORTFOLIO ANALYTICS
# =============================================================================

def portfolio_ownership(
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-player ownership across a portfolio.

    Parameters
    ----------
    lineups : pd.DataFrame
        Stacked lineups with ``lineup_id`` and ``name`` columns.

    Returns
    -------
    pd.DataFrame
        ``name`` and ``ownership`` (fraction, 0–1) columns, sorted
        descending.
    """
    n_lineups = lineups["lineup_id"].nunique()
    counts = lineups.groupby("name").size().reset_index(name="appearances")
    counts["ownership"] = counts["appearances"] / n_lineups
    return counts.sort_values("ownership", ascending=False).reset_index(drop=True)


def portfolio_correlation(
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    """Compute a player-pair co-occurrence matrix across a portfolio.

    Parameters
    ----------
    lineups : pd.DataFrame
        Stacked lineups with ``lineup_id`` and ``name`` columns.

    Returns
    -------
    pd.DataFrame
        Square DataFrame (players × players) with the fraction of lineups
        in which each pair co-occurs.
    """
    n_lineups = lineups["lineup_id"].nunique()
    players = sorted(lineups["name"].unique())
    cooc = pd.DataFrame(0, index=players, columns=players, dtype=float)

    for _, grp in lineups.groupby("lineup_id"):
        names = grp["name"].tolist()
        for p1 in names:
            for p2 in names:
                cooc.loc[p1, p2] += 1

    return cooc / n_lineups


def top_lineups(
    lineups: pd.DataFrame,
    actuals: pd.DataFrame,
    top_n: int = 5,
    score_col: str = "actual_points",
) -> pd.DataFrame:
    """Return the top-N lineups by actual score.

    Parameters
    ----------
    lineups : pd.DataFrame
    actuals : pd.DataFrame
    top_n : int
    score_col : str

    Returns
    -------
    pd.DataFrame
        Stacked DataFrame of the top-N lineups, with ``total_score``
        attached to every row.
    """
    scored = score_portfolio(lineups, actuals, score_col)
    top_ids = scored.head(top_n)["lineup_id"].tolist()
    top = lineups[lineups["lineup_id"].isin(top_ids)].copy()
    top = top.merge(scored[["lineup_id", "total_score"]], on="lineup_id", how="left")
    return top.sort_values(["total_score", "lineup_id"], ascending=[False, True]).reset_index(
        drop=True
    )


# =============================================================================
# POSITION / ROSTER VALIDATION
# =============================================================================

def validate_lineup(
    lineup: pd.DataFrame,
    contest_type: str = "classic",
) -> Tuple[bool, list[str]]:
    """Validate a lineup against DraftKings roster rules.

    Parameters
    ----------
    lineup : pd.DataFrame
        Lineup to validate; must have ``position``, ``salary``, and
        (for showdown) ``slot`` columns.
    contest_type : str
        ``"classic"`` or ``"showdown"``.

    Returns
    -------
    (is_valid, errors)
        ``is_valid`` is ``True`` only when *errors* is empty.
    """
    errors: list[str] = []

    if contest_type == "showdown":
        if len(lineup) != DK_SHOWDOWN_LINEUP_SIZE:
            errors.append(
                f"Showdown lineup has {len(lineup)} players; expected {DK_SHOWDOWN_LINEUP_SIZE}."
            )
        if "slot" in lineup.columns:
            cpt_count = (lineup["slot"] == "CPT").sum()
            if cpt_count != 1:
                errors.append(f"Expected 1 CPT slot, found {cpt_count}.")
        total_sal = lineup["salary"].sum()
        if total_sal > SALARY_CAP:
            errors.append(f"Salary {total_sal:,} exceeds cap {SALARY_CAP:,}.")
        return not errors, errors

    # Classic
    if len(lineup) != DK_LINEUP_SIZE:
        errors.append(
            f"Classic lineup has {len(lineup)} players; expected {DK_LINEUP_SIZE}."
        )
    total_sal = lineup["salary"].sum()
    if total_sal > SALARY_CAP:
        errors.append(f"Salary {total_sal:,} exceeds cap {SALARY_CAP:,}.")

    for pos, (mn, _mx) in DK_POS_SLOTS.items():
        count = (lineup["position"] == pos).sum()
        if count < mn:
            errors.append(f"Position {pos}: need at least {mn}, have {count}.")

    return not errors, errors


def validate_portfolio(
    lineups: pd.DataFrame,
    contest_type: str = "classic",
) -> pd.DataFrame:
    """Validate every lineup in a portfolio.

    Returns
    -------
    pd.DataFrame
        Columns: ``lineup_id``, ``is_valid``, ``errors``.
    """
    rows = []
    for lid, grp in lineups.groupby("lineup_id"):
        is_valid, errors = validate_lineup(grp, contest_type=contest_type)
        rows.append({"lineup_id": lid, "is_valid": is_valid, "errors": errors})
    return pd.DataFrame(rows)


# =============================================================================
# PLAYER / POOL UTILITIES
# =============================================================================

def filter_pool(
    pool: pd.DataFrame,
    min_salary: int | None = None,
    max_salary: int | None = None,
    min_projection: float | None = None,
    positions: list[str] | None = None,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """Apply ad-hoc filters to a prepared pool.

    All parameters are optional; only specified constraints are applied.

    Parameters
    ----------
    pool : pd.DataFrame
    min_salary, max_salary : int or None
    min_projection : float or None
    positions : list[str] or None
    teams : list[str] or None  (requires ``team`` column)

    Returns
    -------
    pd.DataFrame
        Filtered pool, reset index.
    """
    df = pool.copy()
    if min_salary is not None:
        df = df[df["salary"] >= min_salary]
    if max_salary is not None:
        df = df[df["salary"] <= max_salary]
    if min_projection is not None:
        df = df[df["projection"] >= min_projection]
    if positions:
        df = df[df["position"].isin(positions)]
    if teams and "team" in df.columns:
        df = df[df["team"].isin(teams)]
    return df.reset_index(drop=True)


def pool_summary(
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Compute summary statistics for a prepared player pool.

    Returns
    -------
    pd.DataFrame
        Stats grouped by ``position``: count, mean salary, mean projection,
        mean value.
    """
    cols = ["position", "salary", "projection", "value"]
    present = [c for c in cols if c in pool.columns]
    grp = pool[present].groupby("position")
    agg = grp.agg(["count", "mean"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


# =============================================================================
# INJURY / AVAILABILITY HELPERS
# =============================================================================

def flag_injured(
    pool: pd.DataFrame,
    statuses: list[str] | None = None,
) -> pd.DataFrame:
    """Add a boolean ``is_injured`` column to the pool.

    Parameters
    ----------
    pool : pd.DataFrame
        Must contain an ``injury_status`` column.
    statuses : list[str] or None
        Statuses to flag as injured.  Defaults to
        ``list(_INJURY_OUT_STATUSES)``.

    Returns
    -------
    pd.DataFrame
        Pool with ``is_injured`` column added (does not drop any rows).
    """
    statuses = statuses or list(_INJURY_OUT_STATUSES)
    if "injury_status" not in pool.columns:
        pool = pool.copy()
        pool["is_injured"] = False
        return pool
    pool = pool.copy()
    pool["is_injured"] = (
        pool["injury_status"].fillna("").str.strip().isin(statuses)
    )
    return pool


def available_pool(
    pool: pd.DataFrame,
    statuses: list[str] | None = None,
) -> pd.DataFrame:
    """Return only players not flagged with an out-type injury status.

    Thin convenience wrapper around :func:`flag_injured` + boolean filter.

    Parameters
    ----------
    pool : pd.DataFrame
    statuses : list[str] or None
        Defaults to ``list(_INJURY_OUT_STATUSES)``.

    Returns
    -------
    pd.DataFrame
        Filtered pool, index reset.
    """
    flagged = flag_injured(pool, statuses)
    return flagged[~flagged["is_injured"]].drop(columns="is_injured").reset_index(drop=True)


# =============================================================================
# CONFIG INTROSPECTION
# =============================================================================

def describe_config(config: Dict[str, Any] | None = None) -> None:
    """Pretty-print a run-config to stdout."""
    import pprint
    cfg = config or DEFAULT_CONFIG
    pprint.pprint(cfg)


def config_diff(
    base: Dict[str, Any] | None = None,
    override: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return keys in *override* that differ from *base*.

    Parameters
    ----------
    base : dict or None  → ``DEFAULT_CONFIG``
    override : dict or None  → ``DEFAULT_CONFIG``

    Returns
    -------
    dict
        ``{key: (base_value, override_value)}`` for every differing key.
    """
    b = base or DEFAULT_CONFIG
    o = override or DEFAULT_CONFIG
    diff = {}
    all_keys = set(b) | set(o)
    for k in all_keys:
        bv = b.get(k)
        ov = o.get(k)
        if bv != ov:
            diff[k] = (bv, ov)
    return diff


# =============================================================================
# SLATE METADATA
# =============================================================================

def slate_info(
    pool: pd.DataFrame,
) -> Dict[str, Any]:
    """Return a metadata dict describing the current slate.

    Parameters
    ----------
    pool : pd.DataFrame
        Prepared player pool.

    Returns
    -------
    dict
        Keys: ``n_players``, ``n_teams``, ``salary_range``,
        ``projection_range``, ``positions``.
    """
    info: Dict[str, Any] = {}
    info["n_players"] = len(pool)
    if "team" in pool.columns:
        info["n_teams"] = pool["team"].nunique()
    info["salary_range"] = (
        int(pool["salary"].min()),
        int(pool["salary"].max()),
    )
    info["projection_range"] = (
        float(pool["projection"].min()),
        float(pool["projection"].max()),
    )
    if "position" in pool.columns:
        info["positions"] = sorted(pool["position"].unique().tolist())
    return info


# =============================================================================
# BACKWARD-COMPAT SHIMS  (deprecated – will be removed in a future release)
# =============================================================================

def get_player_pool(
    sport: str = "NBA",
    data_mode: str = "historical",
    slate_date: str | None = None,
    yakos_root: str | None = None,
) -> pd.DataFrame:
    """Deprecated alias for :func:`load_player_pool`.

    .. deprecated::
        Use :func:`load_player_pool` directly.
    """
    import warnings
    warnings.warn(
        "get_player_pool() is deprecated; use load_player_pool() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_player_pool(
        sport=sport,
        data_mode=data_mode,
        slate_date=slate_date,
        yakos_root=yakos_root,
    )


def prep_pool(
    df: pd.DataFrame,
    sport: str = "NBA",
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Deprecated alias for :func:`prepare_pool`.

    .. deprecated::
        Use :func:`prepare_pool` directly.
    """
    import warnings
    warnings.warn(
        "prep_pool() is deprecated; use prepare_pool() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return prepare_pool(df, sport=sport, config=config)


def optimize(
    pool: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
) -> pd.DataFrame:
    """Deprecated alias for :func:`build_lineup`.

    .. deprecated::
        Use :func:`build_lineup` directly.
    """
    import warnings
    warnings.warn(
        "optimize() is deprecated; use build_lineup() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_lineup(pool, config=config, locked=locked, excluded=excluded)


def optimize_with_exposure(
    pool: pd.DataFrame,
    n_lineups: int = 20,
    config: Dict[str, Any] | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
    max_exposure: float = 0.6,
) -> pd.DataFrame:
    """Deprecated alias for :func:`build_lineups_with_exposure`.

    .. deprecated::
        Use :func:`build_lineups_with_exposure` directly.
    """
    import warnings
    warnings.warn(
        "optimize_with_exposure() is deprecated; "
        "use build_lineups_with_exposure() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_lineups_with_exposure(
        pool,
        n_lineups=n_lineups,
        config=config,
        locked=locked,
        excluded=excluded,
        max_exposure=max_exposure,
    )


def run_lineups_from_config(
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Deprecated stub – raises RuntimeError.

    In the old codebase ``run_lineups_from_config`` was the single
    entry-point.  It has been replaced by the more explicit
    :func:`run_classic` and :func:`run_showdown` functions.  Call
    :func:`build_lineups_with_exposure` + :func:`build_run_config` or
    :func:`run_classic`/:func:`run_showdown` to migrate call-sites that
    previously called :func:`build_lineups_with_exposure` directly."""
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
