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
        Raw player-pool frame from the source (columns vary by sport/mode).
    """
    sport = sport.upper()
    root = yakos_root or YAKOS_ROOT

    if data_mode == "historical":
        ds = slate_date or str(date.today())
        if sport == "NBA":
            path = os.path.join(root, "data", "nba", "pool", f"{ds}.parquet")
        elif sport == "PGA":
            path = os.path.join(root, "data", "pga", "pool", f"{ds}.parquet")
        else:
            raise ValueError(f"Unsupported sport: {sport}")
        return pd.read_parquet(path)

    elif data_mode == "live":
        if sport == "NBA":
            from yak_core.ingest_nba import fetch_nba_pool
            return fetch_nba_pool()
        elif sport == "PGA":
            from yak_core.ingest_pga import fetch_pga_pool
            return fetch_pga_pool()
        else:
            raise ValueError(f"Unsupported sport: {sport}")

    else:
        raise ValueError(f"Unknown data_mode: {data_mode!r}")


# =============================================================================
# PLAYER POOL PREPARATION
# =============================================================================

def prepare_pool(
    df: pd.DataFrame,
    sport: str = "NBA",
    contest_type: str = "classic",
    cfg: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalise and filter the raw player pool.

    Parameters
    ----------
    df : pd.DataFrame
        Raw frame from :func:`load_player_pool`.
    sport : str
        ``"NBA"`` or ``"PGA"``.
    contest_type : str
        ``"classic"`` (default) or ``"showdown"``.
    cfg : dict or None
        Configuration overrides; merged on top of ``DEFAULT_CONFIG``.

    Returns
    -------
    pd.DataFrame
        Cleaned, filtered player pool ready for the optimiser.
    """
    config = {**DEFAULT_CONFIG, **(cfg or {})}
    sport = sport.upper()

    # ── Column rename: normalise to canonical names ───────────────────────────
    rename_map: dict[str, str] = {}
    if sport == "NBA":
        rename_map = {
            "Name": "name",
            "Roster Position": "pos",
            "Position": "position",
            "Salary": "salary",
            "TeamAbbrev": "team",
            "AvgPointsPerGame": "avg_pts",
            "Game Info": "game_info",
            "ID": "dk_id",
        }
    elif sport == "PGA":
        rename_map = {
            "Name": "name",
            "Roster Position": "pos",
            "Position": "position",
            "Salary": "salary",
            "TeamAbbrev": "team",
            "AvgPointsPerGame": "avg_pts",
            "Game Info": "game_info",
            "ID": "dk_id",
        }

    df = df.rename(columns=rename_map)

    # ── Deduplicate columns ───────────────────────────────────────────────────
    # Guard against duplicate column names (e.g. both 'pos' and 'position'
    # mapping to 'position' after the rename above).  If two columns share a
    # name, df['col'] returns a DataFrame instead of a Series, which crashes
    # any subsequent .str accessor.  Keep only the first occurrence.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # ── Canonical position: prefer 'position', fall back to 'pos' ─────────────
    if "position" not in df.columns and "pos" in df.columns:
        df = df.rename(columns={"pos": "position"})

    # ── Numeric coercion ──────────────────────────────────────────────────────
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["avg_pts"] = pd.to_numeric(df["avg_pts"], errors="coerce")

    # ── Drop rows missing essentials ──────────────────────────────────────────
    df = df.dropna(subset=["name", "salary", "position"])
    df = df[df["salary"] > 0]

    # ── Showdown: duplicate each player as CPT and FLEX ───────────────────────
    if contest_type == "showdown":
        cpt = df.copy()
        cpt["slot"] = "CPT"
        cpt["salary"] = (cpt["salary"] * DK_SHOWDOWN_CAPTAIN_MULTIPLIER).round(-2)
        cpt["proj"] = cpt.get("proj", cpt["avg_pts"]) * DK_SHOWDOWN_CAPTAIN_MULTIPLIER
        flex = df.copy()
        flex["slot"] = "FLEX"
        flex["proj"] = flex.get("proj", flex["avg_pts"])
        df = pd.concat([cpt, flex], ignore_index=True)
    else:
        if "proj" not in df.columns:
            df["proj"] = df["avg_pts"]

    # ── Sport-specific position normalisation ─────────────────────────────────
    if sport == "NBA":
        df = _normalise_nba_positions(df, contest_type, config)
    elif sport == "PGA":
        df = _normalise_pga_positions(df, contest_type, config)

    df = df.reset_index(drop=True)
    return df


# =============================================================================
# POSITION NORMALISATION HELPERS
# =============================================================================

def _normalise_nba_positions(
    df: pd.DataFrame,
    contest_type: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Expand multi-position eligibility for NBA classic/showdown slates."""
    if contest_type == "showdown":
        # Showdown only uses CPT/FLEX slots — positional eligibility is uniform.
        return df

    # DraftKings exports 'Roster Position' as slash-separated eligibility,
    # e.g. "PG/SG" means the player can fill either slot.  We explode each
    # player into one row per eligible position.
    if "position" not in df.columns:
        return df

    df["position"] = df["position"].astype(str)
    rows = []
    for _, row in df.iterrows():
        positions = [p.strip() for p in row["position"].split("/")]
        for pos in positions:
            new_row = row.copy()
            new_row["position"] = pos
            rows.append(new_row)

    if not rows:
        return df

    expanded = pd.DataFrame(rows).reset_index(drop=True)
    return expanded


def _normalise_pga_positions(
    df: pd.DataFrame,
    contest_type: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """PGA pools use a single 'G' position — nothing to explode."""
    if "position" in df.columns:
        df["position"] = df["position"].str.upper().str.strip()
    return df


# =============================================================================
# PROJECTION HELPERS
# =============================================================================

def apply_projections(
    df: pd.DataFrame,
    proj: pd.Series | Dict[str, float],
) -> pd.DataFrame:
    """Overwrite the ``proj`` column with user-supplied projections.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared pool (output of :func:`prepare_pool`).
    proj : pd.Series or dict
        Mapping of player name → projected points.  Players whose names
        are not found in *proj* retain their existing ``proj`` value.

    Returns
    -------
    pd.DataFrame
        Pool with updated projections.
    """
    if isinstance(proj, dict):
        proj = pd.Series(proj)
    df = df.copy()
    df["proj"] = df["name"].map(proj).fillna(df["proj"])
    return df


def apply_ownership(
    df: pd.DataFrame,
    own: pd.Series | Dict[str, float],
) -> pd.DataFrame:
    """Attach projected-ownership percentages to the pool.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared pool.
    own : pd.Series or dict
        Mapping of player name → projected ownership (0–100 scale).

    Returns
    -------
    pd.DataFrame
        Pool with a ``own_pct`` column added / overwritten.
    """
    if isinstance(own, dict):
        own = pd.Series(own)
    df = df.copy()
    df["own_pct"] = df["name"].map(own).fillna(0.0)
    return df


# =============================================================================
# LINEUP OPTIMIZER
# =============================================================================

def build_lineups(
    df: pd.DataFrame,
    sport: str = "NBA",
    contest_type: str = "classic",
    n: int = 20,
    cfg: Dict[str, Any] | None = None,
) -> list[pd.DataFrame]:
    """Generate *n* unique DFS lineups via integer-linear programming.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared pool (output of :func:`prepare_pool`).  Must contain
        columns: ``name``, ``position``, ``salary``, ``proj``, ``team``.
    sport : str
        ``"NBA"`` or ``"PGA"``.
    contest_type : str
        ``"classic"`` or ``"showdown"``.
    n : int
        Number of lineups to generate (default 20).
    cfg : dict or None
        Config overrides merged on top of ``DEFAULT_CONFIG``.

    Returns
    -------
    list of pd.DataFrame
        Each element is a single lineup DataFrame (one row per player slot).
    """
    config = {**DEFAULT_CONFIG, **(cfg or {})}
    sport = sport.upper()

    if contest_type == "showdown":
        return _build_showdown_lineups(df, sport, n, config)
    else:
        return _build_classic_lineups(df, sport, n, config)


# ── Classic optimizer ─────────────────────────────────────────────────────────

def _build_classic_lineups(
    df: pd.DataFrame,
    sport: str,
    n: int,
    config: Dict[str, Any],
) -> list[pd.DataFrame]:
    """ILP optimizer for classic (non-showdown) contests."""
    lineups: list[pd.DataFrame] = []
    excluded_sets: list[frozenset] = []

    pos_slots = DK_POS_SLOTS  # e.g. {"PG": 1, "SG": 1, ..., "UTIL": 1}
    lineup_size = DK_LINEUP_SIZE

    min_teams = config.get("min_teams", 2)
    max_from_team = config.get("max_from_team", 8)
    min_salary = config.get("min_salary", 49000)
    max_salary = SALARY_CAP
    exposure_cap = config.get("exposure_cap", 1.0)

    # Pre-compute: for each positional slot, which player-rows are eligible?
    slot_eligibility: dict[str, list[int]] = {
        slot: df.index[df["position"] == slot].tolist()
        for slot in pos_slots
    }

    teams = df["team"].unique().tolist()

    for lineup_idx in range(n):
        prob = pulp.LpProblem(f"lineup_{lineup_idx}", pulp.LpMaximize)

        # Decision variables: x[i] = 1 if player i is in the lineup
        x = pulp.LpVariable.dicts("x", df.index, cat="Binary")

        # ── Objective ─────────────────────────────────────────────────────────
        prob += pulp.lpSum(df.loc[i, "proj"] * x[i] for i in df.index)

        # ── Lineup size ───────────────────────────────────────────────────────
        prob += pulp.lpSum(x[i] for i in df.index) == lineup_size

        # ── Salary constraints ────────────────────────────────────────────────
        prob += pulp.lpSum(df.loc[i, "salary"] * x[i] for i in df.index) <= max_salary
        prob += pulp.lpSum(df.loc[i, "salary"] * x[i] for i in df.index) >= min_salary

        # ── Positional constraints ────────────────────────────────────────────
        for slot, count in pos_slots.items():
            eligible = slot_eligibility.get(slot, [])
            prob += pulp.lpSum(x[i] for i in eligible) == count

        # ── Team constraints ──────────────────────────────────────────────────
        for team in teams:
            team_idx = df.index[df["team"] == team].tolist()
            prob += pulp.lpSum(x[i] for i in team_idx) <= max_from_team

        # min_teams: we need players from at least min_teams distinct teams
        # Enforce via binary team indicators t[team] = 1 if ≥1 player from team
        t = pulp.LpVariable.dicts("t", teams, cat="Binary")
        for team in teams:
            team_idx = df.index[df["team"] == team].tolist()
            if team_idx:
                prob += t[team] <= pulp.lpSum(x[i] for i in team_idx)
                prob += (
                    pulp.lpSum(x[i] for i in team_idx)
                    <= len(team_idx) * t[team]
                )
        prob += pulp.lpSum(t[team] for team in teams) >= min_teams

        # ── Exposure cap ──────────────────────────────────────────────────────
        if exposure_cap < 1.0 and lineups:
            max_appearances = int(exposure_cap * n)
            for i in df.index:
                appearances = sum(
                    1 for prev in lineups
                    if i in prev.index
                )
                if appearances >= max_appearances:
                    prob += x[i] == 0

        # ── Uniqueness: exclude previously generated lineups ──────────────────
        for excl in excluded_sets:
            prob += pulp.lpSum(x[i] for i in excl) <= lineup_size - 1

        # ── Solve ─────────────────────────────────────────────────────────────
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        selected = [i for i in df.index if pulp.value(x[i]) == 1]
        lineup_df = df.loc[selected].copy()
        lineups.append(lineup_df)
        excluded_sets.append(frozenset(selected))

    return lineups


# ── Showdown optimizer ────────────────────────────────────────────────────────

def _build_showdown_lineups(
    df: pd.DataFrame,
    sport: str,
    n: int,
    config: Dict[str, Any],
) -> list[pd.DataFrame]:
    """ILP optimizer for DraftKings showdown (CPT + FLEX) contests."""
    lineups: list[pd.DataFrame] = []
    excluded_sets: list[frozenset] = []

    lineup_size = DK_SHOWDOWN_LINEUP_SIZE  # typically 6
    slots = DK_SHOWDOWN_SLOTS  # {"CPT": 1, "FLEX": 5}

    max_salary = SALARY_CAP
    min_salary = config.get("min_salary", 49000)
    max_from_team = config.get("max_from_team", 4)
    min_teams = config.get("min_teams", 2)
    exposure_cap = config.get("exposure_cap", 1.0)

    slot_eligibility: dict[str, list[int]] = {
        slot: df.index[df["slot"] == slot].tolist()
        for slot in slots
    }

    teams = df["team"].dropna().unique().tolist()

    for lineup_idx in range(n):
        prob = pulp.LpProblem(f"showdown_{lineup_idx}", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", df.index, cat="Binary")

        prob += pulp.lpSum(df.loc[i, "proj"] * x[i] for i in df.index)

        prob += pulp.lpSum(x[i] for i in df.index) == lineup_size
        prob += pulp.lpSum(df.loc[i, "salary"] * x[i] for i in df.index) <= max_salary
        prob += pulp.lpSum(df.loc[i, "salary"] * x[i] for i in df.index) >= min_salary

        for slot, count in slots.items():
            eligible = slot_eligibility.get(slot, [])
            prob += pulp.lpSum(x[i] for i in eligible) == count

        for team in teams:
            team_idx = df.index[df["team"] == team].tolist()
            prob += pulp.lpSum(x[i] for i in team_idx) <= max_from_team

        t = pulp.LpVariable.dicts("t", teams, cat="Binary")
        for team in teams:
            team_idx = df.index[df["team"] == team].tolist()
            if team_idx:
                prob += t[team] <= pulp.lpSum(x[i] for i in team_idx)
                prob += (
                    pulp.lpSum(x[i] for i in team_idx)
                    <= len(team_idx) * t[team]
                )
        prob += pulp.lpSum(t[team] for team in teams) >= min_teams

        if exposure_cap < 1.0 and lineups:
            max_appearances = int(exposure_cap * n)
            for i in df.index:
                appearances = sum(
                    1 for prev in lineups
                    if i in prev.index
                )
                if appearances >= max_appearances:
                    prob += x[i] == 0

        for excl in excluded_sets:
            prob += pulp.lpSum(x[i] for i in excl) <= lineup_size - 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        selected = [i for i in df.index if pulp.value(x[i]) == 1]
        lineup_df = df.loc[selected].copy()
        lineups.append(lineup_df)
        excluded_sets.append(frozenset(selected))

    return lineups


# =============================================================================
# LINEUP EXPORT
# =============================================================================

def format_lineups_for_dk(
    lineups: list[pd.DataFrame],
    sport: str = "NBA",
    contest_type: str = "classic",
) -> pd.DataFrame:
    """Convert a list of lineup DataFrames into the DraftKings upload format.

    DraftKings expects a CSV where each row is one lineup and each column
    corresponds to a positional slot (e.g. ``PG``, ``SG``, …, ``UTIL``).
    Cell values are the player's DraftKings ID (``dk_id``).

    Parameters
    ----------
    lineups : list of pd.DataFrame
        Output of :func:`build_lineups`.
    sport : str
        ``"NBA"`` or ``"PGA"``.
    contest_type : str
        ``"classic"`` or ``"showdown"``.

    Returns
    -------
    pd.DataFrame
        One row per lineup, columns = positional slot headers.
    """
    if not lineups:
        return pd.DataFrame()

    sport = sport.upper()

    if contest_type == "showdown":
        slot_order = list(DK_SHOWDOWN_SLOTS.keys())  # ["CPT", "FLEX", ...]
        rows = []
        for lu in lineups:
            row: dict[str, str] = {}
            for slot in slot_order:
                players = lu[lu["slot"] == slot]
                row[slot] = "|".join(players["dk_id"].astype(str).tolist())
            rows.append(row)
        return pd.DataFrame(rows, columns=slot_order)

    # Classic
    if sport == "NBA":
        slot_order = list(DK_POS_SLOTS.keys())
    else:
        slot_order = ["G", "G", "G", "G", "G", "G"]

    rows = []
    for lu in lineups:
        row = {}
        assigned: set[int] = set()
        for slot in slot_order:
            candidates = lu[(lu["position"] == slot) & (~lu.index.isin(assigned))]
            if not candidates.empty:
                chosen = candidates.index[0]
                row[slot] = str(lu.loc[chosen, "dk_id"])
                assigned.add(chosen)
            else:
                row[slot] = ""
        rows.append(row)

    return pd.DataFrame(rows, columns=slot_order)


def export_lineups(
    lineups: list[pd.DataFrame],
    path: str,
    sport: str = "NBA",
    contest_type: str = "classic",
) -> None:
    """Write lineups to a CSV file in DraftKings upload format.

    Parameters
    ----------
    lineups : list of pd.DataFrame
        Output of :func:`build_lineups`.
    path : str
        Destination file path.
    sport : str
        ``"NBA"`` or ``"PGA"``.
    contest_type : str
        ``"classic"`` or ``"showdown"``.
    """
    dk_df = format_lineups_for_dk(lineups, sport=sport, contest_type=contest_type)
    dk_df.to_csv(path, index=False)


# =============================================================================
# CONVENIENCE END-TO-END RUNNER
# =============================================================================

def run(
    sport: str = "NBA",
    contest_type: str = "classic",
    data_mode: str = "historical",
    slate_date: str | None = None,
    n: int = 20,
    cfg: Dict[str, Any] | None = None,
    proj: Dict[str, float] | None = None,
    own: Dict[str, float] | None = None,
    export_path: str | None = None,
    yakos_root: str | None = None,
) -> Tuple[list[pd.DataFrame], pd.DataFrame]:
    """End-to-end pipeline: load → prepare → project → optimise → (export).

    Parameters
    ----------
    sport : str
        ``"NBA"`` or ``"PGA"``.
    contest_type : str
        ``"classic"`` or ``"showdown"``.
    data_mode : str
        ``"historical"`` or ``"live"``.
    slate_date : str or None
        ISO 8601 date for historical mode.
    n : int
        Number of lineups to generate.
    cfg : dict or None
        Config overrides.
    proj : dict or None
        Custom projections ``{name: pts}``.
    own : dict or None
        Projected ownership ``{name: pct}``.
    export_path : str or None
        If provided, write lineups CSV to this path.
    yakos_root : str or None
        Repository root override.

    Returns
    -------
    (lineups, pool) : tuple
        *lineups* is a list of lineup DataFrames; *pool* is the prepared
        player pool used by the optimiser.
    """
    raw = load_player_pool(
        sport=sport,
        data_mode=data_mode,
        slate_date=slate_date,
        yakos_root=yakos_root,
    )
    pool = prepare_pool(raw, sport=sport, contest_type=contest_type, cfg=cfg)

    if proj:
        pool = apply_projections(pool, proj)
    if own:
        pool = apply_ownership(pool, own)

    lineups = build_lineups(pool, sport=sport, contest_type=contest_type, n=n, cfg=cfg)

    if export_path:
        export_lineups(lineups, export_path, sport=sport, contest_type=contest_type)

    return lineups, pool


# =============================================================================
# STACKING HELPERS
# =============================================================================

def add_game_stack_constraint(
    prob: pulp.LpProblem,
    x: dict,
    df: pd.DataFrame,
    game: str,
    min_stack: int = 3,
) -> pulp.LpProblem:
    """Add a constraint requiring ≥ *min_stack* players from *game*.

    Parameters
    ----------
    prob : pulp.LpProblem
        The ILP problem instance to modify in-place.
    x : dict
        Decision variable dict ``{index: LpVariable}`` from the optimizer.
    df : pd.DataFrame
        Prepared player pool.
    game : str
        Game identifier string as it appears in the ``game_info`` column,
        e.g. ``"BOS@MIA 07:00PM ET"``.
    min_stack : int
        Minimum number of players required from *game*.

    Returns
    -------
    pulp.LpProblem
        The modified problem (same object, returned for convenience).
    """
    if "game_info" not in df.columns:
        return prob
    game_idx = df.index[df["game_info"].str.contains(game, na=False)].tolist()
    prob += pulp.lpSum(x[i] for i in game_idx) >= min_stack
    return prob


def add_team_stack_constraint(
    prob: pulp.LpProblem,
    x: dict,
    df: pd.DataFrame,
    team: str,
    min_stack: int = 3,
) -> pulp.LpProblem:
    """Require ≥ *min_stack* players from *team*.

    Parameters
    ----------
    prob : pulp.LpProblem
    x : dict
    df : pd.DataFrame
    team : str
        Team abbreviation as it appears in the ``team`` column.
    min_stack : int

    Returns
    -------
    pulp.LpProblem
    """
    team_idx = df.index[df["team"] == team].tolist()
    prob += pulp.lpSum(x[i] for i in team_idx) >= min_stack
    return prob


# =============================================================================
# PLAYER LOCKING / EXCLUSION
# =============================================================================

def lock_players(
    df: pd.DataFrame,
    prob: pulp.LpProblem,
    x: dict,
    names: list[str],
) -> pulp.LpProblem:
    """Force specific players into every lineup.

    Parameters
    ----------
    df : pd.DataFrame
    prob : pulp.LpProblem
    x : dict
    names : list of str
        Player names to lock.

    Returns
    -------
    pulp.LpProblem
    """
    for name in names:
        idxs = df.index[df["name"] == name].tolist()
        for i in idxs:
            prob += x[i] == 1
    return prob


def exclude_players(
    df: pd.DataFrame,
    prob: pulp.LpProblem,
    x: dict,
    names: list[str],
) -> pulp.LpProblem:
    """Exclude specific players from every lineup.

    Parameters
    ----------
    df : pd.DataFrame
    prob : pulp.LpProblem
    x : dict
    names : list of str
        Player names to exclude.

    Returns
    -------
    pulp.LpProblem
    """
    for name in names:
        idxs = df.index[df["name"] == name].tolist()
        for i in idxs:
            prob += x[i] == 0
    return prob


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def lineup_summary(lineups: list[pd.DataFrame]) -> pd.DataFrame:
    """Summarise key metrics across all generated lineups.

    Parameters
    ----------
    lineups : list of pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per lineup with columns:
        ``lineup_num``, ``total_salary``, ``total_proj``, ``n_teams``.
    """
    records = []
    for i, lu in enumerate(lineups):
        records.append({
            "lineup_num": i + 1,
            "total_salary": lu["salary"].sum(),
            "total_proj": lu["proj"].sum(),
            "n_teams": lu["team"].nunique(),
        })
    return pd.DataFrame(records)


def player_exposure(lineups: list[pd.DataFrame]) -> pd.DataFrame:
    """Calculate each player's exposure across all lineups.

    Parameters
    ----------
    lineups : list of pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: ``name``, ``appearances``, ``exposure_pct``.
    """
    if not lineups:
        return pd.DataFrame(columns=["name", "appearances", "exposure_pct"])

    counts: dict[str, int] = {}
    for lu in lineups:
        for name in lu["name"]:
            counts[name] = counts.get(name, 0) + 1

    total = len(lineups)
    records = [
        {"name": name, "appearances": cnt, "exposure_pct": round(cnt / total * 100, 1)}
        for name, cnt in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return pd.DataFrame(records)


def value_players(
    df: pd.DataFrame,
    min_salary: int = 3500,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top value plays ranked by proj / salary * 1000.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared pool.
    min_salary : int
        Filter out players below this salary.
    top_n : int
        Number of players to return.

    Returns
    -------
    pd.DataFrame
    """
    df = df[df["salary"] >= min_salary].copy()
    df["value"] = df["proj"] / df["salary"] * 1000
    return df.nlargest(top_n, "value")[["name", "position", "salary", "proj", "value"]]


# =============================================================================
# SIMULATION / MONTE CARLO
# =============================================================================

def simulate_projections(
    df: pd.DataFrame,
    sigma_pct: float = 0.15,
    seed: int | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise to projections for Monte Carlo simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Pool with a ``proj`` column.
    sigma_pct : float
        Standard deviation as a fraction of the projected value (default 15%).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with noisy projections.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    noise = rng.normal(loc=1.0, scale=sigma_pct, size=len(df))
    df["proj"] = (df["proj"] * noise).clip(lower=0)
    return df


def run_montecarlo(
    sport: str = "NBA",
    contest_type: str = "classic",
    data_mode: str = "historical",
    slate_date: str | None = None,
    n_lineups: int = 20,
    n_sims: int = 100,
    sigma_pct: float = 0.15,
    cfg: Dict[str, Any] | None = None,
    export_dir: str | None = None,
    yakos_root: str | None = None,
) -> pd.DataFrame:
    """Run Monte Carlo simulation and return aggregated exposure stats.

    For each simulation, projections are jittered and *n_lineups* lineups
    are generated; the results are aggregated into a player exposure table.

    Parameters
    ----------
    sport, contest_type, data_mode, slate_date, cfg, yakos_root
        Passed through to :func:`load_player_pool` / :func:`prepare_pool`.
    n_lineups : int
        Lineups to generate per simulation run.
    n_sims : int
        Number of Monte Carlo iterations.
    sigma_pct : float
        Projection noise level (see :func:`simulate_projections`).
    export_dir : str or None
        If provided, write per-simulation CSVs to this directory.

    Returns
    -------
    pd.DataFrame
        Aggregated exposure table with columns
        ``name``, ``mean_exposure_pct``, ``median_exposure_pct``, ``std_exposure_pct``.
    """
    raw = load_player_pool(
        sport=sport,
        data_mode=data_mode,
        slate_date=slate_date,
        yakos_root=yakos_root,
    )
    base_pool = prepare_pool(raw, sport=sport, contest_type=contest_type, cfg=cfg)

    all_exposures: list[pd.DataFrame] = []

    for sim_i in range(n_sims):
        sim_pool = simulate_projections(base_pool, sigma_pct=sigma_pct, seed=sim_i)
        lineups = build_lineups(
            sim_pool,
            sport=sport,
            contest_type=contest_type,
            n=n_lineups,
            cfg=cfg,
        )
        exp = player_exposure(lineups)
        exp["sim"] = sim_i
        all_exposures.append(exp)

        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            export_lineups(
                lineups,
                os.path.join(export_dir, f"sim_{sim_i:04d}.csv"),
                sport=sport,
                contest_type=contest_type,
            )

    combined = pd.concat(all_exposures, ignore_index=True)
    agg = (
        combined.groupby("name")["exposure_pct"]
        .agg(
            mean_exposure_pct="mean",
            median_exposure_pct="median",
            std_exposure_pct="std",
        )
        .reset_index()
        .sort_values("mean_exposure_pct", ascending=False)
    )
    return agg


# =============================================================================
# CONTEST ENTRY MANAGEMENT
# =============================================================================

def deduplicate_lineups(
    lineups: list[pd.DataFrame],
    min_diff: int = 2,
) -> list[pd.DataFrame]:
    """Remove lineups that are too similar to earlier lineups.

    Two lineups are considered duplicates when they differ by fewer than
    *min_diff* players.

    Parameters
    ----------
    lineups : list of pd.DataFrame
    min_diff : int
        Minimum number of differing players required to keep a lineup.

    Returns
    -------
    list of pd.DataFrame
    """
    unique: list[pd.DataFrame] = []
    for lu in lineups:
        lu_set = frozenset(lu["name"].tolist())
        if all(
            len(lu_set.symmetric_difference(frozenset(prev["name"].tolist()))) >= min_diff
            for prev in unique
        ):
            unique.append(lu)
    return unique


def cap_exposure(
    lineups: list[pd.DataFrame],
    max_pct: float = 0.6,
) -> list[pd.DataFrame]:
    """Remove lineups to bring every player's exposure under *max_pct*.

    This is a greedy post-processor: lineups are visited in order and a
    lineup is dropped if it would push any player over the cap.

    Parameters
    ----------
    lineups : list of pd.DataFrame
    max_pct : float
        Maximum allowed exposure fraction (0–1).

    Returns
    -------
    list of pd.DataFrame
    """
    total = len(lineups)
    max_appearances = int(max_pct * total)
    counts: dict[str, int] = {}
    kept: list[pd.DataFrame] = []
    for lu in lineups:
        names = lu["name"].tolist()
        if all(counts.get(n, 0) < max_appearances for n in names):
            kept.append(lu)
            for n in names:
                counts[n] = counts.get(n, 0) + 1
    return kept


# =============================================================================
# SLATE UTILITIES
# =============================================================================

def list_available_slates(
    sport: str = "NBA",
    yakos_root: str | None = None,
) -> list[str]:
    """List parquet files available in the local historical cache.

    Parameters
    ----------
    sport : str
    yakos_root : str or None

    Returns
    -------
    list of str
        Sorted list of date strings (``"YYYY-MM-DD"``) for which cached
        pool data exists.
    """
    import glob as _glob
    root = yakos_root or YAKOS_ROOT
    sport = sport.upper()
    if sport == "NBA":
        pattern = os.path.join(root, "data", "nba", "pool", "*.parquet")
    elif sport == "PGA":
        pattern = os.path.join(root, "data", "pga", "pool", "*.parquet")
    else:
        raise ValueError(f"Unsupported sport: {sport}")
    files = sorted(_glob.glob(pattern))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def filter_by_salary_range(
    df: pd.DataFrame,
    min_salary: int,
    max_salary: int,
) -> pd.DataFrame:
    """Return only players within the specified salary range."""
    return df[(df["salary"] >= min_salary) & (df["salary"] <= max_salary)].copy()


def filter_by_team(
    df: pd.DataFrame,
    teams: list[str],
) -> pd.DataFrame:
    """Return only players belonging to the specified teams."""
    return df[df["team"].isin(teams)].copy()


def filter_by_position(
    df: pd.DataFrame,
    positions: list[str],
) -> pd.DataFrame:
    """Return only players eligible for the specified positions."""
    return df[df["position"].isin(positions)].copy()


# =============================================================================
# ADVANCED METRICS
# =============================================================================

def add_ceiling(
    df: pd.DataFrame,
    sigma_pct: float = 0.25,
) -> pd.DataFrame:
    """Append a ``ceiling`` column (proj + 1σ)."""
    df = df.copy()
    df["ceiling"] = df["proj"] * (1 + sigma_pct)
    return df


def add_floor(
    df: pd.DataFrame,
    sigma_pct: float = 0.25,
) -> pd.DataFrame:
    """Append a ``floor`` column (proj − 1σ, clipped at 0)."""
    df = df.copy()
    df["floor"] = (df["proj"] * (1 - sigma_pct)).clip(lower=0)
    return df


def rank_players(
    df: pd.DataFrame,
    by: str = "proj",
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the pool sorted and ranked by *by*."""
    df = df.copy().sort_values(by, ascending=ascending)
    df["rank"] = range(1, len(df) + 1)
    return df


# =============================================================================
# CORRELATION HELPERS
# =============================================================================

def teammate_correlation(
    lineups: list[pd.DataFrame],
    min_appearances: int = 5,
) -> pd.DataFrame:
    """Return a co-appearance count matrix for players across lineups.

    Parameters
    ----------
    lineups : list of pd.DataFrame
    min_appearances : int
        Only include pairs where both players appear at least this many
        times across all lineups.

    Returns
    -------
    pd.DataFrame
        Square matrix of co-appearance counts indexed/columned by player name.
    """
    from collections import Counter
    import itertools

    pair_counts: Counter = Counter()
    solo_counts: Counter = Counter()
    for lu in lineups:
        names = sorted(lu["name"].tolist())
        for name in names:
            solo_counts[name] += 1
        for pair in itertools.combinations(names, 2):
            pair_counts[pair] += 1

    eligible = {n for n, c in solo_counts.items() if c >= min_appearances}
    eligible_pairs = {
        pair: cnt for pair, cnt in pair_counts.items()
        if pair[0] in eligible and pair[1] in eligible
    }

    players = sorted(eligible)
    mat = pd.DataFrame(0, index=players, columns=players)
    for (a, b), cnt in eligible_pairs.items():
        mat.loc[a, b] = cnt
        mat.loc[b, a] = cnt
    np.fill_diagonal(mat.values, [solo_counts[p] for p in players])
    return mat


# =============================================================================
# DEPRECATED / COMPATIBILITY SHIMS
# =============================================================================

def build_pool(*args, **kwargs):
    """Deprecated alias for :func:`prepare_pool`.  Will be removed in v2.0."""
    import warnings
    warnings.warn(
        "build_pool() is deprecated; use prepare_pool() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return prepare_pool(*args, **kwargs)


def generate_lineups(*args, **kwargs):
    """Deprecated alias for :func:`build_lineups`.  Will be removed in v2.0."""
    import warnings
    warnings.warn(
        "generate_lineups() is deprecated; use build_lineups() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_lineups(*args, **kwargs)


def write_lineups(*args, **kwargs):
    """Deprecated alias for :func:`export_lineups`.  Will be removed in v2.0.

    .. deprecated::
        Use :func:`export_lineups` or :func:`yak_core.publishing` instead.
    """
    import warnings
    warnings.warn(
        "write_lineups() is deprecated. "
        "Use yak_core.publishing instead."
    )
