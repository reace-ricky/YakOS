"""Monte Carlo simulation for YakOS DFS optimizer."""
from __future__ import annotations

import dataclasses
import enum
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from yak_core.edge import compute_empirical_std  # noqa: E402
from yak_core.sim_rating import compute_pipeline_ratings  # noqa: E402

# Status values that make a player ineligible for sims.
_INELIGIBLE_STATUSES = {
    "OUT", "IR", "INJ", "SUSPENDED", "SUSP",
    "G-LEAGUE", "G_LEAGUE", "GLEAGUE",
    "DND", "NA", "O",
}


def compute_sim_eligible(
    pool_df: pd.DataFrame,
    min_proj_minutes: float = 4.0,
    exclude_out_ir: bool = True,
    today_teams: Optional[List[str]] = None,
    minutes_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute the ``sim_eligible`` column for every player in the pool.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Expected columns (all optional): ``status``, ``minutes``,
        ``proj_minutes``, ``team``.
    min_proj_minutes : float
        Minimum projected minutes required for ``sim_eligible = True``
        (default 4.0).  Players with minutes ≤ min_proj_minutes are
        excluded.  Set to ``0`` to skip the minutes filter.
    exclude_out_ir : bool
        If ``True`` (default), players whose ``status`` matches a known
        ineligible value (OUT, IR, G-League, Suspended, etc.) are excluded.
    today_teams : list of str, optional
        If provided, players whose ``team`` is **not** in this list are
        excluded.
    minutes_col : str, optional
        Explicit column name to use for the minutes filter.  When provided
        the auto-detection logic is bypassed:

        * ``"proj_minutes"`` — use projected minutes (live slate).
        * ``"actual_minutes"`` — use actual minutes (historical slate).

        If the specified column is not present in the pool the minutes
        filter is skipped silently.  When *None* (default) the column is
        auto-detected: ``"minutes"`` if present, else ``"proj_minutes"``.

    Returns
    -------
    pd.DataFrame
        Copy of *pool_df* with a boolean ``sim_eligible`` column added (or
        overwritten if it already exists).  Existing ``sim_eligible = False``
        overrides are preserved when ``pool_df`` already contains the column —
        i.e. manual overrides are respected.
    """
    df = pool_df.copy()

    # Preserve any manual overrides already in the DataFrame.
    # Start everyone as eligible; manual False overrides are re-applied at the
    # end so they can't be accidentally reset by the rule engine.
    has_existing = "sim_eligible" in df.columns
    if has_existing:
        manual_false = df.index[~df["sim_eligible"].astype(bool)]
    else:
        manual_false = df.index[[]]

    df["sim_eligible"] = True

    # ── Status-based exclusions ───────────────────────────────────────────
    if exclude_out_ir and "status" in df.columns:
        norm = df["status"].fillna("").astype(str).str.strip().str.upper()
        df.loc[norm.isin(_INELIGIBLE_STATUSES), "sim_eligible"] = False

    # ── Minutes-based exclusions ──────────────────────────────────────────
    # When minutes_col is explicitly specified use it (live → proj_minutes,
    # historical → actual_minutes).  Otherwise auto-detect: prefer 'minutes',
    # fall back to 'proj_minutes' for API-loaded pools.
    if minutes_col is not None:
        _minutes_col: Optional[str] = minutes_col if minutes_col in df.columns else None
    else:
        _minutes_col = (
            "minutes" if "minutes" in df.columns
            else ("proj_minutes" if "proj_minutes" in df.columns else None)
        )
    if _minutes_col is not None and min_proj_minutes > 0:
        mins = pd.to_numeric(df[_minutes_col], errors="coerce").fillna(0)
        df.loc[mins <= min_proj_minutes, "sim_eligible"] = False

    # ── Team-based exclusions ─────────────────────────────────────────────
    if today_teams is not None and "team" in df.columns:
        df.loc[~df["team"].isin(today_teams), "sim_eligible"] = False

    # Re-apply any manual False overrides
    if len(manual_false):
        df.loc[manual_false, "sim_eligible"] = False

    return df


class ContestType(enum.Enum):
    """DK contest archetypes used to derive dynamic smash/bust thresholds."""
    CASH = "CASH"
    SE_SMALL = "SE_SMALL"          # single-entry / 3-max / small-field
    GPP_LARGE = "GPP_LARGE"        # large-field GPP / lotto

    @classmethod
    def _missing_(cls, value: object) -> "ContestType":  # type: ignore[override]
        """Case-insensitive lookup by value string."""
        if isinstance(value, str):
            for member in cls:
                if member.value.upper() == value.upper():
                    return member
        return cls.GPP_LARGE


@dataclasses.dataclass
class LineupSimSummary:
    """Per-lineup Monte Carlo summary statistics and dynamic thresholds."""
    lineup_id: Any
    median_score: float
    stdev_score: float
    p15_score: float
    p85_score: float
    smash_threshold: float = 0.0
    bust_threshold: float = 0.0
    p90_score: float = 0.0  # 90th-percentile score; used as smash bar (top 10% of distribution)
    p30_score: float = 0.0  # 30th-percentile score; used as bust bar (bottom 30% of distribution)


# Contest-calibrated absolute overrides.
# All values are ``None`` so every contest type uses fully dynamic,
# lineup-specific percentile-based thresholds (p90 = smash bar, p30 = bust bar).
# This ensures each lineup's thresholds reflect its own simulated distribution
# rather than a single fixed score that applies to every lineup equally.
CONTEST_ABSOLUTE_THRESHOLDS: Dict[ContestType, Dict[str, Optional[float]]] = {
    ContestType.CASH:      {"smash": None, "bust": None},
    ContestType.SE_SMALL:  {"smash": None, "bust": None},
    ContestType.GPP_LARGE: {"smash": None, "bust": None},
}

# Percentile levels used to derive contest-level smash/bust thresholds from the
# pooled field of simulated lineup totals.  p90 → top 10% finish; p30 → losing
# region / min-cash boundary.  Adjust during calibration sprints.
CONTEST_SMASH_PERCENTILE: int = 90
CONTEST_BUST_PERCENTILE: int = 30

# Minimum ownership (%) required to compute a Leverage Score.  Below this value
# the score is set to ``NaN`` to avoid spurious infinity-like values from near-zero
# denominators.
MIN_OWNERSHIP_FOR_LEVERAGE: float = 0.1


def summarize_lineup_sims(scores: List[float]) -> LineupSimSummary:
    """Compute distribution statistics for a list of simulated lineup totals.

    Parameters
    ----------
    scores : list of float
        Raw simulated DK score totals for a single lineup.

    Returns
    -------
    LineupSimSummary
        Contains ``median_score``, ``stdev_score``, ``p15_score``, ``p85_score``,
        ``p90_score``, and ``p30_score``.
        The ``smash_threshold`` and ``bust_threshold`` fields are initialised to
        ``0.0`` and should be populated by :func:`compute_thresholds`.
    """
    arr = np.asarray(scores, dtype=float)
    return LineupSimSummary(
        lineup_id=None,
        median_score=float(np.median(arr)),
        stdev_score=float(arr.std()),
        p15_score=float(np.percentile(arr, 15)),
        p85_score=float(np.percentile(arr, 85)),
        p90_score=float(np.percentile(arr, 90)),
        p30_score=float(np.percentile(arr, 30)),
    )


def compute_thresholds(
    summary: LineupSimSummary,
    contest_type: ContestType = ContestType.GPP_LARGE,
) -> None:
    """Set ``smash_threshold`` and ``bust_threshold`` on *summary* in place.

    Uses distribution-relative formulas scaled per contest type, with optional
    absolute overrides from :data:`CONTEST_ABSOLUTE_THRESHOLDS`.

    When ``p90_score`` and ``p30_score`` are present on *summary* (populated by
    :func:`summarize_lineup_sims`), the dynamic path uses them directly so that
    smash = top 10 % and bust = bottom 30 % of each lineup's own sim distribution.
    Otherwise falls back to the stdev-multiplier formulas below.

    Fallback dynamic formulas (applied when p90/p30 are absent and override is ``None``):

    * ``CASH``      — smash = median + 0.5 × stdev, bust = median − 1.0 × stdev
    * ``SE_SMALL``  — smash = median + 1.0 × stdev, bust = median − 1.0 × stdev
    * ``GPP_LARGE`` — smash = median + 1.5 × stdev, bust = median − 1.0 × stdev
    """
    _multipliers: Dict[ContestType, Tuple[float, float]] = {
        ContestType.CASH:      (0.5, 1.0),
        ContestType.SE_SMALL:  (1.0, 1.0),
        ContestType.GPP_LARGE: (1.5, 1.0),
    }
    smash_mult, bust_mult = _multipliers.get(contest_type, (1.5, 1.0))

    # Prefer percentile-based when populated by summarize_lineup_sims (p90/p30 > 0)
    if summary.p90_score > 0:
        dynamic_smash = summary.p90_score
    else:
        dynamic_smash = summary.median_score + smash_mult * summary.stdev_score

    if summary.p30_score > 0:
        dynamic_bust = summary.p30_score
    else:
        dynamic_bust = summary.median_score - bust_mult * summary.stdev_score

    overrides = CONTEST_ABSOLUTE_THRESHOLDS.get(contest_type, {})
    smash_override = overrides.get("smash")
    bust_override = overrides.get("bust")
    summary.smash_threshold = float(smash_override) if smash_override is not None else dynamic_smash
    summary.bust_threshold = float(bust_override) if bust_override is not None else dynamic_bust


def compute_smash_bust_rates(
    scores: List[float],
    smash_threshold: float,
    bust_threshold: float,
) -> Tuple[float, float]:
    """Return ``(smash_pct, bust_pct)`` from raw sim scores and thresholds.

    Parameters
    ----------
    scores : list of float
        Raw simulated totals for a single lineup.
    smash_threshold : float
        Scores *≥* this value count as smashes.
    bust_threshold : float
        Scores *≤* this value count as busts.

    Returns
    -------
    tuple[float, float]
        ``(smash_pct, bust_pct)`` each in the range ``[0.0, 1.0]``.
    """
    arr = np.asarray(scores, dtype=float)
    total = len(arr)
    if total == 0:
        return 0.0, 0.0
    smash_pct = float((arr >= smash_threshold).sum()) / total
    bust_pct = float((arr <= bust_threshold).sum()) / total
    return smash_pct, bust_pct


def run_monte_carlo_for_lineups(
    lineups_df: pd.DataFrame,
    n_sims: int = 500,
    volatility_mode: str = "standard",
    contest_type: ContestType = ContestType.GPP_LARGE,
    _return_scores: bool = False,
) -> "pd.DataFrame | Tuple[pd.DataFrame, Dict[Any, np.ndarray]]":
    """Run Monte Carlo simulations on lineup projections.

    For each lineup, simulates ``n_sims`` outcomes by sampling from
    per-player normal distributions (mean=proj, std derived from
    ceil/floor when available).  Smash and bust thresholds are computed
    at the **contest level** — pooling all simulated lineup totals across
    the entire field, then taking the 90th percentile as the smash bar and
    the 30th percentile as the bust bar.  Every lineup in the same run
    shares the same contest-level thresholds so that Smash% / Bust% vary
    naturally across lineups (instead of being ``≈ 10% / 30%`` by
    construction).

    Parameters
    ----------
    lineups_df : pd.DataFrame
        Long-format lineup table with ``lineup_index`` and ``proj`` columns.
        Optional: ``ceil``, ``floor`` columns improve variance estimates.
    n_sims : int, optional
        Number of simulation iterations per lineup (default 500).
    volatility_mode : str, optional
        ``"low"`` / ``"standard"`` / ``"high"`` — scales default variance when
        ceil/floor are unavailable.
    contest_type : ContestType, optional
        Contest archetype (default ``ContestType.GPP_LARGE``).
    _return_scores : bool, optional
        When ``True`` the function returns a ``(DataFrame, scores_dict)`` tuple
        where ``scores_dict`` maps each ``lineup_index`` to its raw 1-D array of
        ``n_sims`` simulated totals.  Useful for downstream histogram rendering.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        Per-lineup summary with columns:
        ``lineup_index``, ``sim_mean``, ``sim_std``,
        ``median_points``, ``sim_p85``, ``sim_p15``,
        ``smash_threshold``, ``bust_threshold``,
        ``contest_smash_score``, ``contest_bust_score``,
        ``smash_pct``, ``bust_pct``, ``contest_type``.
        ``smash_prob`` and ``bust_prob`` are kept as aliases for
        ``smash_pct`` / ``bust_pct`` for backwards compatibility.

        ``smash_threshold`` / ``bust_threshold`` are the contest-level values
        (identical for all rows in a single run).  Per-lineup distribution
        statistics (Avg, Std Dev, Median, P85, P15) remain as read-only
        diagnostic columns.
    """
    if lineups_df.empty or "lineup_index" not in lineups_df.columns:
        if _return_scores:
            return pd.DataFrame(), {}
        return pd.DataFrame()

    # Volatility mode → multiplier on top of empirical calibration.
    # "standard" (1.0) = the backtest-calibrated baseline; low/high scale it.
    vol_map = {"low": 0.65, "standard": 1.0, "high": 1.45}
    variance_mult = vol_map.get(volatility_mode, 1.0)

    rng = np.random.RandomState(42)

    # ── Phase 1: run all lineup sims, collect totals ─────────────────────────
    per_lineup: List[Tuple[Any, np.ndarray, "LineupSimSummary"]] = []

    # Derive contest_mode from contest_type enum for variance dampening
    _contest_mode = contest_type.value.lower() if hasattr(contest_type, 'value') else str(contest_type).lower()
    # Map known contest types to variance mode keys
    _cm_map = {"gpp_large": "gpp_150", "gpp_small": "gpp_20", "single_entry": "se_3max"}
    _contest_mode = _cm_map.get(_contest_mode, _contest_mode)

    for lu_id, grp in lineups_df.groupby("lineup_index"):
        projs = grp["proj"].fillna(0).values.astype(float)
        salaries = pd.to_numeric(grp.get("salary", pd.Series(6000, index=grp.index)), errors="coerce").fillna(6000).values.astype(float)

        # PGA: use DataGolf per-player std_dev when available
        _dg_std = None
        if "std_dev" in grp.columns:
            _dg_std_raw = pd.to_numeric(grp["std_dev"], errors="coerce").fillna(0).values.astype(float)
            if (_dg_std_raw > 0).any():
                _dg_std = _dg_std_raw

        # Empirical variance (PGA path uses DataGolf std_dev; NBA uses salary-bracket model)
        stds = compute_empirical_std(projs, salaries, variance_mult=variance_mult, contest_mode=_contest_mode, std_dev=_dg_std)

        # (n_sims × n_players) outcome matrix
        sim_matrix = rng.normal(
            loc=projs[None, :],
            scale=stds[None, :],
            size=(n_sims, len(projs)),
        )
        sim_matrix = np.clip(sim_matrix, 0, None)
        totals = sim_matrix.sum(axis=1)
        summary = summarize_lineup_sims(totals.tolist())
        summary.lineup_id = lu_id
        per_lineup.append((lu_id, totals, summary))

    if not per_lineup:
        if _return_scores:
            return pd.DataFrame(), {}
        return pd.DataFrame()

    # ── Phase 2: contest-level thresholds from the pooled field ──────────────
    all_arr = np.concatenate([totals for _, totals, _ in per_lineup])
    contest_smash = float(np.percentile(all_arr, CONTEST_SMASH_PERCENTILE))
    contest_bust = float(np.percentile(all_arr, CONTEST_BUST_PERCENTILE))

    # ── Phase 3: per-lineup results against contest-level thresholds ─────────
    results = []
    scores_dict: Dict[Any, np.ndarray] = {}

    for lu_id, totals, summary in per_lineup:
        smash_pct, bust_pct = compute_smash_bust_rates(
            totals.tolist(), contest_smash, contest_bust
        )
        results.append({
            "lineup_index": lu_id,
            "sim_mean": round(float(totals.mean()), 2),
            "sim_std": round(float(totals.std()), 2),
            "median_points": round(summary.median_score, 2),
            "sim_p85": round(summary.p85_score, 2),
            "sim_p15": round(summary.p15_score, 2),
            # Contest-level thresholds (same for all lineups in this run)
            "smash_threshold": round(contest_smash, 2),
            "bust_threshold": round(contest_bust, 2),
            "contest_smash_score": round(contest_smash, 2),
            "contest_bust_score": round(contest_bust, 2),
            "smash_pct": round(smash_pct, 3),
            "bust_pct": round(bust_pct, 3),
            # Backwards-compatible aliases
            "smash_prob": round(smash_pct, 3),
            "bust_prob": round(bust_pct, 3),
            "contest_type": contest_type.value,
        })
        if _return_scores:
            scores_dict[lu_id] = totals

    df = pd.DataFrame(results)
    if _return_scores:
        return df, scores_dict
    return df


def simulate_live_updates(
    pool_df: pd.DataFrame,
    news_updates: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Apply live news / injury / lineup-change updates to the player pool.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Current player pool with ``player_name`` and ``proj`` columns.
    news_updates : list of dict
        Each dict may contain:

        * ``player_name`` (str) — required
        * ``status`` (str) — ``"OUT"``, ``"QUESTIONABLE"``, ``"GTD"``, ``"IN"``,
          ``"UPGRADED"``
        * ``proj_adj`` (float) — direct fantasy-point adjustment (takes priority)
        * ``minutes_change`` (float) — minutes delta; converted via ~1.5 FP/min

    Returns
    -------
    pd.DataFrame
        Updated pool with adjusted projections.
    """
    updated = pool_df.copy()

    status_multipliers: Dict[str, float] = {
        "OUT": 0.0,
        "QUESTIONABLE": 0.35,
        "GTD": 0.65,
        "IN": 1.0,
        "UPGRADED": 1.25,
    }
    fp_per_min = 1.5  # rough DK NBA approximation

    for update in news_updates:
        pname = update.get("player_name", "")
        mask = updated["player_name"] == pname
        if not mask.any():
            continue

        if update.get("proj_adj") is not None:
            updated.loc[mask, "proj"] = (
                updated.loc[mask, "proj"] + float(update["proj_adj"])
            ).clip(lower=0)
        elif update.get("status"):
            mult = status_multipliers.get(update["status"].upper(), 1.0)
            updated.loc[mask, "proj"] = (updated.loc[mask, "proj"] * mult).clip(lower=0)

        if update.get("minutes_change") is not None:
            fp_adj = float(update["minutes_change"]) * fp_per_min
            updated.loc[mask, "proj"] = (
                updated.loc[mask, "proj"] + fp_adj
            ).clip(lower=0)

    return updated


def backtest_sim(
    hist_df: pd.DataFrame,
    n_sims: int = 500,
    volatility_mode: str = "standard",
) -> Dict[str, Any]:
    """Backtest the Monte Carlo sim against historical actual scores.

    Runs :func:`run_monte_carlo_for_lineups` on each historical lineup using
    the pre-game projections recorded in *hist_df*, then compares the
    simulation's predicted distribution to the actual DraftKings score that
    each lineup achieved.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Historical lineup data.  Required columns: ``lineup_id``, ``proj``,
        ``actual``.  Optional: ``ceil``, ``floor`` (improve variance estimates).
        Each row represents one player in one lineup.
    n_sims : int, optional
        Monte Carlo iterations per lineup (default 500).
    volatility_mode : str, optional
        ``"low"`` / ``"standard"`` / ``"high"`` — scales default variance when
        ceil/floor are unavailable.

    Returns
    -------
    dict
        ``lineup_df``        — per-lineup DataFrame with columns
                               ``lineup_id``, ``sim_mean``, ``sim_std``,
                               ``sim_p15``, ``sim_p85``, ``actual``,
                               ``error``, ``within_range``.

        ``sim_mae``          — Mean Absolute Error: |sim_mean − actual|.

        ``sim_rmse``         — Root Mean Squared Error.

        ``sim_bias``         — Average (sim_mean − actual);
                               positive = sim over-projects.

        ``within_range_pct`` — Percentage of lineups where the actual score
                               falls within [sim_p15, sim_p85].

        ``n_lineups``        — Number of lineups evaluated.
    """
    required = {"lineup_id", "proj", "actual"}
    if hist_df.empty or not required.issubset(hist_df.columns):
        return {
            "lineup_df": pd.DataFrame(),
            "sim_mae": 0.0,
            "sim_rmse": 0.0,
            "sim_bias": 0.0,
            "within_range_pct": 0.0,
            "n_lineups": 0,
        }

    # Rename lineup_id → lineup_index for run_monte_carlo_for_lineups
    sim_input = hist_df.rename(columns={"lineup_id": "lineup_index"})

    sim_results = run_monte_carlo_for_lineups(
        sim_input, n_sims=n_sims, volatility_mode=volatility_mode
    )

    if sim_results.empty:
        return {
            "lineup_df": pd.DataFrame(),
            "sim_mae": 0.0,
            "sim_rmse": 0.0,
            "sim_bias": 0.0,
            "within_range_pct": 0.0,
            "n_lineups": 0,
        }

    # Compute actual score per lineup
    actual_scores = (
        hist_df.groupby("lineup_id")["actual"]
        .sum()
        .reset_index()
        .rename(columns={"lineup_id": "lineup_index"})
    )

    merged = sim_results.merge(actual_scores, on="lineup_index", how="inner")
    merged = merged.rename(columns={"lineup_index": "lineup_id"})
    merged["error"] = merged["sim_mean"] - merged["actual"]
    merged["within_range"] = (
        (merged["actual"] >= merged["sim_p15"])
        & (merged["actual"] <= merged["sim_p85"])
    )

    errors = merged["error"]
    n = len(merged)
    sim_mae = float(errors.abs().mean()) if n > 0 else 0.0
    sim_rmse = float(np.sqrt((errors ** 2).mean())) if n > 0 else 0.0
    sim_bias = float(errors.mean()) if n > 0 else 0.0
    within_range_pct = float(merged["within_range"].mean() * 100) if n > 0 else 0.0

    return {
        "lineup_df": merged.reset_index(drop=True),
        "sim_mae": round(sim_mae, 2),
        "sim_rmse": round(sim_rmse, 2),
        "sim_bias": round(sim_bias, 2),
        "within_range_pct": round(within_range_pct, 1),
        "n_lineups": n,
    }


def compute_player_anomaly_table(
    pool_df: pd.DataFrame,
    lineup_df: pd.DataFrame,
    n_sims: int = 500,
    cal_knobs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compute a per-player anomaly / leverage table from Monte Carlo sim results.

    For each player that appears in *lineup_df*, runs **lineup-level** Monte Carlo
    simulations (one sim per complete lineup, not per individual player), computes
    contest-level smash/bust thresholds from the pooled field distribution, then
    calculates each player's Smash% and Bust% as the fraction of contest sims where
    a lineup containing that player exceeds or falls below those thresholds.

    This produces Smash% / Bust% values that **vary naturally** across players —
    players in high-quality lineups will have Smash% > 10%, low-quality lineups
    Smash% < 10% — rather than being locked at the field average by construction.

    Leverage Score is defined as ``Smash% / Own%`` (higher means the player is
    a bigger upside play relative to expected ownership — more leverage in GPP).
    When ``Own%`` is below 0.1 the score is set to ``NaN`` to avoid spurious
    infinity-like values.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Must include a name column (``player_name`` or ``name``)
        and a ``proj`` column.  Optional: ``salary``,
        ``ownership`` / ``own%`` / ``proj_own``, ``ceil``, ``floor``.
    lineup_df : pd.DataFrame
        Long-format lineup table (as returned by ``run_optimizer``).  Must
        include a name column and ``lineup_index``.
    n_sims : int, optional
        Lineup-level simulation iterations (default 500).
    cal_knobs : dict, optional
        Calibration knobs.  Supported keys:

        * ``ceiling_boost``   (float, default 1.0) — scale upside player outcomes
        * ``floor_dampen``    (float, default 1.0) — compress downside outcomes
        * ``smash_threshold`` (float, optional) — ratio override: contest smash =
          this × mean lineup projection.  When absent, the 90th percentile of
          the pooled field scores is used (≈ top 10% of field).
        * ``bust_threshold``  (float, optional) — ratio override: contest bust =
          this × mean lineup projection.  When absent, the 30th percentile is
          used (≈ bottom 30% of field).

    Returns
    -------
    pd.DataFrame
        Sorted by Leverage Score descending.  Columns:
        ``Player``, ``Proj``, ``Salary``, ``Own%``, ``Smash%``, ``Bust%``,
        ``Leverage Score``, ``Value Trap``, ``Flag``.
        Empty DataFrame when inputs are insufficient.
    """
    if pool_df is None:
        pool_df = pd.DataFrame()
    if lineup_df.empty:
        return pd.DataFrame()

    knobs = cal_knobs or {}
    ceiling_boost = float(knobs.get("ceiling_boost", 1.0))
    floor_dampen = float(knobs.get("floor_dampen", 1.0))
    # When set, these override the contest threshold with: threshold = ratio × mean_lineup_proj
    smash_thr_ratio = knobs.get("smash_threshold")   # None → use p90 of field
    bust_thr_ratio = knobs.get("bust_threshold")     # None → use p30 of field

    # Find name column in lineup_df
    lu_name_col = next(
        (c for c in ("player_name", "name") if c in lineup_df.columns), None
    )
    if lu_name_col is None:
        return pd.DataFrame()

    # Normalise lineup_df to have "player_name" column
    lu = lineup_df.copy()
    if lu_name_col != "player_name":
        lu = lu.rename(columns={lu_name_col: "player_name"})
    # Normalise ownership column in lineup_df to "own%" for internal use.
    # own_proj is the canonical column; ownership/Own%/proj_own accepted as legacy aliases.
    # ownership column used here: first available from lu in priority order ("own_proj" → "ownership" → "Own%" → "proj_own"), renamed to "own%"
    for _src in ("own_proj", "ownership", "Own%", "proj_own"):
        if _src in lu.columns and "own%" not in lu.columns:
            lu = lu.rename(columns={_src: "own%"})
            break

    if "lineup_index" not in lu.columns:
        return pd.DataFrame()

    # Build pool lookup keyed by name.
    # own_proj is the required canonical ownership column on pool_df.
    # For backward compatibility, fall back to "ownership" alias when own_proj absent.
    _pool_sim_cols = ("name", "proj", "salary", "own%", "ceil", "floor")
    pool_lookup: dict = {}
    if not pool_df.empty:
        pool = pool_df.copy()
        if "player_name" in pool.columns and "name" not in pool.columns:
            pool = pool.rename(columns={"player_name": "name"})
        # Map own_proj → own% (canonical); fall back to ownership alias if needed
        if "own_proj" in pool.columns:
            pool = pool.rename(columns={"own_proj": "own%"})  # ownership column used here: pool["own_proj"] (canonical), renamed to "own%"
        elif "ownership" in pool.columns and "own%" not in pool.columns:
            pool = pool.rename(columns={"ownership": "own%"})  # ownership column used here: pool["ownership"] (backward-compat alias), renamed to "own%"
        if "name" in pool.columns:
            _keep = [c for c in _pool_sim_cols if c in pool.columns]
            pool_lookup = (
                pool[_keep]
                .drop_duplicates(subset=["name"])
                .set_index("name")
                .to_dict("index")
            )

    rng = np.random.RandomState(42)

    # ── Phase 1: lineup-level simulations ────────────────────────────────────
    lineup_totals: Dict[Any, np.ndarray] = {}  # lineup_index → (n_sims,) array

    for lu_id, grp in lu.groupby("lineup_index"):
        projs_list: List[float] = []
        sals_list: List[float] = []

        for _, player_row in grp.iterrows():
            pname = player_row["player_name"]
            pool_row = pool_lookup.get(pname, {})

            def _get_field(field: str, default=None, _pr=player_row, _pool=pool_row):
                val = _pr.get(field, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = _pool.get(field, default)
                return val

            proj = float(pd.to_numeric(_get_field("proj", 0), errors="coerce") or 0)
            sal = float(pd.to_numeric(_get_field("salary", 6000), errors="coerce") or 6000)
            projs_list.append(proj)
            sals_list.append(sal)

        if not projs_list or all(p == 0 for p in projs_list):
            continue

        projs_arr = np.array(projs_list)
        sals_arr = np.array(sals_list)

        # PGA: check if pool has DataGolf std_dev for per-player variance
        _pat_dg_std = None
        if not pool_df.empty:
            _std_col = pool_df.get("std_dev")
            if _std_col is not None:
                _std_vals = pd.to_numeric(_std_col, errors="coerce").fillna(0)
                if (_std_vals > 0).any():
                    # Build std_dev array matching player order in this lineup
                    _std_lookup = dict(zip(
                        pool_df.get("player_name", pool_df.get("name", pd.Series())),
                        _std_vals,
                    ))
                    _std_arr = np.array([
                        float(_std_lookup.get(grp.iloc[j]["player_name"], 0))
                        for j in range(len(grp))
                    ])
                    if (_std_arr > 0).any():
                        _pat_dg_std = _std_arr

        # Empirical variance (PGA uses DataGolf std_dev; NBA uses salary-bracket model)
        stds_arr = compute_empirical_std(projs_arr, sals_arr, std_dev=_pat_dg_std)

        sim_matrix = rng.normal(
            loc=projs_arr[None, :],
            scale=stds_arr[None, :],
            size=(n_sims, len(projs_arr)),
        )
        sim_matrix = np.clip(sim_matrix, 0, None)

        # Apply calibration knobs
        above = sim_matrix > projs_arr[None, :]
        sim_matrix = np.where(
            above,
            projs_arr[None, :] + (sim_matrix - projs_arr[None, :]) * ceiling_boost,
            projs_arr[None, :] - (projs_arr[None, :] - sim_matrix) * floor_dampen,
        )
        sim_matrix = np.clip(sim_matrix, 0, None)
        lineup_totals[lu_id] = sim_matrix.sum(axis=1)

    if not lineup_totals:
        return pd.DataFrame()

    # ── Phase 2: contest-level thresholds from the pooled field ──────────────
    all_totals = np.concatenate(list(lineup_totals.values()))
    mean_lineup_proj = float(np.mean([lt.mean() for lt in lineup_totals.values()]))

    if smash_thr_ratio is not None:
        contest_smash = float(smash_thr_ratio) * mean_lineup_proj
    else:
        contest_smash = float(np.percentile(all_totals, CONTEST_SMASH_PERCENTILE))

    if bust_thr_ratio is not None:
        contest_bust = float(bust_thr_ratio) * mean_lineup_proj
    else:
        contest_bust = float(np.percentile(all_totals, CONTEST_BUST_PERCENTILE))

    # ── Phase 3: per-player metrics from lineup scores ────────────────────────
    # Map each player → list of lineup_ids they appear in
    player_to_lineups: Dict[str, List[Any]] = {}
    for lu_id, grp in lu.groupby("lineup_index"):
        if lu_id not in lineup_totals:
            continue
        for pname in grp["player_name"].dropna().tolist():
            player_to_lineups.setdefault(pname, []).append(lu_id)

    # Collect player metadata from any lineup row (first occurrence)
    player_meta: Dict[str, dict] = {}
    for _, row in lu.drop_duplicates(subset=["player_name"]).iterrows():
        pname = row["player_name"]
        pool_row = pool_lookup.get(pname, {})

        def _get_meta(field: str, default=None, _r=row, _pool=pool_row):
            val = _r.get(field, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = _pool.get(field, default)
            return val

        player_meta[pname] = {
            "proj": float(pd.to_numeric(_get_meta("proj", 0), errors="coerce") or 0),
            "salary": float(pd.to_numeric(_get_meta("salary", 0), errors="coerce") or 0),
            "own_pct": float(pd.to_numeric(_get_meta("own%", 0), errors="coerce") or 0),
        }

    rows = []
    for pname, lu_ids in player_to_lineups.items():
        meta = player_meta.get(pname, {})
        proj = meta.get("proj", 0.0)
        if proj <= 0:
            continue
        salary = meta.get("salary", 0.0)
        own_pct = meta.get("own_pct", 0.0)

        # Pool all contest sims for lineups containing this player
        player_scores = np.concatenate([lineup_totals[li] for li in lu_ids])

        smash_pct = float((player_scores >= contest_smash).mean() * 100.0)
        bust_pct = float((player_scores <= contest_bust).mean() * 100.0)

        # Leverage: NaN when Own% < MIN_OWNERSHIP_FOR_LEVERAGE to avoid spurious infinity-like values
        if own_pct >= MIN_OWNERSHIP_FOR_LEVERAGE:
            leverage: float = smash_pct / own_pct
        else:
            leverage = float("nan")

        rows.append({
            "Player": pname,
            "Proj": round(proj, 1),
            "Salary": int(salary),
            "Own%": round(own_pct, 1),
            "Smash%": round(smash_pct, 1),
            "Bust%": round(bust_pct, 1),
            "Leverage Score": round(leverage, 2) if not np.isnan(leverage) else float("nan"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Value Trap: Bust% > 40% AND Salary > median salary
    if df["Salary"].sum() > 0:
        median_sal = df["Salary"].median()
        df["Value Trap"] = (df["Bust%"] > 40.0) & (df["Salary"] > median_sal)
    else:
        df["Value Trap"] = False

    # High Leverage flag: LeverageScore >= 3 AND Own% <= 15
    def _flag(r: "pd.Series") -> str:
        lev = r["Leverage Score"]
        if not np.isnan(lev) and lev >= 3.0 and r["Own%"] <= 15.0:
            return "🔥 HIGH LEVERAGE"
        return ""

    df["Flag"] = df.apply(_flag, axis=1)

    return df.sort_values("Leverage Score", ascending=False, na_position="last").reset_index(drop=True)


def build_sim_player_accuracy_table(
    pool_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    hit_threshold: float = 10.0,
) -> Dict[str, Any]:
    """Build a per-player sim projection vs actuals accuracy table.

    Joins the player pool projections to actual fantasy-point results, then
    computes a set of calibration KPIs to measure how well the sim's input
    projections matched reality.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with projections.  Must contain a name column
        (``player_name`` or ``name``) and a ``proj`` column.
    actuals_df : pd.DataFrame
        Actual results.  Must contain a name column (``player_name`` or
        ``name``) and an actuals column (``actual`` or ``actual_fp``).
    hit_threshold : float, optional
        Absolute-error threshold (in fantasy points) used for the hit-rate
        metric.  A player is a "hit" when ``|error| <= hit_threshold``
        (default 10.0 FP).

    Returns
    -------
    dict
        ``player_df``   — per-player DataFrame with columns:
                          ``name``, ``proj``, ``actual``, ``error``,
                          ``abs_error``, ``pct_error``.

        ``mae``         — Mean Absolute Error: mean(|proj − actual|).

        ``rmse``        — Root Mean Squared Error.

        ``bias``        — Mean error (proj − actual); positive = over-projected.

        ``hit_rate``    — Percentage of players where |error| ≤ ``hit_threshold``.

        ``r2``          — R² between proj and actual (0–1; higher is better).

        ``n_players``   — Number of matched players used in the calculation.
    """
    _empty: Dict[str, Any] = {
        "player_df": pd.DataFrame(),
        "mae": 0.0,
        "rmse": 0.0,
        "bias": 0.0,
        "hit_rate": 0.0,
        "r2": 0.0,
        "n_players": 0,
    }

    if pool_df.empty or actuals_df.empty:
        return _empty

    # Normalise pool — accept 'player_name' or 'name'
    pool = pool_df.copy()
    if "player_name" in pool.columns and "name" not in pool.columns:
        pool = pool.rename(columns={"player_name": "name"})
    if "name" not in pool.columns or "proj" not in pool.columns:
        return _empty

    # Normalise actuals — accept 'player_name'/'name' and 'actual'/'actual_fp'
    acts = actuals_df.copy()
    if "player_name" in acts.columns and "name" not in acts.columns:
        acts = acts.rename(columns={"player_name": "name"})
    if "actual_fp" in acts.columns and "actual" not in acts.columns:
        acts = acts.rename(columns={"actual_fp": "actual"})
    if "name" not in acts.columns or "actual" not in acts.columns:
        return _empty

    pool_sub = pool[["name", "proj"]].drop_duplicates(subset=["name"])
    pool_sub["proj"] = pd.to_numeric(pool_sub["proj"], errors="coerce")

    # Average actuals per player in case the contest-results export lists the
    # same player multiple times (e.g. duplicate rows in an entry CSV)
    acts_sub = (
        acts[["name", "actual"]]
        .assign(actual=lambda d: pd.to_numeric(d["actual"], errors="coerce"))
        .groupby("name", as_index=False)["actual"]
        .mean()
    )

    merged = pool_sub.merge(acts_sub, on="name", how="inner").dropna(
        subset=["proj", "actual"]
    )

    if merged.empty:
        return _empty

    merged = merged.copy()
    merged["error"] = merged["proj"] - merged["actual"]
    merged["abs_error"] = merged["error"].abs()
    merged["pct_error"] = np.where(
        merged["actual"] != 0,
        (merged["error"] / merged["actual"].abs()) * 100.0,
        np.nan,
    )
    merged = merged.reset_index(drop=True)

    errors = merged["error"]
    n = len(merged)

    mae = float(merged["abs_error"].mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    bias = float(errors.mean())
    hit_rate = float((merged["abs_error"] <= hit_threshold).mean() * 100.0)

    # R² between proj and actual
    ss_res = float((errors ** 2).sum())
    ss_tot = float(((merged["actual"] - merged["actual"].mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "player_df": merged[["name", "proj", "actual", "error", "abs_error", "pct_error"]],
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "hit_rate": round(hit_rate, 1),
        "r2": round(r2, 3),
        "n_players": n,
    }


# ---------------------------------------------------------------------------
# Sims Pipeline
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sims"


def run_sims_pipeline(
    pool: pd.DataFrame,
    lineups_df: pd.DataFrame,
    contest_type: str = "GPP_20",
    n_sims: int = 10000,
    variance: float = 1.0,
    slate_date: str = "",
    draft_group_id: Optional[int] = None,
    output_dir: Optional[str] = None,
    top_x_pct: float = 0.15,
    itm_pct: float = 0.50,
) -> pd.DataFrame:
    """Run the full sims pipeline for a slate and persist lineup-level metrics.

    Generates lineup-level metrics (projection, total_pown, top_X% finish
    rate, ITM rate, sim ROI, leverage) for every lineup in *lineups_df*,
    computes the YakOS Sim Rating, and writes the results to a Parquet file
    under *output_dir* for later calibration.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool with ``player_name``, ``proj``, ``ownership`` columns.
    lineups_df : pd.DataFrame
        Lineup rows in long format (one player per row) with at least
        ``lineup_index`` and ``player_name`` columns.
    contest_type : str
        Contest archetype (e.g. "GPP_150", "GPP_20", "SE_3MAX", "CASH").
    n_sims : int
        Monte Carlo iterations.
    variance : float
        Variance multiplier applied to player score distributions.
    slate_date : str
        ISO date string used for the output filename.
    draft_group_id : int, optional
        DK draft group ID; stored in the output for join-back during
        calibration.
    output_dir : str, optional
        Directory to write the Parquet file.  Defaults to
        ``data/sims/`` relative to the repo root.
    top_x_pct : float
        Fraction of the field considered "top-X%".  Default 0.15 (top 15%).
    itm_pct : float
        In-the-money fraction.  Default 0.50 (cash-line).

    Returns
    -------
    pd.DataFrame
        Lineup-level pipeline output with one row per lineup_index.
        Columns: ``lineup_index``, ``projection``, ``total_pown``,
        ``top_x_rate``, ``itm_rate``, ``sim_roi``, ``leverage``,
        ``yakos_sim_rating``, ``rating_bucket``.
    """
    if pool.empty or lineups_df.empty:
        return pd.DataFrame()

    # Require the canonical projected ownership column.
    # apply_ownership() (in yak_core/ownership.py) populates own_proj from
    # external POWN data, legacy columns, or a salary-rank fallback.
    if "own_proj" not in pool.columns:
        raise ValueError(
            "Expected 'own_proj' column on pool before running sims. "
            "Call apply_ownership() (yak_core/ownership.py) to populate "
            "own_proj from external POWN data or a salary-rank estimate."
        )

    # Build a player-level projection/ownership lookup
    p_proj: Dict[str, float] = {}
    p_own: Dict[str, float] = {}

    for _, row in pool.iterrows():
        pname = str(row.get("player_name", ""))
        if not pname:
            continue
        proj_val = float(row.get("proj", 0) or 0)
        own_val = float(row.get("own_proj", 5.0) or 5.0)  # ownership column used here: pool["own_proj"] (canonical projected ownership)
        p_proj[pname] = proj_val
        p_own[pname] = own_val

    # Monte Carlo: simulate lineup totals
    rng = np.random.default_rng(seed=42)
    lineup_indices = sorted(lineups_df["lineup_index"].unique().tolist()) if "lineup_index" in lineups_df.columns else []
    records: List[Dict[str, Any]] = []

    # Pre-simulate all player scores for all iterations
    # Uses empirical salary-bracket variance model (calibrated from 21-slate backtest)
    all_player_sims: Dict[str, np.ndarray] = {}
    _all_names = list(p_proj.keys())
    _all_projs = np.array([p_proj[n] for n in _all_names])
    _all_sals = np.array([p_own.get(n, 6000) for n in _all_names])  # salary lookup below
    # Build salary lookup from pool
    p_sal: Dict[str, float] = {}
    for _, row in pool.iterrows():
        pname = str(row.get("player_name", ""))
        if pname:
            p_sal[pname] = float(row.get("salary", 6000) or 6000)
    _all_sals = np.array([p_sal.get(n, 6000) for n in _all_names])
    # Derive contest_mode for variance dampening
    _pipe_contest_mode = contest_type.strip().lower().replace(" ", "_")
    _pipe_cm_map = {"gpp_main": "gpp", "gpp_early": "gpp", "gpp_late": "gpp", "cash_main": "cash"}
    _pipe_contest_mode = _pipe_cm_map.get(_pipe_contest_mode, _pipe_contest_mode)
    # PGA: use DataGolf per-player std_dev when available
    _pipe_dg_std = None
    if "std_dev" in pool.columns:
        _std_series = pd.to_numeric(pool["std_dev"], errors="coerce").fillna(0)
        if (_std_series > 0).any():
            _std_map = dict(zip(pool["player_name"], _std_series))
            _pipe_dg_std = np.array([float(_std_map.get(n, 0)) for n in _all_names])
    _all_stds = compute_empirical_std(_all_projs, _all_sals, variance_mult=variance, contest_mode=_pipe_contest_mode, std_dev=_pipe_dg_std)
    for i, pname in enumerate(_all_names):
        all_player_sims[pname] = rng.normal(_all_projs[i], _all_stds[i], n_sims)

    # Build per-lineup totals across all simulations
    all_lineup_totals: Dict[Any, np.ndarray] = {}
    for lu_idx in lineup_indices:
        lu = lineups_df[lineups_df["lineup_index"] == lu_idx]
        players = lu["player_name"].dropna().tolist() if "player_name" in lu.columns else []
        lu_totals = np.zeros(n_sims)
        for pname in players:
            if pname in all_player_sims:
                lu_totals += all_player_sims[pname]
        all_lineup_totals[lu_idx] = lu_totals

    # Derive field-wide percentile thresholds from the pooled sim distributions
    if all_lineup_totals:
        pooled_totals = np.concatenate(list(all_lineup_totals.values()))
        top_x_threshold = float(np.nanpercentile(pooled_totals, (1.0 - top_x_pct) * 100))
        itm_threshold = float(np.nanpercentile(pooled_totals, (1.0 - itm_pct) * 100))
    else:
        top_x_threshold = 0.0
        itm_threshold = 0.0

    for lu_idx in lineup_indices:
        lu = lineups_df[lineups_df["lineup_index"] == lu_idx]
        players = lu["player_name"].dropna().tolist() if "player_name" in lu.columns else []

        # Lineup-level base metrics
        projection = sum(p_proj.get(p, 0) for p in players)
        total_pown_raw = sum(p_own.get(p, 0) for p in players) / max(len(players), 1)
        total_pown_frac = total_pown_raw / 100.0  # convert pct to fraction

        lu_totals = all_lineup_totals.get(lu_idx, np.zeros(n_sims))
        median_total = float(np.nanmedian(lu_totals))

        # Top-X% finish rate (fraction of sims where lineup beats top_x threshold)
        top_x_rate = float(np.mean(lu_totals >= top_x_threshold))

        # ITM rate (fraction of sims where lineup beats cash line)
        itm_rate = float(np.mean(lu_totals >= itm_threshold))

        # Sim ROI = (median_sim_total - projected_total) / projected_total
        sim_roi = float((median_total - projection) / max(projection, 1.0))

        # Leverage: top_x_rate relative to field-average ownership
        avg_own = float(np.mean([p_own.get(p, 5.0) for p in players])) if players else 5.0
        own_frac = max(avg_own / 100.0, 0.01)
        leverage = float(top_x_rate / own_frac) if own_frac > 0 else 1.0

        records.append({
            "lineup_index":  lu_idx,
            "projection":    round(projection, 2),
            "total_pown":    round(total_pown_frac, 4),
            "top_x_rate":   round(top_x_rate, 4),
            "itm_rate":      round(itm_rate, 4),
            "sim_roi":       round(sim_roi, 4),
            "leverage":      round(leverage, 3),
            "contest_type":  contest_type,
            "slate_date":    slate_date,
            "draft_group_id": draft_group_id,
        })

    if not records:
        return pd.DataFrame()

    pipeline_df = pd.DataFrame(records)
    pipeline_df = compute_pipeline_ratings(pipeline_df, contest_type=contest_type)

    # Persist to Parquet
    _save_pipeline_output(pipeline_df, slate_date=slate_date,
                          draft_group_id=draft_group_id,
                          contest_type=contest_type,
                          output_dir=output_dir)

    return pipeline_df


def _save_pipeline_output(
    df: pd.DataFrame,
    slate_date: str = "",
    draft_group_id: Optional[int] = None,
    contest_type: str = "",
    output_dir: Optional[str] = None,
) -> Optional[Path]:
    """Write pipeline output to a Parquet file.

    Returns the Path written, or None if write failed.
    """
    try:
        base = Path(output_dir) if output_dir else _DEFAULT_DATA_DIR
        base.mkdir(parents=True, exist_ok=True)
        dg_part = f"_dg{draft_group_id}" if draft_group_id else ""
        ct_part = f"_{contest_type.lower()}" if contest_type else ""
        date_part = slate_date.replace("-", "") if slate_date else "nodate"
        fname = f"sims_{date_part}{dg_part}{ct_part}.parquet"
        out_path = base / fname
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        return None


def load_pipeline_output(
    slate_date: str = "",
    draft_group_id: Optional[int] = None,
    contest_type: str = "",
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load a previously saved pipeline output Parquet file.

    Parameters
    ----------
    slate_date : str
        ISO date string matching the filename.
    draft_group_id : int, optional
        DK draft group ID.
    contest_type : str
        Contest archetype string.
    output_dir : str, optional
        Directory to search.  Defaults to ``data/sims/``.

    Returns
    -------
    pd.DataFrame
        Pipeline output, or empty DataFrame if not found.
    """
    base = Path(output_dir) if output_dir else _DEFAULT_DATA_DIR
    dg_part = f"_dg{draft_group_id}" if draft_group_id else ""
    ct_part = f"_{contest_type.lower()}" if contest_type else ""
    date_part = slate_date.replace("-", "") if slate_date else "nodate"
    fname = f"sims_{date_part}{dg_part}{ct_part}.parquet"
    path = base / fname
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def run_calibration_pipeline(
    historical_sims_dir: Optional[str] = None,
    dk_results_df: Optional[pd.DataFrame] = None,
    min_bucket_samples: int = 20,
) -> pd.DataFrame:
    """Load historical sims output and compute bucket-level realized ROI.

    Joins the pipeline output (from ``run_sims_pipeline``) to actual DK
    results (payouts, finish positions) and computes bucket-level realized
    ROI and top-finish rates.  Results are accumulated across all Parquet
    files found in *historical_sims_dir*.

    This function performs bulk bucket updates only — it does not
    micro-adjust individual lineup weights.  The intention is to collect
    sufficient volume per bucket (≥ *min_bucket_samples*) before updating
    rating weights.

    Parameters
    ----------
    historical_sims_dir : str, optional
        Directory containing historical ``sims_*.parquet`` files.
        Defaults to ``data/sims/``.
    dk_results_df : pd.DataFrame, optional
        Actual DK contest results.  Must contain ``lineup_index``
        (or ``entry_id``), ``payout`` (float), ``finish_position`` (int),
        and ``total_entries`` (int) columns.  When *None*, only the pipeline
        metrics are aggregated (no realized ROI).
    min_bucket_samples : int
        Minimum rows per bucket before that bucket's summary is reported.

    Returns
    -------
    pd.DataFrame
        Bucket-level calibration summary with columns:
        ``rating_bucket``, ``n``, ``avg_yakos_rating``,
        ``avg_top_x_rate``, ``avg_itm_rate``, ``avg_sim_roi``,
        ``realized_roi`` (if DK results provided),
        ``top_finish_rate`` (if DK results provided),
        ``meets_threshold`` (bool: n >= min_bucket_samples).
    """
    base = Path(historical_sims_dir) if historical_sims_dir else _DEFAULT_DATA_DIR
    parquet_files = sorted(base.glob("sims_*.parquet")) if base.exists() else []

    frames: List[pd.DataFrame] = []
    for pf in parquet_files:
        try:
            frames.append(pd.read_parquet(pf))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    all_sims = pd.concat(frames, ignore_index=True)

    # Join actual DK results if provided
    if dk_results_df is not None and not dk_results_df.empty:
        dk = dk_results_df.copy()
        # Normalise join key
        if "entry_id" in dk.columns and "lineup_index" not in dk.columns:
            dk = dk.rename(columns={"entry_id": "lineup_index"})
        if "lineup_index" in dk.columns:
            all_sims = all_sims.merge(
                dk[["lineup_index"] + [c for c in ["payout", "finish_position", "total_entries"]
                                       if c in dk.columns]],
                on="lineup_index",
                how="left",
            )
            if "payout" in all_sims.columns and "finish_position" in all_sims.columns:
                all_sims["realized_roi"] = (
                    pd.to_numeric(all_sims["payout"], errors="coerce").fillna(0) - 1.0
                )
                all_sims["top_finish"] = (
                    pd.to_numeric(all_sims["finish_position"], errors="coerce").le(
                        pd.to_numeric(
                            all_sims.get("total_entries", pd.Series(dtype=float)),
                            errors="coerce",
                        ).fillna(1000) * 0.15
                    )
                ).astype(int)

    if "rating_bucket" not in all_sims.columns:
        return pd.DataFrame()

    agg_dict: Dict[str, Any] = {
        "yakos_sim_rating":  ("yakos_sim_rating", "mean"),
        "avg_top_x_rate":    ("top_x_rate", "mean"),
        "avg_itm_rate":      ("itm_rate", "mean"),
        "avg_sim_roi":       ("sim_roi", "mean"),
    }
    if "realized_roi" in all_sims.columns:
        agg_dict["realized_roi"] = ("realized_roi", "mean")
    if "top_finish" in all_sims.columns:
        agg_dict["top_finish_rate"] = ("top_finish", "mean")

    # Build named aggregation
    named_agg = {k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg_dict.items()}

    summary = (
        all_sims.groupby("rating_bucket", as_index=False)
        .agg(n=("lineup_index", "count"), **named_agg)
        .rename(columns={"yakos_sim_rating": "avg_yakos_rating"})
    )
    summary["meets_threshold"] = summary["n"] >= min_bucket_samples

    # Round for readability
    for col in ["avg_yakos_rating", "avg_top_x_rate", "avg_itm_rate", "avg_sim_roi",
                "realized_roi", "top_finish_rate"]:
        if col in summary.columns:
            summary[col] = summary[col].round(4)

    return summary.sort_values("rating_bucket").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Contest-type sims wrapper
# ---------------------------------------------------------------------------

# Contest-specific parameter presets.
# field_size: approximate number of entrants (used for payout/odds estimates).
# top_x_pct: fraction of field considered "top-X%".
# itm_pct: fraction of field that finishes in-the-money.
# stack_min: minimum players from same team encouraged in a lineup.
_CONTEST_PARAMS: Dict[str, Dict[str, Any]] = {
    "GPP_150": {"field_size": 150, "top_x_pct": 0.10, "itm_pct": 0.25, "stack_min": 2},
    "GPP_20":  {"field_size": 20,  "top_x_pct": 0.15, "itm_pct": 0.35, "stack_min": 2},
    "SE_3MAX": {"field_size": 3,   "top_x_pct": 0.33, "itm_pct": 0.50, "stack_min": 1},
    "CASH":    {"field_size": 2,   "top_x_pct": 0.50, "itm_pct": 0.50, "stack_min": 0},
}
_DEFAULT_CONTEST_PARAMS: Dict[str, Any] = {
    "field_size": 20,
    "top_x_pct": 0.15,
    "itm_pct": 0.35,
    "stack_min": 1,
}


def run_sims_for_contest_type(
    edge_df: pd.DataFrame,
    contest_type: str,
    lineup_filter: Optional[List[str]] = None,
    n_sims: int = 10000,
    variance: float = 1.0,
) -> pd.DataFrame:
    """Run player-level Monte Carlo sims scoped to a specific contest type.

    Uses contest-specific parameters (field size, payout structure, stack rules)
    to shape the smash/bust thresholds and compute player-level sim metrics.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Player edge metrics table (output of ``compute_edge_metrics``).
        Required columns: ``player_name``, ``proj``, ``own_pct``.
        Optional: ``floor``, ``ceil``.
    contest_type : str
        Contest archetype key: ``"GPP_150"``, ``"GPP_20"``, ``"SE_3MAX"``,
        ``"CASH"``.  Unknown values fall back to ``_DEFAULT_CONTEST_PARAMS``.
    lineup_filter : list of str, optional
        When provided, restrict the simulation to only the player names in
        this list.  Useful for backtesting a specific set of lineups.
    n_sims : int
        Number of Monte Carlo iterations (default 10 000).
    variance : float
        Variance multiplier applied to player score distributions (default 1.0).

    Returns
    -------
    pd.DataFrame
        Player-level sim results with columns:
        ``player_name``, ``proj``, ``own_pct``, ``smash_prob``, ``bust_prob``,
        ``leverage``, ``contest_type``.
        Always sorted by ``smash_prob`` descending.
    """
    if edge_df is None or edge_df.empty:
        return pd.DataFrame(columns=["player_name", "proj", "own_pct",
                                     "smash_prob", "bust_prob", "leverage",
                                     "contest_type"])

    params = _CONTEST_PARAMS.get(contest_type.upper(), _DEFAULT_CONTEST_PARAMS)
    top_x_pct: float = params["top_x_pct"]

    df = edge_df.copy()

    # Apply lineup filter when provided
    if lineup_filter:
        df = df[df["player_name"].isin(lineup_filter)].copy()

    if df.empty:
        return pd.DataFrame(columns=["player_name", "proj", "own_pct",
                                     "smash_prob", "bust_prob", "leverage",
                                     "contest_type"])

    proj = pd.to_numeric(df.get("proj", 0), errors="coerce").fillna(0)
    salary = pd.to_numeric(df.get("salary", pd.Series(6000, index=df.index)), errors="coerce").fillna(6000)
    own = pd.to_numeric(df.get("own_pct", 5.0), errors="coerce").fillna(5.0)

    rng = np.random.default_rng(seed=42)
    # Empirical variance — PGA uses DataGolf std_dev when available
    _ct_mode = contest_type.strip().lower().replace(" ", "_")
    _ct_dg_std = None
    if "std_dev" in df.columns:
        _ct_std = pd.to_numeric(df["std_dev"], errors="coerce").fillna(0).values
        if (_ct_std > 0).any():
            _ct_dg_std = _ct_std
    std = compute_empirical_std(proj.values, salary.values, variance_mult=variance, contest_mode=_ct_mode, std_dev=_ct_dg_std)
    proj_vals = proj.values

    # Simulate n_sims score draws per player
    scores = rng.normal(loc=proj_vals[:, None], scale=std[:, None], size=(len(df), n_sims))

    # Derive thresholds from the contest-type top-X percentile
    field_median = float(np.median(scores))
    smash_thresh = float(np.percentile(scores, (1.0 - top_x_pct) * 100))
    bust_thresh = float(np.percentile(scores, params["itm_pct"] * 50))

    smash_prob = (scores >= smash_thresh).mean(axis=1)
    bust_prob = (scores <= bust_thresh).mean(axis=1)

    own_safe = own.values.clip(0.1)
    leverage = np.where(own.values >= 0.1, proj_vals / own_safe, np.nan)

    out = df[["player_name"]].copy()
    out["proj"] = proj_vals
    out["own_pct"] = own.values
    out["smash_prob"] = smash_prob
    out["bust_prob"] = bust_prob
    out["leverage"] = leverage
    out["contest_type"] = contest_type
    out["field_median"] = field_median

    return out.sort_values("smash_prob", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

_SIMS_ROUND_COLS = [
    "proj",
    "floor",
    "ceil",
    "proj_minutes",
    "ownership",
    "smash_prob",
    "bust_prob",
    "leverage",
]


def prepare_sims_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and format a sims results DataFrame for display in Streamlit.

    Steps applied (in order):

    1. Drop players who did not play — rows where ``mp_actual`` is 0 or NaN
       (column is skipped when absent so player-level results without actuals
       are unaffected).
    2. Convert ``ownership`` from a 0–1 fraction to a percentage value.
       The conversion is applied only when all non-null ownership values are
       ≤ 1, which avoids double-converting data already stored as percentages
       (e.g. 25.0 rather than 0.25).
    3. Round :data:`_SIMS_ROUND_COLS` to 1 decimal place.
    4. Rename ``ownership`` → ``own_pct`` for UI clarity.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sims output.  Expected columns (all optional beyond what your
        pipeline produces): ``mp_actual``, ``proj``, ``floor``, ``ceil``,
        ``ownership``, ``smash_prob``, ``bust_prob``, ``leverage``.

    Returns
    -------
    pd.DataFrame
        Cleaned, display-ready copy of *df*.
    """
    df = df.copy()

    # 1. Remove DNPs when mp_actual is available
    if "mp_actual" in df.columns:
        df = df[pd.to_numeric(df["mp_actual"], errors="coerce").fillna(0) > 0]

    # 2. Convert ownership 0–1 → percentage (guard against already-pct data)
    if "ownership" in df.columns:
        own_numeric = pd.to_numeric(df["ownership"], errors="coerce")
        if own_numeric.dropna().le(1).all():
            df["ownership"] = own_numeric * 100

    # 3. Round key numeric columns to 1 decimal place
    existing_round_cols = [c for c in _SIMS_ROUND_COLS if c in df.columns]
    df[existing_round_cols] = df[existing_round_cols].round(1)

    # Cast salary to int — DK salaries are always whole-dollar amounts.
    if "salary" in df.columns:
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)

    # 4. Rename for UI friendliness
    rename_map = {}
    if "ownership" in df.columns:
        rename_map["ownership"] = "own_pct"
    if "proj_minutes" in df.columns:
        rename_map["proj_minutes"] = "Mins"
    if rename_map:
        df = df.rename(columns=rename_map)

    return df
