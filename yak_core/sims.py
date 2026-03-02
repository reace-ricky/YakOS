"""Monte Carlo simulation for YakOS DFS optimizer."""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
) -> pd.DataFrame:
    """Compute the ``sim_eligible`` column for every player in the pool.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Expected columns (all optional): ``status``, ``minutes``,
        ``team``.
    min_proj_minutes : float
        Minimum projected minutes required for ``sim_eligible = True``
        (default 4.0).  Players with ``minutes ≤ min_proj_minutes`` are
        excluded.  Set to ``0`` to skip the minutes filter.
    exclude_out_ir : bool
        If ``True`` (default), players whose ``status`` matches a known
        ineligible value (OUT, IR, G-League, Suspended, etc.) are excluded.
    today_teams : list of str, optional
        If provided, players whose ``team`` is **not** in this list are
        excluded.

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
    if "minutes" in df.columns and min_proj_minutes > 0:
        mins = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
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
) -> pd.DataFrame:
    """Run Monte Carlo simulations on lineup projections.

    For each lineup, simulates ``n_sims`` outcomes by sampling from
    per-player normal distributions (mean=proj, std derived from
    ceil/floor when available).  Smash and bust thresholds are computed
    dynamically from each lineup's own sim distribution using
    :func:`summarize_lineup_sims` and :func:`compute_thresholds`.

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
        Contest archetype used to derive per-lineup smash/bust thresholds
        (default ``ContestType.GPP_LARGE``).

    Returns
    -------
    pd.DataFrame
        Per-lineup summary with columns:
        ``lineup_index``, ``sim_mean``, ``sim_std``,
        ``median_points``, ``sim_p85``, ``sim_p15``,
        ``smash_threshold``, ``bust_threshold``,
        ``smash_pct``, ``bust_pct``, ``contest_type``.
        ``smash_prob`` and ``bust_prob`` are retained as aliases for
        ``smash_pct`` and ``bust_pct`` for backwards compatibility.
    """
    if lineups_df.empty or "lineup_index" not in lineups_df.columns:
        return pd.DataFrame()

    vol_map = {"low": 0.10, "standard": 0.18, "high": 0.28}
    default_vol = vol_map.get(volatility_mode, 0.18)

    rng = np.random.RandomState(42)
    results = []

    for lu_id, grp in lineups_df.groupby("lineup_index"):
        projs = grp["proj"].fillna(0).values.astype(float)

        # Derive per-player std from ceil/floor when available
        if "ceil" in grp.columns and "floor" in grp.columns:
            ceil_series = pd.to_numeric(grp["ceil"], errors="coerce")
            floor_series = pd.to_numeric(grp["floor"], errors="coerce")
            ceil_v = ceil_series.where(ceil_series.notna(), other=pd.Series(projs * 1.3, index=grp.index)).values.astype(float)
            floor_v = floor_series.where(floor_series.notna(), other=pd.Series(projs * 0.7, index=grp.index)).values.astype(float)
            stds = (ceil_v - floor_v) / 4.0
            stds = np.clip(stds, projs * 0.05, projs * 0.6)
        else:
            stds = projs * default_vol

        # (n_sims × n_players) outcome matrix
        sim_matrix = rng.normal(
            loc=projs[None, :],
            scale=stds[None, :],
            size=(n_sims, len(projs)),
        )
        sim_matrix = np.clip(sim_matrix, 0, None)
        totals = sim_matrix.sum(axis=1)

        # Dynamic thresholds from this lineup's own distribution
        summary = summarize_lineup_sims(totals.tolist())
        compute_thresholds(summary, contest_type)
        smash_pct, bust_pct = compute_smash_bust_rates(
            totals.tolist(), summary.smash_threshold, summary.bust_threshold
        )

        results.append({
            "lineup_index": lu_id,
            "sim_mean": round(float(totals.mean()), 2),
            "sim_std": round(float(totals.std()), 2),
            "median_points": round(summary.median_score, 2),
            "sim_p85": round(summary.p85_score, 2),
            "sim_p15": round(summary.p15_score, 2),
            "smash_threshold": round(summary.smash_threshold, 2),
            "bust_threshold": round(summary.bust_threshold, 2),
            "smash_pct": round(smash_pct, 3),
            "bust_pct": round(bust_pct, 3),
            # Backwards-compatible aliases
            "smash_prob": round(smash_pct, 3),
            "bust_prob": round(bust_pct, 3),
            "contest_type": contest_type.value,
        })

    return pd.DataFrame(results)


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

    For each player that appears in *lineup_df*, simulates *n_sims* individual
    outcomes from a normal distribution centred on their projection, then
    classifies each outcome as a **smash** (``outcome ≥ smash_threshold × proj``)
    or a **bust** (``outcome ≤ bust_threshold × proj``).

    Leverage Score is defined as ``Smash% / Own%`` (higher means the player is
    a bigger upside play relative to expected ownership — more leverage in GPP).

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Must include a name column (``player_name`` or ``name``)
        and a ``proj`` column.  Optional: ``salary``,
        ``ownership`` / ``own%``, ``ceil``, ``floor``.
    lineup_df : pd.DataFrame
        Long-format lineup table (as returned by ``run_optimizer``).  Must
        include a name column so we know which players appear in the lineups.
    n_sims : int, optional
        Per-player simulation iterations (default 500).
    cal_knobs : dict, optional
        Calibration knobs.  Supported keys:

        * ``ceiling_boost``   (float, default 1.0) — multiply upside outcomes
        * ``floor_dampen``    (float, default 1.0) — compress downside outcomes
        * ``smash_threshold`` (float, optional) — ratio multiplier: smash if outcome ≥ this × proj.
          When absent, the 90th percentile of each player's sim outcomes is used (top 10%).
        * ``bust_threshold``  (float, optional) — ratio multiplier: bust if outcome ≤ this × proj.
          When absent, the 30th percentile of each player's sim outcomes is used (bottom 30%).

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
    # smash_threshold / bust_threshold in cal_knobs are ratio multipliers (e.g. 1.3 = 130% of proj).
    # When absent (None), percentile-based defaults are used instead:
    #   smash = p90 of player's sim outcomes (top 10%)
    #   bust  = p30 of player's sim outcomes (bottom 30%)
    smash_thr = knobs.get("smash_threshold")   # None → use p90
    bust_thr = knobs.get("bust_threshold")     # None → use p30

    # Find name column in lineup_df — this defines which players are in the lineup
    lu_name_col = next(
        (c for c in ("player_name", "name") if c in lineup_df.columns), None
    )
    if lu_name_col is None:
        return pd.DataFrame()

    # Use lineup_df as the authoritative source of players to simulate.
    # Deduplicate so each player is only simulated once regardless of how many
    # lineups they appear in.
    lu_uniq = (
        lineup_df
        .drop_duplicates(subset=[lu_name_col])
        .dropna(subset=[lu_name_col])
        .copy()
    )
    if lu_uniq.empty:
        return pd.DataFrame()

    if lu_name_col != "name":
        lu_uniq = lu_uniq.rename(columns={lu_name_col: "name"})

    # Build a pool lookup keyed by name to enrich lineup rows with any
    # supplemental columns (ownership, etc.) not already present in lineup_df.
    # Only the columns relevant for simulation are kept to minimise memory use.
    _pool_sim_cols = ("name", "proj", "salary", "own%", "ownership", "Own%", "proj_own", "ceil", "floor")
    pool_lookup: dict = {}
    if not pool_df.empty:
        pool = pool_df.copy()
        if "player_name" in pool.columns and "name" not in pool.columns:
            pool = pool.rename(columns={"player_name": "name"})
        # Normalise ownership column in pool — prefer own%, then ownership/Own%, then proj_own
        for _src in ("ownership", "Own%", "proj_own"):
            if _src in pool.columns and "own%" not in pool.columns:
                pool = pool.rename(columns={_src: "own%"})
                break
        if "name" in pool.columns:
            _keep = [c for c in _pool_sim_cols if c in pool.columns]
            pool_lookup = (
                pool[_keep]
                .drop_duplicates(subset=["name"])
                .set_index("name")
                .to_dict("index")
            )

    # Normalise ownership column in lu_uniq as well
    for _src in ("ownership", "Own%", "proj_own"):
        if _src in lu_uniq.columns and "own%" not in lu_uniq.columns:
            lu_uniq = lu_uniq.rename(columns={_src: "own%"})
            break

    rng = np.random.RandomState(42)
    rows = []
    for _, row in lu_uniq.iterrows():
        name = row["name"]

        # For each field, prefer the lineup row value; fall back to pool_lookup
        pool_row = pool_lookup.get(name, {})

        def _get(field: str, default=None):
            val = row.get(field, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = pool_row.get(field, default)
            return val

        proj = float(pd.to_numeric(_get("proj", 0), errors="coerce") or 0)
        if proj <= 0:
            continue

        salary = float(pd.to_numeric(_get("salary", 0), errors="coerce") or 0)
        own_pct = float(pd.to_numeric(_get("own%", 0), errors="coerce") or 0)

        # Derive std from ceil/floor when available
        ceil_raw = pd.to_numeric(_get("ceil", np.nan), errors="coerce")
        floor_raw = pd.to_numeric(_get("floor", np.nan), errors="coerce")
        ceil_val = float(ceil_raw) if pd.notna(ceil_raw) else proj * 1.3
        floor_val = float(floor_raw) if pd.notna(floor_raw) else proj * 0.7
        std = float(np.clip((ceil_val - floor_val) / 4.0, proj * 0.05, proj * 0.6))

        # Simulate per-player outcomes
        outcomes = rng.normal(loc=proj, scale=std, size=n_sims)
        outcomes = np.clip(outcomes, 0, None)

        # Apply calibration knobs: ceiling_boost scales upside, floor_dampen scales downside
        above = outcomes > proj
        outcomes = np.where(
            above,
            proj + (outcomes - proj) * ceiling_boost,
            proj - (proj - outcomes) * floor_dampen,
        )
        outcomes = np.clip(outcomes, 0, None)

        # Smash/bust thresholds: use ratio multiplier when explicitly provided via
        # cal_knobs, otherwise use percentile-based thresholds (top 10% / bottom 30%).
        if smash_thr is not None:
            smash_threshold_val = float(smash_thr) * proj
        else:
            smash_threshold_val = float(np.percentile(outcomes, 90))

        if bust_thr is not None:
            bust_threshold_val = float(bust_thr) * proj
        else:
            bust_threshold_val = float(np.percentile(outcomes, 30))

        smash_pct = float((outcomes >= smash_threshold_val).mean() * 100.0)
        bust_pct = float((outcomes <= bust_threshold_val).mean() * 100.0)
        leverage = smash_pct / own_pct if own_pct > 0 else smash_pct

        rows.append({
            "Player": name,
            "Proj": round(proj, 1),
            "Salary": int(salary),
            "Own%": round(own_pct, 1),
            "Smash%": round(smash_pct, 1),
            "Bust%": round(bust_pct, 1),
            "Leverage Score": round(leverage, 2),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Value Trap: Bust% > 40% AND Salary > median salary in the pool
    if df["Salary"].sum() > 0:
        median_sal = df["Salary"].median()
        df["Value Trap"] = (df["Bust%"] > 40.0) & (df["Salary"] > median_sal)
    else:
        df["Value Trap"] = False

    # High Leverage flag
    df["Flag"] = df["Leverage Score"].apply(
        lambda x: "🔥 HIGH LEVERAGE" if x > 3.0 else ""
    )

    return df.sort_values("Leverage Score", ascending=False).reset_index(drop=True)


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
