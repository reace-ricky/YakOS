"""Monte Carlo simulation for YakOS DFS optimizer."""
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# DK NBA scoring thresholds (classic 8-man roster)
SMASH_THRESHOLD = 300.0  # strong GPP score
BUST_THRESHOLD = 230.0   # likely cash-game miss

# Private aliases kept for backwards compatibility
_SMASH_THRESHOLD = SMASH_THRESHOLD
_BUST_THRESHOLD = BUST_THRESHOLD


def run_monte_carlo_for_lineups(
    lineups_df: pd.DataFrame,
    n_sims: int = 500,
    volatility_mode: str = "standard",
) -> pd.DataFrame:
    """Run Monte Carlo simulations on lineup projections.

    For each lineup, simulates ``n_sims`` outcomes by sampling from
    per-player normal distributions (mean=proj, std derived from
    ceil/floor when available).

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

    Returns
    -------
    pd.DataFrame
        Per-lineup summary with columns:
        ``lineup_index``, ``sim_mean``, ``sim_std``,
        ``smash_prob``, ``bust_prob``, ``median_points``,
        ``sim_p85``, ``sim_p15``.
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

        results.append({
            "lineup_index": lu_id,
            "sim_mean": round(float(totals.mean()), 2),
            "sim_std": round(float(totals.std()), 2),
            "smash_prob": round(float((totals >= _SMASH_THRESHOLD).mean()), 3),
            "bust_prob": round(float((totals <= _BUST_THRESHOLD).mean()), 3),
            "median_points": round(float(np.median(totals)), 2),
            "sim_p85": round(float(np.percentile(totals, 85)), 2),
            "sim_p15": round(float(np.percentile(totals, 15)), 2),
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
