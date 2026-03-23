"""yak_core.ricky_projections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ricky's recency-weighted game-log projection model.

Design
------
- Primary signal: recency-weighted rolling averages from Tank01 game logs.
  Weights: fp_5 (50%), fp_10 (30%), fp_20 (20%).
- Blended with salary-implied baseline (70% rolling / 30% salary when data
  available; pure salary when data is absent).
- Calibratable via per-position or per-player additive adjustments.
- Produces ``ricky_proj``, ``ricky_floor``, and ``ricky_ceil`` columns.

Usage
-----
    from yak_core.ricky_projections import compute_ricky_proj
    pool_df = compute_ricky_proj(pool_df, cfg)
    # pool_df now has ricky_proj / ricky_floor / ricky_ceil columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rolling-window weights (must sum to 1.0)
_ROLLING_WEIGHTS: Dict[str, float] = {
    "rolling_fp_5": 0.50,
    "rolling_fp_10": 0.30,
    "rolling_fp_20": 0.20,
}

# Blend: rolling signal vs salary baseline
_ROLLING_VS_SALARY = 0.70  # 70% rolling, 30% salary
_SALARY_ONLY = 1.00        # fallback when no rolling data

# Floor/ceil spread multipliers by salary tier ($/1k)
# Higher-salaried players are more predictable → tighter range
_SPREAD_KNOTS_SAL_K = [3.0, 4.0, 6.0, 8.0, 10.0, 15.0]
_SPREAD_KNOTS_MULT = [0.55, 0.52, 0.45, 0.38, 0.30, 0.25]

# Salary-implied baseline constant (FP per $1K)
_DEFAULT_FP_PER_K: float = 4.0

# Salary-curve knots for the fallback non-linear baseline
_SAL_CURVE_KNOTS_SAL = [0, 4000, 6000, 8000, 10000, 15000]
_SAL_CURVE_KNOTS_FPK = [2.5, 2.5, 3.5, 4.0, 4.5, 5.0]
_SAL_CURVE_CAP = 35.0


# ---------------------------------------------------------------------------
# Core projection function
# ---------------------------------------------------------------------------


def compute_ricky_proj(
    pool_df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
    adjustments: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute ``ricky_proj``, ``ricky_floor``, and ``ricky_ceil`` for every
    player in *pool_df*.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool.  Required column: ``salary``.
        Optional columns: ``rolling_fp_5``, ``rolling_fp_10``, ``rolling_fp_20``.
    cfg : dict, optional
        Config overrides.  Keys used:
          * ``FP_PER_K`` (float, default 4.0) – salary-implied baseline multiplier.
          * ``RICKY_ROLLING_VS_SALARY`` (float, default 0.70) – weight given to the
            rolling signal when data is present.
    adjustments : dict, optional
        Per-player additive calibration adjustments keyed by ``player_name``.
        E.g. ``{"LeBron James": +2.5, "Nikola Jokic": +1.0}``.

    Returns
    -------
    pd.DataFrame
        A copy of *pool_df* with three new (or updated) columns:
        ``ricky_proj``, ``ricky_floor``, ``ricky_ceil``.
    """
    cfg = cfg or {}
    adjustments = adjustments or {}

    df = pool_df.copy()
    n = len(df)

    fp_per_k = float(cfg.get("FP_PER_K", _DEFAULT_FP_PER_K))
    rolling_vs_salary = float(cfg.get("RICKY_ROLLING_VS_SALARY", _ROLLING_VS_SALARY))

    # ------------------------------------------------------------------
    # 1. Salary-implied baseline (always computed)
    # ------------------------------------------------------------------
    sal = pd.to_numeric(df.get("salary", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    sal_proj = sal * fp_per_k / 1000.0

    # ------------------------------------------------------------------
    # 2. Rolling game-log signal
    # ------------------------------------------------------------------
    rolling_cols_present = [c for c in _ROLLING_WEIGHTS if c in df.columns]

    if rolling_cols_present:
        weighted_sum = pd.Series(0.0, index=df.index)
        weight_total = pd.Series(0.0, index=df.index)

        for col, w in _ROLLING_WEIGHTS.items():
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            valid = vals.notna() & (vals > 0)
            weighted_sum += vals.fillna(0.0) * w * valid.astype(float)
            weight_total += w * valid.astype(float)

        has_rolling = weight_total > 0
        rolling_avg = (weighted_sum / weight_total.replace(0.0, 1.0)).clip(lower=0.0)

        # Blend: rolling signal for players with data, salary-implied for others
        ricky_proj = sal_proj.copy()
        ricky_proj[has_rolling] = (
            rolling_avg[has_rolling] * rolling_vs_salary
            + sal_proj[has_rolling] * (1.0 - rolling_vs_salary)
        )
    else:
        # No rolling columns at all — use non-linear salary curve
        sal_k = sal
        curve_fpk = np.interp(sal_k, _SAL_CURVE_KNOTS_SAL, _SAL_CURVE_KNOTS_FPK)
        ricky_proj = pd.Series(
            (sal_k * curve_fpk / 1000.0).clip(upper=_SAL_CURVE_CAP),
            index=df.index,
        )
        has_rolling = pd.Series(False, index=df.index)

    ricky_proj = ricky_proj.clip(lower=0.0)

    # ------------------------------------------------------------------
    # 3. Calibratable per-player adjustments
    # ------------------------------------------------------------------
    if adjustments and "player_name" in df.columns:
        adj_series = df["player_name"].map(adjustments).fillna(0.0)
        ricky_proj = (ricky_proj + adj_series).clip(lower=0.0)

    # ------------------------------------------------------------------
    # 4. Floor / ceil using salary-tier spread multipliers
    # ------------------------------------------------------------------
    sal_k_for_spread = (sal / 1000.0).clip(lower=3.0)
    spread_mult = pd.Series(
        np.interp(sal_k_for_spread, _SPREAD_KNOTS_SAL_K, _SPREAD_KNOTS_MULT),
        index=df.index,
    ).clip(lower=0.25, upper=0.55)

    # Refine spread with rolling variance when both fp_5 and fp_10 are available
    if "rolling_fp_5" in df.columns and "rolling_fp_10" in df.columns:
        fp5 = pd.to_numeric(df["rolling_fp_5"], errors="coerce")
        fp10 = pd.to_numeric(df["rolling_fp_10"], errors="coerce")
        both_valid = fp5.notna() & fp10.notna()
        if both_valid.any():
            rmean = ((fp5.fillna(0.0) + fp10.fillna(0.0)) / 2.0).replace(0.0, 1.0)
            rdiff = (fp5.fillna(0.0) - fp10.fillna(0.0)).abs()
            rcv = (rdiff / rmean).clip(lower=0.05, upper=0.60)
            spread_mult[both_valid] = (
                rcv[both_valid] * 0.60 + spread_mult[both_valid] * 0.40
            ).clip(lower=0.25, upper=0.55)

    ricky_floor = (ricky_proj * (1.0 - spread_mult)).round(2).clip(lower=0.0)
    ricky_ceil = (ricky_proj * (1.0 + spread_mult)).round(2)

    df["ricky_proj"] = ricky_proj.round(2)
    df["ricky_floor"] = ricky_floor
    df["ricky_ceil"] = ricky_ceil

    n_rolling = int(has_rolling.sum()) if rolling_cols_present else 0
    print(
        f"[ricky_projections] {n_rolling}/{n} players had rolling game-log data"
    )

    return df


# ---------------------------------------------------------------------------
# Archive-level helpers
# ---------------------------------------------------------------------------


def build_ricky_proj_from_archive(
    archive_df: pd.DataFrame,
    target_date: str,
    n_games_5: int = 5,
    n_games_10: int = 10,
    n_games_20: int = 20,
) -> pd.DataFrame:
    """Compute per-player rolling averages from a historical game-log archive.

    Parameters
    ----------
    archive_df : pd.DataFrame
        Historical game log with columns: ``player_name``, ``game_date``,
        ``fantasy_points`` (DK FP), and optionally ``minutes``.
    target_date : str
        ISO date string (``YYYY-MM-DD``).  Only games *before* this date are
        used to compute rolling averages.
    n_games_5, n_games_10, n_games_20 : int
        Number of most-recent games used for each rolling window.

    Returns
    -------
    pd.DataFrame
        One row per player with columns: ``player_name``, ``rolling_fp_5``,
        ``rolling_fp_10``, ``rolling_fp_20``, ``rolling_min_5``,
        ``rolling_min_10``, ``rolling_min_20``.
    """
    df = archive_df.copy()

    # Normalize column names
    if "fantasy_points" not in df.columns and "actual_fp" in df.columns:
        df = df.rename(columns={"actual_fp": "fantasy_points"})
    if "game_date" not in df.columns and "slate_date" in df.columns:
        df = df.rename(columns={"slate_date": "game_date"})

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    cutoff = pd.Timestamp(target_date)
    df = df[df["game_date"] < cutoff].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "player_name",
            "rolling_fp_5", "rolling_fp_10", "rolling_fp_20",
            "rolling_min_5", "rolling_min_10", "rolling_min_20",
        ])

    df = df.sort_values(["player_name", "game_date"])

    def _tail_avg(arr: np.ndarray, n: int) -> Optional[float]:
        tail = arr[-n:] if len(arr) >= 1 else np.array([])
        if len(tail) == 0:
            return None
        return float(np.mean(tail))

    records = []
    for player, grp in df.groupby("player_name", sort=False):
        fp_vals = grp["fantasy_points"].dropna().values
        min_vals = grp["minutes"].dropna().values if "minutes" in grp.columns else np.array([])

        rec = {
            "player_name": player,
            "rolling_fp_5": _tail_avg(fp_vals, n_games_5),
            "rolling_fp_10": _tail_avg(fp_vals, n_games_10),
            "rolling_fp_20": _tail_avg(fp_vals, n_games_20),
            "rolling_min_5": _tail_avg(min_vals, n_games_5),
            "rolling_min_10": _tail_avg(min_vals, n_games_10),
            "rolling_min_20": _tail_avg(min_vals, n_games_20),
        }
        records.append(rec)

    return pd.DataFrame(records)
