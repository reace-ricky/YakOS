"""yak_core.pop_catalyst -- Pop Catalyst detection for edge analysis.

Computes per-player "pop catalyst" signals that identify players with
situational upside the base projection model doesn't capture.  These
signals feed directly into ``compute_edge_metrics`` as a 5th edge-score
component so Ricky's Edge Analysis surfaces pop candidates automatically.

Signals
-------
1. **Injury Minutes Opportunity** (``injury_opp``)
   Uses ``injury_bump_fp`` from injury_cascade — how much projection
   boost this player got from teammates being OUT.  Normalised 0-1
   within the slate.

2. **Salary Stickiness** (``salary_lag``)
   Compares current DK salary to what recent production would price at.
   If a player's last-5 rolling FP implies a higher salary tier but DK
   hasn't adjusted, there's a pricing edge.  Normalised 0-1.

3. **Minutes Trend** (``minutes_trend``)
   Compares rolling 3-game minutes (``rolling_min_5`` proxy) to rolling
   10-game minutes.  A positive trend means the player is moving into a
   bigger role.  Normalised 0-1.

4. **Recent Ceiling Game** (``ceiling_flash``)
   Checks if the player's max recent FP (from game logs) significantly
   exceeds their current projection.  A player who has *shown* they can
   pop is more likely to pop again.  Uses ``rolling_fp_5`` max vs proj.

5. **Pace / Game Environment** (``pace_environment``)
   Flags players in high-scoring game environments (high vegas total),
   especially when their salary doesn't reflect the pace-up.  Also
   adjusts for spread: big underdogs get a slight boost (forced to keep
   shooting), big favorites get dampened (blowout risk).  Normalised
   0-1 within the slate using ``vegas_total`` and ``vegas_spread``.

Output
------
``compute_pop_catalyst(pool_df)`` adds columns to the pool:

- ``pop_catalyst_score``    : float 0-1, composite of all 5 signals
- ``pop_catalyst_tag``      : str, human-readable tag (e.g. "Injury Opp + Salary Lag")
- ``pop_injury_opp``        : float 0-1, individual signal
- ``pop_salary_lag``        : float 0-1, individual signal
- ``pop_minutes_trend``     : float 0-1, individual signal
- ``pop_ceiling_flash``     : float 0-1, individual signal
- ``pop_pace_environment``  : float 0-1, individual signal

The composite ``pop_catalyst_score`` is used by ``compute_edge_metrics``
as a weighted component of ``edge_score``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS: Dict[str, float] = {
    "injury_opp":       0.30,   # biggest driver historically (67% of spikes)
    "salary_lag":       0.25,   # pricing inefficiency
    "minutes_trend":    0.15,   # role expansion
    "ceiling_flash":    0.15,   # proven upside
    "pace_environment": 0.15,   # game environment / vegas total
}

# ---------------------------------------------------------------------------
# Thresholds and constants
# ---------------------------------------------------------------------------

# Injury opportunity: minimum bump (FP) to be considered meaningful
_MIN_INJURY_BUMP_FP = 1.5

# Salary stickiness: approximate DK FP-to-salary ratio (NBA)
# ~$1,000 salary per ~4.0 FP of expected production
_FP_TO_SALARY_RATIO = 250.0  # $250 per 1 DK FP

# Salary lag: minimum implied salary gap ($) to flag
_MIN_SALARY_GAP = 500

# Minutes trend: minimum delta (minutes) for trend to matter
_MIN_MINUTES_DELTA = 2.0

# Ceiling flash: ratio of recent-max-FP to projection to flag
_CEILING_FLASH_RATIO = 1.5  # 50% above projection = ceiling shown

# Minimum projection to even consider (ignore near-zero players)
_MIN_PROJ_FOR_POP = 5.0

# Minimum pop_catalyst_score to generate a tag (avoid noise)
_MIN_TAG_SCORE = 0.15


# ---------------------------------------------------------------------------
# Individual signal computations
# ---------------------------------------------------------------------------

def _compute_injury_opp(df: pd.DataFrame) -> pd.Series:
    """Signal 1: Injury-driven minutes opportunity.

    Players with ``injury_bump_fp > 0`` have gained projection from
    teammates being OUT.  Higher bump = more opportunity unlocked.
    Normalised 0-1 within the slate.
    """
    _raw = df["injury_bump_fp"] if "injury_bump_fp" in df.columns else pd.Series(0.0, index=df.index)
    bump = pd.to_numeric(_raw, errors="coerce").fillna(0.0)

    # Zero out tiny bumps (noise)
    bump = bump.where(bump >= _MIN_INJURY_BUMP_FP, 0.0)

    # Normalise 0-1 within slate
    _max = float(bump.max())
    if _max > 0:
        return (bump / _max).clip(0.0, 1.0)
    return pd.Series(0.0, index=df.index)


def _safe_col(df: pd.DataFrame, *col_names: str) -> pd.Series:
    """Return first matching column as numeric Series; 0.0 if none found."""
    for c in col_names:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _compute_salary_lag(df: pd.DataFrame) -> pd.Series:
    """Signal 2: Salary stickiness — salary lagging recent production.

    Compares current DK salary to what recent rolling FP would imply.
    If rolling_fp_5 implies a higher salary, the player is underpriced
    relative to their current role.
    """
    salary = _safe_col(df, "salary")
    rolling_fp = _safe_col(df, "rolling_fp_5", "rolling_fp_10")

    # What salary would their recent production command?
    implied_salary = rolling_fp * _FP_TO_SALARY_RATIO

    # Gap: how much they're underpriced
    gap = (implied_salary - salary).clip(lower=0.0)

    # Only flag meaningful gaps
    gap = gap.where(gap >= _MIN_SALARY_GAP, 0.0)

    # Normalise 0-1 — $3,000+ gap is a full signal
    return (gap / 3000.0).clip(0.0, 1.0)


def _compute_minutes_trend(df: pd.DataFrame) -> pd.Series:
    """Signal 3: Minutes trend — recent minutes exceeding baseline.

    Compares rolling 5-game minutes to rolling 10-game (or 20-game) minutes.
    A positive delta indicates role expansion.
    """
    min_recent = _safe_col(df, "rolling_min_5")
    min_baseline = _safe_col(df, "rolling_min_10", "rolling_min_20")

    delta = (min_recent - min_baseline).clip(lower=0.0)

    # Only flag meaningful trends
    delta = delta.where(delta >= _MIN_MINUTES_DELTA, 0.0)

    # Normalise 0-1 — 10+ min jump is a full signal
    return (delta / 10.0).clip(0.0, 1.0)


def _compute_ceiling_flash(df: pd.DataFrame) -> pd.Series:
    """Signal 4: Recent ceiling game — has the player shown they can pop?

    Compares recent rolling max FP to current projection.  A player who
    recently put up 1.5x+ their projection has demonstrated the ceiling.

    Uses rolling_fp_5 as a proxy since we don't have per-game max in the
    pool.  If rolling_fp_5 > proj * ratio, the player has consistently
    outperformed recently (or had a big game pulling the average up).
    """
    proj = _safe_col(df, "proj")
    rolling_fp = _safe_col(df, "rolling_fp_5")

    # Ratio of recent production to projection
    safe_proj = proj.clip(lower=1.0)
    ratio = rolling_fp / safe_proj

    # Score: how far above the flash threshold
    # ratio of 1.5 = threshold (score 0), 2.5 = full signal (score 1)
    score = ((ratio - _CEILING_FLASH_RATIO) / 1.0).clip(0.0, 1.0)

    # Zero out players with negligible projections
    score = score.where(proj >= _MIN_PROJ_FOR_POP, 0.0)

    return score


def _compute_pace_environment(df: pd.DataFrame) -> pd.Series:
    """Signal 5: Game environment — high vegas total = more stats available.

    Players in games with above-average totals get a boost, especially
    when their salary doesn't reflect the pace-up.  A $5K player in a
    240-total game has more breakout upside than one in a 210-total game.

    Also factors in spread: players on big underdogs get a slight boost
    (garbage time / forced to keep shooting), while big favorites get
    dampened (blowout risk = bench early).

    Uses ``vegas_total`` and ``vegas_spread`` from the pool.
    Players with missing or zero vegas_total receive 0.0.
    """
    total = _safe_col(df, "vegas_total").replace(0.0, np.nan)
    spread = _safe_col(df, "vegas_spread")

    # Players with no game total data get 0
    missing_mask = total.isna()

    # Slate average total (only from valid rows)
    slate_avg = float(total.mean()) if total.notna().any() else 220.0

    # Raw pace signal: how far above slate average this game is
    pace_raw = (total - slate_avg) / slate_avg

    # Underdog boost: spread > 5 means this team is a big underdog
    pace_raw = pace_raw.where(spread <= 5, pace_raw + 0.1)

    # Favorite dampen: spread < -8 means this team is a big favourite
    pace_raw = pace_raw.where(spread >= -8, pace_raw - 0.1)

    # Zero out players without valid game totals
    pace_raw = pace_raw.where(~missing_mask, 0.0).fillna(0.0)

    # Clip negatives to 0 before normalising (negative = below average)
    pace_raw = pace_raw.clip(lower=0.0)

    # Normalise 0-1 within the slate
    _max = float(pace_raw.max())
    if _max > 0:
        return (pace_raw / _max).clip(0.0, 1.0)
    return pd.Series(0.0, index=df.index)


# ---------------------------------------------------------------------------
# Tag generation
# ---------------------------------------------------------------------------

_SIGNAL_LABELS: Dict[str, str] = {
    "injury_opp":       "Injury Opp",
    "salary_lag":       "Salary Lag",
    "minutes_trend":    "Min Trend",
    "ceiling_flash":    "Ceiling Flash",
    "pace_environment": "Pace/Env",
}


def _build_tag(row: pd.Series) -> str:
    """Build a human-readable pop catalyst tag from individual signals."""
    parts: List[str] = []
    for signal_key, label in _SIGNAL_LABELS.items():
        col = f"pop_{signal_key}"
        if col in row.index and float(row[col]) > 0.10:
            parts.append(label)

    if not parts:
        return ""
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_pop_catalyst(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pop catalyst signals and add them to the pool.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Player pool with columns from DK + Tank01 enrichment + injury cascade.
        Expected columns (all optional, gracefully degrade):
        ``injury_bump_fp``, ``salary``, ``rolling_fp_5``, ``rolling_fp_10``,
        ``rolling_min_5``, ``rolling_min_10``, ``proj``,
        ``vegas_total``, ``vegas_spread``.

    Returns
    -------
    pd.DataFrame
        Copy of pool with new columns: ``pop_catalyst_score``,
        ``pop_catalyst_tag``, ``pop_injury_opp``, ``pop_salary_lag``,
        ``pop_minutes_trend``, ``pop_ceiling_flash``, ``pop_pace_environment``.
    """
    if pool_df is None or pool_df.empty:
        return pool_df

    df = pool_df.copy()

    # Compute individual signals
    df["pop_injury_opp"] = _compute_injury_opp(df)
    df["pop_salary_lag"] = _compute_salary_lag(df)
    df["pop_minutes_trend"] = _compute_minutes_trend(df)
    df["pop_ceiling_flash"] = _compute_ceiling_flash(df)
    df["pop_pace_environment"] = _compute_pace_environment(df)

    # Composite score (weighted average)
    df["pop_catalyst_score"] = (
        df["pop_injury_opp"]       * SIGNAL_WEIGHTS["injury_opp"]
        + df["pop_salary_lag"]     * SIGNAL_WEIGHTS["salary_lag"]
        + df["pop_minutes_trend"]  * SIGNAL_WEIGHTS["minutes_trend"]
        + df["pop_ceiling_flash"]  * SIGNAL_WEIGHTS["ceiling_flash"]
        + df["pop_pace_environment"] * SIGNAL_WEIGHTS["pace_environment"]
    ).round(3)

    # Zero out players with negligible projections
    proj = _safe_col(df, "proj")
    df.loc[proj < _MIN_PROJ_FOR_POP, "pop_catalyst_score"] = 0.0

    # Generate tags
    df["pop_catalyst_tag"] = df.apply(_build_tag, axis=1)

    # Clear tags for players below threshold
    df.loc[df["pop_catalyst_score"] < _MIN_TAG_SCORE, "pop_catalyst_tag"] = ""

    return df
