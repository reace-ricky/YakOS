"""yak_core.ricky_rank — Ricky SE lineup ranking helper.

Takes a lineup-level summary DataFrame (one row per lineup, with columns
aggregated from the player-level lineups DF) and tags the top 1–3 lineups
as SE Core / SE Spicy / SE Alt.

Purpose
-------
Test the ranking on historical Sim Lab batches before promoting it into
the main Optimizer or Ricky Edge Analysis.  This module has NO side-effects
on the main optimizer config or any state objects.

Ranking formula
---------------
    ricky_score = W_GPP * norm(gpp_score)
                + W_CEIL * norm(ceiling)
                + W_LEV * norm(leverage)

where norm() is min-max normalization per-slate and the weights are defined
as module-level constants for easy tuning.

Usage
-----
    from yak_core.ricky_rank import rank_lineups_for_se
    ranked = rank_lineups_for_se(summary_df)
    shortlist = ranked[ranked["ricky_tag"] != ""]
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

# ── Ranking weights (easy to tune) ──────────────────────────────────────────
# Must sum to 1.0.  w_gpp >= w_ceil >= w_lev by default.
RICKY_W_GPP: float = 0.50   # weight on total gpp_score across lineup
RICKY_W_CEIL: float = 0.30  # weight on total ceiling (sum of per-player ceil)
RICKY_W_LEV: float = 0.20   # weight on leverage metric (low-own edge)

# Number of lineups to tag.  SE Core = rank 1, SE Spicy = rank 2,
# SE Alt = rank 3.  Set to 2 to skip Alt tagging.
RICKY_TAG_COUNT: int = 3

_TAG_MAP: Dict[int, str] = {
    1: "SE Core",
    2: "SE Spicy",
    3: "SE Alt",
}


def _norm01(s: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1].  Constant series → 0.5."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def rank_lineups_for_se(
    summary: pd.DataFrame,
    *,
    w_gpp: Optional[float] = None,
    w_ceil: Optional[float] = None,
    w_lev: Optional[float] = None,
    tag_count: Optional[int] = None,
) -> pd.DataFrame:
    """Rank lineups and tag the top few for SE play.

    Parameters
    ----------
    summary : DataFrame
        One row per lineup.  Expected columns (all numeric, lineup-level totals):

        - ``lineup_index`` — lineup identifier
        - ``total_gpp_score`` — sum of player gpp_score values in lineup
        - ``total_ceil`` — sum of player ceil values in lineup
        - ``avg_own_pct`` — mean player ownership in lineup (lower = more leverage)
        - ``total_proj`` — sum of player proj values (for display)
        - ``total_actual`` — sum of player actual FP (may be 0 in live mode)
        - ``total_salary`` — sum of player salary

        Missing columns are tolerated; the corresponding weight component
        will be set to 0.

    w_gpp, w_ceil, w_lev : float, optional
        Override module-level weights for this call.

    tag_count : int, optional
        Override RICKY_TAG_COUNT (number of lineups to tag).

    Returns
    -------
    DataFrame
        Copy of ``summary`` with three new columns:

        - ``ricky_score`` — the combined ranking key (higher = better)
        - ``ricky_rank`` — 1-based rank (1 = best)
        - ``ricky_tag`` — "SE Core", "SE Spicy", "SE Alt", or ""
    """
    if summary.empty:
        summary = summary.copy()
        summary["ricky_score"] = pd.Series(dtype=float)
        summary["ricky_rank"] = pd.Series(dtype=int)
        summary["ricky_tag"] = pd.Series(dtype=str)
        return summary

    _w_gpp = w_gpp if w_gpp is not None else RICKY_W_GPP
    _w_ceil = w_ceil if w_ceil is not None else RICKY_W_CEIL
    _w_lev = w_lev if w_lev is not None else RICKY_W_LEV
    _tag_count = tag_count if tag_count is not None else RICKY_TAG_COUNT

    out = summary.copy()

    # ── Normalize each component ────────────────────────────────────────────
    gpp_norm = _norm01(
        pd.to_numeric(out.get("total_gpp_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    )
    ceil_norm = _norm01(
        pd.to_numeric(out.get("total_ceil", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    )

    # Leverage: lower ownership = MORE leverage, so we invert avg_own_pct.
    avg_own = pd.to_numeric(
        out.get("avg_own_pct", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    # Invert: high leverage = low ownership
    lev_norm = _norm01(-avg_own)

    # ── Composite score ─────────────────────────────────────────────────────
    out["ricky_score"] = (
        _w_gpp * gpp_norm
        + _w_ceil * ceil_norm
        + _w_lev * lev_norm
    ).round(4)

    # ── Rank (1 = best, dense ranking) ──────────────────────────────────────
    out["ricky_rank"] = (
        out["ricky_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    # ── Tag top N ───────────────────────────────────────────────────────────
    out["ricky_tag"] = ""
    for rank_val in range(1, _tag_count + 1):
        tag = _TAG_MAP.get(rank_val, "")
        if tag:
            mask = out["ricky_rank"] == rank_val
            out.loc[mask, "ricky_tag"] = tag

    return out
