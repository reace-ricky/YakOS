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
    ricky_score = w_gpp * norm(gpp_score)
                + w_ceil * norm(ceiling)
                - w_own * norm(avg_own_pct)

where norm() is min-max normalization per-slate.  Ownership is SUBTRACTED
so high-ownership lineups are penalized (lower score = chalky).

Usage
-----
    from yak_core.ricky_rank import rank_lineups_for_se
    ranked = rank_lineups_for_se(summary_df, w_gpp=1.0, w_ceil=0.8, w_own=0.3)
    shortlist = ranked[ranked["ricky_tag"] != ""]
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

# ── Default ranking weights ─────────────────────────────────────────────────
# These are the slider defaults shown in the UI.  They do NOT need to sum
# to 1.0 — the formula uses raw weighted components so the user can crank
# any knob independently.
RICKY_W_GPP: float = 1.0    # GPP score weight (primary)
RICKY_W_CEIL: float = 0.8   # ceiling weight (almost as important)
RICKY_W_OWN: float = 0.3    # ownership penalty (soft — not the main driver)

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
    w_own: Optional[float] = None,
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
        - ``avg_own_pct`` — mean player ownership in lineup (higher = chalkier)
        - ``total_proj`` — sum of player proj values (for display)
        - ``total_actual`` — sum of player actual FP (may be 0 in live mode)
        - ``total_salary`` — sum of player salary

        Missing columns are tolerated; the corresponding weight component
        will be set to 0.

    w_gpp : float, optional
        Weight on GPP score.  Default 1.0.
    w_ceil : float, optional
        Weight on ceiling.  Default 0.8.
    w_own : float, optional
        Ownership penalty weight (subtracted).  Default 0.3.
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
    _w_own = w_own if w_own is not None else RICKY_W_OWN
    _tag_count = tag_count if tag_count is not None else RICKY_TAG_COUNT

    out = summary.copy()

    # ── Normalize each component ────────────────────────────────────────────
    gpp_norm = _norm01(
        pd.to_numeric(out.get("total_gpp_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    )
    ceil_norm = _norm01(
        pd.to_numeric(out.get("total_ceil", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    )
    own_norm = _norm01(
        pd.to_numeric(out.get("avg_own_pct", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    )

    # ── Composite score ─────────────────────────────────────────────────────
    # Ownership is SUBTRACTED — higher ownership = lower score (penalty).
    out["ricky_score"] = (
        _w_gpp * gpp_norm
        + _w_ceil * ceil_norm
        - _w_own * own_norm
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
