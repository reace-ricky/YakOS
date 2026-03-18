"""yak_core.ricky_signals -- Compute Ricky's edge signals for the Edge Analysis page.

Each player gets evaluated across multiple signal dimensions:
  1. Injury cascade benefit  (injury_bump_fp > 0 from injury_cascade.py)
  2. Ownership vs projection mismatch  (underpriced by the field)
  3. Salary value efficiency  (FP/$1K relative to slate average)
  4. Leverage score  (smash_prob / ownership -- upside per % owned)
  5. Salary stickiness  (cheap player with high proj -- DK slow to adjust)

Signals are combined into a composite edge_rank that drives the
"Top Edges" list on Ricky's Edge Analysis page.

Usage
-----
    from yak_core.ricky_signals import compute_ricky_signals, generate_slate_overview

    signals_df = compute_ricky_signals(pool)
    overview = generate_slate_overview(pool, signals_df)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal weight defaults (overridden by edge_feedback weights when available)
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "injury_cascade": 0.25,
    "own_proj_mismatch": 0.25,
    "salary_value": 0.20,
    "leverage": 0.20,
    "salary_stickiness": 0.10,
}


# ---------------------------------------------------------------------------
# Signal badge labels for the UI
# ---------------------------------------------------------------------------
SIGNAL_BADGES: Dict[str, str] = {
    "injury_cascade": "Injury Cascade",
    "own_proj_mismatch": "Ownership Mismatch",
    "salary_value": "Salary Value",
    "leverage": "Leverage",
    "salary_stickiness": "Salary Sticky",
}


def _safe_numeric(series: "pd.Series", default: float = 0.0) -> "pd.Series":
    return pd.to_numeric(series, errors="coerce").fillna(default)


# Mapping from edge_feedback signal names → ricky_signals weight keys.
# edge_feedback tracks hit rates on these 5 signals;
# we map each to the ricky_signals dimension it most informs.
_EDGE_FB_TO_RICKY: Dict[str, str] = {
    "high_leverage": "leverage",
    "salary_value": "salary_value",
    # Pruned: low_ownership_upside (0% hit rate), chalk_fade (0 calls),
    #         smash_candidate (0% hit rate) — removed from edge_feedback.py
}


def _load_feedback_weights() -> Dict[str, float]:
    """Load learned signal weights from edge feedback, fall back to defaults.

    The edge_feedback system stores per-signal hit rates under its own key
    names (high_leverage, salary_value, etc.).  We map those into the 5
    ricky_signals dimensions, blend with defaults using a 60/40 split
    (feedback/default), and renormalize so weights sum to 1.0.
    """
    try:
        import json
        import os
        from yak_core.config import YAKOS_ROOT

        path = os.path.join(YAKOS_ROOT, "data", "edge_feedback", "signal_weights.json")
        if not os.path.exists(path):
            return dict(_DEFAULT_WEIGHTS)

        with open(path) as f:
            data = json.load(f)

        # Direct key match (if weights file already uses ricky keys)
        weights_raw = data.get("weights", data)
        if isinstance(weights_raw, dict) and any(k in weights_raw for k in _DEFAULT_WEIGHTS):
            merged = dict(_DEFAULT_WEIGHTS)
            merged.update({k: float(v) for k, v in weights_raw.items() if k in _DEFAULT_WEIGHTS})
            total = sum(merged.values())
            if total > 0:
                return {k: round(v / total, 4) for k, v in merged.items()}
            return dict(_DEFAULT_WEIGHTS)

        # Map from edge_feedback signal names → ricky dimensions
        signal_stats = data.get("signal_stats", {})
        if not signal_stats:
            return dict(_DEFAULT_WEIGHTS)

        # Accumulate weighted hit rates per ricky dimension
        fb_accum: Dict[str, float] = {k: 0.0 for k in _DEFAULT_WEIGHTS}
        fb_count: Dict[str, int] = {k: 0 for k in _DEFAULT_WEIGHTS}
        for fb_sig, stats in signal_stats.items():
            ricky_key = _EDGE_FB_TO_RICKY.get(fb_sig)
            if ricky_key and stats.get("weighted_hit_rate", 0) > 0:
                fb_accum[ricky_key] += stats["weighted_hit_rate"]
                fb_count[ricky_key] += 1

        # Average per dimension, then blend 60% feedback / 40% default
        merged = {}
        _FB_BLEND = 0.6
        for key, default_w in _DEFAULT_WEIGHTS.items():
            if fb_count[key] > 0:
                fb_w = fb_accum[key] / fb_count[key]
                merged[key] = _FB_BLEND * fb_w + (1 - _FB_BLEND) * default_w
            else:
                merged[key] = default_w

        # Renormalize to sum to 1.0
        total = sum(merged.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in merged.items()}
    except Exception:
        pass
    return dict(_DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Per-signal scoring (each returns 0..1 normalised score)
# ---------------------------------------------------------------------------

def _score_injury_cascade(df: pd.DataFrame) -> pd.Series:
    """Players benefiting from injury cascades (minutes redistribution)."""
    bump = _safe_numeric(df.get("injury_bump_fp", pd.Series(0.0, index=df.index)))
    if bump.max() <= 0:
        return pd.Series(0.0, index=df.index)
    # Normalise to 0-1: a +5 FP bump is near-max signal
    return (bump / 5.0).clip(0, 1)


def _score_own_proj_mismatch(df: pd.DataFrame) -> pd.Series:
    """Ownership lower than projection rank suggests -- field is undervaluing."""
    proj = _safe_numeric(df.get("proj", pd.Series(0.0, index=df.index)))
    own = _safe_numeric(df.get("ownership", df.get("own_pct", pd.Series(15.0, index=df.index))))

    if proj.max() <= 0:
        return pd.Series(0.0, index=df.index)

    # Compute projection rank percentile (1.0 = best projections)
    proj_pct = proj.rank(pct=True)
    # Compute ownership rank percentile (1.0 = highest owned)
    own_pct = own.rank(pct=True)

    # Mismatch = how much higher their projection rank is vs ownership rank
    # Positive = underowned relative to projections
    mismatch = (proj_pct - own_pct).clip(0, 1)
    return mismatch


def _score_salary_value(df: pd.DataFrame) -> pd.Series:
    """FP per $1K relative to slate median -- salary hasn't caught up.

    Qualifiers (Fix 3): Only flag when rolling_min_5 >= 24 AND floor >= 0.65 * proj.
    This filters out players who are cheap because they're in bad situations
    (low minutes, low floor) rather than genuine salary value.
    """
    proj = _safe_numeric(df.get("proj", pd.Series(0.0, index=df.index)))
    sal = _safe_numeric(df.get("salary", pd.Series(5000.0, index=df.index)))
    sal_k = (sal / 1000.0).clip(lower=1.0)
    val = proj / sal_k

    median_val = val.median()
    if median_val <= 0:
        return pd.Series(0.0, index=df.index)

    # How far above median value this player sits, normalised
    # 2x median value → score of 1.0
    score = ((val - median_val) / median_val).clip(0, 1)

    # Apply floor/minutes qualifiers to suppress false positives.
    # Only zero out players who FAIL the qualifiers — if the columns
    # don't exist, skip the filter gracefully (don't penalize).
    has_rolling_min = "rolling_min_5" in df.columns
    has_floor = "floor" in df.columns

    if has_rolling_min or has_floor:
        disqualify = pd.Series(False, index=df.index)
        if has_rolling_min:
            rolling_min = _safe_numeric(df["rolling_min_5"])
            # Only disqualify players with actual rolling data that's below threshold
            has_data = rolling_min > 0
            disqualify = disqualify | (has_data & (rolling_min < 24))
        if has_floor:
            floor_vals = _safe_numeric(df["floor"])
            has_data = floor_vals > 0
            disqualify = disqualify | (has_data & (floor_vals < 0.65 * proj))
        score = score.where(~disqualify, 0.0)

    return score


def _score_leverage(df: pd.DataFrame) -> pd.Series:
    """Upside per percent owned -- GPP differentiators.

    When rolling minutes data is available (rolling_min_5), we boost
    players whose recent minutes trend upward (gaining role) but
    haven't been reflected in ownership yet.
    """
    smash = _safe_numeric(df.get("smash_prob", pd.Series(0.0, index=df.index)))
    own = _safe_numeric(df.get("ownership", df.get("own_pct", pd.Series(15.0, index=df.index))))
    own_safe = own.clip(lower=1.0)

    lev = smash / (own_safe / 100.0)
    if lev.max() <= 0:
        return pd.Series(0.0, index=df.index)

    # Minutes trend boost: if recent 5-game mins > 10-game mins, player is
    # gaining role — extra leverage the field hasn't priced in.
    rolling_min_5 = _safe_numeric(df.get("rolling_min_5", pd.Series(0.0, index=df.index)))
    rolling_min_10 = _safe_numeric(df.get("rolling_min_10", pd.Series(0.0, index=df.index)))
    if rolling_min_5.max() > 0 and rolling_min_10.max() > 0:
        mins_trend = (rolling_min_5 - rolling_min_10).clip(lower=0)
        # Normalize: +5 min trend = 0.5 boost, capped at 1.0
        trend_boost = (mins_trend / 10.0).clip(0, 0.5)
        lev = lev + (trend_boost * lev.max())

    # Normalise: leverage of 5+ is near-max
    return (lev / 5.0).clip(0, 1)


def _score_salary_stickiness(df: pd.DataFrame) -> pd.Series:
    """Cheap players with high projections -- DK salary is slow to adjust.

    Classic edge: a $4K player projecting like a $6K player. DK sets
    salaries days before lock and doesn't react to late news.

    When rolling game log stats are available (rolling_fp_5, rolling_min_5),
    we use actual recent performance instead of projection rank alone.
    This catches players whose recent production outstrips their stale salary.
    """
    proj = _safe_numeric(df.get("proj", pd.Series(0.0, index=df.index)))
    sal = _safe_numeric(df.get("salary", pd.Series(5000.0, index=df.index)))

    if proj.max() <= 0 or sal.max() <= 0:
        return pd.Series(0.0, index=df.index)

    # If we have real game log rolling stats, use them for a stronger signal
    rolling_fp = _safe_numeric(df.get("rolling_fp_5", pd.Series(0.0, index=df.index)))
    has_rolling = rolling_fp.max() > 0

    if has_rolling:
        # Use rolling 5-game FP average as "what salary should be"
        # Compare actual salary vs salary implied by recent performance
        sal_k = (sal / 1000.0).clip(lower=1.0)
        actual_val = rolling_fp / sal_k  # recent FP per $1K
        median_val = actual_val.median()
        if median_val > 0:
            # Players producing well above their salary tier
            score = ((actual_val - median_val) / median_val).clip(0, 1)
            return score

    # Fallback: projection-rank-based estimate
    proj_rank = proj.rank(pct=True)
    sal_range = sal.max() - sal.min()
    expected_sal = sal.min() + proj_rank * sal_range

    # Stickiness = how much cheaper actual salary is than expected
    gap = (expected_sal - sal) / max(sal_range, 1.0)
    return gap.clip(0, 1)


# ---------------------------------------------------------------------------
# Main composite computation
# ---------------------------------------------------------------------------

def compute_ricky_signals(
    pool: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Score every player across all edge signals and return ranked results.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool from The Lab (must have player_name, salary, proj at minimum).
    weights : dict, optional
        Override signal weights.  Loaded from edge feedback if None.

    Returns
    -------
    pd.DataFrame
        Original pool columns plus:
        - sig_injury, sig_own_mismatch, sig_value, sig_leverage, sig_sticky (0-1 each)
        - edge_composite (weighted sum, 0-1)
        - active_signals (list of signal names firing > 0.3)
        - signal_badges (comma-separated display labels)
        - edge_rank (1 = best edge)
    """
    if pool.empty or "player_name" not in pool.columns:
        return pool.copy()

    df = pool.copy()
    w = weights or _load_feedback_weights()

    # Compute individual signals
    df["sig_injury"] = _score_injury_cascade(df)
    df["sig_own_mismatch"] = _score_own_proj_mismatch(df)
    df["sig_value"] = _score_salary_value(df)
    df["sig_leverage"] = _score_leverage(df)
    df["sig_sticky"] = _score_salary_stickiness(df)

    # Weighted composite
    df["edge_composite"] = (
        df["sig_injury"] * w.get("injury_cascade", 0.25)
        + df["sig_own_mismatch"] * w.get("own_proj_mismatch", 0.25)
        + df["sig_value"] * w.get("salary_value", 0.20)
        + df["sig_leverage"] * w.get("leverage", 0.20)
        + df["sig_sticky"] * w.get("salary_stickiness", 0.10)
    )

    # Active signals (threshold: 0.3 out of 1.0, except leverage which is 0.50)
    _sig_map = {
        "sig_injury": "injury_cascade",
        "sig_own_mismatch": "own_proj_mismatch",
        "sig_value": "salary_value",
        "sig_leverage": "leverage",
        "sig_sticky": "salary_stickiness",
    }
    _threshold = 0.3
    _leverage_threshold = 0.50

    # Leverage percentile cutoff: only flag top 25% of leverage scores on the slate
    _leverage_p75 = float(df["sig_leverage"].quantile(0.75)) if len(df) > 0 else 0.0

    def _active(row):
        active = []
        for col, key in _sig_map.items():
            val = row.get(col, 0)
            if key == "leverage":
                # Tighter threshold (0.50) + must be in top 25% of slate
                if val >= _leverage_threshold and val >= _leverage_p75:
                    active.append(key)
            else:
                if val >= _threshold:
                    active.append(key)
        return active

    df["active_signals"] = df.apply(_active, axis=1)
    df["signal_badges"] = df["active_signals"].apply(
        lambda sigs: ", ".join(SIGNAL_BADGES.get(s, s) for s in sigs) if sigs else ""
    )
    df["n_signals"] = df["active_signals"].apply(len)

    # Rank: higher composite = better edge = lower rank number
    df["edge_rank"] = df["edge_composite"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("edge_rank")

    return df


# ---------------------------------------------------------------------------
# Slate overview generator
# ---------------------------------------------------------------------------

def generate_slate_overview(
    pool: pd.DataFrame,
    signals_df: pd.DataFrame,
    contest_type: str = "GPP",
) -> Dict[str, Any]:
    """Generate a 4-5 bullet slate overview + recommendation.

    Returns
    -------
    dict with keys:
        bullets : list[str]   -- 4-5 analysis bullets
        recommendation : str  -- one-line recommendation
        top_plays : list[str] -- top 3 player names
        top_fades : list[str] -- top 3 fades
        injury_impact : str   -- summary of injury cascades
    """
    bullets: List[str] = []
    recommendation = ""
    top_plays: List[str] = []
    top_fades: List[str] = []
    injury_impact = ""

    if pool.empty or signals_df.empty:
        return {
            "bullets": ["No pool data available."],
            "recommendation": "Load a slate in The Lab first.",
            "top_plays": [],
            "top_fades": [],
            "injury_impact": "",
        }

    # Reset indexes to avoid misaligned boolean indexing after sort
    pool = pool.reset_index(drop=True)
    signals_df = signals_df.reset_index(drop=True)

    # IMPORTANT: Use signals_df for all column lookups since it's sorted by
    # edge_rank and its rows won't match pool order after reset_index.
    sal = _safe_numeric(signals_df.get("salary", pd.Series()))
    proj = _safe_numeric(signals_df.get("proj", pd.Series()))
    own = _safe_numeric(signals_df.get(
        "ownership", signals_df.get("own_pct", pd.Series())
    ))
    n_players = len(signals_df)

    # Track player names used in bullets so we never repeat a name.
    _used_names: set = set()

    # ------------------------------------------------------------------
    # Bullet 1: ANCHOR STUDS — $7K+ players with best edge.
    # "Who are you paying up for tonight?"
    # ------------------------------------------------------------------
    _edge_col = "edge_score" if "edge_score" in signals_df.columns else "edge_composite"
    _has_edge = _edge_col in signals_df.columns and signals_df[_edge_col].max() > 0

    _stud_mask = sal >= 7000
    _studs = signals_df[_stud_mask]
    if _has_edge and not _studs.empty:
        _studs = _studs.nlargest(3, _edge_col)
    elif not _studs.empty:
        _studs = _studs.nlargest(3, "proj")
    else:
        _studs = pd.DataFrame()

    if not _studs.empty:
        parts = []
        for _, r in _studs.iterrows():
            name = r.get("player_name", "?")
            s = r.get("salary", 0)
            p = r.get("proj", 0)
            parts.append(f"{name} (${s:,.0f}/{p:.1f})")
            _used_names.add(name)
        bullets.append(f"Anchor studs: {', '.join(parts)}")

    # ------------------------------------------------------------------
    # Bullet 2: VALUE / LEVERAGE — cheap players with the best edge.
    # "Where are you saving salary and still getting upside?"
    # Excludes anyone already in Bullet 1.
    # ------------------------------------------------------------------
    _val_mask = sal < 7000
    _vals = signals_df[_val_mask]
    if _has_edge and not _vals.empty:
        _vals = _vals.nlargest(10, _edge_col)  # grab extra so we can filter
    elif not _vals.empty:
        _vals = _vals.nlargest(10, "proj")

    # Prefer players with pop catalyst or leverage signals
    _pop_col = "pop_catalyst_score"
    if _pop_col in _vals.columns:
        _vals = _vals.sort_values(
            [_pop_col, _edge_col if _has_edge else "proj"], ascending=False
        )
    _vals = _vals[~_vals["player_name"].isin(_used_names)].head(3)

    if not _vals.empty:
        parts = []
        for _, r in _vals.iterrows():
            name = r.get("player_name", "?")
            s = r.get("salary", 0)
            p = r.get("proj", 0)
            # Add the pop catalyst tag if it exists for context
            tag = r.get("pop_catalyst_tag", "")
            if tag:
                parts.append(f"{name} (${s:,.0f} — {tag})")
            else:
                parts.append(f"{name} (${s:,.0f}/{p:.1f})")
            _used_names.add(name)
        bullets.append(f"Value plays: {', '.join(parts)}")

    # ------------------------------------------------------------------
    # Bullet 3: CHALK / FADES — ownership traps or high-owned fades.
    # "Who should you avoid or at least be aware of?"
    # Excludes anyone already named.
    # ------------------------------------------------------------------
    _own_col = "ownership" if "ownership" in signals_df.columns else "own_pct"
    if not own.empty and own.max() > 0:
        # Fades: high ownership + low edge signal
        _fade_mask = (own >= 15) & (sal >= 6000)
        if "sig_own_mismatch" in signals_df.columns:
            _fade_mask = _fade_mask & (signals_df["sig_own_mismatch"] <= 0.15)
        _fades = signals_df[_fade_mask & ~signals_df["player_name"].isin(_used_names)]
        _fades = _fades.nlargest(3, _own_col if _own_col in _fades.columns else "proj")

        # Chalk: high ownership but might still be good
        chalk = signals_df[(own >= 25) & ~signals_df["player_name"].isin(_used_names)]
        chalk = chalk.nlargest(2, _own_col if _own_col in chalk.columns else "proj")

        if not _fades.empty:
            fade_parts = []
            for _, r in _fades.iterrows():
                name = r.get("player_name", "?")
                o = r.get("ownership", r.get("own_pct", 0))
                fade_parts.append(f"{name} ({o:.0f}%)")
                _used_names.add(name)
            top_fades = _fades["player_name"].tolist()
            bullets.append(f"Fade candidates: {', '.join(fade_parts)} — chalk without edge")
        elif not chalk.empty:
            chalk_parts = []
            for _, r in chalk.iterrows():
                name = r.get("player_name", "?")
                o = r.get("ownership", r.get("own_pct", 0))
                chalk_parts.append(f"{name} ({o:.0f}%)")
                _used_names.add(name)
            bullets.append(f"Chalk alert: {', '.join(chalk_parts)} — pay up or fade")
        else:
            bullets.append("Ownership spread is flat — no extreme chalk")
    else:
        bullets.append("No ownership data — can't assess chalk")

    # ------------------------------------------------------------------
    # Bullet 4: SLATE TEXTURE — what kind of night is it?
    # Injury-driven? Minutes shifts? Condensed edges?
    # Only names NOT already used.
    # ------------------------------------------------------------------
    bump = _safe_numeric(signals_df.get("injury_bump_fp", pd.Series(0.0, index=signals_df.index)))
    pop_score = _safe_numeric(signals_df.get("pop_catalyst_score", pd.Series(0.0, index=signals_df.index)))

    cascade_count = int((bump > 0.5).sum())
    pop_count = int((pop_score >= 0.15).sum())

    _rolling5 = _safe_numeric(signals_df.get("rolling_min_5", pd.Series(0.0)))
    _rolling10 = _safe_numeric(signals_df.get("rolling_min_10", pd.Series(0.0)))
    _has_game_logs = _rolling5.max() > 0 and _rolling10.max() > 0
    riser_count = 0
    if _has_game_logs:
        _trend = _rolling5 - _rolling10
        riser_count = int((_trend > 3).sum())

    texture_parts = []
    if cascade_count > 0:
        texture_parts.append(f"{cascade_count} injury cascade{'s' if cascade_count != 1 else ''}")
        injury_impact = f"{cascade_count} players benefiting from injury cascades"
    if pop_count > 0:
        texture_parts.append(f"{pop_count} pop catalyst{'s' if pop_count != 1 else ''}")
        if not injury_impact:
            injury_impact = f"{pop_count} players with pop catalyst signals"
    if riser_count > 0:
        texture_parts.append(f"{riser_count} minutes riser{'s' if riser_count != 1 else ''}")

    if texture_parts:
        bullets.append(f"Slate drivers: {', '.join(texture_parts)} creating opportunity")
    else:
        bullets.append("Clean slate — no major injury or role shifts detected")

    # ------------------------------------------------------------------
    # Populate top_plays / top_fades for downstream use
    # ------------------------------------------------------------------
    top_edges_df = signals_df.nlargest(3, _edge_col if _has_edge else "proj")
    top_plays = top_edges_df["player_name"].tolist()

    # Trim to 4 bullets max — keep it tight
    bullets = bullets[:4]

    # Recommendation
    n_strong = (signals_df["n_signals"] >= 2).sum()
    if n_strong >= 5:
        recommendation = (
            f"Strong edge night — {n_strong} players with 2+ converging signals. "
            f"{'Play volume in GPP.' if 'GPP' in contest_type.upper() else 'Lock in core plays.'}"
        )
    elif n_strong >= 2:
        recommendation = (
            f"Moderate edges — {n_strong} multi-signal plays. "
            f"{'Be selective, target 1-2 stacks with leverage.' if 'GPP' in contest_type.upper() else 'Build around confirmed value.'}"
        )
    else:
        recommendation = (
            "Thin edge night — few converging signals. "
            "Consider smaller exposure or focus on chalk avoidance."
        )

    return {
        "bullets": bullets,
        "recommendation": recommendation,
        "top_plays": top_plays,
        "top_fades": top_fades,
        "injury_impact": injury_impact,
    }
