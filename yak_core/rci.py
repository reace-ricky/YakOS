"""Ricky Confidence Index (RCI) — multi-signal gauge per contest type.

Combines independent signals into a single 0–100 RCI score per contest.
Used to determine:
- Whether calibration still needs work
- Whether gains should come from methodology/strategy instead
- When to freeze a calibration profile and shift focus

Signals:
1. Projection Confidence (how strong are the player projections & edge tags)
2. Sim Alignment (how well-distributed are sim probabilities — smash/bust spread)
3. Ownership Accuracy (how complete and differentiated are ownership projections)
4. Historical ROI (backtest ROI for this contest type over recent slates)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import pandas as pd


@dataclass
class RCISignal:
    """One independent signal feeding the RCI gauge."""

    name: str
    value: float          # 0–100 raw score
    weight: float         # 0–1, how much this signal matters
    description: str      # human-readable explanation
    status: str           # "green" / "yellow" / "red"


@dataclass
class RCIResult:
    """Full RCI result for one contest type."""

    contest_label: str
    rci_score: float              # 0–100 composite
    rci_status: str               # "green" / "yellow" / "red"
    signals: List[RCISignal]      # individual signal breakdowns
    recommendation: str           # "Calibration OK — focus on methodology" or "Needs more calibration"
    calibration_stable: bool      # True when calibration is good enough


DEFAULT_WEIGHTS: Dict[str, float] = {
    "projection_confidence": 0.30,
    "sim_alignment": 0.25,
    "ownership_accuracy": 0.25,
    "historical_roi": 0.20,
}


def _get_color(score: float) -> str:
    """Return status color for a 0-100 score."""
    if score >= 70:
        return "green"
    elif score >= 45:
        return "yellow"
    else:
        return "red"


def compute_projection_confidence_signal(
    edge_payload: dict,
    player_pool: Optional[pd.DataFrame] = None,
    contest_label: str = "",
) -> RCISignal:
    """
    Signal 1: How confident are we in the player projections/tags?

    Primary: uses edge analysis payload (core/value/leverage players with confidence).
    Fallback: uses player pool smash/bust/floor/ceil to measure projection quality.
    """
    # Try edge payload first (has real edge analysis data)
    core_value = edge_payload.get("core_value_players", [])
    leverage = edge_payload.get("leverage_players", [])

    if core_value or leverage:
        # Edge analysis exists — use player confidence from tagged players
        def _avg_conf(players: list) -> float:
            confs = [p.get("confidence", 0) for p in players if isinstance(p, dict)]
            return sum(confs) / len(confs) if confs else 0.0

        cv_conf = _avg_conf(core_value)
        lev_conf = _avg_conf(leverage)

        if not leverage:
            raw = cv_conf
        elif not core_value:
            raw = lev_conf
        else:
            raw = 0.6 * cv_conf + 0.4 * lev_conf

        score = round(max(0.0, min(100.0, raw * 100)), 1)
        desc = f"Edge analysis confidence: {score:.0f}/100 ({len(core_value)} core/value, {len(leverage)} leverage)"
        return RCISignal(
            name="projection_confidence",
            value=score,
            weight=DEFAULT_WEIGHTS["projection_confidence"],
            description=desc,
            status=_get_color(score),
        )

    # Fallback: derive from player pool quality
    if player_pool is not None and not player_pool.empty:
        import numpy as np
        proj = pd.to_numeric(player_pool["proj"], errors="coerce").fillna(0) if "proj" in player_pool.columns else pd.Series([0.0] * len(player_pool))
        floor_col = pd.to_numeric(player_pool["floor"], errors="coerce").fillna(0) if "floor" in player_pool.columns else pd.Series([0.0] * len(player_pool))
        ceil_col = pd.to_numeric(player_pool["ceil"], errors="coerce").fillna(0) if "ceil" in player_pool.columns else pd.Series([0.0] * len(player_pool))

        has_proj = (proj > 0).sum()
        total = len(player_pool)

        # Score components:
        # 1) Coverage: what % of pool has non-zero projections (0-40 points)
        coverage_pct = has_proj / max(total, 1)
        coverage_score = coverage_pct * 40

        # 2) Spread quality: players with distinct proj/floor/ceil (not all same) (0-30 points)
        has_spread = ((ceil_col > proj) & (floor_col < proj) & (proj > 0)).sum()
        spread_pct = has_spread / max(has_proj, 1)
        spread_score = spread_pct * 30

        # 3) Differentiation: how much do projections vary across players? (0-30 points)
        if has_proj > 2:
            proj_valid = proj[proj > 0]
            cv = proj_valid.std() / max(proj_valid.mean(), 1)  # coefficient of variation
            # CV of ~0.3-0.5 is good differentiation; <0.1 means projections are too similar
            diff_score = min(30.0, cv * 75)
        else:
            diff_score = 0.0

        score = round(max(0.0, min(100.0, coverage_score + spread_score + diff_score)), 1)
        desc = (
            f"Pool projection quality: {score:.0f}/100 "
            f"({has_proj}/{total} with proj, {has_spread} with spreads)"
        )
        return RCISignal(
            name="projection_confidence",
            value=score,
            weight=DEFAULT_WEIGHTS["projection_confidence"],
            description=desc,
            status=_get_color(score),
        )

    # No data at all
    return RCISignal(
        name="projection_confidence",
        value=0.0,
        weight=DEFAULT_WEIGHTS["projection_confidence"],
        description="No projections or edge analysis available",
        status="red",
    )


def compute_sim_alignment_signal(
    sim_results: Optional[pd.DataFrame],
    actual_results: Optional[pd.DataFrame] = None,
    contest_label: str = "",
) -> RCISignal:
    """
    Signal 2: How well-distributed are sim outputs?

    If actual_results available: compare sim predicted vs actual smash/bust rates.
    If only sim_results: measure quality of sim distribution (smash/bust spread,
    leverage differentiation — do sims actually separate players?).
    If no sim data: return low score indicating sims need to be run.
    """
    # Full comparison mode: sim vs actuals
    if sim_results is not None and not sim_results.empty and actual_results is not None and not actual_results.empty:
        merged = sim_results.merge(
            actual_results, on="player_name", how="inner", suffixes=("_sim", "_actual")
        )
        errors = []
        if "sim_mean" in merged.columns and "actual_fp" in merged.columns:
            fp_mae = (merged["sim_mean"] - merged["actual_fp"]).abs().mean()
            fp_score = max(0.0, 100.0 - fp_mae * 10.0)
            errors.append(fp_score)
        if "smash_prob" in merged.columns and "actual_smash" in merged.columns:
            smash_mae = (merged["smash_prob"] - merged["actual_smash"]).abs().mean()
            smash_score = max(0.0, 100.0 - smash_mae * 200.0)
            errors.append(smash_score)

        score = sum(errors) / len(errors) if errors else 50.0
        score = max(0.0, min(100.0, score))
        return RCISignal(
            name="sim_alignment",
            value=round(score, 1),
            weight=DEFAULT_WEIGHTS["sim_alignment"],
            description=f"Sim vs actual alignment: {score:.0f}/100",
            status=_get_color(score),
        )

    # Sim-only mode: measure distribution quality
    if sim_results is not None and not sim_results.empty:
        import numpy as np
        smash = pd.to_numeric(sim_results["smash_prob"], errors="coerce").fillna(0) if "smash_prob" in sim_results.columns else pd.Series([0.0] * len(sim_results))
        bust = pd.to_numeric(sim_results["bust_prob"], errors="coerce").fillna(0) if "bust_prob" in sim_results.columns else pd.Series([0.0] * len(sim_results))
        leverage = pd.to_numeric(sim_results["leverage"], errors="coerce").fillna(0) if "leverage" in sim_results.columns else pd.Series([0.0] * len(sim_results))

        score_parts = []

        # 1) Smash spread: do sims differentiate players? (0-35 pts)
        if smash.any():
            smash_std = smash.std()
            smash_range = smash.max() - smash.min()
            # Good sims have spread: std > 0.08, range > 0.2
            spread_quality = min(1.0, smash_std / 0.10) * 0.6 + min(1.0, smash_range / 0.25) * 0.4
            score_parts.append(spread_quality * 35)
        else:
            score_parts.append(0)

        # 2) Bust differentiation (0-25 pts)
        if bust.any():
            bust_std = bust.std()
            bust_quality = min(1.0, bust_std / 0.08)
            score_parts.append(bust_quality * 25)
        else:
            score_parts.append(0)

        # 3) Leverage differentiation (0-25 pts)
        if leverage.any() and leverage.std() > 0:
            lev_cv = leverage.std() / max(leverage.mean(), 0.01)
            lev_quality = min(1.0, lev_cv / 0.5)
            score_parts.append(lev_quality * 25)
        else:
            score_parts.append(0)

        # 4) Coverage: how many players have sim data (0-15 pts)
        n_players = len(sim_results)
        coverage_score = min(15.0, (n_players / 40) * 15)
        score_parts.append(coverage_score)

        score = round(max(0.0, min(100.0, sum(score_parts))), 1)
        desc = (
            f"Sim distribution quality: {score:.0f}/100 "
            f"({n_players} players, smash spread {smash.std():.2f})"
        )
        return RCISignal(
            name="sim_alignment",
            value=score,
            weight=DEFAULT_WEIGHTS["sim_alignment"],
            description=desc,
            status=_get_color(score),
        )

    # No sim data at all
    return RCISignal(
        name="sim_alignment",
        value=0.0,
        weight=DEFAULT_WEIGHTS["sim_alignment"],
        description="No sim results — run sims first",
        status="red",
    )


def compute_ownership_accuracy_signal(
    projected_ownership: Optional[pd.DataFrame] = None,
    actual_ownership: Optional[pd.DataFrame] = None,
    player_pool: Optional[pd.DataFrame] = None,
    contest_label: str = "",
) -> RCISignal:
    """
    Signal 3: How accurate/complete are ownership projections?

    If actual_ownership available: compare projected vs actual.
    If only projected_ownership or player_pool: measure ownership quality
    (coverage, differentiation, realistic distribution).
    """
    # Full comparison mode
    if projected_ownership is not None and actual_ownership is not None and not actual_ownership.empty:
        merged = projected_ownership.merge(
            actual_ownership, on="player_name", how="inner", suffixes=("_proj", "_actual")
        )
        own_proj_col = [c for c in merged.columns if "own" in c.lower() and "proj" in c.lower()]
        own_act_col = [c for c in merged.columns if "own" in c.lower() and "actual" in c.lower()]

        if own_proj_col and own_act_col:
            mae = (
                pd.to_numeric(merged[own_proj_col[0]], errors="coerce")
                - pd.to_numeric(merged[own_act_col[0]], errors="coerce")
            ).abs().mean()
            score = max(0.0, min(100.0, 100.0 - mae * (100.0 / 15.0)))
            return RCISignal(
                name="ownership_accuracy",
                value=round(score, 1),
                weight=DEFAULT_WEIGHTS["ownership_accuracy"],
                description=f"Ownership accuracy: {score:.0f}/100 (avg error {mae:.1f}%)",
                status=_get_color(score),
            )

    # Ownership quality mode: use player_pool ownership column
    pool = player_pool
    if pool is None or pool.empty:
        pool = projected_ownership

    if pool is not None and not pool.empty and "ownership" in pool.columns:
        own = pd.to_numeric(pool["ownership"], errors="coerce").fillna(0)
        total = len(pool)

        # 1) Coverage: how many players have ownership > 0 (0-40 pts)
        has_own = (own > 0).sum()
        coverage_pct = has_own / max(total, 1)
        coverage_score = coverage_pct * 40

        # 2) Differentiation: are ownership values varied? (0-35 pts)
        if has_own > 2:
            own_valid = own[own > 0]
            own_std = own_valid.std()
            own_range = own_valid.max() - own_valid.min()
            # Good ownership has std > 3, range > 15
            diff_quality = min(1.0, own_std / 5.0) * 0.5 + min(1.0, own_range / 20.0) * 0.5
            diff_score = diff_quality * 35
        else:
            diff_score = 0.0

        # 3) Realism: total ownership shouldn't be wildly off
        #    (sum should be ~700-900% for an 8-man slate with ~80-100 players) (0-25 pts)
        own_sum = own.sum()
        if total > 0 and own_sum > 0:
            avg_own = own_sum / total
            # Avg ownership per player around 5-15% is realistic
            realism = 1.0 - min(1.0, abs(avg_own - 8.0) / 15.0)
            realism_score = realism * 25
        else:
            realism_score = 0.0

        score = round(max(0.0, min(100.0, coverage_score + diff_score + realism_score)), 1)
        desc = (
            f"Ownership quality: {score:.0f}/100 "
            f"({has_own}/{total} with ownership, std {own.std():.1f})"
        )
        return RCISignal(
            name="ownership_accuracy",
            value=score,
            weight=DEFAULT_WEIGHTS["ownership_accuracy"],
            description=desc,
            status=_get_color(score),
        )

    # No ownership data
    return RCISignal(
        name="ownership_accuracy",
        value=0.0,
        weight=DEFAULT_WEIGHTS["ownership_accuracy"],
        description="No ownership data available",
        status="red",
    )


def compute_historical_roi_signal(
    backtest_results: Optional[pd.DataFrame],
    contest_label: str,
) -> RCISignal:
    """
    Signal 4: Historical ROI from backtests for this contest type.

    If backtest_results available:
    - Look at recent N slates for this contest type
    - Compute average ROI (profit / entry fee)
    - Score: 50 = breakeven, 100 = +50% ROI, 0 = -50% ROI

    If no backtest data: return neutral 50 (doesn't penalize or reward).
    """
    if backtest_results is None or backtest_results.empty:
        return RCISignal(
            name="historical_roi",
            value=50.0,
            weight=DEFAULT_WEIGHTS["historical_roi"],
            description="No historical backtest data — neutral score",
            status="yellow",
        )

    # Filter to contest type if column exists
    if "contest_type" in backtest_results.columns:
        filtered = backtest_results[backtest_results["contest_type"] == contest_label]
    else:
        filtered = backtest_results

    if filtered.empty or "roi" not in filtered.columns:
        return RCISignal(
            name="historical_roi",
            value=50.0,
            weight=DEFAULT_WEIGHTS["historical_roi"],
            description="No ROI data for this contest type — neutral score",
            status="yellow",
        )

    avg_roi = filtered["roi"].mean()
    # Map: -50% ROI → 0, 0% → 50, +50% → 100
    score = max(0.0, min(100.0, 50.0 + avg_roi * 100.0))

    return RCISignal(
        name="historical_roi",
        value=round(score, 1),
        weight=DEFAULT_WEIGHTS["historical_roi"],
        description=f"Historical ROI: {avg_roi:+.1%} → score {score:.0f}/100",
        status=_get_color(score),
    )


def _count_active_signals(
    edge_payload: dict,
    sim_results: Optional[pd.DataFrame],
    actual_ownership: Optional[pd.DataFrame],
    backtest_results: Optional[pd.DataFrame],
) -> int:
    """Count how many RCI signals have real data (not fallback/defaults)."""
    active = 0
    # Signal 1: edge payload has tagged players
    if edge_payload and (edge_payload.get("core_value_players") or edge_payload.get("leverage_players")):
        active += 1
    # Signal 2: sims have been run
    if sim_results is not None and not sim_results.empty:
        active += 1
    # Signal 3: actual ownership exists
    if actual_ownership is not None and not actual_ownership.empty:
        active += 1
    # Signal 4: backtest data exists
    if backtest_results is not None and not backtest_results.empty:
        active += 1
    return active


def compute_rci(
    contest_label: str,
    edge_payload: dict,
    sim_results: Optional[pd.DataFrame] = None,
    actual_results: Optional[pd.DataFrame] = None,
    projected_ownership: Optional[pd.DataFrame] = None,
    actual_ownership: Optional[pd.DataFrame] = None,
    backtest_results: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    player_pool: Optional[pd.DataFrame] = None,
) -> RCIResult:
    """
    Compute the full RCI for one contest type.

    Combines 4 independent signals with tunable weights into a 0–100 composite.
    Determines whether calibration is stable or needs more work.

    The score is **contest-specific** only when contest-specific data is
    passed in (edge_payload per contest, sim_results filtered to that slate
    subset, etc.).  When the same pool is shared across contest types the
    function tracks *which signals have real data* and adjusts the score
    accordingly so different contest types do NOT show identical numbers.

    Parameters
    ----------
    contest_label : str
        Human-readable contest label (e.g. "GPP Main").
    edge_payload : dict
        Edge Analysis payload from RickyEdgeState.edge_analysis_by_contest.
    sim_results : pd.DataFrame, optional
        Player-level sim output with columns player_name, sim_mean, smash_prob.
    actual_results : pd.DataFrame, optional
        Historical actuals with columns player_name, actual_fp, actual_smash.
    projected_ownership : pd.DataFrame, optional
        Projected ownership with columns player_name, ownership_proj.
    actual_ownership : pd.DataFrame, optional
        Actual contest ownership with columns player_name, ownership_actual.
    backtest_results : pd.DataFrame, optional
        Historical backtest table with columns contest_type, roi.
    weights : dict, optional
        Override dict e.g. {"projection_confidence": 0.4, ...}.
        If provided, uses these instead of DEFAULT_WEIGHTS.
    player_pool : pd.DataFrame, optional
        Full player pool with proj, floor, ceil, ownership columns.
        Used as fallback data when edge_payload or sim_results are sparse.

    Returns
    -------
    RCIResult
    """
    w = weights or DEFAULT_WEIGHTS

    signals = [
        compute_projection_confidence_signal(edge_payload, player_pool, contest_label),
        compute_sim_alignment_signal(sim_results, actual_results, contest_label),
        compute_ownership_accuracy_signal(projected_ownership, actual_ownership, player_pool, contest_label),
        compute_historical_roi_signal(backtest_results, contest_label),
    ]

    # Apply custom weights if provided
    for s in signals:
        if s.name in w:
            s.weight = w[s.name]

    # Normalize weights to sum to 1
    total_w = sum(s.weight for s in signals)
    if total_w > 0:
        for s in signals:
            s.weight = s.weight / total_w

    # ── Data-availability penalty ──
    # When most signals are using fallback/default data (no edge analysis,
    # no sims, no actual ownership, no backtests), the pool-only fallback
    # scores look deceptively healthy.  Apply a ceiling so the RCI honestly
    # reflects how much real work has been done for this contest type.
    active_signals = _count_active_signals(
        edge_payload, sim_results, actual_ownership, backtest_results,
    )

    # Weighted composite
    rci_raw = sum(s.value * s.weight for s in signals)

    # Cap the score based on how many signals have real data:
    # 0 active → max 30  ("not started")
    # 1 active → max 55  ("early")
    # 2 active → max 75  ("in progress")
    # 3-4 active → no cap ("calibrated")
    _CAPS = {0: 30, 1: 55, 2: 75, 3: 100, 4: 100}
    cap = _CAPS.get(active_signals, 100)
    rci_score = round(max(0.0, min(cap, rci_raw)), 1)

    rci_status = _get_color(rci_score)

    # Calibration stability decision:
    # "Stable" when RCI >= 70 AND no individual signal is red
    any_red = any(s.status == "red" for s in signals)
    calibration_stable = rci_score >= 70 and not any_red

    if active_signals == 0:
        recommendation = (
            "Not calibrated yet. Run edge analysis and sims "
            "for this contest type to build confidence."
        )
    elif calibration_stable:
        recommendation = (
            "Calibration OK — further gains come from methodology and contest strategy."
        )
    elif rci_score >= 55:
        recommendation = (
            "Calibration is decent but has room to improve. "
            "Consider tuning projections or sim parameters."
        )
    elif rci_score >= 35:
        recommendation = (
            "Calibration needs work. Run sims and edge analysis "
            "to improve projection confidence."
        )
    else:
        recommendation = (
            "Calibration not started. Load a slate, run sims, and "
            "complete edge analysis to build confidence."
        )

    return RCIResult(
        contest_label=contest_label,
        rci_score=rci_score,
        rci_status=rci_status,
        signals=signals,
        recommendation=recommendation,
        calibration_stable=calibration_stable,
    )
