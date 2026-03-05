"""Ricky Confidence Index (RCI) — multi-signal gauge per contest type.

Combines independent signals into a single 0–100 RCI score per contest.
Used to determine:
- Whether calibration still needs work
- Whether gains should come from methodology/strategy instead
- When to freeze a calibration profile and shift focus

Signals:
1. Projection Confidence (from edge_metrics — how confident are we in player tags/projections)
2. Sim Alignment (how well do sims match realized results from backtests)
3. Ownership Accuracy (how close projected ownership was to actual)
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


def compute_projection_confidence_signal(
    edge_payload: dict,
) -> RCISignal:
    """
    Signal 1: How confident are we in the player projections/tags?
    Uses compute_ricky_confidence_for_contest from edge_metrics.
    """
    from yak_core.edge_metrics import compute_ricky_confidence_for_contest, get_confidence_color
    score = compute_ricky_confidence_for_contest(edge_payload)
    return RCISignal(
        name="projection_confidence",
        value=score,
        weight=DEFAULT_WEIGHTS["projection_confidence"],
        description=f"Edge analysis confidence: {score:.0f}/100",
        status=get_confidence_color(score),
    )


def compute_sim_alignment_signal(
    sim_results: Optional[pd.DataFrame],
    actual_results: Optional[pd.DataFrame],
) -> RCISignal:
    """
    Signal 2: How well do sim distributions match realized outcomes?

    If actual_results available (from historical backtest):
    - Compare sim predicted smash rates vs actual smash rates
    - Compare sim predicted bust rates vs actual bust rates
    - Compare sim mean vs actual mean fantasy points
    - Score = 100 - (mean absolute error across metrics, scaled)

    If no actual_results: return neutral 50 with "yellow" status.
    """
    if sim_results is None or actual_results is None or actual_results.empty:
        return RCISignal(
            name="sim_alignment",
            value=50.0,
            weight=DEFAULT_WEIGHTS["sim_alignment"],
            description="No backtest data available — neutral score",
            status="yellow",
        )

    # Compare sim predictions to actuals
    # Merge on player_name
    merged = sim_results.merge(
        actual_results, on="player_name", how="inner", suffixes=("_sim", "_actual")
    )

    errors = []
    # Mean FP error
    if "sim_mean" in merged.columns and "actual_fp" in merged.columns:
        fp_mae = (merged["sim_mean"] - merged["actual_fp"]).abs().mean()
        # Normalize: 0 error = 100, 10+ error = 0
        fp_score = max(0.0, 100.0 - fp_mae * 10.0)
        errors.append(fp_score)

    # Smash rate accuracy
    if "smash_prob" in merged.columns and "actual_smash" in merged.columns:
        smash_mae = (merged["smash_prob"] - merged["actual_smash"]).abs().mean()
        smash_score = max(0.0, 100.0 - smash_mae * 200.0)
        errors.append(smash_score)

    score = sum(errors) / len(errors) if errors else 50.0
    score = max(0.0, min(100.0, score))

    from yak_core.edge_metrics import get_confidence_color
    return RCISignal(
        name="sim_alignment",
        value=round(score, 1),
        weight=DEFAULT_WEIGHTS["sim_alignment"],
        description=f"Sim vs actual alignment: {score:.0f}/100",
        status=get_confidence_color(score),
    )


def compute_ownership_accuracy_signal(
    projected_ownership: Optional[pd.DataFrame],
    actual_ownership: Optional[pd.DataFrame],
) -> RCISignal:
    """
    Signal 3: How accurate were our ownership projections?

    Compare projected ownership % vs actual ownership % per player.
    Score = 100 - mean absolute ownership error (scaled).

    If no actual ownership data: return neutral 50.
    """
    if projected_ownership is None or actual_ownership is None or actual_ownership.empty:
        return RCISignal(
            name="ownership_accuracy",
            value=50.0,
            weight=DEFAULT_WEIGHTS["ownership_accuracy"],
            description="No actual ownership data — neutral score",
            status="yellow",
        )

    merged = projected_ownership.merge(
        actual_ownership, on="player_name", how="inner", suffixes=("_proj", "_actual")
    )

    own_proj_col = [c for c in merged.columns if "own" in c.lower() and "proj" in c.lower()]
    own_act_col = [c for c in merged.columns if "own" in c.lower() and "actual" in c.lower()]

    if not own_proj_col or not own_act_col:
        return RCISignal(
            name="ownership_accuracy",
            value=50.0,
            weight=DEFAULT_WEIGHTS["ownership_accuracy"],
            description="Ownership columns not found — neutral score",
            status="yellow",
        )

    # Prefer the most specific match; take the first candidate if multiple exist
    proj_col = own_proj_col[0]
    act_col = own_act_col[0]
    mae = (
        pd.to_numeric(merged[proj_col], errors="coerce")
        - pd.to_numeric(merged[act_col], errors="coerce")
    ).abs().mean()
    # 0% error = 100, 15%+ avg error = 0
    score = max(0.0, min(100.0, 100.0 - mae * (100.0 / 15.0)))

    from yak_core.edge_metrics import get_confidence_color
    return RCISignal(
        name="ownership_accuracy",
        value=round(score, 1),
        weight=DEFAULT_WEIGHTS["ownership_accuracy"],
        description=f"Ownership accuracy: {score:.0f}/100 (avg error {mae:.1f}%)",
        status=get_confidence_color(score),
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

    If no backtest data: return neutral 50.
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

    from yak_core.edge_metrics import get_confidence_color
    return RCISignal(
        name="historical_roi",
        value=round(score, 1),
        weight=DEFAULT_WEIGHTS["historical_roi"],
        description=f"Historical ROI: {avg_roi:+.1%} → score {score:.0f}/100",
        status=get_confidence_color(score),
    )


def compute_rci(
    contest_label: str,
    edge_payload: dict,
    sim_results: Optional[pd.DataFrame] = None,
    actual_results: Optional[pd.DataFrame] = None,
    projected_ownership: Optional[pd.DataFrame] = None,
    actual_ownership: Optional[pd.DataFrame] = None,
    backtest_results: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
) -> RCIResult:
    """
    Compute the full RCI for one contest type.

    Combines 4 independent signals with tunable weights into a 0–100 composite.
    Determines whether calibration is stable or needs more work.

    Parameters
    ----------
    contest_label : str
        Human-readable contest label (e.g. "GPP - 20 Max").
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

    Returns
    -------
    RCIResult
    """
    w = weights or DEFAULT_WEIGHTS

    signals = [
        compute_projection_confidence_signal(edge_payload),
        compute_sim_alignment_signal(sim_results, actual_results),
        compute_ownership_accuracy_signal(projected_ownership, actual_ownership),
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

    # Weighted composite
    rci_score = sum(s.value * s.weight for s in signals)
    rci_score = round(max(0.0, min(100.0, rci_score)), 1)

    from yak_core.edge_metrics import get_confidence_color
    rci_status = get_confidence_color(rci_score)

    # Calibration stability decision:
    # "Stable" when RCI >= 70 AND no individual signal is red
    any_red = any(s.status == "red" for s in signals)
    calibration_stable = rci_score >= 70 and not any_red

    if calibration_stable:
        recommendation = (
            "Calibration OK — further gains come from methodology and contest strategy."
        )
    elif rci_score >= 55:
        recommendation = (
            "Calibration is decent but has room to improve. "
            "Consider tuning projections or sim parameters."
        )
    else:
        recommendation = (
            "Calibration needs work. Focus on projection accuracy and sim alignment "
            "before trusting methodology conclusions."
        )

    return RCIResult(
        contest_label=contest_label,
        rci_score=rci_score,
        rci_status=rci_status,
        signals=signals,
        recommendation=recommendation,
        calibration_stable=calibration_stable,
    )
