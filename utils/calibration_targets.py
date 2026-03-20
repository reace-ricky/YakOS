"""Calibration target ranges and nudge guidance for Sim Lab batch results.

Each profile key maps to a dict of metric → (min, max) acceptable range.
``evaluate_metric`` returns a status string and directional nudge text.
"""
from __future__ import annotations

from typing import Optional

# ── Target ranges by profile_key ─────────────────────────────────────────────
# Each entry: metric_name → (lo, hi)  — both ends inclusive.
# A metric within [lo, hi] is "green"; within 10 % of a boundary is "yellow";
# outside is "red".

CALIBRATION_TARGETS: dict[str, dict[str, tuple[float, float]]] = {
    "classic_gpp_main": {
        "mae": (5.5, 7.5),
        "bias": (-1.0, 1.0),
        "correlation": (0.78, 1.0),
        "avg_score": (320.0, 400.0),
        "ownership_sum": (90.0, 130.0),
        "top_1pct_rate": (0.04, 1.0),
    },
    "classic_gpp_20max": {
        "mae": (5.5, 7.5),
        "bias": (-1.0, 1.0),
        "correlation": (0.78, 1.0),
        "avg_score": (310.0, 390.0),
        "ownership_sum": (80.0, 120.0),
        "top_1pct_rate": (0.03, 1.0),
        "lineup_diversity_min_cores": (3.0, 20.0),
    },
    "classic_gpp_se": {
        "mae": (5.5, 7.5),
        "bias": (-1.0, 1.0),
        "correlation": (0.78, 1.0),
        "avg_score": (305.0, 380.0),
        "ownership_sum": (60.0, 90.0),
        "top_1pct_rate": (0.03, 1.0),
    },
    "classic_cash": {
        "mae": (5.0, 7.0),
        "bias": (-0.5, 0.5),
        "correlation": (0.80, 1.0),
        "avg_score": (290.0, 360.0),
        "cash_rate": (0.55, 1.0),
    },
    "showdown_gpp": {
        "mae": (6.0, 9.0),
        "bias": (-1.5, 1.5),
        "correlation": (0.70, 1.0),
        "avg_score": (140.0, 200.0),
        "ownership_sum": (70.0, 120.0),
    },
    "showdown_cash": {
        "mae": (6.0, 9.0),
        "bias": (-1.0, 1.0),
        "correlation": (0.72, 1.0),
        "avg_score": (130.0, 180.0),
        "cash_rate": (0.55, 1.0),
    },
}

# ── Ricky Ranking Weight targets ─────────────────────────────────────────────
# Applied to ALL GPP profiles.  These calibrate the lineup *selection* layer
# (which 3 out of N lineups get the SE Core / Spicy / Alt tags).
# Metrics are computed per-slate then averaged across the batch.
#
#   ricky_top3_lift  — % by which Ricky top-3 avg actual beats pool avg actual
#                      (positive = good).  Range: want +3 % to +20 %.
#   ricky_top3_hit   — fraction of slates where ≥1 Ricky pick is in the actual
#                      top-5 of the pool.  Range: want ≥ 0.30.
_RICKY_WEIGHT_TARGETS: dict[str, tuple[float, float]] = {
    "ricky_top3_lift": (3.0, 20.0),
    "ricky_top3_hit":  (0.30, 1.0),
}

# Inject into every GPP profile (not cash profiles)
_GPP_PROFILES = [
    "classic_gpp_main", "classic_gpp_20max", "classic_gpp_se",
    "showdown_gpp",
]
for _pk in _GPP_PROFILES:
    if _pk in CALIBRATION_TARGETS:
        CALIBRATION_TARGETS[_pk].update(_RICKY_WEIGHT_TARGETS)

# ── Human-readable display labels ────────────────────────────────────────────
METRIC_LABELS: dict[str, str] = {
    "mae": "Projection MAE",
    "bias": "Projection Bias",
    "correlation": "Correlation",
    "avg_score": "Avg Lineup Score",
    "ownership_sum": "Avg Ownership Sum",
    "top_1pct_rate": "Top-1% Hit Rate",
    "cash_rate": "Cash Rate",
    "lineup_diversity_min_cores": "Lineup Diversity (unique cores)",
    "ricky_top3_lift": "Ricky Top-3 Lift %",
    "ricky_top3_hit": "Ricky Top-3 Hit Rate",
}

# ── Nudge text: (metric, direction) → recommendation ─────────────────────────
# direction is "low" or "high".
NUDGE_TEXT: dict[tuple[str, str], str] = {
    ("mae", "low"): (
        "Projections are very tight — may be overfitting. "
        "Consider widening variance."
    ),
    ("mae", "high"): (
        "Projections are loose. "
        "Review player model or apply correction factors."
    ),
    ("bias", "low"): (
        "Under-projecting. "
        "Increase overall bias correction or check position-specific corrections."
    ),
    ("bias", "high"): "Over-projecting. Decrease bias correction.",
    ("correlation", "low"): (
        "Low correlation — projection model may need retraining on recent slates."
    ),
    ("correlation", "high"): "",  # higher is always better
    ("avg_score", "low"): (
        "Lineups scoring too low. "
        "Raise projection floor or reduce punt plays."
    ),
    ("avg_score", "high"): (
        "Avg score very high — may indicate too much chalk. "
        "Consider adding more ownership diversity."
    ),
    ("ownership_sum", "low"): (
        "Too contrarian — may be sacrificing floor. "
        "Reduce chalk penalty."
    ),
    ("ownership_sum", "high"): (
        "Too chalky — increase chalk penalty or raise uniqueness floor."
    ),
    ("top_1pct_rate", "low"): (
        "Not enough ceiling. "
        "Increase correlation weight, allow more stacking, raise ceiling boost."
    ),
    ("top_1pct_rate", "high"): "",  # n/a
    ("cash_rate", "low"): (
        "Cash rate below target. "
        "Raise floor weight, lower ceiling weight, increase min projection threshold."
    ),
    ("cash_rate", "high"): "",  # n/a
    ("lineup_diversity_min_cores", "low"): (
        "Not enough differentiation across lineups. "
        "Lower max exposure, increase diversity penalty."
    ),
    ("lineup_diversity_min_cores", "high"): "",  # n/a
    ("ricky_top3_lift", "low"): (
        "Ricky picks not outperforming the pool. "
        "Increase ceiling weight or reduce ownership penalty."
    ),
    ("ricky_top3_lift", "high"): "",  # higher lift is always good
    ("ricky_top3_hit", "low"): (
        "Ricky picks rarely land in the actual top 5. "
        "Increase GPP score weight to lean on projection quality."
    ),
    ("ricky_top3_hit", "high"): "",  # higher hit rate is always good
}


def evaluate_metric(
    metric_name: str,
    value: float,
    profile_key: str,
) -> tuple[str, str, str]:
    """Evaluate a metric against the target range for a profile key.

    Parameters
    ----------
    metric_name:
        One of the keys in ``CALIBRATION_TARGETS`` (e.g. ``"mae"``).
    value:
        The observed batch value for this metric.
    profile_key:
        Canonical profile key (e.g. ``"classic_gpp_main"``).

    Returns
    -------
    status : str
        ``"green"``, ``"yellow"``, or ``"red"``.
    dot : str
        Emoji shorthand for the status (🟢 / 🟡 / 🔴).
    nudge : str
        Directional recommendation, or ``"On target"`` when within range.
    """
    targets = CALIBRATION_TARGETS.get(profile_key, {})
    if metric_name not in targets:
        return "grey", "⚪", "No target defined"

    lo, hi = targets[metric_name]

    # Determine direction
    if value < lo:
        direction = "low"
    elif value > hi:
        direction = "high"
    else:
        direction = "on_target"

    if direction == "on_target":
        return "green", "🟢", "On target"

    # Yellow zone: within 10 % of the boundary that was breached
    span = hi - lo if hi != lo else 1.0
    margin = abs(span) * 0.10
    if direction == "low" and value >= lo - margin:
        status = "yellow"
        dot = "🟡"
    elif direction == "high" and value <= hi + margin:
        status = "yellow"
        dot = "🟡"
    else:
        status = "red"
        dot = "🔴"

    nudge = NUDGE_TEXT.get((metric_name, direction), "")
    if not nudge:
        nudge = "Review this metric."

    return status, dot, nudge


def get_target_display(metric_name: str, profile_key: str) -> str:
    """Return a human-readable target range string, e.g. ``'5.5 – 7.5'``."""
    targets = CALIBRATION_TARGETS.get(profile_key, {})
    if metric_name not in targets:
        return "—"
    lo, hi = targets[metric_name]
    # For metrics where hi=1.0 represents the natural maximum (rates, correlation),
    # show as ">= lo" instead of "lo – 1.0".
    _unbounded_hi_metrics = {"top_1pct_rate", "cash_rate", "correlation", "ricky_top3_hit"}
    if metric_name in _unbounded_hi_metrics and hi == 1.0 and lo < 1.0:
        if metric_name in ("top_1pct_rate", "cash_rate"):
            return f"≥ {lo:.0%}"
        return f"≥ {lo}"
    return f"{lo} – {hi}"
