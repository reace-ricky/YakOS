"""Config Evolution analysis for the Calibration Lab.

Analyzes config_history.json and active_config.json to produce
before/after comparisons, parameter trends, insights, and confidence
assessments that make training progress visible at a glance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Human-readable labels for slider keys
PARAM_LABELS: Dict[str, str] = {
    "proj_weight": "Proj Weight",
    "upside_weight": "Upside Weight",
    "boom_weight": "Boom Weight",
    "own_penalty_strength": "Own Penalty",
    "low_own_boost": "Low Own Boost",
    "own_neutral_pct": "Own Neutral %",
    "max_punt_players": "Max Punts",
    "min_mid_players": "Min Mid Players",
    "game_diversity_pct": "Game Diversity %",
    "stud_exposure": "Stud Exposure %",
    "mid_exposure": "Mid Exposure %",
    "value_exposure": "Value Exposure %",
}


@dataclass
class ParamEvolution:
    """Tracks how a single parameter evolved across slates."""

    key: str
    label: str
    default_value: float
    current_value: float
    # value at each slate snapshot: list of (slate_date, value)
    history: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def change(self) -> float:
        return self.current_value - self.default_value

    @property
    def changed(self) -> bool:
        return abs(self.change) > 1e-9

    @property
    def direction(self) -> str:
        if self.change > 1e-9:
            return "up"
        elif self.change < -1e-9:
            return "down"
        return "unchanged"

    @property
    def direction_arrow(self) -> str:
        if self.direction == "up":
            return "↑"
        elif self.direction == "down":
            return "↓"
        return "—"

    @property
    def confidence(self) -> str:
        """Assess confidence based on consistency of direction across slates."""
        if len(self.history) < 2:
            return "low"
        deltas = []
        for i in range(1, len(self.history)):
            d = self.history[i][1] - self.history[i - 1][1]
            if abs(d) > 1e-9:
                deltas.append(d)
        if not deltas:
            return "stable"
        positive = sum(1 for d in deltas if d > 0)
        negative = sum(1 for d in deltas if d < 0)
        total = len(deltas)
        # High confidence if >= 75% moves in same direction
        if positive / total >= 0.75 or negative / total >= 0.75:
            return "high"
        # Medium if >= 50%
        if positive / total >= 0.5 or negative / total >= 0.5:
            return "medium"
        return "low"

    def trend_description(self) -> str:
        """Generate a human-readable insight about this parameter's trend."""
        if not self.changed and not self.history:
            return f"{self.label} has stayed at default ({self.default_value}) — well-calibrated."

        n_slates = len(self.history)
        if n_slates < 2:
            if self.changed:
                return (
                    f"{self.label} moved from {self.default_value} to "
                    f"{self.current_value} — only 1 data point so far."
                )
            return f"{self.label} is at default ({self.default_value})."

        # Count directional moves
        increases = 0
        decreases = 0
        for i in range(1, len(self.history)):
            d = self.history[i][1] - self.history[i - 1][1]
            if d > 1e-9:
                increases += 1
            elif d < -1e-9:
                decreases += 1

        total_moves = increases + decreases
        if total_moves == 0:
            return f"{self.label} has been stable at {self.current_value} across all {n_slates} slates — well-calibrated."

        vals_str = " → ".join(f"{v}" for _, v in self.history)

        if self.confidence == "high":
            if increases > decreases:
                return (
                    f"{self.label} has increased on {increases} of {n_slates - 1} changes "
                    f"({vals_str}) — strong signal."
                )
            return (
                f"{self.label} has decreased on {decreases} of {n_slates - 1} changes "
                f"({vals_str}) — strong signal."
            )
        elif self.confidence == "low":
            return (
                f"{self.label} bounced between values ({vals_str}) — "
                f"no clear direction, may need more data."
            )
        else:
            dominant = "up" if increases > decreases else "down"
            return (
                f"{self.label} trending {dominant} ({vals_str}) — moderate signal."
            )


@dataclass
class EvolutionSummary:
    """Full evolution analysis result."""

    contest_type: str
    slates_trained: List[str]
    total_changes: int
    params: List[ParamEvolution]
    maturity_label: str
    maturity_recommendation: str

    @property
    def changed_params(self) -> List[ParamEvolution]:
        return [p for p in self.params if p.changed]

    @property
    def unchanged_params(self) -> List[ParamEvolution]:
        return [p for p in self.params if not p.changed]

    @property
    def high_confidence_params(self) -> List[ParamEvolution]:
        return [p for p in self.params if p.confidence == "high" and p.changed]

    @property
    def low_confidence_params(self) -> List[ParamEvolution]:
        return [p for p in self.params if p.confidence == "low" and p.changed]


def analyze_evolution(
    active_config: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    defaults: Dict[str, Any],
    contest_type: str = "gpp",
) -> Optional[EvolutionSummary]:
    """Analyze config evolution for a given contest type.

    Returns None if there's no active config or no training data.
    """
    if not active_config:
        return None

    ct = contest_type.lower()
    ct_config = active_config.get(ct, {})
    current_values = ct_config.get("values", {})
    slates_trained = ct_config.get("slates_trained", [])

    if not current_values and not slates_trained:
        return None

    # Filter history to this contest type
    ct_history = [h for h in history if h.get("contest_type", "gpp") == ct]

    # Build per-parameter evolution
    # Group history entries by slate_date to get the parameter value at each slate
    slate_snapshots: Dict[str, Dict[str, float]] = {}
    for entry in ct_history:
        slate = entry.get("slate_date")
        vals = entry.get("values", {})
        if slate and vals:
            # Keep the latest values for each slate (in case of multiple entries per slate)
            slate_snapshots[slate] = vals

    # Order snapshots by the slate order in slates_trained
    ordered_slates = []
    for s in slates_trained:
        if s in slate_snapshots:
            ordered_slates.append((s, slate_snapshots[s]))

    # Also include snapshots for slates not in slates_trained (e.g., from history)
    seen = set(slates_trained)
    for entry in ct_history:
        slate = entry.get("slate_date")
        if slate and slate not in seen and slate in slate_snapshots:
            ordered_slates.append((slate, slate_snapshots[slate]))
            seen.add(slate)

    # Build ParamEvolution for each tracked parameter
    tracked_keys = [k for k in PARAM_LABELS if k in defaults]
    params: List[ParamEvolution] = []
    total_changes = 0

    for key in tracked_keys:
        default_val = defaults.get(key, 0)
        current_val = current_values.get(key, default_val)

        # Build history trace for this param
        param_history: List[Tuple[str, float]] = []
        # Start with default as implicit first point if we have slates
        if ordered_slates:
            # Add the value at each slate snapshot
            for slate_date, snap_vals in ordered_slates:
                val = snap_vals.get(key, default_val)
                param_history.append((slate_date, val))

        pe = ParamEvolution(
            key=key,
            label=PARAM_LABELS[key],
            default_value=default_val,
            current_value=current_val,
            history=param_history,
        )
        if pe.changed:
            total_changes += 1
        params.append(pe)

    # Maturity assessment
    n_slates = len(slates_trained)
    if n_slates == 0:
        maturity_label = "Not started"
        maturity_rec = "Train on at least 1 slate to begin calibration."
    elif n_slates <= 2:
        maturity_label = "Early stage"
        maturity_rec = f"{n_slates} slate(s) trained — recommend 8-10 for stable config."
    elif n_slates <= 5:
        maturity_label = "Developing"
        maturity_rec = f"{n_slates} slates trained — trends emerging. Recommend 8-10 for stability."
    elif n_slates <= 9:
        maturity_label = "Maturing"
        maturity_rec = f"{n_slates} slates trained — config settling. A few more slates for full confidence."
    else:
        maturity_label = "Stable"
        maturity_rec = f"{n_slates} slates trained — config is well-calibrated."

    return EvolutionSummary(
        contest_type=ct,
        slates_trained=slates_trained,
        total_changes=total_changes,
        params=params,
        maturity_label=maturity_label,
        maturity_recommendation=maturity_rec,
    )
