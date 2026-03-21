"""Prescriptive nudge parameter mappings for calibration guidance (RIG-8).

Maps ``(metric_name, direction)`` → list of parameter nudge rule dicts.
Each rule specifies the exact ``config.py`` key to adjust, a formula for
computing the suggested value, and the slider bounds used in the Sim Lab
config panel.

Includes guardrails (absolute min/max bounds), cross-effect conflict
detection, and proportional step sizing based on how far off-target a
metric is.

Usage::

    from utils.nudge_params import NUDGE_PARAM_RULES, get_nudge_suggestions

    suggestions = get_nudge_suggestions(
        metric_name="ownership_sum",
        batch_value=145.0,
        profile_key="classic_gpp_main",
        current_overrides=sandbox_overrides,
        preset_name="GPP Main",
        off_target_metrics={"avg_score": "low", "bias": "high"},
    )
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Parameter guardrails — absolute bounds that nudges must never exceed
# ---------------------------------------------------------------------------

PARAM_GUARDRAILS: dict[str, tuple[float, float]] = {
    "GPP_PROJ_WEIGHT": (0.05, 0.60),           # was (0.20, 0.60) — DS says 0.15, allow down to 0.05
    "GPP_UPSIDE_WEIGHT": (0.0, 0.70),            # was (0.10, 0.60) — DS says 0.45 for GPP, 0.0 for Cash
    "GPP_BOOM_WEIGHT": (0.0, 0.50),             # was (0.10, 0.50) — DS says 0.35, allow 0.0
    "GPP_OWN_PENALTY_STRENGTH": (0.0, 2.5),     # was (0.5, 2.5) — DS says 0.30, must allow below 0.5
    "GPP_BUST_PENALTY": (0.0, 0.30),
    "GPP_LOW_OWN_THRESHOLD": (0.02, 0.40),      # was (0.05, 0.40) — allow tighter threshold
    "GPP_SMASH_WEIGHT": (0.0, 0.30),            # DS says 0.0 (binary treatment)
    "GPP_LEVERAGE_WEIGHT": (0.0, 0.30),          # DS says 0.0
    "OWN_WEIGHT": (0.0, 0.30),                  # DS says 0.0
    "MAX_EXPOSURE": (0.20, 1.0),
    "MIN_UNIQUES": (0, 4),
    # Sniper signals
    "GPP_BOOM_SPREAD_WEIGHT": (0.0, 0.50),
    "GPP_SNIPER_WEIGHT": (0.0, 0.50),
    "GPP_EFFICIENCY_WEIGHT": (0.0, 0.30),
    "CASH_FLOOR_WEIGHT": (0.0, 1.0),
    "MIN_PLAYER_MINUTES": (0, 30),
    # Ricky weights
    "w_gpp": (0.0, 2.0),                        # DS says 0.0 is optimal
    "w_ceil": (0.0, 2.0),
    "w_own": (0.0, 2.0),                        # DS says +0.15 (positive, not penalty)
}


# ---------------------------------------------------------------------------
# Cross-effect map — which metrics a param change affects
# ---------------------------------------------------------------------------
# Maps param → list of (metric_name, effect_direction)
# effect_direction: "+" means increasing the param helps this metric,
#                   "-" means increasing the param hurts this metric

PARAM_CROSS_EFFECTS: dict[str, list[tuple[str, str]]] = {
    "GPP_PROJ_WEIGHT": [
        ("avg_score", "+"),
        ("bias", "+"),
        ("mae", "-"),
    ],
    "GPP_BOOM_WEIGHT": [
        ("avg_score", "-"),
        ("top_1pct_rate", "+"),
    ],
    "GPP_UPSIDE_WEIGHT": [
        ("avg_score", "+"),
        ("top_1pct_rate", "+"),
    ],
    "GPP_OWN_PENALTY_STRENGTH": [
        ("ownership_sum", "-"),
        ("avg_score", "-"),
    ],
    "GPP_BUST_PENALTY": [
        ("mae", "-"),
        ("avg_score", "-"),
    ],
}


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _steps_over(val: float, hi: float) -> float:
    """Number of 10%-of-hi steps the value exceeds the target max."""
    span = max(abs(hi) * 0.10, 1.0)
    return max((val - hi) / span, 0.0)


def _steps_under(val: float, lo: float) -> float:
    """Number of 10%-of-lo steps the value falls below the target min."""
    span = max(abs(lo) * 0.10, 1.0)
    return max((lo - val) / span, 0.0)


def _severity(batch_value: float, lo: float, hi: float) -> float:
    """How far off-target is the metric? Returns 0.0-2.0 scale factor.

    Used to make nudge steps proportional to deviation size:
    - ~0.1 → small nudge
    - ~0.5 → medium nudge
    - ~1.0 → full step
    - ~2.0 → large step (capped)
    """
    span = hi - lo if hi != lo else 1.0
    if batch_value < lo:
        return min((lo - batch_value) / span, 2.0)
    elif batch_value > hi:
        return min((batch_value - hi) / span, 2.0)
    return 0.0


# ---------------------------------------------------------------------------
# Nudge rule schema
# ---------------------------------------------------------------------------
# Each rule dict has:
#   param       : str   — config.py key (e.g. "GPP_OWN_PENALTY_STRENGTH")
#   label       : str   — human-readable slider label
#   has_slider  : bool  — True if this param has a Sim Lab slider
#   slider_min  : float — lower bound (for clamping the suggestion)
#   slider_max  : float — upper bound
#   step        : float — slider step size (used for display)
#   description : str   — one-line rationale shown in the table
#   compute     : callable(current_val, batch_val, lo, hi) → suggested_val
# ---------------------------------------------------------------------------

NUDGE_PARAM_RULES: dict[tuple[str, str], list[dict[str, Any]]] = {
    # ── Ownership Sum ─────────────────────────────────────────────────────────
    ("ownership_sum", "high"): [
        {
            "param": "GPP_OWN_PENALTY_STRENGTH",
            "label": "Own Penalty Strength",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 3.0,
            "step": 0.1,
            "description": "Increase chalk penalty (DS: chalk outperforms, keep moderate)",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.1 * max(0.3, min(_severity(val, lo, hi), 1.5)), 0.5), 1
            ),
        },
        {
            "param": "GPP_OWN_CAP",
            "label": "Ownership Cap",
            "has_slider": False,
            "slider_min": 2.0,
            "slider_max": 10.0,
            "step": 0.5,
            "description": "Lower max lineup ownership cap",
            "compute": lambda cur, val, lo, hi: round(
                max(cur - 0.5 * max(0.3, min(_severity(val, lo, hi), 1.5)), 2.0), 1
            ),
        },
    ],
    ("ownership_sum", "low"): [
        {
            "param": "GPP_OWN_PENALTY_STRENGTH",
            "label": "Own Penalty Strength",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 3.0,
            "step": 0.1,
            "description": "Reduce chalk penalty",
            "compute": lambda cur, val, lo, hi: round(
                max(cur - 0.2 * max(0.3, min(_severity(val, lo, hi), 1.5)), 0.0), 1
            ),
        },
        {
            "param": "GPP_LOW_OWN_THRESHOLD",
            "label": "Low Own Threshold",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 0.50,
            "step": 0.05,
            "description": "Lower threshold to reduce contrarian exposure",
            "compute": lambda cur, val, lo, hi: round(
                max(cur - 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 0.0), 2
            ),
        },
    ],
    # ── Average Score ─────────────────────────────────────────────────────────
    ("avg_score", "low"): [
        {
            "param": "GPP_PROJ_WEIGHT",
            "label": "Proj Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 1.0,
            "step": 0.05,
            "description": "Increase projection weight",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 1.0), 2
            ),
        },
    ],
    ("avg_score", "high"): [
        {
            "param": "GPP_OWN_PENALTY_STRENGTH",
            "label": "Own Penalty Strength",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 3.0,
            "step": 0.1,
            "description": "Add ownership diversity to reduce chalk-driven scores",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.1 * max(0.3, min(_severity(val, lo, hi), 1.5)), 3.0), 1
            ),
        },
    ],
    # ── Top-1% Hit Rate ────────────────────────────────────────────────────────
    ("top_1pct_rate", "low"): [
        {
            "param": "GPP_BOOM_WEIGHT",
            "label": "Boom Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 1.0,
            "step": 0.05,
            "description": "Raise ceiling/boom weight",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 1.0), 2
            ),
        },
        {
            "param": "GPP_UPSIDE_WEIGHT",
            "label": "Upside Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 1.0,
            "step": 0.05,
            "description": "Raise sim-ceiling (upside) weight",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 1.0), 2
            ),
        },
    ],
    # ── Cash Rate ─────────────────────────────────────────────────────────────
    ("cash_rate", "low"): [
        {
            "param": "CASH_FLOOR_WEIGHT",
            "label": "Cash Floor Weight",
            "has_slider": False,
            "slider_min": 0.0,
            "slider_max": 1.0,
            "step": 0.05,
            "description": "Raise floor weight in cash scoring",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 1.0), 2
            ),
        },
        {
            "param": "GPP_BUST_PENALTY",
            "label": "Bust Penalty",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 0.50,
            "step": 0.05,
            "description": "Penalise high-bust-risk players more",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 0.5), 2
            ),
        },
    ],
    # ── Lineup Diversity ──────────────────────────────────────────────────────
    ("lineup_diversity_min_cores", "low"): [
        {
            "param": "MAX_EXPOSURE",
            "label": "Max Exposure",
            "has_slider": True,
            "slider_min": 0.1,
            "slider_max": 1.0,
            "step": 0.05,
            "description": "Lower max exposure per player",
            "compute": lambda cur, val, lo, hi: round(
                max(cur - 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 0.1), 2
            ),
        },
        {
            "param": "MIN_UNIQUES",
            "label": "Min Uniques",
            "has_slider": True,
            "slider_min": 0,
            "slider_max": 5,
            "step": 1,
            "description": "Require more unique players between lineups",
            "compute": lambda cur, val, lo, hi: int(min(cur + 1, 5)),
        },
    ],
    # ── MAE ───────────────────────────────────────────────────────────────────
    ("mae", "high"): [
        {
            "param": "GPP_BUST_PENALTY",
            "label": "Bust Penalty",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 0.50,
            "step": 0.05,
            "description": "Penalise high-variance busts to reduce projection error",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 0.5), 2
            ),
        },
    ],
    # ── Bias ─────────────────────────────────────────────────────────────────
    ("bias", "high"): [
        {
            "param": "GPP_PROJ_WEIGHT",
            "label": "Proj Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 1.0,
            "step": 0.05,
            "description": "Reduce over-reliance on (over-)projections",
            "compute": lambda cur, val, lo, hi: round(
                max(cur - 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 0.0), 2
            ),
        },
    ],
    ("bias", "low"): [
        {
            "param": "GPP_PROJ_WEIGHT",
            "label": "Proj Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 1.0,
            "step": 0.05,
            "description": "Increase reliance on projections to correct under-projection",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.05 * max(0.3, min(_severity(val, lo, hi), 1.5)), 1.0), 2
            ),
        },
    ],

    # ── Ricky Ranking Weight nudges ──────────────────────────────────────────
    # These target the Ricky weight sliders (w_gpp, w_ceil, w_own), which
    # live in session_state under sim_lab_ricky_weights_{preset}.  The
    # "storage" key tells the Apply handler to write there instead of the
    # sandbox config dict.
    ("ricky_top3_lift", "low"): [
        {
            "param": "w_ceil",
            "label": "Ceiling Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 2.0,
            "step": 0.05,
            "storage": "ricky_weights",
            "description": "Increase ceiling emphasis to pick higher-upside lineups",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.10, 2.0), 2),
        },
        {
            "param": "w_own",
            "label": "Own Penalty",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 2.0,
            "step": 0.05,
            "storage": "ricky_weights",
            "description": "Reduce ownership penalty so ranking favors quality over contrarian",
            "compute": lambda cur, val, lo, hi: round(max(cur - 0.10, 0.0), 2),
        },
    ],
    # ── Ricky Rank Correlation ────────────────────────────────────────────────
    ("ricky_rank_corr", "low"): [
        {
            "param": "w_ceil",
            "label": "Ceiling Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 2.0,
            "step": 0.05,
            "storage": "ricky_weights",
            "description": "Increase ceiling emphasis — strongest within-date signal (r=0.341)",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.10, 2.0), 2),
        },
        {
            "param": "w_own",
            "label": "Own Weight (positive)",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 2.0,
            "step": 0.05,
            "storage": "ricky_weights",
            "description": "Increase ownership weight — chalk outperforms (within-date r=+0.256)",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 2.0), 2),
        },
    ],
    ("ricky_top3_hit", "low"): [
        {
            "param": "w_ceil",
            "label": "Ceiling Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 2.0,
            "step": 0.05,
            "storage": "ricky_weights",
            "description": "Increase ceiling to favor boom lineups in top-5 contention",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.10, 2.0), 2),
        },
        {
            "param": "w_own",
            "label": "Own Weight (positive)",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 2.0,
            "step": 0.05,
            "storage": "ricky_weights",
            "description": "Lean on chalk — higher-owned lineups cash more often (r=+0.256)",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 2.0), 2),
        },
    ],
}


# ---------------------------------------------------------------------------
# Conflict detection helpers
# ---------------------------------------------------------------------------

def _check_cross_effects(
    param: str,
    direction: str,
    off_target_metrics: dict[str, str],
) -> str | None:
    """Check if changing *param* in *direction* would worsen another off-target metric.

    Parameters
    ----------
    param : str
        Config key being nudged.
    direction : str
        ``"up"`` if the suggested value is higher than current, ``"down"`` otherwise.
    off_target_metrics : dict
        Maps metric_name → ``"low"`` | ``"high"`` for currently off-target metrics.

    Returns
    -------
    str or None
        Warning message if a conflict is detected, else ``None``.
    """
    effects = PARAM_CROSS_EFFECTS.get(param, [])
    if not effects:
        return None

    for affected_metric, effect_sign in effects:
        if affected_metric not in off_target_metrics:
            continue

        off_dir = off_target_metrics[affected_metric]

        # "Would this change make the off-target metric worse?"
        # If metric is "low" and the change decreases it (effect "-" when going up,
        # or effect "+" when going down), that's bad.
        # If metric is "high" and the change increases it, that's also bad.
        if direction == "up":
            # Increasing param: "+" effects go up, "-" effects go down
            worsens = (off_dir == "high" and effect_sign == "+") or \
                      (off_dir == "low" and effect_sign == "-")
        else:
            # Decreasing param: "+" effects go down, "-" effects go up
            worsens = (off_dir == "low" and effect_sign == "+") or \
                      (off_dir == "high" and effect_sign == "-")

        if worsens:
            label = affected_metric.replace("_", " ").title()
            return f"May worsen {label}"

    return None


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_nudge_suggestions(
    metric_name: str,
    batch_value: float,
    lo: float,
    hi: float,
    current_overrides: dict[str, Any],
    preset_defaults: dict[str, Any],
    ricky_weights: dict[str, float] | None = None,
    off_target_metrics: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return a list of prescriptive nudge suggestions for an off-target metric.

    Parameters
    ----------
    metric_name:
        Metric key from ``CALIBRATION_TARGETS`` (e.g. ``"ownership_sum"``).
    batch_value:
        The observed batch value for this metric.
    lo, hi:
        Target range boundaries from ``CALIBRATION_TARGETS``.
    current_overrides:
        The current sandbox override dict (from session state).
    preset_defaults:
        The merged config dict for the active preset (from ``merge_config``).
    ricky_weights:
        Current Ricky Ranking weights dict ({"w_gpp": ..., "w_ceil": ..., "w_own": ...}).
        Used to resolve current values for rules with ``storage='ricky_weights'``.
    off_target_metrics:
        Dict mapping metric_name → "low"|"high" for all currently off-target metrics.
        Used for cross-effect conflict detection.

    Returns
    -------
    list of dicts, each with keys:
        param, label, has_slider, description,
        current_value, suggested_value, changed,
        warning (str|None), clamped (bool)
    """
    if batch_value < lo:
        direction = "low"
    elif batch_value > hi:
        direction = "high"
    else:
        return []  # on target — no suggestions

    rules = NUDGE_PARAM_RULES.get((metric_name, direction), [])
    if not rules:
        return []

    # Lazy-load DEFAULT_CONFIG for the final fallback.
    # Import directly from config submodule to avoid yak_core/__init__ chain.
    try:
        import importlib.util as _ilu
        import os as _os
        _cfg_path = _os.path.join(_os.path.dirname(__file__), "..", "yak_core", "config.py")
        _spec = _ilu.spec_from_file_location("_yak_config", _cfg_path)
        _mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
        _DEFAULT_CFG: dict[str, Any] = _mod.DEFAULT_CONFIG
    except Exception:
        # Hard-coded fallback for the most commonly nudged parameters
        _DEFAULT_CFG = {
            "GPP_OWN_PENALTY_STRENGTH": 1.0,
            "GPP_OWN_CAP": 6.0,
            "GPP_LOW_OWN_THRESHOLD": 0.40,
            "GPP_PROJ_WEIGHT": 0.30,
            "GPP_UPSIDE_WEIGHT": 0.30,
            "GPP_BOOM_WEIGHT": 0.35,
            "GPP_BUST_PENALTY": 0.10,
            "MAX_EXPOSURE": 0.60,
            "MIN_UNIQUES": 0,
            "CASH_FLOOR_WEIGHT": 0.6,
        }

    _ricky_w = ricky_weights or {}
    _off_target = off_target_metrics or {}

    results = []
    for rule in rules:
        param = rule["param"]
        storage = rule.get("storage", "sandbox")

        # Resolve current value based on storage target
        if storage == "ricky_weights":
            current_val = _ricky_w.get(param, rule["slider_min"])
        else:
            # overrides → merged preset → DEFAULT_CONFIG → slider_min
            current_val = current_overrides.get(
                param,
                preset_defaults.get(
                    param,
                    _DEFAULT_CFG.get(param, rule["slider_min"]),
                ),
            )
        try:
            suggested_val = rule["compute"](current_val, batch_value, lo, hi)
        except Exception:
            continue

        # Clamp to slider bounds first
        suggested_val = max(rule["slider_min"], min(rule["slider_max"], suggested_val))

        # Apply guardrails (tighter than slider bounds)
        clamped = False
        if param in PARAM_GUARDRAILS:
            g_min, g_max = PARAM_GUARDRAILS[param]
            if suggested_val < g_min:
                suggested_val = g_min
                clamped = True
            elif suggested_val > g_max:
                suggested_val = g_max
                clamped = True
            # For int params (MIN_UNIQUES), keep as int
            if isinstance(rule["slider_min"], int) and isinstance(rule["slider_max"], int):
                suggested_val = int(suggested_val)

        # Cross-effect conflict detection
        warning = None
        if storage != "ricky_weights" and _off_target:
            nudge_dir = "up" if suggested_val > current_val else "down"
            warning = _check_cross_effects(param, nudge_dir, _off_target)

        results.append({
            "param": param,
            "label": rule["label"],
            "has_slider": rule["has_slider"],
            "storage": storage,
            "description": rule["description"],
            "current_value": current_val,
            "suggested_value": suggested_val,
            "changed": suggested_val != current_val,
            "warning": warning,
            "clamped": clamped,
        })

    return results
