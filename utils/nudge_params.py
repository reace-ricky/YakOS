"""Prescriptive nudge parameter mappings for calibration guidance (RIG-8).

Maps ``(metric_name, direction)`` → list of parameter nudge rule dicts.
Each rule specifies the exact ``config.py`` key to adjust, a formula for
computing the suggested value, and the slider bounds used in the Sim Lab
config panel.

Usage::

    from utils.nudge_params import NUDGE_PARAM_RULES, get_nudge_suggestions

    suggestions = get_nudge_suggestions(
        metric_name="ownership_sum",
        batch_value=145.0,
        profile_key="classic_gpp_main",
        current_overrides=sandbox_overrides,
        preset_name="GPP Main",
    )
"""
from __future__ import annotations

from typing import Any


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
            "description": "Increase chalk penalty",
            "compute": lambda cur, val, lo, hi: round(
                min(cur + 0.2 * max(1.0, _steps_over(val, hi)), 3.0), 1
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
                max(cur - 0.5 * max(1.0, _steps_over(val, hi)), 2.0), 1
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
                max(cur - 0.2 * max(1.0, _steps_under(val, lo)), 0.0), 1
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
                max(cur - 0.05, 0.0), 2
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
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 1.0), 2),
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
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.1, 3.0), 1),
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
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 1.0), 2),
        },
        {
            "param": "GPP_UPSIDE_WEIGHT",
            "label": "Upside Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 1.0,
            "step": 0.05,
            "description": "Raise sim-ceiling (upside) weight",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 1.0), 2),
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
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 1.0), 2),
        },
        {
            "param": "GPP_BUST_PENALTY",
            "label": "Bust Penalty",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 0.50,
            "step": 0.05,
            "description": "Penalise high-bust-risk players more",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 0.5), 2),
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
            "compute": lambda cur, val, lo, hi: round(max(cur - 0.05, 0.1), 2),
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
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 0.5), 2),
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
            "compute": lambda cur, val, lo, hi: round(max(cur - 0.05, 0.0), 2),
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
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 1.0), 2),
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
    ("ricky_top3_hit", "low"): [
        {
            "param": "w_gpp",
            "label": "GPP Score Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 2.0,
            "step": 0.05,
            "storage": "ricky_weights",
            "description": "Increase GPP score weight to lean on projection quality",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.10, 2.0), 2),
        },
        {
            "param": "w_ceil",
            "label": "Ceiling Weight",
            "has_slider": True,
            "slider_min": 0.0,
            "slider_max": 2.0,
            "step": 0.05,
            "storage": "ricky_weights",
            "description": "Increase ceiling to favor boom lineups in top-5 contention",
            "compute": lambda cur, val, lo, hi: round(min(cur + 0.05, 2.0), 2),
        },
    ],
}


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

    Returns
    -------
    list of dicts, each with keys:
        param, label, has_slider, description,
        current_value, suggested_value, changed
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

        # Clamp to slider bounds
        suggested_val = max(rule["slider_min"], min(rule["slider_max"], suggested_val))

        results.append({
            "param": param,
            "label": rule["label"],
            "has_slider": rule["has_slider"],
            "storage": storage,
            "description": rule["description"],
            "current_value": current_val,
            "suggested_value": suggested_val,
            "changed": suggested_val != current_val,
        })

    return results
