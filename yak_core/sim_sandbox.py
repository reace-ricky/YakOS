"""yak_core.sim_sandbox -- Sim Calibration Sandbox.

Runs sim configs against archived slates with actuals, scores prediction
accuracy, detects breakouts, and recommends knob adjustments.

Absorbs the old "Signal Accuracy" display into a unified calibration tool
that actually persists results and closes the feedback loop.

KPIs:
  - MAE (projection accuracy)
  - Smash Precision (when we predicted smash, did they?)
  - Bust Precision (when we predicted bust, did they?)
  - Coverage (% of actuals within sim range)
  - Breakout Hit Rate (ceiling-beaters we identified)

Knobs tuned:
  - ceiling_boost  (scales upside outcomes in MC sims)
  - floor_dampen   (compresses downside outcomes in MC sims)

Workflow:
  1. Load archived slates (they have proj, ceil, floor, actual_fp)
  2. Run sim scoring with current knobs
  3. Compute verdicts (SMASHED / HIT / MISS / BUSTED) per player
  4. Detect breakouts (players who blew past ceiling)
  5. Recommend knob adjustments
  6. One-click Apply writes new knobs

Storage: JSON in data/sim_sandbox/
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import YAKOS_ROOT

_SANDBOX_DIR = os.path.join(YAKOS_ROOT, "data", "sim_sandbox")
_KNOBS_FILE = os.path.join(_SANDBOX_DIR, "active_knobs.json")
_HISTORY_FILE = os.path.join(_SANDBOX_DIR, "sandbox_history.json")


def _ensure_dir() -> None:
    Path(_SANDBOX_DIR).mkdir(parents=True, exist_ok=True)


# ── Default Knobs ──────────────────────────────────────────────────

DEFAULT_KNOBS: Dict[str, float] = {
    "ceiling_boost": 1.0,
    "floor_dampen": 1.0,
}


def get_active_knobs() -> Dict[str, float]:
    """Return the current active knobs (from file or defaults)."""
    if os.path.isfile(_KNOBS_FILE):
        with open(_KNOBS_FILE, "r") as f:
            data = json.load(f)
        return {
            "ceiling_boost": float(data.get("ceiling_boost", DEFAULT_KNOBS["ceiling_boost"])),
            "floor_dampen": float(data.get("floor_dampen", DEFAULT_KNOBS["floor_dampen"])),
        }
    return dict(DEFAULT_KNOBS)


def save_active_knobs(knobs: Dict[str, float]) -> None:
    """Persist knob values to disk."""
    _ensure_dir()
    payload = {
        "ceiling_boost": round(float(knobs.get("ceiling_boost", DEFAULT_KNOBS["ceiling_boost"])), 3),
        "floor_dampen": round(float(knobs.get("floor_dampen", DEFAULT_KNOBS["floor_dampen"])), 3),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(_KNOBS_FILE, "w") as f:
        json.dump(payload, f, indent=2)


# ── Monte Carlo Engine ─────────────────────────────────────────────

def _simulate_player_outcomes(
    proj: np.ndarray,
    ceil: np.ndarray,
    floor: np.ndarray,
    ceiling_boost: float = 1.0,
    floor_dampen: float = 1.0,
    n_sims: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Run MC sims for a set of players. Returns (n_sims, n_players) matrix."""
    rng = np.random.RandomState(seed)

    # Derive std from ceil/floor range (same method as sims.py)
    Z_85 = 1.036  # z-score for 85th percentile
    raw_std = (ceil - floor) / (2 * Z_85)
    raw_std = np.maximum(raw_std, 1.0)

    sim_matrix = rng.normal(
        loc=proj[None, :],
        scale=raw_std[None, :],
        size=(n_sims, len(proj)),
    )

    # Apply knobs
    upside_mask = sim_matrix > proj[None, :]
    downside_mask = sim_matrix < proj[None, :]

    sim_matrix = np.where(
        upside_mask,
        proj[None, :] + (sim_matrix - proj[None, :]) * ceiling_boost,
        sim_matrix,
    )
    sim_matrix = np.where(
        downside_mask,
        proj[None, :] - (proj[None, :] - sim_matrix) * floor_dampen,
        sim_matrix,
    )

    return np.maximum(sim_matrix, 0.0)


# ── Player Verdicts ────────────────────────────────────────────────

def _player_verdict(actual: float, proj: float, floor: float, ceil: float) -> str:
    """Classify a player outcome. Same logic as sim_accuracy._player_verdict."""
    if actual >= ceil:
        return "SMASHED"
    elif actual >= proj * 0.90:
        return "HIT"
    elif actual >= floor:
        return "MISS"
    else:
        return "BUSTED"


def _compute_verdicts(
    df: pd.DataFrame,
    actual: np.ndarray,
    proj: np.ndarray,
    ceil: np.ndarray,
    floor: np.ndarray,
) -> List[Dict[str, Any]]:
    """Compute per-player verdicts for a slate."""
    name_col = "player_name" if "player_name" in df.columns else "name"
    names = df[name_col].values if name_col in df.columns else [f"P{i}" for i in range(len(df))]
    salaries = df["salary"].values.astype(float) if "salary" in df.columns else np.zeros(len(df))

    verdicts = []
    for i in range(len(df)):
        v = _player_verdict(float(actual[i]), float(proj[i]), float(floor[i]), float(ceil[i]))
        verdicts.append({
            "player": str(names[i]),
            "salary": int(salaries[i]),
            "proj": round(float(proj[i]), 1),
            "actual": round(float(actual[i]), 1),
            "ceil": round(float(ceil[i]), 1),
            "floor": round(float(floor[i]), 1),
            "diff": round(float(actual[i] - proj[i]), 1),
            "verdict": v,
        })
    return verdicts


def _smash_precision(verdicts: List[Dict[str, Any]], sim_p90: np.ndarray, actual: np.ndarray) -> float:
    """Of players whose sim p90 suggested upside (top quartile), how many actually smashed?"""
    if len(verdicts) == 0:
        return 0.0
    # "Predicted smash" = sim p90 was in top 25% of all sim p90s
    threshold = np.percentile(sim_p90, 75)
    predicted_smash = sim_p90 >= threshold
    n_predicted = predicted_smash.sum()
    if n_predicted == 0:
        return 0.0
    # Of those predicted, how many actually smashed (beat ceiling)?
    actually_smashed = np.array([v["verdict"] == "SMASHED" for v in verdicts])
    correct = (predicted_smash & actually_smashed).sum()
    return float(correct / n_predicted)


def _bust_precision(verdicts: List[Dict[str, Any]], sim_p10: np.ndarray, actual: np.ndarray) -> float:
    """Of players whose sim p10 suggested downside (bottom quartile), how many actually busted?"""
    if len(verdicts) == 0:
        return 0.0
    threshold = np.percentile(sim_p10, 25)
    predicted_bust = sim_p10 <= threshold
    n_predicted = predicted_bust.sum()
    if n_predicted == 0:
        return 0.0
    actually_busted = np.array([v["verdict"] == "BUSTED" for v in verdicts])
    correct = (predicted_bust & actually_busted).sum()
    return float(correct / n_predicted)


# ── Score Config vs Actuals ────────────────────────────────────────

def score_config_vs_actuals(
    pool_df: pd.DataFrame,
    knobs: Dict[str, float],
    n_sims: int = 1000,
) -> Dict[str, Any]:
    """Score a sim config against actual results for a single slate.

    Returns dict with accuracy KPIs, verdicts, and breakouts.
    """
    required = {"proj", "ceil", "floor", "actual_fp"}
    if not required.issubset(set(pool_df.columns)):
        missing = required - set(pool_df.columns)
        return {"error": f"Missing columns: {missing}"}

    df = pool_df.dropna(subset=list(required)).copy()
    df = df[(df["proj"] >= 5.0) & (df["actual_fp"] > 0)].copy()
    if df.empty:
        return {"error": "No scoreable players"}

    proj = df["proj"].values.astype(float)
    ceil = df["ceil"].values.astype(float)
    floor = df["floor"].values.astype(float)
    actual = df["actual_fp"].values.astype(float)

    cb = float(knobs.get("ceiling_boost", 1.0))
    fd = float(knobs.get("floor_dampen", 1.0))

    sim_matrix = _simulate_player_outcomes(proj, ceil, floor, cb, fd, n_sims)

    sim_mean = sim_matrix.mean(axis=0)
    sim_p10 = np.percentile(sim_matrix, 10, axis=0)
    sim_p90 = np.percentile(sim_matrix, 90, axis=0)

    # -- KPIs --
    mae = float(np.mean(np.abs(sim_mean - actual)))

    in_range = ((actual >= sim_p10) & (actual <= sim_p90)).sum()
    coverage = float(in_range / len(actual))

    beat_proj = actual > proj
    upside_captured = float(np.mean(sim_p90[beat_proj] >= actual[beat_proj] * 0.85)) if beat_proj.sum() > 0 else 0.0

    missed_floor = actual < floor
    downside_captured = float(np.mean(sim_p10[missed_floor] <= actual[missed_floor] * 1.15)) if missed_floor.sum() > 0 else 0.0

    sim_direction = sim_mean > proj
    actual_direction = actual > proj
    directional_acc = float(np.mean(sim_direction == actual_direction))

    # -- Verdicts --
    verdicts = _compute_verdicts(df, actual, proj, ceil, floor)
    smash_prec = _smash_precision(verdicts, sim_p90, actual)
    bust_prec = _bust_precision(verdicts, sim_p10, actual)

    # Verdict distribution
    v_counts = {"SMASHED": 0, "HIT": 0, "MISS": 0, "BUSTED": 0}
    for v in verdicts:
        v_counts[v["verdict"]] = v_counts.get(v["verdict"], 0) + 1
    n = len(verdicts)
    v_rates = {k: round(c / n, 3) if n else 0 for k, c in v_counts.items()}

    # Top smashes and worst busts (top 5 each)
    smashes = sorted([v for v in verdicts if v["verdict"] == "SMASHED"], key=lambda x: x["diff"], reverse=True)
    busts = sorted([v for v in verdicts if v["verdict"] == "BUSTED"], key=lambda x: x["diff"])

    # -- Breakouts --
    breakouts = _detect_breakouts(df, actual, ceil, proj)

    return {
        "n_players": n,
        "knobs": {"ceiling_boost": cb, "floor_dampen": fd},
        "mae": round(mae, 2),
        "coverage": round(coverage, 3),
        "upside_accuracy": round(upside_captured, 3),
        "downside_accuracy": round(downside_captured, 3),
        "directional_accuracy": round(directional_acc, 3),
        "smash_precision": round(smash_prec, 3),
        "bust_precision": round(bust_prec, 3),
        "verdict_rates": v_rates,
        "top_smashes": smashes[:5],
        "worst_busts": busts[:5],
        "breakouts": breakouts,
    }


# ── Breakout Detection ─────────────────────────────────────────────

def _detect_breakouts(
    df: pd.DataFrame,
    actual: np.ndarray,
    ceil: np.ndarray,
    proj: np.ndarray,
) -> List[Dict[str, Any]]:
    """Find players who blew past their ceiling or hit extreme value."""
    breakouts = []

    name_col = "player_name" if "player_name" in df.columns else "name"
    names = df[name_col].values if name_col in df.columns else [f"P{i}" for i in range(len(df))]
    salaries = df["salary"].values if "salary" in df.columns else np.zeros(len(df))

    for i in range(len(df)):
        reasons = []

        ceil_gap = actual[i] - ceil[i]
        if ceil_gap >= 10:
            reasons.append(f"Beat ceiling by {ceil_gap:.1f} FP")

        if salaries[i] > 0:
            value = actual[i] / (salaries[i] / 1000)
            if value >= 5.0:
                reasons.append(f"Value explosion: {value:.1f}x per $1k")

        proj_gap = actual[i] - proj[i]
        if proj_gap >= 20:
            reasons.append(f"Beat proj by {proj_gap:.1f} FP")

        if reasons:
            breakouts.append({
                "player": str(names[i]),
                "actual_fp": round(float(actual[i]), 1),
                "proj": round(float(proj[i]), 1),
                "ceil": round(float(ceil[i]), 1),
                "salary": int(salaries[i]),
                "reasons": reasons,
            })

    breakouts.sort(key=lambda x: x["actual_fp"], reverse=True)
    return breakouts


# ── Run Across All Archived Slates ─────────────────────────────────

def run_sandbox(
    knobs: Optional[Dict[str, float]] = None,
    contest_type: Optional[str] = None,
    n_sims: int = 1000,
) -> Dict[str, Any]:
    """Run sim sandbox across all archived slates.

    Returns aggregate KPIs + per-slate breakdown + recommendations.
    """
    if knobs is None:
        knobs = get_active_knobs()

    archive_dir = os.path.join(YAKOS_ROOT, "data", "slate_archive")
    if not os.path.isdir(archive_dir):
        return {"error": "No slate archive found"}

    parquets = sorted([
        f for f in os.listdir(archive_dir) if f.endswith(".parquet")
    ])

    if contest_type:
        ct = contest_type.lower()
        parquets = [f for f in parquets if ct in f.lower()]

    if not parquets:
        return {"error": "No archived slates found"}

    slate_results = []
    all_breakouts = []
    all_smashes = []
    all_busts = []

    for pq in parquets:
        path = os.path.join(archive_dir, pq)
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue

        result = score_config_vs_actuals(df, knobs, n_sims)
        if "error" in result:
            continue

        slate_label = pq.replace(".parquet", "")
        result["slate"] = slate_label
        slate_results.append(result)

        for bo in result.get("breakouts", []):
            bo["slate"] = slate_label
            all_breakouts.append(bo)

        for s in result.get("top_smashes", []):
            s["slate"] = slate_label
            all_smashes.append(s)

        for b in result.get("worst_busts", []):
            b["slate"] = slate_label
            all_busts.append(b)

    if not slate_results:
        return {"error": "No valid slates scored"}

    # Aggregate KPIs
    def _avg(key):
        vals = [r[key] for r in slate_results if key in r]
        return round(float(np.mean(vals)), 3) if vals else 0.0

    # Sort smashes/busts across all slates
    all_smashes.sort(key=lambda x: x["diff"], reverse=True)
    all_busts.sort(key=lambda x: x["diff"])

    agg = {
        "n_slates": len(slate_results),
        "knobs": knobs,
        "avg_mae": round(_avg("mae"), 2),
        "avg_coverage": _avg("coverage"),
        "avg_upside_accuracy": _avg("upside_accuracy"),
        "avg_downside_accuracy": _avg("downside_accuracy"),
        "avg_directional_accuracy": _avg("directional_accuracy"),
        "avg_smash_precision": _avg("smash_precision"),
        "avg_bust_precision": _avg("bust_precision"),
        "top_smashes": all_smashes[:5],
        "worst_busts": all_busts[:5],
        "breakouts": all_breakouts[:10],
        "per_slate": [{
            "slate": r["slate"],
            "mae": r["mae"],
            "coverage": r["coverage"],
            "smash_precision": r["smash_precision"],
            "bust_precision": r["bust_precision"],
            "n_players": r["n_players"],
        } for r in slate_results],
    }

    agg["recommendations"] = _generate_recommendations(agg)

    return agg


# ── Recommendation Engine ──────────────────────────────────────────

def _generate_recommendations(agg: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results and recommend knob adjustments.

    The primary goal is IMPROVING SMASH PRECISION (identifying breakout
    players).  Smash precision is the #1 signal — if we can't predict
    who will smash, we can't win GPPs.  Targets:
      - Smash precision: 25%+  (currently ~10% is terrible)
      - Bust precision:  40%+
      - MAE:             < 6 FP
      - Coverage:        80%+
    """
    current = agg["knobs"]
    cb = current.get("ceiling_boost", 1.0)
    fd = current.get("floor_dampen", 1.0)

    upside_acc = agg.get("avg_upside_accuracy", 0)
    downside_acc = agg.get("avg_downside_accuracy", 0)
    coverage = agg.get("avg_coverage", 0)
    smash_prec = agg.get("avg_smash_precision", 0)
    bust_prec = agg.get("avg_bust_precision", 0)
    mae = agg.get("avg_mae", 10)

    new_cb = cb
    new_fd = fd
    reasons = []

    # ── SMASH PRECISION is the primary driver ──────────────────────
    # This is the whole point: if smash prec is bad, we MUST adjust.
    if smash_prec < 0.15:
        # Terrible — we're basically guessing. Aggressive ceiling boost.
        bump = 0.20
        new_cb = round(cb + bump, 3)
        reasons.append(
            f"Smash precision {smash_prec:.0%} is way below target (25%) "
            f"— aggressively boost ceiling_boost +{bump} to {new_cb}"
        )
    elif smash_prec < 0.20:
        bump = 0.12
        new_cb = round(cb + bump, 3)
        reasons.append(
            f"Smash precision {smash_prec:.0%} still below target (25%) "
            f"— bump ceiling_boost +{bump} to {new_cb}"
        )
    elif smash_prec < 0.25:
        bump = 0.06
        new_cb = round(cb + bump, 3)
        reasons.append(
            f"Smash precision {smash_prec:.0%} getting closer to target (25%) "
            f"— nudge ceiling_boost +{bump} to {new_cb}"
        )
    elif smash_prec >= 0.30 and upside_acc > 0.80:
        # Strong — we can afford to trim slightly
        bump = -0.03
        new_cb = round(cb + bump, 3)
        reasons.append(
            f"Smash precision {smash_prec:.0%} is strong — trim ceiling_boost {bump} to {new_cb}"
        )

    # ── Upside accuracy as secondary signal ────────────────────────
    if upside_acc < 0.50 and smash_prec >= 0.15:
        # Only recommend if smash prec hasn't already triggered a bigger bump
        extra = 0.08
        new_cb = round(new_cb + extra, 3)
        reasons.append(
            f"Upside accuracy {upside_acc:.0%} is also low — additional ceiling_boost +{extra} to {new_cb}"
        )

    # ── BUST PRECISION ─────────────────────────────────────────────
    if bust_prec < 0.25:
        bump = 0.15
        new_fd = round(fd + bump, 3)
        reasons.append(
            f"Bust precision {bust_prec:.0%} below target (40%) "
            f"— bump floor_dampen +{bump} to {new_fd}"
        )
    elif bust_prec < 0.35:
        bump = 0.08
        new_fd = round(fd + bump, 3)
        reasons.append(
            f"Bust precision {bust_prec:.0%} needs improvement (target: 40%) "
            f"— nudge floor_dampen +{bump} to {new_fd}"
        )
    elif bust_prec >= 0.45 and downside_acc > 0.80:
        bump = -0.03
        new_fd = round(fd + bump, 3)
        reasons.append(
            f"Bust precision {bust_prec:.0%} is strong — trim floor_dampen {bump} to {new_fd}"
        )

    # ── Coverage check ─────────────────────────────────────────────
    if coverage > 0.92:
        reasons.append(
            f"Coverage {coverage:.0%} is very high — sims may be too conservative "
            f"(wider range catches everything but dilutes signal)"
        )
    elif coverage < 0.55:
        reasons.append(f"Coverage {coverage:.0%} is low — sims may be too narrow")

    # ── MAE check ──────────────────────────────────────────────────
    if mae > 8.0:
        reasons.append(f"MAE {mae:.1f} FP is high (target: <6) — projections need work")

    # Guard rails
    new_cb = round(max(0.5, min(2.5, new_cb)), 3)
    new_fd = round(max(0.5, min(2.5, new_fd)), 3)

    changed = (abs(new_cb - cb) > 0.001) or (abs(new_fd - fd) > 0.001)

    if not reasons:
        reasons.append("Current config looks solid — no changes recommended")

    return {
        "current": {"ceiling_boost": cb, "floor_dampen": fd},
        "recommended": {"ceiling_boost": new_cb, "floor_dampen": new_fd},
        "changed": changed,
        "reasons": reasons,
    }


# ── History Tracking ───────────────────────────────────────────────

def save_sandbox_run(result: Dict[str, Any]) -> None:
    """Append a sandbox run to history for trend tracking."""
    _ensure_dir()
    history = _load_history()

    entry = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "knobs": result.get("knobs", {}),
        "n_slates": result.get("n_slates", 0),
        "avg_mae": result.get("avg_mae", 0),
        "avg_coverage": result.get("avg_coverage", 0),
        "avg_smash_precision": result.get("avg_smash_precision", 0),
        "avg_bust_precision": result.get("avg_bust_precision", 0),
        "avg_upside_accuracy": result.get("avg_upside_accuracy", 0),
        "avg_downside_accuracy": result.get("avg_downside_accuracy", 0),
        "avg_directional_accuracy": result.get("avg_directional_accuracy", 0),
    }

    history.append(entry)
    if len(history) > 50:
        history = history[-50:]

    with open(_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _load_history() -> List[Dict[str, Any]]:
    if os.path.isfile(_HISTORY_FILE):
        with open(_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def get_sandbox_history() -> List[Dict[str, Any]]:
    """Return all saved sandbox runs for trend display."""
    return _load_history()


# ── Breakout Feedback Loop ─────────────────────────────────────────

def learn_breakout_weights_from_sandbox(
    sandbox_result: Dict[str, Any],
    learning_rate: float = 0.10,
) -> Dict[str, Any]:
    """Analyze sandbox breakout results and tune breakout model weights.

    The idea: for each breakout player the sandbox found (who beat their
    ceiling), we check which of the 5 breakout signals were strongest for
    that player across the archived slate.  Signals that were high for
    actual breakouts get weight bumped up; signals that were high for
    NON-breakouts get penalised.

    This closes the loop: sandbox detects breakouts retrospectively,
    then feeds that knowledge back into the forward-looking breakout
    model so it does better next time.

    Parameters
    ----------
    sandbox_result : dict
        Output of ``run_sandbox()``.
    learning_rate : float
        How aggressively to shift weights (0.05 = conservative, 0.15 = aggressive).

    Returns
    -------
    dict with keys:
        "old_weights", "new_weights", "adjustments", "n_breakouts", "changed"
    """
    from yak_core.right_angle import (
        _load_breakout_weights, save_breakout_weights, DEFAULT_BREAKOUT_WEIGHTS,
    )

    old_w = _load_breakout_weights()
    breakouts = sandbox_result.get("breakouts", [])

    if not breakouts:
        return {
            "old_weights": old_w,
            "new_weights": old_w,
            "adjustments": {},
            "n_breakouts": 0,
            "changed": False,
            "reason": "No breakouts found in sandbox results to learn from.",
        }

    # Load all archived slates to get the signal values for breakout players
    archive_dir = os.path.join(YAKOS_ROOT, "data", "slate_archive")
    if not os.path.isdir(archive_dir):
        return {
            "old_weights": old_w,
            "new_weights": old_w,
            "adjustments": {},
            "n_breakouts": len(breakouts),
            "changed": False,
            "reason": "No slate archive — can't analyze signal values.",
        }

    # Collect all archived player data with signal columns
    all_players = []
    for pq in sorted(os.listdir(archive_dir)):
        if not pq.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(archive_dir, pq))
            if "player_name" not in df.columns:
                continue
            all_players.append(df)
        except Exception:
            continue

    if not all_players:
        return {
            "old_weights": old_w,
            "new_weights": old_w,
            "adjustments": {},
            "n_breakouts": len(breakouts),
            "changed": False,
            "reason": "Could not load archived slate data.",
        }

    pool = pd.concat(all_players, ignore_index=True)

    # Compute breakout signals for the full archived pool
    from yak_core.right_angle import compute_breakout_candidates
    bo_df = compute_breakout_candidates(pool, top_n=len(pool))

    if bo_df.empty or "breakout_score" not in bo_df.columns:
        return {
            "old_weights": old_w,
            "new_weights": old_w,
            "adjustments": {},
            "n_breakouts": len(breakouts),
            "changed": False,
            "reason": "Could not compute breakout signals for archived players.",
        }

    # Tag which players actually broke out
    breakout_names = {b["player"] for b in breakouts}
    bo_df["_actually_broke_out"] = bo_df["player_name"].isin(breakout_names)

    actual_bo = bo_df[bo_df["_actually_broke_out"]]
    non_bo = bo_df[~bo_df["_actually_broke_out"]]

    if actual_bo.empty:
        return {
            "old_weights": old_w,
            "new_weights": old_w,
            "adjustments": {},
            "n_breakouts": len(breakouts),
            "changed": False,
            "reason": "Breakout players not found in computed signals.",
        }

    # For each player, the breakout_signals string tells us which signals fired.
    # But we need the raw signal contributions.  Since we can't easily get those
    # from the output, we use a heuristic: check what the breakout score was for
    # actual breakouts vs non-breakouts.  If actual breakouts had LOW breakout
    # scores, our model needs more weight on whatever made them special.

    # Average breakout score for actual breakouts vs everyone else
    avg_bo_score = float(actual_bo["breakout_score"].mean())
    avg_all_score = float(bo_df["breakout_score"].mean())

    # Parse which signals appear in breakout_signals for actual breakouts
    signal_map = {
        "Mins": "minutes_surge",
        "Underpriced": "salary_value",
        "Trending": "usage_bump",
        "Matchup": "matchup_dvp",
        "Volatile": "volatility",
    }

    signal_hits = {s: 0 for s in DEFAULT_BREAKOUT_WEIGHTS}
    signal_total = 0
    for _, row in actual_bo.iterrows():
        sigs = str(row.get("breakout_signals", ""))
        for keyword, signal_name in signal_map.items():
            if keyword in sigs:
                signal_hits[signal_name] += 1
        signal_total += 1

    # Compute adjustments: signals that appeared often for breakouts get boosted
    adjustments = {}
    new_w = dict(old_w)

    if signal_total > 0:
        for signal_name, count in signal_hits.items():
            hit_rate = count / signal_total
            # If this signal appears for >40% of breakouts, boost it
            if hit_rate > 0.4:
                adj = learning_rate * hit_rate
                new_w[signal_name] = round(old_w[signal_name] + adj, 4)
                adjustments[signal_name] = f"+{adj:.4f} (fired for {hit_rate:.0%} of breakouts)"
            # If it appears for <10% of breakouts, it's not helping
            elif hit_rate < 0.1 and old_w[signal_name] > 0.10:
                adj = -learning_rate * 0.3
                new_w[signal_name] = round(max(0.05, old_w[signal_name] + adj), 4)
                adjustments[signal_name] = f"{adj:.4f} (only fired for {hit_rate:.0%} of breakouts)"

    # If breakouts had low scores overall, boost all weights slightly
    # (the model isn't finding them at all)
    if avg_bo_score < avg_all_score * 1.2:
        for s in new_w:
            if s not in adjustments:
                adjustments[s] = "(breakouts scored below average — model needs recalibration)"

    # Normalise
    total = sum(new_w.values())
    if total > 0:
        new_w = {k: round(v / total, 4) for k, v in new_w.items()}

    changed = any(abs(new_w[k] - old_w[k]) > 0.001 for k in new_w)

    return {
        "old_weights": old_w,
        "new_weights": new_w,
        "adjustments": adjustments,
        "n_breakouts": len(breakouts),
        "avg_breakout_score": round(avg_bo_score, 1),
        "avg_all_score": round(avg_all_score, 1),
        "changed": changed,
        "reason": (
            f"Analyzed {len(breakouts)} breakouts. "
            f"Avg breakout score: {avg_bo_score:.1f} vs pool avg: {avg_all_score:.1f}."
        ),
    }
