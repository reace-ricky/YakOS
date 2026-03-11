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
    """Of players whose sim p90 predicted upside, how many actually smashed?

    "Predicted smash" = sim p90 exceeds the player's ceiling.  This is an
    absolute threshold so that ceiling_boost directly controls who enters
    the predicted set — higher boost → more p90s exceed ceiling → the
    prediction set changes, and precision becomes knob-sensitive.

    Falls back to top-quartile ranking when no player's p90 clears their
    ceiling (possible at very low ceiling_boost values).
    """
    if len(verdicts) == 0:
        return 0.0

    ceils = np.array([v["ceil"] for v in verdicts])
    predicted_smash = sim_p90 >= ceils
    n_predicted = int(predicted_smash.sum())

    # Fallback: if no p90 beats any ceiling, use top-quartile ranking so
    # the metric isn't always 0% at conservative settings.
    if n_predicted == 0:
        threshold = np.percentile(sim_p90, 75)
        predicted_smash = sim_p90 >= threshold
        n_predicted = int(predicted_smash.sum())
    if n_predicted == 0:
        return 0.0

    actually_smashed = np.array([v["verdict"] == "SMASHED" for v in verdicts])
    correct = int((predicted_smash & actually_smashed).sum())
    return float(correct / n_predicted)


def _bust_precision(verdicts: List[Dict[str, Any]], sim_p10: np.ndarray, actual: np.ndarray) -> float:
    """Of players whose sim p10 predicted downside, how many actually busted?

    "Predicted bust" = sim p10 falls below the player's floor.  Same
    absolute-threshold logic as _smash_precision so floor_dampen controls
    who enters the prediction set.
    """
    if len(verdicts) == 0:
        return 0.0

    floors = np.array([v["floor"] for v in verdicts])
    predicted_bust = sim_p10 <= floors
    n_predicted = int(predicted_bust.sum())

    # Fallback: bottom-quartile ranking if no p10 breaches any floor.
    if n_predicted == 0:
        threshold = np.percentile(sim_p10, 25)
        predicted_bust = sim_p10 <= threshold
        n_predicted = int(predicted_bust.sum())
    if n_predicted == 0:
        return 0.0

    actually_busted = np.array([v["verdict"] == "BUSTED" for v in verdicts])
    correct = int((predicted_bust & actually_busted).sum())
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
    sport: Optional[str] = None,
    n_sims: int = 1000,
) -> Dict[str, Any]:
    """Run sim sandbox across all archived slates.

    Parameters
    ----------
    sport : str, optional
        "NBA" or "PGA". Filters archived slates by filename prefix.
        PGA files start with ``pga_``, NBA files do not.
        None = include all slates.

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

    # Sport filter: PGA files are prefixed "pga_", NBA files are not
    if sport:
        _sp = sport.upper()
        if _sp == "PGA":
            parquets = [f for f in parquets if f.startswith("pga_")]
        elif _sp == "NBA":
            parquets = [f for f in parquets if not f.startswith("pga_")]

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

    # Deduplicate smashes/busts by player name across slates.
    # A player can appear in multiple slates (e.g. GPP + Cash on the same day)
    # with slightly different projections. Keep the entry with the most extreme diff.
    def _dedup_by_player(entries: List[Dict[str, Any]], keep_max: bool) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            name = e["player"]
            if name not in best:
                best[name] = e
            elif keep_max and e["diff"] > best[name]["diff"]:
                best[name] = e
            elif not keep_max and e["diff"] < best[name]["diff"]:
                best[name] = e
        return list(best.values())

    all_smashes = _dedup_by_player(all_smashes, keep_max=True)
    all_busts = _dedup_by_player(all_busts, keep_max=False)

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

    Knob triggers (independent — each fires on its own):

    ceiling_boost:
      - smash_prec < 15%  → big bump (+0.12)
      - smash_prec < 25%  → nudge  (+0.06)
      - smash_prec > 35%  → trim   (-0.05)  (over-predicting upside)
      - upside_acc < 50%  → bump   (+0.10)
      - upside_acc < 65%  → nudge  (+0.05)

    floor_dampen:
      - bust_prec < 20%   → bump   (+0.12)
      - bust_prec < 35%   → nudge  (+0.06)
      - bust_prec > 50%   → trim   (-0.05)
      - downside_acc < 50% → bump  (+0.10)
      - downside_acc < 65% → nudge (+0.05)

    Multiple reasons can fire. The largest applicable bump wins per knob.
    """
    current = agg["knobs"]
    cb = current.get("ceiling_boost", 1.0)
    fd = current.get("floor_dampen", 1.0)

    upside_acc = agg.get("avg_upside_accuracy", 0)
    downside_acc = agg.get("avg_downside_accuracy", 0)
    coverage = agg.get("avg_coverage", 0)
    smash_prec = agg.get("avg_smash_precision", 0)
    bust_prec = agg.get("avg_bust_precision", 0)

    cb_delta = 0.0
    fd_delta = 0.0
    reasons = []

    # ── Ceiling boost: smash precision is the primary signal ──────────
    if smash_prec < 0.15:
        _d = 0.12
        cb_delta = max(cb_delta, _d)
        reasons.append(f"Smash precision {smash_prec:.0%} is very low — bump ceiling_boost +{_d}")
    elif smash_prec < 0.25:
        _d = 0.06
        cb_delta = max(cb_delta, _d)
        reasons.append(f"Smash precision {smash_prec:.0%} below target 25% — nudge ceiling_boost +{_d}")
    elif smash_prec > 0.35:
        _d = -0.05
        cb_delta = min(cb_delta, _d)
        reasons.append(f"Smash precision {smash_prec:.0%} is high — trim ceiling_boost {_d}")

    # Upside accuracy as secondary signal
    if upside_acc < 0.50:
        _d = 0.10
        if _d > cb_delta:
            cb_delta = _d
        reasons.append(f"Upside accuracy {upside_acc:.0%} is low — ceiling_boost +{_d}")
    elif upside_acc < 0.65:
        _d = 0.05
        if _d > cb_delta:
            cb_delta = _d
        reasons.append(f"Upside accuracy {upside_acc:.0%} could improve — ceiling_boost +{_d}")

    # ── Floor dampen: bust precision is the primary signal ────────────
    if bust_prec < 0.20:
        _d = 0.12
        fd_delta = max(fd_delta, _d)
        reasons.append(f"Bust precision {bust_prec:.0%} is very low — bump floor_dampen +{_d}")
    elif bust_prec < 0.35:
        _d = 0.06
        fd_delta = max(fd_delta, _d)
        reasons.append(f"Bust precision {bust_prec:.0%} below target 35% — nudge floor_dampen +{_d}")
    elif bust_prec > 0.50:
        _d = -0.05
        fd_delta = min(fd_delta, _d)
        reasons.append(f"Bust precision {bust_prec:.0%} is high — trim floor_dampen {_d}")

    # Downside accuracy as secondary signal
    if downside_acc < 0.50:
        _d = 0.10
        if _d > fd_delta:
            fd_delta = _d
        reasons.append(f"Downside accuracy {downside_acc:.0%} is low — floor_dampen +{_d}")
    elif downside_acc < 0.65:
        _d = 0.05
        if _d > fd_delta:
            fd_delta = _d
        reasons.append(f"Downside accuracy {downside_acc:.0%} could improve — floor_dampen +{_d}")

    # ── Coverage check (advisory only, no knob change) ───────────────
    if coverage > 0.90:
        reasons.append(f"Coverage {coverage:.0%} is very high — sims may be too conservative")
    elif coverage < 0.60:
        reasons.append(f"Coverage {coverage:.0%} is low — sims may be too aggressive")

    # Apply deltas
    new_cb = round(cb + cb_delta, 3)
    new_fd = round(fd + fd_delta, 3)

    # Guard rails
    new_cb = round(max(0.5, min(2.0, new_cb)), 3)
    new_fd = round(max(0.5, min(2.0, new_fd)), 3)

    # changed flag follows the deltas, not new-vs-old comparison.
    # The old check (abs(new - old) > 0.001) would say "no changes"
    # when knobs had already been applied but KPIs hadn't improved.
    # Also respect guard rails: if we hit the cap, no point recommending.
    changed = (abs(new_cb - cb) > 0.001) or (abs(new_fd - fd) > 0.001)
    if not changed and (abs(cb_delta) > 0.001 or abs(fd_delta) > 0.001):
        # Deltas fired but the knobs didn't move (already at cap or
        # already at the recommended value). Tell the user.
        if cb >= 2.0 or fd >= 2.0:
            reasons.append("Knobs are at their maximum — further tuning won't help. "
                           "The issue is upstream (projection quality, not sim width).")
        else:
            # Deltas fired and knobs aren't at cap, so apply them.
            changed = True

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
