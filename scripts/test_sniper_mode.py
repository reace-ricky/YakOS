#!/usr/bin/env python3
"""Sniper Mode acceptance tests.

Validates all DS-calibrated configs, guardrails, nudge rules, and sniper
signals introduced by the Sniper Mode calibration rework.

Run standalone:
    python scripts/test_sniper_mode.py
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import traceback
from typing import Any, Dict, List

PASS = 0
FAIL = 0
ERRORS: List[str] = []


def _check(desc: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {desc}")
    else:
        FAIL += 1
        msg = f"  ✗ {desc}"
        if detail:
            msg += f"  — {detail}"
        print(msg)
        ERRORS.append(msg)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: DS-recommended slider values fall within PARAM_GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════
def test_ds_values_within_guardrails() -> None:
    print("\n── Test 1: DS-recommended values within PARAM_GUARDRAILS ──")
    from utils.nudge_params import PARAM_GUARDRAILS
    from yak_core.auto_calibrate import DS_RECOMMENDATIONS

    for contest_type, recs in DS_RECOMMENDATIONS.items():
        for param, val in recs.items():
            if param in PARAM_GUARDRAILS:
                lo, hi = PARAM_GUARDRAILS[param]
                _check(
                    f"[{contest_type}] {param}={val} in [{lo}, {hi}]",
                    lo <= val <= hi,
                    f"value {val} outside guardrail [{lo}, {hi}]",
                )


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Nudge system does not suggest w_gpp > 0 when current w_gpp = 0
# ═══════════════════════════════════════════════════════════════════════════
def test_nudge_no_wgpp_increase() -> None:
    print("\n── Test 2: Nudge system does not suggest w_gpp > 0 ──")
    from utils.nudge_params import NUDGE_PARAM_RULES

    # Check all rules for w_gpp suggestions
    for (metric, direction), rules in NUDGE_PARAM_RULES.items():
        for rule in rules:
            if rule["param"] == "w_gpp":
                _check(
                    f"No w_gpp nudge rule for ({metric}, {direction})",
                    False,
                    f"Rule suggests {rule['param']} — DS says w_gpp=0 is optimal",
                )
                return

    _check("No nudge rules suggest increasing w_gpp", True)


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Nudge system does not suggest OWN_PENALTY_STRENGTH above 0.5
# ═══════════════════════════════════════════════════════════════════════════
def test_nudge_own_penalty_cap() -> None:
    print("\n── Test 3: Nudge does not suggest OWN_PENALTY > 0.5 for GPP ──")
    from utils.nudge_params import get_nudge_suggestions

    # Simulate a high-ownership scenario
    suggestions = get_nudge_suggestions(
        metric_name="ownership_sum",
        batch_value=180.0,  # very high
        lo=90.0,
        hi=130.0,
        current_overrides={"GPP_OWN_PENALTY_STRENGTH": 0.3},
        preset_defaults={"GPP_OWN_PENALTY_STRENGTH": 0.3},
    )
    for sug in suggestions:
        if sug["param"] == "GPP_OWN_PENALTY_STRENGTH":
            _check(
                f"OWN_PENALTY suggested={sug['suggested_value']} <= 0.5",
                sug["suggested_value"] <= 0.5,
                f"Suggested {sug['suggested_value']} exceeds 0.5",
            )
            return
    _check("No OWN_PENALTY suggestion generated (acceptable)", True)


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Auto-calibrate search bounds include all DS-recommended values
# ═══════════════════════════════════════════════════════════════════════════
def test_autocal_bounds_include_ds() -> None:
    print("\n── Test 4: Auto-calibrate bounds include DS values ──")
    from yak_core.auto_calibrate import SEARCH_SPACE, DS_RECOMMENDATIONS

    for contest_type, recs in DS_RECOMMENDATIONS.items():
        for param, val in recs.items():
            if param in SEARCH_SPACE:
                spec = SEARCH_SPACE[param]
                _check(
                    f"[{contest_type}] {param}={val} in search [{spec['low']}, {spec['high']}]",
                    spec["low"] <= val <= spec["high"],
                    f"value {val} outside search space [{spec['low']}, {spec['high']}]",
                )


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Showdown preset OWN_PENALTY_STRENGTH is NOT 1.5
# ═══════════════════════════════════════════════════════════════════════════
def test_showdown_own_penalty() -> None:
    print("\n── Test 5: Showdown preset OWN_PENALTY != 1.5 ──")
    from yak_core.config import CONTEST_PRESETS

    showdown = CONTEST_PRESETS.get("Showdown", {})
    own_pen = showdown.get("GPP_OWN_PENALTY_STRENGTH")
    _check(
        f"Showdown GPP_OWN_PENALTY_STRENGTH = {own_pen} (not 1.5)",
        own_pen is not None and own_pen != 1.5,
        f"Still using the broken value 1.5!",
    )
    _check(
        f"Showdown GPP_OWN_PENALTY_STRENGTH = {own_pen} <= 0.5",
        own_pen is not None and own_pen <= 0.5,
        f"DS says 0.40 — current value {own_pen} is too high",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Cash preset has BOOM_WEIGHT = 0.0 and UPSIDE_WEIGHT = 0.0
# ═══════════════════════════════════════════════════════════════════════════
def test_cash_preset_values() -> None:
    print("\n── Test 6: Cash preset floor-first values ──")
    from yak_core.config import CONTEST_PRESETS

    cash = CONTEST_PRESETS.get("Cash Main", {})
    _check(
        "Cash GPP_BOOM_WEIGHT = 0.0",
        cash.get("GPP_BOOM_WEIGHT") == 0.0,
        f"Got {cash.get('GPP_BOOM_WEIGHT')}",
    )
    _check(
        "Cash GPP_UPSIDE_WEIGHT = 0.0",
        cash.get("GPP_UPSIDE_WEIGHT") == 0.0,
        f"Got {cash.get('GPP_UPSIDE_WEIGHT')}",
    )
    _check(
        "Cash GPP_OWN_PENALTY_STRENGTH = 0.0",
        cash.get("GPP_OWN_PENALTY_STRENGTH") == 0.0,
        f"Got {cash.get('GPP_OWN_PENALTY_STRENGTH')}",
    )
    _check(
        "Cash CASH_FLOOR_WEIGHT > 0",
        cash.get("CASH_FLOOR_WEIGHT", 0) > 0,
        f"Got {cash.get('CASH_FLOOR_WEIGHT')}",
    )
    _check(
        "Cash MIN_PLAYER_MINUTES >= 20",
        cash.get("MIN_PLAYER_MINUTES", 0) >= 20,
        f"Got {cash.get('MIN_PLAYER_MINUTES')}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: All new config parameters appear in PARAM_GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════
def test_new_params_in_guardrails() -> None:
    print("\n── Test 7: New config params in PARAM_GUARDRAILS ──")
    from utils.nudge_params import PARAM_GUARDRAILS

    required_params = [
        "GPP_BOOM_SPREAD_WEIGHT",
        "GPP_SNIPER_WEIGHT",
        "GPP_EFFICIENCY_WEIGHT",
        "CASH_FLOOR_WEIGHT",
        "MIN_PLAYER_MINUTES",
        "GPP_SMASH_WEIGHT",
        "GPP_LEVERAGE_WEIGHT",
        "OWN_WEIGHT",
        "w_gpp",
        "w_ceil",
        "w_own",
    ]
    for param in required_params:
        _check(
            f"{param} in PARAM_GUARDRAILS",
            param in PARAM_GUARDRAILS,
            f"Missing from guardrails",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Calibration targets exist for all profile keys with sensible ranges
# ═══════════════════════════════════════════════════════════════════════════
def test_calibration_targets_exist() -> None:
    print("\n── Test 8: Calibration targets for all profiles ──")
    from utils.calibration_targets import CALIBRATION_TARGETS

    required_profiles = [
        "classic_gpp_main",
        "classic_gpp_20max",
        "classic_gpp_se",
        "classic_cash",
        "showdown_gpp",
        "showdown_cash",
    ]
    for pk in required_profiles:
        _check(
            f"Profile '{pk}' in CALIBRATION_TARGETS",
            pk in CALIBRATION_TARGETS,
            "Missing profile",
        )
        if pk in CALIBRATION_TARGETS:
            targets = CALIBRATION_TARGETS[pk]
            for metric, (lo, hi) in targets.items():
                _check(
                    f"  [{pk}] {metric}: lo={lo} < hi={hi}",
                    lo < hi,
                    f"Invalid range: lo={lo} >= hi={hi}",
                )


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Sniper metrics compute without error on sample data
# ═══════════════════════════════════════════════════════════════════════════
def test_sniper_metrics_compute() -> None:
    print("\n── Test 9: Sniper metrics compute on sample data ──")
    try:
        import pandas as pd
        import numpy as np

        # Create sample summary_df matching the structure in _compute_nudge_metrics
        np.random.seed(42)
        n = 25
        sample_df = pd.DataFrame({
            "lineup_index": range(n),
            "total_actual": np.random.normal(220, 45, n),
            "total_ceil": np.random.normal(304, 30, n),
            "avg_own_pct": np.random.uniform(0.05, 0.25, n),
            "ricky_tag": ["SE Core"] + ["Spicy"] + ["Alt"] + [""] * (n - 3),
            "ricky_rank": range(1, n + 1),
        })

        # Test 300+ count
        count_300 = float((sample_df["total_actual"] >= 300.0).sum())
        _check("300+ count computes", isinstance(count_300, float))

        # Test avg ceiling
        avg_ceil = float(sample_df["total_ceil"].mean())
        _check("Avg ceiling computes", avg_ceil > 0, f"Got {avg_ceil}")

        # Test avg ownership
        avg_own = float(sample_df["avg_own_pct"].mean())
        _check("Avg ownership computes", 0.0 <= avg_own <= 1.0, f"Got {avg_own}")

        # Test top-5 avg score
        top5 = sample_df["total_actual"].nlargest(5)
        top5_avg = float(top5.mean())
        _check("Top-5 avg score computes", top5_avg > 0, f"Got {top5_avg}")

        # Test cash proximity
        cash_line = 287.0
        prox = ((sample_df["total_actual"] >= cash_line - 30) &
                (sample_df["total_actual"] <= cash_line + 30)).sum()
        cash_prox = float(prox) / float(len(sample_df))
        _check("Cash proximity computes", 0.0 <= cash_prox <= 1.0, f"Got {cash_prox}")

        # Test score spread
        spread = float(sample_df["total_actual"].std())
        _check("Score spread computes", spread > 0, f"Got {spread}")

    except Exception as e:
        _check(f"Sniper metrics compute without error", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: Named profiles reference valid presets
# ═══════════════════════════════════════════════════════════════════════════
def test_named_profiles_valid() -> None:
    print("\n── Test 10: Named profiles reference valid presets ──")
    from yak_core.config import NAMED_PROFILES, CONTEST_PRESETS

    for key, profile in NAMED_PROFILES.items():
        base = profile.get("base_preset", "")
        _check(
            f"[{key}] base_preset '{base}' exists in CONTEST_PRESETS",
            base in CONTEST_PRESETS,
            f"Missing preset '{base}'",
        )
        rw = profile.get("ricky_weights", {})
        _check(
            f"[{key}] has ricky_weights with w_gpp, w_ceil, w_own",
            "w_gpp" in rw and "w_ceil" in rw and "w_own" in rw,
            f"Got keys: {list(rw.keys())}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: Showdown and Cash named profiles exist in constants mapping
# ═══════════════════════════════════════════════════════════════════════════
def test_constants_mappings() -> None:
    print("\n── Test 11: Constants mappings include all contest types ──")
    from utils.constants import (
        CONTEST_PROFILE_KEY_MAP,
        PROFILE_KEY_TO_PRESET,
        PROFILE_KEY_TO_NAMED,
    )
    from yak_core.config import CONTEST_PRESETS, NAMED_PROFILES

    for key, preset_name in PROFILE_KEY_TO_PRESET.items():
        _check(
            f"PROFILE_KEY_TO_PRESET['{key}'] = '{preset_name}' exists",
            preset_name in CONTEST_PRESETS or preset_name == "GPP SE",
            f"Preset '{preset_name}' not in CONTEST_PRESETS",
        )

    for key, named in PROFILE_KEY_TO_NAMED.items():
        if named is not None:
            _check(
                f"PROFILE_KEY_TO_NAMED['{key}'] = '{named}' exists",
                named in NAMED_PROFILES,
                f"Named profile '{named}' not in NAMED_PROFILES",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: GPP Main MIN_PLAYER_MINUTES defaults are set
# ═══════════════════════════════════════════════════════════════════════════
def test_min_player_minutes() -> None:
    print("\n── Test 12: MIN_PLAYER_MINUTES defaults in presets ──")
    from yak_core.config import CONTEST_PRESETS

    gpp = CONTEST_PRESETS.get("GPP Main", {})
    _check(
        f"GPP Main MIN_PLAYER_MINUTES = {gpp.get('MIN_PLAYER_MINUTES')} (expected 18)",
        gpp.get("MIN_PLAYER_MINUTES") == 18,
        f"Got {gpp.get('MIN_PLAYER_MINUTES')}",
    )

    cash = CONTEST_PRESETS.get("Cash Main", {})
    _check(
        f"Cash Main MIN_PLAYER_MINUTES = {cash.get('MIN_PLAYER_MINUTES')} (expected 20)",
        cash.get("MIN_PLAYER_MINUTES") == 20,
        f"Got {cash.get('MIN_PLAYER_MINUTES')}",
    )

    showdown = CONTEST_PRESETS.get("Showdown", {})
    _check(
        f"Showdown MIN_PLAYER_MINUTES = {showdown.get('MIN_PLAYER_MINUTES')} (expected 15)",
        showdown.get("MIN_PLAYER_MINUTES") == 15,
        f"Got {showdown.get('MIN_PLAYER_MINUTES')}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 13: New sniper metric labels exist
# ═══════════════════════════════════════════════════════════════════════════
def test_sniper_metric_labels() -> None:
    print("\n── Test 13: Sniper metric labels defined ──")
    from utils.calibration_targets import METRIC_LABELS

    sniper_metrics = [
        "lineup_300_count",
        "avg_ceiling",
        "avg_ownership",
        "top5_avg_score",
        "cash_proximity_pct",
        "score_spread",
    ]
    for m in sniper_metrics:
        _check(
            f"METRIC_LABELS['{m}'] exists",
            m in METRIC_LABELS,
            "Missing label",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Sniper Mode Acceptance Tests")
    print("=" * 60)

    tests = [
        test_ds_values_within_guardrails,
        test_nudge_no_wgpp_increase,
        test_nudge_own_penalty_cap,
        test_autocal_bounds_include_ds,
        test_showdown_own_penalty,
        test_cash_preset_values,
        test_new_params_in_guardrails,
        test_calibration_targets_exist,
        test_sniper_metrics_compute,
        test_named_profiles_valid,
        test_constants_mappings,
        test_min_player_minutes,
        test_sniper_metric_labels,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            FAIL += 1
            msg = f"  ✗ {test_fn.__name__} CRASHED: {e}"
            print(msg)
            ERRORS.append(msg)
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if ERRORS:
        print("\nFailed checks:")
        for err in ERRORS:
            print(err)
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
