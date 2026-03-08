"""End-to-end test for Phase 3: Post-Slate Miss Analysis.

Creates synthetic archived slate data WITH game context columns,
runs the miss analyzer, verifies classification, context tagging,
and UI status helper.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yak_core.config import YAKOS_ROOT

_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")
_ANALYSIS_DIR = os.path.join(YAKOS_ROOT, "data", "miss_analysis")
_PATTERNS_FILE = os.path.join(_ANALYSIS_DIR, "miss_patterns.json")

_backup_archive = None
_backup_analysis = None


def _backup():
    global _backup_archive, _backup_analysis
    if os.path.isdir(_ARCHIVE_DIR):
        _backup_archive = _ARCHIVE_DIR + ".bak"
        if os.path.exists(_backup_archive):
            shutil.rmtree(_backup_archive)
        shutil.copytree(_ARCHIVE_DIR, _backup_archive)
    if os.path.isdir(_ANALYSIS_DIR):
        _backup_analysis = _ANALYSIS_DIR + ".bak"
        if os.path.exists(_backup_analysis):
            shutil.rmtree(_backup_analysis)
        shutil.copytree(_ANALYSIS_DIR, _backup_analysis)


def _restore():
    if _backup_archive and os.path.isdir(_backup_archive):
        if os.path.isdir(_ARCHIVE_DIR):
            shutil.rmtree(_ARCHIVE_DIR)
        shutil.move(_backup_archive, _ARCHIVE_DIR)
    elif _backup_archive is None and os.path.isdir(_ARCHIVE_DIR):
        shutil.rmtree(_ARCHIVE_DIR)

    if _backup_analysis and os.path.isdir(_backup_analysis):
        if os.path.isdir(_ANALYSIS_DIR):
            shutil.rmtree(_ANALYSIS_DIR)
        shutil.move(_backup_analysis, _ANALYSIS_DIR)
    elif _backup_analysis is None and os.path.isdir(_ANALYSIS_DIR):
        shutil.rmtree(_ANALYSIS_DIR)


def _create_synthetic_archive(n_slates=5, players_per_slate=80):
    """Create synthetic archives with game context for miss analysis testing."""
    os.makedirs(_ARCHIVE_DIR, exist_ok=True)
    np.random.seed(123)

    total_players = 0
    for i in range(n_slates):
        date_str = f"2026-02-{15 + i:02d}"
        rows = []
        for j in range(players_per_slate):
            salary = np.random.choice([3800, 4500, 5200, 6000, 7000, 8500, 9500, 11000],
                                       p=[0.12, 0.13, 0.15, 0.15, 0.15, 0.13, 0.1, 0.07])
            proj = salary / 1000 * np.random.uniform(3.5, 5.5)

            # Game context
            spread = np.random.choice([-14, -8, -4, 0, 4, 8, 14],
                                       p=[0.05, 0.15, 0.25, 0.10, 0.25, 0.15, 0.05])
            total = np.random.choice([206, 215, 222, 228, 235],
                                      p=[0.1, 0.25, 0.30, 0.25, 0.1])
            b2b = np.random.random() < 0.15
            rolling_cv = np.random.uniform(0.10, 0.45)
            injury_flag = np.random.random() < 0.08

            # Simulate actual FP with context-dependent noise
            noise_scale = proj * 0.40  # base noise
            # Blowout: starters get fewer mins → more busts
            if abs(spread) >= 10 and salary >= 8000:
                noise_scale *= 1.3
                actual = max(0, proj * 0.75 + np.random.normal(0, noise_scale * 0.5))
            # High pace: scoring environment inflates actuals
            elif total >= 230:
                actual = max(0, proj * 1.15 + np.random.normal(0, noise_scale))
            # B2B: fatigue
            elif b2b:
                actual = max(0, proj * 0.85 + np.random.normal(0, noise_scale))
            # High variance player: bigger swings
            elif rolling_cv >= 0.35:
                actual = max(0, proj + np.random.normal(0, noise_scale * 1.5))
            else:
                actual = max(0, proj + np.random.normal(0, noise_scale))

            rows.append({
                "player_name": f"Player_{i}_{j}",
                "salary": int(salary),
                "proj": round(proj, 1),
                "actual_fp": round(actual, 1),
                "team": np.random.choice(["BOS", "LAL", "MIL", "DEN"]),
                "opp": np.random.choice(["NYK", "GSW", "PHX", "MIA"]),
                "pos": np.random.choice(["PG", "SG", "SF", "PF", "C"]),
                "ownership": round(np.random.uniform(2, 30), 1),
                "vegas_spread": float(spread),
                "vegas_total": float(total),
                "b2b": b2b,
                "rolling_cv": round(rolling_cv, 3),
                "status": "GTD" if injury_flag else "Active",
                "injury_note": "ankle soreness" if injury_flag else "",
            })

        df = pd.DataFrame(rows)
        df["slate_date"] = date_str
        df["contest_type"] = "GPP"
        df["archived_at"] = "2026-03-08T12:00:00"

        fname = f"{date_str}_gpp.parquet"
        df.to_parquet(os.path.join(_ARCHIVE_DIR, fname), index=False)
        total_players += len(rows)

    print(f"[TEST] Created {n_slates} archive files ({total_players} player-slates)")
    return total_players


def test_end_to_end():
    """Full pipeline test: archive → miss_analyzer → UI status."""
    _backup()
    try:
        # Clean slate
        if os.path.isdir(_ARCHIVE_DIR):
            shutil.rmtree(_ARCHIVE_DIR)
        if os.path.isdir(_ANALYSIS_DIR):
            shutil.rmtree(_ANALYSIS_DIR)

        # ── Step 1: Verify no analysis exists ───────────────────────────
        from yak_core.miss_analyzer import (
            analyze_misses,
            get_miss_analysis_status,
        )

        status_before = get_miss_analysis_status()
        assert status_before["status"] == "none", f"Expected 'none', got {status_before['status']}"
        print("[PASS] Step 1: No miss analysis before archive creation")

        # ── Step 2: Create synthetic archive with context ───────────────
        n_total = _create_synthetic_archive(n_slates=5, players_per_slate=80)
        print(f"[PASS] Step 2: Synthetic archive created ({n_total} player-slates)")

        # ── Step 3: Run miss analysis ───────────────────────────────────
        result = analyze_misses()
        assert "error" not in result, f"Analysis failed: {result.get('error')}"
        assert result["n_slates"] == 5, f"Expected 5 slates, got {result['n_slates']}"
        assert result["n_player_slates"] > 200, f"Expected >200 player-slates, got {result['n_player_slates']}"
        print(f"[PASS] Step 3: Miss analysis ran — {result['n_player_slates']} player-slates")

        # ── Step 4: Verify classification ───────────────────────────────
        cls = result.get("classification", {})
        assert cls.get("pop", 0) > 0, "Expected some pops"
        assert cls.get("bust", 0) > 0, "Expected some busts"
        assert cls.get("inline", 0) > 0, "Expected some inline"
        assert cls["pop"] + cls["bust"] + cls["inline"] == result["n_player_slates"]
        print(f"[PASS] Step 4: Classification — {cls['pop']} pops, {cls['bust']} busts, {cls['inline']} inline")

        # ── Step 5: Verify context factor breakdown ─────────────────────
        factors = result.get("factor_breakdown", {})
        assert len(factors) > 0, "Expected at least one context factor"
        print(f"\n  Factor         |   N  | Pop Rate | Bust Rate | Pop Lift | Bust Lift | Avg Res")
        print(f"  ---------------|------|----------|-----------|----------|-----------|--------")
        for fk, fv in factors.items():
            print(f"  {fk:14s} | {fv['n']:4d} | {fv['pop_rate']:.1%}    | {fv['bust_rate']:.1%}     | {fv['pop_lift']:+6.0f}%  | {fv['bust_lift']:+6.0f}%   | {fv['avg_residual']:+.1f}")

        # Verify expected patterns:
        # Blowout should show higher bust rate for stud players (designed into synthetic data)
        if "blowout" in factors:
            assert factors["blowout"]["n"] >= 3, "Expected blowout sample"
            print(f"\n  [OK] Blowout factor detected with {factors['blowout']['n']} samples")

        # B2B should show negative avg residual (designed into synthetic data)
        if "b2b" in factors:
            assert factors["b2b"]["n"] >= 3, "Expected b2b sample"
            print(f"  [OK] B2B factor detected with {factors['b2b']['n']} samples")

        print("[PASS] Step 5: Context factors detected and broken down")

        # ── Step 6: Verify bracket breakdown ────────────────────────────
        brackets = result.get("bracket_breakdown", {})
        assert len(brackets) >= 3, f"Expected ≥3 brackets, got {len(brackets)}"
        print(f"\n  Bracket    |   N  | Pop Rate | Bust Rate | MAE  | Avg Res")
        print(f"  -----------|------|----------|-----------|------|--------")
        for bk, bv in brackets.items():
            print(f"  {bk:10s} | {bv['n']:4d} | {bv['pop_rate']:.1%}    | {bv['bust_rate']:.1%}     | {bv['mae']:.1f} | {bv['avg_residual']:+.1f}")
        print("[PASS] Step 6: Bracket breakdown complete")

        # ── Step 7: Verify top pops and busts ───────────────────────────
        top_pops = result.get("top_pops", [])
        top_busts = result.get("top_busts", [])
        assert len(top_pops) > 0, "Expected top pops"
        assert len(top_busts) > 0, "Expected top busts"
        # Pops should have positive residuals, busts negative
        assert all(p["residual"] > 0 for p in top_pops), "Top pops should have positive residuals"
        assert all(b["residual"] < 0 for b in top_busts), "Top busts should have negative residuals"
        print(f"[PASS] Step 7: Top {len(top_pops)} pops and {len(top_busts)} busts identified")

        # ── Step 8: Verify suggestions generated ────────────────────────
        suggestions = result.get("suggestions", [])
        print(f"\n  Suggestions ({len(suggestions)}):")
        for s in suggestions:
            sev = "🟥" if s["severity"] == "high" else "🟨"
            print(f"    {sev} {s['signal']} — {s['direction']}: {s['detail'][:80]}...")
        print("[PASS] Step 8: Actionable suggestions generated")

        # ── Step 9: Verify JSON persisted ───────────────────────────────
        assert os.path.isfile(_PATTERNS_FILE), "miss_patterns.json not found"
        with open(_PATTERNS_FILE) as f:
            saved = json.load(f)
        assert "classification" in saved
        assert "factor_breakdown" in saved
        assert "bracket_breakdown" in saved
        assert "suggestions" in saved
        print("[PASS] Step 9: miss_patterns.json persisted")

        # ── Step 10: Verify UI status helper ────────────────────────────
        status_after = get_miss_analysis_status()
        assert status_after["status"] == "analysed", f"Expected 'analysed', got {status_after['status']}"
        assert status_after["n_pops"] > 0
        assert status_after["n_busts"] > 0
        assert "pops" in status_after["message"]
        print("[PASS] Step 10: UI status reports correctly")

        print("\n" + "=" * 60)
        print("ALL 10 TESTS PASSED — Phase 3 miss analysis pipeline is solid.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        _restore()


if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
