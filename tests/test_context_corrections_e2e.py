"""End-to-end test: miss analysis → context corrections → applied to pool.

Creates synthetic archive data, runs miss analysis to generate context
corrections, then applies those corrections to a synthetic pool and
verifies adjustments are correct.
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
_CORRECTIONS_FILE = os.path.join(_ANALYSIS_DIR, "context_corrections.json")

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


def _create_biased_archive(n_slates=6, players_per_slate=80):
    """Create archive with strong, detectable biases for testing."""
    os.makedirs(_ARCHIVE_DIR, exist_ok=True)
    np.random.seed(999)

    for i in range(n_slates):
        date_str = f"2026-02-{20 + i:02d}"
        rows = []
        for j in range(players_per_slate):
            salary = np.random.choice([4000, 5500, 7000, 9000, 11000])
            proj = salary / 1000 * 4.5

            # Assign context flags
            is_blowout = np.random.random() < 0.20
            is_high_pace = np.random.random() < 0.15
            is_b2b = np.random.random() < 0.12
            is_injury = np.random.random() < 0.10

            spread = -12.0 if is_blowout else -4.0
            total = 234.0 if is_high_pace else 220.0

            # Build strong biases into the data:
            # Blowout: model over-projects by ~5 FP on average
            # High pace: model under-projects by ~4 FP
            # B2B: model over-projects by ~3 FP
            # Injury: model under-projects by ~3 FP
            bias = 0.0
            if is_blowout:
                bias -= 5.0  # actual is lower than proj
            if is_high_pace:
                bias += 4.0  # actual is higher than proj
            if is_b2b:
                bias -= 3.0
            if is_injury:
                bias += 3.0

            actual = max(1, proj + bias + np.random.normal(0, proj * 0.15))

            rows.append({
                "player_name": f"TestPlayer_{i}_{j}",
                "salary": int(salary),
                "proj": round(proj, 1),
                "actual_fp": round(actual, 1),
                "team": "BOS",
                "opp": "LAL",
                "pos": "SF",
                "ownership": 10.0,
                "vegas_spread": float(spread),
                "vegas_total": float(total),
                "b2b": is_b2b,
                "rolling_cv": 0.20,
                "status": "GTD" if is_injury else "Active",
                "injury_note": "knee soreness" if is_injury else "",
            })

        df = pd.DataFrame(rows)
        df["slate_date"] = date_str
        df["contest_type"] = "GPP"
        df["archived_at"] = "2026-03-08T12:00:00"
        df.to_parquet(os.path.join(_ARCHIVE_DIR, f"{date_str}_gpp.parquet"), index=False)

    print(f"[TEST] Created {n_slates} biased archive files")


def test_end_to_end():
    """Full pipeline: archive → miss analysis → context corrections → apply to pool."""
    _backup()
    try:
        # Clean
        if os.path.isdir(_ARCHIVE_DIR):
            shutil.rmtree(_ARCHIVE_DIR)
        if os.path.isdir(_ANALYSIS_DIR):
            shutil.rmtree(_ANALYSIS_DIR)

        # ── Step 1: Create biased archive ───────────────────────────────
        _create_biased_archive()
        print("[PASS] Step 1: Biased archive created")

        # ── Step 2: Run miss analysis (generates context corrections) ───
        from yak_core.miss_analyzer import analyze_misses, get_context_corrections

        result = analyze_misses()
        assert "error" not in result, f"Analysis failed: {result.get('error')}"
        assert "context_corrections" in result, "Missing context_corrections in result"
        print("[PASS] Step 2: Miss analysis ran")

        # ── Step 3: Verify context corrections computed ─────────────────
        ctx_corr = result["context_corrections"]
        factors = ctx_corr.get("factors", {})
        n_active = ctx_corr.get("n_active", 0)
        assert n_active > 0, f"Expected active corrections, got {n_active}"

        print(f"\n  Factor         | Active | Correction | Raw Residual | N")
        print(f"  ---------------|--------|------------|--------------|----")
        for fk, fv in factors.items():
            active = "✅" if fv.get("active") else "⬜"
            corr = fv.get("correction_fp", 0)
            raw = fv.get("raw_residual", 0)
            n = fv.get("n", 0)
            print(f"  {fk:14s} | {active}     | {corr:+.2f} FP   | {raw:+.2f} FP      | {n}")

        # Verify expected directions:
        # Blowout should have NEGATIVE correction (model over-projects)
        blowout = factors.get("blowout", {})
        if blowout.get("active"):
            assert blowout["correction_fp"] < 0, f"Blowout correction should be negative, got {blowout['correction_fp']}"
            print(f"\n  [OK] Blowout correction is negative ({blowout['correction_fp']:+.2f}) — dampening over-projection")

        # High pace should have POSITIVE correction (model under-projects)
        hp = factors.get("high_pace", {})
        if hp.get("active"):
            assert hp["correction_fp"] > 0, f"High pace correction should be positive, got {hp['correction_fp']}"
            print(f"  [OK] High pace correction is positive ({hp['correction_fp']:+.2f}) — boosting under-projection")

        # B2B should have NEGATIVE correction
        b2b = factors.get("b2b", {})
        if b2b.get("active"):
            assert b2b["correction_fp"] < 0, f"B2B correction should be negative, got {b2b['correction_fp']}"
            print(f"  [OK] B2B correction is negative ({b2b['correction_fp']:+.2f}) — dampening over-projection")

        print("[PASS] Step 3: Context corrections computed with correct directions")

        # ── Step 4: Verify corrections file persisted ───────────────────
        assert os.path.isfile(_CORRECTIONS_FILE), "context_corrections.json not found"
        loaded = get_context_corrections()
        assert loaded.get("n_active", 0) > 0
        print("[PASS] Step 4: context_corrections.json persisted and loadable")

        # ── Step 5: Apply corrections to a synthetic pool ───────────────
        from yak_core.calibration_feedback import apply_context_corrections

        # Create a pool with mixed contexts
        pool = pd.DataFrame([
            {"player_name": "Blowout Star",    "proj": 40.0, "salary": 9000, "vegas_spread": -14.0, "vegas_total": 220.0, "b2b": False, "status": "Active", "injury_note": ""},
            {"player_name": "High Pace Guy",   "proj": 30.0, "salary": 7000, "vegas_spread": -3.0,  "vegas_total": 235.0, "b2b": False, "status": "Active", "injury_note": ""},
            {"player_name": "B2B Warrior",     "proj": 25.0, "salary": 6000, "vegas_spread": -5.0,  "vegas_total": 222.0, "b2b": True,  "status": "Active", "injury_note": ""},
            {"player_name": "GTD Player",      "proj": 35.0, "salary": 8000, "vegas_spread": -4.0,  "vegas_total": 220.0, "b2b": False, "status": "GTD",    "injury_note": "ankle"},
            {"player_name": "Normal Player",   "proj": 28.0, "salary": 7000, "vegas_spread": -3.0,  "vegas_total": 222.0, "b2b": False, "status": "Active", "injury_note": ""},
            {"player_name": "Multi Context",   "proj": 32.0, "salary": 8000, "vegas_spread": -11.0, "vegas_total": 232.0, "b2b": True,  "status": "GTD",    "injury_note": "knee"},
        ])

        original_projs = pool["proj"].copy()
        adjusted_pool = apply_context_corrections(pool, loaded)

        print(f"\n  Player          | Orig Proj | Adj Proj | Context Corr | Context")
        print(f"  ----------------|-----------|----------|--------------|--------")
        for idx, row in adjusted_pool.iterrows():
            name = row["player_name"]
            orig = original_projs[idx]
            adj = row["proj"]
            corr = row.get("context_correction", 0)
            ctx = []
            if abs(float(row.get("vegas_spread", 0))) >= 10: ctx.append("blowout")
            if float(row.get("vegas_total", 0)) >= 230: ctx.append("high_pace")
            if row.get("b2b"): ctx.append("b2b")
            if str(row.get("status", "")).upper() in ("GTD", "QUESTIONABLE"): ctx.append("injury")
            print(f"  {name:15s} | {orig:9.1f} | {adj:8.1f} | {corr:+12.2f} | {', '.join(ctx) or 'none'}")

        # Verify specific adjustments
        blowout_star = adjusted_pool[adjusted_pool["player_name"] == "Blowout Star"].iloc[0]
        assert blowout_star["proj"] < 40.0, "Blowout Star should have reduced projection"

        high_pace_guy = adjusted_pool[adjusted_pool["player_name"] == "High Pace Guy"].iloc[0]
        assert high_pace_guy["proj"] > 30.0, "High Pace Guy should have boosted projection"

        normal_player = adjusted_pool[adjusted_pool["player_name"] == "Normal Player"].iloc[0]
        assert normal_player["proj"] == 28.0, "Normal Player should be unchanged"

        multi_ctx = adjusted_pool[adjusted_pool["player_name"] == "Multi Context"].iloc[0]
        # Multi context has blowout + high_pace + b2b + injury — corrections stack
        assert multi_ctx.get("context_correction", 0) != 0, "Multi Context should have stacked corrections"

        print("[PASS] Step 5: Context corrections applied correctly to pool")

        # ── Step 6: Verify no corrections applied when no data ──────────
        # Remove corrections file and test graceful fallback
        os.remove(_CORRECTIONS_FILE)
        fallback_pool = apply_context_corrections(pool)
        assert (fallback_pool["proj"] == pool["proj"]).all(), "Pool should be unchanged when no corrections file exists"
        print("[PASS] Step 6: Graceful fallback when no corrections")

        print("\n" + "=" * 60)
        print("ALL 6 TESTS PASSED — Context corrections pipeline wired end-to-end.")
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
