"""End-to-end test for Phase 2: Dynamic Variance from Archived Slates.

Creates synthetic archived slate data, runs the variance learner,
verifies edge.py picks up learned ratios, and checks the UI status helper.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

# Ensure yak_core is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yak_core.config import YAKOS_ROOT

# ---------------------------------------------------------------------------
# Test paths — we create a temp archive to avoid polluting real data
# ---------------------------------------------------------------------------
_ARCHIVE_DIR = os.path.join(YAKOS_ROOT, "data", "slate_archive")
_MODEL_DIR = os.path.join(YAKOS_ROOT, "data", "variance_model")
_RATIOS_FILE = os.path.join(_MODEL_DIR, "learned_ratios.json")

# Backup existing files so we can restore after test
_backup_archive = None
_backup_ratios = None


def _backup():
    """Back up existing archive + model so test is non-destructive."""
    global _backup_archive, _backup_ratios
    if os.path.isdir(_ARCHIVE_DIR):
        _backup_archive = _ARCHIVE_DIR + ".bak"
        if os.path.exists(_backup_archive):
            shutil.rmtree(_backup_archive)
        shutil.copytree(_ARCHIVE_DIR, _backup_archive)
    if os.path.isfile(_RATIOS_FILE):
        _backup_ratios = _RATIOS_FILE + ".bak"
        shutil.copy2(_RATIOS_FILE, _backup_ratios)


def _restore():
    """Restore original archive + model."""
    # Restore archive
    if _backup_archive and os.path.isdir(_backup_archive):
        if os.path.isdir(_ARCHIVE_DIR):
            shutil.rmtree(_ARCHIVE_DIR)
        shutil.move(_backup_archive, _ARCHIVE_DIR)
    elif _backup_archive is None:
        # No original archive existed — remove what we created
        if os.path.isdir(_ARCHIVE_DIR):
            shutil.rmtree(_ARCHIVE_DIR)

    # Restore ratios
    if _backup_ratios and os.path.isfile(_backup_ratios):
        shutil.move(_backup_ratios, _RATIOS_FILE)
    elif _backup_ratios is None:
        # No original ratios existed — remove what we created
        if os.path.isfile(_RATIOS_FILE):
            os.remove(_RATIOS_FILE)


def _create_synthetic_archive(n_slates=5, players_per_slate=60):
    """Create synthetic archived slate parquet files with realistic NBA data."""
    os.makedirs(_ARCHIVE_DIR, exist_ok=True)

    np.random.seed(42)
    salary_tiers = {
        "lt5k":     (3500, 4900, 12),   # (salary_lo, salary_hi, n_players)
        "5_65k":    (5000, 6400, 15),
        "65_8k":    (6500, 7900, 15),
        "8_10k":    (8000, 9900, 12),
        "10k_plus": (10000, 12000, 6),
    }

    for i in range(n_slates):
        date_str = f"2026-02-{10 + i:02d}"
        rows = []
        for tier, (lo, hi, n) in salary_tiers.items():
            for j in range(n):
                salary = np.random.randint(lo, hi + 1)
                # Projection scales with salary
                proj = salary / 1000 * np.random.uniform(3.5, 5.5)
                # Actual = proj + noise (the noise IS what we're measuring)
                noise_ratio = {
                    "lt5k": 1.0,
                    "5_65k": 0.60,
                    "65_8k": 0.42,
                    "8_10k": 0.33,
                    "10k_plus": 0.28,
                }[tier]
                actual = max(0, proj + np.random.normal(0, proj * noise_ratio))
                rows.append({
                    "player_name": f"Player_{tier}_{j}",
                    "salary": salary,
                    "proj": round(proj, 1),
                    "actual_fp": round(actual, 1),
                    "team": "BOS",
                    "opp": "LAL",
                    "pos": "SF",
                    "ownership": np.random.uniform(2, 30),
                })

        df = pd.DataFrame(rows)
        df["slate_date"] = date_str
        df["contest_type"] = "GPP"
        df["archived_at"] = "2026-03-08T00:00:00"

        fname = f"{date_str}_gpp.parquet"
        df.to_parquet(os.path.join(_ARCHIVE_DIR, fname), index=False)

    print(f"[TEST] Created {n_slates} synthetic archive files in {_ARCHIVE_DIR}")
    return n_slates * sum(t[2] for t in salary_tiers.values())


def test_end_to_end():
    """Full pipeline test: archive → variance_learner → edge.py → status."""
    _backup()
    try:
        # Clean slate — remove existing archive and model
        if os.path.isdir(_ARCHIVE_DIR):
            shutil.rmtree(_ARCHIVE_DIR)
        if os.path.isfile(_RATIOS_FILE):
            os.remove(_RATIOS_FILE)

        # ── Step 1: Verify no model exists ──────────────────────────────
        from yak_core.variance_learner import (
            get_variance_model_status,
            load_learned_ratios,
            recalculate_variance_model,
        )

        status_before = get_variance_model_status()
        assert status_before["status"] == "static", f"Expected 'static', got {status_before['status']}"
        assert load_learned_ratios() is None, "Expected no learned ratios before archive"
        print("[PASS] Step 1: No learned model before archive creation")

        # ── Step 2: Create synthetic archive ────────────────────────────
        n_total = _create_synthetic_archive(n_slates=5, players_per_slate=60)
        print(f"[PASS] Step 2: Synthetic archive created ({n_total} player-slates)")

        # ── Step 3: Run variance recalculation ──────────────────────────
        result = recalculate_variance_model()
        assert "error" not in result, f"Recalculation failed: {result.get('error')}"
        assert result["n_slates"] == 5, f"Expected 5 slates, got {result['n_slates']}"
        assert result["n_player_slates"] > 100, f"Expected >100 player-slates, got {result['n_player_slates']}"

        brackets = result.get("brackets", {})
        n_learned = sum(1 for b in brackets.values() if b.get("using") == "learned")
        assert n_learned == 5, f"Expected 5/5 learned brackets, got {n_learned}"
        print(f"[PASS] Step 3: Variance model recalculated — {n_learned}/5 brackets learned")

        # ── Step 4: Verify learned ratios are reasonable ─────────────────
        static_fallbacks = {"lt5k": 1.04, "5_65k": 0.64, "65_8k": 0.44, "8_10k": 0.35, "10k_plus": 0.30}
        ratios = result.get("ratios", {})
        print("\n  Bracket    | Static | Learned |  Δ%")
        print("  -----------|--------|---------|------")
        for bk in ["lt5k", "5_65k", "65_8k", "8_10k", "10k_plus"]:
            s = static_fallbacks[bk]
            l = ratios.get(bk, 0)
            delta = (l - s) / s * 100
            print(f"  {bk:10s} | {s:.3f}  | {l:.3f}  | {delta:+.1f}%")
            # Verify clamping: must be within 50%-200% of static
            assert l >= s * 0.5, f"{bk} learned ratio {l} below floor ({s * 0.5})"
            assert l <= s * 2.0, f"{bk} learned ratio {l} above cap ({s * 2.0})"
        print("[PASS] Step 4: All learned ratios within sanity bounds")

        # ── Step 5: Verify edge.py picks up learned ratios ──────────────
        from yak_core.edge import (
            _EMPIRICAL_VOL_RATIO,
            _STATIC_VOL_RATIO,
            compute_empirical_std,
            reload_variance_ratios,
        )

        reloaded = reload_variance_ratios()
        assert reloaded, "reload_variance_ratios returned False"

        # Verify at least one ratio differs from static
        any_different = any(
            abs(_EMPIRICAL_VOL_RATIO[k] - _STATIC_VOL_RATIO[k]) > 1e-6
            for k in _STATIC_VOL_RATIO
        )
        assert any_different, "No learned ratios differ from static after reload"
        print("[PASS] Step 5: edge.py loaded learned ratios")

        # ── Step 6: Verify compute_empirical_std uses learned values ─────
        test_proj = np.array([20.0, 25.0, 30.0, 35.0, 45.0])
        test_sal = np.array([4000, 5500, 7000, 9000, 11000])
        std_result = compute_empirical_std(test_proj, test_sal)
        assert len(std_result) == 5, f"Expected 5 std values, got {len(std_result)}"
        assert all(s > 0 for s in std_result), "All std values should be positive"

        # Compare to what static would produce
        static_std = test_proj * np.array([1.04, 0.64, 0.44, 0.35, 0.30])
        any_std_different = any(abs(std_result[i] - static_std[i]) > 0.01 for i in range(5))
        assert any_std_different, "compute_empirical_std should differ from static after learning"
        print("[PASS] Step 6: compute_empirical_std using learned values")

        # ── Step 7: Verify get_variance_model_status for UI ─────────────
        status_after = get_variance_model_status()
        assert status_after["status"] == "learned", f"Expected 'learned', got {status_after['status']}"
        assert status_after["n_learned_brackets"] == 5
        assert status_after["n_total_brackets"] == 5
        assert status_after["n_slates"] == 5
        assert "computed_at" in status_after
        assert "5/5 brackets learned" in status_after["message"]
        print("[PASS] Step 7: UI status correctly reports learned model")

        # ── Step 8: Verify JSON file persisted ──────────────────────────
        assert os.path.isfile(_RATIOS_FILE), "learned_ratios.json not found"
        with open(_RATIOS_FILE) as f:
            saved = json.load(f)
        assert "ratios" in saved
        assert "brackets" in saved
        assert "computed_at" in saved
        print("[PASS] Step 8: learned_ratios.json persisted correctly")

        print("\n" + "=" * 60)
        print("ALL 8 TESTS PASSED — Phase 2 variance pipeline is solid.")
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
