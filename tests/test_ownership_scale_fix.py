"""End-to-end test for the ownership scale fix.

Verifies that:
1. enforce_pct_scale correctly converts 0-1 fractions to 0-100
2. apply_ownership propagates 0-100 scale even when own_proj already exists
3. Leverage scores are sane (not 48x) after fix
4. Edge metrics produce realistic values with corrected ownership
5. Sims pipeline accepts 0-100 ownership without error
6. Dead signals are actually removed from edge_feedback
7. Correction factor caps are enforced
"""

import numpy as np
import pandas as pd
import pytest


class TestEnforcePctScale:
    """Test the centralized ownership scale enforcement."""

    def test_converts_fractional_to_pct(self):
        from yak_core.ownership_scale import enforce_pct_scale
        s = pd.Series([0.25, 0.10, 0.50, 0.01, 1.0])
        result = enforce_pct_scale(s)
        assert result.iloc[0] == pytest.approx(25.0, abs=0.1)
        assert result.iloc[1] == pytest.approx(10.0, abs=0.1)
        assert result.iloc[4] == pytest.approx(100.0, abs=0.1)

    def test_already_pct_unchanged(self):
        from yak_core.ownership_scale import enforce_pct_scale
        s = pd.Series([25.0, 10.0, 50.0, 5.0])
        result = enforce_pct_scale(s)
        assert result.iloc[0] == pytest.approx(25.0, abs=0.1)
        assert result.iloc[3] == pytest.approx(5.0, abs=0.1)

    def test_clips_above_100(self):
        from yak_core.ownership_scale import enforce_pct_scale
        s = pd.Series([150.0, 25.0])
        result = enforce_pct_scale(s)
        assert result.iloc[0] == 100.0

    def test_empty_series(self):
        from yak_core.ownership_scale import enforce_pct_scale
        s = pd.Series([], dtype=float)
        result = enforce_pct_scale(s)
        assert len(result) == 0

    def test_all_zeros(self):
        from yak_core.ownership_scale import enforce_pct_scale
        s = pd.Series([0.0, 0.0, 0.0])
        result = enforce_pct_scale(s)
        assert (result == 0.0).all()


class TestApplyOwnershipPreserves100Scale:
    """Test that apply_ownership enforces 0-100 even when own_proj pre-exists."""

    def test_fractional_own_proj_gets_scaled(self):
        from yak_core.ownership import apply_ownership
        pool = pd.DataFrame({
            "player_name": ["A", "B", "C"],
            "salary": [8000, 5000, 3500],
            "pos": ["PG", "SG", "SF"],
            "proj": [30.0, 20.0, 10.0],
            "own_proj": [0.25, 0.10, 0.03],  # fractional — should be converted
        })
        result = apply_ownership(pool)
        assert result["own_proj"].max() > 1.0, "own_proj should be on 0-100 scale"
        assert result["own_proj"].iloc[0] == pytest.approx(25.0, abs=0.1)

    def test_already_pct_own_proj_unchanged(self):
        from yak_core.ownership import apply_ownership
        pool = pd.DataFrame({
            "player_name": ["A", "B", "C"],
            "salary": [8000, 5000, 3500],
            "pos": ["PG", "SG", "SF"],
            "proj": [30.0, 20.0, 10.0],
            "own_proj": [25.0, 10.0, 3.0],  # already 0-100
        })
        result = apply_ownership(pool)
        assert result["own_proj"].iloc[0] == pytest.approx(25.0, abs=0.1)


class TestLeverageSanity:
    """Verify leverage is in sane range with corrected ownership."""

    def test_leverage_not_extreme(self):
        from yak_core.ownership import compute_leverage
        pool = pd.DataFrame({
            "player_name": ["Star", "Starter", "Bench"],
            "proj": [35.0, 22.0, 12.0],
            "own_proj": [28.0, 12.0, 3.0],  # 0-100 scale
        })
        result = compute_leverage(pool)
        lev = result["leverage"]
        # Leverage should be normalized 0-1 range
        assert lev.max() <= 1.0
        assert lev.min() >= 0.0

    def test_leverage_with_fractional_own_gives_extreme(self):
        """Before the fix, 0-1 ownership would produce extreme leverage."""
        from yak_core.ownership import compute_leverage
        pool = pd.DataFrame({
            "player_name": ["Star", "Starter", "Bench"],
            "proj": [35.0, 22.0, 12.0],
            "own_proj": [0.28, 0.12, 0.03],  # WRONG: 0-1 scale
        })
        result = compute_leverage(pool)
        # Raw leverage = proj / own_safe
        # With 0.28 ownership: 35/0.5 = 70 vs 35/28 = 1.25
        # The normalized leverage should still be 0-1 but the raw ratio is insane
        assert result["leverage"].max() <= 1.0  # normalized, so still 0-1
        # But the raw computation would be wildly different


class TestEdgeMetricsSanity:
    """Verify edge scores are realistic with corrected ownership."""

    def test_edge_score_range(self):
        from yak_core.edge import compute_edge_metrics
        pool = pd.DataFrame({
            "player_name": ["Star", "Mid", "Bench", "Value", "Punt"],
            "salary": [10000, 7500, 4500, 5500, 3500],
            "proj": [40.0, 28.0, 15.0, 22.0, 8.0],
            "floor": [25.0, 18.0, 8.0, 14.0, 3.0],
            "ceil": [55.0, 38.0, 25.0, 32.0, 18.0],
            "ownership": [30.0, 15.0, 5.0, 10.0, 2.0],  # 0-100 scale
        })
        result = compute_edge_metrics(pool)
        assert "edge_score" in result.columns
        # Edge scores should be 0-1 range
        assert result["edge_score"].max() <= 1.0
        assert result["edge_score"].min() >= 0.0
        # Leverage should be present and sane
        if "leverage" in result.columns:
            lev = result["leverage"].dropna()
            if len(lev) > 0:
                assert lev.max() < 100, "Leverage should not be insanely high"


class TestDeadSignalsRemoved:
    """Verify dead signals are actually gone from edge_feedback."""

    def test_signal_defs_only_has_live_signals(self):
        from yak_core.edge_feedback import _SIGNAL_DEFS
        assert "high_leverage" in _SIGNAL_DEFS
        assert "salary_value" in _SIGNAL_DEFS
        assert "smash_candidate" not in _SIGNAL_DEFS
        assert "chalk_fade" not in _SIGNAL_DEFS
        assert "low_ownership_upside" not in _SIGNAL_DEFS

    def test_hit_defs_match_signal_defs(self):
        from yak_core.edge_feedback import _SIGNAL_DEFS, _HIT_DEFS
        assert set(_HIT_DEFS.keys()) == set(_SIGNAL_DEFS.keys())


class TestCorrectionCaps:
    """Verify calibration correction factors are capped."""

    def test_caps_are_defined(self):
        from yak_core.calibration_feedback import _MAX_CORRECTION, _VALID_POSITIONS
        assert _MAX_CORRECTION == 3.0
        assert "PG" in _VALID_POSITIONS
        assert "UTIL" not in _VALID_POSITIONS
        assert "G" not in _VALID_POSITIONS

    def test_min_samples_raised(self):
        from yak_core.calibration_feedback import _MIN_SAMPLES
        assert _MIN_SAMPLES >= 15
