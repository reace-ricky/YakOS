"""Tests for smash_prob ceiling-gap fix, bust_prob floor-gap fix, and leverage quality gate."""

import pandas as pd
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yak_core.edge import compute_edge_metrics, _ceiling_gap_factor, _compute_smash_bust


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(**overrides) -> pd.DataFrame:
    """Build a single-player pool with sane defaults."""
    row = {
        "player_name": "Test Player",
        "salary": 6000,
        "proj": 20.0,
        "ceil": 28.0,     # 1.4x proj — normal
        "floor": 14.0,    # 0.7x proj — normal
        "ownership": 15.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _edge(pool: pd.DataFrame) -> pd.DataFrame:
    return compute_edge_metrics(pool)


# ---------------------------------------------------------------------------
# Ceiling Gap Factor
# ---------------------------------------------------------------------------

class TestCeilingGapFactor:
    """Test the continuous _ceiling_gap_factor interpolation."""

    def test_very_compressed_ceiling(self):
        """ceil/proj 1.05 → smash_gap ≈ 0.125 (interpolated)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([10.5])   # 1.05x
        floor = pd.Series([7.0])
        smash_gap, _ = _ceiling_gap_factor(proj, ceil, floor)
        assert smash_gap.iloc[0] == pytest.approx(0.125, abs=0.01)

    def test_compressed_ceiling(self):
        """ceil/proj 1.15 → smash_gap ≈ 0.35 (interpolated)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([11.5])   # 1.15x
        floor = pd.Series([7.0])
        smash_gap, _ = _ceiling_gap_factor(proj, ceil, floor)
        assert smash_gap.iloc[0] == pytest.approx(0.35, abs=0.01)

    def test_below_normal_ceiling(self):
        """ceil/proj 1.25 → smash_gap ≈ 0.65 (interpolated)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([12.5])   # 1.25x
        floor = pd.Series([7.0])
        smash_gap, _ = _ceiling_gap_factor(proj, ceil, floor)
        assert smash_gap.iloc[0] == pytest.approx(0.65, abs=0.01)

    def test_normal_ceiling(self):
        """ceil/proj 1.40 → smash_gap = 1.00 (anchor point)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([14.0])   # 1.40x
        floor = pd.Series([7.0])
        smash_gap, _ = _ceiling_gap_factor(proj, ceil, floor)
        assert smash_gap.iloc[0] == pytest.approx(1.00)

    def test_above_normal_ceiling(self):
        """ceil/proj 1.50 → smash_gap = 1.10 (anchor point)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([15.0])   # 1.50x
        floor = pd.Series([7.0])
        smash_gap, _ = _ceiling_gap_factor(proj, ceil, floor)
        assert smash_gap.iloc[0] == pytest.approx(1.10)

    def test_huge_ceiling(self):
        """ceil/proj 2.00 → smash_gap = 1.25 (extrapolated max)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([20.0])   # 2.00x
        floor = pd.Series([7.0])
        smash_gap, _ = _ceiling_gap_factor(proj, ceil, floor)
        assert smash_gap.iloc[0] == pytest.approx(1.25)


class TestFloorGapFactor:
    """Test bust_gap continuous interpolation from _ceiling_gap_factor."""

    def test_very_safe_floor(self):
        """floor/proj 0.95 → bust_gap ≈ 0.05 (near-zero bust risk)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([14.0])
        floor = pd.Series([9.5])   # 0.95x
        _, bust_gap = _ceiling_gap_factor(proj, ceil, floor)
        assert bust_gap.iloc[0] == pytest.approx(0.05, abs=0.01)

    def test_safe_floor(self):
        """floor/proj 0.85 → bust_gap ≈ 0.325 (interpolated)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([14.0])
        floor = pd.Series([8.5])   # 0.85x
        _, bust_gap = _ceiling_gap_factor(proj, ceil, floor)
        assert bust_gap.iloc[0] == pytest.approx(0.325, abs=0.01)

    def test_normal_floor(self):
        """floor/proj 0.70 → bust_gap ≈ 0.85 (interpolated)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([14.0])
        floor = pd.Series([7.0])   # 0.70x
        _, bust_gap = _ceiling_gap_factor(proj, ceil, floor)
        assert bust_gap.iloc[0] == pytest.approx(0.85, abs=0.01)

    def test_risky_floor(self):
        """floor/proj 0.55 → bust_gap = 1.15 (anchor point)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([14.0])
        floor = pd.Series([5.5])   # 0.55x
        _, bust_gap = _ceiling_gap_factor(proj, ceil, floor)
        assert bust_gap.iloc[0] == pytest.approx(1.15)

    def test_extreme_bust_floor(self):
        """floor/proj 0.30 → bust_gap = 1.40 (max bust risk)."""
        proj = pd.Series([10.0])
        ceil = pd.Series([14.0])
        floor = pd.Series([3.0])   # 0.30x
        _, bust_gap = _ceiling_gap_factor(proj, ceil, floor)
        assert bust_gap.iloc[0] == pytest.approx(1.40)


# ---------------------------------------------------------------------------
# Barnhizer Scenario (the bug report)
# ---------------------------------------------------------------------------

class TestBarnhizerFix:
    """Brooks Barnhizer: proj=9.43, ceil=10.75, salary<$5K.

    Old model gave smash_prob ~0.43 (entire bracket base rate).
    New model should significantly dampen this because the ceiling
    is only 1.14x the projection — barely any upside room.
    """

    def test_barnhizer_smash_is_dampened(self):
        pool = _make_pool(
            player_name="Brooks Barnhizer",
            salary=3800,
            proj=9.43,
            ceil=10.75,
            floor=8.5,
            ownership=5.0,
        )
        edge = _edge(pool)
        smash = float(edge["smash_prob"].iloc[0])
        # Old value was ~0.43, new should be < 0.20
        assert smash < 0.20, f"Barnhizer smash_prob={smash:.3f} should be < 0.20"

    def test_barnhizer_bust_is_dampened(self):
        """Barnhizer's floor is 0.90x proj — very safe. bust_prob should drop."""
        pool = _make_pool(
            player_name="Brooks Barnhizer",
            salary=3800,
            proj=9.43,
            ceil=10.75,
            floor=8.5,
            ownership=5.0,
        )
        edge = _edge(pool)
        bust = float(edge["bust_prob"].iloc[0])
        # With floor/proj = 0.90, bust_gap = 0.10, so bust should be very low
        assert bust < 0.10, f"Barnhizer bust_prob={bust:.3f} should be < 0.10"

    def test_normal_value_play_unaffected(self):
        """A normal value play with standard ceiling ratio should be unaffected."""
        pool = _make_pool(
            player_name="Normal Value Guy",
            salary=4500,
            proj=15.0,
            ceil=21.0,    # 1.40x — normal
            floor=10.5,   # 0.70x — normal
            ownership=8.0,
        )
        edge = _edge(pool)
        smash = float(edge["smash_prob"].iloc[0])
        # Should still be near the bracket base rate (0.43 for <$5K, adjusted)
        assert smash > 0.30, f"Normal player smash_prob={smash:.3f} should be > 0.30"


# ---------------------------------------------------------------------------
# Leverage Quality Gate
# ---------------------------------------------------------------------------

class TestLeverageQualityGate:
    """Players with proj < 10 FP should get NaN leverage."""

    def test_low_proj_gets_nan_leverage(self):
        pool = _make_pool(
            player_name="Junk Punt",
            salary=3500,
            proj=5.0,
            ceil=7.0,
            floor=3.5,
            ownership=0.5,
        )
        edge = _edge(pool)
        lev = edge["leverage"].iloc[0]
        assert pd.isna(lev), f"Low-proj player should have NaN leverage, got {lev}"

    def test_above_threshold_gets_real_leverage(self):
        pool = _make_pool(
            player_name="Real Player",
            salary=5500,
            proj=18.0,
            ceil=25.0,
            floor=12.0,
            ownership=3.0,
        )
        edge = _edge(pool)
        lev = float(edge["leverage"].iloc[0])
        assert not pd.isna(lev), "Above-threshold player should have real leverage"
        assert lev > 0, f"Leverage should be positive, got {lev}"

    def test_low_ownership_still_nan_below_threshold(self):
        """Even with micro-ownership, low proj means no leverage."""
        pool = _make_pool(
            player_name="Micro-Own Punt",
            salary=3500,
            proj=4.0,
            ceil=5.6,
            floor=2.8,
            ownership=0.2,
        )
        edge = _edge(pool)
        lev = edge["leverage"].iloc[0]
        assert pd.isna(lev), f"Expected NaN leverage for proj=4.0, got {lev}"

    def test_stud_retains_leverage(self):
        """$10K stud with high projection should have leverage."""
        pool = _make_pool(
            player_name="Big Stud",
            salary=10500,
            proj=48.0,
            ceil=65.0,
            floor=35.0,
            ownership=28.0,
        )
        edge = _edge(pool)
        lev = float(edge["leverage"].iloc[0])
        assert not pd.isna(lev), "Stud should have leverage"
        assert lev > 0


# ---------------------------------------------------------------------------
# Multi-player slate: leverage normalisation
# ---------------------------------------------------------------------------

class TestLeverageNormalisation:
    """Verify that junk punts don't compress everyone else's leverage."""

    def test_junk_doesnt_inflate_max(self):
        """With the quality gate, the junk punt gets NaN and doesn't set the max."""
        pool = pd.DataFrame([
            {"player_name": "Junk Punt", "salary": 3500, "proj": 4.0,
             "ceil": 5.6, "floor": 2.8, "ownership": 0.3},
            {"player_name": "Real Contrarian", "salary": 5500, "proj": 18.0,
             "ceil": 25.0, "floor": 12.0, "ownership": 3.0},
            {"player_name": "Normal Mid", "salary": 7000, "proj": 25.0,
             "ceil": 35.0, "floor": 17.5, "ownership": 20.0},
        ])
        edge = _edge(pool)
        junk = edge[edge["player_name"] == "Junk Punt"]["leverage"].iloc[0]
        real = float(edge[edge["player_name"] == "Real Contrarian"]["leverage"].iloc[0])
        mid = float(edge[edge["player_name"] == "Normal Mid"]["leverage"].iloc[0])

        assert pd.isna(junk), "Junk punt should have NaN leverage"
        assert real > mid, "Real contrarian should have higher leverage than normal mid"


# ---------------------------------------------------------------------------
# Display Format
# ---------------------------------------------------------------------------

class TestDisplayFormat:
    """Test the shared display_format module."""

    def test_normalise_ownership_fractional(self):
        from yak_core.display_format import normalise_ownership
        s = pd.Series([0.25, 0.10, 0.50])
        result = normalise_ownership(s)
        assert result.iloc[0] == pytest.approx(25.0)
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(50.0)

    def test_normalise_ownership_already_pct(self):
        from yak_core.display_format import normalise_ownership
        s = pd.Series([25.0, 10.0, 50.0])
        result = normalise_ownership(s)
        assert result.iloc[0] == pytest.approx(25.0)
        assert result.iloc[1] == pytest.approx(10.0)

    def test_standard_player_format_keys(self):
        from yak_core.display_format import standard_player_format
        df = pd.DataFrame({
            "salary": [5000],
            "own_pct": [15.0],
            "proj": [20.0],
            "smash_prob": [0.30],
            "leverage": [1.5],
        })
        fmt = standard_player_format(df)
        assert fmt["salary"] == "${:,.0f}"
        assert fmt["own_pct"] == "{:.1f}%"
        assert fmt["proj"] == "{:.1f}"
        assert fmt["smash_prob"] == "{:.2f}"
        assert fmt["leverage"] == "{:.2f}"
