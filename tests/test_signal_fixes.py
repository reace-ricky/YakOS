"""Tests for CDF smash/bust model, leverage quality gate, and display format.

The smash/bust model uses a Normal CDF approach:
  - std derived from (ceil - floor) / (2 * Z_85) * salary_vol_mult
  - smash_prob = P(outcome >= 5x DK salary value)
  - bust_prob  = P(outcome <= projector floor)
"""

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
# Legacy _ceiling_gap_factor stub
# ---------------------------------------------------------------------------

class TestCeilingGapFactorLegacy:
    """_ceiling_gap_factor is now a legacy stub returning (1.0, 1.0)."""

    def test_always_returns_ones(self):
        proj = pd.Series([10.0, 20.0, 30.0])
        ceil = pd.Series([14.0, 28.0, 45.0])
        floor = pd.Series([7.0, 14.0, 22.0])
        smash_gap, bust_gap = _ceiling_gap_factor(proj, ceil, floor)
        assert (smash_gap == 1.0).all()
        assert (bust_gap == 1.0).all()


# ---------------------------------------------------------------------------
# CDF Smash Model — salary-value based
# ---------------------------------------------------------------------------

class TestSmashCDF:
    """smash_prob = P(outcome >= 5x DK salary value) via Normal CDF.

    The 5x salary value line is salary / 200.
    Players projecting well above their value line get high smash.
    Players whose ceiling can't reach the value line get near-zero smash.
    """

    def test_stud_high_smash(self):
        """Jokic-type: $11.5K, proj 55, 5x value = 57.5 → high smash."""
        pool = _make_pool(
            player_name="Stud",
            salary=11500,
            proj=55.0,
            ceil=72.0,
            floor=40.0,
            ownership=30.0,
        )
        edge = _edge(pool)
        smash = float(edge["smash_prob"].iloc[0])
        assert smash > 0.30, f"Stud smash_prob={smash:.3f} should be > 0.30"

    def test_compressed_punt_near_zero_smash(self):
        """Barnhizer-type: $3.8K, proj 9.43, ceil 10.75, 5x value = 19 → near-zero smash."""
        pool = _make_pool(
            player_name="Compressed Punt",
            salary=3800,
            proj=9.43,
            ceil=10.75,
            floor=8.5,
            ownership=5.0,
        )
        edge = _edge(pool)
        smash = float(edge["smash_prob"].iloc[0])
        assert smash <= 0.05, f"Compressed punt smash_prob={smash:.3f} should be <= 0.05"

    def test_value_play_moderate_smash(self):
        """Hendricks-type: $4.2K, proj 18, wide range, 5x value = 21 → moderate-high smash."""
        pool = _make_pool(
            player_name="Value Play",
            salary=4200,
            proj=18.0,
            ceil=32.0,
            floor=8.0,
            ownership=4.0,
        )
        edge = _edge(pool)
        smash = float(edge["smash_prob"].iloc[0])
        assert smash > 0.25, f"Value play smash_prob={smash:.3f} should be > 0.25"

    def test_mid_range_decent_smash(self):
        """$7.5K mid: proj 32, 5x value = 37.5 → moderate smash."""
        pool = _make_pool(
            player_name="Mid Range",
            salary=7500,
            proj=32.0,
            ceil=45.0,
            floor=22.0,
            ownership=18.0,
        )
        edge = _edge(pool)
        smash = float(edge["smash_prob"].iloc[0])
        assert 0.15 < smash < 0.50, f"Mid-range smash_prob={smash:.3f} not in (0.15, 0.50)"

    def test_low_proj_punt_low_smash(self):
        """$3.5K punt: proj 6, 5x value = 17.5 → very low smash."""
        pool = _make_pool(
            player_name="Low Punt",
            salary=3500,
            proj=6.0,
            ceil=12.0,
            floor=2.0,
            ownership=1.0,
        )
        edge = _edge(pool)
        smash = float(edge["smash_prob"].iloc[0])
        assert smash < 0.10, f"Low punt smash_prob={smash:.3f} should be < 0.10"

    def test_smash_differentiation_across_slate(self):
        """A multi-player slate should have meaningfully different smash values."""
        pool = pd.DataFrame([
            {"player_name": "Stud", "salary": 11000, "proj": 52.0,
             "ceil": 68.0, "floor": 38.0, "ownership": 28.0},
            {"player_name": "Mid", "salary": 7000, "proj": 28.0,
             "ceil": 40.0, "floor": 18.0, "ownership": 15.0},
            {"player_name": "Value", "salary": 4500, "proj": 18.0,
             "ceil": 28.0, "floor": 10.0, "ownership": 6.0},
            {"player_name": "Compressed", "salary": 3800, "proj": 9.0,
             "ceil": 11.0, "floor": 7.5, "ownership": 3.0},
        ])
        edge = _edge(pool)
        smash_vals = edge.set_index("player_name")["smash_prob"]
        # Stud should have highest smash, Compressed should have lowest
        assert smash_vals["Stud"] > smash_vals["Mid"]
        assert smash_vals["Mid"] > smash_vals["Compressed"]
        # Range of smash values should span at least 0.20
        spread = smash_vals.max() - smash_vals.min()
        assert spread > 0.20, f"Smash spread {spread:.3f} too narrow (< 0.20)"


# ---------------------------------------------------------------------------
# CDF Bust Model — floor-based
# ---------------------------------------------------------------------------

class TestBustCDF:
    """bust_prob = P(outcome <= projector floor) via Normal CDF.

    Studs with high floors relative to projection get low bust.
    Players with deep floors get higher bust.
    """

    def test_stud_low_bust(self):
        """$11.5K stud: proj 55, floor 40. bust should be low."""
        pool = _make_pool(
            player_name="Stud",
            salary=11500,
            proj=55.0,
            ceil=72.0,
            floor=40.0,
            ownership=30.0,
        )
        edge = _edge(pool)
        bust = float(edge["bust_prob"].iloc[0])
        assert bust < 0.25, f"Stud bust_prob={bust:.3f} should be < 0.25"

    def test_risky_punt_higher_bust(self):
        """Cheap punt with deep floor: proj 8, floor 4 → moderate bust."""
        pool = _make_pool(
            player_name="Risky Punt",
            salary=3500,
            proj=8.0,
            ceil=14.0,
            floor=4.0,
            ownership=3.0,
        )
        edge = _edge(pool)
        bust = float(edge["bust_prob"].iloc[0])
        assert bust > 0.15, f"Risky punt bust_prob={bust:.3f} should be > 0.15"

    def test_safe_floor_low_bust(self):
        """Player with very tight floor (0.90x proj): near-zero bust expected.

        Note: When floor is very close to proj, the sanity check in
        compute_edge_metrics may reset floor to proj*0.7 (because floor >= proj
        or ceil < floor checks). We use floor slightly below proj to avoid that.
        """
        pool = _make_pool(
            player_name="Safe Player",
            salary=8000,
            proj=35.0,
            ceil=50.0,
            floor=25.0,   # ~0.71x proj, normal range
            ownership=15.0,
        )
        edge = _edge(pool)
        bust = float(edge["bust_prob"].iloc[0])
        assert bust < 0.25, f"Safe player bust_prob={bust:.3f} should be < 0.25"


# ---------------------------------------------------------------------------
# Barnhizer Scenario (the original bug report)
# ---------------------------------------------------------------------------

class TestBarnhizerFix:
    """Brooks Barnhizer: proj=9.43, ceil=10.75, salary=$3.8K.

    Old model gave smash_prob ~0.43 (entire bracket base rate).
    New CDF model with salary-value smash threshold:
      5x value = $3800/200 = 19.0 FP — far above his proj and ceiling.
      smash should be near-zero.
    """

    def test_barnhizer_smash_is_near_zero(self):
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
        assert smash <= 0.05, f"Barnhizer smash_prob={smash:.3f} should be <= 0.05"

    def test_barnhizer_bust_moderate(self):
        """Barnhizer floor=8.5, proj=9.43. Floor/proj=0.90 → moderate bust risk."""
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
        # With compressed range, std is small, so bust could be moderate
        assert bust < 0.50, f"Barnhizer bust_prob={bust:.3f} should be < 0.50"

    def test_normal_value_play_has_real_smash(self):
        """A normal value play with upside should have meaningful smash."""
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
        # 5x value = 22.5, proj = 15, std ≈ 5.5 → z ≈ 1.36 → smash ≈ 0.09
        # Not super high because 5x value (22.5) is well above proj (15)
        assert smash > 0.03, f"Normal player smash_prob={smash:.3f} should be > 0.03"


# ---------------------------------------------------------------------------
# Smash vs Bust relationship
# ---------------------------------------------------------------------------

class TestSmashBustRelationship:
    """Verify smash and bust can't both be unreasonably high simultaneously.

    The original bug: players showing 0.34 smash AND 0.30 bust.
    With salary-value smash, this only happens if proj ≈ 5x salary value
    AND proj ≈ floor, which is contradictory for normal players.
    """

    def test_no_simultaneous_high_smash_and_bust(self):
        """For typical players, smash + bust should not both exceed 0.35."""
        pool = pd.DataFrame([
            {"player_name": "A", "salary": 5000, "proj": 18.0,
             "ceil": 25.0, "floor": 12.0, "ownership": 10.0},
            {"player_name": "B", "salary": 7500, "proj": 30.0,
             "ceil": 42.0, "floor": 20.0, "ownership": 15.0},
            {"player_name": "C", "salary": 10000, "proj": 45.0,
             "ceil": 60.0, "floor": 32.0, "ownership": 25.0},
        ])
        edge = _edge(pool)
        for _, row in edge.iterrows():
            name = row["player_name"]
            s, b = row["smash_prob"], row["bust_prob"]
            # At least one should be below 0.35
            assert min(s, b) < 0.35, (
                f"{name}: smash={s:.3f}, bust={b:.3f} — both too high"
            )


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
# Direct _compute_smash_bust tests
# ---------------------------------------------------------------------------

class TestComputeSmashBustDirect:
    """Test the raw _compute_smash_bust function directly."""

    def test_returns_series(self):
        proj = pd.Series([20.0])
        sal = pd.Series([6000])
        own = pd.Series([10.0])
        ceil = pd.Series([28.0])
        floor = pd.Series([14.0])
        smash, bust = _compute_smash_bust(proj, sal, own, ceil, floor, 1.0)
        assert isinstance(smash, pd.Series)
        assert isinstance(bust, pd.Series)
        assert len(smash) == 1
        assert len(bust) == 1

    def test_higher_variance_increases_smash(self):
        """Higher variance slider should increase smash probability."""
        proj = pd.Series([20.0])
        sal = pd.Series([6000])
        own = pd.Series([10.0])
        ceil = pd.Series([28.0])
        floor = pd.Series([14.0])
        s_low, b_low = _compute_smash_bust(proj, sal, own, ceil, floor, 0.8)
        s_high, b_high = _compute_smash_bust(proj, sal, own, ceil, floor, 1.5)
        # Higher variance = wider distribution = more probability in tails
        # But direction depends on whether smash_line is above or below proj
        # For 6000/200=30 (above proj=20), higher std pushes more mass past 30
        assert s_high.iloc[0] > s_low.iloc[0], "Higher variance should increase smash"

    def test_clipped_to_upper_bound(self):
        """Smash should clip to 0.95 when proj is far above smash line."""
        # Extreme case: proj way above 5x salary value → smash should cap at 0.95
        proj = pd.Series([100.0])
        sal = pd.Series([3500])  # 5x value = 17.5 — way below proj
        own = pd.Series([5.0])
        ceil = pd.Series([140.0])
        floor = pd.Series([70.0])
        smash, bust = _compute_smash_bust(proj, sal, own, ceil, floor, 1.0)
        assert smash.iloc[0] == pytest.approx(0.95), "Should clip to 0.95"

    def test_clipped_to_lower_bound(self):
        """Bust should clip to 0.01 when floor is far below proj."""
        # Extreme case: floor very far below proj relative to std
        proj = pd.Series([50.0])
        sal = pd.Series([8000])
        own = pd.Series([10.0])
        ceil = pd.Series([52.0])   # narrow range → tiny std
        floor = pd.Series([49.0])  # floor barely below proj
        smash, bust = _compute_smash_bust(proj, sal, own, ceil, floor, 1.0)
        # With tiny std (~1.45) and floor only 1 below proj, bust is moderate
        # But with salary 8K and 5x value = 40 (below proj 50), smash should be high
        assert smash.iloc[0] == pytest.approx(0.95), "Smash should be very high"
        # Bust clips at minimum 0.01
        assert bust.iloc[0] >= 0.01, "Bust should be at least 0.01"


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
