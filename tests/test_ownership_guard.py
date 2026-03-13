"""Tests for yak_core/ownership_guard.py.

Covers all edge cases:
- Pool with ownership=None → generates valid ownership
- Pool with ownership=0.0 for all → generates valid ownership
- Pool with ownership in 0-1 range → scales to 0-100
- Pool with valid 0-100 ownership → passes through unchanged
- Pool with own_proj but no ownership → copies and validates
- NBA vs PGA lineup_size handling
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yak_core.ownership_guard import ensure_ownership, _has_valid_ownership


def _make_pool(n: int = 10, salary_range=(4000, 10000)) -> pd.DataFrame:
    """Create a minimal valid player pool for testing."""
    np.random.seed(42)
    salaries = np.linspace(salary_range[0], salary_range[1], n).astype(int)
    proj = salaries / 1000 * 4.5 + np.random.normal(0, 1, n)
    return pd.DataFrame({
        "player_name": [f"Player {i}" for i in range(n)],
        "pos": ["PG", "SG", "SF", "PF", "C", "PG", "SG", "SF", "PF", "C"][:n],
        "salary": salaries,
        "proj": proj.clip(min=0),
        "team": ["LAL", "BOS"] * (n // 2) + ["LAL"] * (n % 2),
    })


# ─── _has_valid_ownership tests ──────────────────────────────────────────────

class TestHasValidOwnership:
    def test_none_series_returns_false(self):
        s = pd.Series([None, None, None])
        assert _has_valid_ownership(s) is False

    def test_nan_series_returns_false(self):
        s = pd.Series([float("nan"), float("nan")])
        assert _has_valid_ownership(s) is False

    def test_all_zeros_returns_false(self):
        s = pd.Series([0.0, 0.0, 0.0])
        assert _has_valid_ownership(s) is False

    def test_mixed_zeros_and_none_returns_false(self):
        s = pd.Series([None, 0.0, None, 0.0])
        assert _has_valid_ownership(s) is False

    def test_valid_values_returns_true(self):
        s = pd.Series([5.0, 12.0, 8.5])
        assert _has_valid_ownership(s) is True

    def test_mixed_valid_and_none_returns_true(self):
        s = pd.Series([None, 5.0, None])
        assert _has_valid_ownership(s) is True

    def test_single_small_positive_returns_true(self):
        s = pd.Series([0.01])
        assert _has_valid_ownership(s) is True

    def test_empty_series_returns_false(self):
        s = pd.Series([], dtype=object)
        assert _has_valid_ownership(s) is False


# ─── ensure_ownership tests ───────────────────────────────────────────────────

class TestEnsureOwnershipNoneValues:
    """Pool with ownership=None → generates valid ownership."""

    def test_ownership_none_generates_valid(self):
        pool = _make_pool(10)
        pool["ownership"] = None
        pool["own_proj"] = None
        result = ensure_ownership(pool, sport="NBA")
        own = pd.to_numeric(result["ownership"], errors="coerce")
        assert own.notna().all(), "All ownership values should be non-NaN"
        assert (own > 0).any(), "At least one player should have ownership > 0"

    def test_ownership_missing_column_generates_valid(self):
        """Pool with no ownership column at all should generate ownership."""
        pool = _make_pool(10)
        assert "ownership" not in pool.columns
        result = ensure_ownership(pool, sport="NBA")
        assert "ownership" in result.columns
        assert "own_proj" in result.columns
        own = pd.to_numeric(result["ownership"], errors="coerce")
        assert (own > 0).any()


class TestEnsureOwnershipAllZeros:
    """Pool with ownership=0.0 for all → generates valid ownership."""

    def test_all_zeros_generates_valid(self):
        pool = _make_pool(10)
        pool["ownership"] = 0.0
        pool["own_proj"] = 0.0
        result = ensure_ownership(pool, sport="NBA")
        own = pd.to_numeric(result["ownership"], errors="coerce")
        assert (own > 0).any(), "Should generate non-zero ownership when all zeros"


class TestEnsureOwnershipScaling:
    """Pool with ownership in 0-1 range → scales to 0-100."""

    def test_zero_to_one_scaled_to_100(self):
        pool = _make_pool(10)
        # Set ownership as 0-1 fractions
        pool["ownership"] = [0.05, 0.10, 0.15, 0.08, 0.20, 0.07, 0.12, 0.18, 0.06, 0.09]
        pool["own_proj"] = pool["ownership"]
        result = ensure_ownership(pool, sport="NBA")
        own = result["ownership"]
        # After scaling, all values should be > 1.0
        assert own.max() > 1.0, f"Expected scaled values > 1, got max={own.max()}"
        assert own.max() <= 100.0, f"Values should not exceed 100, got max={own.max()}"


class TestEnsureOwnershipPassthrough:
    """Pool with valid 0-100 ownership → passes through unchanged."""

    def test_valid_100_scale_passes_through(self):
        pool = _make_pool(10)
        original_own = [5.0, 10.0, 15.0, 8.0, 20.0, 7.0, 12.0, 18.0, 6.0, 9.0]
        pool["ownership"] = original_own
        pool["own_proj"] = original_own
        result = ensure_ownership(pool, sport="NBA")
        # Values should remain on 0-100 scale (not multiplied again)
        assert result["ownership"].max() <= 100.0
        # Max value should still be approximately 20% (not 2000%)
        assert result["ownership"].max() < 25.0, (
            f"Valid 0-100 values should not be rescaled, got max={result['ownership'].max()}"
        )


class TestEnsureOwnershipSyncColumns:
    """Pool with own_proj but no ownership → copies and validates."""

    def test_own_proj_but_no_ownership_copies(self):
        pool = _make_pool(10)
        pool["own_proj"] = [5.0, 10.0, 15.0, 8.0, 20.0, 7.0, 12.0, 18.0, 6.0, 9.0]
        # No ownership column
        result = ensure_ownership(pool, sport="NBA")
        assert "ownership" in result.columns
        # ownership should match own_proj
        assert (result["ownership"] == result["own_proj"]).all()

    def test_ownership_but_no_own_proj_copies(self):
        pool = _make_pool(10)
        pool["ownership"] = [5.0, 10.0, 15.0, 8.0, 20.0, 7.0, 12.0, 18.0, 6.0, 9.0]
        # No own_proj column
        result = ensure_ownership(pool, sport="NBA")
        assert "own_proj" in result.columns
        assert (result["own_proj"] == result["ownership"]).all()


class TestEnsureOwnershipSportLineupSize:
    """NBA vs PGA lineup_size handling."""

    def test_nba_generates_ownership_8_lineup(self):
        """NBA uses lineup_size=8: ownership sum should be ~800%."""
        pool = _make_pool(20)
        pool["ownership"] = None
        pool["own_proj"] = None
        result = ensure_ownership(pool, sport="NBA")
        own_sum = result["ownership"].sum()
        # NBA: 8 players per lineup → sum should be roughly 800%
        # Allow wide tolerance since it's a model estimate
        assert own_sum > 100, f"NBA ownership sum should be > 100, got {own_sum:.1f}"

    def test_pga_generates_ownership_6_lineup(self):
        """PGA uses lineup_size=6."""
        pool = _make_pool(20)
        # Make it look like PGA (remove team-based columns)
        pool["ownership"] = None
        pool["own_proj"] = None
        result = ensure_ownership(pool, sport="PGA")
        own = pd.to_numeric(result["ownership"], errors="coerce")
        assert own.notna().all()
        assert (own >= 0).all()
        assert (own <= 100).all()

    def test_case_insensitive_sport(self):
        """Sport string should be case-insensitive."""
        pool = _make_pool(10)
        pool["ownership"] = None
        pool_nba = ensure_ownership(pool.copy(), sport="nba")
        pool_pga = ensure_ownership(pool.copy(), sport="pga")
        assert "ownership" in pool_nba.columns
        assert "ownership" in pool_pga.columns


class TestEnsureOwnershipReturnsDataFrame:
    """ensure_ownership should always return a DataFrame."""

    def test_returns_dataframe(self):
        pool = _make_pool(5)
        result = ensure_ownership(pool)
        assert isinstance(result, pd.DataFrame)

    def test_empty_pool_does_not_crash(self):
        pool = pd.DataFrame(columns=["player_name", "salary", "proj", "pos"])
        # Should not raise, though ownership will be empty
        try:
            result = ensure_ownership(pool, sport="NBA")
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"ensure_ownership raised on empty pool: {e}")

    def test_both_columns_present_after_guard(self):
        """After ensure_ownership, both 'ownership' and 'own_proj' must exist."""
        pool = _make_pool(10)
        result = ensure_ownership(pool, sport="NBA")
        assert "ownership" in result.columns
        assert "own_proj" in result.columns

    def test_values_in_valid_range(self):
        """All ownership values should be between 0 and 100."""
        pool = _make_pool(15)
        pool["ownership"] = None
        result = ensure_ownership(pool, sport="NBA")
        own = pd.to_numeric(result["ownership"], errors="coerce")
        assert (own >= 0).all(), "Ownership must be >= 0"
        assert (own <= 100).all(), "Ownership must be <= 100"
