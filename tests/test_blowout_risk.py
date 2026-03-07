"""Tests for yak_core.blowout_risk module."""

import pandas as pd
import pytest

from yak_core.blowout_risk import (
    DAMPENING,
    SPREAD_THRESHOLD,
    apply_blowout_cascade,
    compute_blowout_adjustment,
    _get_tier,
)


# ---------------------------------------------------------------------------
# Unit tests: _get_tier
# ---------------------------------------------------------------------------

class TestGetTier:
    def test_starter(self):
        assert _get_tier(32.0) == "starter"

    def test_starter_lite(self):
        assert _get_tier(25.0) == "starter_lite"

    def test_rotation(self):
        assert _get_tier(18.0) == "rotation"

    def test_low_rotation(self):
        assert _get_tier(10.0) == "low_rotation"

    def test_deep_bench(self):
        assert _get_tier(4.0) == "deep_bench"

    def test_zero_minutes(self):
        assert _get_tier(0.0) == "deep_bench"

    def test_boundary_28(self):
        assert _get_tier(28.0) == "starter"

    def test_boundary_22(self):
        assert _get_tier(22.0) == "starter_lite"

    def test_boundary_15(self):
        assert _get_tier(15.0) == "rotation"

    def test_boundary_8(self):
        assert _get_tier(8.0) == "low_rotation"


# ---------------------------------------------------------------------------
# Unit tests: compute_blowout_adjustment
# ---------------------------------------------------------------------------

class TestComputeBlowoutAdjustment:
    def test_below_threshold_returns_zero(self):
        adj = compute_blowout_adjustment(spread=5.0, proj_minutes=32.0)
        assert adj == 0.0

    def test_at_threshold_returns_zero(self):
        adj = compute_blowout_adjustment(spread=SPREAD_THRESHOLD, proj_minutes=32.0)
        assert adj == 0.0

    def test_starter_favorite_loses_minutes(self):
        adj = compute_blowout_adjustment(spread=15.0, proj_minutes=32.0, side="favorite")
        assert adj < 0, "Starters on the favored team should lose minutes"

    def test_starter_underdog_loses_minutes(self):
        adj = compute_blowout_adjustment(spread=15.0, proj_minutes=32.0, side="underdog")
        assert adj < 0, "Starters on the underdog should also lose minutes"

    def test_deep_bench_favorite_gains_minutes(self):
        adj = compute_blowout_adjustment(spread=15.0, proj_minutes=4.0, side="favorite")
        assert adj > 0, "Deep bench on favored team should gain minutes"

    def test_deep_bench_underdog_gains_minutes(self):
        adj = compute_blowout_adjustment(spread=15.0, proj_minutes=4.0, side="underdog")
        assert adj > 0, "Deep bench on underdog should gain minutes"

    def test_larger_spread_bigger_adjustment(self):
        adj_small = abs(compute_blowout_adjustment(spread=12.0, proj_minutes=32.0))
        adj_big = abs(compute_blowout_adjustment(spread=20.0, proj_minutes=32.0))
        assert adj_big > adj_small, "Larger spread should produce bigger adjustments"

    def test_capped_at_5_minutes(self):
        adj = compute_blowout_adjustment(spread=50.0, proj_minutes=32.0)
        assert abs(adj) <= 5.0, "Adjustment should be capped at 5 minutes"

    def test_dampening_applied(self):
        # Without dampening: (15-8) * 0.146 = 1.022
        # With dampening at 0.4: 1.022 * 0.4 = 0.409 → rounds to 0.4
        adj = compute_blowout_adjustment(spread=15.0, proj_minutes=32.0, side="favorite")
        # Should be dampened (less than raw slope would suggest)
        raw = (15.0 - SPREAD_THRESHOLD) * 0.146
        assert abs(adj) < raw, "Adjustment should be dampened"

    def test_negative_spread_treated_as_absolute(self):
        adj_pos = compute_blowout_adjustment(spread=15.0, proj_minutes=32.0)
        adj_neg = compute_blowout_adjustment(spread=-15.0, proj_minutes=32.0)
        assert adj_pos == adj_neg

    def test_rotation_player_small_adjustment(self):
        adj = compute_blowout_adjustment(spread=15.0, proj_minutes=18.0, side="favorite")
        # Rotation players should have smaller adjustments than starters
        starter_adj = compute_blowout_adjustment(spread=15.0, proj_minutes=32.0, side="favorite")
        assert abs(adj) < abs(starter_adj) or abs(adj) == abs(starter_adj)


# ---------------------------------------------------------------------------
# Integration tests: apply_blowout_cascade
# ---------------------------------------------------------------------------

def _make_pool():
    """Create a test pool with players at different minute tiers."""
    return pd.DataFrame([
        {"player_name": "Star A", "team": "LAL", "pos": "SF", "salary": 10000,
         "proj": 45.0, "proj_minutes": 34.0, "status": "Active"},
        {"player_name": "Starter B", "team": "LAL", "pos": "PG", "salary": 7000,
         "proj": 32.0, "proj_minutes": 30.0, "status": "Active"},
        {"player_name": "Rotation C", "team": "LAL", "pos": "SG", "salary": 5000,
         "proj": 22.0, "proj_minutes": 20.0, "status": "Active"},
        {"player_name": "Bench D", "team": "LAL", "pos": "PF", "salary": 3500,
         "proj": 12.0, "proj_minutes": 12.0, "status": "Active"},
        {"player_name": "Deep E", "team": "LAL", "pos": "C", "salary": 3000,
         "proj": 5.0, "proj_minutes": 5.0, "status": "Active"},
        # Opponent
        {"player_name": "Opp Star", "team": "BOS", "pos": "SF", "salary": 10000,
         "proj": 44.0, "proj_minutes": 33.0, "status": "Active"},
        {"player_name": "Opp Bench", "team": "BOS", "pos": "PG", "salary": 3000,
         "proj": 6.0, "proj_minutes": 6.0, "status": "Active"},
    ])


class TestApplyBlowoutCascade:
    def test_no_spreads_returns_unchanged(self):
        pool = _make_pool()
        result, report = apply_blowout_cascade(pool, {})
        assert report == []
        pd.testing.assert_frame_equal(result[["player_name", "proj"]], pool[["player_name", "proj"]])

    def test_small_spread_no_adjustments(self):
        pool = _make_pool()
        spreads = {"game1": {"favorite": "LAL", "underdog": "BOS", "spread": 5.0}}
        result, report = apply_blowout_cascade(pool, spreads)
        assert report == []

    def test_large_spread_adjusts_both_teams(self):
        pool = _make_pool()
        spreads = {"game1": {"favorite": "LAL", "underdog": "BOS", "spread": 15.0}}
        result, report = apply_blowout_cascade(pool, spreads)
        assert len(report) == 1
        assert report[0]["spread"] == 15.0
        # Should have adjustments for both teams
        teams = set(a["team"] for a in report[0]["adjustments"])
        assert "LAL" in teams
        assert "BOS" in teams

    def test_starters_lose_minutes(self):
        pool = _make_pool()
        spreads = {"game1": {"favorite": "LAL", "underdog": "BOS", "spread": 15.0}}
        result, report = apply_blowout_cascade(pool, spreads)
        star_a = result[result["player_name"] == "Star A"].iloc[0]
        assert star_a["blowout_min_adj"] < 0, "Star should lose minutes in blowout"

    def test_deep_bench_gains_minutes(self):
        pool = _make_pool()
        spreads = {"game1": {"favorite": "LAL", "underdog": "BOS", "spread": 15.0}}
        result, report = apply_blowout_cascade(pool, spreads)
        deep_e = result[result["player_name"] == "Deep E"].iloc[0]
        assert deep_e["blowout_min_adj"] > 0, "Deep bench should gain minutes in blowout"

    def test_fp_adjusted_proportionally(self):
        pool = _make_pool()
        spreads = {"game1": {"favorite": "LAL", "underdog": "BOS", "spread": 15.0}}
        result, report = apply_blowout_cascade(pool, spreads)
        star_a = result[result["player_name"] == "Star A"].iloc[0]
        # FP should decrease when minutes decrease
        assert star_a["blowout_fp_adj"] < 0

    def test_empty_pool_returns_empty(self):
        pool = pd.DataFrame()
        result, report = apply_blowout_cascade(pool, {"g": {"favorite": "LAL", "underdog": "BOS", "spread": 15}})
        assert report == []

    def test_none_pool_returns_none(self):
        result, report = apply_blowout_cascade(None, {"g": {"favorite": "LAL", "underdog": "BOS", "spread": 15}})
        assert report == []

    def test_original_minutes_preserved(self):
        pool = _make_pool()
        spreads = {"game1": {"favorite": "LAL", "underdog": "BOS", "spread": 15.0}}
        result, report = apply_blowout_cascade(pool, spreads)
        star_a = result[result["player_name"] == "Star A"].iloc[0]
        assert star_a["original_proj_minutes"] == 34.0

    def test_minutes_floor_at_zero(self):
        pool = pd.DataFrame([
            {"player_name": "Tiny", "team": "LAL", "pos": "PG", "salary": 3000,
             "proj": 1.0, "proj_minutes": 1.0, "status": "Active"},
        ])
        spreads = {"game1": {"favorite": "BOS", "underdog": "LAL", "spread": 25.0}}
        result, report = apply_blowout_cascade(pool, spreads)
        assert result.iloc[0]["proj_minutes"] >= 0
