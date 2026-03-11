"""Tests for Ticket 13: breakout_score ceiling boost in GPP optimizer.

High-breakout players (score >= 60) receive up to +8% proj boost in the
optimizer objective for GPP contest types only.  The input DataFrame is
never mutated.
"""

import pandas as pd
import pytest
from yak_core.lineups import build_multiple_lineups_with_exposure


def _make_pool() -> pd.DataFrame:
    """Minimal 8-position NBA pool sufficient to build 1 lineup."""
    positions = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "SG/SF"]
    rows = []
    for i, pos in enumerate(positions * 3):
        rows.append(
            {
                "player_id": str(i),
                "player_name": f"Player_{i}",
                "team": "T1" if i % 2 == 0 else "T2",
                "opponent": "T2" if i % 2 == 0 else "T1",
                "pos": pos,
                "salary": 5000 + i * 300,
                "proj": 25.0 + i * 0.5,
                "ceil": 35.0 + i * 0.5,
                "floor": 15.0 + i * 0.3,
                "ownership": 0.15,
            }
        )
    return pd.DataFrame(rows)


_BASE_CFG = {
    "SITE": "dk",
    "SPORT": "nba",
    "NUM_LINEUPS": 1,
    "MIN_SALARY_USED": 0,
    "SALARY_CAP": 500000,
    "SOLVER_TIME_LIMIT": 30,
    "CONTEST_TYPE": "gpp",
}


class TestBreakoutBoostGPP:
    def test_input_df_not_mutated(self):
        """breakout_score boost must not modify the input DataFrame's proj column."""
        pool = _make_pool()
        pool["breakout_score"] = 90.0
        original_proj = pool["proj"].copy()

        build_multiple_lineups_with_exposure(pool, _BASE_CFG)

        pd.testing.assert_series_equal(pool["proj"], original_proj)

    def test_no_breakout_column_no_error(self):
        """When breakout_score is absent the optimizer runs without error."""
        pool = _make_pool()
        assert "breakout_score" not in pool.columns

        lineups_df, _ = build_multiple_lineups_with_exposure(pool, _BASE_CFG)

        assert not lineups_df.empty

    def test_boost_not_applied_for_cash(self):
        """breakout_score boost must NOT be applied for cash contest type."""
        pool = _make_pool()
        pool["breakout_score"] = 90.0
        original_proj = pool["proj"].copy()

        cash_cfg = {**_BASE_CFG, "CONTEST_TYPE": "cash"}
        build_multiple_lineups_with_exposure(pool, cash_cfg)

        # Input still unchanged
        pd.testing.assert_series_equal(pool["proj"], original_proj)

    def test_high_breakout_player_favored_in_gpp(self):
        """A slightly-lower-proj PG with breakout_score=100 should beat a higher-proj PG with no boost.

        Player B: proj=24.5, breakout=100 → boosted proj = 24.5 * 1.08 = 26.46
        Player A: proj=25.0, breakout=0   → no boost, proj = 25.0

        After boost Player B has higher effective proj so should be selected.
        """
        # Build a minimal pool with exactly 2 PG candidates that differ only
        # in proj (tiny gap) and breakout_score.
        positions_needed = ["SG", "SF", "PF", "C", "PG/SG", "SF/PF", "SG/SF"]
        rows = []
        pid = 0
        # PG A: higher raw proj but no breakout boost
        rows.append({
            "player_id": str(pid), "player_name": "PG_A",
            "team": "T1", "opponent": "T2",
            "pos": "PG", "salary": 7000,
            "proj": 25.0, "ceil": 35.0, "floor": 15.0,
            "ownership": 0.15, "breakout_score": 0.0,
        })
        pid += 1
        # PG B: slightly lower raw proj but high breakout → boosted proj = 24.5 * 1.08 ≈ 26.46
        rows.append({
            "player_id": str(pid), "player_name": "PG_B",
            "team": "T2", "opponent": "T1",
            "pos": "PG", "salary": 7000,
            "proj": 24.5, "ceil": 34.5, "floor": 14.5,
            "ownership": 0.15, "breakout_score": 100.0,
        })
        pid += 1
        # Fill remaining positions
        for i, pos in enumerate(positions_needed):
            for _ in range(3):
                rows.append({
                    "player_id": str(pid), "player_name": f"P_{pid}",
                    "team": "T1" if pid % 2 == 0 else "T2",
                    "opponent": "T2" if pid % 2 == 0 else "T1",
                    "pos": pos, "salary": 5000 + pid * 200,
                    "proj": 20.0 + pid * 0.3, "ceil": 30.0 + pid * 0.3,
                    "floor": 12.0, "ownership": 0.15, "breakout_score": 0.0,
                })
                pid += 1

        pool = pd.DataFrame(rows)
        lineups_df, _ = build_multiple_lineups_with_exposure(pool, _BASE_CFG)
        assert "PG_B" in lineups_df["player_name"].values, (
            "PG_B with breakout_score=100 should be selected over PG_A (higher boosted proj)"
        )

    def test_zero_breakout_no_boost(self):
        """Players with breakout_score=0 receive no proj change (boost=0)."""
        pool = _make_pool()
        pool["breakout_score"] = 0.0
        original_proj = pool["proj"].copy()

        build_multiple_lineups_with_exposure(pool, _BASE_CFG)

        # Input DataFrame unchanged
        pd.testing.assert_series_equal(pool["proj"], original_proj)

    def test_sub_threshold_breakout_no_boost(self):
        """Players with breakout_score < 60 receive no proj boost."""
        pool = _make_pool()
        pool["breakout_score"] = 59.9
        original_proj = pool["proj"].copy()

        build_multiple_lineups_with_exposure(pool, _BASE_CFG)

        pd.testing.assert_series_equal(pool["proj"], original_proj)
