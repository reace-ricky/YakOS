"""Tests for compute_player_anomaly_table — per-player sim anomaly / leverage."""

import numpy as np
import pandas as pd
import pytest
from yak_core.sims import compute_player_anomaly_table


def _make_pool(n: int = 5) -> pd.DataFrame:
    """Minimal pool with player_name, proj, salary, own%, ceil, floor."""
    players = [
        ("LeBron James", 45.0, 8800, 25.0, 58.5, 31.5),
        ("Stephen Curry", 38.0, 7800, 5.0, 49.4, 26.6),
        ("Nikola Jokic", 52.0, 10500, 40.0, 67.6, 36.4),
        ("Kevin Durant", 40.0, 8200, 15.0, 52.0, 28.0),
        ("Luka Doncic", 48.0, 9600, 30.0, 62.4, 33.6),
    ]
    return pd.DataFrame(
        players[:n],
        columns=["player_name", "proj", "salary", "own%", "ceil", "floor"],
    )


def _make_lineups(pool: pd.DataFrame) -> pd.DataFrame:
    """Minimal lineup DataFrame referencing all pool players."""
    rows = [
        {"player_name": name, "lineup_index": 1}
        for name in pool["player_name"].tolist()
    ]
    return pd.DataFrame(rows)


class TestComputePlayerAnomalyTableReturnShape:
    def test_returns_dataframe(self):
        result = compute_player_anomaly_table(_make_pool(), _make_lineups(_make_pool()))
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self):
        df = compute_player_anomaly_table(_make_pool(), _make_lineups(_make_pool()))
        for col in ("Player", "Proj", "Salary", "Own%", "Smash%", "Bust%",
                    "Leverage Score", "Value Trap", "Flag"):
            assert col in df.columns, f"missing column: {col}"

    def test_row_count_matches_lineup_players(self):
        pool = _make_pool()
        lu = _make_lineups(pool)
        df = compute_player_anomaly_table(pool, lu)
        assert len(df) == len(pool)

    def test_sorted_by_leverage_score_descending(self):
        df = compute_player_anomaly_table(_make_pool(), _make_lineups(_make_pool()))
        assert (df["Leverage Score"].diff().dropna() <= 0).all()


class TestComputePlayerAnomalyTableEdgeCases:
    def test_empty_pool_returns_empty(self):
        result = compute_player_anomaly_table(pd.DataFrame(), _make_lineups(_make_pool()))
        assert result.empty

    def test_empty_lineup_returns_empty(self):
        result = compute_player_anomaly_table(_make_pool(), pd.DataFrame())
        assert result.empty

    def test_empty_pool_still_sims_when_lineup_has_proj(self):
        """When pool is empty but lineup has proj data, sims should still run."""
        lu = pd.DataFrame([{
            "player_name": "TestPlayer", "lineup_index": 0, "proj": 35.0,
            "salary": 7000, "own%": 10.0, "ceil": 50.0, "floor": 20.0,
        }])
        df = compute_player_anomaly_table(pd.DataFrame(), lu)
        assert len(df) == 1
        assert df.iloc[0]["Player"] == "TestPlayer"
        assert 0.0 <= df.iloc[0]["Smash%"] <= 100.0
        assert 0.0 <= df.iloc[0]["Bust%"] <= 100.0
        assert df.iloc[0]["Proj"] == 35.0
        assert df.iloc[0]["Salary"] == 7000

    def test_accepts_name_column_in_pool(self):
        pool = _make_pool().rename(columns={"player_name": "name"})
        lu = _make_lineups(_make_pool())
        df = compute_player_anomaly_table(pool, lu)
        assert not df.empty

    def test_accepts_name_column_in_lineup(self):
        pool = _make_pool()
        lu = _make_lineups(pool).rename(columns={"player_name": "name"})
        df = compute_player_anomaly_table(pool, lu)
        assert not df.empty

    def test_players_not_in_lineup_excluded(self):
        pool = _make_pool()
        # Only include first 3 players in the lineups
        lu = _make_lineups(pool.head(3))
        df = compute_player_anomaly_table(pool, lu)
        assert len(df) == 3

    def test_pool_player_not_in_lineup_never_simulated(self):
        """Players that are in the pool but NOT in any lineup must not appear in the result."""
        pool = _make_pool()  # 5 players
        # Lineup contains only 2 of the 5 pool players
        lu = pd.DataFrame([
            {"player_name": "LeBron James", "lineup_index": 0},
            {"player_name": "Stephen Curry", "lineup_index": 0},
        ])
        df = compute_player_anomaly_table(pool, lu)
        assert len(df) == 2
        assert set(df["Player"]) == {"LeBron James", "Stephen Curry"}
        # Verify pool-only players are absent
        assert "Nikola Jokic" not in df["Player"].values
        assert "Kevin Durant" not in df["Player"].values
        assert "Luka Doncic" not in df["Player"].values

    def test_pool_without_salary_column(self):
        pool = _make_pool().drop(columns=["salary"])
        lu = _make_lineups(_make_pool())
        df = compute_player_anomaly_table(pool, lu)
        assert not df.empty
        assert (df["Salary"] == 0).all()

    def test_pool_without_own_pct_column(self):
        pool = _make_pool().drop(columns=["own%"])
        lu = _make_lineups(_make_pool())
        df = compute_player_anomaly_table(pool, lu)
        assert not df.empty
        assert (df["Own%"] == 0).all()

    def test_proj_own_column_used_when_own_pct_absent(self):
        """Pool with proj_own (not own%) should still populate Own% via column normalisation."""
        pool = _make_pool().drop(columns=["own%"])
        pool["proj_own"] = [15.0, 5.0, 40.0, 12.0, 25.0]
        lu = _make_lineups(_make_pool())
        df = compute_player_anomaly_table(pool, lu)
        assert not df.empty
        # At least one player should have non-zero ownership from proj_own
        assert (df["Own%"] > 0).any()

    def test_proj_own_leverages_low_ownership_player(self):
        """Player with low proj_own should have higher leverage than high-proj_own player."""
        pool = pd.DataFrame([
            {"player_name": "LowOwn", "proj": 35.0, "salary": 6500, "proj_own": 2.0},
            {"player_name": "HighOwn", "proj": 35.0, "salary": 8500, "proj_own": 30.0},
        ])
        lu = pd.DataFrame([
            {"player_name": "LowOwn", "lineup_index": 1},
            {"player_name": "HighOwn", "lineup_index": 1},
        ])
        df = compute_player_anomaly_table(pool, lu, n_sims=500)
        low_own_row = df[df["Player"] == "LowOwn"].iloc[0]
        high_own_row = df[df["Player"] == "HighOwn"].iloc[0]
        assert low_own_row["Leverage Score"] > high_own_row["Leverage Score"]


class TestComputePlayerAnomalyTableMetrics:
    def test_smash_pct_between_0_and_100(self):
        df = compute_player_anomaly_table(_make_pool(), _make_lineups(_make_pool()))
        assert (df["Smash%"] >= 0.0).all() and (df["Smash%"] <= 100.0).all()

    def test_bust_pct_between_0_and_100(self):
        df = compute_player_anomaly_table(_make_pool(), _make_lineups(_make_pool()))
        assert (df["Bust%"] >= 0.0).all() and (df["Bust%"] <= 100.0).all()

    def test_leverage_score_non_negative(self):
        df = compute_player_anomaly_table(_make_pool(), _make_lineups(_make_pool()))
        assert (df["Leverage Score"] >= 0.0).all()

    def test_high_leverage_flag_when_score_gt_3(self):
        """Low own% player should have high leverage score and get the flag."""
        pool = pd.DataFrame([
            {"player_name": "LowOwn", "proj": 40.0, "salary": 6000, "own%": 1.0},
        ])
        lu = pd.DataFrame([{"player_name": "LowOwn", "lineup_index": 1}])
        df = compute_player_anomaly_table(
            pool, lu, n_sims=500,
            cal_knobs={"ceiling_boost": 2.0, "smash_threshold": 1.1},
        )
        assert df.iloc[0]["Flag"] == "🔥 HIGH LEVERAGE"

    def test_no_high_leverage_flag_when_score_le_3(self):
        """High own% player should not get the flag."""
        pool = pd.DataFrame([
            {"player_name": "HighOwn", "proj": 40.0, "salary": 8000, "own%": 60.0},
        ])
        lu = pd.DataFrame([{"player_name": "HighOwn", "lineup_index": 1}])
        df = compute_player_anomaly_table(pool, lu, n_sims=200)
        assert df.iloc[0]["Flag"] == ""

    def test_value_trap_flag_when_bust_and_high_salary(self):
        """High-salary player with very high bust rate gets Value Trap flag."""
        pool = pd.DataFrame([
            {"player_name": "ValueTrap", "proj": 40.0, "salary": 10000, "own%": 30.0},
            {"player_name": "CheapBust", "proj": 10.0, "salary": 3000, "own%": 5.0},
        ])
        lu = pd.DataFrame([
            {"player_name": "ValueTrap", "lineup_index": 1},
            {"player_name": "CheapBust", "lineup_index": 1},
        ])
        # bust_threshold=0.99 means bust if outcome < 0.99*proj; with default floor_dampen=1.0
        # roughly ~50% of outcomes fall below proj so Bust% should be substantially > 40%
        df = compute_player_anomaly_table(
            pool, lu, n_sims=1000,
            cal_knobs={"bust_threshold": 0.99},
        )
        trap_row = df[df["Player"] == "ValueTrap"].iloc[0]
        cheap_row = df[df["Player"] == "CheapBust"].iloc[0]
        # Both should have high Bust% (almost all outcomes below 0.99*proj qualify)
        assert trap_row["Bust%"] > 40.0
        assert cheap_row["Bust%"] > 40.0
        # ValueTrap has higher salary (> median), so it should be the value trap
        assert trap_row["Value Trap"] == True  # noqa: E712  (np.True_ != True with `is`)
        # CheapBust has lower salary (at/below median), should NOT be flagged
        assert trap_row["Value Trap"] != cheap_row["Value Trap"]

    def test_cal_knobs_ceiling_boost_increases_smash(self):
        """ceiling_boost > 1 should increase Smash% when an explicit ratio threshold is used."""
        pool = pd.DataFrame([{"player_name": "P1", "proj": 30.0, "salary": 7000, "own%": 20.0}])
        lu = pd.DataFrame([{"player_name": "P1", "lineup_index": 1}])
        # Use an explicit ratio threshold so ceiling_boost affects the smash probability
        baseline = compute_player_anomaly_table(
            pool, lu, n_sims=1000, cal_knobs={"ceiling_boost": 1.0, "smash_threshold": 1.2}
        )
        boosted = compute_player_anomaly_table(
            pool, lu, n_sims=1000, cal_knobs={"ceiling_boost": 2.0, "smash_threshold": 1.2}
        )
        assert boosted.iloc[0]["Smash%"] >= baseline.iloc[0]["Smash%"]

    def test_cal_knobs_smash_threshold_reduces_smash(self):
        """Higher smash_threshold means fewer outcomes qualify as smash."""
        pool = pd.DataFrame([{"player_name": "P1", "proj": 30.0, "salary": 7000, "own%": 20.0}])
        lu = pd.DataFrame([{"player_name": "P1", "lineup_index": 1}])
        low_thr = compute_player_anomaly_table(pool, lu, n_sims=1000, cal_knobs={"smash_threshold": 1.1})
        high_thr = compute_player_anomaly_table(pool, lu, n_sims=1000, cal_knobs={"smash_threshold": 1.8})
        assert low_thr.iloc[0]["Smash%"] >= high_thr.iloc[0]["Smash%"]

    def test_cal_knobs_bust_threshold_affects_bust(self):
        """Higher bust_threshold means more outcomes qualify as busts."""
        pool = pd.DataFrame([{"player_name": "P1", "proj": 30.0, "salary": 7000, "own%": 20.0}])
        lu = pd.DataFrame([{"player_name": "P1", "lineup_index": 1}])
        low_thr = compute_player_anomaly_table(pool, lu, n_sims=1000, cal_knobs={"bust_threshold": 0.1})
        high_thr = compute_player_anomaly_table(pool, lu, n_sims=1000, cal_knobs={"bust_threshold": 0.9})
        assert high_thr.iloc[0]["Bust%"] >= low_thr.iloc[0]["Bust%"]

    def test_default_cal_knobs_when_none_provided(self):
        """Passing cal_knobs=None should not raise and should return results."""
        pool = _make_pool(3)
        lu = _make_lineups(pool)
        df = compute_player_anomaly_table(pool, lu, cal_knobs=None)
        assert not df.empty

    def test_missing_cal_knobs_keys_use_defaults(self):
        """Passing an empty cal_knobs dict should use all defaults."""
        pool = _make_pool(3)
        lu = _make_lineups(pool)
        df = compute_player_anomaly_table(pool, lu, cal_knobs={})
        assert not df.empty

    def test_default_smash_pct_near_ten_percent(self):
        """Default (percentile-based) smash threshold produces ~10% smash rate."""
        pool = pd.DataFrame([
            {"player_name": "P1", "proj": 30.0, "salary": 7000, "own%": 20.0,
             "ceil": 42.0, "floor": 18.0},
        ])
        lu = pd.DataFrame([{"player_name": "P1", "lineup_index": 1}])
        df = compute_player_anomaly_table(pool, lu, n_sims=2000, cal_knobs={})
        # p90 threshold → smash_pct should be very close to 10%
        smash = df.iloc[0]["Smash%"]
        assert 7.0 <= smash <= 13.0, f"expected ~10% smash, got {smash}"

    def test_default_bust_pct_near_thirty_percent(self):
        """Default (percentile-based) bust threshold produces ~30% bust rate."""
        pool = pd.DataFrame([
            {"player_name": "P1", "proj": 30.0, "salary": 7000, "own%": 20.0,
             "ceil": 42.0, "floor": 18.0},
        ])
        lu = pd.DataFrame([{"player_name": "P1", "lineup_index": 1}])
        df = compute_player_anomaly_table(pool, lu, n_sims=2000, cal_knobs={})
        # p30 threshold → bust_pct should be very close to 30%
        bust = df.iloc[0]["Bust%"]
        assert 26.0 <= bust <= 35.0, f"expected ~30% bust, got {bust}"
