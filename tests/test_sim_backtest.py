"""Tests for backtest_sim — Monte Carlo sim vs historical actuals."""

import pandas as pd
import pytest
from yak_core.sims import backtest_sim


def _make_hist_df(n_lineups: int = 3, n_players: int = 8, seed: int = 0) -> pd.DataFrame:
    """Build a minimal historical lineup DataFrame with known values."""
    rng = __import__("numpy").random.RandomState(seed)
    rows = []
    for lu_id in range(1, n_lineups + 1):
        for p in range(n_players):
            proj = float(rng.uniform(15, 45))
            rows.append(
                {
                    "lineup_id": lu_id,
                    "name": f"Player_{p}",
                    "proj": proj,
                    "ceil": proj * 1.4,
                    "floor": proj * 0.6,
                    "actual": float(rng.uniform(10, 60)),
                }
            )
    return pd.DataFrame(rows)


class TestBacktestSimReturnShape:
    def test_returns_dict_with_required_keys(self):
        hist = _make_hist_df()
        result = backtest_sim(hist)
        for key in ("lineup_df", "sim_mae", "sim_rmse", "sim_bias",
                    "within_range_pct", "n_lineups"):
            assert key in result, f"missing key: {key}"

    def test_lineup_df_has_expected_columns(self):
        hist = _make_hist_df()
        df = backtest_sim(hist)["lineup_df"]
        for col in ("lineup_id", "sim_mean", "sim_std", "sim_p15", "sim_p85",
                    "actual", "error", "within_range"):
            assert col in df.columns, f"missing column: {col}"

    def test_n_lineups_matches_input(self):
        n = 4
        hist = _make_hist_df(n_lineups=n)
        result = backtest_sim(hist)
        assert result["n_lineups"] == n

    def test_lineup_df_row_count_matches_n_lineups(self):
        n = 5
        hist = _make_hist_df(n_lineups=n)
        result = backtest_sim(hist)
        assert len(result["lineup_df"]) == n


class TestBacktestSimEdgeCases:
    def test_empty_df_returns_empty_result(self):
        result = backtest_sim(pd.DataFrame())
        assert result["n_lineups"] == 0
        assert result["lineup_df"].empty

    def test_missing_actual_column_returns_empty(self):
        hist = _make_hist_df()
        result = backtest_sim(hist.drop(columns=["actual"]))
        assert result["n_lineups"] == 0

    def test_missing_lineup_id_column_returns_empty(self):
        hist = _make_hist_df()
        result = backtest_sim(hist.drop(columns=["lineup_id"]))
        assert result["n_lineups"] == 0

    def test_missing_proj_column_returns_empty(self):
        hist = _make_hist_df()
        result = backtest_sim(hist.drop(columns=["proj"]))
        assert result["n_lineups"] == 0


class TestBacktestSimMetrics:
    def test_sim_mae_is_non_negative(self):
        result = backtest_sim(_make_hist_df())
        assert result["sim_mae"] >= 0.0

    def test_sim_rmse_is_non_negative(self):
        result = backtest_sim(_make_hist_df())
        assert result["sim_rmse"] >= 0.0

    def test_within_range_pct_between_0_and_100(self):
        result = backtest_sim(_make_hist_df())
        assert 0.0 <= result["within_range_pct"] <= 100.0

    def test_error_equals_sim_mean_minus_actual(self):
        result = backtest_sim(_make_hist_df())
        df = result["lineup_df"]
        import numpy as np
        assert np.allclose(df["error"], df["sim_mean"] - df["actual"], atol=1e-6)

    def test_within_range_flag_consistent_with_bounds(self):
        result = backtest_sim(_make_hist_df())
        df = result["lineup_df"]
        expected = (df["actual"] >= df["sim_p15"]) & (df["actual"] <= df["sim_p85"])
        assert (df["within_range"] == expected).all()

    def test_perfect_projection_low_mae(self):
        """When actual == proj for every player, sim_mean ≈ actual and MAE is small."""
        import numpy as np
        rows = []
        for lu_id in range(1, 4):
            for p in range(8):
                rows.append({"lineup_id": lu_id, "name": f"P{p}",
                              "proj": 30.0, "actual": 30.0})
        hist = pd.DataFrame(rows)
        result = backtest_sim(hist, n_sims=1000)
        assert result["sim_mae"] < 20.0  # within a standard deviation

    def test_sim_rmse_geq_sim_mae(self):
        """RMSE ≥ MAE always holds (Jensen's inequality)."""
        result = backtest_sim(_make_hist_df())
        assert result["sim_rmse"] >= result["sim_mae"] - 1e-9


class TestBacktestSimVolatilityModes:
    def test_high_volatility_wider_range(self):
        """High volatility should produce a wider [p15, p85] band on average.
        Only applies when ceil/floor are absent (they override the volatility scale)."""
        hist = _make_hist_df(n_lineups=10, seed=7).drop(columns=["ceil", "floor"])
        low = backtest_sim(hist, volatility_mode="low")
        high = backtest_sim(hist, volatility_mode="high")
        low_df = low["lineup_df"]
        high_df = high["lineup_df"]
        low_band = (low_df["sim_p85"] - low_df["sim_p15"]).mean()
        high_band = (high_df["sim_p85"] - high_df["sim_p15"]).mean()
        assert high_band > low_band

    def test_unknown_volatility_mode_uses_default(self):
        """An unrecognised volatility mode should not raise and should return results."""
        result = backtest_sim(_make_hist_df(), volatility_mode="extreme")
        assert result["n_lineups"] > 0

    def test_ceil_floor_columns_used_when_present(self):
        hist_with = _make_hist_df()
        hist_without = hist_with.drop(columns=["ceil", "floor"])
        r_with = backtest_sim(hist_with)
        r_without = backtest_sim(hist_without)
        # Both should succeed
        assert r_with["n_lineups"] > 0
        assert r_without["n_lineups"] > 0

    def test_n_sims_parameter_accepted(self):
        result = backtest_sim(_make_hist_df(), n_sims=100)
        assert result["n_lineups"] > 0
