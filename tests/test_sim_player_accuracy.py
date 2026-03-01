"""Tests for build_sim_player_accuracy_table — sim projections vs actuals."""

import numpy as np
import pandas as pd
import pytest
from yak_core.sims import build_sim_player_accuracy_table


def _make_pool(players=None):
    """Build a minimal pool DataFrame with player_name and proj."""
    if players is None:
        players = [
            ("LeBron James", 45.0),
            ("Stephen Curry", 38.0),
            ("Nikola Jokic", 52.0),
            ("Kevin Durant", 40.0),
            ("Luka Doncic", 48.0),
        ]
    return pd.DataFrame(players, columns=["player_name", "proj"])


def _make_actuals(players=None):
    """Build a minimal actuals DataFrame with player_name and actual_fp."""
    if players is None:
        players = [
            ("LeBron James", 50.0),
            ("Stephen Curry", 35.0),
            ("Nikola Jokic", 60.0),
            ("Kevin Durant", 42.0),
            ("Luka Doncic", 45.0),
        ]
    return pd.DataFrame(players, columns=["player_name", "actual_fp"])


class TestBuildSimPlayerAccuracyTableReturnShape:
    def test_returns_dict_with_required_keys(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        for key in ("player_df", "mae", "rmse", "bias", "hit_rate", "r2", "n_players"):
            assert key in result, f"missing key: {key}"

    def test_player_df_has_expected_columns(self):
        df = build_sim_player_accuracy_table(_make_pool(), _make_actuals())["player_df"]
        for col in ("name", "proj", "actual", "error", "abs_error", "pct_error"):
            assert col in df.columns, f"missing column: {col}"

    def test_n_players_matches_matched_count(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        assert result["n_players"] == len(_make_pool())

    def test_player_df_row_count_matches_n_players(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        assert len(result["player_df"]) == result["n_players"]


class TestBuildSimPlayerAccuracyTableEdgeCases:
    def test_empty_pool_returns_empty(self):
        result = build_sim_player_accuracy_table(pd.DataFrame(), _make_actuals())
        assert result["n_players"] == 0
        assert result["player_df"].empty

    def test_empty_actuals_returns_empty(self):
        result = build_sim_player_accuracy_table(_make_pool(), pd.DataFrame())
        assert result["n_players"] == 0
        assert result["player_df"].empty

    def test_no_matching_names_returns_empty(self):
        actuals = pd.DataFrame([("Nobody McFake", 30.0)], columns=["player_name", "actual_fp"])
        result = build_sim_player_accuracy_table(_make_pool(), actuals)
        assert result["n_players"] == 0

    def test_partial_match(self):
        actuals = pd.DataFrame(
            [("LeBron James", 50.0), ("Zion Williamson", 40.0)],
            columns=["player_name", "actual_fp"],
        )
        result = build_sim_player_accuracy_table(_make_pool(), actuals)
        assert result["n_players"] == 1
        assert result["player_df"]["name"].iloc[0] == "LeBron James"

    def test_accepts_name_column_instead_of_player_name(self):
        pool = _make_pool().rename(columns={"player_name": "name"})
        result = build_sim_player_accuracy_table(pool, _make_actuals())
        assert result["n_players"] == len(pool)

    def test_accepts_actual_column_instead_of_actual_fp(self):
        actuals = _make_actuals().rename(columns={"actual_fp": "actual"})
        result = build_sim_player_accuracy_table(_make_pool(), actuals)
        assert result["n_players"] == len(_make_pool())

    def test_accepts_player_name_in_actuals_as_name(self):
        actuals = _make_actuals().rename(columns={"player_name": "name"})
        result = build_sim_player_accuracy_table(_make_pool(), actuals)
        assert result["n_players"] == len(_make_pool())


class TestBuildSimPlayerAccuracyTableMetrics:
    def test_mae_is_non_negative(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        assert result["mae"] >= 0.0

    def test_rmse_is_non_negative(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        assert result["rmse"] >= 0.0

    def test_hit_rate_between_0_and_100(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        assert 0.0 <= result["hit_rate"] <= 100.0

    def test_r2_reasonable_range(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        # R2 can be negative (when predictions are worse than mean), but <= 1
        assert result["r2"] <= 1.0

    def test_error_equals_proj_minus_actual(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        df = result["player_df"]
        assert np.allclose(df["error"], df["proj"] - df["actual"], atol=1e-6)

    def test_abs_error_equals_abs_of_error(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        df = result["player_df"]
        assert np.allclose(df["abs_error"], df["error"].abs(), atol=1e-6)

    def test_rmse_geq_mae(self):
        result = build_sim_player_accuracy_table(_make_pool(), _make_actuals())
        assert result["rmse"] >= result["mae"] - 1e-9

    def test_perfect_projection_zero_mae(self):
        """When proj == actual, MAE should be 0."""
        players = [("Alice", 30.0), ("Bob", 25.0), ("Charlie", 40.0)]
        pool = pd.DataFrame(players, columns=["player_name", "proj"])
        actuals = pd.DataFrame(
            [(p, v) for p, v in players], columns=["player_name", "actual_fp"]
        )
        result = build_sim_player_accuracy_table(pool, actuals)
        assert result["mae"] == 0.0
        assert result["bias"] == 0.0

    def test_hit_rate_all_within_threshold(self):
        """When all errors are < 5 FP, hit_rate at threshold=10 should be 100."""
        players = [("Alice", 30.0), ("Bob", 25.0)]
        pool = pd.DataFrame(players, columns=["player_name", "proj"])
        actuals = pd.DataFrame(
            [("Alice", 32.0), ("Bob", 27.0)], columns=["player_name", "actual_fp"]
        )
        result = build_sim_player_accuracy_table(pool, actuals, hit_threshold=10.0)
        assert result["hit_rate"] == 100.0

    def test_hit_rate_none_within_threshold(self):
        """When all errors exceed threshold, hit_rate should be 0."""
        players = [("Alice", 30.0), ("Bob", 25.0)]
        pool = pd.DataFrame(players, columns=["player_name", "proj"])
        actuals = pd.DataFrame(
            [("Alice", 55.0), ("Bob", 50.0)], columns=["player_name", "actual_fp"]
        )
        result = build_sim_player_accuracy_table(pool, actuals, hit_threshold=5.0)
        assert result["hit_rate"] == 0.0

    def test_custom_hit_threshold(self):
        """hit_threshold parameter is respected."""
        players = [("Alice", 30.0), ("Bob", 25.0)]
        pool = pd.DataFrame(players, columns=["player_name", "proj"])
        actuals = pd.DataFrame(
            [("Alice", 37.0), ("Bob", 27.0)], columns=["player_name", "actual_fp"]
        )
        # Alice error=7, Bob error=2 → threshold 5 → only Bob hits
        r5 = build_sim_player_accuracy_table(pool, actuals, hit_threshold=5.0)
        assert r5["hit_rate"] == 50.0
        # Threshold 10 → both hit
        r10 = build_sim_player_accuracy_table(pool, actuals, hit_threshold=10.0)
        assert r10["hit_rate"] == 100.0

    def test_duplicate_player_in_actuals_averaged(self):
        """When a player appears multiple times in actuals (multiple lineups), their
        actual scores are averaged before computing error."""
        pool = pd.DataFrame([("LeBron", 40.0)], columns=["player_name", "proj"])
        actuals = pd.DataFrame(
            [("LeBron", 30.0), ("LeBron", 50.0)], columns=["player_name", "actual_fp"]
        )
        result = build_sim_player_accuracy_table(pool, actuals)
        # avg actual = 40.0 → error = 40 - 40 = 0
        assert result["mae"] == 0.0
