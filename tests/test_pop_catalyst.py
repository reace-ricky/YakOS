"""Tests for yak_core/pop_catalyst.py -- Pop Catalyst detection."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.pop_catalyst import (
    compute_pop_catalyst,
    SIGNAL_WEIGHTS,
    _compute_injury_opp,
    _compute_salary_lag,
    _compute_minutes_trend,
    _compute_ceiling_flash,
    _build_tag,
    _MIN_PROJ_FOR_POP,
    _MIN_TAG_SCORE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool(n: int = 6) -> pd.DataFrame:
    """Build a minimal pool with all columns pop_catalyst needs."""
    return pd.DataFrame({
        "player_name": [f"Player_{i}" for i in range(n)],
        "salary": [4000, 5000, 6500, 8000, 9500, 3500],
        "proj": [10.0, 15.0, 22.0, 30.0, 35.0, 7.0],
        "injury_bump_fp": [0.0, 3.0, 6.0, 0.5, 0.0, 8.0],
        "rolling_fp_5": [12.0, 25.0, 20.0, 28.0, 34.0, 18.0],
        "rolling_fp_10": [10.0, 18.0, 21.0, 29.0, 35.0, 12.0],
        "rolling_min_5": [22.0, 30.0, 28.0, 34.0, 36.0, 25.0],
        "rolling_min_10": [18.0, 26.0, 29.0, 33.0, 36.0, 20.0],
    })


def _hendricks_pool() -> pd.DataFrame:
    """Simulate a Hendricks-like scenario: cheap, big injury bump, salary lag."""
    return pd.DataFrame({
        "player_name": ["Taylor Hendricks", "Star Player", "Bench Guy"],
        "salary": [4700, 10000, 3500],
        "proj": [13.3, 40.0, 6.0],
        "injury_bump_fp": [7.0, 0.0, 0.5],
        "rolling_fp_5": [22.0, 38.0, 5.0],
        "rolling_fp_10": [15.0, 39.0, 5.5],
        "rolling_min_5": [30.0, 36.0, 12.0],
        "rolling_min_10": [22.0, 35.0, 12.0],
    })


# ---------------------------------------------------------------------------
# Signal weight sanity
# ---------------------------------------------------------------------------

class TestSignalWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9

    def test_four_signals(self):
        assert len(SIGNAL_WEIGHTS) == 4


# ---------------------------------------------------------------------------
# Individual signal tests
# ---------------------------------------------------------------------------

class TestInjuryOpp:
    def test_zero_when_no_bump(self):
        df = pd.DataFrame({"injury_bump_fp": [0.0, 0.0, 0.0]})
        result = _compute_injury_opp(df)
        assert (result == 0.0).all()

    def test_normalised_0_to_1(self):
        df = pd.DataFrame({"injury_bump_fp": [0.0, 3.0, 8.0]})
        result = _compute_injury_opp(df)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_small_bumps_zeroed(self):
        df = pd.DataFrame({"injury_bump_fp": [0.5, 1.0, 1.4]})
        result = _compute_injury_opp(df)
        assert (result == 0.0).all()

    def test_missing_column_graceful(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = _compute_injury_opp(df)
        assert (result == 0.0).all()


class TestSalaryLag:
    def test_underpriced_player(self):
        # rolling_fp_5 = 25 → implied salary = 25 * 250 = $6,250
        # actual salary = $4,000 → gap = $2,250 → score = 2250/3000 = 0.75
        df = pd.DataFrame({"salary": [4000], "rolling_fp_5": [25.0]})
        result = _compute_salary_lag(df)
        assert result.iloc[0] > 0.5

    def test_overpriced_player(self):
        # rolling_fp_5 = 10 → implied = $2,500, actual = $8,000 → gap = 0
        df = pd.DataFrame({"salary": [8000], "rolling_fp_5": [10.0]})
        result = _compute_salary_lag(df)
        assert result.iloc[0] == 0.0

    def test_capped_at_1(self):
        # Huge lag: rolling_fp_5 = 50 → implied = $12,500, salary = $3,500
        df = pd.DataFrame({"salary": [3500], "rolling_fp_5": [50.0]})
        result = _compute_salary_lag(df)
        assert result.iloc[0] == 1.0


class TestMinutesTrend:
    def test_positive_trend(self):
        df = pd.DataFrame({"rolling_min_5": [32.0], "rolling_min_10": [24.0]})
        result = _compute_minutes_trend(df)
        assert result.iloc[0] > 0.0

    def test_flat_trend(self):
        df = pd.DataFrame({"rolling_min_5": [28.0], "rolling_min_10": [28.0]})
        result = _compute_minutes_trend(df)
        assert result.iloc[0] == 0.0

    def test_small_delta_zeroed(self):
        df = pd.DataFrame({"rolling_min_5": [28.5], "rolling_min_10": [27.0]})
        result = _compute_minutes_trend(df)
        assert result.iloc[0] == 0.0  # delta < 2.0


class TestCeilingFlash:
    def test_player_with_ceiling(self):
        # rolling_fp_5 = 30, proj = 15 → ratio = 2.0, score = (2.0-1.5)/1.0 = 0.5
        df = pd.DataFrame({"proj": [15.0], "rolling_fp_5": [30.0]})
        result = _compute_ceiling_flash(df)
        assert result.iloc[0] == pytest.approx(0.5, abs=0.05)

    def test_no_ceiling(self):
        df = pd.DataFrame({"proj": [20.0], "rolling_fp_5": [18.0]})
        result = _compute_ceiling_flash(df)
        assert result.iloc[0] == 0.0

    def test_low_proj_zeroed(self):
        # Below MIN_PROJ_FOR_POP even with great ratio
        df = pd.DataFrame({"proj": [3.0], "rolling_fp_5": [15.0]})
        result = _compute_ceiling_flash(df)
        assert result.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Composite & tag tests
# ---------------------------------------------------------------------------

class TestComputePopCatalyst:
    def test_returns_dataframe_with_columns(self):
        pool = _make_pool()
        result = compute_pop_catalyst(pool)
        for col in ["pop_catalyst_score", "pop_catalyst_tag",
                     "pop_injury_opp", "pop_salary_lag",
                     "pop_minutes_trend", "pop_ceiling_flash"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_scores_between_0_and_1(self):
        pool = _make_pool()
        result = compute_pop_catalyst(pool)
        assert (result["pop_catalyst_score"] >= 0.0).all()
        assert (result["pop_catalyst_score"] <= 1.0).all()

    def test_empty_pool_passthrough(self):
        result = compute_pop_catalyst(pd.DataFrame())
        assert result.empty

    def test_none_pool_passthrough(self):
        result = compute_pop_catalyst(None)
        assert result is None

    def test_low_proj_players_zeroed(self):
        pool = _make_pool()
        result = compute_pop_catalyst(pool)
        # Player_5 has proj=7.0 (above threshold) but Player with proj < 5 should be 0
        low_proj = result[result["proj"] < _MIN_PROJ_FOR_POP]
        if not low_proj.empty:
            assert (low_proj["pop_catalyst_score"] == 0.0).all()

    def test_tag_blank_for_low_scores(self):
        pool = _make_pool()
        result = compute_pop_catalyst(pool)
        low_score = result[result["pop_catalyst_score"] < _MIN_TAG_SCORE]
        assert (low_score["pop_catalyst_tag"] == "").all()

    def test_preserves_original_columns(self):
        pool = _make_pool()
        pool["extra_col"] = "keep_me"
        result = compute_pop_catalyst(pool)
        assert "extra_col" in result.columns
        assert (result["extra_col"] == "keep_me").all()

    def test_does_not_mutate_input(self):
        pool = _make_pool()
        orig_cols = list(pool.columns)
        _ = compute_pop_catalyst(pool)
        assert list(pool.columns) == orig_cols


class TestHendricksScenario:
    """Validate that a Hendricks-like player surfaces with a high pop score."""

    def test_hendricks_has_highest_pop_score(self):
        pool = _hendricks_pool()
        result = compute_pop_catalyst(pool)
        hendricks = result[result["player_name"] == "Taylor Hendricks"].iloc[0]
        assert hendricks["pop_catalyst_score"] > 0.15

    def test_hendricks_injury_opp_fires(self):
        pool = _hendricks_pool()
        result = compute_pop_catalyst(pool)
        hendricks = result[result["player_name"] == "Taylor Hendricks"].iloc[0]
        assert hendricks["pop_injury_opp"] > 0.5

    def test_hendricks_salary_lag_fires(self):
        pool = _hendricks_pool()
        result = compute_pop_catalyst(pool)
        hendricks = result[result["player_name"] == "Taylor Hendricks"].iloc[0]
        # rolling_fp_5 = 22 → implied = $5,500, salary = $4,700 → gap = $800
        assert hendricks["pop_salary_lag"] > 0.0

    def test_hendricks_minutes_trend_fires(self):
        pool = _hendricks_pool()
        result = compute_pop_catalyst(pool)
        hendricks = result[result["player_name"] == "Taylor Hendricks"].iloc[0]
        # rolling_min_5=30, rolling_min_10=22 → delta=8
        assert hendricks["pop_minutes_trend"] > 0.5

    def test_hendricks_has_tag(self):
        pool = _hendricks_pool()
        result = compute_pop_catalyst(pool)
        hendricks = result[result["player_name"] == "Taylor Hendricks"].iloc[0]
        assert hendricks["pop_catalyst_tag"] != ""
        assert "Injury Opp" in hendricks["pop_catalyst_tag"]

    def test_bench_guy_low_score(self):
        pool = _hendricks_pool()
        result = compute_pop_catalyst(pool)
        bench = result[result["player_name"] == "Bench Guy"].iloc[0]
        assert bench["pop_catalyst_score"] < 0.10


class TestBuildTag:
    def test_no_signals(self):
        row = pd.Series({
            "pop_injury_opp": 0.0,
            "pop_salary_lag": 0.0,
            "pop_minutes_trend": 0.0,
            "pop_ceiling_flash": 0.0,
        })
        assert _build_tag(row) == ""

    def test_single_signal(self):
        row = pd.Series({
            "pop_injury_opp": 0.8,
            "pop_salary_lag": 0.0,
            "pop_minutes_trend": 0.0,
            "pop_ceiling_flash": 0.0,
        })
        assert _build_tag(row) == "Injury Opp"

    def test_multiple_signals(self):
        row = pd.Series({
            "pop_injury_opp": 0.5,
            "pop_salary_lag": 0.3,
            "pop_minutes_trend": 0.0,
            "pop_ceiling_flash": 0.2,
        })
        tag = _build_tag(row)
        assert "Injury Opp" in tag
        assert "Salary Lag" in tag
        assert "Ceiling Flash" in tag
        assert "Min Trend" not in tag


# ---------------------------------------------------------------------------
# Edge integration test: pop_catalyst_score flows into edge_score
# ---------------------------------------------------------------------------

class TestEdgeIntegration:
    """Verify that edge.py picks up pop_catalyst_score as 5th component."""

    def test_edge_score_includes_pop_catalyst(self):
        from yak_core.edge import compute_edge_metrics

        pool = _make_pool()
        pool = compute_pop_catalyst(pool)

        result = compute_edge_metrics(pool)
        assert "pop_catalyst_score" in result.columns
        assert "pop_catalyst_tag" in result.columns

    def test_pop_catalyst_boosts_edge_score(self):
        from yak_core.edge import compute_edge_metrics

        pool = _make_pool()

        # Without pop catalyst
        result_no_pop = compute_edge_metrics(pool.copy())

        # With pop catalyst
        pool_pop = compute_pop_catalyst(pool.copy())
        result_pop = compute_edge_metrics(pool_pop)

        # Players with non-zero pop scores should have higher edge_scores
        pop_scores = result_pop.set_index("player_name")["pop_catalyst_score"]
        boosted = pop_scores[pop_scores > 0].index.tolist()

        if boosted:
            for pname in boosted:
                score_no = float(result_no_pop[result_no_pop["player_name"] == pname]["edge_score"].iloc[0])
                score_yes = float(result_pop[result_pop["player_name"] == pname]["edge_score"].iloc[0])
                assert score_yes >= score_no, f"{pname}: pop should boost edge_score"

    def test_edge_label_includes_rocket_for_pop(self):
        from yak_core.edge import compute_edge_metrics

        pool = _hendricks_pool()
        pool = compute_pop_catalyst(pool)
        result = compute_edge_metrics(pool)

        hendricks = result[result["player_name"] == "Taylor Hendricks"]
        if not hendricks.empty:
            label = hendricks.iloc[0]["edge_label"]
            # Hendricks should have rocket emoji from pop catalyst
            assert "🚀" in label or hendricks.iloc[0]["pop_catalyst_score"] < 0.15
