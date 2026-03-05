"""Tests for yak_core/edge.py – compute_edge_metrics."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.edge import compute_edge_metrics, EDGE_DF_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": [f"P{i}" for i in range(n)],
        "pos": ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"][:n],
        "salary": [5000 + i * 500 for i in range(n)],
        "proj": [20.0 + i * 2 for i in range(n)],
        "floor": [14.0 + i for i in range(n)],
        "ceil": [30.0 + i * 3 for i in range(n)],
        "ownership": [5.0 + i * 2 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# compute_edge_metrics
# ---------------------------------------------------------------------------

class TestComputeEdgeMetrics:
    def test_returns_dataframe(self):
        pool = _make_pool()
        result = compute_edge_metrics(pool)
        assert isinstance(result, pd.DataFrame)

    def test_empty_pool_returns_empty_df(self):
        result = compute_edge_metrics(pd.DataFrame())
        assert result.empty

    def test_none_pool_returns_empty_df(self):
        result = compute_edge_metrics(None)
        assert result.empty

    def test_output_has_required_columns(self):
        pool = _make_pool()
        result = compute_edge_metrics(pool)
        for col in EDGE_DF_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_edge_score_descending(self):
        pool = _make_pool()
        result = compute_edge_metrics(pool)
        scores = result["edge_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_smash_prob_between_0_and_1(self):
        pool = _make_pool()
        result = compute_edge_metrics(pool)
        assert (result["smash_prob"].between(0, 1)).all()

    def test_bust_prob_between_0_and_1(self):
        pool = _make_pool()
        result = compute_edge_metrics(pool)
        assert (result["bust_prob"].between(0, 1)).all()

    def test_edge_score_non_negative(self):
        pool = _make_pool()
        result = compute_edge_metrics(pool)
        assert (result["edge_score"] >= 0).all()

    def test_edge_label_column_is_string(self):
        pool = _make_pool()
        result = compute_edge_metrics(pool)
        assert result["edge_label"].dtype == object

    def test_missing_player_name_raises(self):
        pool = pd.DataFrame({"salary": [5000], "proj": [20.0]})
        with pytest.raises(ValueError, match="player_name"):
            compute_edge_metrics(pool)

    def test_calibration_proj_multiplier_applied(self):
        pool = _make_pool(n=3)
        base_result = compute_edge_metrics(pool, calibration_state={})
        cal_result = compute_edge_metrics(pool, calibration_state={"proj_multiplier": 2.0})
        # Higher proj should yield different (typically higher) edge scores
        # for at least some players
        assert not base_result["proj"].equals(cal_result["proj"])

    def test_calibration_ceiling_boost_applied(self):
        pool = _make_pool(n=3)
        base_result = compute_edge_metrics(pool, calibration_state={})
        cal_result = compute_edge_metrics(pool, calibration_state={"ceiling_boost": 0.5})
        # Ceiling boost changes the smash threshold → probabilities should differ
        assert not base_result["smash_prob"].reset_index(drop=True).equals(
            cal_result["smash_prob"].reset_index(drop=True)
        )

    def test_preserves_extra_columns(self):
        pool = _make_pool()
        pool["extra"] = "value"
        result = compute_edge_metrics(pool)
        assert "extra" in result.columns

    def test_own_pct_from_ownership_column(self):
        pool = _make_pool()
        result = compute_edge_metrics(pool)
        # own_pct should match the input ownership values
        assert "own_pct" in result.columns
        assert (result["own_pct"] > 0).any()

    def test_leverage_nan_for_near_zero_ownership(self):
        pool = _make_pool(n=2)
        pool["ownership"] = [0.0, 0.05]  # Below MIN_OWN_FOR_LEVERAGE
        result = compute_edge_metrics(pool)
        # leverage should be NaN for these players
        assert result["leverage"].isna().any()

    def test_variance_affects_smash_bust(self):
        pool = _make_pool()
        low_var = compute_edge_metrics(pool, variance=0.5)
        high_var = compute_edge_metrics(pool, variance=2.0)
        # Higher variance → more spread → different probs
        assert not low_var["smash_prob"].equals(high_var["smash_prob"])

    def test_single_player_pool(self):
        pool = pd.DataFrame({
            "player_name": ["Solo"],
            "salary": [7000],
            "proj": [35.0],
            "ownership": [15.0],
        })
        result = compute_edge_metrics(pool)
        assert len(result) == 1
        assert result.iloc[0]["player_name"] == "Solo"
