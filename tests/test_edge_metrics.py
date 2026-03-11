"""Tests for yak_core/edge.py – compute_edge_metrics
and yak_core/edge_metrics.py – Ricky Confidence helpers.
"""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.edge import compute_edge_metrics, EDGE_DF_COLUMNS
from yak_core.edge_metrics import (
    compute_player_confidence,
    compute_pool_confidence,
    compute_ricky_confidence_for_contest,
    get_confidence_color,
)


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
        # Use players across different salary brackets so ceiling_boost
        # produces varying effects on normalised ceil_magnitude.
        pool = pd.DataFrame({
            "player_name": ["Star", "Mid", "Cheap"],
            "salary": [10000, 7000, 4000],
            "proj": [45.0, 22.0, 6.0],
            "floor": [35.0, 16.0, 3.0],
            "ceil": [60.0, 30.0, 12.0],
            "ownership": [25.0, 12.0, 5.0],
        })
        base_result = compute_edge_metrics(pool, calibration_state={})
        cal_result = compute_edge_metrics(pool, calibration_state={"ceiling_boost": 0.5})
        # Ceiling boost changes edge_score via ceil_magnitude (not smash_prob,
        # which is now driven by the empirical salary/ownership model).
        base_es = base_result.set_index("player_name")["edge_score"]
        cal_es = cal_result.set_index("player_name")["edge_score"]
        assert not base_es.equals(cal_es)

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


# ---------------------------------------------------------------------------
# yak_core/edge_metrics.py — Ricky Confidence helpers
# ---------------------------------------------------------------------------

class TestComputePlayerConfidence:
    def test_high_smash_low_bust_high_confidence(self):
        row = pd.Series({"proj": 30.0, "floor": 20.0, "ceil": 40.0, "smash_prob": 0.7, "bust_prob": 0.1})
        score = compute_player_confidence(row)
        assert score > 0.7

    def test_high_bust_low_smash_low_confidence(self):
        row = pd.Series({"proj": 30.0, "floor": 10.0, "ceil": 50.0, "smash_prob": 0.1, "bust_prob": 0.8})
        score = compute_player_confidence(row)
        assert score < 0.4

    def test_zero_proj_returns_zero(self):
        row = pd.Series({"proj": 0.0, "floor": 0.0, "ceil": 10.0, "smash_prob": 0.5, "bust_prob": 0.1})
        assert compute_player_confidence(row) == 0.0

    def test_result_within_0_and_1(self):
        row = pd.Series({"proj": 25.0, "floor": 18.0, "ceil": 35.0, "smash_prob": 0.5, "bust_prob": 0.2})
        score = compute_player_confidence(row)
        assert 0.0 <= score <= 1.0

    def test_percentage_probabilities_normalized(self):
        """smash/bust values > 1 (i.e., 0–100 scale) are normalized."""
        row = pd.Series({"proj": 30.0, "floor": 20.0, "ceil": 40.0, "smash_prob": 70.0, "bust_prob": 10.0})
        score = compute_player_confidence(row)
        assert score > 0.7


class TestComputePoolConfidence:
    def test_returns_series_same_length(self):
        pool = pd.DataFrame({
            "proj": [20.0, 30.0, 0.0],
            "floor": [14.0, 22.0, 0.0],
            "ceil": [28.0, 40.0, 10.0],
            "smash_prob": [0.5, 0.7, 0.0],
            "bust_prob": [0.2, 0.1, 0.0],
        })
        result = compute_pool_confidence(pool)
        assert isinstance(result, pd.Series)
        assert len(result) == len(pool)

    def test_zero_proj_rows_score_zero(self):
        pool = pd.DataFrame({
            "proj": [0.0],
            "floor": [0.0],
            "ceil": [10.0],
            "smash_prob": [0.5],
            "bust_prob": [0.1],
        })
        result = compute_pool_confidence(pool)
        assert result.iloc[0] == 0.0


class TestComputeRickyConfidenceForContest:
    def test_core_value_and_leverage_combined(self):
        payload = {
            "core_value_players": [{"confidence": 0.8}] * 5,
            "leverage_players": [{"confidence": 0.6}] * 3,
        }
        score = compute_ricky_confidence_for_contest(payload)
        # 0.6 * 0.8 + 0.4 * 0.6 = 0.72  → 72.0
        assert 71.0 <= score <= 75.0

    def test_empty_payload_returns_zero(self):
        assert compute_ricky_confidence_for_contest({}) == 0.0

    def test_only_core_value_players(self):
        payload = {"core_value_players": [{"confidence": 0.9}]}
        score = compute_ricky_confidence_for_contest(payload)
        assert score == pytest.approx(90.0, abs=1.0)

    def test_only_leverage_players(self):
        payload = {"leverage_players": [{"confidence": 0.5}]}
        score = compute_ricky_confidence_for_contest(payload)
        assert score == pytest.approx(50.0, abs=1.0)

    def test_result_between_0_and_100(self):
        payload = {
            "core_value_players": [{"confidence": 1.0}] * 3,
            "leverage_players": [{"confidence": 1.0}] * 3,
        }
        assert 0.0 <= compute_ricky_confidence_for_contest(payload) <= 100.0


class TestGetConfidenceColor:
    def test_high_score_green(self):
        assert get_confidence_color(85) == "green"

    def test_mid_score_yellow(self):
        assert get_confidence_color(65) == "yellow"

    def test_low_score_red(self):
        assert get_confidence_color(40) == "red"

    def test_boundary_80_green(self):
        assert get_confidence_color(80) == "green"

    def test_boundary_60_yellow(self):
        assert get_confidence_color(60) == "yellow"

    def test_boundary_59_red(self):
        assert get_confidence_color(59) == "red"


# ---------------------------------------------------------------------------
# DVP integration in compute_edge_metrics
# ---------------------------------------------------------------------------

class TestComputeEdgeMetricsDvpIntegration:
    """Tests that DVP matchup boost flows correctly through compute_edge_metrics."""

    def _make_pool_with_opp(self) -> pd.DataFrame:
        return pd.DataFrame({
            "player_name": ["PG_Soft", "PG_Tough", "C_Neutral"],
            "pos":         ["PG",      "PG",       "C"],
            "opponent":    ["LAL",     "BOS",      "GSW"],
            "salary":      [7000,      7000,       6000],
            "proj":        [30.0,      30.0,       25.0],
            "floor":       [22.0,      22.0,       18.0],
            "ceil":        [42.0,      42.0,       35.0],
            "ownership":   [15.0,      15.0,       12.0],
        })

    def _make_dvp(self) -> "pd.DataFrame":
        import pandas as _pd
        # LAL allows 46.0 PG (soft), BOS allows 40.0 PG (tough), league avg 43.0
        return _pd.DataFrame({
            "Team": ["LAL", "BOS", "GSW"],
            "PG":   [46.0,  40.0,  43.0],
            "C":    [39.0,  36.0,  37.5],
        })

    def test_dvp_matchup_boost_column_present(self):
        pool = self._make_pool_with_opp()
        result = compute_edge_metrics(pool)
        assert "dvp_matchup_boost" in result.columns

    def test_dvp_matchup_boost_zero_without_dvp_data(self, tmp_path, monkeypatch):
        """When no DVP file is on disk, dvp_matchup_boost should be 0.0."""
        import yak_core.edge as edge_mod
        monkeypatch.setattr(edge_mod, "load_dvp_table", lambda: None)
        pool = self._make_pool_with_opp()
        result = compute_edge_metrics(pool)
        assert (result["dvp_matchup_boost"] == 0.0).all()

    def test_soft_matchup_has_higher_boost_than_tough(self, monkeypatch):
        import pandas as _pd
        import yak_core.edge as edge_mod
        dvp = self._make_dvp()
        monkeypatch.setattr(edge_mod, "load_dvp_table", lambda: dvp)
        pool = self._make_pool_with_opp()
        result = compute_edge_metrics(pool).set_index("player_name")
        assert result.loc["PG_Soft", "dvp_matchup_boost"] > result.loc["PG_Tough", "dvp_matchup_boost"]

    def test_soft_matchup_raises_edge_score(self, monkeypatch):
        import pandas as _pd
        import yak_core.edge as edge_mod
        dvp = self._make_dvp()
        # Run once with DVP, once without
        monkeypatch.setattr(edge_mod, "load_dvp_table", lambda: dvp)
        pool = self._make_pool_with_opp()
        with_dvp = compute_edge_metrics(pool).set_index("player_name")

        monkeypatch.setattr(edge_mod, "load_dvp_table", lambda: None)
        without_dvp = compute_edge_metrics(pool).set_index("player_name")

        # PG_Soft faces a soft matchup — edge score should be >= without DVP
        assert with_dvp.loc["PG_Soft", "edge_score"] >= without_dvp.loc["PG_Soft", "edge_score"]

    def test_no_dvp_defaults_neutral_edge_score_weight(self, monkeypatch):
        """Without DVP data dvp_norm=0.5 (neutral); edge_score should still sum correctly."""
        import yak_core.edge as edge_mod
        monkeypatch.setattr(edge_mod, "load_dvp_table", lambda: None)
        pool = self._make_pool_with_opp()
        result = compute_edge_metrics(pool)
        assert (result["edge_score"] >= 0).all()

    def test_pool_without_pos_or_opponent_still_works(self, monkeypatch):
        """Pool with no pos/opponent columns should not crash — falls back to 0 boost."""
        import yak_core.edge as edge_mod
        dvp = self._make_dvp()
        monkeypatch.setattr(edge_mod, "load_dvp_table", lambda: dvp)
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "salary":      [6000, 7000],
            "proj":        [22.0, 30.0],
            "floor":       [15.0, 22.0],
            "ceil":        [32.0, 42.0],
            "ownership":   [10.0, 18.0],
        })
        result = compute_edge_metrics(pool)
        # No pos/opponent → get_dvp_boost returns 0.0 for all
        assert (result["dvp_matchup_boost"] == 0.0).all()
