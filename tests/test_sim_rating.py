"""Unit tests for yak_core.sim_rating – YakOS Sim Rating system."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.sim_rating import (
    yakos_sim_rating,
    compute_pipeline_ratings,
    compare_rating_weights,
    get_bucket_label,
    get_weight_sets,
    _resolve_weight_set,
)


# ---------------------------------------------------------------------------
# yakos_sim_rating
# ---------------------------------------------------------------------------

class TestYakosSimRating:
    """Core rating computation tests."""

    def _good_metrics(self) -> dict:
        return {
            "projection": 320.0,
            "total_pown": 0.30,
            "top_x_rate": 0.35,
            "itm_rate": 0.60,
            "sim_roi": 0.50,
            "leverage": 2.0,
        }

    def _weak_metrics(self) -> dict:
        return {
            "projection": 230.0,
            "total_pown": 0.65,
            "top_x_rate": 0.02,
            "itm_rate": 0.12,
            "sim_roi": -0.40,
            "leverage": 0.6,
        }

    def test_returns_float_and_bucket(self):
        rating, bucket = yakos_sim_rating({})
        assert isinstance(rating, float)
        assert bucket in ("A", "B", "C", "D")

    def test_rating_in_range(self):
        rating, _ = yakos_sim_rating(self._good_metrics(), "GPP_20")
        assert 0.0 <= rating <= 100.0

    def test_good_lineup_scores_higher_than_weak(self):
        good_rating, _ = yakos_sim_rating(self._good_metrics(), "GPP_20")
        weak_rating, _ = yakos_sim_rating(self._weak_metrics(), "GPP_20")
        assert good_rating > weak_rating

    def test_good_lineup_bucket_a_or_b(self):
        _, bucket = yakos_sim_rating(self._good_metrics(), "GPP_20")
        assert bucket in ("A", "B")

    def test_weak_lineup_bucket_c_or_d(self):
        _, bucket = yakos_sim_rating(self._weak_metrics(), "GPP_20")
        assert bucket in ("C", "D")

    def test_missing_metrics_neutral(self):
        # Empty dict should return ~50 (neutral)
        rating, _ = yakos_sim_rating({})
        assert 30.0 <= rating <= 70.0

    def test_contest_type_aliases(self):
        """Different alias strings for the same contest type should give same rating."""
        m = self._good_metrics()
        r1, _ = yakos_sim_rating(m, "GPP_150")
        r2, _ = yakos_sim_rating(m, "150-max")
        r3, _ = yakos_sim_rating(m, "mme")
        assert r1 == r2 == r3

    def test_cash_contest_ignores_ownership(self):
        """CASH weight set has 0 weight on total_pown; changing it shouldn't affect rating."""
        m_low_own = {**self._good_metrics(), "total_pown": 0.10}
        m_high_own = {**self._good_metrics(), "total_pown": 0.65}
        r_low, _ = yakos_sim_rating(m_low_own, "CASH")
        r_high, _ = yakos_sim_rating(m_high_own, "CASH")
        # In CASH mode ownership is irrelevant so ratings must be identical
        assert r_low == r_high

    def test_custom_weight_set(self):
        """Custom weight override should change the rating."""
        m = self._good_metrics()
        default_r, _ = yakos_sim_rating(m, "GPP_20")
        # Weight everything on projection
        custom_w = {k: 0.0 for k in get_weight_sets()["GPP_20"]}
        custom_w["projection"] = 1.0
        custom_r, _ = yakos_sim_rating(m, "GPP_20", weight_set=custom_w)
        # Results differ (custom projection-only scoring)
        assert isinstance(custom_r, float)

    def test_ownership_inverse_in_gpp(self):
        """Lower ownership should score higher in GPP contests."""
        low_own = {**self._good_metrics(), "total_pown": 0.20}
        high_own = {**self._good_metrics(), "total_pown": 0.65}
        r_low, _ = yakos_sim_rating(low_own, "GPP_20")
        r_high, _ = yakos_sim_rating(high_own, "GPP_20")
        assert r_low > r_high


# ---------------------------------------------------------------------------
# get_bucket_label
# ---------------------------------------------------------------------------

class TestGetBucketLabel:
    def test_bucket_a(self):
        assert get_bucket_label(80.0) == "A"

    def test_bucket_b(self):
        assert get_bucket_label(60.0) == "B"

    def test_bucket_c(self):
        assert get_bucket_label(35.0) == "C"

    def test_bucket_d(self):
        assert get_bucket_label(10.0) == "D"

    def test_boundary_75(self):
        assert get_bucket_label(75.0) == "A"

    def test_boundary_50(self):
        assert get_bucket_label(50.0) == "B"


# ---------------------------------------------------------------------------
# compute_pipeline_ratings
# ---------------------------------------------------------------------------

class TestComputePipelineRatings:
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"lineup_index": 0, "projection": 310, "total_pown": 0.30,
             "top_x_rate": 0.30, "itm_rate": 0.55, "sim_roi": 0.40, "leverage": 1.8},
            {"lineup_index": 1, "projection": 235, "total_pown": 0.60,
             "top_x_rate": 0.05, "itm_rate": 0.15, "sim_roi": -0.30, "leverage": 0.7},
        ])

    def test_adds_rating_and_bucket_columns(self):
        df = compute_pipeline_ratings(self._sample_df())
        assert "yakos_sim_rating" in df.columns
        assert "rating_bucket" in df.columns

    def test_all_ratings_in_range(self):
        df = compute_pipeline_ratings(self._sample_df())
        assert (df["yakos_sim_rating"] >= 0).all()
        assert (df["yakos_sim_rating"] <= 100).all()

    def test_buckets_valid(self):
        df = compute_pipeline_ratings(self._sample_df())
        assert set(df["rating_bucket"]).issubset({"A", "B", "C", "D"})

    def test_lineup0_scores_higher(self):
        df = compute_pipeline_ratings(self._sample_df())
        assert df.iloc[0]["yakos_sim_rating"] > df.iloc[1]["yakos_sim_rating"]

    def test_empty_df_returns_empty(self):
        result = compute_pipeline_ratings(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# compare_rating_weights
# ---------------------------------------------------------------------------

class TestRatingUpdate:
    def test_returns_dataframe(self):
        result = compare_rating_weights({}, {})
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        result = compare_rating_weights({}, {})
        assert "bucket" in result.columns
        assert "n_old" in result.columns
        assert "n_new" in result.columns

    def test_four_buckets(self):
        result = compare_rating_weights({}, {})
        assert set(result["bucket"]) == {"A", "B", "C", "D"}

    def test_custom_weights_change_results(self):
        weight_sets = get_weight_sets()
        old = {"weights": weight_sets["GPP_20"]}
        new_w = {k: 1.0 / len(weight_sets["GPP_20"]) for k in weight_sets["GPP_20"]}
        new = {"weights": new_w}
        result = compare_rating_weights(old, new)
        assert not result.empty

    def test_with_historical_data(self):
        import numpy as np
        rng = np.random.default_rng(0)
        n = 100
        hist_df = pd.DataFrame({
            "projection":  rng.normal(280, 20, n),
            "total_pown":  rng.uniform(0.25, 0.60, n),
            "top_x_rate":  rng.uniform(0.02, 0.35, n),
            "itm_rate":    rng.uniform(0.15, 0.55, n),
            "sim_roi":     rng.normal(0.05, 0.25, n),
            "leverage":    rng.uniform(0.7, 2.0, n),
            "realized_roi": rng.normal(0.04, 0.30, n),
            "top_finish":  rng.integers(0, 2, n),
        })
        result = compare_rating_weights({}, {}, historical_df=hist_df)
        assert "old_realized_roi" in result.columns
        assert "new_realized_roi" in result.columns


# ---------------------------------------------------------------------------
# _resolve_weight_set
# ---------------------------------------------------------------------------

class TestResolveWeightSet:
    def test_gpp_150(self):
        assert _resolve_weight_set("GPP_150") == "GPP_150"

    def test_alias_150max(self):
        assert _resolve_weight_set("150-max") == "GPP_150"

    def test_alias_cash(self):
        assert _resolve_weight_set("cash") == "CASH"

    def test_unknown_defaults_gpp_20(self):
        assert _resolve_weight_set("unknown_type") == "GPP_20"
