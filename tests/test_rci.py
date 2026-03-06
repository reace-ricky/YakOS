"""Tests for yak_core/rci.py — RCI engine."""
from __future__ import annotations

import pandas as pd
import pytest

from yak_core.rci import (
    RCISignal,
    RCIResult,
    DEFAULT_WEIGHTS,
    compute_projection_confidence_signal,
    compute_sim_alignment_signal,
    compute_ownership_accuracy_signal,
    compute_historical_roi_signal,
    compute_rci,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_payload_full() -> dict:
    return {
        "core_value_players": [{"confidence": 0.8}] * 4,
        "leverage_players": [{"confidence": 0.6}] * 2,
    }


def _sim_results_df() -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": [f"P{i}" for i in range(5)],
        "sim_mean": [25.0, 30.0, 20.0, 35.0, 28.0],
        "smash_prob": [0.2, 0.4, 0.1, 0.5, 0.3],
    })


def _actual_results_df() -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": [f"P{i}" for i in range(5)],
        "actual_fp": [24.0, 29.5, 21.0, 33.0, 28.0],
        "actual_smash": [0.2, 0.4, 0.1, 0.5, 0.3],
    })


def _projected_ownership_df() -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": [f"P{i}" for i in range(5)],
        "ownership_proj": [10.0, 25.0, 5.0, 30.0, 15.0],
    })


def _actual_ownership_df() -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": [f"P{i}" for i in range(5)],
        "ownership_actual": [11.0, 23.0, 6.0, 32.0, 14.0],
    })


def _backtest_df() -> pd.DataFrame:
    return pd.DataFrame({
        "contest_type": ["GPP - 20 Max"] * 5,
        "roi": [0.10, 0.20, -0.05, 0.15, 0.05],
    })


# ---------------------------------------------------------------------------
# RCISignal
# ---------------------------------------------------------------------------

class TestRCISignal:
    def test_instantiation(self):
        sig = RCISignal(
            name="test_signal",
            value=75.0,
            weight=0.25,
            description="Test",
            status="green",
        )
        assert sig.name == "test_signal"
        assert sig.value == 75.0
        assert sig.weight == 0.25
        assert sig.status == "green"


# ---------------------------------------------------------------------------
# compute_projection_confidence_signal
# ---------------------------------------------------------------------------

class TestProjectionConfidenceSignal:
    def test_returns_rci_signal(self):
        sig = compute_projection_confidence_signal(_edge_payload_full())
        assert isinstance(sig, RCISignal)
        assert sig.name == "projection_confidence"

    def test_score_between_0_and_100(self):
        sig = compute_projection_confidence_signal(_edge_payload_full())
        assert 0.0 <= sig.value <= 100.0

    def test_empty_payload_returns_low_score(self):
        sig = compute_projection_confidence_signal({})
        assert sig.value == 0.0

    def test_status_reflects_score(self):
        sig = compute_projection_confidence_signal(_edge_payload_full())
        if sig.value >= 70:
            assert sig.status == "green"
        elif sig.value >= 45:
            assert sig.status == "yellow"
        else:
            assert sig.status == "red"

    def test_default_weight(self):
        sig = compute_projection_confidence_signal(_edge_payload_full())
        assert sig.weight == DEFAULT_WEIGHTS["projection_confidence"]


# ---------------------------------------------------------------------------
# compute_sim_alignment_signal
# ---------------------------------------------------------------------------

class TestSimAlignmentSignal:
    def test_returns_low_when_no_data(self):
        sig = compute_sim_alignment_signal(None, None)
        assert sig.value == 0.0
        assert sig.status == "red"

    def test_uses_sim_distribution_when_no_actuals(self):
        sig = compute_sim_alignment_signal(_sim_results_df(), pd.DataFrame())
        # Should compute from sim distribution quality, not return neutral 50
        assert sig.value > 0.0
        assert sig.name == "sim_alignment"

    def test_with_data_returns_signal(self):
        sig = compute_sim_alignment_signal(_sim_results_df(), _actual_results_df())
        assert isinstance(sig, RCISignal)
        assert sig.name == "sim_alignment"
        assert 0.0 <= sig.value <= 100.0

    def test_perfect_alignment_high_score(self):
        """When sim_mean == actual_fp, error is 0 → score near 100."""
        sim = pd.DataFrame({
            "player_name": ["A", "B"],
            "sim_mean": [30.0, 25.0],
            "smash_prob": [0.3, 0.2],
        })
        actual = pd.DataFrame({
            "player_name": ["A", "B"],
            "actual_fp": [30.0, 25.0],
            "actual_smash": [0.3, 0.2],
        })
        sig = compute_sim_alignment_signal(sim, actual)
        assert sig.value >= 90.0

    def test_default_weight(self):
        sig = compute_sim_alignment_signal(None, None)
        assert sig.weight == DEFAULT_WEIGHTS["sim_alignment"]


# ---------------------------------------------------------------------------
# compute_ownership_accuracy_signal
# ---------------------------------------------------------------------------

class TestOwnershipAccuracySignal:
    def test_returns_low_when_no_data(self):
        sig = compute_ownership_accuracy_signal(None, None)
        assert sig.value == 0.0
        assert sig.status == "red"

    def test_with_data_returns_signal(self):
        sig = compute_ownership_accuracy_signal(
            _projected_ownership_df(), _actual_ownership_df()
        )
        assert isinstance(sig, RCISignal)
        assert sig.name == "ownership_accuracy"
        assert 0.0 <= sig.value <= 100.0

    def test_perfect_accuracy_high_score(self):
        proj = pd.DataFrame({
            "player_name": ["A", "B"],
            "ownership_proj": [10.0, 20.0],
        })
        actual = pd.DataFrame({
            "player_name": ["A", "B"],
            "ownership_actual": [10.0, 20.0],
        })
        sig = compute_ownership_accuracy_signal(proj, actual)
        assert sig.value == 100.0

    def test_large_error_low_score(self):
        proj = pd.DataFrame({
            "player_name": ["A"],
            "ownership_proj": [5.0],
        })
        actual = pd.DataFrame({
            "player_name": ["A"],
            "ownership_actual": [50.0],  # 45% avg error → score = 0
        })
        sig = compute_ownership_accuracy_signal(proj, actual)
        assert sig.value == 0.0

    def test_default_weight(self):
        sig = compute_ownership_accuracy_signal(None, None)
        assert sig.weight == DEFAULT_WEIGHTS["ownership_accuracy"]

    def test_missing_columns_returns_low(self):
        """DataFrames without 'own' columns and no pool → low score."""
        proj = pd.DataFrame({"player_name": ["A"], "salary": [5000]})
        actual = pd.DataFrame({"player_name": ["A"], "salary": [5000]})
        sig = compute_ownership_accuracy_signal(proj, actual)
        # No ownership columns in actual comparison, no player_pool → 0
        assert sig.value == 0.0


# ---------------------------------------------------------------------------
# compute_historical_roi_signal
# ---------------------------------------------------------------------------

class TestHistoricalROISignal:
    def test_returns_neutral_when_no_data(self):
        sig = compute_historical_roi_signal(None, "GPP - 20 Max")
        assert sig.value == 50.0
        assert sig.status == "yellow"

    def test_with_data_returns_signal(self):
        sig = compute_historical_roi_signal(_backtest_df(), "GPP - 20 Max")
        assert isinstance(sig, RCISignal)
        assert sig.name == "historical_roi"
        assert 0.0 <= sig.value <= 100.0

    def test_positive_roi_above_50(self):
        df = pd.DataFrame({"contest_type": ["GPP"], "roi": [0.20]})
        sig = compute_historical_roi_signal(df, "GPP")
        assert sig.value > 50.0

    def test_negative_roi_below_50(self):
        df = pd.DataFrame({"contest_type": ["GPP"], "roi": [-0.30]})
        sig = compute_historical_roi_signal(df, "GPP")
        assert sig.value < 50.0

    def test_breakeven_roi_near_50(self):
        df = pd.DataFrame({"contest_type": ["GPP"], "roi": [0.0]})
        sig = compute_historical_roi_signal(df, "GPP")
        assert abs(sig.value - 50.0) < 1.0

    def test_no_matching_contest_type_returns_neutral(self):
        df = pd.DataFrame({"contest_type": ["Cash"], "roi": [0.20]})
        sig = compute_historical_roi_signal(df, "GPP")
        assert sig.value == 50.0

    def test_default_weight(self):
        sig = compute_historical_roi_signal(None, "GPP")
        assert sig.weight == DEFAULT_WEIGHTS["historical_roi"]


# ---------------------------------------------------------------------------
# compute_rci
# ---------------------------------------------------------------------------

class TestComputeRCI:
    def test_returns_rci_result(self):
        result = compute_rci("GPP - 20 Max", _edge_payload_full())
        assert isinstance(result, RCIResult)

    def test_score_between_0_and_100(self):
        result = compute_rci("GPP - 20 Max", _edge_payload_full())
        assert 0.0 <= result.rci_score <= 100.0

    def test_four_signals_present(self):
        result = compute_rci("GPP - 20 Max", _edge_payload_full())
        assert len(result.signals) == 4
        signal_names = {s.name for s in result.signals}
        assert signal_names == {
            "projection_confidence",
            "sim_alignment",
            "ownership_accuracy",
            "historical_roi",
        }

    def test_weights_sum_to_one(self):
        result = compute_rci("GPP - 20 Max", _edge_payload_full())
        total = sum(s.weight for s in result.signals)
        assert abs(total - 1.0) < 1e-9

    def test_calibration_stable_when_all_data_present(self):
        """All signals with data + score >= 70 → calibration_stable=True."""
        payload = {
            "core_value_players": [{"confidence": 1.0}] * 5,
            "leverage_players": [{"confidence": 1.0}] * 3,
        }
        result = compute_rci(
            "GPP - 20 Max", payload,
            sim_results=_sim_results_df(),
            actual_results=_actual_results_df(),
            projected_ownership=_projected_ownership_df(),
            actual_ownership=_actual_ownership_df(),
            backtest_results=_backtest_df(),
        )
        # With full data, projection confidence is high, sim alignment good,
        # ownership accurate, ROI positive → should be stable
        assert result.rci_score >= 70
        assert not any(s.status == "red" for s in result.signals)
        assert result.calibration_stable is True

    def test_calibration_not_stable_when_red_signal(self):
        """If any signal is red, calibration is not stable even if RCI >= 70."""
        result = compute_rci("GPP - 20 Max", {})
        # Empty payload → projection_confidence = 0 → red
        # Even if composite were >=70, red signal blocks stability
        assert not result.calibration_stable

    def test_recommendation_text_varies_by_score(self):
        # High RCI (from perfect edge payload, neutral signals)
        payload = {"core_value_players": [{"confidence": 1.0}] * 10}
        r = compute_rci("GPP - 20 Max", payload)
        assert isinstance(r.recommendation, str)
        assert len(r.recommendation) > 0

    def test_custom_weights_applied(self):
        """Custom weights should override defaults."""
        custom = {
            "projection_confidence": 1.0,
            "sim_alignment": 0.0,
            "ownership_accuracy": 0.0,
            "historical_roi": 0.0,
        }
        result = compute_rci("GPP - 20 Max", _edge_payload_full(), weights=custom)
        # All weight on projection_confidence → normalized weight = 1.0
        pc_signal = next(s for s in result.signals if s.name == "projection_confidence")
        assert abs(pc_signal.weight - 1.0) < 1e-9
        # Score should equal the projection_confidence score
        assert abs(result.rci_score - pc_signal.value) < 0.5

    def test_rci_status_matches_score(self):
        result = compute_rci("GPP - 20 Max", _edge_payload_full())
        if result.rci_score >= 70:
            assert result.rci_status == "green"
        elif result.rci_score >= 45:
            assert result.rci_status == "yellow"
        else:
            assert result.rci_status == "red"

    def test_with_backtest_data(self):
        result = compute_rci(
            "GPP - 20 Max",
            _edge_payload_full(),
            backtest_results=_backtest_df(),
        )
        assert isinstance(result, RCIResult)
        roi_signal = next(s for s in result.signals if s.name == "historical_roi")
        # Avg ROI = 0.09 → score = 50 + 9 = 59 → yellow
        assert roi_signal.value > 50.0

    def test_contest_label_stored(self):
        result = compute_rci("Cash", _edge_payload_full())
        assert result.contest_label == "Cash"
