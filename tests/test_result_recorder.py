"""Tests for yak_core.result_recorder — feedback loop bridge."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

_repo = str(Path(__file__).resolve().parent.parent)
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from yak_core.result_recorder import (
    merge_actuals_into_pool,
    actuals_from_ocr,
    record_contest_results,
    get_feedback_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool(n: int = 10) -> pd.DataFrame:
    """Minimal pool with projections."""
    names = [f"Player {i}" for i in range(n)]
    pos_cycle = (["PG", "SG", "SF", "PF", "C"] * (n // 5 + 1))[:n]
    team_cycle = (["BOS", "LAL", "MIA", "DEN", "PHX"] * (n // 5 + 1))[:n]
    return pd.DataFrame({
        "player_name": names,
        "pos": pos_cycle,
        "team": team_cycle,
        "salary": [s * 1000 for s in range(4, 4 + n)],
        "proj": [float(15 + i) for i in range(n)],
        "floor": [float(10 + i) for i in range(n)],
        "ceil": [float(25 + i) for i in range(n)],
        "ownership": [float(10 + i * 2) for i in range(n)],
        "smash_prob": [0.1 + i * 0.03 for i in range(n)],
        "bust_prob": [0.3 - i * 0.02 for i in range(n)],
        "leverage": [1.0 + i * 0.2 for i in range(n)],
    })


def _make_actuals(pool: pd.DataFrame, noise: float = 3.0) -> pd.DataFrame:
    """Generate actuals with some noise relative to projections."""
    np.random.seed(42)
    n = len(pool)
    return pd.DataFrame({
        "player_name": pool["player_name"].values,
        "actual_fp": pool["proj"].values + np.random.randn(n) * noise,
    })


# ---------------------------------------------------------------------------
# merge_actuals_into_pool
# ---------------------------------------------------------------------------

class TestMergeActuals:
    def test_basic_merge(self):
        pool = _make_pool(5)
        actuals = pd.DataFrame({
            "player_name": ["Player 0", "Player 2", "Player 4"],
            "actual_fp": [20.0, 25.0, 30.0],
        })
        merged = merge_actuals_into_pool(pool, actuals)
        assert "actual_fp" in merged.columns
        assert merged.loc[merged["player_name"] == "Player 0", "actual_fp"].iloc[0] == 20.0
        assert merged.loc[merged["player_name"] == "Player 2", "actual_fp"].iloc[0] == 25.0
        assert pd.isna(merged.loc[merged["player_name"] == "Player 1", "actual_fp"].iloc[0])

    def test_fuzzy_name_match(self):
        pool = pd.DataFrame({"player_name": ["De'Aaron Fox"], "salary": [7000], "proj": [30.0]})
        actuals = pd.DataFrame({"player_name": ["DeAaron Fox"], "actual_fp": [35.0]})
        merged = merge_actuals_into_pool(pool, actuals)
        assert merged["actual_fp"].iloc[0] == 35.0

    def test_empty_actuals(self):
        pool = _make_pool(3)
        merged = merge_actuals_into_pool(pool, pd.DataFrame())
        assert "actual_fp" not in merged.columns or merged["actual_fp"].isna().all()

    def test_none_actuals(self):
        pool = _make_pool(3)
        merged = merge_actuals_into_pool(pool, None)
        assert len(merged) == len(pool)


# ---------------------------------------------------------------------------
# actuals_from_ocr
# ---------------------------------------------------------------------------

class TestActualsFromOCR:
    def test_basic_conversion(self):
        # Mock a ContestResult-like object
        class FakePlayer:
            def __init__(self, name, pts, sal, pos):
                self.player_name = name
                self.points = pts
                self.salary = sal
                self.pos = pos

        class FakeResult:
            players = [
                FakePlayer("LeBron James", 52.5, 10200, "SF"),
                FakePlayer("Nikola Jokic", 61.0, 11400, "C"),
            ]

        df = actuals_from_ocr(FakeResult())
        assert len(df) == 2
        assert df.iloc[0]["player_name"] == "LeBron James"
        assert df.iloc[0]["actual_fp"] == 52.5
        assert df.iloc[1]["salary"] == 11400


# ---------------------------------------------------------------------------
# record_contest_results (integration)
# ---------------------------------------------------------------------------

class TestRecordContestResults:
    def test_full_pipeline(self, tmp_path):
        """End-to-end: pool + actuals → both feedback systems fire."""
        pool = _make_pool(10)
        actuals = _make_actuals(pool)

        # Patch YAKOS_ROOT so feedback files go to tmp
        with mock.patch("yak_core.calibration_feedback.YAKOS_ROOT", str(tmp_path)), \
             mock.patch("yak_core.calibration_feedback._FEEDBACK_DIR", str(tmp_path / "cal")), \
             mock.patch("yak_core.calibration_feedback._HISTORY_FILE", str(tmp_path / "cal" / "slate_errors.json")), \
             mock.patch("yak_core.calibration_feedback._CORRECTIONS_FILE", str(tmp_path / "cal" / "corrections.json")), \
             mock.patch("yak_core.edge_feedback.YAKOS_ROOT", str(tmp_path)), \
             mock.patch("yak_core.edge_feedback._FEEDBACK_DIR", str(tmp_path / "edge")), \
             mock.patch("yak_core.edge_feedback._SIGNAL_FILE", str(tmp_path / "edge" / "signal_history.json")), \
             mock.patch("yak_core.edge_feedback._WEIGHTS_FILE", str(tmp_path / "edge" / "signal_weights.json")):

            result = record_contest_results(
                slate_date="2026-03-07",
                pool_df=pool,
                actuals_df=actuals,
                contest_type="GPP",
            )

        assert result["players_matched"] == 10
        assert result["calibration"] is not None
        assert "error" not in result.get("calibration", {})
        assert result["edge_feedback"] is not None
        assert result["slate_date"] == "2026-03-07"

    def test_no_match(self):
        pool = _make_pool(5)
        actuals = pd.DataFrame({
            "player_name": ["Nobody Here", "Also Nobody"],
            "actual_fp": [10.0, 20.0],
        })
        result = record_contest_results("2026-03-07", pool, actuals)
        assert result["players_matched"] == 0
        assert "error" in result

    def test_empty_pool(self):
        result = record_contest_results("2026-03-07", pd.DataFrame(), pd.DataFrame())
        assert "error" in result


# ---------------------------------------------------------------------------
# get_feedback_status
# ---------------------------------------------------------------------------

class TestFeedbackStatus:
    def test_no_data(self):
        store: dict = {}
        status = get_feedback_status(store)
        assert status["total_slates_calibration"] == 0
        assert not status["calibration_ready"]


# ---------------------------------------------------------------------------
# Signal weight loading with edge feedback data
# ---------------------------------------------------------------------------

class TestSignalWeightBridge:
    def test_defaults_when_no_file(self, tmp_path):
        """With no feedback file, ricky_signals should return defaults."""
        from yak_core.ricky_signals import _load_feedback_weights, _DEFAULT_WEIGHTS
        with mock.patch("yak_core.config.YAKOS_ROOT", str(tmp_path)):
            w = _load_feedback_weights()
        assert abs(sum(w.values()) - 1.0) < 0.01
        assert set(w.keys()) == set(_DEFAULT_WEIGHTS.keys())

    def test_direct_key_match(self, tmp_path):
        """If weights file uses ricky keys directly, load them."""
        from yak_core.ricky_signals import _load_feedback_weights
        fb_dir = tmp_path / "data" / "edge_feedback"
        fb_dir.mkdir(parents=True)
        weights_data = {
            "weights": {
                "injury_cascade": 0.30,
                "own_proj_mismatch": 0.30,
                "salary_value": 0.15,
                "leverage": 0.15,
                "salary_stickiness": 0.10,
            }
        }
        (fb_dir / "signal_weights.json").write_text(json.dumps(weights_data))

        with mock.patch("yak_core.config.YAKOS_ROOT", str(tmp_path)):
            w = _load_feedback_weights()
        # Should sum to 1.0
        assert abs(sum(w.values()) - 1.0) < 0.01
        # injury_cascade should be boosted relative to defaults
        assert w["injury_cascade"] >= 0.25

    def test_edge_feedback_key_mapping(self, tmp_path):
        """If weights file uses edge_feedback signal names, map them."""
        from yak_core.ricky_signals import _load_feedback_weights
        fb_dir = tmp_path / "data" / "edge_feedback"
        fb_dir.mkdir(parents=True)
        weights_data = {
            "signal_stats": {
                "high_leverage": {"weighted_hit_rate": 0.6, "total_flagged": 50},
                "low_ownership_upside": {"weighted_hit_rate": 0.4, "total_flagged": 30},
                "chalk_fade": {"weighted_hit_rate": 0.3, "total_flagged": 20},
                "salary_value": {"weighted_hit_rate": 0.5, "total_flagged": 40},
                "smash_candidate": {"weighted_hit_rate": 0.45, "total_flagged": 35},
            },
            "weights": {
                "high_leverage": 0.267,
                "low_ownership_upside": 0.178,
                "chalk_fade": 0.133,
                "salary_value": 0.222,
                "smash_candidate": 0.200,
            },
            "n_slates": 5,
        }
        (fb_dir / "signal_weights.json").write_text(json.dumps(weights_data))

        with mock.patch("yak_core.config.YAKOS_ROOT", str(tmp_path)):
            w = _load_feedback_weights()

        assert abs(sum(w.values()) - 1.0) < 0.02
        # All 5 keys should be present
        assert set(w.keys()) == {"injury_cascade", "own_proj_mismatch", "salary_value", "leverage", "salary_stickiness"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
