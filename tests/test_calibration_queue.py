"""Tests for yak_core.calibration queue functions."""

import pandas as pd
import pytest
from yak_core.calibration import (
    action_queue_items,
    get_calibration_queue,
)


def _make_hist_df() -> pd.DataFrame:
    rows = [
        {"slate_date": "2026-02-25", "contest_name": "GPP", "lineup_id": 1,
         "pos": "PG", "team": "CLE", "name": "Alice", "salary": 7000,
         "proj": 30.0, "proj_own": 18.0, "own": 20.0, "actual": 35.0},
        {"slate_date": "2026-02-25", "contest_name": "GPP", "lineup_id": 1,
         "pos": "SG", "team": "HOU", "name": "Bob", "salary": 5500,
         "proj": 22.0, "proj_own": 14.0, "own": 15.0, "actual": 28.0},
        {"slate_date": "2026-02-25", "contest_name": "GPP", "lineup_id": 2,
         "pos": "PG", "team": "LAL", "name": "Carol", "salary": 6200,
         "proj": 28.0, "proj_own": 22.0, "own": 25.0, "actual": 42.0},
        {"slate_date": "2026-02-24", "contest_name": "50/50", "lineup_id": 3,
         "pos": "C", "team": "BOS", "name": "Dan", "salary": 8000,
         "proj": 38.0, "proj_own": 28.0, "own": 30.0, "actual": 50.0},
    ]
    return pd.DataFrame(rows)


class TestGetCalibrationQueue:
    def test_returns_most_recent_dates(self):
        hist = _make_hist_df()
        queue = get_calibration_queue(hist, prior_dates=1)
        assert set(queue["slate_date"].unique()) == {"2026-02-25"}

    def test_queue_status_is_pending(self):
        hist = _make_hist_df()
        queue = get_calibration_queue(hist)
        assert (queue["queue_status"] == "pending").all()

    def test_empty_input_returns_empty(self):
        queue = get_calibration_queue(pd.DataFrame())
        assert queue.empty

    def test_proj_and_proj_own_preserved(self):
        hist = _make_hist_df()
        queue = get_calibration_queue(hist, prior_dates=1)
        assert "proj" in queue.columns, "proj column must be present in calibration queue"
        assert "proj_own" in queue.columns, "proj_own column must be present in calibration queue"
        assert (queue["proj"] > 0).all()
        assert (queue["proj_own"] > 0).all()


class TestActionQueueItemsByName:
    def _make_queue(self):
        hist = _make_hist_df()
        return get_calibration_queue(hist, prior_dates=2)

    def test_action_by_player_name_reviewed(self):
        queue = self._make_queue()
        updated = action_queue_items(queue, ["Alice"], "reviewed", id_col="name")
        assert (updated[updated["name"] == "Alice"]["queue_status"] == "reviewed").all()
        assert (updated[updated["name"] == "Bob"]["queue_status"] == "pending").all()

    def test_action_questioned_is_valid(self):
        queue = self._make_queue()
        updated = action_queue_items(queue, ["Bob"], "questioned", id_col="name")
        assert (updated[updated["name"] == "Bob"]["queue_status"] == "questioned").all()

    def test_action_dismissed(self):
        queue = self._make_queue()
        updated = action_queue_items(queue, ["Carol"], "dismissed", id_col="name")
        assert (updated[updated["name"] == "Carol"]["queue_status"] == "dismissed").all()

    def test_invalid_action_raises(self):
        queue = self._make_queue()
        with pytest.raises(ValueError):
            action_queue_items(queue, ["Alice"], "unknown_action", id_col="name")

    def test_original_df_not_mutated(self):
        queue = self._make_queue()
        original_statuses = queue["queue_status"].copy()
        action_queue_items(queue, ["Alice"], "reviewed", id_col="name")
        pd.testing.assert_series_equal(queue["queue_status"], original_statuses)


class TestActionQueueItemsByRowId:
    def _make_queue(self):
        hist = _make_hist_df()
        return get_calibration_queue(hist, prior_dates=2)

    def test_action_by_row_id(self):
        queue = self._make_queue()
        first_idx = queue.index[0]
        updated = action_queue_items(queue, [first_idx], "reviewed", id_col="row_id")
        assert updated.loc[first_idx, "queue_status"] == "reviewed"
        # Other rows should remain pending
        assert (updated.loc[queue.index[1:], "queue_status"] == "pending").all()

    def test_questioned_by_row_id(self):
        queue = self._make_queue()
        idx = queue.index[1]
        updated = action_queue_items(queue, [idx], "questioned", id_col="row_id")
        assert updated.loc[idx, "queue_status"] == "questioned"


class TestActionQueueDefaultIdCol:
    def test_default_id_col_is_name(self):
        hist = _make_hist_df()
        queue = get_calibration_queue(hist, prior_dates=1)
        updated = action_queue_items(queue, ["Alice"], "reviewed")
        assert (updated[updated["name"] == "Alice"]["queue_status"] == "reviewed").all()
