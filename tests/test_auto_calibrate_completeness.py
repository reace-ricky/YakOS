"""Tests for auto-calibrate incomplete date detection."""

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from yak_core.auto_calibrate import (
    IncompleteSlateError,
    SlateCompleteness,
    _DEFAULT_COMPLETENESS_THRESHOLD,
    check_slate_completeness,
    filter_incomplete_dates,
)


# ---------------------------------------------------------------------------
# SlateCompleteness dataclass
# ---------------------------------------------------------------------------

class TestSlateCompleteness:
    def test_complete_slate(self):
        sc = SlateCompleteness(
            slate_date=date(2026, 3, 15),
            is_complete=True,
            reason="",
            completeness_pct=92.0,
            total_players=50,
            players_with_actuals=46,
            is_future=False,
        )
        assert sc.is_complete
        assert sc.completeness_pct == 92.0

    def test_incomplete_slate(self):
        sc = SlateCompleteness(
            slate_date=date(2026, 3, 20),
            is_complete=False,
            reason="Only 5/50 players have actual FP",
            completeness_pct=10.0,
            total_players=50,
            players_with_actuals=5,
            is_future=False,
        )
        assert not sc.is_complete
        assert sc.completeness_pct == 10.0


# ---------------------------------------------------------------------------
# IncompleteSlateError
# ---------------------------------------------------------------------------

class TestIncompleteSlateError:
    def test_is_value_error(self):
        assert issubclass(IncompleteSlateError, ValueError)

    def test_message(self):
        err = IncompleteSlateError("test message")
        assert str(err) == "test message"


# ---------------------------------------------------------------------------
# check_slate_completeness
# ---------------------------------------------------------------------------

class TestCheckSlateCompleteness:
    def test_future_date(self):
        future = date.today() + timedelta(days=5)
        result = check_slate_completeness(future)
        assert not result.is_complete
        assert result.is_future
        assert "future" in result.reason.lower()

    def test_today_date(self):
        result = check_slate_completeness(date.today())
        assert not result.is_complete
        assert result.is_future
        assert "today" in result.reason.lower()

    @patch("yak_core.auto_calibrate._load_ricky_pool_for_date", return_value=pd.DataFrame())
    @patch("yak_core.auto_calibrate._SLATE_ARCHIVE_DIR", new=Path("/nonexistent_slate_dir"))
    @patch("yak_core.auto_calibrate._get_nba_api_key", return_value="")
    def test_no_api_key(self, _mock_key, _mock_archive):
        past = date.today() - timedelta(days=3)
        result = check_slate_completeness(past)
        assert not result.is_complete
        assert "API key" in result.reason

    @patch("yak_core.auto_calibrate._load_ricky_pool_for_date", return_value=pd.DataFrame())
    @patch("yak_core.auto_calibrate._SLATE_ARCHIVE_DIR", new=Path("/nonexistent_slate_dir"))
    @patch("yak_core.auto_calibrate._get_nba_api_key", return_value="test-key")
    @patch("yak_core.live.fetch_actuals_from_api")
    @patch("yak_core.live.fetch_live_dfs")
    def test_complete_date(self, mock_pool, mock_actuals, _mock_key, _mock_archive):
        past = date.today() - timedelta(days=3)
        pool = pd.DataFrame({
            "player_name": [f"Player{i}" for i in range(50)],
            "salary": [5000] * 50,
            "proj": [25.0] * 50,
        })
        mock_pool.return_value = pool
        actuals = pd.DataFrame({
            "player_name": [f"Player{i}" for i in range(50)],
            "actual_fp": [25.0] * 45 + [0.0] * 5,
        })
        mock_actuals.return_value = actuals

        result = check_slate_completeness(past)
        assert result.is_complete
        assert result.completeness_pct == 90.0
        assert result.players_with_actuals == 45
        assert result.total_players == 50

    @patch("yak_core.auto_calibrate._load_ricky_pool_for_date", return_value=pd.DataFrame())
    @patch("yak_core.auto_calibrate._SLATE_ARCHIVE_DIR", new=Path("/nonexistent_slate_dir"))
    @patch("yak_core.auto_calibrate._get_nba_api_key", return_value="test-key")
    @patch("yak_core.live.fetch_actuals_from_api")
    @patch("yak_core.live.fetch_live_dfs")
    def test_incomplete_date_low_actuals(self, mock_pool, mock_actuals, _mock_key, _mock_archive):
        past = date.today() - timedelta(days=2)
        pool = pd.DataFrame({
            "player_name": [f"Player{i}" for i in range(50)],
            "salary": [5000] * 50,
            "proj": [25.0] * 50,
        })
        mock_pool.return_value = pool
        actuals = pd.DataFrame({
            "player_name": [f"Player{i}" for i in range(50)],
            "actual_fp": [15.0] * 10 + [0.0] * 40,
        })
        mock_actuals.return_value = actuals

        result = check_slate_completeness(past)
        assert not result.is_complete
        assert result.completeness_pct == 20.0
        assert "below" in result.reason.lower()

    @patch("yak_core.auto_calibrate._load_ricky_pool_for_date", return_value=pd.DataFrame())
    @patch("yak_core.auto_calibrate._SLATE_ARCHIVE_DIR", new=Path("/nonexistent_slate_dir"))
    @patch("yak_core.auto_calibrate._get_nba_api_key", return_value="test-key")
    @patch("yak_core.live.fetch_actuals_from_api")
    @patch("yak_core.live.fetch_live_dfs")
    def test_custom_threshold(self, mock_pool, mock_actuals, _mock_key, _mock_archive):
        past = date.today() - timedelta(days=3)
        pool = pd.DataFrame({
            "player_name": [f"Player{i}" for i in range(100)],
            "salary": [5000] * 100,
            "proj": [25.0] * 100,
        })
        mock_pool.return_value = pool
        actuals = pd.DataFrame({
            "player_name": [f"Player{i}" for i in range(100)],
            "actual_fp": [20.0] * 50 + [0.0] * 50,
        })
        mock_actuals.return_value = actuals

        # 50% actuals — passes at 40% threshold
        result = check_slate_completeness(past, completeness_threshold=40.0)
        assert result.is_complete

        # 50% actuals — fails at 60% threshold
        result = check_slate_completeness(past, completeness_threshold=60.0)
        assert not result.is_complete

    @patch("yak_core.auto_calibrate._load_ricky_pool_for_date", return_value=pd.DataFrame())
    @patch("yak_core.auto_calibrate._SLATE_ARCHIVE_DIR", new=Path("/nonexistent_slate_dir"))
    @patch("yak_core.auto_calibrate._get_nba_api_key", return_value="test-key")
    @patch("yak_core.live.fetch_live_dfs", side_effect=RuntimeError("API down"))
    def test_pool_fetch_failure(self, _mock_pool, _mock_key, _mock_archive):
        past = date.today() - timedelta(days=3)
        result = check_slate_completeness(past)
        assert not result.is_complete
        assert "Failed to fetch player pool" in result.reason


# ---------------------------------------------------------------------------
# filter_incomplete_dates
# ---------------------------------------------------------------------------

class TestFilterIncompleteDates:
    def test_filters_future_dates_fast(self):
        today = date.today()
        dates = [
            today - timedelta(days=5),
            today - timedelta(days=3),
            today,
            today + timedelta(days=1),
        ]
        with patch(
            "yak_core.auto_calibrate.check_slate_completeness"
        ) as mock_check:
            # Past dates are complete
            mock_check.return_value = SlateCompleteness(
                slate_date=dates[0],
                is_complete=True,
                reason="",
                completeness_pct=90.0,
                total_players=50,
                players_with_actuals=45,
                is_future=False,
            )
            complete, skipped = filter_incomplete_dates(dates)

        # Future/today dates skipped without API call
        assert len(skipped) == 2
        assert len(complete) == 2
        # check_slate_completeness only called for past dates
        assert mock_check.call_count == 2

    def test_progress_callback(self):
        today = date.today()
        dates = [today - timedelta(days=1), today + timedelta(days=1)]
        cb = MagicMock()
        with patch(
            "yak_core.auto_calibrate.check_slate_completeness"
        ) as mock_check:
            mock_check.return_value = SlateCompleteness(
                slate_date=dates[0],
                is_complete=True,
                reason="",
                completeness_pct=95.0,
                total_players=50,
                players_with_actuals=47,
                is_future=False,
            )
            filter_incomplete_dates(dates, progress_callback=cb)

        assert cb.call_count == 2


# ---------------------------------------------------------------------------
# Default threshold sanity
# ---------------------------------------------------------------------------

def test_default_threshold_is_reasonable():
    assert 10 <= _DEFAULT_COMPLETENESS_THRESHOLD <= 80
