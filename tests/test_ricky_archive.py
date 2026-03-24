"""Tests for yak_core.ricky_archive — RickyArchive single source of truth."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from yak_core.ricky_archive import (
    RICKY_ARCHIVE_PATH,
    has_enough_dates,
    load_pool_for_date,
    scan_ricky_dates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_archive_df(*dates_str) -> pd.DataFrame:
    """Return a minimal archive DataFrame with one player per date."""
    rows = []
    for ds in dates_str:
        rows.append({
            "player_name": f"Player_{ds}",
            "game_date": pd.Timestamp(ds),
            "fantasy_points": 25.0,
            "salary": 6000,
            "pos": "SF",
            "team": "LAL",
            "opp": "BOS",
            "minutes": 30.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# scan_ricky_dates
# ---------------------------------------------------------------------------

class TestScanRickyDates:
    def test_no_archive_file(self, tmp_path):
        """Returns empty list when archive.parquet does not exist."""
        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", tmp_path / "missing.parquet"):
            result = scan_ricky_dates()
        assert result == []

    def test_returns_unique_dates_sorted(self, tmp_path):
        """Unique dates sorted most-recent-first."""
        df = _make_archive_df("2026-03-01", "2026-02-15", "2026-03-01")  # dup
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path):
            result = scan_ricky_dates()

        assert result == [date(2026, 3, 1), date(2026, 2, 15)]

    def test_min_date_filter(self, tmp_path):
        """min_date parameter filters out older entries."""
        df = _make_archive_df("2026-01-10", "2026-02-20", "2026-03-05")
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path):
            result = scan_ricky_dates(min_date="2026-02-01")

        assert date(2026, 1, 10) not in result
        assert date(2026, 2, 20) in result
        assert date(2026, 3, 5) in result

    def test_bad_parquet_returns_empty(self, tmp_path):
        """Gracefully returns empty list when parquet is unreadable."""
        bad_file = tmp_path / "archive.parquet"
        bad_file.write_text("not a parquet file")

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", bad_file):
            result = scan_ricky_dates()

        assert result == []


# ---------------------------------------------------------------------------
# load_pool_for_date
# ---------------------------------------------------------------------------

class TestLoadPoolForDate:
    def test_no_archive_returns_empty(self, tmp_path):
        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", tmp_path / "missing.parquet"):
            result = load_pool_for_date(date(2026, 3, 1))
        assert result.empty

    def test_date_not_in_archive_returns_empty(self, tmp_path):
        df = _make_archive_df("2026-02-15")
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path):
            result = load_pool_for_date(date(2026, 3, 1))

        assert result.empty

    def test_returns_pool_for_matching_date(self, tmp_path):
        df = _make_archive_df("2026-03-01", "2026-02-15")
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path):
            result = load_pool_for_date(date(2026, 3, 1))

        assert not result.empty
        assert len(result) == 1
        assert "Player_2026-03-01" in result["player_name"].values

    def test_pool_has_required_columns(self, tmp_path):
        df = _make_archive_df("2026-03-01")
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path):
            result = load_pool_for_date(date(2026, 3, 1))

        for col in ("player_name", "salary", "proj", "actual_fp", "own_proj", "slate_date"):
            assert col in result.columns, f"Missing column: {col}"

    def test_proj_equals_actual_fp(self, tmp_path):
        """proj and actual_fp should both reflect fantasy_points."""
        df = _make_archive_df("2026-03-01")
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path):
            result = load_pool_for_date(date(2026, 3, 1))

        assert (result["proj"] == result["actual_fp"]).all()
        assert result["proj"].iloc[0] == 25.0


# ---------------------------------------------------------------------------
# has_enough_dates
# ---------------------------------------------------------------------------

class TestHasEnoughDates:
    def test_false_when_archive_missing(self, tmp_path):
        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", tmp_path / "missing.parquet"):
            assert not has_enough_dates(3)

    def test_false_when_too_few_dates(self, tmp_path):
        df = _make_archive_df("2026-03-01", "2026-02-15")
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path):
            assert not has_enough_dates(3)

    def test_true_when_enough_dates(self, tmp_path):
        df = _make_archive_df("2026-03-01", "2026-02-15", "2026-01-10")
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        with patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path):
            assert has_enough_dates(3)


# ---------------------------------------------------------------------------
# scan_archive_dates in auto_calibrate uses RickyArchive as primary source
# ---------------------------------------------------------------------------

class TestScanArchiveDatesIntegration:
    def test_ricky_archive_takes_priority(self, tmp_path):
        """Dates from RickyArchive appear in scan_archive_dates result."""
        df = _make_archive_df("2026-03-01", "2026-02-15")
        archive_path = tmp_path / "archive.parquet"
        df.to_parquet(archive_path, index=False)

        # Empty slate_archive directory
        slate_dir = tmp_path / "slate_archive"
        slate_dir.mkdir()

        with (
            patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", archive_path),
            patch("yak_core.auto_calibrate._SLATE_ARCHIVE_DIR", slate_dir),
        ):
            from yak_core.auto_calibrate import scan_archive_dates
            result = scan_archive_dates()

        assert date(2026, 3, 1) in result
        assert date(2026, 2, 15) in result

    def test_returns_empty_when_both_sources_empty(self, tmp_path):
        """Empty list when neither archive nor slate_archive has data."""
        slate_dir = tmp_path / "slate_archive"
        slate_dir.mkdir()

        with (
            patch("yak_core.ricky_archive.RICKY_ARCHIVE_PATH", tmp_path / "missing.parquet"),
            patch("yak_core.auto_calibrate._SLATE_ARCHIVE_DIR", slate_dir),
        ):
            from yak_core.auto_calibrate import scan_archive_dates
            result = scan_archive_dates()

        assert result == []
