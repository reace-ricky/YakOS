"""Tests for yak_core/dvp.py — DvP baseline loader and utilities."""

import io
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from yak_core.dvp import (
    DVP_STALE_DAYS,
    compute_league_averages,
    dvp_staleness_days,
    load_dvp_table,
    parse_dvp_upload,
    save_dvp_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_bytes(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode())


def _fantasy_pros_csv() -> io.BytesIO:
    return _csv_bytes(
        "Team,PG_FPPG_Allowed,SG_FPPG_Allowed,SF_FPPG_Allowed,PF_FPPG_Allowed,C_FPPG_Allowed\n"
        "LAL,44.1,42.0,40.5,39.8,38.2\n"
        "BOS,41.3,39.7,38.1,37.6,36.0\n"
        "GSW,46.0,43.5,41.0,40.2,39.5\n"
    )


def _human_readable_csv() -> io.BytesIO:
    return _csv_bytes(
        "Team,PG Allowed,SG Allowed,SF Allowed,PF Allowed,C Allowed\n"
        "LAL,44.1,42.0,40.5,39.8,38.2\n"
        "BOS,41.3,39.7,38.1,37.6,36.0\n"
    )


def _already_canonical_csv() -> io.BytesIO:
    return _csv_bytes(
        "Team,PG,SG,SF,PF,C\n"
        "LAL,44.1,42.0,40.5,39.8,38.2\n"
        "BOS,41.3,39.7,38.1,37.6,36.0\n"
    )


def _opp_team_col_csv() -> io.BytesIO:
    """Team column named 'Opponent' instead of 'Team'."""
    return _csv_bytes(
        "Opponent,PG_FPPG_Allowed,SG_FPPG_Allowed,SF_FPPG_Allowed,PF_FPPG_Allowed,C_FPPG_Allowed\n"
        "LAL,44.1,42.0,40.5,39.8,38.2\n"
    )


def _missing_positions_csv() -> io.BytesIO:
    """Only PG and C columns."""
    return _csv_bytes(
        "Team,PG_FPPG_Allowed,C_FPPG_Allowed\n"
        "LAL,44.1,38.2\n"
        "BOS,41.3,36.0\n"
    )


def _non_numeric_csv() -> io.BytesIO:
    return _csv_bytes(
        "Team,PG_FPPG_Allowed,SG_FPPG_Allowed\n"
        "LAL,44.1,N/A\n"
        "BOS,41.3,39.7\n"
    )


# ---------------------------------------------------------------------------
# parse_dvp_upload
# ---------------------------------------------------------------------------

class TestParseDvpUpload:
    def test_fantasypros_format_columns(self):
        df = parse_dvp_upload(_fantasy_pros_csv())
        assert set(df.columns) == {"Team", "PG", "SG", "SF", "PF", "C"}

    def test_fantasypros_row_count(self):
        df = parse_dvp_upload(_fantasy_pros_csv())
        assert len(df) == 3

    def test_fantasypros_values(self):
        df = parse_dvp_upload(_fantasy_pros_csv())
        row = df[df["Team"] == "LAL"].iloc[0]
        assert row["PG"] == pytest.approx(44.1)
        assert row["C"] == pytest.approx(38.2)

    def test_human_readable_format(self):
        df = parse_dvp_upload(_human_readable_csv())
        assert "PG" in df.columns
        assert "C" in df.columns

    def test_already_canonical_format(self):
        df = parse_dvp_upload(_already_canonical_csv())
        assert list(df.columns) == ["Team", "PG", "SG", "SF", "PF", "C"]

    def test_opponent_team_column_renamed(self):
        df = parse_dvp_upload(_opp_team_col_csv())
        assert "Team" in df.columns
        assert "Opponent" not in df.columns

    def test_missing_positions_only_available_cols(self):
        df = parse_dvp_upload(_missing_positions_csv())
        assert "PG" in df.columns
        assert "C" in df.columns
        assert "SG" not in df.columns

    def test_non_numeric_becomes_nan(self):
        df = parse_dvp_upload(_non_numeric_csv())
        assert pd.isna(df.loc[df["Team"] == "LAL", "SG"].values[0])
        assert df.loc[df["Team"] == "BOS", "SG"].values[0] == pytest.approx(39.7)

    def test_position_cols_are_float(self):
        df = parse_dvp_upload(_fantasy_pros_csv())
        for pos in ["PG", "SG", "SF", "PF", "C"]:
            assert df[pos].dtype == float

    def test_no_extra_original_cols_leaked(self):
        """Columns not in the keep-list should be dropped."""
        raw = _csv_bytes(
            "Team,PG_FPPG_Allowed,Extra_Col\n"
            "LAL,44.1,99\n"
        )
        df = parse_dvp_upload(raw)
        assert "Extra_Col" not in df.columns


# ---------------------------------------------------------------------------
# compute_league_averages
# ---------------------------------------------------------------------------

class TestComputeLeagueAverages:
    def _make_dvp(self):
        return pd.DataFrame({
            "Team": ["LAL", "BOS", "GSW"],
            "PG": [44.1, 41.3, 46.0],
            "SG": [42.0, 39.7, 43.5],
            "SF": [40.5, 38.1, 41.0],
            "PF": [39.8, 37.6, 40.2],
            "C": [38.2, 36.0, 39.5],
        })

    def test_keys_are_positions(self):
        avgs = compute_league_averages(self._make_dvp())
        assert set(avgs.keys()) == {"PG", "SG", "SF", "PF", "C"}

    def test_pg_average(self):
        avgs = compute_league_averages(self._make_dvp())
        expected = round((44.1 + 41.3 + 46.0) / 3, 2)
        assert avgs["PG"] == pytest.approx(expected)

    def test_c_average(self):
        avgs = compute_league_averages(self._make_dvp())
        expected = round((38.2 + 36.0 + 39.5) / 3, 2)
        assert avgs["C"] == pytest.approx(expected)

    def test_rounded_to_two_decimals(self):
        avgs = compute_league_averages(self._make_dvp())
        for v in avgs.values():
            assert v == round(v, 2)

    def test_missing_position_col_excluded(self):
        df = pd.DataFrame({"Team": ["LAL"], "PG": [44.1]})
        avgs = compute_league_averages(df)
        assert "PG" in avgs
        assert "SG" not in avgs

    def test_all_nan_position_excluded(self):
        df = pd.DataFrame({"Team": ["LAL", "BOS"], "PG": [float("nan"), float("nan")]})
        avgs = compute_league_averages(df)
        assert "PG" not in avgs

    def test_partial_nan_ignores_nan(self):
        df = pd.DataFrame({"Team": ["LAL", "BOS"], "PG": [44.0, float("nan")]})
        avgs = compute_league_averages(df)
        assert avgs["PG"] == pytest.approx(44.0)

    def test_single_team(self):
        df = pd.DataFrame({"Team": ["LAL"], "C": [38.2]})
        avgs = compute_league_averages(df)
        assert avgs["C"] == pytest.approx(38.2)


# ---------------------------------------------------------------------------
# save_dvp_table / load_dvp_table
# ---------------------------------------------------------------------------

class TestSaveLoadDvpTable:
    def _make_df(self):
        return pd.DataFrame({
            "Team": ["LAL", "BOS"],
            "PG": [44.1, 41.3],
            "C": [38.2, 36.0],
        })

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "dvp.csv"
        save_dvp_table(self._make_df(), path)
        assert path.exists()

    def test_load_returns_dataframe(self, tmp_path):
        path = tmp_path / "dvp.csv"
        save_dvp_table(self._make_df(), path)
        df = load_dvp_table(path)
        assert df is not None
        assert isinstance(df, pd.DataFrame)

    def test_roundtrip_values(self, tmp_path):
        path = tmp_path / "dvp.csv"
        original = self._make_df()
        save_dvp_table(original, path)
        loaded = load_dvp_table(path)
        assert loaded["PG"].tolist() == pytest.approx([44.1, 41.3])

    def test_load_missing_file_returns_none(self, tmp_path):
        result = load_dvp_table(tmp_path / "nonexistent.csv")
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "subdir" / "nested" / "dvp.csv"
        save_dvp_table(self._make_df(), path)
        assert path.exists()


# ---------------------------------------------------------------------------
# dvp_staleness_days
# ---------------------------------------------------------------------------

class TestDvpStalenessDays:
    def test_returns_none_for_missing_file(self, tmp_path):
        assert dvp_staleness_days(tmp_path / "no_file.csv") is None

    def test_new_file_is_near_zero_days(self, tmp_path):
        path = tmp_path / "dvp.csv"
        path.write_text("Team,PG\nLAL,44.1\n")
        age = dvp_staleness_days(path)
        assert age is not None
        assert age < 1.0  # freshly written file should be < 1 day old

    def test_old_file_exceeds_stale_threshold(self, tmp_path):
        path = tmp_path / "dvp.csv"
        path.write_text("Team,PG\nLAL,44.1\n")
        # Back-date the modification time by 8 days
        old_mtime = time.time() - (8 * 86400)
        os.utime(path, (old_mtime, old_mtime))
        age = dvp_staleness_days(path)
        assert age is not None
        assert age > DVP_STALE_DAYS

    def test_stale_days_constant_is_7(self):
        assert DVP_STALE_DAYS == 7
