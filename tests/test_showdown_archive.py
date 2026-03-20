"""Tests for showdown salary archiving and overlay."""

import os
import tempfile
import pandas as pd
import pytest

# Patch _ARCHIVE_DIR before importing the module
_tmp_dir = tempfile.mkdtemp()


def _patch_archive_dir(monkeypatch):
    monkeypatch.setattr("yak_core.slate_archive._ARCHIVE_DIR", _tmp_dir)


class TestArchiveShowdownSalaries:
    """archive_showdown_salaries persists a parquet with expected schema."""

    def test_creates_parquet_with_correct_columns(self, monkeypatch, tmp_path):
        monkeypatch.setattr("yak_core.slate_archive._ARCHIVE_DIR", str(tmp_path))
        # Stub out sync_feedback_async so it doesn't try GitHub
        monkeypatch.setattr("yak_core.slate_archive.sync_feedback_async", lambda **kw: None)

        from yak_core.slate_archive import archive_showdown_salaries

        players = [
            {"name": "Joel Embiid", "team": "PHI", "position": "C", "salary": 10400, "dk_player_id": "12345"},
            {"name": "Nikola Jokic", "team": "DEN", "position": "C", "salary": 11000, "dk_player_id": "67890"},
        ]
        path = archive_showdown_salaries(
            players=players,
            draft_group_id=99999,
            away="PHI",
            home="DEN",
            slate_date="2026-03-17",
        )
        assert os.path.isfile(path)
        df = pd.read_parquet(path)
        assert len(df) == 2
        assert set(df.columns) >= {"player_name", "team", "salary", "roster_position", "matchup"}
        assert df["roster_position"].unique().tolist() == ["UTIL"]
        assert df["matchup"].unique().tolist() == ["PHI@DEN"]

    def test_empty_players_returns_empty_string(self, monkeypatch, tmp_path):
        monkeypatch.setattr("yak_core.slate_archive._ARCHIVE_DIR", str(tmp_path))
        monkeypatch.setattr("yak_core.slate_archive.sync_feedback_async", lambda **kw: None)

        from yak_core.slate_archive import archive_showdown_salaries
        result = archive_showdown_salaries([], 0, "PHI", "DEN", "2026-03-17")
        assert result == ""


class TestLoadShowdownSalaries:
    """load_showdown_salaries reads back archived data."""

    def test_loads_exact_matchup(self, monkeypatch, tmp_path):
        monkeypatch.setattr("yak_core.slate_archive._ARCHIVE_DIR", str(tmp_path))
        monkeypatch.setattr("yak_core.slate_archive.sync_feedback_async", lambda **kw: None)

        from yak_core.slate_archive import archive_showdown_salaries, load_showdown_salaries

        archive_showdown_salaries(
            players=[
                {"name": "Joel Embiid", "team": "PHI", "position": "C", "salary": 10400, "dk_player_id": "1"},
                {"name": "Nikola Jokic", "team": "DEN", "position": "C", "salary": 11000, "dk_player_id": "2"},
            ],
            draft_group_id=99999,
            away="PHI", home="DEN",
            slate_date="2026-03-17",
        )
        sal_map = load_showdown_salaries("2026-03-17", "PHI", "DEN")
        assert sal_map["Joel Embiid"] == 10400
        assert sal_map["Nikola Jokic"] == 11000

    def test_no_archive_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr("yak_core.slate_archive._ARCHIVE_DIR", str(tmp_path))
        from yak_core.slate_archive import load_showdown_salaries
        assert load_showdown_salaries("2099-01-01") == {}


class TestLoadAllShowdownSalaries:
    """load_all_showdown_salaries merges ALL matchups for a date."""

    def test_merges_multiple_matchups(self, monkeypatch, tmp_path):
        monkeypatch.setattr("yak_core.slate_archive._ARCHIVE_DIR", str(tmp_path))
        monkeypatch.setattr("yak_core.slate_archive.sync_feedback_async", lambda **kw: None)

        from yak_core.slate_archive import archive_showdown_salaries, load_all_showdown_salaries

        archive_showdown_salaries(
            players=[{"name": "Joel Embiid", "team": "PHI", "position": "C", "salary": 10400, "dk_player_id": "1"}],
            draft_group_id=100, away="PHI", home="DEN", slate_date="2026-03-17",
        )
        archive_showdown_salaries(
            players=[{"name": "Shai GA", "team": "OKC", "position": "PG", "salary": 9800, "dk_player_id": "2"}],
            draft_group_id=200, away="OKC", home="ORL", slate_date="2026-03-17",
        )
        merged = load_all_showdown_salaries("2026-03-17")
        assert "Joel Embiid" in merged
        assert "Shai GA" in merged
        assert merged["Joel Embiid"] == 10400
        assert merged["Shai GA"] == 9800

    def test_no_archives_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr("yak_core.slate_archive._ARCHIVE_DIR", str(tmp_path))
        from yak_core.slate_archive import load_all_showdown_salaries
        assert load_all_showdown_salaries("2099-01-01") == {}
