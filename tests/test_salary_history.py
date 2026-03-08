"""Tests for yak_core/salary_history.py.

Covers:
- _to_fl_date date reformatting
- SalaryHistoryClient.get_draft_group_ids JSON parsing
- SalaryHistoryClient.get_draftables DataFrame construction
- SalaryHistoryClient._pick_primary_group selection logic
- SalaryHistoryClient.load_cached_salaries / save_salaries cache round-trip
- SalaryHistoryClient.get_historical_salaries orchestration
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from yak_core.salary_history import SalaryHistoryClient, _to_fl_date


# ---------------------------------------------------------------------------
# _to_fl_date helpers
# ---------------------------------------------------------------------------

class TestToFlDate:
    def test_standard_date(self):
        assert _to_fl_date("2026-02-28") == "2_28_2026"

    def test_no_zero_padding_on_month(self):
        assert _to_fl_date("2026-03-05") == "3_5_2026"

    def test_december(self):
        assert _to_fl_date("2025-12-31") == "12_31_2025"

    def test_january_first(self):
        assert _to_fl_date("2026-01-01") == "1_1_2026"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            _to_fl_date("20260228")


# ---------------------------------------------------------------------------
# get_draft_group_ids
# ---------------------------------------------------------------------------

class TestGetDraftGroupIds:
    def _client(self):
        return SalaryHistoryClient()

    def test_parses_valid_response(self):
        payload = [
            {
                "DraftGroupId": 12345,
                "GameCount": 8,
                "ContestSuffix": "",
                "DisplayName": "NBA Main Slate",
                "StartTime": "2026-02-28T18:00:00",
            },
            {
                "DraftGroupId": 12346,
                "GameCount": 3,
                "ContestSuffix": "(Night)",
                "DisplayName": "NBA Night Slate",
                "StartTime": "2026-02-28T21:00:00",
            },
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.salary_history.requests.get", return_value=mock_resp):
            groups = self._client().get_draft_group_ids("2026-02-28")

        assert len(groups) == 2
        assert groups[0]["draft_group_id"] == 12345
        assert groups[0]["suffix"] == ""
        assert groups[1]["draft_group_id"] == 12346
        assert groups[1]["suffix"] == "(Night)"

    def test_empty_array_returns_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.salary_history.requests.get", return_value=mock_resp):
            groups = self._client().get_draft_group_ids("2026-02-28")

        assert groups == []

    def test_request_failure_returns_empty_list(self):
        with patch(
            "yak_core.salary_history.requests.get",
            side_effect=Exception("Connection refused"),
        ):
            groups = self._client().get_draft_group_ids("2026-02-28")

        assert groups == []

    def test_non_list_response_returns_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "not found"}
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.salary_history.requests.get", return_value=mock_resp):
            groups = self._client().get_draft_group_ids("2026-02-28")

        assert groups == []

    def test_uses_fl_date_format_in_url(self):
        captured_url = []

        def fake_get(url, **kwargs):
            captured_url.append(url)
            mock_resp = MagicMock()
            mock_resp.json.return_value = []
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        with patch("yak_core.salary_history.requests.get", side_effect=fake_get):
            self._client().get_draft_group_ids("2026-03-05")

        assert "3_5_2026" in captured_url[0]


# ---------------------------------------------------------------------------
# get_draftables
# ---------------------------------------------------------------------------

class TestGetDraftables:
    def _client(self):
        return SalaryHistoryClient()

    def _make_draftable(self, name, position, team, salary, player_id):
        return {
            "displayName": name,
            "position": position,
            "teamAbbreviation": team,
            "salary": salary,
            "playerId": player_id,
        }

    def test_parses_draftables(self):
        payload = {
            "draftables": [
                self._make_draftable("LeBron James", "SF", "LAL", 9800, 214152),
                self._make_draftable("Stephen Curry", "PG", "GSW", 10200, 214153),
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.salary_history.requests.get", return_value=mock_resp):
            df = self._client().get_draftables(12345)

        assert len(df) == 2
        assert list(df.columns) == ["player_name", "position", "team", "opp", "game_info", "game_time", "salary", "player_dk_id"]
        assert df.iloc[0]["player_name"] == "LeBron James"
        assert df.iloc[0]["salary"] == 9800
        assert df.iloc[1]["team"] == "GSW"

    def test_empty_draftables_returns_empty_df(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"draftables": []}
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.salary_history.requests.get", return_value=mock_resp):
            df = self._client().get_draftables(12345)

        assert df.empty
        assert "player_name" in df.columns

    def test_request_failure_returns_empty_df(self):
        with patch(
            "yak_core.salary_history.requests.get",
            side_effect=Exception("timeout"),
        ):
            df = self._client().get_draftables(12345)

        assert df.empty

    def test_team_normalized_to_upper(self):
        payload = {
            "draftables": [
                {"displayName": "X", "position": "PG", "teamAbbreviation": "lal", "salary": 5000, "playerId": 1}
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.salary_history.requests.get", return_value=mock_resp):
            df = self._client().get_draftables(99)

        assert df.iloc[0]["team"] == "LAL"


# ---------------------------------------------------------------------------
# _pick_primary_group
# ---------------------------------------------------------------------------

class TestPickPrimaryGroup:
    def test_empty_list_returns_none(self):
        assert SalaryHistoryClient._pick_primary_group([]) is None

    def test_single_group_returned(self):
        groups = [{"draft_group_id": 1, "game_count": 5, "suffix": "", "display_name": "Main", "start_time": ""}]
        result = SalaryHistoryClient._pick_primary_group(groups)
        assert result["draft_group_id"] == 1

    def test_prefers_empty_suffix_over_night(self):
        groups = [
            {"draft_group_id": 1, "game_count": 3, "suffix": "(Night)", "display_name": "Night", "start_time": ""},
            {"draft_group_id": 2, "game_count": 8, "suffix": "", "display_name": "Main", "start_time": ""},
        ]
        result = SalaryHistoryClient._pick_primary_group(groups)
        assert result["draft_group_id"] == 2

    def test_picks_highest_game_count_among_main_slates(self):
        groups = [
            {"draft_group_id": 1, "game_count": 6, "suffix": "", "display_name": "Main A", "start_time": ""},
            {"draft_group_id": 2, "game_count": 10, "suffix": "", "display_name": "Main B", "start_time": ""},
        ]
        result = SalaryHistoryClient._pick_primary_group(groups)
        assert result["draft_group_id"] == 2

    def test_falls_back_to_highest_game_count_when_all_have_suffix(self):
        groups = [
            {"draft_group_id": 1, "game_count": 3, "suffix": "(Night)", "display_name": "Night", "start_time": ""},
            {"draft_group_id": 2, "game_count": 7, "suffix": "(Early)", "display_name": "Early", "start_time": ""},
        ]
        result = SalaryHistoryClient._pick_primary_group(groups)
        assert result["draft_group_id"] == 2


# ---------------------------------------------------------------------------
# Cache (load / save) round-trip
# ---------------------------------------------------------------------------

class TestSalaryCache:
    def test_load_returns_none_when_no_cache(self, tmp_path):
        client = SalaryHistoryClient()
        # Patch the cache dir to a tmp location
        with patch("yak_core.salary_history._SALARY_CACHE_DIR", tmp_path):
            result = client.load_cached_salaries("2026-01-01")
        assert result is None

    def test_save_and_load_round_trip(self, tmp_path):
        df = pd.DataFrame({
            "player_name": ["LeBron James"],
            "position": ["SF"],
            "team": ["LAL"],
            "salary": [9800],
            "player_dk_id": [214152],
        })
        client = SalaryHistoryClient()
        with patch("yak_core.salary_history._SALARY_CACHE_DIR", tmp_path):
            client.save_salaries("2026-02-28", df)
            loaded = client.load_cached_salaries("2026-02-28")

        assert loaded is not None
        assert len(loaded) == 1
        assert loaded.iloc[0]["player_name"] == "LeBron James"

    def test_save_creates_directory_if_missing(self, tmp_path):
        nested = tmp_path / "nested" / "cache"
        df = pd.DataFrame({"player_name": ["A"], "salary": [5000]})
        client = SalaryHistoryClient()
        with patch("yak_core.salary_history._SALARY_CACHE_DIR", nested):
            client.save_salaries("2026-03-01", df)
        assert (nested / "2026-03-01.parquet").exists()


# ---------------------------------------------------------------------------
# get_historical_salaries orchestration
# ---------------------------------------------------------------------------

class TestGetHistoricalSalaries:
    def _make_groups(self):
        return [
            {"draft_group_id": 99001, "game_count": 8, "suffix": "", "display_name": "Main", "start_time": ""},
        ]

    def _make_draftables_df(self):
        return pd.DataFrame({
            "player_name": ["Player A", "Player B"],
            "position": ["PG", "SG"],
            "team": ["LAL", "GSW"],
            "salary": [8000, 7500],
            "player_dk_id": [1, 2],
        })

    def test_returns_df_and_attaches_draft_group_id(self, tmp_path):
        client = SalaryHistoryClient()
        with (
            patch.object(client, "get_draft_group_ids", return_value=self._make_groups()),
            patch.object(client, "get_draftables", return_value=self._make_draftables_df()),
            patch("yak_core.salary_history._SALARY_CACHE_DIR", tmp_path),
            patch("yak_core.salary_history.time.sleep"),
        ):
            df = client.get_historical_salaries("2026-02-28")

        assert len(df) == 2
        assert df.attrs.get("draft_group_id") == 99001

    def test_saves_cache_file(self, tmp_path):
        client = SalaryHistoryClient()
        with (
            patch.object(client, "get_draft_group_ids", return_value=self._make_groups()),
            patch.object(client, "get_draftables", return_value=self._make_draftables_df()),
            patch("yak_core.salary_history._SALARY_CACHE_DIR", tmp_path),
            patch("yak_core.salary_history.time.sleep"),
        ):
            client.get_historical_salaries("2026-02-28")

        assert (tmp_path / "2026-02-28.parquet").exists()

    def test_returns_empty_df_when_no_groups(self, tmp_path):
        client = SalaryHistoryClient()
        with (
            patch.object(client, "get_draft_group_ids", return_value=[]),
            patch("yak_core.salary_history._SALARY_CACHE_DIR", tmp_path),
            patch("yak_core.salary_history.time.sleep"),
        ):
            df = client.get_historical_salaries("2026-02-28")

        assert df.empty

    def test_sleeps_between_calls(self, tmp_path):
        client = SalaryHistoryClient()
        sleep_calls = []
        with (
            patch.object(client, "get_draft_group_ids", return_value=self._make_groups()),
            patch.object(client, "get_draftables", return_value=self._make_draftables_df()),
            patch("yak_core.salary_history._SALARY_CACHE_DIR", tmp_path),
            patch("yak_core.salary_history.time.sleep", side_effect=sleep_calls.append),
        ):
            client.get_historical_salaries("2026-02-28")

        assert len(sleep_calls) >= 1
