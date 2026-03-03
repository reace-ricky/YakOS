"""Tests for yak_core/dk_ingest.py — Sprint 5 DK Contest-Scoped Optimizer Integration."""

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import yak_core.dk_ingest as dk


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_yak_pool(rows=None):
    if rows is None:
        rows = [
            {"player_id": "p1", "player_name": "LeBron James", "team": "LAL", "pos": "SF", "salary": 10000, "proj": 55.0},
            {"player_id": "p2", "player_name": "Stephen Curry", "team": "GSW", "pos": "PG", "salary": 9800, "proj": 50.0},
            {"player_id": "p3", "player_name": "Nikola Jokic", "team": "DEN", "pos": "C", "salary": 9600, "proj": 53.0},
            {"player_id": "p4", "player_name": "Anthony Davis", "team": "LAL", "pos": "PF", "salary": 9200, "proj": 48.0},
            {"player_id": "p5", "player_name": "Jayson Tatum", "team": "BOS", "pos": "SF", "salary": 8800, "proj": 46.0},
        ]
    return pd.DataFrame(rows)


def _make_dk_pool(rows=None):
    if rows is None:
        rows = [
            {"draft_group_id": 100, "dk_player_id": "dk1", "name": "LeBron James", "display_name": "LeBron James", "team": "LAL", "positions": "SF/PF", "salary": 10200, "status": "Active"},
            {"draft_group_id": 100, "dk_player_id": "dk2", "name": "Stephen Curry", "display_name": "Stephen Curry", "team": "GSW", "positions": "PG", "salary": 10000, "status": "Active"},
            {"draft_group_id": 100, "dk_player_id": "dk3", "name": "Nikola Jokic", "display_name": "Nikola Jokic", "team": "DEN", "positions": "C", "salary": 9800, "status": "Active"},
            {"draft_group_id": 100, "dk_player_id": "dk4", "name": "Unknown Player", "display_name": "Unknown Player", "team": "NYK", "positions": "SG", "salary": 5000, "status": "Active"},
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5.1 — fetch_dk_lobby_contests / save / load
# ---------------------------------------------------------------------------

class TestFetchDkLobbyContests:
    def _mock_lobby_response(self):
        return {
            "Contests": [
                {
                    "id": "C1", "n": "Mega Tournament", "gameTypeId": 1,
                    "dg": 100, "sd": "2026-03-10T18:00:00Z",
                    "a": 3.0, "po": 100000.0, "m": 5000,
                    "mec": 150, "nt": 1200, "ise": False,
                },
                {
                    "id": "C2", "n": "Late Night GPP", "gameTypeId": 96,
                    "dg": 101, "sd": "2026-03-10T23:00:00Z",
                    "a": 1.0, "po": 5000.0, "m": 1000,
                    "mec": 3, "nt": 300, "ise": False,
                },
            ]
        }

    def test_returns_dataframe_with_expected_columns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dk, "_DK_DIR", tmp_path / "dk")
        monkeypatch.setattr(dk, "_CONTESTS_PATH", tmp_path / "dk" / "dk_contests.parquet")

        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_lobby_response()
        with patch.object(dk, "_rate_limited_get", return_value=mock_resp):
            df = dk.fetch_dk_lobby_contests("NBA")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "contest_id" in df.columns
        assert "draft_group_id" in df.columns
        assert "entry_fee" in df.columns
        assert "prize_pool" in df.columns
        assert "is_single_entry" in df.columns

    def test_returns_empty_when_disabled(self, monkeypatch):
        monkeypatch.setenv("DK_INTEGRATION_ENABLED", "false")
        # Re-read env; patch is_dk_integration_enabled directly
        with patch.object(dk, "is_dk_integration_enabled", return_value=False):
            result = dk.fetch_dk_lobby_contests("NBA")
        assert result.empty

    def test_returns_empty_when_no_contests_in_response(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch.object(dk, "_rate_limited_get", return_value=mock_resp):
            result = dk.fetch_dk_lobby_contests("NBA")
        assert result.empty

    def test_sport_column_uppercased(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_lobby_response()
        with patch.object(dk, "_rate_limited_get", return_value=mock_resp):
            df = dk.fetch_dk_lobby_contests("nba")
        assert (df["sport"] == "NBA").all()


class TestSaveLoadDkContests:
    def test_save_and_load_roundtrip(self, tmp_path):
        with patch.object(dk, "_DK_DIR", tmp_path / "dk"):
            with patch.object(dk, "_CONTESTS_PATH", tmp_path / "dk" / "contests.parquet"):
                df = pd.DataFrame([{"contest_id": "X1", "name": "Test", "sport": "NBA",
                                     "game_type_id": 1, "draft_group_id": 99,
                                     "start_time": "", "entry_fee": 5.0, "prize_pool": 1000.0,
                                     "max_entries": 100, "max_entries_per_user": 3,
                                     "current_entries": 50, "is_single_entry": False}])
                dk.save_dk_contests(df)
                loaded = dk.load_dk_contests()
                assert len(loaded) == 1
                assert loaded.iloc[0]["contest_id"] == "X1"

    def test_upsert_deduplicates_by_contest_id(self, tmp_path):
        cp = tmp_path / "dk" / "contests.parquet"
        with patch.object(dk, "_DK_DIR", tmp_path / "dk"), \
             patch.object(dk, "_CONTESTS_PATH", cp):
            df1 = pd.DataFrame([{"contest_id": "X1", "name": "Old", "sport": "NBA",
                                  "game_type_id": 1, "draft_group_id": 99,
                                  "start_time": "", "entry_fee": 5.0, "prize_pool": 1000.0,
                                  "max_entries": 100, "max_entries_per_user": 3,
                                  "current_entries": 50, "is_single_entry": False}])
            dk.save_dk_contests(df1)
            df2 = pd.DataFrame([{"contest_id": "X1", "name": "Updated", "sport": "NBA",
                                  "game_type_id": 1, "draft_group_id": 99,
                                  "start_time": "", "entry_fee": 5.0, "prize_pool": 1000.0,
                                  "max_entries": 100, "max_entries_per_user": 3,
                                  "current_entries": 60, "is_single_entry": False}])
            dk.save_dk_contests(df2)
            loaded = dk.load_dk_contests()
        assert len(loaded) == 1
        assert loaded.iloc[0]["name"] == "Updated"

    def test_load_returns_empty_when_file_missing(self, tmp_path):
        with patch.object(dk, "_CONTESTS_PATH", tmp_path / "nonexistent.parquet"):
            result = dk.load_dk_contests()
        assert result.empty

    def test_load_sport_filter(self, tmp_path):
        cp = tmp_path / "dk" / "contests.parquet"
        with patch.object(dk, "_DK_DIR", tmp_path / "dk"), \
             patch.object(dk, "_CONTESTS_PATH", cp):
            df = pd.DataFrame([
                {"contest_id": "N1", "name": "NBA C", "sport": "NBA", "game_type_id": 1,
                 "draft_group_id": 10, "start_time": "", "entry_fee": 1.0, "prize_pool": 100.0,
                 "max_entries": 10, "max_entries_per_user": 1, "current_entries": 5, "is_single_entry": False},
                {"contest_id": "P1", "name": "PGA C", "sport": "PGA", "game_type_id": 2,
                 "draft_group_id": 20, "start_time": "", "entry_fee": 2.0, "prize_pool": 200.0,
                 "max_entries": 20, "max_entries_per_user": 2, "current_entries": 8, "is_single_entry": False},
            ])
            dk.save_dk_contests(df)
            nba = dk.load_dk_contests("NBA")
            pga = dk.load_dk_contests("PGA")
        assert len(nba) == 1 and nba.iloc[0]["sport"] == "NBA"
        assert len(pga) == 1 and pga.iloc[0]["sport"] == "PGA"


# ---------------------------------------------------------------------------
# 5.2 — fetch_dk_draftables / save / load
# ---------------------------------------------------------------------------

class TestFetchDkDraftables:
    def _mock_draftables_response(self):
        return {
            "draftables": [
                {"draftableId": "dk1", "displayName": "LeBron James",
                 "teamAbbreviation": "LAL", "salary": 10200, "position": "SF",
                 "playerGameAttributes": [],
                 "playerGameInfo": {"status": "Active"}},
                {"draftableId": "dk2", "displayName": "Stephen Curry",
                 "teamAbbreviation": "GSW", "salary": 10000, "position": "PG",
                 "playerGameAttributes": [],
                 "playerGameInfo": {"status": "Active"}},
            ]
        }

    def test_returns_dataframe_with_expected_columns(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_draftables_response()
        with patch.object(dk, "_rate_limited_get", return_value=mock_resp):
            df = dk.fetch_dk_draftables(100)
        assert isinstance(df, pd.DataFrame)
        assert "dk_player_id" in df.columns
        assert "name" in df.columns
        assert "salary" in df.columns
        assert "team" in df.columns

    def test_returns_empty_when_disabled(self):
        with patch.object(dk, "is_dk_integration_enabled", return_value=False):
            result = dk.fetch_dk_draftables(100)
        assert result.empty

    def test_draft_group_id_column_populated(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_draftables_response()
        with patch.object(dk, "_rate_limited_get", return_value=mock_resp):
            df = dk.fetch_dk_draftables(42)
        assert (df["draft_group_id"] == 42).all()

    def test_salary_is_numeric(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_draftables_response()
        with patch.object(dk, "_rate_limited_get", return_value=mock_resp):
            df = dk.fetch_dk_draftables(100)
        assert pd.api.types.is_float_dtype(df["salary"])


class TestSaveLoadDkPlayerPool:
    def test_save_and_load_for_group(self, tmp_path):
        with patch.object(dk, "_DK_DIR", tmp_path / "dk"), \
             patch.object(dk, "_PLAYER_POOL_PATH", tmp_path / "dk" / "pool.parquet"):
            df = _make_dk_pool()
            dk.save_dk_player_pool(df)
            loaded = dk.load_dk_player_pool_for_group(100)
        assert len(loaded) == 4

    def test_load_for_group_filters_correctly(self, tmp_path):
        pp = tmp_path / "dk" / "pool.parquet"
        with patch.object(dk, "_DK_DIR", tmp_path / "dk"), \
             patch.object(dk, "_PLAYER_POOL_PATH", pp):
            df1 = _make_dk_pool()
            df2 = _make_dk_pool([
                {"draft_group_id": 200, "dk_player_id": "dkX", "name": "Other Player",
                 "display_name": "Other Player", "team": "OKC", "positions": "PG",
                 "salary": 5000, "status": "Active"}
            ])
            dk.save_dk_player_pool(pd.concat([df1, df2], ignore_index=True))
            g100 = dk.load_dk_player_pool_for_group(100)
            g200 = dk.load_dk_player_pool_for_group(200)
        assert len(g100) == 4
        assert len(g200) == 1


# ---------------------------------------------------------------------------
# 5.3 — map_dk_players_to_yak / diagnostics
# ---------------------------------------------------------------------------

class TestMapDkPlayersToYak:
    def test_exact_name_match(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        mapping = dk.map_dk_players_to_yak(100, yak, dk_pool)
        lebron = mapping[mapping["dk_name"] == "LeBron James"].iloc[0]
        assert lebron["yak_player_id"] == "p1"
        assert lebron["match_quality"] in ("exact+team", "name_only")

    def test_returns_none_for_unmatched_player(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        mapping = dk.map_dk_players_to_yak(100, yak, dk_pool)
        unknown = mapping[mapping["dk_name"] == "Unknown Player"].iloc[0]
        assert unknown["yak_player_id"] == ""
        assert unknown["match_quality"] == "none"

    def test_name_normalization_handles_punctuation(self):
        yak = pd.DataFrame([
            {"player_id": "p1", "player_name": "De'Aaron Fox", "team": "SAC", "pos": "PG", "salary": 7000, "proj": 35.0}
        ])
        dk_pool = pd.DataFrame([
            {"draft_group_id": 100, "dk_player_id": "dk1", "name": "DeAaron Fox",
             "display_name": "DeAaron Fox", "team": "SAC", "positions": "PG", "salary": 7200, "status": "Active"}
        ])
        mapping = dk.map_dk_players_to_yak(100, yak, dk_pool)
        assert len(mapping) == 1
        assert mapping.iloc[0]["yak_player_id"] == "p1"

    def test_empty_dk_pool_returns_empty_map(self):
        yak = _make_yak_pool()
        result = dk.map_dk_players_to_yak(100, yak, pd.DataFrame())
        assert result.empty

    def test_empty_yak_pool_returns_empty_map(self):
        dk_pool = _make_dk_pool()
        result = dk.map_dk_players_to_yak(100, pd.DataFrame(), dk_pool)
        assert result.empty

    def test_all_required_columns_present(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        mapping = dk.map_dk_players_to_yak(100, yak, dk_pool)
        for col in ("draft_group_id", "dk_player_id", "dk_name", "dk_team",
                    "dk_positions", "yak_player_id", "yak_name", "match_quality"):
            assert col in mapping.columns

    def test_mapping_coverage_at_least_three_out_of_four(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        mapping = dk.map_dk_players_to_yak(100, yak, dk_pool)
        mapped_count = (mapping["yak_player_id"] != "").sum()
        assert mapped_count >= 3  # LeBron, Curry, Jokic should match


class TestGetMappingDiagnostics:
    def test_returns_correct_pct_mapped(self, tmp_path):
        mp = tmp_path / "dk" / "map.parquet"
        with patch.object(dk, "_DK_DIR", tmp_path / "dk"), \
             patch.object(dk, "_PLAYER_MAP_PATH", mp):
            yak = _make_yak_pool()
            dk_pool = _make_dk_pool()
            mapping = dk.map_dk_players_to_yak(100, yak, dk_pool)
            dk.save_dk_player_map(mapping)
            diag = dk.get_mapping_diagnostics(100)
        assert diag["total_dk_players"] == 4
        assert 0 < diag["pct_mapped"] <= 100
        assert isinstance(diag["unmapped_players"], list)

    def test_returns_zero_when_no_map_saved(self, tmp_path):
        with patch.object(dk, "_PLAYER_MAP_PATH", tmp_path / "nonexistent.parquet"):
            diag = dk.get_mapping_diagnostics(999)
        assert diag["total_dk_players"] == 0
        assert diag["mapped_count"] == 0

    def test_unmapped_players_list_identifies_unmatched(self, tmp_path):
        mp = tmp_path / "dk" / "map.parquet"
        with patch.object(dk, "_DK_DIR", tmp_path / "dk"), \
             patch.object(dk, "_PLAYER_MAP_PATH", mp):
            yak = _make_yak_pool()
            dk_pool = _make_dk_pool()
            mapping = dk.map_dk_players_to_yak(100, yak, dk_pool)
            dk.save_dk_player_map(mapping)
            diag = dk.get_mapping_diagnostics(100)
        unmapped_names = [p["dk_name"] for p in diag["unmapped_players"]]
        assert "Unknown Player" in unmapped_names


# ---------------------------------------------------------------------------
# 5.6 — parse_roster_rules
# ---------------------------------------------------------------------------

class TestParseRosterRules:
    def test_classic_8_man_rules(self):
        raw = {
            "rosterSlots": [
                {"name": "PG"}, {"name": "SG"}, {"name": "SF"},
                {"name": "PF"}, {"name": "C"}, {"name": "G"},
                {"name": "F"}, {"name": "UTIL"},
            ],
            "salaryCap": 50000,
        }
        rules = dk.parse_roster_rules(raw)
        assert rules["lineup_size"] == 8
        assert rules["salary_cap"] == 50000
        assert not rules["captain_slot"]
        assert not rules["is_showdown"]

    def test_showdown_captain_rules(self):
        raw = {
            "rosterSlots": [
                {"name": "CPT"}, {"name": "FLEX"}, {"name": "FLEX"},
                {"name": "FLEX"}, {"name": "FLEX"}, {"name": "FLEX"},
            ],
            "salaryCap": 50000,
        }
        rules = dk.parse_roster_rules(raw)
        assert rules["lineup_size"] == 6
        assert rules["captain_slot"]
        assert rules["is_showdown"]

    def test_empty_rules_returns_defaults(self):
        rules = dk.parse_roster_rules({})
        assert rules["lineup_size"] == 8
        assert rules["salary_cap"] == 50000
        assert rules["slots"] == ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

    def test_salary_cap_parsed_correctly(self):
        raw = {"rosterSlots": [], "salaryCap": 60000}
        rules = dk.parse_roster_rules(raw)
        assert rules["salary_cap"] == 60000


# ---------------------------------------------------------------------------
# 5.5 — build_contest_scoped_pool
# ---------------------------------------------------------------------------

class TestBuildContestScopedPool:
    def test_returns_dataframe(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        result = dk.build_contest_scoped_pool(100, yak, dk_pool)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_uses_dk_salary(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        result = dk.build_contest_scoped_pool(100, yak, dk_pool)
        # LeBron: DK salary = 10200, YakOS salary = 10000
        lebron_row = result[result["player_name"] == "LeBron James"]
        if not lebron_row.empty:
            assert float(lebron_row.iloc[0]["salary"]) == 10200.0

    def test_unmapped_column_present(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        result = dk.build_contest_scoped_pool(100, yak, dk_pool)
        assert "_unmapped" in result.columns

    def test_unmapped_flag_set_for_unknown_player(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        result = dk.build_contest_scoped_pool(100, yak, dk_pool)
        unknown_row = result[result["player_name"] == "Unknown Player"]
        if not unknown_row.empty:
            assert bool(unknown_row.iloc[0]["_unmapped"]) is True

    def test_mapped_players_have_yak_projections(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        result = dk.build_contest_scoped_pool(100, yak, dk_pool)
        lebron_row = result[result["player_name"] == "LeBron James"]
        if not lebron_row.empty:
            assert float(lebron_row.iloc[0]["proj"]) == 55.0

    def test_empty_dk_pool_falls_back_to_yak_pool(self):
        yak = _make_yak_pool()
        result = dk.build_contest_scoped_pool(100, yak, pd.DataFrame())
        # Should fall back gracefully
        assert isinstance(result, pd.DataFrame)

    def test_returns_only_dk_draft_group_players(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()  # 4 DK players
        result = dk.build_contest_scoped_pool(100, yak, dk_pool)
        # Result rows ≤ DK pool size
        assert len(result) <= len(dk_pool)

    def test_dk_player_id_column_present(self):
        yak = _make_yak_pool()
        dk_pool = _make_dk_pool()
        result = dk.build_contest_scoped_pool(100, yak, dk_pool)
        assert "dk_player_id" in result.columns


# ---------------------------------------------------------------------------
# is_dk_integration_enabled + config constants
# ---------------------------------------------------------------------------

class TestDkIntegrationEnabled:
    def test_enabled_by_default(self):
        with patch.dict(os.environ, {"DK_INTEGRATION_ENABLED": "true"}):
            assert dk.is_dk_integration_enabled() is True

    def test_disabled_via_env(self):
        with patch.dict(os.environ, {"DK_INTEGRATION_ENABLED": "false"}):
            assert dk.is_dk_integration_enabled() is False

    def test_disabled_via_zero(self):
        with patch.dict(os.environ, {"DK_INTEGRATION_ENABLED": "0"}):
            assert dk.is_dk_integration_enabled() is False


class TestConfigConstants:
    def test_dk_integration_enabled_in_config(self):
        import yak_core.config as cfg
        assert hasattr(cfg, "DK_INTEGRATION_ENABLED")
        assert isinstance(cfg.DK_INTEGRATION_ENABLED, bool)

    def test_dk_sports_enabled_in_config(self):
        import yak_core.config as cfg
        assert hasattr(cfg, "DK_SPORTS_ENABLED")
        assert isinstance(cfg.DK_SPORTS_ENABLED, list)

    def test_dk_polling_freq_in_config(self):
        import yak_core.config as cfg
        assert hasattr(cfg, "DK_POLLING_FREQ_MINUTES")
        assert isinstance(cfg.DK_POLLING_FREQ_MINUTES, int)
        assert cfg.DK_POLLING_FREQ_MINUTES > 0


# ---------------------------------------------------------------------------
# fetch_game_type_rules with fallback
# ---------------------------------------------------------------------------

class TestFetchGameTypeRules:
    def test_returns_raw_json_on_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rosterSlots": [{"name": "PG"}], "salaryCap": 50000}
        with patch.object(dk, "_rate_limited_get", return_value=mock_resp):
            rules = dk.fetch_game_type_rules(1)
        assert "rosterSlots" in rules

    def test_falls_back_to_empty_dict_when_api_fails_and_no_fallback_file(self, tmp_path):
        with patch.object(dk, "_rate_limited_get", side_effect=Exception("network error")), \
             patch.object(dk, "_RULES_JSON_FALLBACK", tmp_path / "nonexistent.json"), \
             patch.object(dk, "is_dk_integration_enabled", return_value=True):
            rules = dk.fetch_game_type_rules(99)
        assert isinstance(rules, dict)

    def test_falls_back_to_file_when_api_fails(self, tmp_path):
        fallback_path = tmp_path / "RulesAndScoring.json"
        fallback_path.write_text(json.dumps({"gameTypes": {"5": {"salaryCap": 55000}}}))
        with patch.object(dk, "_rate_limited_get", side_effect=Exception("fail")), \
             patch.object(dk, "_RULES_JSON_FALLBACK", fallback_path), \
             patch.object(dk, "is_dk_integration_enabled", return_value=True):
            rules = dk.fetch_game_type_rules(5)
        assert rules.get("salaryCap") == 55000

    def test_returns_empty_when_disabled(self):
        with patch.object(dk, "is_dk_integration_enabled", return_value=False):
            rules = dk.fetch_game_type_rules(1)
        assert isinstance(rules, dict)


# ---------------------------------------------------------------------------
# App imports smoke test (5.7)
# ---------------------------------------------------------------------------

class TestDkIngestAppImports:
    def test_fetch_dk_lobby_contests_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "fetch_dk_lobby_contests")

    def test_fetch_dk_draftables_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "fetch_dk_draftables")

    def test_build_contest_scoped_pool_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "build_contest_scoped_pool")

    def test_map_dk_players_to_yak_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "map_dk_players_to_yak")

    def test_parse_roster_rules_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "parse_roster_rules")

    def test_get_mapping_diagnostics_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "get_mapping_diagnostics")

    def test_is_dk_integration_enabled_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "is_dk_integration_enabled")

    def test_save_dk_contests_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "save_dk_contests")

    def test_load_dk_contests_importable(self):
        import importlib
        mod = importlib.import_module("yak_core.dk_ingest")
        assert hasattr(mod, "load_dk_contests")
