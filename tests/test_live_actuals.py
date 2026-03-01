"""Tests for Tank01 actuals fetching: _calc_dk_nba_fp,
_fetch_actuals_from_box_scores, fetch_actuals_from_api, and fetch_live_dfs."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from yak_core.live import (
    _calc_dk_nba_fp,
    _fetch_actuals_from_box_scores,
    fetch_actuals_from_api,
    fetch_live_dfs,
)

# ---------------------------------------------------------------------------
# _calc_dk_nba_fp
# ---------------------------------------------------------------------------

class TestCalcDkNbaFp:
    def test_basic_scoring(self):
        stats = {"pts": 20, "reb": 5, "ast": 4, "stl": 1, "blk": 1, "TOV": 2, "fg3m": 2}
        fp = _calc_dk_nba_fp(stats)
        # 20 + 2*0.5 + 5*1.25 + 4*1.5 + 1*2 + 1*2 - 2*0.5 = 20+1+6.25+6+2+2-1 = 36.25
        assert fp == pytest.approx(36.25)

    def test_double_double_bonus(self):
        stats = {"pts": 10, "reb": 10, "ast": 2}
        fp = _calc_dk_nba_fp(stats)
        # 10 + 10*1.25 + 2*1.5 = 10+12.5+3 = 25.5 + 1.5 dd bonus = 27.0
        assert fp == pytest.approx(27.0)

    def test_triple_double_bonus(self):
        stats = {"pts": 10, "reb": 10, "ast": 10}
        fp = _calc_dk_nba_fp(stats)
        # 10 + 12.5 + 15 = 37.5 + 3 td bonus = 40.5
        assert fp == pytest.approx(40.5)

    def test_only_double_double_when_three_cats_not_met(self):
        # pts=10, reb=10, ast=9 → exactly 2 cats >= 10 → dd bonus (+1.5), not td
        stats = {"pts": 10, "reb": 10, "ast": 9}
        fp = _calc_dk_nba_fp(stats)
        # 10 + 12.5 + 13.5 = 36 + 1.5 dd = 37.5
        assert fp == pytest.approx(37.5)

    def test_zero_stats(self):
        assert _calc_dk_nba_fp({}) == 0.0

    def test_alternative_field_names(self):
        # Use "points", "rebounds", "assists", "steals", "blocks", "turnovers"
        stats = {"points": 15, "rebounds": 8, "assists": 3, "steals": 2, "blocks": 1, "turnovers": 1}
        fp = _calc_dk_nba_fp(stats)
        # 15 + 8*1.25 + 3*1.5 + 2*2 + 1*2 - 1*0.5 = 15+10+4.5+4+2-0.5 = 35.0
        assert fp == pytest.approx(35.0)

    def test_negative_fp_possible_with_turnovers(self):
        stats = {"TOV": 5}
        fp = _calc_dk_nba_fp(stats)
        assert fp == pytest.approx(-2.5)

    def test_three_pm_bonus(self):
        stats = {"pts": 6, "fg3m": 2}  # 2 threes = 6 pts + 0.5*2 bonus
        fp = _calc_dk_nba_fp(stats)
        assert fp == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# _fetch_actuals_from_box_scores
# ---------------------------------------------------------------------------

def _make_response(json_body, status=200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_body
    m.raise_for_status = MagicMock()
    return m


class TestFetchActualsFromBoxScores:
    CFG = {"RAPIDAPI_KEY": "test-key"}
    DATE = "20260227"

    def _games_response(self, games):
        return _make_response({"statusCode": 200, "body": {"games": games}})

    def _box_response(self, player_stats):
        return _make_response({"statusCode": 200, "body": {"playerStats": player_stats}})

    @patch("yak_core.live.requests.get")
    def test_fetches_precomputed_dk_fp(self, mock_get):
        games = [{"gameID": "G1"}]
        players = [
            {"displayName": "Alice", "fantasyPoints": {"DraftKings": 42.5}},
            {"displayName": "Bob", "fantasyPoints": {"DraftKings": 35.0}},
        ]
        mock_get.side_effect = [self._games_response(games), self._box_response(players)]

        df = _fetch_actuals_from_box_scores(self.DATE, self.CFG)
        assert list(df.columns) == ["player_name", "actual_fp"]
        assert df.loc[df["player_name"] == "Alice", "actual_fp"].iloc[0] == pytest.approx(42.5)

    @patch("yak_core.live.requests.get")
    def test_falls_back_to_calc_when_no_precomputed_fp(self, mock_get):
        games = [{"gameID": "G1"}]
        players = [{"displayName": "Charlie", "pts": 20, "reb": 5, "ast": 4}]
        mock_get.side_effect = [self._games_response(games), self._box_response(players)]

        df = _fetch_actuals_from_box_scores(self.DATE, self.CFG)
        assert df["player_name"].iloc[0] == "Charlie"
        assert df["actual_fp"].iloc[0] > 0

    @patch("yak_core.live.requests.get")
    def test_multiple_games(self, mock_get):
        games = [{"gameID": "G1"}, {"gameID": "G2"}]
        players_g1 = [{"displayName": "Alice", "fantasyPoints": {"DraftKings": 30.0}}]
        players_g2 = [{"displayName": "Bob", "fantasyPoints": {"DraftKings": 25.0}}]
        mock_get.side_effect = [
            self._games_response(games),
            self._box_response(players_g1),
            self._box_response(players_g2),
        ]

        df = _fetch_actuals_from_box_scores(self.DATE, self.CFG)
        assert len(df) == 2
        assert set(df["player_name"]) == {"Alice", "Bob"}

    @patch("yak_core.live.requests.get")
    def test_raises_if_no_games(self, mock_get):
        mock_get.return_value = _make_response({"statusCode": 200, "body": {"games": []}})
        with pytest.raises(ValueError, match="No games found"):
            _fetch_actuals_from_box_scores(self.DATE, self.CFG)

    @patch("yak_core.live.requests.get")
    def test_raises_if_no_players_in_box_scores(self, mock_get):
        games = [{"gameID": "G1"}]
        mock_get.side_effect = [
            self._games_response(games),
            self._box_response([]),
        ]
        with pytest.raises(ValueError, match="No player actuals parsed"):
            _fetch_actuals_from_box_scores(self.DATE, self.CFG)

    @patch("yak_core.live.requests.get")
    def test_handles_body_as_list(self, mock_get):
        """If games endpoint returns body as a plain list of games."""
        games = [{"gameID": "G1"}]
        players = [{"displayName": "Dave", "fantasyPoints": {"DraftKings": 18.0}}]
        mock_get.side_effect = [
            _make_response({"statusCode": 200, "body": games}),
            self._box_response(players),
        ]
        df = _fetch_actuals_from_box_scores(self.DATE, self.CFG)
        assert df["player_name"].iloc[0] == "Dave"

    @patch("yak_core.live.requests.get")
    def test_skips_entries_without_name(self, mock_get):
        games = [{"gameID": "G1"}]
        players = [
            {"displayName": "", "fantasyPoints": {"DraftKings": 10.0}},
            {"displayName": "Eve", "fantasyPoints": {"DraftKings": 22.0}},
        ]
        mock_get.side_effect = [self._games_response(games), self._box_response(players)]
        df = _fetch_actuals_from_box_scores(self.DATE, self.CFG)
        assert len(df) == 1
        assert df["player_name"].iloc[0] == "Eve"

    @patch("yak_core.live.requests.get")
    def test_date_normalisation_with_hyphens(self, mock_get):
        """YYYY-MM-DD date format should be accepted."""
        games = [{"gameID": "G1"}]
        players = [{"displayName": "Frank", "fantasyPoints": {"DraftKings": 28.0}}]
        mock_get.side_effect = [self._games_response(games), self._box_response(players)]
        df = _fetch_actuals_from_box_scores("2026-02-27", self.CFG)
        assert not df.empty

    @patch("yak_core.live.requests.get")
    def test_dkpoints_key_variant(self, mock_get):
        """fantasyPoints.dkPoints key variant should be parsed."""
        games = [{"gameID": "G1"}]
        players = [{"displayName": "Grace", "fantasyPoints": {"dkPoints": 33.0}}]
        mock_get.side_effect = [self._games_response(games), self._box_response(players)]
        df = _fetch_actuals_from_box_scores(self.DATE, self.CFG)
        assert df["actual_fp"].iloc[0] == pytest.approx(33.0)

    @patch("yak_core.live.requests.get")
    def test_flat_fantasy_points_value(self, mock_get):
        """fantasyPoints as a plain float (not nested dict) is accepted."""
        games = [{"gameID": "G1"}]
        players = [{"displayName": "Hank", "fantasyPoints": 45.0}]
        mock_get.side_effect = [self._games_response(games), self._box_response(players)]
        df = _fetch_actuals_from_box_scores(self.DATE, self.CFG)
        assert df["actual_fp"].iloc[0] == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# fetch_actuals_from_api (integration-level with mocks)
# ---------------------------------------------------------------------------

class TestFetchActualsFromApi:
    CFG = {"RAPIDAPI_KEY": "test-key"}
    DATE = "20260227"

    @patch("yak_core.live._fetch_actuals_from_box_scores")
    @patch("yak_core.live.fetch_live_dfs")
    def test_uses_dfs_endpoint_when_it_works(self, mock_dfs, mock_box):
        """When getNBADFS returns data, box scores should NOT be called."""
        mock_dfs.return_value = pd.DataFrame([
            {"player_name": "Alice", "proj": 42.5},
        ])
        df = fetch_actuals_from_api(self.DATE, self.CFG)
        mock_box.assert_not_called()
        assert df["player_name"].iloc[0] == "Alice"
        assert df["actual_fp"].iloc[0] == pytest.approx(42.5)

    @patch("yak_core.live._fetch_actuals_from_box_scores")
    @patch("yak_core.live.fetch_live_dfs")
    def test_falls_back_to_box_scores_when_dfs_raises(self, mock_dfs, mock_box):
        """When getNBADFS raises, the box-score fallback should be used."""
        mock_dfs.side_effect = ValueError("No DFS player rows parsed for 20260227")
        mock_box.return_value = pd.DataFrame([
            {"player_name": "Bob", "actual_fp": 35.0},
        ])
        df = fetch_actuals_from_api(self.DATE, self.CFG)
        mock_box.assert_called_once()
        assert df["player_name"].iloc[0] == "Bob"

    @patch("yak_core.live._fetch_actuals_from_box_scores")
    @patch("yak_core.live.fetch_live_dfs")
    def test_falls_back_when_dfs_returns_empty(self, mock_dfs, mock_box):
        """When getNBADFS returns an empty DataFrame, use box scores."""
        mock_dfs.return_value = pd.DataFrame()
        mock_box.return_value = pd.DataFrame([
            {"player_name": "Carol", "actual_fp": 27.0},
        ])
        df = fetch_actuals_from_api(self.DATE, self.CFG)
        mock_box.assert_called_once()
        assert len(df) == 1

    @patch("yak_core.live._fetch_actuals_from_box_scores")
    @patch("yak_core.live.fetch_live_dfs")
    def test_raises_runtime_error_when_both_fail(self, mock_dfs, mock_box):
        """Both endpoints failing should raise RuntimeError."""
        mock_dfs.side_effect = ValueError("No DFS player rows parsed for 20260227")
        mock_box.side_effect = ValueError("No games found for 20260227")
        with pytest.raises(RuntimeError, match="Tank01 actuals API error"):
            fetch_actuals_from_api(self.DATE, self.CFG)

    @patch("yak_core.live._fetch_actuals_from_box_scores")
    @patch("yak_core.live.fetch_live_dfs")
    def test_date_with_hyphens_accepted(self, mock_dfs, mock_box):
        """YYYY-MM-DD date is normalised correctly."""
        mock_dfs.side_effect = ValueError("No DFS data")
        mock_box.return_value = pd.DataFrame([{"player_name": "Dave", "actual_fp": 20.0}])
        df = fetch_actuals_from_api("2026-02-27", self.CFG)
        # box scores called with clean YYYYMMDD key
        mock_box.assert_called_once_with("20260227", self.CFG)
        assert not df.empty


# ---------------------------------------------------------------------------
# fetch_live_dfs — body dict unwrapping
# ---------------------------------------------------------------------------

def _make_dfs_response(json_body, status=200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_body
    m.raise_for_status = MagicMock()
    return m


_SAMPLE_PLAYERS = [
    {"playerID": "1", "longName": "Alice", "teamAbv": "BOS", "pos": "SF",
     "salary": "7500", "fantasyPoints": "35.0"},
    {"playerID": "2", "longName": "Bob", "teamAbv": "LAL", "pos": "PG",
     "salary": "8000", "fantasyPoints": "40.0"},
]


class TestFetchLiveDfsBodyParsing:
    """Tests for fetch_live_dfs body dict unwrapping (Tank01 API fix)."""

    CFG = {"RAPIDAPI_KEY": "test-key"}
    DATE = "20260227"

    @patch("yak_core.live.requests.get")
    def test_body_as_list(self, mock_get):
        """Plain list body is handled correctly."""
        mock_get.return_value = _make_dfs_response(
            {"statusCode": 200, "body": _SAMPLE_PLAYERS}
        )
        df = fetch_live_dfs(self.DATE, self.CFG)
        assert len(df) == 2
        assert set(df["player_name"]) == {"Alice", "Bob"}

    @patch("yak_core.live.requests.get")
    def test_body_as_draftkings_dict(self, mock_get):
        """body = {'DraftKings': [...]} is unwrapped correctly."""
        mock_get.return_value = _make_dfs_response(
            {"statusCode": 200, "body": {"DraftKings": _SAMPLE_PLAYERS}}
        )
        df = fetch_live_dfs(self.DATE, self.CFG)
        assert len(df) == 2
        assert "Alice" in df["player_name"].values

    @patch("yak_core.live.requests.get")
    def test_body_as_dk_dict(self, mock_get):
        """body = {'DK': [...]} variant is unwrapped correctly."""
        mock_get.return_value = _make_dfs_response(
            {"statusCode": 200, "body": {"DK": _SAMPLE_PLAYERS}}
        )
        df = fetch_live_dfs(self.DATE, self.CFG)
        assert len(df) == 2

    @patch("yak_core.live.requests.get")
    def test_body_dict_fallback_to_longest_list(self, mock_get):
        """Dict body with unknown key falls back to longest list value."""
        mock_get.return_value = _make_dfs_response(
            {"statusCode": 200, "body": {"unknownSite": _SAMPLE_PLAYERS, "meta": []}}
        )
        df = fetch_live_dfs(self.DATE, self.CFG)
        assert len(df) == 2

    @patch("yak_core.live.requests.get")
    def test_salary_as_string_is_parsed(self, mock_get):
        """Salary strings (e.g. '7500') are converted to int."""
        mock_get.return_value = _make_dfs_response(
            {"statusCode": 200, "body": _SAMPLE_PLAYERS}
        )
        df = fetch_live_dfs(self.DATE, self.CFG)
        assert df["salary"].dtype in (int, "int64", "int32")
        assert df["salary"].iloc[0] == 7500
