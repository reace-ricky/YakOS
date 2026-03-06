"""Tests for yak_core/dff_ingest.py — DailyFantasyFuel fallback pool ingest."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import yak_core.dff_ingest as dff


# ---------------------------------------------------------------------------
# Sample DFF HTML fixtures
# ---------------------------------------------------------------------------

_DFF_HTML_PLAYER_ROWS = """
<html><body>
<table>
<tr data-name="LeBron James" data-pos="SF" data-team="LAL" data-opp="BOS"
    data-salary="9800" data-proj="48.5" data-own="24.3" data-status="Active"
    data-ou="225.5" data-spread="-3.5"></tr>
<tr data-name="Stephen Curry" data-pos="PG" data-team="GSW" data-opp="LAC"
    data-salary="9400" data-proj="45.2" data-own="22.1" data-status="Active"
    data-ou="218.0" data-spread="-2.0"></tr>
<tr data-name="Nikola Jokic" data-pos="C" data-team="DEN" data-opp="MIL"
    data-salary="10200" data-proj="53.1" data-own="19.8" data-status="Active"
    data-ou="221.5" data-spread="-4.5"></tr>
<tr data-name="Kawhi Leonard" data-pos="SF" data-team="LAC" data-opp="GSW"
    data-salary="8800" data-proj="0.0" data-own="2.1" data-status="OUT"
    data-ou="218.0" data-spread="2.0"></tr>
</table>
</body></html>
"""

_DFF_HTML_ALTERNATIVE_ATTRS = """
<html><body>
<div class="player-row" data-player-name="Anthony Davis" data-position="PF"
     data-teamabbrev="LAL" data-opp="BOS" data-dksalary="9200"
     data-fpts="44.0" data-ownership="18.5" data-injury-status="Active"
     data-over-under="225.5" data-spread="-3.5"></div>
</body></html>
"""

_DFF_HTML_NO_PLAYERS = """
<html><body>
<p>No projections available yet for today's slate.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# _DFFHTMLParser tests
# ---------------------------------------------------------------------------

class TestDFFHTMLParser:
    def test_parses_player_rows_with_data_name(self):
        parser = dff._DFFHTMLParser()
        parser.feed(_DFF_HTML_PLAYER_ROWS)
        assert len(parser.rows) == 4

    def test_player_row_contains_expected_keys(self):
        parser = dff._DFFHTMLParser()
        parser.feed(_DFF_HTML_PLAYER_ROWS)
        first = parser.rows[0]
        assert first["name"] == "LeBron James"
        assert first["pos"] == "SF"
        assert first["team"] == "LAL"
        assert first["salary"] == "9800"
        assert first["proj"] == "48.5"

    def test_no_player_rows_in_empty_html(self):
        parser = dff._DFFHTMLParser()
        parser.feed(_DFF_HTML_NO_PLAYERS)
        assert len(parser.rows) == 0

    def test_ignores_non_row_tags(self):
        """Non-row tags (h1, p, a, etc.) should not generate player rows."""
        html = """<html><body>
        <h1 data-name="LeBron James" data-pos="SF" data-team="LAL" data-salary="9800">Title</h1>
        <p data-name="Stephen Curry">paragraph</p>
        <tr data-name="Nikola Jokic" data-pos="C" data-team="DEN" data-salary="10200">row</tr>
        </body></html>"""
        parser = dff._DFFHTMLParser()
        parser.feed(html)
        # Only the tr row should be captured
        assert len(parser.rows) == 1
        assert parser.rows[0]["name"] == "Nikola Jokic"

    def test_ignores_non_person_name_values(self):
        """Attribute values that don't look like a person's name are skipped."""
        html = """<table>
        <tr data-name="12345" data-salary="9000"></tr>
        <tr data-name="A" data-salary="9000"></tr>
        <tr data-name="LeBron James" data-salary="9800"></tr>
        </table>"""
        parser = dff._DFFHTMLParser()
        parser.feed(html)
        assert len(parser.rows) == 1
        assert parser.rows[0]["name"] == "LeBron James"


# ---------------------------------------------------------------------------
# _extract_field tests
# ---------------------------------------------------------------------------

class TestExtractField:
    def test_returns_first_matching_alias(self):
        row = {"salary": "9000", "dksalary": "9200"}
        result = dff._extract_field(row, ["salary", "dksalary"])
        assert result == "9000"

    def test_falls_through_to_second_alias(self):
        row = {"dksalary": "9200"}
        result = dff._extract_field(row, ["salary", "dksalary"])
        assert result == "9200"

    def test_returns_none_when_no_match(self):
        row = {"other_key": "value"}
        result = dff._extract_field(row, ["salary", "dksalary"])
        assert result is None

    def test_skips_empty_string_values(self):
        row = {"salary": "", "dksalary": "9200"}
        result = dff._extract_field(row, ["salary", "dksalary"])
        # Empty string → falsy → skip to next alias
        assert result == "9200"


# ---------------------------------------------------------------------------
# _parse_rows tests
# ---------------------------------------------------------------------------

class TestParseRows:
    def _sample_rows(self):
        return [
            {
                "name": "LeBron James", "pos": "sf", "team": "lal", "opp": "bos",
                "salary": "9800", "proj": "48.5", "own": "24.3",
                "status": "Active", "ou": "225.5", "spread": "-3.5",
            },
            {
                "name": "Nikola Jokic", "pos": "C", "team": "DEN", "opp": "MIL",
                "salary": "10200", "proj": "53.1", "own": "19.8",
                "status": "Active", "ou": "221.5", "spread": "-4.5",
            },
        ]

    def test_returns_dataframe_with_expected_columns(self):
        df = dff._parse_rows(self._sample_rows())
        for col in ("player_name", "pos", "team", "opp", "salary", "proj",
                    "ownership", "status", "vegas_total", "vegas_spread"):
            assert col in df.columns, f"Missing column: {col}"

    def test_salary_is_int(self):
        df = dff._parse_rows(self._sample_rows())
        assert df["salary"].dtype in (int, "int32", "int64")
        assert df.loc[0, "salary"] == 9800

    def test_proj_is_float(self):
        df = dff._parse_rows(self._sample_rows())
        assert df["proj"].dtype in (float, "float32", "float64")
        assert df.loc[0, "proj"] == pytest.approx(48.5)

    def test_pos_uppercased(self):
        df = dff._parse_rows(self._sample_rows())
        # "sf" should be normalised to "SF"
        assert df.loc[0, "pos"] == "SF"

    def test_team_uppercased(self):
        df = dff._parse_rows(self._sample_rows())
        assert df.loc[0, "team"] == "LAL"

    def test_rows_without_salary_dropped(self):
        rows = [
            {"name": "LeBron James", "pos": "SF", "team": "LAL", "salary": "0"},
            {"name": "Nikola Jokic", "pos": "C", "team": "DEN", "salary": "10200"},
        ]
        df = dff._parse_rows(rows)
        assert len(df) == 1
        assert df.loc[0, "player_name"] == "Nikola Jokic"

    def test_rows_without_name_dropped(self):
        rows = [
            {"pos": "SF", "team": "LAL", "salary": "9800"},
            {"name": "Nikola Jokic", "pos": "C", "team": "DEN", "salary": "10200"},
        ]
        df = dff._parse_rows(rows)
        assert len(df) == 1
        assert df.loc[0, "player_name"] == "Nikola Jokic"

    def test_empty_rows_returns_empty_dataframe(self):
        df = dff._parse_rows([])
        assert df.empty
        assert "player_name" in df.columns

    def test_vegas_total_is_numeric(self):
        df = dff._parse_rows(self._sample_rows())
        assert df.loc[0, "vegas_total"] == pytest.approx(225.5)


# ---------------------------------------------------------------------------
# fetch_dff_pool tests (with mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchDffPool:
    def test_returns_dataframe_on_valid_html(self):
        mock_resp = MagicMock()
        mock_resp.text = _DFF_HTML_PLAYER_ROWS
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp):
            df = dff.fetch_dff_pool(sport="NBA")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "player_name" in df.columns
        assert "salary" in df.columns
        assert "proj" in df.columns

    def test_returns_correct_player_count(self):
        mock_resp = MagicMock()
        mock_resp.text = _DFF_HTML_PLAYER_ROWS
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp):
            df = dff.fetch_dff_pool(sport="NBA")

        # 4 rows in HTML but Kawhi has salary 8800 (> 0), so all 4 should be included
        assert len(df) == 4

    def test_returns_empty_dataframe_on_http_error(self):
        import requests as req_lib
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError("403 Forbidden")

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp):
            df = dff.fetch_dff_pool(sport="NBA")

        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "player_name" in df.columns

    def test_returns_empty_dataframe_on_no_players(self):
        mock_resp = MagicMock()
        mock_resp.text = _DFF_HTML_NO_PLAYERS
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp):
            df = dff.fetch_dff_pool(sport="NBA")

        assert df.empty

    def test_returns_empty_dataframe_for_unsupported_sport(self):
        df = dff.fetch_dff_pool(sport="CRICKET")
        assert df.empty

    def test_alternative_attribute_names_parsed(self):
        """DFF sometimes uses different data-* attribute names; parser handles both."""
        mock_resp = MagicMock()
        mock_resp.text = _DFF_HTML_ALTERNATIVE_ATTRS
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp):
            df = dff.fetch_dff_pool(sport="NBA")

        assert len(df) == 1
        assert df.loc[0, "player_name"] == "Anthony Davis"
        assert df.loc[0, "pos"] == "PF"
        assert df.loc[0, "salary"] == 9200

    def test_status_preserved(self):
        mock_resp = MagicMock()
        mock_resp.text = _DFF_HTML_PLAYER_ROWS
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp):
            df = dff.fetch_dff_pool(sport="NBA")

        kawhi = df[df["player_name"] == "Kawhi Leonard"]
        assert not kawhi.empty
        assert kawhi.iloc[0]["status"] == "OUT"

    def test_uses_correct_nba_url(self):
        mock_resp = MagicMock()
        mock_resp.text = _DFF_HTML_PLAYER_ROWS
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp) as mock_get:
            dff.fetch_dff_pool(sport="NBA")
            call_url = mock_get.call_args[0][0]
            assert "dailyfantasyfuel.com" in call_url
            assert "nba" in call_url.lower()

    def test_date_str_param_does_not_error(self):
        """date_str param is accepted but ignored (DFF always serves today's pool)."""
        mock_resp = MagicMock()
        mock_resp.text = _DFF_HTML_PLAYER_ROWS
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp):
            df = dff.fetch_dff_pool(sport="NBA", date_str="2026-03-06")

        assert len(df) == 4

    def test_empty_dataframe_has_expected_columns(self):
        df = dff._empty_dff_pool()
        for col in ("player_name", "pos", "team", "opp", "salary", "proj",
                    "ownership", "status", "vegas_total", "vegas_spread"):
            assert col in df.columns

    def test_pool_compatible_with_yakos_pipeline(self):
        """Check that the returned DataFrame has all columns expected by YakOS pipeline."""
        mock_resp = MagicMock()
        mock_resp.text = _DFF_HTML_PLAYER_ROWS
        mock_resp.raise_for_status.return_value = None

        with patch("yak_core.dff_ingest.requests.get", return_value=mock_resp):
            df = dff.fetch_dff_pool(sport="NBA")

        # YakOS pipeline expects at minimum: player_name, pos, team, salary
        for col in ("player_name", "pos", "team", "salary"):
            assert col in df.columns
            assert df[col].notna().all(), f"Column {col} has NaN values"
