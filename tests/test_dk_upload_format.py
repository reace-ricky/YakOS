"""Tests for to_dk_upload_format — DraftKings bulk upload CSV export."""

import pandas as pd
import pytest
from yak_core.lineups import to_dk_upload_format
from yak_core.config import DK_POS_SLOTS

_META_COLS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
_EXPECTED_COLS = _META_COLS + list(DK_POS_SLOTS)


def _make_lineups(n_lineups: int = 3) -> pd.DataFrame:
    """Build a minimal long-format lineups DataFrame."""
    rows = []
    for lu in range(n_lineups):
        for i, slot in enumerate(DK_POS_SLOTS):
            rows.append({
                "lineup_index": lu,
                "slot": slot,
                "player_name": f"Player_{lu}_{i}",
                "team": f"T{lu % 3 + 1}",
                "salary": 5000 + i * 200,
                "proj": 20.0 + i,
            })
    return pd.DataFrame(rows)


class TestDkUploadFormat:
    def test_empty_input_returns_correct_columns(self):
        df = to_dk_upload_format(pd.DataFrame())
        assert list(df.columns) == _EXPECTED_COLS
        assert len(df) == 0

    def test_one_row_per_lineup(self):
        lineups = _make_lineups(n_lineups=5)
        result = to_dk_upload_format(lineups)
        assert len(result) == 5

    def test_three_lineups(self):
        lineups = _make_lineups(n_lineups=3)
        result = to_dk_upload_format(lineups)
        assert len(result) == 3

    def test_correct_columns(self):
        lineups = _make_lineups()
        result = to_dk_upload_format(lineups)
        assert list(result.columns) == _EXPECTED_COLS

    def test_meta_columns_are_blank(self):
        lineups = _make_lineups()
        result = to_dk_upload_format(lineups)
        for col in _META_COLS:
            assert (result[col] == "").all(), f"Column '{col}' should be blank"

    def test_player_name_in_slot_columns(self):
        lineups = _make_lineups(n_lineups=2)
        result = to_dk_upload_format(lineups)
        # Each slot column should be non-empty
        for slot in DK_POS_SLOTS:
            assert result[slot].str.len().gt(0).all(), f"Slot '{slot}' has empty cells"

    def test_player_format_includes_team(self):
        lineups = _make_lineups(n_lineups=1)
        result = to_dk_upload_format(lineups)
        # Check "Name (TEAM)" format
        for slot in DK_POS_SLOTS:
            cell = result[slot].iloc[0]
            assert "(" in cell and ")" in cell, (
                f"Slot '{slot}' cell '{cell}' should be 'Name (TEAM)' format"
            )

    def test_missing_lineup_index_column(self):
        """DataFrame without lineup_index returns empty result with correct cols."""
        df = pd.DataFrame({"slot": ["PG"], "player_name": ["X"], "team": ["Y"]})
        result = to_dk_upload_format(df)
        assert len(result) == 0
        assert list(result.columns) == _EXPECTED_COLS

    def test_slot_order_matches_dk_pos_slots(self):
        lineups = _make_lineups(n_lineups=1)
        result = to_dk_upload_format(lineups)
        slot_cols = [c for c in result.columns if c in DK_POS_SLOTS]
        assert slot_cols == list(DK_POS_SLOTS)

    def test_team_nan_omitted_from_cell(self):
        """If team is NaN, player cell should just be the name with no '(nan)'."""
        row = {
            "lineup_index": 0,
            "slot": "PG",
            "player_name": "Test Player",
            "team": float("nan"),
            "salary": 5000,
            "proj": 20.0,
        }
        # Fill remaining slots so lineup is valid
        rows = [row]
        for i, slot in enumerate(DK_POS_SLOTS[1:], start=1):
            rows.append({
                "lineup_index": 0,
                "slot": slot,
                "player_name": f"P_{i}",
                "team": "T1",
                "salary": 5000,
                "proj": 20.0,
            })
        lineups = pd.DataFrame(rows)
        result = to_dk_upload_format(lineups)
        pg_cell = result["PG"].iloc[0]
        assert "nan" not in pg_cell.lower(), f"'nan' should not appear in cell: {pg_cell}"
