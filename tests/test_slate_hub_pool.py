"""Tests for player pool deduplication and numeric rounding in slate_hub.

Validates the two pool-cleanup steps added to pages/1_slate_hub.py:
  1. Group-then-deduplicate: aggregate numeric cols per (player_name, team, pos, salary),
     with a single-key drop_duplicates fallback when group/agg cols are absent.
  2. Float columns rounded to 1 decimal place before display.
"""

from __future__ import annotations

import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers that mirror the logic in 1_slate_hub.py so we can unit-test it.
# ---------------------------------------------------------------------------

def _deduplicate_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Group-then-dedup: aggregate duplicate rows per player across sources.

    Groups by (player_name, team, pos, salary) and averages numeric projection
    columns.  Falls back to single-key drop_duplicates when group or agg cols
    are not present.
    """
    _player_identity_cols = ["player_name", "team", "pos", "salary"]
    _numeric_agg_cols = ["proj", "floor", "ceil", "proj_minutes", "ownership"]
    _group_cols = [c for c in _player_identity_cols if c in pool.columns]
    _agg_cols = {
        c: "mean"
        for c in _numeric_agg_cols
        if c in pool.columns
    }
    if _group_cols and _agg_cols:
        _extra_cols = [c for c in pool.columns if c not in _group_cols and c not in _agg_cols]
        _extra_agg = {c: "first" for c in _extra_cols}
        pool = pool.groupby(_group_cols, as_index=False).agg({**_agg_cols, **_extra_agg})
    else:
        dedup_key = "dk_player_id" if "dk_player_id" in pool.columns else "player_name"
        if "proj" in pool.columns:
            pool = pool.sort_values("proj", ascending=False)
        pool = pool.drop_duplicates(subset=[dedup_key], keep="first")
    return pool.reset_index(drop=True)


_FLOAT_PREVIEW_COLS = ["proj", "floor", "ceil", "proj_minutes", "ownership", "actual_fp"]


def _round_preview(df: pd.DataFrame) -> pd.DataFrame:
    """Round float preview columns to 1 decimal place."""
    df = df.copy()
    float_cols = [c for c in _FLOAT_PREVIEW_COLS if c in df.columns]
    df[float_cols] = df[float_cols].round(1)
    return df


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplicatePool:
    def test_no_duplicates_unchanged(self):
        pool = pd.DataFrame({
            "player_name": ["A", "B", "C"],
            "team": ["T1", "T2", "T3"],
            "pos": ["PG", "SG", "SF"],
            "salary": [8000, 7500, 7000],
            "proj": [20.0, 18.0, 15.0],
        })
        result = _deduplicate_pool(pool)
        assert len(result) == 3

    def test_groupby_averages_proj_for_duplicates(self):
        """Duplicate rows for the same (name/team/pos/salary) should be averaged."""
        pool = pd.DataFrame({
            "player_name": ["A", "A", "B"],
            "team": ["T1", "T1", "T2"],
            "pos": ["PG", "PG", "SG"],
            "salary": [8000, 8000, 7500],
            "proj": [18.0, 22.0, 15.0],
        })
        result = _deduplicate_pool(pool)
        assert len(result) == 2
        player_a = result[result["player_name"] == "A"]
        assert player_a["proj"].iloc[0] == 20.0  # mean of 18.0 and 22.0

    def test_groupby_averages_floor_ceil_ownership(self):
        """All aggregated numeric cols are averaged per group."""
        pool = pd.DataFrame({
            "player_name": ["X", "X"],
            "team": ["T1", "T1"],
            "pos": ["C", "C"],
            "salary": [9000, 9000],
            "proj": [30.0, 20.0],
            "floor": [20.0, 10.0],
            "ceil": [40.0, 30.0],
            "ownership": [0.2, 0.4],
        })
        result = _deduplicate_pool(pool)
        assert len(result) == 1
        assert result["proj"].iloc[0] == 25.0
        assert result["floor"].iloc[0] == 15.0
        assert result["ceil"].iloc[0] == 35.0
        assert result["ownership"].iloc[0] == pytest.approx(0.3)

    def test_different_salary_keeps_separate_rows(self):
        """Same player name but different salary slot → separate rows."""
        pool = pd.DataFrame({
            "player_name": ["A", "A"],
            "team": ["T1", "T1"],
            "pos": ["CPT", "FLEX"],
            "salary": [12000, 8000],
            "proj": [30.0, 20.0],
        })
        result = _deduplicate_pool(pool)
        assert len(result) == 2

    def test_groupby_used_when_player_name_present(self):
        """player_name is a group col, so duplicates are averaged (not kept by max)."""
        pool = pd.DataFrame({
            "player_name": ["A", "A", "B"],
            "proj": [10.0, 14.0, 8.0],
        })
        result = _deduplicate_pool(pool)
        assert len(result) == 2
        a_row = result[result["player_name"] == "A"]
        assert a_row["proj"].iloc[0] == 12.0  # mean of 10.0 and 14.0

    def test_fallback_when_no_group_or_agg_cols(self):
        """When neither group cols nor agg cols are present, no error is raised."""
        pool = pd.DataFrame({
            "dk_player_id": ["1", "1", "2"],
            "some_col": ["x", "x", "y"],
        })
        result = _deduplicate_pool(pool)
        # No proj col → agg_cols empty → fallback path, single-key drop_duplicates
        assert len(result) == 2

    def test_dedup_result_is_reset_index(self):
        pool = pd.DataFrame({
            "player_name": ["A", "A"],
            "team": ["T1", "T1"],
            "pos": ["PG", "PG"],
            "salary": [8000, 8000],
            "proj": [5.0, 9.0],
        })
        result = _deduplicate_pool(pool)
        assert list(result.index) == [0]

    def test_empty_pool_returns_empty(self):
        pool = pd.DataFrame(columns=["player_name", "team", "pos", "salary", "proj"])
        result = _deduplicate_pool(pool)
        assert result.empty

    def test_extra_columns_preserved(self):
        """Non-group, non-agg columns are kept (first value per group)."""
        pool = pd.DataFrame({
            "player_name": ["A", "A"],
            "team": ["T1", "T1"],
            "pos": ["PG", "PG"],
            "salary": [8000, 8000],
            "proj": [18.0, 22.0],
            "dk_player_id": ["99", "99"],
        })
        result = _deduplicate_pool(pool)
        assert "dk_player_id" in result.columns
        assert result["dk_player_id"].iloc[0] == "99"


# ---------------------------------------------------------------------------
# Rounding tests
# ---------------------------------------------------------------------------

class TestRoundPreview:
    def test_rounds_to_one_decimal(self):
        df = pd.DataFrame({
            "player_name": ["A"],
            "proj": [24.5678],
            "floor": [18.3333],
            "ceil": [31.999],
            "proj_minutes": [34.12345],
            "ownership": [0.2567],
        })
        result = _round_preview(df)
        assert result["proj"].iloc[0] == 24.6
        assert result["floor"].iloc[0] == 18.3
        assert result["ceil"].iloc[0] == 32.0
        assert result["proj_minutes"].iloc[0] == 34.1
        assert result["ownership"].iloc[0] == 0.3

    def test_skips_missing_float_cols(self):
        """Only present float columns are rounded; no KeyError for absent ones."""
        df = pd.DataFrame({
            "player_name": ["A"],
            "proj": [20.555],
        })
        result = _round_preview(df)
        assert result["proj"].iloc[0] == 20.6
        # actual_fp, floor, ceil, etc. were not in df — no error

    def test_non_float_cols_unchanged(self):
        df = pd.DataFrame({
            "player_name": ["A"],
            "salary": [7500],
            "proj": [22.345],
        })
        result = _round_preview(df)
        assert result["salary"].iloc[0] == 7500
        assert result["proj"].iloc[0] == 22.3

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"proj": [19.999]})
        _ = _round_preview(df)
        assert df["proj"].iloc[0] == 19.999  # original unchanged
