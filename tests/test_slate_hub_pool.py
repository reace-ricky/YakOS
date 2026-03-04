"""Tests for player pool deduplication, numeric rounding, and ineligible-player
filtering in slate_hub.

Validates the cleanup steps added to pages/1_slate_hub.py:
  1. Group-then-deduplicate: aggregate numeric cols per (player_name, team, pos, salary),
     with a single-key drop_duplicates fallback when group/agg cols are absent.
  2. Float columns rounded to 1 decimal place before display.
  3. _filter_ineligible_players: removes OUT/DND/IR players and zero-minutes players.
  4. _enrich_pool proj_minutes zeroing: players with ineligible status get proj_minutes=0.
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


# ---------------------------------------------------------------------------
# _filter_ineligible_players helper (mirrors logic in pages/1_slate_hub.py)
# ---------------------------------------------------------------------------

_INELIGIBLE_STATUSES = {
    "OUT", "IR", "INJ", "SUSPENDED", "SUSP",
    "G-LEAGUE", "G_LEAGUE", "GLEAGUE",
    "DND", "NA", "O",
}


def _filter_ineligible_players(pool: pd.DataFrame) -> pd.DataFrame:
    """Mirror of the function added to pages/1_slate_hub.py for unit testing."""
    df = pool.copy()
    if "status" in df.columns:
        inelig_mask = (
            df["status"].fillna("").astype(str).str.strip().str.upper()
            .isin(_INELIGIBLE_STATUSES)
        )
        df = df[~inelig_mask]
    mins_col = "proj_minutes" if "proj_minutes" in df.columns else (
        "minutes" if "minutes" in df.columns else None
    )
    if mins_col is not None:
        mins = pd.to_numeric(df[mins_col], errors="coerce").fillna(0)
        df = df[mins > 0]
    return df.reset_index(drop=True)


class TestFilterIneligiblePlayers:
    """Unit tests for _filter_ineligible_players."""

    def _make_pool(self):
        return pd.DataFrame({
            "player_name": ["SGA", "LeBron", "KD", "Kawhi", "Harden"],
            "team": ["OKC", "LAL", "PHX", "LAC", "LAC"],
            "status": ["OUT", "Active", "Active", "DND", "Active"],
            "proj_minutes": [0.0, 35.0, 32.0, 0.0, 28.0],
            "salary": [10200, 9800, 9400, 8800, 8400],
            "proj": [0.0, 48.0, 43.0, 0.0, 38.0],
        })

    def test_out_player_removed(self):
        result = _filter_ineligible_players(self._make_pool())
        assert "SGA" not in result["player_name"].values

    def test_dnd_player_removed(self):
        result = _filter_ineligible_players(self._make_pool())
        assert "Kawhi" not in result["player_name"].values

    def test_active_players_kept(self):
        result = _filter_ineligible_players(self._make_pool())
        assert set(result["player_name"].values) == {"LeBron", "KD", "Harden"}

    def test_zero_proj_minutes_removed(self):
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "status": ["Active", "Active"],
            "proj_minutes": [0.0, 30.0],
        })
        result = _filter_ineligible_players(pool)
        assert len(result) == 1
        assert result["player_name"].iloc[0] == "B"

    def test_uses_minutes_col_when_no_proj_minutes(self):
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "status": ["Active", "Active"],
            "minutes": [0.0, 30.0],
        })
        result = _filter_ineligible_players(pool)
        assert len(result) == 1
        assert result["player_name"].iloc[0] == "B"

    def test_no_status_col_only_minutes_filter_applied(self):
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "proj_minutes": [5.0, 0.0],
        })
        result = _filter_ineligible_players(pool)
        assert len(result) == 1
        assert result["player_name"].iloc[0] == "A"

    def test_no_minutes_col_only_status_filter_applied(self):
        pool = pd.DataFrame({
            "player_name": ["A", "B"],
            "status": ["OUT", "Active"],
        })
        result = _filter_ineligible_players(pool)
        assert len(result) == 1
        assert result["player_name"].iloc[0] == "B"

    def test_empty_pool_returns_empty(self):
        pool = pd.DataFrame(columns=["player_name", "status", "proj_minutes"])
        result = _filter_ineligible_players(pool)
        assert result.empty

    def test_does_not_mutate_original(self):
        pool = self._make_pool()
        original_len = len(pool)
        _filter_ineligible_players(pool)
        assert len(pool) == original_len

    def test_ir_player_removed(self):
        pool = pd.DataFrame({
            "player_name": ["X"],
            "status": ["IR"],
            "proj_minutes": [30.0],
        })
        result = _filter_ineligible_players(pool)
        assert result.empty

    def test_case_insensitive_status(self):
        pool = pd.DataFrame({
            "player_name": ["X"],
            "status": ["out"],
            "proj_minutes": [30.0],
        })
        result = _filter_ineligible_players(pool)
        assert result.empty

    def test_result_index_reset(self):
        pool = self._make_pool()
        result = _filter_ineligible_players(pool)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# Ineligible-status proj_minutes zeroing (mirrors _enrich_pool logic)
# ---------------------------------------------------------------------------

class TestEnrichPoolProjMinutesZeroing:
    """Verify that OUT/DND/IR players get proj_minutes=0 regardless of salary."""

    def _zero_inelig_minutes(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Minimal replica of the ineligible-zeroing step in _enrich_pool."""
        df = pool.copy()
        if "status" in df.columns:
            inelig_mask = (
                df["status"].fillna("").astype(str).str.strip().str.upper()
                .isin(_INELIGIBLE_STATUSES)
            )
            df.loc[inelig_mask, "proj_minutes"] = 0.0
        return df

    def test_out_player_gets_zero_proj_minutes(self):
        pool = pd.DataFrame({
            "player_name": ["SGA"],
            "status": ["OUT"],
            "salary": [10200],
            "proj_minutes": [34.0],  # salary-based value that would be non-zero
        })
        result = self._zero_inelig_minutes(pool)
        assert result.loc[0, "proj_minutes"] == 0.0

    def test_dnd_player_gets_zero_proj_minutes(self):
        pool = pd.DataFrame({
            "player_name": ["Kawhi"],
            "status": ["DND"],
            "salary": [8800],
            "proj_minutes": [32.0],
        })
        result = self._zero_inelig_minutes(pool)
        assert result.loc[0, "proj_minutes"] == 0.0

    def test_active_player_keeps_salary_based_proj_minutes(self):
        pool = pd.DataFrame({
            "player_name": ["LeBron"],
            "status": ["Active"],
            "salary": [9800],
            "proj_minutes": [35.0],
        })
        result = self._zero_inelig_minutes(pool)
        assert result.loc[0, "proj_minutes"] == 35.0

    def test_no_status_col_leaves_minutes_unchanged(self):
        pool = pd.DataFrame({
            "player_name": ["A"],
            "proj_minutes": [30.0],
        })
        result = self._zero_inelig_minutes(pool)
        assert result.loc[0, "proj_minutes"] == 30.0


# ---------------------------------------------------------------------------
# Tank01 status merge logic
# ---------------------------------------------------------------------------

class TestTank01StatusMerge:
    """Verify that Tank01 date-specific status overrides the DK status."""

    def _apply_tank01_status_override(
        self, pool: pd.DataFrame, tank01_pool: pd.DataFrame
    ) -> pd.DataFrame:
        """Replica of the Tank01 status-override step in 1_slate_hub.py."""
        merge_cols = ["player_name"]
        for col in ("status",):
            if col in tank01_pool.columns:
                merge_cols.append(col)

        merged = pool.merge(
            tank01_pool[merge_cols],
            on="player_name",
            how="left",
            suffixes=("", "_tank01"),
        )
        if "status_tank01" in merged.columns:
            tank01_norm = (
                merged["status_tank01"].fillna("").astype(str).str.strip().str.upper()
            )
            has_t01 = tank01_norm != ""
            merged.loc[has_t01, "status"] = merged.loc[has_t01, "status_tank01"]
            merged = merged.drop(columns=["status_tank01"])
        return merged

    def test_tank01_out_overrides_dk_active(self):
        dk_pool = pd.DataFrame({
            "player_name": ["SGA"],
            "status": ["Active"],  # DK shows Active (stale)
            "salary": [10200],
        })
        tank01 = pd.DataFrame({
            "player_name": ["SGA"],
            "status": ["OUT"],    # Tank01 shows OUT on the slate date
        })
        result = self._apply_tank01_status_override(dk_pool, tank01)
        assert result.loc[0, "status"] == "OUT"

    def test_tank01_status_used_when_available(self):
        dk_pool = pd.DataFrame({
            "player_name": ["LeBron"],
            "status": ["GTD"],
            "salary": [9800],
        })
        tank01 = pd.DataFrame({
            "player_name": ["LeBron"],
            "status": ["Active"],
        })
        result = self._apply_tank01_status_override(dk_pool, tank01)
        assert result.loc[0, "status"] == "Active"

    def test_dk_status_kept_when_tank01_has_no_match(self):
        dk_pool = pd.DataFrame({
            "player_name": ["KD"],
            "status": ["Active"],
            "salary": [9400],
        })
        tank01 = pd.DataFrame({
            "player_name": ["SGA"],  # different player
            "status": ["OUT"],
        })
        result = self._apply_tank01_status_override(dk_pool, tank01)
        # KD not in tank01 → left join yields NaN for status_tank01 → DK status kept
        assert result.loc[0, "status"] == "Active"

    def test_empty_tank01_status_does_not_override(self):
        dk_pool = pd.DataFrame({
            "player_name": ["SGA"],
            "status": ["Active"],
            "salary": [10200],
        })
        tank01 = pd.DataFrame({
            "player_name": ["SGA"],
            "status": [""],  # blank tank01 status → don't override
        })
        result = self._apply_tank01_status_override(dk_pool, tank01)
        assert result.loc[0, "status"] == "Active"
