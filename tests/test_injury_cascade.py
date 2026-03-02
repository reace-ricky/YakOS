"""Tests for yak_core.injury_cascade (Sprint 2)."""

import pandas as pd
import pytest

from yak_core.injury_cascade import (
    KEY_INJURY_MIN_MINUTES,
    MAX_PLAYER_MINUTES,
    apply_injury_cascade,
    find_key_injuries,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_pool(**kwargs) -> pd.DataFrame:
    """Build a minimal 8-player pool. Accepts column overrides via kwargs."""
    rows = [
        # OUT star PG – 28 projected minutes → key injury
        {
            "player_name": "Star PG", "team": "LAL", "pos": "PG",
            "salary": 9000, "proj": 48.0, "proj_minutes": 28.0, "status": "OUT",
        },
        # Active PG – same position as Star PG → gets 60%
        {
            "player_name": "Backup PG", "team": "LAL", "pos": "PG",
            "salary": 4500, "proj": 22.0, "proj_minutes": 18.0, "status": "Active",
        },
        # Active SG – backcourt (adjacent if Star PG is also backcourt, but
        # same group as PG → adjacent only for frontcourt injuries)
        {
            "player_name": "SG One", "team": "LAL", "pos": "SG",
            "salary": 5500, "proj": 28.0, "proj_minutes": 22.0, "status": "Active",
        },
        # Active SF – frontcourt → adjacent to Star PG (backcourt)
        {
            "player_name": "SF One", "team": "LAL", "pos": "SF",
            "salary": 6000, "proj": 30.0, "proj_minutes": 25.0, "status": "Active",
        },
        # Active PF – frontcourt → adjacent to Star PG (backcourt)
        {
            "player_name": "PF One", "team": "LAL", "pos": "PF",
            "salary": 5800, "proj": 27.0, "proj_minutes": 24.0, "status": "Active",
        },
        # Active C – frontcourt → adjacent to Star PG (backcourt)
        {
            "player_name": "C One", "team": "LAL", "pos": "C",
            "salary": 5200, "proj": 24.0, "proj_minutes": 20.0, "status": "Active",
        },
        # GSW players – should NOT be affected by LAL injury
        {
            "player_name": "GSW PG", "team": "GSW", "pos": "PG",
            "salary": 7000, "proj": 35.0, "proj_minutes": 30.0, "status": "Active",
        },
        {
            "player_name": "GSW SF", "team": "GSW", "pos": "SF",
            "salary": 6000, "proj": 29.0, "proj_minutes": 24.0, "status": "Active",
        },
    ]
    df = pd.DataFrame(rows)
    for col, val in kwargs.items():
        df[col] = val
    return df


# ---------------------------------------------------------------------------
# find_key_injuries
# ---------------------------------------------------------------------------

class TestFindKeyInjuries:
    def test_finds_out_player_with_enough_minutes(self):
        pool = _make_pool()
        ki = find_key_injuries(pool)
        assert len(ki) == 1
        assert ki.iloc[0]["player_name"] == "Star PG"

    def test_ignores_active_players(self):
        pool = _make_pool()
        ki = find_key_injuries(pool)
        assert all(ki["status"].str.upper().isin({"OUT", "IR"}))

    def test_ignores_out_player_below_minute_threshold(self):
        pool = _make_pool()
        pool.loc[pool["player_name"] == "Star PG", "proj_minutes"] = 10.0
        ki = find_key_injuries(pool)
        assert ki.empty

    def test_ir_status_counts_as_key_injury(self):
        pool = _make_pool()
        pool.loc[pool["player_name"] == "Star PG", "status"] = "IR"
        ki = find_key_injuries(pool)
        assert len(ki) == 1

    def test_empty_pool_returns_empty(self):
        assert find_key_injuries(pd.DataFrame()).empty

    def test_missing_status_column_returns_empty(self):
        pool = _make_pool().drop(columns=["status"])
        assert find_key_injuries(pool).empty

    def test_missing_proj_minutes_column_returns_empty(self):
        pool = _make_pool().drop(columns=["proj_minutes"])
        assert find_key_injuries(pool).empty

    def test_exactly_at_threshold(self):
        pool = _make_pool()
        pool.loc[pool["player_name"] == "Star PG", "proj_minutes"] = KEY_INJURY_MIN_MINUTES
        ki = find_key_injuries(pool)
        assert len(ki) == 1

    def test_multiple_key_injuries(self):
        pool = _make_pool()
        # Also make SF One OUT with high minutes
        pool.loc[pool["player_name"] == "SF One", "status"] = "OUT"
        ki = find_key_injuries(pool)
        assert len(ki) == 2


# ---------------------------------------------------------------------------
# apply_injury_cascade
# ---------------------------------------------------------------------------

class TestApplyInjuryCascade:
    def test_returns_dataframe_and_report(self):
        pool = _make_pool()
        updated, report = apply_injury_cascade(pool)
        assert isinstance(updated, pd.DataFrame)
        assert isinstance(report, list)

    def test_original_proj_preserved(self):
        pool = _make_pool()
        updated, _ = apply_injury_cascade(pool)
        assert "original_proj" in updated.columns
        # OUT player: original_proj should match original pool proj
        star_orig = pool.loc[pool["player_name"] == "Star PG", "proj"].iloc[0]
        row = updated[updated["player_name"] == "Star PG"].iloc[0]
        assert row["original_proj"] == star_orig

    def test_adjusted_proj_column_exists(self):
        pool = _make_pool()
        updated, _ = apply_injury_cascade(pool)
        assert "adjusted_proj" in updated.columns

    def test_injury_bump_fp_column_exists(self):
        pool = _make_pool()
        updated, _ = apply_injury_cascade(pool)
        assert "injury_bump_fp" in updated.columns

    def test_backup_pg_gets_bumped(self):
        """Backup PG is same-position as OUT Star PG → should receive a bump."""
        pool = _make_pool()
        updated, _ = apply_injury_cascade(pool)
        backup = updated[updated["player_name"] == "Backup PG"].iloc[0]
        assert backup["injury_bump_fp"] > 0
        assert backup["adjusted_proj"] > backup["original_proj"]

    def test_frontcourt_gets_bumped(self):
        """SF/PF/C are frontcourt (adjacent to PG backcourt) → should get bumped."""
        pool = _make_pool()
        updated, _ = apply_injury_cascade(pool)
        for name in ["SF One", "PF One", "C One"]:
            row = updated[updated["player_name"] == name].iloc[0]
            assert row["injury_bump_fp"] > 0, f"{name} should have bump"

    def test_other_team_not_affected(self):
        """GSW players should NOT be affected by LAL injury."""
        pool = _make_pool()
        updated, _ = apply_injury_cascade(pool)
        for name in ["GSW PG", "GSW SF"]:
            row = updated[updated["player_name"] == name].iloc[0]
            assert row["injury_bump_fp"] == 0.0
            assert row["adjusted_proj"] == row["original_proj"]

    def test_out_player_proj_unchanged(self):
        """The OUT player's projection should stay the same (they don't play)."""
        pool = _make_pool()
        updated, _ = apply_injury_cascade(pool)
        out_row = updated[updated["player_name"] == "Star PG"].iloc[0]
        assert out_row["injury_bump_fp"] == 0.0
        assert out_row["adjusted_proj"] == out_row["original_proj"]

    def test_proj_equals_adjusted_proj(self):
        """After cascade, pool['proj'] should equal pool['adjusted_proj']."""
        pool = _make_pool()
        updated, _ = apply_injury_cascade(pool)
        assert (updated["proj"] == updated["adjusted_proj"]).all()

    def test_max_minutes_cap(self):
        """No player's implied total projected minutes should exceed MAX_PLAYER_MINUTES."""
        pool = _make_pool()
        # Give the OUT player unrealistically high minutes to force cap
        pool.loc[pool["player_name"] == "Star PG", "proj_minutes"] = 40.0
        updated, _ = apply_injury_cascade(pool)
        for _, row in updated.iterrows():
            if row["status"] in ("OUT", "IR"):
                continue
            extra = row["injury_bump_fp"]
            orig_mins = pool.loc[pool["player_name"] == row["player_name"], "proj_minutes"].iloc[0]
            fp_per_min = row["original_proj"] / orig_mins if orig_mins > 0 else 0
            # Use a small tolerance to account for rounding in stored injury_bump_fp
            extra_mins = extra / fp_per_min if fp_per_min > 0 else 0
            assert orig_mins + extra_mins <= MAX_PLAYER_MINUTES + 0.02, (
                f"{row['player_name']} exceeded cap: {orig_mins + extra_mins:.3f}"
            )

    def test_cascade_report_structure(self):
        pool = _make_pool()
        _, report = apply_injury_cascade(pool)
        assert len(report) == 1
        entry = report[0]
        assert entry["out_player"] == "Star PG"
        assert entry["team"] == "LAL"
        assert entry["out_proj_mins"] == 28.0
        assert "beneficiaries" in entry
        assert len(entry["beneficiaries"]) > 0

    def test_cascade_report_beneficiary_fields(self):
        pool = _make_pool()
        _, report = apply_injury_cascade(pool)
        for b in report[0]["beneficiaries"]:
            assert "name" in b
            assert "original_proj" in b
            assert "adjusted_proj" in b
            assert "bump" in b
            assert "salary" in b
            assert "new_value_multiple" in b
            assert b["adjusted_proj"] >= b["original_proj"]
            assert b["bump"] >= 0

    def test_cascade_report_beneficiaries_sorted_by_bump_desc(self):
        pool = _make_pool()
        _, report = apply_injury_cascade(pool)
        bumps = [b["bump"] for b in report[0]["beneficiaries"]]
        assert bumps == sorted(bumps, reverse=True)

    def test_no_key_injuries_returns_empty_report(self):
        pool = _make_pool()
        pool["status"] = "Active"  # make everyone active
        updated, report = apply_injury_cascade(pool)
        assert report == []
        # proj should still equal original_proj
        assert (updated["proj"] == updated["original_proj"]).all()

    def test_empty_pool_returns_empty_report(self):
        updated, report = apply_injury_cascade(pd.DataFrame())
        assert isinstance(updated, pd.DataFrame)
        assert report == []

    def test_value_multiple_uses_adjusted_proj(self):
        pool = _make_pool()
        _, report = apply_injury_cascade(pool)
        for b in report[0]["beneficiaries"]:
            expected = b["adjusted_proj"] / (b["salary"] / 1000.0)
            # new_value_multiple is rounded to 2 dp; allow small rounding tolerance
            assert abs(b["new_value_multiple"] - expected) < 0.01

    def test_missing_proj_minutes_column_no_crash(self):
        pool = _make_pool().drop(columns=["proj_minutes"])
        updated, report = apply_injury_cascade(pool)
        assert isinstance(updated, pd.DataFrame)
        assert report == []

    def test_missing_status_column_no_crash(self):
        pool = _make_pool().drop(columns=["status"])
        updated, report = apply_injury_cascade(pool)
        assert isinstance(updated, pd.DataFrame)
        assert report == []

    def test_multiple_injuries_accumulate(self):
        """Two OUT players on same team → teammates accumulate bumps from both."""
        pool = _make_pool()
        # Also make SG One OUT
        pool.loc[pool["player_name"] == "SG One", "status"] = "OUT"
        pool.loc[pool["player_name"] == "SG One", "proj_minutes"] = 25.0
        updated, report = apply_injury_cascade(pool)
        assert len(report) == 2
        # Backup PG should now get bumps from both SG One (backcourt adj)
        # and from Star PG (same-pos)
        backup = updated[updated["player_name"] == "Backup PG"].iloc[0]
        assert backup["injury_bump_fp"] > 0
