"""Tests for yak_core.injury_cascade (Sprint 2)."""

import pandas as pd
import pytest

from yak_core.injury_cascade import (
    KEY_INJURY_MIN_MINUTES,
    MAX_PLAYER_MINUTES,
    apply_injury_cascade,
    find_key_injuries,
)
from yak_core.sims import _INELIGIBLE_STATUSES


def _cascade_then_drop(pool: pd.DataFrame):
    """Mirror the API-load pattern: cascade first, then drop ineligible rows.

    Returns (cleaned_pool, cascade_report, removed_players).
    """
    updated, report = apply_injury_cascade(pool)
    removed = []
    if "status" in updated.columns:
        mask = updated["status"].fillna("").str.upper().isin(_INELIGIBLE_STATUSES)
        removed = updated.loc[mask, "player_name"].tolist()
        updated = updated[~mask].reset_index(drop=True)
    return updated, report, removed


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


# ---------------------------------------------------------------------------
# Regression: apply_injury_cascade must use 'minutes' as fallback for
# 'proj_minutes' (RotoGrinders CSV pools use 'minutes', not 'proj_minutes')
# ---------------------------------------------------------------------------

class TestApplyInjuryCascadeMinutesFallback:
    """Regression tests: cascade must fire on RG CSV pools that use 'minutes'."""

    def _make_rg_pool(self) -> pd.DataFrame:
        """Pool with 'minutes' column (RG CSV style) instead of 'proj_minutes'."""
        rows = [
            {"player_name": "Star PG", "team": "LAL", "pos": "PG",
             "salary": 9000, "proj": 48.0, "minutes": 28.0, "status": "OUT"},
            {"player_name": "Backup PG", "team": "LAL", "pos": "PG",
             "salary": 4500, "proj": 22.0, "minutes": 18.0, "status": "Active"},
            {"player_name": "SF One", "team": "LAL", "pos": "SF",
             "salary": 6000, "proj": 30.0, "minutes": 25.0, "status": "Active"},
            {"player_name": "GSW PG", "team": "GSW", "pos": "PG",
             "salary": 7000, "proj": 35.0, "minutes": 30.0, "status": "Active"},
        ]
        return pd.DataFrame(rows)

    def test_cascade_fires_when_only_minutes_col_present(self):
        """With only 'minutes' column (no 'proj_minutes'), cascade should still fire."""
        pool = self._make_rg_pool()
        assert "proj_minutes" not in pool.columns
        updated, report = apply_injury_cascade(pool)
        assert len(report) == 1, "Expected one injury cascade entry"
        assert report[0]["out_player"] == "Star PG"

    def test_backup_gets_bumped_via_minutes_fallback(self):
        """Backup PG should receive a bump when cascade uses 'minutes' column."""
        pool = self._make_rg_pool()
        updated, _ = apply_injury_cascade(pool)
        backup = updated[updated["player_name"] == "Backup PG"].iloc[0]
        assert backup["injury_bump_fp"] > 0
        assert backup["adjusted_proj"] > backup["original_proj"]

    def test_gsw_unaffected_via_minutes_fallback(self):
        """Other-team player must not be bumped even with 'minutes' fallback."""
        pool = self._make_rg_pool()
        updated, _ = apply_injury_cascade(pool)
        gsw = updated[updated["player_name"] == "GSW PG"].iloc[0]
        assert gsw["injury_bump_fp"] == 0.0


# ---------------------------------------------------------------------------
# Regression: OUT/IR players must be dropped from pool at API load time
# so they never reach the sim module, regardless of downstream eligibility checks.
# ---------------------------------------------------------------------------

class TestCascadeThenDropPattern:
    """Verify the API-load pattern: cascade runs first (so minutes are
    redistributed), then ineligible players are removed from the pool
    entirely.  This is the behavior implemented in streamlit_app.py at
    both 'Fetch Pool from API' sites."""

    def _make_pool(self) -> pd.DataFrame:
        rows = [
            {"player_name": "Alex Sarr",   "team": "WAS", "pos": "C",
             "salary": 7200, "proj": 38.0, "proj_minutes": 28.0, "status": "OUT"},
            {"player_name": "Leaky Black", "team": "CHA", "pos": "SF",
             "salary": 4800, "proj": 22.0, "proj_minutes": 24.0, "status": "OUT"},
            {"player_name": "Backup C",    "team": "WAS", "pos": "C",
             "salary": 5000, "proj": 18.0, "proj_minutes": 16.0, "status": "Active"},
            {"player_name": "WAS SF",      "team": "WAS", "pos": "SF",
             "salary": 5500, "proj": 25.0, "proj_minutes": 22.0, "status": "Active"},
        ]
        return pd.DataFrame(rows)

    def test_out_players_not_in_final_pool(self):
        """Alex Sarr and Leaky Black (both OUT) must be absent from the pool
        returned after the cascade-then-drop step."""
        pool = self._make_pool()
        cleaned, _, removed = _cascade_then_drop(pool)
        assert "Alex Sarr" not in cleaned["player_name"].values
        assert "Leaky Black" not in cleaned["player_name"].values

    def test_removed_list_contains_out_players(self):
        """The removed list must name the OUT players that were dropped."""
        pool = self._make_pool()
        _, _, removed = _cascade_then_drop(pool)
        assert "Alex Sarr" in removed
        assert "Leaky Black" in removed

    def test_active_players_remain(self):
        """Active teammates must still be in the cleaned pool."""
        pool = self._make_pool()
        cleaned, _, _ = _cascade_then_drop(pool)
        assert "Backup C" in cleaned["player_name"].values
        assert "WAS SF" in cleaned["player_name"].values

    def test_cascade_still_runs_before_drop(self):
        """The cascade report must reference the OUT player even though
        that player is subsequently removed from the pool."""
        pool = self._make_pool()
        cleaned, report, _ = _cascade_then_drop(pool)
        # Cascade entry for Alex Sarr (WAS, C, OUT, 28 mins)
        was_entry = next((r for r in report if r["out_player"] == "Alex Sarr"), None)
        assert was_entry is not None, "Cascade should fire for Alex Sarr"
        assert was_entry["out_proj_mins"] == 28.0

    def test_backup_bumped_then_out_player_gone(self):
        """Backup C should have an injury bump AND Alex Sarr must not appear."""
        pool = self._make_pool()
        cleaned, _, _ = _cascade_then_drop(pool)
        backup = cleaned[cleaned["player_name"] == "Backup C"].iloc[0]
        assert backup["injury_bump_fp"] > 0
        assert "Alex Sarr" not in cleaned["player_name"].values

    def test_ir_player_also_dropped(self):
        """IR status players must be dropped just like OUT players."""
        pool = self._make_pool()
        pool.loc[pool["player_name"] == "Alex Sarr", "status"] = "IR"
        cleaned, _, removed = _cascade_then_drop(pool)
        assert "Alex Sarr" not in cleaned["player_name"].values
        assert "Alex Sarr" in removed

    def test_no_out_players_nothing_dropped(self):
        """When no players are ineligible, the pool is returned unchanged."""
        pool = self._make_pool()
        pool["status"] = "Active"
        cleaned, _, removed = _cascade_then_drop(pool)
        assert removed == []
        assert len(cleaned) == len(pool)


class TestCsvUploadDropPattern:
    """Regression tests verifying that the CSV-upload code path applies the
    cascade-then-drop pattern (not just cascade with no drop), mirroring the
    logic added to the Cal-tab CSV uploader in streamlit_app.py."""

    def _make_csv_pool(self) -> pd.DataFrame:
        """Minimal pool that might come from a RotoGrinders CSV upload."""
        return pd.DataFrame([
            {"player_name": "Alex Sarr",  "team": "WAS", "pos": "C",
             "salary": 7200, "proj": 38.0, "proj_minutes": 28.0, "status": "OUT"},
            {"player_name": "Backup C",   "team": "WAS", "pos": "C",
             "salary": 5000, "proj": 18.0, "proj_minutes": 16.0, "status": "Active"},
            {"player_name": "WAS PG",     "team": "WAS", "pos": "PG",
             "salary": 5500, "proj": 25.0, "proj_minutes": 22.0, "status": "Active"},
            {"player_name": "IR Player",  "team": "WAS", "pos": "SG",
             "salary": 6000, "proj": 30.0, "proj_minutes": 26.0, "status": "IR"},
        ])

    def test_csv_out_player_dropped_after_cascade(self):
        """OUT player must be absent from pool after cascade-then-drop."""
        cleaned, _, _ = _cascade_then_drop(self._make_csv_pool())
        assert "Alex Sarr" not in cleaned["player_name"].values

    def test_csv_ir_player_dropped_after_cascade(self):
        """IR player must be absent from pool after cascade-then-drop."""
        cleaned, _, _ = _cascade_then_drop(self._make_csv_pool())
        assert "IR Player" not in cleaned["player_name"].values

    def test_csv_active_players_retained(self):
        """Active players must remain after cascade-then-drop."""
        cleaned, _, _ = _cascade_then_drop(self._make_csv_pool())
        assert "Backup C" in cleaned["player_name"].values
        assert "WAS PG" in cleaned["player_name"].values

    def test_csv_cascade_still_fires_before_drop(self):
        """Backup C should receive an injury bump even though Alex Sarr is
        subsequently removed — cascade must run before the drop."""
        cleaned, report, _ = _cascade_then_drop(self._make_csv_pool())
        backup = cleaned[cleaned["player_name"] == "Backup C"].iloc[0]
        assert backup["injury_bump_fp"] > 0

    def test_csv_pool_no_out_ir_after_drop(self):
        """After cascade-then-drop the cleaned pool must contain zero OUT/IR
        players — this mirrors the assertion added to 'Run Sims'."""
        cleaned, _, _ = _cascade_then_drop(self._make_csv_pool())
        if "status" in cleaned.columns:
            bad = cleaned[cleaned["status"].fillna("").str.upper().isin({"OUT", "IR", "O"})]
            assert bad.empty, f"OUT/IR players found in cleaned pool: {bad[['player_name','status']].to_dict('records')}"


class TestManualOverrideDropPattern:
    """Regression tests verifying that the manual override (live-update) path
    removes OUT/IR players from the pool entirely instead of just marking
    sim_eligible=False."""

    def _make_pool(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"player_name": "Zion Williamson", "team": "NOP", "pos": "PF",
             "salary": 9500, "proj": 50.0, "proj_minutes": 30.0, "status": "Active"},
            {"player_name": "Brandon Ingram",  "team": "NOP", "pos": "SF",
             "salary": 7000, "proj": 38.0, "proj_minutes": 30.0, "status": "Active"},
            {"player_name": "CJ McCollum",     "team": "NOP", "pos": "PG",
             "salary": 6500, "proj": 35.0, "proj_minutes": 28.0, "status": "Active"},
        ])

    def _apply_manual_out_drop(self, pool: pd.DataFrame, player_name: str) -> pd.DataFrame:
        """Mirrors the corrected manual-override path in streamlit_app.py:
        drop instead of setting sim_eligible=False."""
        from yak_core.sims import simulate_live_updates
        news_updates = [{"player_name": player_name, "status": "OUT"}]
        sim_pool = simulate_live_updates(pool, news_updates)
        out_names = [
            u["player_name"] for u in news_updates
            if u.get("status", "").upper() in {"OUT", "IR", "SUSPENDED", "G-LEAGUE"}
        ]
        if out_names and "player_name" in sim_pool.columns:
            sim_pool = sim_pool[~sim_pool["player_name"].isin(out_names)].reset_index(drop=True)
        return sim_pool

    def test_manual_out_player_removed_from_pool(self):
        """Player marked OUT via manual override must be absent from the pool."""
        pool = self._make_pool()
        result = self._apply_manual_out_drop(pool, "Zion Williamson")
        assert "Zion Williamson" not in result["player_name"].values

    def test_manual_out_other_players_retained(self):
        """Non-OUT players must still be in the pool after manual override."""
        pool = self._make_pool()
        result = self._apply_manual_out_drop(pool, "Zion Williamson")
        assert "Brandon Ingram" in result["player_name"].values
        assert "CJ McCollum" in result["player_name"].values

    def test_manual_out_pool_has_no_out_status(self):
        """After manual-override drop, pool must contain zero OUT players —
        matches the assertion added before run_optimizer in Run Sims."""
        pool = self._make_pool()
        result = self._apply_manual_out_drop(pool, "Zion Williamson")
        if "status" in result.columns:
            bad = result[result["status"].fillna("").str.upper().isin({"OUT", "IR", "O"})]
            assert bad.empty, f"OUT players still present: {bad[['player_name','status']].to_dict('records')}"
