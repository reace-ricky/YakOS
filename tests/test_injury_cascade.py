"""Tests for yak_core.injury_cascade (Sprint 2)."""

import pandas as pd
import pytest

from yak_core.injury_cascade import (
    KEY_INJURY_MIN_MINUTES,
    MAX_PLAYER_MINUTES,
    _PRIMARY_BACKUP_BOOST_MULT,
    _PRIMARY_BACKUP_MAX_EXTRA_MINS,
    _find_primary_backup_idx,
    _primary_backup_boost,
    apply_injury_cascade,
    apply_minutes_gap_redistribution,
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


# ---------------------------------------------------------------------------
# _primary_backup_boost unit tests
# ---------------------------------------------------------------------------

class TestPrimaryBackupBoost:
    """Unit tests for the _primary_backup_boost helper."""

    def _make_eligible(self) -> pd.DataFrame:
        """Small eligible DataFrame with players across position/minute tiers."""
        rows = [
            # Same position as out PG, mid-rotation (18 min) → primary backup
            {"player_name": "Backup PG", "pos": "PG", "proj_minutes": 18.0},
            # Same position as out PG, but only 10 min (below 12 threshold)
            {"player_name": "Deep Bench PG", "pos": "PG", "proj_minutes": 10.0},
            # Different position
            {"player_name": "SG One", "pos": "SG", "proj_minutes": 22.0},
            # Same position but starter (28+ min → above 28 threshold)
            {"player_name": "Starter PG", "pos": "PG", "proj_minutes": 30.0},
        ]
        return pd.DataFrame(rows)

    def test_identifies_same_pos_candidate(self):
        """Primary backup must be the same-pos player in 12–28 min range."""
        df = self._make_eligible()
        weights = {i: 10.0 for i in df.index}
        boosted = _primary_backup_boost(weights, df, "PG", 28.0)
        # Backup PG is index 0; should have 2x weight
        assert boosted[0] == pytest.approx(10.0 * _PRIMARY_BACKUP_BOOST_MULT)

    def test_applies_2x_multiplier(self):
        """The primary backup's weight must be exactly _PRIMARY_BACKUP_BOOST_MULT × original."""
        df = self._make_eligible()
        weights = {i: 5.0 * (i + 1) for i in df.index}  # varied weights
        boosted = _primary_backup_boost(weights, df, "PG", 28.0)
        orig_w = weights[0]
        assert boosted[0] == pytest.approx(orig_w * _PRIMARY_BACKUP_BOOST_MULT)

    def test_non_primary_weights_unchanged(self):
        """All players except the primary backup must keep their original weight."""
        df = self._make_eligible()
        weights = {i: 8.0 for i in df.index}
        boosted = _primary_backup_boost(weights, df, "PG", 28.0)
        for idx in [1, 2, 3]:
            assert boosted[idx] == pytest.approx(8.0)

    def test_no_candidate_returns_unchanged(self):
        """If no same-pos player is in 12–28 min range, return weights unchanged."""
        df = self._make_eligible()
        # Ask for SF position — no SF in the eligible set
        weights = {i: 10.0 for i in df.index}
        boosted = _primary_backup_boost(weights, df, "SF", 25.0)
        assert boosted == weights

    def test_highest_minutes_chosen_among_same_pos(self):
        """When multiple same-pos candidates exist, the one with highest minutes wins."""
        rows = [
            {"player_name": "PG A", "pos": "PG", "proj_minutes": 15.0},
            {"player_name": "PG B", "pos": "PG", "proj_minutes": 20.0},  # highest
            {"player_name": "SG One", "pos": "SG", "proj_minutes": 22.0},
        ]
        df = pd.DataFrame(rows)
        weights = {i: 10.0 for i in df.index}
        boosted = _primary_backup_boost(weights, df, "PG", 28.0)
        # Index 1 (PG B, 20 min) should be boosted
        assert boosted[1] == pytest.approx(10.0 * _PRIMARY_BACKUP_BOOST_MULT)
        assert boosted[0] == pytest.approx(10.0)  # PG A unchanged

    def test_above_28min_not_a_candidate(self):
        """A same-pos player with >= 28 min is NOT in the primary backup tier."""
        rows = [
            {"player_name": "PG Starter", "pos": "PG", "proj_minutes": 28.0},
            {"player_name": "SG One", "pos": "SG", "proj_minutes": 22.0},
        ]
        df = pd.DataFrame(rows)
        weights = {i: 10.0 for i in df.index}
        boosted = _primary_backup_boost(weights, df, "PG", 28.0)
        assert boosted == weights  # no candidate, unchanged

    def test_exactly_12min_qualifies(self):
        """A same-pos player at exactly 12 min is the lower bound (inclusive)."""
        rows = [
            {"player_name": "PG Edge", "pos": "PG", "proj_minutes": 12.0},
            {"player_name": "C One", "pos": "C", "proj_minutes": 20.0},
        ]
        df = pd.DataFrame(rows)
        weights = {i: 10.0 for i in df.index}
        boosted = _primary_backup_boost(weights, df, "PG", 25.0)
        assert boosted[0] == pytest.approx(10.0 * _PRIMARY_BACKUP_BOOST_MULT)


# ---------------------------------------------------------------------------
# Integration tests: primary backup boost in apply_injury_cascade
# ---------------------------------------------------------------------------

class TestPrimaryBackupBoostIntegration:
    """Verify that primary backup boost concentrates minutes correctly in the
    full apply_injury_cascade pipeline."""

    def _make_pool_with_clear_backup(self) -> pd.DataFrame:
        """Pool where Backup PG (18 min) is the clear primary backup for Star PG (OUT, 28 min).

        All other teammates are different positions, ensuring the 2x boost has
        a measurable effect on minute concentration.
        """
        return pd.DataFrame([
            {"player_name": "Star PG", "team": "LAL", "pos": "PG",
             "salary": 9000, "proj": 50.0, "proj_minutes": 28.0, "status": "OUT"},
            {"player_name": "Backup PG", "team": "LAL", "pos": "PG",
             "salary": 4500, "proj": 20.0, "proj_minutes": 18.0, "status": "Active"},
            {"player_name": "SF One", "team": "LAL", "pos": "SF",
             "salary": 6000, "proj": 30.0, "proj_minutes": 25.0, "status": "Active"},
            {"player_name": "PF One", "team": "LAL", "pos": "PF",
             "salary": 5500, "proj": 27.0, "proj_minutes": 24.0, "status": "Active"},
            {"player_name": "C One", "team": "LAL", "pos": "C",
             "salary": 5200, "proj": 24.0, "proj_minutes": 20.0, "status": "Active"},
        ])

    def test_primary_backup_gets_most_minutes(self):
        """After boost, Backup PG should receive more extra minutes than any other player."""
        pool = self._make_pool_with_clear_backup()
        updated, report = apply_injury_cascade(pool)
        # Gather extra_minutes for active LAL players
        extra_by_player = {}
        for b in report[0]["beneficiaries"]:
            extra_by_player[b["name"]] = b["extra_minutes"]
        backup_extra = extra_by_player.get("Backup PG", 0)
        others_extra = [v for k, v in extra_by_player.items() if k != "Backup PG"]
        assert backup_extra > max(others_extra), (
            f"Backup PG ({backup_extra:.1f} min) should beat max secondary "
            f"({max(others_extra):.1f} min)"
        )

    def test_primary_backup_12min_cap_enforced(self):
        """Backup PG must never receive more than _PRIMARY_BACKUP_MAX_EXTRA_MINS
        extra minutes from a single injury, even with the boost applied."""
        pool = self._make_pool_with_clear_backup()
        updated, report = apply_injury_cascade(pool)
        for b in report[0]["beneficiaries"]:
            if b["name"] == "Backup PG":
                assert b["extra_minutes"] <= _PRIMARY_BACKUP_MAX_EXTRA_MINS + 0.01, (
                    f"Backup PG extra_minutes {b['extra_minutes']:.2f} exceeds cap "
                    f"{_PRIMARY_BACKUP_MAX_EXTRA_MINS}"
                )

    def test_overflow_redistributed_total_minutes_conserved(self):
        """Total extra minutes distributed must equal the OUT player's projected minutes
        (or less, only if headroom constraints prevent full redistribution)."""
        pool = self._make_pool_with_clear_backup()
        updated, report = apply_injury_cascade(pool)
        total_extra = sum(b["extra_minutes"] for b in report[0]["beneficiaries"])
        out_mins = report[0]["out_proj_mins"]
        # Total distributed must not exceed original out_mins
        assert total_extra <= out_mins + 0.05

    def test_no_primary_backup_weights_unchanged(self):
        """If there is no same-pos candidate in 12–28 min range, the cascade
        should still run (no crash) and distribute normally."""
        pool = pd.DataFrame([
            {"player_name": "Star C", "team": "BOS", "pos": "C",
             "salary": 9000, "proj": 50.0, "proj_minutes": 30.0, "status": "OUT"},
            # Two SG players — no C backup in 12-28 min range
            {"player_name": "SG A", "team": "BOS", "pos": "SG",
             "salary": 5000, "proj": 25.0, "proj_minutes": 22.0, "status": "Active"},
            {"player_name": "SF A", "team": "BOS", "pos": "SF",
             "salary": 5200, "proj": 26.0, "proj_minutes": 24.0, "status": "Active"},
        ])
        updated, report = apply_injury_cascade(pool)
        assert len(report) == 1
        # Both players should still get bumped (normal cascade fires)
        bumps = {b["name"]: b["bump"] for b in report[0]["beneficiaries"]}
        assert bumps["SG A"] > 0
        assert bumps["SF A"] > 0


# ---------------------------------------------------------------------------
# apply_minutes_gap_redistribution tests
# ---------------------------------------------------------------------------

class TestApplyMinutesGapRedistribution:
    """Tests for the minutes gap redistribution function that catches
    structural gaps from players not on the DK slate."""

    def _make_gap_pool(self, total_proj_minutes: float = 160.0) -> pd.DataFrame:
        """Build a pool for one team with a specified total of projected minutes.

        Creates 5 active players whose proj_minutes sum to ``total_proj_minutes``.
        Each player gets an equal share of the total minutes.
        """
        per_player = total_proj_minutes / 5
        rows = [
            {"player_name": f"Player {i}", "team": "LAL", "pos": pos,
             "salary": 5000 + i * 500, "proj": per_player * 1.0,
             "proj_minutes": per_player, "status": "Active"}
            for i, pos in enumerate(["PG", "SG", "SF", "PF", "C"])
        ]
        return pd.DataFrame(rows)

    def test_gap_detected_and_minutes_redistributed(self):
        """Team with 160 projected minutes → 80 min gap > 30 threshold → redistribution fires."""
        pool = self._make_gap_pool(total_proj_minutes=160.0)
        result = apply_minutes_gap_redistribution(pool)
        # All players should receive a bump
        assert (result["minutes_gap_bump_min"] > 0).all()
        assert (result["minutes_gap_bump_fp"] > 0).all()
        # proj_minutes should increase
        assert result["proj_minutes"].sum() > 160.0
        # proj should increase
        assert result["proj"].sum() > pool["proj"].sum()

    def test_no_gap_below_threshold(self):
        """Team with 230 projected minutes → gap = 10 < 30 threshold → no redistribution."""
        pool = self._make_gap_pool(total_proj_minutes=230.0)
        result = apply_minutes_gap_redistribution(pool)
        assert (result["minutes_gap_bump_min"] == 0).all()
        assert (result["minutes_gap_bump_fp"] == 0).all()
        # Projections unchanged
        assert result["proj"].sum() == pytest.approx(pool["proj"].sum())

    def test_no_gap_at_240(self):
        """Team already at 240 projected minutes → gap = 0 → no redistribution."""
        pool = self._make_gap_pool(total_proj_minutes=240.0)
        result = apply_minutes_gap_redistribution(pool)
        assert (result["minutes_gap_bump_min"] == 0).all()
        assert (result["minutes_gap_bump_fp"] == 0).all()

    def test_no_player_exceeds_40_minutes(self):
        """After redistribution, no player should exceed MAX_PLAYER_MINUTES (40)."""
        pool = self._make_gap_pool(total_proj_minutes=160.0)
        result = apply_minutes_gap_redistribution(pool)
        assert (result["proj_minutes"] <= MAX_PLAYER_MINUTES + 0.1).all(), (
            f"Players exceeded 40 min cap: {result[['player_name','proj_minutes']].to_dict('records')}"
        )

    def test_no_player_exceeds_40_with_high_base_minutes(self):
        """Players starting near 40 min should be capped, not over-boosted."""
        rows = [
            {"player_name": "Star", "team": "LAL", "pos": "PG",
             "salary": 9000, "proj": 50.0, "proj_minutes": 38.0, "status": "Active"},
            {"player_name": "Bench1", "team": "LAL", "pos": "SG",
             "salary": 4000, "proj": 15.0, "proj_minutes": 12.0, "status": "Active"},
            {"player_name": "Bench2", "team": "LAL", "pos": "SF",
             "salary": 4500, "proj": 18.0, "proj_minutes": 14.0, "status": "Active"},
        ]
        pool = pd.DataFrame(rows)
        # Total = 64 min, gap = 176 → huge gap, but Star is nearly capped
        result = apply_minutes_gap_redistribution(pool)
        assert (result["proj_minutes"] <= MAX_PLAYER_MINUTES + 0.1).all()

    def test_empty_pool_returns_empty(self):
        """Empty DataFrame should be returned unchanged."""
        result = apply_minutes_gap_redistribution(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_none_pool_returns_none(self):
        """None input should be returned as-is."""
        result = apply_minutes_gap_redistribution(None)
        assert result is None

    def test_missing_proj_minutes_column_noop(self):
        """Pool without proj_minutes column should be returned unchanged."""
        pool = self._make_gap_pool()
        pool = pool.drop(columns=["proj_minutes"])
        result = apply_minutes_gap_redistribution(pool)
        assert "minutes_gap_bump_min" not in result.columns

    def test_missing_proj_column_noop(self):
        """Pool without proj column should be returned unchanged."""
        pool = self._make_gap_pool()
        pool = pool.drop(columns=["proj"])
        result = apply_minutes_gap_redistribution(pool)
        assert "minutes_gap_bump_min" not in result.columns

    def test_other_team_unaffected(self):
        """Players on a team with normal minutes should not be redistributed."""
        gap_pool = self._make_gap_pool(total_proj_minutes=160.0)
        # Add a GSW team with normal minutes (~240)
        gsw_rows = pd.DataFrame([
            {"player_name": "GSW PG", "team": "GSW", "pos": "PG",
             "salary": 7000, "proj": 35.0, "proj_minutes": 30.0, "status": "Active"},
            {"player_name": "GSW SG", "team": "GSW", "pos": "SG",
             "salary": 6000, "proj": 30.0, "proj_minutes": 28.0, "status": "Active"},
            {"player_name": "GSW SF", "team": "GSW", "pos": "SF",
             "salary": 6500, "proj": 32.0, "proj_minutes": 26.0, "status": "Active"},
            {"player_name": "GSW PF", "team": "GSW", "pos": "PF",
             "salary": 5500, "proj": 28.0, "proj_minutes": 25.0, "status": "Active"},
            {"player_name": "GSW C", "team": "GSW", "pos": "C",
             "salary": 7500, "proj": 38.0, "proj_minutes": 32.0, "status": "Active"},
            {"player_name": "GSW Bench1", "team": "GSW", "pos": "PG",
             "salary": 4000, "proj": 15.0, "proj_minutes": 18.0, "status": "Active"},
            {"player_name": "GSW Bench2", "team": "GSW", "pos": "SG",
             "salary": 3800, "proj": 12.0, "proj_minutes": 16.0, "status": "Active"},
            {"player_name": "GSW Bench3", "team": "GSW", "pos": "SF",
             "salary": 3500, "proj": 10.0, "proj_minutes": 15.0, "status": "Active"},
            {"player_name": "GSW Bench4", "team": "GSW", "pos": "C",
             "salary": 3600, "proj": 11.0, "proj_minutes": 14.0, "status": "Active"},
            {"player_name": "GSW Bench5", "team": "GSW", "pos": "PF",
             "salary": 3700, "proj": 10.0, "proj_minutes": 12.0, "status": "Active"},
        ])
        pool = pd.concat([gap_pool, gsw_rows], ignore_index=True)
        result = apply_minutes_gap_redistribution(pool)
        # GSW players should have zero bump
        gsw_mask = result["team"] == "GSW"
        assert (result.loc[gsw_mask, "minutes_gap_bump_min"] == 0).all()
        # LAL players should have bumps
        lal_mask = result["team"] == "LAL"
        assert (result.loc[lal_mask, "minutes_gap_bump_min"] > 0).all()

    def test_out_players_excluded_from_redistribution(self):
        """OUT players should not receive any gap redistribution."""
        rows = [
            {"player_name": "OUT Star", "team": "LAL", "pos": "PG",
             "salary": 9000, "proj": 48.0, "proj_minutes": 28.0, "status": "OUT"},
            {"player_name": "Active PG", "team": "LAL", "pos": "PG",
             "salary": 5000, "proj": 20.0, "proj_minutes": 18.0, "status": "Active"},
            {"player_name": "Active SG", "team": "LAL", "pos": "SG",
             "salary": 5500, "proj": 22.0, "proj_minutes": 20.0, "status": "Active"},
        ]
        pool = pd.DataFrame(rows)
        # Active total = 38 min, gap = 202 → redistribution fires
        result = apply_minutes_gap_redistribution(pool)
        out_row = result[result["player_name"] == "OUT Star"].iloc[0]
        assert out_row["minutes_gap_bump_min"] == 0
        assert out_row["minutes_gap_bump_fp"] == 0

    def test_fp_bump_uses_fp_per_min_rate(self):
        """The FP bump should equal extra_minutes × (original_proj / original_proj_minutes)."""
        rows = [
            {"player_name": "Player A", "team": "LAL", "pos": "PG",
             "salary": 5000, "proj": 30.0, "proj_minutes": 20.0, "status": "Active"},
            {"player_name": "Player B", "team": "LAL", "pos": "SG",
             "salary": 5000, "proj": 20.0, "proj_minutes": 20.0, "status": "Active"},
        ]
        pool = pd.DataFrame(rows)
        # Total = 40 min, gap = 200 → redistribution
        original_proj_a = 30.0
        original_min_a = 20.0
        fp_per_min_a = original_proj_a / original_min_a  # 1.5

        result = apply_minutes_gap_redistribution(pool)
        row_a = result[result["player_name"] == "Player A"].iloc[0]
        extra_min = row_a["minutes_gap_bump_min"]
        expected_fp = round(extra_min * fp_per_min_a, 2)
        assert row_a["minutes_gap_bump_fp"] == pytest.approx(expected_fp, abs=0.1)

    def test_adjusted_proj_column_updated(self):
        """If adjusted_proj column exists (post-cascade), it should be updated."""
        pool = self._make_gap_pool(total_proj_minutes=160.0)
        pool["adjusted_proj"] = pool["proj"]
        result = apply_minutes_gap_redistribution(pool)
        # adjusted_proj should match proj after redistribution
        assert (result["proj"] == result["adjusted_proj"]).all()

    def test_exactly_at_threshold_no_redistribution(self):
        """Team with gap exactly == 30 (threshold) should NOT trigger redistribution."""
        pool = self._make_gap_pool(total_proj_minutes=210.0)
        # gap = 240 - 210 = 30, threshold is 30, condition is gap <= 30 → skip
        result = apply_minutes_gap_redistribution(pool)
        assert (result["minutes_gap_bump_min"] == 0).all()

    def test_gap_just_above_threshold(self):
        """Team with gap = 31 (just above threshold) should trigger redistribution."""
        # Use 8 players so each has ~26 min (well below 40 cap), ensuring headroom > 0
        rows = [
            {"player_name": f"P{i}", "team": "LAL", "pos": pos,
             "salary": 5000, "proj": 26.0, "proj_minutes": 26.125,
             "status": "Active"}
            for i, pos in enumerate(["PG", "SG", "SF", "PF", "C", "PG", "SG", "SF"])
        ]
        pool = pd.DataFrame(rows)
        # Total = 26.125 * 8 = 209 min, gap = 31 > 30 → fires
        result = apply_minutes_gap_redistribution(pool)
        assert (result["minutes_gap_bump_min"] > 0).any()
