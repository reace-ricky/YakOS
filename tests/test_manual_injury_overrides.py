"""Tests for manual injury override functions in yak_core/live.py."""

import os
import tempfile

import pandas as pd
import pytest

from yak_core.live import (
    apply_manual_injury_overrides_to_pool,
    load_manual_injury_overrides,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(**kwargs) -> pd.DataFrame:
    """Build a minimal player pool for testing."""
    defaults = {
        "player_name": ["Alex Sarr", "Cooper McNeeley", "Active Player"],
        "player_id": ["SARR_001", "940541851189", "ACT_001"],
        "team": ["WAS", "IND", "LAL"],
        "status": ["Active", "Active", "Active"],
        "salary": [7800, 3500, 8000],
        "proj": [30.0, 5.0, 40.0],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def _write_overrides_csv(tmp_dir: str, rows: list) -> str:
    """Write a manual_injuries.csv to tmp_dir and return the path."""
    import csv

    path = os.path.join(tmp_dir, "manual_injuries.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# load_manual_injury_overrides
# ---------------------------------------------------------------------------

class TestLoadManualInjuryOverrides:
    def test_returns_empty_df_when_file_missing(self, monkeypatch, tmp_path):
        """Should return empty DataFrame with correct columns when CSV is absent."""
        import yak_core.live as _live
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        result = load_manual_injury_overrides()
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["playerID", "player", "designation"]
        assert result.empty

    def test_returns_empty_df_on_corrupt_csv(self, monkeypatch, tmp_path):
        """Should silently return empty DataFrame when file is unreadable."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "manual_injuries.csv").write_bytes(b"\x00\xff\xfe bad bytes")
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        result = load_manual_injury_overrides()
        assert result.empty

    def test_filters_inactive_rows(self, monkeypatch, tmp_path):
        """Rows where active=False should not be returned."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [
            {"playerID": "001", "player": "Active Player", "designation": "Out", "notes": "", "active": "True"},
            {"playerID": "002", "player": "Inactive Player", "designation": "Out", "notes": "", "active": "False"},
        ]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        result = load_manual_injury_overrides()
        assert len(result) == 1
        assert result.iloc[0]["player"] == "Active Player"

    def test_returns_active_rows(self, monkeypatch, tmp_path):
        """Rows where active=True should be returned."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [
            {"playerID": "001", "player": "Alex Sarr", "designation": "Out", "notes": "hamstring", "active": "True"},
        ]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        result = load_manual_injury_overrides()
        assert len(result) == 1
        assert result.iloc[0]["player"] == "Alex Sarr"
        assert result.iloc[0]["designation"] == "Out"

    def test_accepts_numeric_one_as_active(self, monkeypatch, tmp_path):
        """active=1 should be treated as active=True."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [
            {"playerID": "001", "player": "Player A", "designation": "Out", "notes": "", "active": "1"},
        ]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        result = load_manual_injury_overrides()
        assert len(result) == 1

    def test_returns_only_required_columns(self, monkeypatch, tmp_path):
        """Result should contain only playerID, player, designation columns."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [
            {"playerID": "001", "player": "Alex Sarr", "team": "WAS",
             "designation": "Out", "notes": "hamstring", "active": "True"},
        ]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        result = load_manual_injury_overrides()
        for col in ["playerID", "player", "designation"]:
            assert col in result.columns
        assert "notes" not in result.columns
        assert "team" not in result.columns


# ---------------------------------------------------------------------------
# apply_manual_injury_overrides_to_pool
# ---------------------------------------------------------------------------

class TestApplyManualInjuryOverridesToPool:
    def test_returns_dataframe(self, monkeypatch, tmp_path):
        """Should always return a DataFrame."""
        import yak_core.live as _live
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, monkeypatch, tmp_path):
        """Should return a copy and not modify the input DataFrame."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "", "player": "Alex Sarr", "designation": "Out", "notes": "", "active": "True"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        original_status = pool["status"].tolist()
        apply_manual_injury_overrides_to_pool(pool)
        assert pool["status"].tolist() == original_status

    def test_no_overrides_returns_unchanged_pool(self, monkeypatch, tmp_path):
        """When CSV is absent, pool should be returned unchanged."""
        import yak_core.live as _live
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        assert result["status"].tolist() == pool["status"].tolist()

    def test_matches_by_player_name(self, monkeypatch, tmp_path):
        """Override should apply when player name matches."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "", "player": "Alex Sarr", "designation": "Out", "notes": "", "active": "True"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        sarr_row = result[result["player_name"] == "Alex Sarr"]
        assert sarr_row["status"].iloc[0] == "OUT"

    def test_matches_by_player_id(self, monkeypatch, tmp_path):
        """Override should apply when Tank01 playerID matches pool's player_id."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "940541851189", "player": "Cooper McNeeley",
                 "designation": "Out", "notes": "", "active": "True"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        cm_row = result[result["player_id"] == "940541851189"]
        assert cm_row["status"].iloc[0] == "OUT"

    def test_active_player_not_overridden(self, monkeypatch, tmp_path):
        """Players not in the override list should keep their original status."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "", "player": "Alex Sarr", "designation": "Out", "notes": "", "active": "True"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        active_row = result[result["player_name"] == "Active Player"]
        assert active_row["status"].iloc[0] == "Active"

    def test_inactive_override_ignored(self, monkeypatch, tmp_path):
        """Rows with active=False should not affect the pool."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "", "player": "Alex Sarr", "designation": "Out",
                 "notes": "", "active": "False"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        sarr_row = result[result["player_name"] == "Alex Sarr"]
        assert sarr_row["status"].iloc[0] == "Active"

    def test_day_to_day_maps_to_gtd(self, monkeypatch, tmp_path):
        """'Day-To-Day' designation should map to 'GTD' status."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "", "player": "Alex Sarr", "designation": "Day-To-Day",
                 "notes": "", "active": "True"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        sarr_row = result[result["player_name"] == "Alex Sarr"]
        assert sarr_row["status"].iloc[0] == "GTD"

    def test_creates_status_column_if_missing(self, monkeypatch, tmp_path):
        """Should create a status column with defaults if pool has none."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "", "player": "Alex Sarr", "designation": "Out",
                 "notes": "", "active": "True"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool().drop(columns=["status"])
        result = apply_manual_injury_overrides_to_pool(pool)
        assert "status" in result.columns
        sarr_row = result[result["player_name"] == "Alex Sarr"]
        assert sarr_row["status"].iloc[0] == "OUT"

    def test_name_matching_is_case_insensitive(self, monkeypatch, tmp_path):
        """Player name matching should be case-insensitive."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "", "player": "ALEX SARR", "designation": "Out",
                 "notes": "", "active": "True"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        sarr_row = result[result["player_name"] == "Alex Sarr"]
        assert sarr_row["status"].iloc[0] == "OUT"

    def test_player_id_takes_priority_over_name(self, monkeypatch, tmp_path):
        """When playerID matches, it should be used (not fall through to name)."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [{"playerID": "940541851189", "player": "Cooper McNeeley",
                 "designation": "Out", "notes": "", "active": "True"}]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        # Should be overridden by ID match, not fall to name
        cm_row = result[result["player_id"] == "940541851189"]
        assert cm_row["status"].iloc[0] == "OUT"

    def test_multiple_overrides_applied(self, monkeypatch, tmp_path):
        """Multiple active override rows should all be applied."""
        import yak_core.live as _live
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        rows = [
            {"playerID": "", "player": "Alex Sarr", "designation": "Out", "notes": "", "active": "True"},
            {"playerID": "940541851189", "player": "Cooper McNeeley", "designation": "Out", "notes": "", "active": "True"},
        ]
        _write_overrides_csv(str(cfg_dir), rows)
        monkeypatch.setattr(_live, "YAKOS_ROOT", str(tmp_path))
        pool = _make_pool()
        result = apply_manual_injury_overrides_to_pool(pool)
        assert result[result["player_name"] == "Alex Sarr"]["status"].iloc[0] == "OUT"
        assert result[result["player_id"] == "940541851189"]["status"].iloc[0] == "OUT"
