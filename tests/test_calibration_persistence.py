"""Tests for Calibration Lab persistent config storage."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.calibration_persistence import (
    ACTIVE_CONFIG_PATH,
    CONFIG_HISTORY_PATH,
    OPTIMIZER_OVERRIDES_PATH,
    _SLIDER_KEYS,
    append_config_history,
    apply_config_to_optimizer,
    get_active_slider_values,
    load_active_config,
    load_config_history,
    load_optimizer_overrides,
    reset_active_config,
    save_active_config,
)


@pytest.fixture(autouse=True)
def _clean_config_files(tmp_path, monkeypatch):
    """Redirect config paths to tmp_path so tests don't touch real data."""
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    monkeypatch.setattr("app.calibration_persistence.CALIBRATION_DIR", cal_dir)
    monkeypatch.setattr("app.calibration_persistence.ACTIVE_CONFIG_PATH", cal_dir / "active_config.json")
    monkeypatch.setattr("app.calibration_persistence.CONFIG_HISTORY_PATH", cal_dir / "config_history.json")
    monkeypatch.setattr("app.calibration_persistence.OPTIMIZER_OVERRIDES_PATH", cal_dir / "optimizer_overrides.json")


def _sample_values():
    return {
        "proj_weight": 0.25,
        "upside_weight": 0.35,
        "boom_weight": 0.40,
        "own_penalty_strength": 1.2,
        "low_own_boost": 0.5,
        "own_neutral_pct": 15,
        "max_punt_players": 1,
        "min_mid_players": 4,
        "game_diversity_pct": 65,
        "stud_exposure": 50,
        "mid_exposure": 35,
        "value_exposure": 25,
    }


class TestActiveConfig:
    def test_load_returns_none_when_missing(self):
        assert load_active_config() is None

    def test_save_and_load_roundtrip(self):
        vals = _sample_values()
        saved = save_active_config(vals, slate_date="2026-03-14")
        loaded = load_active_config()
        assert loaded is not None
        # Now returns per-contest-type dict
        gpp = loaded["gpp"]
        assert gpp["name"] == "GPP Working Config"
        assert gpp["values"]["proj_weight"] == 0.25
        assert "2026-03-14" in gpp["slates_trained"]

    def test_save_preserves_existing_slates(self):
        vals = _sample_values()
        save_active_config(vals, slate_date="2026-03-14")
        save_active_config(vals, slate_date="2026-03-15")
        loaded = load_active_config()
        assert loaded["gpp"]["slates_trained"] == ["2026-03-14", "2026-03-15"]

    def test_save_deduplicates_slate_dates(self):
        vals = _sample_values()
        save_active_config(vals, slate_date="2026-03-14")
        save_active_config(vals, slate_date="2026-03-14")
        loaded = load_active_config()
        assert loaded["gpp"]["slates_trained"].count("2026-03-14") == 1

    def test_save_without_slate_date(self):
        vals = _sample_values()
        save_active_config(vals)
        loaded = load_active_config()
        assert loaded["gpp"]["slates_trained"] == []

    def test_only_slider_keys_persisted(self):
        vals = _sample_values()
        vals["extra_garbage"] = 999
        save_active_config(vals)
        loaded = load_active_config()
        assert "extra_garbage" not in loaded["gpp"]["values"]

    def test_get_active_slider_values(self):
        save_active_config(_sample_values())
        vals = get_active_slider_values()
        assert vals is not None
        assert vals["proj_weight"] == 0.25

    def test_get_active_slider_values_none_when_missing(self):
        assert get_active_slider_values() is None

    def test_save_creates_all_contest_types(self):
        vals = _sample_values()
        saved = save_active_config(vals, contest_type="cash")
        assert "gpp" in saved
        assert "cash" in saved
        assert "showdown" in saved
        assert saved["cash"]["values"]["proj_weight"] == 0.25

    def test_contest_types_independent(self):
        vals_gpp = _sample_values()
        vals_cash = _sample_values()
        vals_cash["proj_weight"] = 0.70
        save_active_config(vals_gpp, contest_type="gpp")
        save_active_config(vals_cash, contest_type="cash")
        loaded = load_active_config()
        assert loaded["gpp"]["values"]["proj_weight"] == 0.25
        assert loaded["cash"]["values"]["proj_weight"] == 0.70

    def test_get_active_slider_values_per_contest_type(self):
        vals_gpp = _sample_values()
        vals_cash = _sample_values()
        vals_cash["proj_weight"] = 0.80
        save_active_config(vals_gpp, contest_type="gpp")
        save_active_config(vals_cash, contest_type="cash")
        assert get_active_slider_values(contest_type="gpp")["proj_weight"] == 0.25
        assert get_active_slider_values(contest_type="cash")["proj_weight"] == 0.80


class TestConfigHistory:
    def test_empty_history(self):
        assert load_config_history() == []

    def test_append_and_load(self):
        vals = _sample_values()
        append_config_history("apply_recommendations", vals, slate_date="2026-03-14")
        history = load_config_history()
        assert len(history) == 1
        assert history[0]["action"] == "apply_recommendations"
        assert history[0]["slate_date"] == "2026-03-14"

    def test_tracks_changes(self):
        old = _sample_values()
        new = dict(old)
        new["proj_weight"] = 0.60
        append_config_history("apply_recommendations", new, old_values=old)
        history = load_config_history()
        assert "proj_weight" in history[0]["changes"]
        assert history[0]["changes"]["proj_weight"]["from"] == 0.25
        assert history[0]["changes"]["proj_weight"]["to"] == 0.60

    def test_no_changes_when_identical(self):
        vals = _sample_values()
        append_config_history("save_checkpoint", vals, old_values=vals)
        history = load_config_history()
        assert history[0]["changes"] == {}

    def test_multiple_entries_append(self):
        vals = _sample_values()
        append_config_history("first", vals)
        append_config_history("second", vals)
        history = load_config_history()
        assert len(history) == 2
        assert history[0]["action"] == "first"
        assert history[1]["action"] == "second"

    def test_history_tagged_with_contest_type(self):
        vals = _sample_values()
        append_config_history("test_action", vals, contest_type="showdown")
        history = load_config_history()
        assert history[0]["contest_type"] == "showdown"


class TestResetConfig:
    def test_reset_clears_slates(self):
        vals = _sample_values()
        save_active_config(vals, slate_date="2026-03-14")
        reset_active_config(vals)
        loaded = load_active_config()
        assert loaded["gpp"]["slates_trained"] == []

    def test_reset_records_history(self):
        vals = _sample_values()
        reset_active_config(vals)
        history = load_config_history()
        assert any(e["action"] == "reset_to_defaults" for e in history)

    def test_reset_only_affects_target_contest_type(self):
        vals = _sample_values()
        save_active_config(vals, slate_date="2026-03-14", contest_type="gpp")
        save_active_config(vals, slate_date="2026-03-14", contest_type="cash")
        reset_active_config(vals, contest_type="gpp")
        loaded = load_active_config()
        assert loaded["gpp"]["slates_trained"] == []
        assert "2026-03-14" in loaded["cash"]["slates_trained"]


class TestApplyToOptimizer:
    def test_writes_overrides_file(self):
        vals = _sample_values()
        save_active_config(vals)
        apply_config_to_optimizer(vals)
        overrides = load_optimizer_overrides()
        assert overrides is not None
        assert "GPP_PROJ_WEIGHT" in overrides

    def test_gpp_weights_normalized(self):
        vals = _sample_values()
        # proj=0.50, upside=0.30, boom=0.20, total=1.0
        save_active_config(vals)
        apply_config_to_optimizer(vals)
        overrides = load_optimizer_overrides()
        assert overrides["GPP_PROJ_WEIGHT"] == pytest.approx(0.25)
        assert overrides["GPP_UPSIDE_WEIGHT"] == pytest.approx(0.35)
        assert overrides["GPP_BOOM_WEIGHT"] == pytest.approx(0.40)

    def test_gpp_weights_normalized_non_unit_sum(self):
        vals = _sample_values()
        vals["proj_weight"] = 2.5
        vals["upside_weight"] = 3.5
        vals["boom_weight"] = 4.0
        save_active_config(vals)
        apply_config_to_optimizer(vals)
        overrides = load_optimizer_overrides()
        assert overrides["GPP_PROJ_WEIGHT"] == pytest.approx(0.25)
        assert overrides["GPP_UPSIDE_WEIGHT"] == pytest.approx(0.35)
        assert overrides["GPP_BOOM_WEIGHT"] == pytest.approx(0.40)

    def test_exposure_converted_to_decimals(self):
        vals = _sample_values()
        save_active_config(vals)
        apply_config_to_optimizer(vals)
        overrides = load_optimizer_overrides()
        assert overrides["TIERED_EXPOSURE_STUD"] == pytest.approx(0.50)
        assert overrides["TIERED_EXPOSURE_MID"] == pytest.approx(0.35)
        assert overrides["TIERED_EXPOSURE_VALUE"] == pytest.approx(0.25)

    def test_game_diversity_converted_to_decimal(self):
        vals = _sample_values()
        save_active_config(vals)
        apply_config_to_optimizer(vals)
        overrides = load_optimizer_overrides()
        assert overrides["MAX_GAME_STACK_RATE"] == pytest.approx(0.65)

    def test_records_history(self):
        vals = _sample_values()
        save_active_config(vals)
        apply_config_to_optimizer(vals)
        history = load_config_history()
        assert any(e["action"] == "apply_to_optimizer" for e in history)

    def test_load_overrides_returns_none_when_missing(self):
        assert load_optimizer_overrides() is None

    def test_per_contest_type_overrides_isolated(self):
        vals_gpp = _sample_values()
        vals_cash = _sample_values()
        vals_cash["proj_weight"] = 0.80
        vals_cash["upside_weight"] = 0.10
        vals_cash["boom_weight"] = 0.10
        apply_config_to_optimizer(vals_gpp, contest_type="gpp")
        apply_config_to_optimizer(vals_cash, contest_type="cash")
        gpp_overrides = load_optimizer_overrides(contest_type="gpp")
        cash_overrides = load_optimizer_overrides(contest_type="cash")
        assert gpp_overrides["GPP_PROJ_WEIGHT"] == pytest.approx(0.25)
        assert cash_overrides["GPP_PROJ_WEIGHT"] == pytest.approx(0.80)


class TestApplyCalibrationOverrides:
    def test_no_overrides_returns_cfg_unchanged(self):
        from yak_core.lineups import apply_calibration_overrides
        cfg = {"GPP_PROJ_WEIGHT": 0.25, "NUM_LINEUPS": 20}
        result = apply_calibration_overrides(cfg)
        assert result == cfg

    def test_overrides_merge_into_cfg(self, tmp_path, monkeypatch):
        from yak_core.lineups import apply_calibration_overrides

        vals = _sample_values()
        vals["proj_weight"] = 0.70
        vals["upside_weight"] = 0.20
        vals["boom_weight"] = 0.10
        save_active_config(vals)
        apply_config_to_optimizer(vals)

        cfg = {"GPP_PROJ_WEIGHT": 0.25, "NUM_LINEUPS": 20}
        result = apply_calibration_overrides(cfg)
        assert result["GPP_PROJ_WEIGHT"] == pytest.approx(0.70)
        assert result["NUM_LINEUPS"] == 20  # non-overridden key preserved

    def test_tiered_exposure_reconstructed(self, tmp_path, monkeypatch):
        from yak_core.lineups import apply_calibration_overrides

        vals = _sample_values()
        vals["stud_exposure"] = 60
        vals["mid_exposure"] = 40
        vals["value_exposure"] = 30
        save_active_config(vals)
        apply_config_to_optimizer(vals)

        cfg = {"TIERED_EXPOSURE": [(9000, 0.50), (6000, 0.35), (0, 0.25)]}
        result = apply_calibration_overrides(cfg)
        assert result["TIERED_EXPOSURE"] == [(9000, 0.60), (6000, 0.40), (0, 0.30)]

    def test_original_cfg_not_mutated(self):
        from yak_core.lineups import apply_calibration_overrides

        vals = _sample_values()
        save_active_config(vals)
        apply_config_to_optimizer(vals)

        cfg = {"GPP_PROJ_WEIGHT": 0.25}
        original_val = cfg["GPP_PROJ_WEIGHT"]
        apply_calibration_overrides(cfg)
        assert cfg["GPP_PROJ_WEIGHT"] == original_val


class TestLegacyMigration:
    """Test that legacy flat configs are migrated to per-contest-type format."""

    def test_legacy_active_config_migrated(self, tmp_path, monkeypatch):
        """A flat active_config.json (no contest-type keys) gets migrated on load."""
        from app.calibration_persistence import ACTIVE_CONFIG_PATH as acp
        legacy = {
            "name": "Old Config",
            "created": "2026-01-01T00:00:00",
            "updated": "2026-01-01T00:00:00",
            "slates_trained": ["2026-01-01"],
            "values": {"proj_weight": 0.60},
        }
        acp.write_text(json.dumps(legacy))
        loaded = load_active_config()
        assert "gpp" in loaded
        assert "cash" in loaded
        assert "showdown" in loaded
        assert loaded["gpp"]["values"]["proj_weight"] == 0.60
        assert loaded["gpp"]["slates_trained"] == ["2026-01-01"]


class TestGitHubSyncOnSave:
    """Verify that _sync_to_github is called after every write operation."""

    @patch("app.calibration_persistence._sync_to_github")
    def test_save_active_config_triggers_sync(self, mock_sync):
        save_active_config(_sample_values(), slate_date="2026-03-14")
        mock_sync.assert_called()
        call_files = mock_sync.call_args[1]["files"]
        assert "data/calibration/active_config.json" in call_files

    @patch("app.calibration_persistence._sync_to_github")
    def test_append_config_history_triggers_sync(self, mock_sync):
        append_config_history("test_action", _sample_values())
        mock_sync.assert_called()
        call_files = mock_sync.call_args[1]["files"]
        assert "data/calibration/config_history.json" in call_files

    @patch("app.calibration_persistence._sync_to_github")
    def test_apply_config_to_optimizer_triggers_sync(self, mock_sync):
        apply_config_to_optimizer(_sample_values())
        mock_sync.assert_called()
        # Last call should include the optimizer overrides file
        last_call_files = mock_sync.call_args[1]["files"]
        assert "data/calibration/optimizer_overrides.json" in last_call_files

    @patch("app.calibration_persistence._sync_to_github")
    def test_reset_active_config_triggers_sync(self, mock_sync):
        reset_active_config(_sample_values())
        mock_sync.assert_called()
        # Should have been called multiple times (from append_config_history + reset itself)
        assert mock_sync.call_count >= 2
