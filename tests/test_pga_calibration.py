"""Tests for sport-keyed calibration feedback and PGA calibration helpers."""

import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nba_pool() -> pd.DataFrame:
    """Minimal NBA pool for calibration."""
    return pd.DataFrame({
        "player_name": ["Alice", "Bob", "Carol"],
        "pos": ["PG", "SG", "C"],
        "salary": [7000, 5500, 8500],
        "proj": [30.0, 22.0, 38.0],
        "actual_fp": [35.0, 18.0, 42.0],
    })


def _make_pga_pool() -> pd.DataFrame:
    """Minimal PGA pool for calibration."""
    return pd.DataFrame({
        "player_name": ["Scheffler", "McIlroy", "Hovland", "Clark"],
        "pos": ["G", "G", "G", "G"],
        "salary": [11000, 10000, 8500, 7000],
        "proj": [90.0, 85.0, 75.0, 65.0],
        "actual_fp": [100.0, 70.0, 80.0, 55.0],
    })


@pytest.fixture
def patched_feedback_dir(tmp_path, monkeypatch):
    """Patch both YAKOS_ROOT and _FEEDBACK_DIR so all paths point to tmp_path."""
    feedback_dir = str(tmp_path / "data" / "calibration_feedback")
    monkeypatch.setattr("yak_core.calibration_feedback.YAKOS_ROOT", str(tmp_path))
    monkeypatch.setattr("yak_core.calibration_feedback._FEEDBACK_DIR", feedback_dir)
    return tmp_path


# ---------------------------------------------------------------------------
# Sport-keyed storage isolation
# ---------------------------------------------------------------------------

class TestSportKeyedStorage:
    """Verify NBA and PGA calibration data are stored separately."""

    def test_record_nba_does_not_affect_pga(self, patched_feedback_dir):
        from yak_core.calibration_feedback import (
            record_slate_errors,
            get_correction_factors,
        )

        nba_pool = _make_nba_pool()
        record_slate_errors("2026-03-01", nba_pool, sport="NBA")

        nba_cf = get_correction_factors(sport="NBA")
        pga_cf = get_correction_factors(sport="PGA")

        assert nba_cf.get("n_slates", 0) > 0
        assert pga_cf.get("n_slates", 0) == 0

    def test_record_pga_does_not_affect_nba(self, patched_feedback_dir):
        from yak_core.calibration_feedback import (
            record_slate_errors,
            get_correction_factors,
        )

        pga_pool = _make_pga_pool()
        record_slate_errors("2026-03-01", pga_pool, sport="PGA")

        pga_cf = get_correction_factors(sport="PGA")
        nba_cf = get_correction_factors(sport="NBA")

        assert pga_cf.get("n_slates", 0) > 0
        assert nba_cf.get("n_slates", 0) == 0

    def test_sport_keyed_file_paths(self, patched_feedback_dir):
        tmp_path = patched_feedback_dir
        from yak_core.calibration_feedback import record_slate_errors

        record_slate_errors("2026-03-01", _make_nba_pool(), sport="NBA")
        record_slate_errors("2026-03-01", _make_pga_pool(), sport="PGA")

        nba_dir = tmp_path / "data" / "calibration_feedback" / "nba"
        pga_dir = tmp_path / "data" / "calibration_feedback" / "pga"

        assert (nba_dir / "slate_errors.json").exists()
        assert (pga_dir / "slate_errors.json").exists()


# ---------------------------------------------------------------------------
# PGA salary bins
# ---------------------------------------------------------------------------

class TestPGASalaryBins:
    """Verify PGA uses golf-appropriate salary bins."""

    def test_pga_bins_are_different_from_nba(self):
        from yak_core.calibration_feedback import _get_salary_config

        nba_bins, nba_labels = _get_salary_config("NBA")
        pga_bins, pga_labels = _get_salary_config("PGA")

        assert nba_bins != pga_bins
        assert nba_labels != pga_labels

    def test_pga_bins_cover_golf_salary_range(self):
        from yak_core.calibration_feedback import _get_salary_config

        pga_bins, pga_labels = _get_salary_config("PGA")

        # PGA salaries typically $6K-$12K
        assert pga_bins[0] == 0
        assert pga_bins[-1] >= 10500
        assert len(pga_labels) == len(pga_bins) - 1

    def test_pga_corrections_use_pga_bins(self, patched_feedback_dir):
        from yak_core.calibration_feedback import (
            record_slate_errors,
            get_correction_factors,
        )

        pool = _make_pga_pool()
        record_slate_errors("2026-03-01", pool, sport="PGA")

        cf = get_correction_factors(sport="PGA")
        # Should have salary_bin corrections keyed on PGA bin labels
        sal_corr = cf.get("by_salary_bin", {})
        if sal_corr:
            # At least one PGA bin label should appear
            pga_bin_labels = {"<6.5K", "6.5-7.5K", "7.5-8.5K", "8.5-9.5K", "9.5-10.5K", "10.5K+"}
            assert any(k in pga_bin_labels for k in sal_corr.keys())


# ---------------------------------------------------------------------------
# PGA valid positions
# ---------------------------------------------------------------------------

class TestPGAPositions:
    """Verify PGA uses 'G' as the only valid position."""

    def test_pga_valid_positions(self):
        from yak_core.calibration_feedback import _get_valid_positions

        pga_positions = _get_valid_positions("PGA")
        assert "G" in pga_positions

    def test_nba_valid_positions_are_different(self):
        from yak_core.calibration_feedback import _get_valid_positions

        nba_positions = _get_valid_positions("NBA")
        pga_positions = _get_valid_positions("PGA")
        assert nba_positions != pga_positions
        assert "PG" in nba_positions


# ---------------------------------------------------------------------------
# Migration of legacy flat files
# ---------------------------------------------------------------------------

class TestLegacyMigration:
    """Verify old flat-file calibration data migrates to nba/ subdirectory."""

    def test_migrate_moves_flat_files(self, patched_feedback_dir):
        tmp_path = patched_feedback_dir
        from yak_core.calibration_feedback import _migrate_legacy_files

        # Create legacy flat files
        cal_dir = tmp_path / "data" / "calibration_feedback"
        cal_dir.mkdir(parents=True, exist_ok=True)
        legacy_errors = {"slates": {"2026-01-01": {"overall": {"mae": 5.0}}}}
        legacy_corrections = {"n_slates": 1}

        (cal_dir / "slate_errors.json").write_text(json.dumps(legacy_errors))
        (cal_dir / "correction_factors.json").write_text(json.dumps(legacy_corrections))

        _migrate_legacy_files()

        # Legacy files should be gone, nba/ subdir should have them
        assert not (cal_dir / "slate_errors.json").exists()
        assert not (cal_dir / "correction_factors.json").exists()
        assert (cal_dir / "nba" / "slate_errors.json").exists()
        assert (cal_dir / "nba" / "correction_factors.json").exists()

        # Content should be preserved
        migrated = json.loads((cal_dir / "nba" / "slate_errors.json").read_text())
        assert migrated == legacy_errors

    def test_migrate_no_op_if_no_legacy_files(self, patched_feedback_dir):
        from yak_core.calibration_feedback import _migrate_legacy_files

        # Should not raise even if dir doesn't exist
        _migrate_legacy_files()


# ---------------------------------------------------------------------------
# Apply corrections with sport param
# ---------------------------------------------------------------------------

class TestApplyCorrections:
    """Verify apply_corrections respects the sport parameter."""

    def test_apply_pga_corrections(self, patched_feedback_dir):
        from yak_core.calibration_feedback import (
            record_slate_errors,
            get_correction_factors,
            apply_corrections,
        )

        pool = _make_pga_pool()
        record_slate_errors("2026-03-01", pool, sport="PGA")
        record_slate_errors("2026-03-02", pool, sport="PGA")

        cf = get_correction_factors(sport="PGA")
        corrected = apply_corrections(pool.copy(), cf, sport="PGA")

        # Should have proj column modified (or at least not crash)
        assert "proj" in corrected.columns
        assert len(corrected) == len(pool)

    def test_nba_corrections_not_applied_to_pga(self, patched_feedback_dir):
        from yak_core.calibration_feedback import (
            record_slate_errors,
            get_correction_factors,
        )

        # Record NBA data only
        nba_pool = _make_nba_pool()
        record_slate_errors("2026-03-01", nba_pool, sport="NBA")

        # PGA should have no corrections
        pga_cf = get_correction_factors(sport="PGA")
        assert pga_cf.get("n_slates", 0) == 0


# ---------------------------------------------------------------------------
# PGA calibration helpers
# ---------------------------------------------------------------------------

class TestPGACalibrationHelpers:
    """Test pga_calibration module helper functions."""

    def test_fetch_pga_actuals_maps_columns(self):
        """fetch_pga_actuals should map total_pts -> actual_fp and add pos=G."""
        from unittest.mock import MagicMock

        mock_dg = MagicMock()
        mock_dg.get_historical_dfs_points.return_value = pd.DataFrame({
            "player_name": ["Scheffler", "McIlroy"],
            "dg_id": [1, 2],
            "salary": [11000, 10000],
            "total_pts": [100.0, 85.0],
            "ownership": [15.0, 12.0],
            "fin_text": ["1", "5"],
        })

        from yak_core.pga_calibration import fetch_pga_actuals

        result = fetch_pga_actuals(mock_dg, event_id=9, year=2026)

        assert "actual_fp" in result.columns
        assert "pos" in result.columns
        assert (result["pos"] == "G").all()
        assert result["actual_fp"].iloc[0] == 100.0

    def test_get_pga_event_list_filters_to_pga(self):
        """get_pga_event_list should filter to PGA events with DK salaries."""
        from unittest.mock import MagicMock

        mock_dg = MagicMock()
        mock_dg.get_dfs_event_list.return_value = pd.DataFrame({
            "event_id": [1, 2, 3],
            "event_name": ["PGA Event", "Euro Event", "PGA NoSalary"],
            "tour": ["pga", "euro", "pga"],
            "dk_salaries": ["yes", "yes", "no"],
            "date": ["2026-03-01", "2026-03-01", "2026-03-01"],
            "calendar_year": [2026, 2026, 2026],
        })

        from yak_core.pga_calibration import get_pga_event_list

        result = get_pga_event_list(mock_dg)

        assert len(result) == 1
        assert result.iloc[0]["event_id"] == 1

    def test_get_pga_event_list_empty_input(self):
        """get_pga_event_list handles empty event list gracefully."""
        from unittest.mock import MagicMock

        mock_dg = MagicMock()
        mock_dg.get_dfs_event_list.return_value = pd.DataFrame()

        from yak_core.pga_calibration import get_pga_event_list

        result = get_pga_event_list(mock_dg)
        assert result.empty
