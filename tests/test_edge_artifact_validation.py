"""Tests for edge artifact validation and cleanup helpers in yak_core/edge_integrity.py.

Validates that:
- validate_edge_artifacts() correctly detects date mismatches and phantom players.
- clean_stale_edge_artifacts() removes the expected files.
- Valid/absent artifacts pass without error.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from yak_core.edge_integrity import validate_edge_artifacts, clean_stale_edge_artifacts


# ---------------------------------------------------------------------------
# validate_edge_artifacts
# ---------------------------------------------------------------------------

class TestValidateEdgeArtifacts:

    def _write_pool(self, tmp_path: Path, players: list) -> None:
        df = pd.DataFrame({"player_name": players, "salary": [5000] * len(players)})
        df.to_parquet(tmp_path / "slate_pool.parquet", index=False)

    def _write_meta(self, tmp_path: Path, date: str) -> None:
        (tmp_path / "slate_meta.json").write_text(json.dumps({"date": date}))

    def _write_edge_state(self, tmp_path: Path, date: str) -> None:
        (tmp_path / "edge_state.json").write_text(json.dumps({"sport": "NBA", "date": date}))

    def _write_edge_analysis(self, tmp_path: Path, core: list, leverage: list = None,
                              value: list = None, fade: list = None) -> None:
        ea = {
            "core_plays": [{"player_name": p} for p in core],
            "leverage_plays": [{"player_name": p} for p in (leverage or [])],
            "value_plays": [{"player_name": p} for p in (value or [])],
            "fade_candidates": [{"player_name": p} for p in (fade or [])],
        }
        (tmp_path / "edge_analysis.json").write_text(json.dumps(ea))

    def test_missing_meta_returns_valid(self, tmp_path):
        """No slate_meta.json → valid (nothing to compare against)."""
        result = validate_edge_artifacts(tmp_path)
        assert result["valid"] is True

    def test_no_edge_files_returns_valid(self, tmp_path):
        """slate_meta.json present but no edge files → valid."""
        self._write_meta(tmp_path, "2026-04-14")
        result = validate_edge_artifacts(tmp_path)
        assert result["valid"] is True
        assert result["meta_date"] == "2026-04-14"

    def test_matching_dates_returns_valid(self, tmp_path):
        """edge_state.json date matches meta date → valid."""
        self._write_meta(tmp_path, "2026-04-14")
        self._write_edge_state(tmp_path, "2026-04-14")
        self._write_pool(tmp_path, ["Player A", "Player B"])
        self._write_edge_analysis(tmp_path, core=["Player A"])
        result = validate_edge_artifacts(tmp_path)
        assert result["valid"] is True
        assert result["phantom_players"] == []

    def test_date_mismatch_detected(self, tmp_path):
        """edge_state.json with a different date → not valid."""
        self._write_meta(tmp_path, "2026-04-14")
        self._write_edge_state(tmp_path, "2026-04-13")
        result = validate_edge_artifacts(tmp_path)
        assert result["valid"] is False
        assert "2026-04-13" in result["reason"]
        assert "2026-04-14" in result["reason"]
        assert result["meta_date"] == "2026-04-14"
        assert result["edge_date"] == "2026-04-13"

    def test_phantom_player_detected(self, tmp_path):
        """Player in edge_analysis not in pool → not valid."""
        self._write_meta(tmp_path, "2026-04-14")
        self._write_edge_state(tmp_path, "2026-04-14")
        self._write_pool(tmp_path, ["Player A", "Player B"])
        self._write_edge_analysis(tmp_path, core=["Player A", "LeBron James"])
        result = validate_edge_artifacts(tmp_path)
        assert result["valid"] is False
        assert "LeBron James" in result["phantom_players"]

    def test_no_phantoms_when_all_in_pool(self, tmp_path):
        """All edge players present in pool → valid."""
        self._write_meta(tmp_path, "2026-04-14")
        self._write_edge_state(tmp_path, "2026-04-14")
        self._write_pool(tmp_path, ["Alpha", "Beta", "Gamma"])
        self._write_edge_analysis(tmp_path, core=["Alpha"], value=["Beta"], fade=["Gamma"])
        result = validate_edge_artifacts(tmp_path)
        assert result["valid"] is True
        assert result["phantom_players"] == []

    def test_phantom_across_multiple_sections(self, tmp_path):
        """Phantom players in leverage/value/fade sections are also detected."""
        self._write_meta(tmp_path, "2026-04-14")
        self._write_edge_state(tmp_path, "2026-04-14")
        self._write_pool(tmp_path, ["Alpha"])
        self._write_edge_analysis(
            tmp_path,
            core=["Alpha"],
            leverage=["Ghost1"],
            value=["Ghost2"],
            fade=["Ghost3"],
        )
        result = validate_edge_artifacts(tmp_path)
        assert result["valid"] is False
        for name in ["Ghost1", "Ghost2", "Ghost3"]:
            assert name in result["phantom_players"]

    def test_date_mismatch_takes_priority_over_phantom_check(self, tmp_path):
        """When date mismatches, we report the date error first (no phantom check)."""
        self._write_meta(tmp_path, "2026-04-14")
        self._write_edge_state(tmp_path, "2026-04-13")
        self._write_pool(tmp_path, ["Player A"])
        self._write_edge_analysis(tmp_path, core=["Phantom"])
        result = validate_edge_artifacts(tmp_path)
        # Date mismatch detected; phantom check is not reached
        assert result["valid"] is False
        assert "2026-04-13" in result["reason"]
        assert "phantom" not in result["reason"].lower()


# ---------------------------------------------------------------------------
# clean_stale_edge_artifacts
# ---------------------------------------------------------------------------

class TestCleanStaleEdgeArtifacts:

    def test_removes_all_edge_files(self, tmp_path):
        """All three edge files are removed when present."""
        for fname in ("edge_state.json", "edge_analysis.json", "signals.parquet"):
            (tmp_path / fname).write_text("{}")
        removed = clean_stale_edge_artifacts(tmp_path)
        assert set(removed) == {"edge_state.json", "edge_analysis.json", "signals.parquet"}
        for fname in removed:
            assert not (tmp_path / fname).exists()

    def test_partial_removal_when_some_missing(self, tmp_path):
        """Only existing files are removed; missing files are silently skipped."""
        (tmp_path / "edge_state.json").write_text("{}")
        # edge_analysis.json and signals.parquet intentionally absent
        removed = clean_stale_edge_artifacts(tmp_path)
        assert removed == ["edge_state.json"]
        assert not (tmp_path / "edge_state.json").exists()

    def test_no_error_when_directory_empty(self, tmp_path):
        """No error when no edge files are present."""
        removed = clean_stale_edge_artifacts(tmp_path)
        assert removed == []

    def test_non_edge_files_are_not_removed(self, tmp_path):
        """slate_pool.parquet and slate_meta.json are untouched."""
        (tmp_path / "slate_pool.parquet").write_bytes(b"data")
        (tmp_path / "slate_meta.json").write_text("{}")
        (tmp_path / "edge_state.json").write_text("{}")
        clean_stale_edge_artifacts(tmp_path)
        assert (tmp_path / "slate_pool.parquet").exists()
        assert (tmp_path / "slate_meta.json").exists()

