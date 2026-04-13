"""Tests for app.data_loader.get_published_version.

Validates that the version string changes when a tracked published file is
updated, and stays stable when nothing changes.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


def test_missing_directory_returns_empty_string(tmp_path, monkeypatch):
    """get_published_version returns '' when the sport directory does not exist."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    assert dl.get_published_version("nba") == ""


def test_stable_when_files_unchanged(tmp_path, monkeypatch):
    """Version is stable across two calls with no file changes."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()
    (sport_dir / "slate_meta.json").write_text("{}")

    v1 = dl.get_published_version("nba")
    v2 = dl.get_published_version("nba")
    assert v1 == v2


def test_version_changes_when_file_updated(tmp_path, monkeypatch):
    """Version changes after a tracked file is re-written."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()
    meta = sport_dir / "slate_meta.json"
    meta.write_text("{}")

    v1 = dl.get_published_version("nba")

    # Force a different mtime by nudging the modification time by 1 second.
    current = meta.stat().st_mtime
    os.utime(meta, (current + 1, current + 1))

    v2 = dl.get_published_version("nba")
    assert v1 != v2


def test_version_changes_when_lineup_file_added(tmp_path, monkeypatch):
    """Version changes when a new *_lineups.parquet file appears."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    sport_dir = tmp_path / "nba"
    sport_dir.mkdir()

    v1 = dl.get_published_version("nba")

    (sport_dir / "classic_gpp_main_lineups.parquet").write_bytes(b"")
    v2 = dl.get_published_version("nba")

    assert v1 != v2


def test_independent_across_sports(tmp_path, monkeypatch):
    """Versions for different sports are independent."""
    import app.data_loader as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    (tmp_path / "nba").mkdir()
    (tmp_path / "pga").mkdir()
    (tmp_path / "nba" / "slate_meta.json").write_text("{}")
    (tmp_path / "pga" / "slate_meta.json").write_text("{}")

    v_nba = dl.get_published_version("nba")
    v_pga = dl.get_published_version("pga")

    # Updating NBA file must not affect PGA version
    meta_nba = tmp_path / "nba" / "slate_meta.json"
    current = meta_nba.stat().st_mtime
    os.utime(meta_nba, (current + 1, current + 1))

    assert dl.get_published_version("nba") != v_nba
    assert dl.get_published_version("pga") == v_pga
