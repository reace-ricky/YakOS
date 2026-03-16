"""Test: RG merge before cascade preserves injury bump projections.

Validates the pipeline ordering fix where _merge_rg_csv() is called BEFORE
apply_injury_cascade() inside _load_nba_pool(), so cascade bumps on top of
RG FPTS are not overwritten by a post-hoc RG merge.
"""
import importlib.util
import io
import os
import sys
import types

# ── Path setup ──
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Mock streamlit before importing app modules ──
_st = types.ModuleType("streamlit")
for attr in ("info", "error", "warning", "markdown", "success",
             "dataframe", "data_editor"):
    setattr(_st, attr, lambda *a, **kw: None)
_st.text_input = lambda *a, **kw: ""
_st.columns = lambda *a, **kw: [_st, _st]
_st.file_uploader = lambda *a, **kw: None
_st.button = lambda *a, **kw: False
_st.secrets = {}
_st.session_state = {}
_ctx = type("ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})
_st.spinner = lambda *a, **kw: _ctx()
_st.expander = lambda *a, **kw: _ctx()
_cc = types.ModuleType("streamlit.column_config")
_cc.CheckboxColumn = lambda *a, **kw: None
_st.column_config = _cc
sys.modules["streamlit"] = _st
sys.modules["streamlit.column_config"] = _cc

import pandas as pd
import pytest

from yak_core.injury_cascade import apply_injury_cascade

# ── Load _merge_rg_csv from app/lab_tab.py ──
_spec = importlib.util.spec_from_file_location(
    "app_lab_tab", os.path.join(_REPO_ROOT, "app", "lab_tab.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["app_lab_tab"] = _mod
_spec.loader.exec_module(_mod)
_merge_rg_csv = _mod._merge_rg_csv


# ── Fixtures ──

def _make_mock_pool():
    """Build a minimal pool DataFrame with one OUT player and two teammates."""
    return pd.DataFrame([
        {
            "player_name": "Star Player",
            "team": "NYK",
            "pos": "PG",
            "salary": 9000,
            "proj": 40.0,
            "proj_minutes": 34.0,
            "status": "OUT",
        },
        {
            "player_name": "Backup Guard",
            "team": "NYK",
            "pos": "PG",
            "salary": 4000,
            "proj": 18.0,
            "proj_minutes": 20.0,
            "status": "Active",
        },
        {
            "player_name": "Wing Player",
            "team": "NYK",
            "pos": "SG",
            "salary": 5500,
            "proj": 22.0,
            "proj_minutes": 25.0,
            "status": "Active",
        },
        {
            "player_name": "Opponent Star",
            "team": "BOS",
            "pos": "SF",
            "salary": 8500,
            "proj": 38.0,
            "proj_minutes": 33.0,
            "status": "Active",
        },
    ])


def _make_rg_csv():
    """Build a mock RG CSV with base FPTS projections."""
    csv_content = "PLAYER,FPTS,FLOOR,CEIL,POWN\n"
    csv_content += "Star Player,40.0,30.0,50.0,5%\n"
    csv_content += "Backup Guard,18.0,12.0,25.0,10%\n"
    csv_content += "Wing Player,22.0,16.0,30.0,8%\n"
    csv_content += "Opponent Star,38.0,28.0,48.0,15%\n"
    return io.StringIO(csv_content)


# ── Tests ──

def test_correct_order_cascade_survives():
    """RG merge THEN cascade -> bumped players have proj > raw RG FPTS."""
    pool = _make_mock_pool()

    # Step 1: RG merge sets base projections
    pool_merged = _merge_rg_csv(pool.copy(), _make_rg_csv())

    backup = pool_merged[pool_merged["player_name"] == "Backup Guard"].iloc[0]
    assert backup["proj"] == 18.0, "RG FPTS should set proj to 18.0"
    assert backup["rg_proj"] == 18.0

    # Step 2: Cascade runs on top of RG projections
    pool_cascaded, report = apply_injury_cascade(pool_merged)

    backup_final = pool_cascaded[pool_cascaded["player_name"] == "Backup Guard"].iloc[0]
    bump = backup_final.get("injury_bump_fp", 0)

    assert bump > 0, (
        f"Backup Guard should receive a cascade bump, got injury_bump_fp={bump}"
    )
    assert backup_final["proj"] > 18.0, (
        f"Backup Guard final proj ({backup_final['proj']}) should exceed raw RG FPTS (18.0) "
        f"due to cascade bump of {bump}"
    )


def test_wrong_order_cascade_overwritten():
    """Cascade THEN RG merge -> bumps are destroyed (documents the old bug)."""
    pool = _make_mock_pool()

    # Step 1: Cascade bumps the backup
    pool_cascaded, _ = apply_injury_cascade(pool.copy())

    backup_after = pool_cascaded[pool_cascaded["player_name"] == "Backup Guard"].iloc[0]
    bump = backup_after.get("injury_bump_fp", 0)
    assert bump > 0, "Cascade should bump Backup Guard"
    assert backup_after["proj"] > 18.0, "Cascaded proj should exceed base"

    # Step 2: RG merge OVERWRITES proj with raw RG FPTS -> bump destroyed
    pool_overwritten = _merge_rg_csv(pool_cascaded, _make_rg_csv())

    backup_ow = pool_overwritten[pool_overwritten["player_name"] == "Backup Guard"].iloc[0]
    assert backup_ow["proj"] == 18.0, (
        f"RG merge should overwrite cascade-adjusted proj back to 18.0, "
        f"got {backup_ow['proj']}"
    )


def test_injury_bump_fp_column_present():
    """After the correct pipeline, injury_bump_fp column shows bump amounts."""
    pool = _make_mock_pool()
    pool_merged = _merge_rg_csv(pool.copy(), _make_rg_csv())
    pool_cascaded, _ = apply_injury_cascade(pool_merged)

    assert "injury_bump_fp" in pool_cascaded.columns, "injury_bump_fp column should exist"

    bumps = pool_cascaded[pool_cascaded["injury_bump_fp"] > 0]
    assert len(bumps) > 0, "At least one player should have a positive injury_bump_fp"

    for _, row in bumps.iterrows():
        expected = row["original_proj"] + row["injury_bump_fp"]
        assert row["proj"] == pytest.approx(expected, abs=0.01), (
            f"{row['player_name']}: proj ({row['proj']}) should equal "
            f"original_proj ({row['original_proj']}) + bump ({row['injury_bump_fp']})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
