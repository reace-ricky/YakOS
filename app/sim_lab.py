"""Sim Lab v2 — Config Tuning + Batch Replay.

Pick a contest preset, adjust knobs across 4 tuning groups,
batch-run all available RG archive dates, and compare configs
via trend chart + summary table.  Batch history persists to
data/sim_lab/batch_history.parquet and syncs to GitHub.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from yak_core.config import (
    CONTEST_PRESETS,
    DEFAULT_CONFIG,
    NAMED_PROFILES,
    merge_config,
)
from utils.constants import (
    NBA_GAME_STYLES, NBA_CONTEST_TYPES_BY_STYLE, CONTEST_PROFILE_KEY_MAP,
    PROFILE_KEY_TO_PRESET, PROFILE_KEY_TO_NAMED,
    PGA_CONTEST_TYPES, PGA_DISPLAY_TO_PRESET,
)
from yak_core.edge import compute_edge_metrics
from yak_core.lineups import (
    build_multiple_lineups_with_exposure,
    build_showdown_lineups,
    prepare_pool,
)
from yak_core.live import fetch_actuals_from_api, fetch_live_dfs
from yak_core.ricky_rank import (
    RICKY_W_CEIL,
    RICKY_W_GPP,
    RICKY_W_OWN,
    rank_lineups_for_se,
)
from yak_core.sim_lab_report import summarize_sim_lab

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NBA_PRESETS = ["GPP Main", "GPP Early", "GPP Late", "Showdown", "Cash Main", "Cash Game"]
_PGA_PRESETS = ["PGA GPP", "PGA Cash", "PGA Showdown"]

# ---------------------------------------------------------------------------
# NBA GPP Archetypes — Sim Lab only, never touches main optimizer config.
#
# Each archetype is a named set of slider overrides applied on top of the
# selected contest preset.  "Default" uses the preset as-is (no overrides).
# Only active for NBA GPP presets (GPP Main, GPP Early, GPP Late).
# ---------------------------------------------------------------------------

_NBA_GPP_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "Default": {
        "description": "Current GPP preset — no overrides",
        "overrides": {},
    },
    "Stars & Scrubs Ceiling": {
        "description": "Heavy stud concentration + max ceiling. "
                       "High boom weight, aggressive ownership penalty, "
                       "fewer mid-range players.",
        "overrides": {
            "GPP_PROJ_WEIGHT": 0.20,
            "GPP_UPSIDE_WEIGHT": 0.35,
            "GPP_BOOM_WEIGHT": 0.45,
            "GPP_OWN_PENALTY_STRENGTH": 1.4,
            "GPP_SMASH_WEIGHT": 0.20,
            "GPP_LEVERAGE_WEIGHT": 0.10,
            "GPP_BUST_PENALTY": 0.05,
            "NUM_LINEUPS": 20,
            "MAX_EXPOSURE": 0.50,
        },
    },
    "Balanced Leverage": {
        "description": "Even weight split + strong ownership fade. "
                       "Targets underowned edges across all salary tiers.",
        "overrides": {
            "GPP_PROJ_WEIGHT": 0.35,
            "GPP_UPSIDE_WEIGHT": 0.30,
            "GPP_BOOM_WEIGHT": 0.25,
            "GPP_OWN_PENALTY_STRENGTH": 1.6,
            "GPP_LEVERAGE_WEIGHT": 0.15,
            "GPP_SMASH_WEIGHT": 0.10,
            "GPP_BUST_PENALTY": 0.12,
            "NUM_LINEUPS": 20,
            "MAX_EXPOSURE": 0.45,
        },
    },
}

_NBA_GPP_ARCHETYPE_NAMES: List[str] = list(_NBA_GPP_ARCHETYPES.keys())

# Presets that support archetypes (GPP-family only)
_ARCHETYPE_ELIGIBLE_PRESETS = {"GPP Main", "GPP Early", "GPP Late"}

_BATCH_COLORS = ["#4dabf7", "#00ff87", "#ffa726", "#ef5350", "#ab47bc", "#26c6da", "#d4e157", "#ff7043"]

_HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "sim_lab"
_HISTORY_FILE = _HISTORY_DIR / "batch_history.parquet"
_HISTORY_REL_PATH = "data/sim_lab/batch_history.parquet"
# Baselines are tracked inside batch_history.parquet via is_baseline flag.
# No separate baselines file needed — single source of truth.

_logger = logging.getLogger(__name__)


def _get_secret(key: str) -> str:
    """Read a secret from Streamlit secrets or env, empty string on miss."""
    import os
    try:
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, "")


def _get_nba_api_key() -> str:
    return _get_secret("TANK01_RAPIDAPI_KEY") or _get_secret("RAPIDAPI_KEY")


def _get_pga_api_key() -> str:
    return _get_secret("DATAGOLF_API_KEY")


def _sandbox_config_key(preset: str) -> str:
    return f"sim_lab_config_{preset}"


def _get_sandbox_overrides(preset: str) -> Dict[str, Any]:
    return dict(st.session_state.get(_sandbox_config_key(preset), {}))


def _config_hash(overrides: Dict[str, Any], archetype: str = "Default") -> str:
    """Deterministic hash of slider overrides + archetype name."""
    payload = {"archetype": archetype, **overrides}
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]



# ---------------------------------------------------------------------------
# Slider / weight persistence (survives page refreshes + cold starts)
# ---------------------------------------------------------------------------

_SLIDER_STATE_FILE = _HISTORY_DIR / "slider_state.json"
_SLIDER_STATE_REL = str(Path("data/sim_lab/slider_state.json"))


def _save_slider_state(preset_name: str, overrides: Dict[str, Any],
                       ricky_weights: Dict[str, float]) -> None:
    """Persist current slider overrides + Ricky weights to disk + GitHub."""
    try:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        state: dict = {}
        if _SLIDER_STATE_FILE.is_file():
            state = json.loads(_SLIDER_STATE_FILE.read_text())
        state.setdefault("configs", {})[preset_name] = {
            "overrides": {k: v for k, v in overrides.items()},
            "ricky_weights": {k: v for k, v in ricky_weights.items()},
        }
        _SLIDER_STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

        try:
            from yak_core.github_persistence import sync_feedback_async
            sync_feedback_async(
                files=[_SLIDER_STATE_REL],
                commit_message="Auto-sync Sim Lab slider state",
            )
        except Exception:
            pass
    except Exception as exc:
        _logger.warning("Failed to save slider state: %s", exc)


def _load_slider_state(preset_name: str) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Load persisted slider overrides + Ricky weights. Returns ({}, {}) on miss."""
    try:
        if _SLIDER_STATE_FILE.is_file():
            state = json.loads(_SLIDER_STATE_FILE.read_text())
            entry = state.get("configs", {}).get(preset_name, {})
            return (
                entry.get("overrides", {}),
                entry.get("ricky_weights", {}),
            )
    except Exception:
        pass
    return {}, {}


# ---------------------------------------------------------------------------
# Batch history persistence
# ---------------------------------------------------------------------------

def _append_batch_history(
    batch: Dict[str, Any],
    *,
    is_baseline: bool = False,
) -> None:
    """Append a batch summary row to the persistent history file.

    Writes to data/sim_lab/batch_history.parquet and syncs to GitHub
    so the history survives Streamlit Cloud restarts.

    Parameters
    ----------
    is_baseline : bool
        If *True*, this row is stored as the active baseline for its preset.
        Previous baselines for the same preset are kept (``is_baseline`` flipped
        to *False*) so long-term trends are never lost.
    """
    successful_runs = batch.get("runs", [])
    best_slate_fp = 0.0
    worst_slate_fp = 0.0
    if successful_runs:
        best_run = max(successful_runs, key=lambda r: r["avg_actual"])
        worst_run = min(successful_runs, key=lambda r: r["avg_actual"])
        best_slate_fp = best_run["avg_actual"]
        worst_slate_fp = worst_run["avg_actual"]

    row = {
        "timestamp": datetime.now().isoformat(),
        "sport": batch.get("runs", [{}])[0].get("sport", "NBA") if successful_runs else "NBA",
        "preset": batch.get("preset", ""),
        "archetype": batch.get("archetype", "Default"),
        "config_hash": batch.get("config_hash", ""),
        "config_label": batch.get("config_label", ""),
        "profile_name": batch.get("profile_name", ""),
        "overrides_json": json.dumps(
            batch.get("overrides", {}), sort_keys=True, default=str,
        ),
        "num_dates": len(successful_runs) + len(batch.get("errors", [])),
        "num_lineups": successful_runs[0].get("num_lineups", 0) if successful_runs else 0,
        "avg_actual": batch.get("avg_actual", 0.0),
        "avg_proj": batch.get("avg_proj", 0.0),
        "best_slate": best_slate_fp,
        "worst_slate": worst_slate_fp,
        "beat_proj_pct": batch.get("beat_proj_pct", 0.0),
        "errors": len(batch.get("errors", [])),
        "is_baseline": is_baseline,
        "removed": False,
    }

    new_df = pd.DataFrame([row])

    try:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        if _HISTORY_FILE.is_file():
            existing = pd.read_parquet(_HISTORY_FILE)
            # Ensure new columns exist on legacy data
            for col, default in [("is_baseline", False), ("removed", False),
                                  ("config_label", ""), ("overrides_json", "{}"),
                                  ("profile_name", "")]:
                if col not in existing.columns:
                    existing[col] = default
            # If promoting a new baseline, demote the old one for same preset
            if is_baseline:
                mask = (
                    (existing["preset"] == batch.get("preset", ""))
                    & (existing["is_baseline"] == True)  # noqa: E712
                )
                existing.loc[mask, "is_baseline"] = False
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_parquet(_HISTORY_FILE, index=False)
        _logger.info("Batch history appended: %s rows total", len(combined))

        # Sync to GitHub so it survives cold starts
        try:
            from yak_core.github_persistence import sync_feedback_async
            sync_feedback_async(
                files=[_HISTORY_REL_PATH],
                commit_message="Auto-sync Sim Lab batch history",
            )
        except Exception as sync_err:
            _logger.warning("GitHub sync failed for batch history: %s", sync_err)

    except Exception as exc:
        _logger.warning("Failed to persist batch history: %s", exc)


def _load_batch_history() -> pd.DataFrame:
    """Load batch history from parquet. Returns empty DataFrame if missing."""
    if _HISTORY_FILE.is_file():
        try:
            return pd.read_parquet(_HISTORY_FILE)
        except Exception as exc:
            _logger.warning("Failed to read batch history: %s", exc)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Baseline helpers  (stored inside batch_history.parquet via is_baseline flag)
# ---------------------------------------------------------------------------

def _promote_baseline(row_timestamp: str, preset_name: str) -> None:
    """Promote a batch_history row to active baseline for its preset.

    Demotes any previous baseline (keeps the row, just flips the flag)
    so long-term trends are preserved.
    """
    if not _HISTORY_FILE.is_file():
        return
    try:
        df = pd.read_parquet(_HISTORY_FILE)
        for col, default in [("is_baseline", False), ("removed", False),
                              ("config_label", ""), ("overrides_json", "{}"),
                              ("profile_name", "")]:
            if col not in df.columns:
                df[col] = default

        # Demote old baseline for this preset
        mask_old = (df["preset"] == preset_name) & (df["is_baseline"] == True)  # noqa: E712
        df.loc[mask_old, "is_baseline"] = False

        # Promote the target row
        mask_new = df["timestamp"] == row_timestamp
        df.loc[mask_new, "is_baseline"] = True

        df.to_parquet(_HISTORY_FILE, index=False)
        _logger.info("Promoted baseline for %s (ts=%s)", preset_name, row_timestamp)

        try:
            from yak_core.github_persistence import sync_feedback_async
            sync_feedback_async(
                files=[_HISTORY_REL_PATH],
                commit_message=f"Promote Sim Lab baseline ({preset_name})",
            )
        except Exception as sync_err:
            _logger.warning("GitHub sync failed: %s", sync_err)
    except Exception as exc:
        _logger.warning("Failed to promote baseline: %s", exc)


def _remove_history_rows(timestamps: List[str]) -> None:
    """Soft-delete rows from batch history by setting removed=True."""
    if not _HISTORY_FILE.is_file() or not timestamps:
        return
    try:
        df = pd.read_parquet(_HISTORY_FILE)
        if "removed" not in df.columns:
            df["removed"] = False
        df.loc[df["timestamp"].isin(timestamps), "removed"] = True
        df.to_parquet(_HISTORY_FILE, index=False)
        _logger.info("Soft-deleted %d batch history rows", len(timestamps))

        try:
            from yak_core.github_persistence import sync_feedback_async
            sync_feedback_async(
                files=[_HISTORY_REL_PATH],
                commit_message="Remove Sim Lab batch history rows",
            )
        except Exception as sync_err:
            _logger.warning("GitHub sync failed: %s", sync_err)
    except Exception as exc:
        _logger.warning("Failed to remove history rows: %s", exc)


def _get_active_baseline(preset_name: str) -> Optional[pd.Series]:
    """Return the active baseline row for a preset, or None."""
    history = _load_batch_history()
    if history.empty or "is_baseline" not in history.columns:
        return None
    bl = history[(history["preset"] == preset_name) & (history["is_baseline"] == True)]  # noqa: E712
    if bl.empty:
        return None
    if "timestamp" in bl.columns:
        bl = bl.sort_values("timestamp", ascending=False)
    return bl.iloc[0]


def _render_history_table() -> None:
    """Render the persistent batch history table (all presets, full log)."""
    history = _load_batch_history()
    if history.empty:
        st.caption("No batch history yet — run a batch to start tracking.")
        return

    # Ensure new columns on legacy data
    for col, default in [("is_baseline", False), ("removed", False),
                          ("config_label", ""), ("profile_name", "")]:
        if col not in history.columns:
            history[col] = default

    # Exclude soft-deleted rows
    history = history[~history["removed"]].copy()
    if history.empty:
        return

    st.subheader("Batch History")

    # Sort newest first
    if "timestamp" in history.columns:
        history = history.sort_values("timestamp", ascending=False).reset_index(drop=True)

    display = history.copy()

    # Format timestamp for readability
    if "timestamp" in display.columns:
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%m/%d %I:%M %p")

    # Add baseline indicator column
    if "is_baseline" in display.columns:
        display["role"] = display["is_baseline"].apply(lambda x: "\u2693 Baseline" if x else "")
    else:
        display["role"] = ""

    col_rename = {
        "timestamp": "When",
        "preset": "Preset",
        "role": "Role",
        "archetype": "Archetype",
        "config_hash": "Config",
        "num_dates": "Dates",
        "num_lineups": "Lineups",
        "avg_actual": "Avg Actual",
        "avg_proj": "Avg Proj",
        "best_slate": "Best Slate",
        "worst_slate": "Worst Slate",
        "beat_proj_pct": "Beat Proj %",
        "errors": "Errors",
    }
    show_cols = [c for c in col_rename if c in display.columns]
    display = display[show_cols].rename(columns=col_rename)

    st.dataframe(
        display.style.format({
            "Avg Actual": "{:.1f}",
            "Avg Proj": "{:.1f}",
            "Best Slate": "{:.1f}",
            "Worst Slate": "{:.1f}",
            "Beat Proj %": "{:.0f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# RG archive helpers (NBA)
# ---------------------------------------------------------------------------

_RG_ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "rg_archive" / "nba"
_RG_DATE_RE = re.compile(r"^rg_(\d{4}-\d{2}-\d{2})\.csv$")


def _scan_rg_dates() -> List[date]:
    """Return dates with RG archive files, sorted most-recent-first."""
    if not _RG_ARCHIVE_DIR.is_dir():
        return []
    dates: List[date] = []
    for f in _RG_ARCHIVE_DIR.iterdir():
        m = _RG_DATE_RE.match(f.name)
        if m:
            try:
                dates.append(date.fromisoformat(m.group(1)))
            except ValueError:
                continue
    dates.sort(reverse=True)
    return dates


def _merge_rg_csv(pool: pd.DataFrame, rg_file: Path) -> pd.DataFrame:
    """Merge RotoGrinders CSV projections into the player pool."""
    try:
        rg = pd.read_csv(rg_file, encoding="utf-8-sig")
    except Exception:
        try:
            rg = pd.read_csv(rg_file, encoding="latin-1")
        except Exception:
            rg = pd.read_csv(rg_file, sep=None, engine="python")

    rg.columns = [c.strip().upper() for c in rg.columns]

    if "PLAYER" not in rg.columns:
        st.error(
            f"RG CSV missing PLAYER column. "
            f"Found columns: {', '.join(rg.columns[:10])}"
        )
        return pool

    rg["_join_name"] = rg["PLAYER"].astype(str).str.strip().str.lower()
    pool["_join_name"] = pool["player_name"].astype(str).str.strip().str.lower()
    pool["rg_proj"] = float("nan")
    rg_lookup = rg.set_index("_join_name")

    n_merged = 0
    n_missing = 0
    for idx, row in pool.iterrows():
        jn = row["_join_name"]
        if jn not in rg_lookup.index:
            n_missing += 1
            continue
        r = rg_lookup.loc[jn]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rg_proj = float(r.get("FPTS", 0) or 0)
        if rg_proj > 0:
            pool.at[idx, "proj"] = rg_proj
            pool.at[idx, "rg_proj"] = rg_proj
            pool.at[idx, "proj_source"] = "rotogrinders"
            n_merged += 1
        rg_sal = r.get("SALARY")
        if rg_sal is not None and not pd.isna(rg_sal):
            rg_sal = float(rg_sal)
            if rg_sal > 0:
                pool.at[idx, "salary"] = int(rg_sal)
        rg_floor = float(r.get("FLOOR", 0) or 0)
        rg_ceil = float(r.get("CEIL", 0) or 0)
        if rg_floor > 0:
            pool.at[idx, "floor"] = rg_floor
        if rg_ceil > 0:
            pool.at[idx, "ceil"] = rg_ceil
        pown_str = str(r.get("POWN", "0%")).replace("%", "").strip()
        try:
            pown_val = float(pown_str)
        except (ValueError, TypeError):
            pown_val = 0.0
        if pown_val > 0:
            pool.at[idx, "ownership"] = pown_val
            pool.at[idx, "own_proj"] = pown_val
        for sim_col in ["SIM15TH", "SIM33RD", "SIM50TH", "SIM66TH", "SIM85TH", "SIM90TH", "SIM99TH"]:
            val = r.get(sim_col)
            if val is not None and not pd.isna(val):
                pool.at[idx, sim_col.lower()] = float(val)
        smash_val = r.get("SMASH")
        if smash_val is not None and not pd.isna(smash_val):
            pool.at[idx, "smash_prob"] = float(smash_val)
    pool.drop(columns=["_join_name"], inplace=True)

    rg_fpts_range = f"{rg['FPTS'].min():.0f}-{rg['FPTS'].max():.0f}" if "FPTS" in rg.columns else "N/A"
    _logger.info(
        "RG merge: %d/%d players matched (%d unmatched) | %d rows | FPTS %s",
        n_merged, len(pool), n_missing, len(rg), rg_fpts_range,
    )
    if n_merged == 0:
        _logger.warning("No players matched from RG file — using YakOS model projections")
    return pool


# ---------------------------------------------------------------------------
# PGA helpers
# ---------------------------------------------------------------------------

def _fetch_pga_pool(api_key: str) -> pd.DataFrame:
    from yak_core.datagolf import DataGolfClient
    dg = DataGolfClient(api_key)
    pool = dg.get_dfs_projections(site="draftkings", slate="main")
    if pool.empty:
        raise ValueError("DataGolf returned an empty projection pool.")
    if "proj_own" in pool.columns and "ownership" not in pool.columns:
        pool["ownership"] = pool["proj_own"]
    if "player_name" not in pool.columns and "dg_id" in pool.columns:
        pool["player_name"] = pool["dg_id"]
    return pool


def _fetch_pga_actuals(api_key: str, slate_date: str = "") -> pd.DataFrame:
    from yak_core.datagolf import DataGolfClient
    from yak_core.pga_calibration import fetch_pga_actuals, get_pga_event_list

    dg = DataGolfClient(api_key)
    events = get_pga_event_list(dg)
    if events.empty:
        raise ValueError("No PGA events found in DataGolf event list.")

    target = slate_date.replace("-", "") if slate_date else ""
    chosen = events.iloc[0]
    if target:
        for _, ev in events.iterrows():
            ev_date = str(ev.get("date", "")).replace("-", "")[:8]
            if ev_date and ev_date <= target:
                chosen = ev
                break

    event_id = int(chosen["event_id"])
    year = int(chosen.get("calendar_year", datetime.now().year))
    actuals = fetch_pga_actuals(dg, event_id, year)
    if actuals.empty:
        raise ValueError(
            f"No actuals returned for event '{chosen.get('event_name', event_id)}' ({year})."
        )
    actuals["actual_fp"] = pd.to_numeric(actuals["actual_fp"], errors="coerce").fillna(0.0)
    return actuals


# ---------------------------------------------------------------------------
# Scatter Plot (Chart.js) — kept for potential future use
# ---------------------------------------------------------------------------

_EDGE_COLORS = {
    "smash": "#00ff87",
    "solid": "#4dabf7",
    "risky": "#ffa726",
    "bust": "#ef5350",
}
_DEFAULT_DOT_COLOR = "#888888"


def _classify_edge(row: pd.Series) -> str:
    smash = float(row.get("smash_prob", 0))
    bust = float(row.get("bust_prob", 0))
    if smash >= 0.35:
        return "smash"
    if smash >= 0.20 and bust < 0.30:
        return "solid"
    if bust >= 0.35:
        return "bust"
    if bust >= 0.25:
        return "risky"
    return "solid"


def _build_scatter_html(player_df: pd.DataFrame) -> str:
    """Build a self-contained Chart.js dark-theme scatter plot."""
    points: List[Dict[str, Any]] = []
    for _, r in player_df.iterrows():
        ec = _classify_edge(r)
        points.append({
            "x": round(float(r.get("proj", 0)), 2),
            "y": round(float(r.get("actual_fp", 0)), 2),
            "label": str(r.get("player_name", "")),
            "salary": int(r.get("salary", 0)),
            "color": _EDGE_COLORS.get(ec, _DEFAULT_DOT_COLOR),
        })
    max_val = max(
        max((p["x"] for p in points), default=1),
        max((p["y"] for p in points), default=1),
    ) * 1.1
    data_json = json.dumps(points, separators=(",", ":"))
    return f"""
<div style="width:100%;height:460px;background:#0f1117;border-radius:8px;padding:8px;">
<canvas id="simScatter"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7"></script>
<script>
(function(){{
const pts={data_json};
const maxV={max_val:.1f};
const ctx=document.getElementById('simScatter').getContext('2d');
new Chart(ctx,{{
  type:'scatter',
  data:{{
    datasets:[
      {{
        label:'Players',
        data:pts.map(p=>({{x:p.x,y:p.y}})),
        backgroundColor:pts.map(p=>p.color),
        pointRadius:5,
        pointHoverRadius:7,
      }},
      {{
        label:'Perfect Line',
        type:'line',
        data:[{{x:0,y:0}},{{x:maxV,y:maxV}}],
        borderColor:'rgba(255,255,255,0.2)',
        borderDash:[5,5],
        pointRadius:0,
        fill:false,
      }}
    ]
  }},
  options:{{
    responsive:true,
    maintainAspectRatio:false,
    plugins:{{
      legend:{{display:false}},
      tooltip:{{
        callbacks:{{
          label:function(ctx){{
            const i=ctx.dataIndex;
            const p=pts[i];
            if(!p) return '';
            return p.label+' | Proj: '+p.x+' | Actual: '+p.y+' | $'+p.salary.toLocaleString();
          }}
        }}
      }}
    }},
    scales:{{
      x:{{
        title:{{display:true,text:'Projected FP',color:'#ccc'}},
        grid:{{color:'rgba(255,255,255,0.06)'}},
        ticks:{{color:'#aaa'}},
        min:0,max:maxV,
      }},
      y:{{
        title:{{display:true,text:'Actual FP',color:'#ccc'}},
        grid:{{color:'rgba(255,255,255,0.06)'}},
        ticks:{{color:'#aaa'}},
        min:0,max:maxV,
      }}
    }}
  }}
}});
}})();
</script>
"""


# ---------------------------------------------------------------------------
# Pipeline (unchanged from v1)
# ---------------------------------------------------------------------------

def _run_pipeline(
    sport: str,
    selected_date: date,
    preset_name: str,
    sandbox_overrides: Dict[str, Any],
    archetype: str = "Default",
    *,
    ricky_w_gpp: Optional[float] = None,
    ricky_w_ceil: Optional[float] = None,
    ricky_w_own: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute the full fetch -> build -> score pipeline. Returns a run dict."""
    date_key = selected_date.strftime("%Y%m%d")
    date_dash = selected_date.strftime("%Y-%m-%d")
    preset = CONTEST_PRESETS[preset_name]
    cfg = merge_config(preset)
    cfg.update(sandbox_overrides)

    if "NUM_LINEUPS" not in cfg or cfg["NUM_LINEUPS"] <= 0:
        cfg["NUM_LINEUPS"] = preset.get("default_lineups", 10)

    # Step 1: Fetch pool
    if sport == "NBA":
        api_key = _get_nba_api_key()
        if not api_key:
            raise ValueError("NBA API key not found. Set RAPIDAPI_KEY or TANK01_RAPIDAPI_KEY.")
        cfg["RAPIDAPI_KEY"] = api_key
        pool_df = fetch_live_dfs(date_key, cfg)
    else:
        api_key = _get_pga_api_key()
        if not api_key:
            raise ValueError("PGA API key not found. Set DATAGOLF_API_KEY.")
        pool_df = _fetch_pga_pool(api_key)

    if pool_df is None or pool_df.empty:
        raise ValueError(f"No DFS pool found for {date_dash}.")

    # Step 2: Merge RG projections (NBA only)
    if sport == "NBA":
        rg_path = _RG_ARCHIVE_DIR / f"rg_{date_dash}.csv"
        if rg_path.is_file():
            pool_df = _merge_rg_csv(pool_df, rg_path)
        else:
            _logger.warning("No RG archive file for %s — using Tank01 projections", date_dash)

    # Step 3: Auto-run Monte Carlo sims (if sim columns missing)
    if sport == "NBA" and "sim90th" not in pool_df.columns and "SIM90TH" not in pool_df.columns:
        try:
            from yak_core.edge import compute_empirical_std
            _proj = pd.to_numeric(pool_df["proj"], errors="coerce").fillna(0)
            _sal = pd.to_numeric(pool_df["salary"], errors="coerce").fillna(0)
            _std = compute_empirical_std(_proj.values, _sal.values, variance_mult=1.0)
            _n_sims = 5000
            _rng = np.random.default_rng(42)
            _sim_matrix = _rng.normal(
                loc=_proj.values[None, :],
                scale=_std[None, :],
                size=(_n_sims, len(_proj)),
            )
            _sim_matrix = np.maximum(_sim_matrix, 0.0)
            for _pct, _col in [(15, "sim15th"), (33, "sim33rd"), (50, "sim50th"),
                                (66, "sim66th"), (85, "sim85th"), (90, "sim90th"), (99, "sim99th")]:
                pool_df[_col] = np.percentile(_sim_matrix, _pct, axis=0).round(2)
            _logger.info("Auto-ran %d player sims — sim columns populated", _n_sims)
        except Exception as _sim_err:
            _logger.warning("Auto-sim failed (%s), continuing with fallback estimates", _sim_err)

    # Step 4: Compute edge
    edge_df = compute_edge_metrics(pool_df, calibration_state=None, sport=sport)

    # Step 4b: Clean NaN/inf values that crash PuLP's LP solver
    numeric_cols = edge_df.select_dtypes(include="number").columns
    edge_df[numeric_cols] = edge_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Step 4c: Drop players with no usable projection
    edge_df = edge_df[edge_df["proj"] > 0].copy()

    # Step 5: Fetch actuals
    if sport == "NBA":
        actuals_df = fetch_actuals_from_api(date_key, cfg)
    else:
        actuals_df = _fetch_pga_actuals(api_key, date_dash)

    # Step 5b: Overlay showdown salaries from archive (if showdown preset)
    is_showdown = "showdown" in preset_name.lower()
    if is_showdown and sport == "NBA":
        try:
            from yak_core.slate_archive import load_all_showdown_salaries
            sd_sal_map = load_all_showdown_salaries(date_dash)
            if sd_sal_map:
                _overlaid = 0
                for idx, row in edge_df.iterrows():
                    pname = str(row.get("player_name", "")).strip()
                    if pname in sd_sal_map:
                        edge_df.at[idx, "salary"] = sd_sal_map[pname]
                        _overlaid += 1
                _logger.info(
                    "Showdown salary overlay: %d/%d players matched from archive",
                    _overlaid, len(sd_sal_map),
                )
            else:
                _logger.warning("No showdown salary archive for %s — using classic salaries", date_dash)
        except Exception as _sd_err:
            _logger.warning("Showdown salary overlay failed (%s) — using classic salaries", _sd_err)

    # Step 6: Build lineups
    prepped = prepare_pool(edge_df, cfg)

    if is_showdown:
        lineups_df, _ = build_showdown_lineups(prepped, cfg)
    else:
        lineups_df, _ = build_multiple_lineups_with_exposure(prepped, cfg)

    if lineups_df is None or lineups_df.empty:
        raise ValueError("Optimizer returned no lineups. Try adjusting config.")

    n_lineups = lineups_df["lineup_index"].nunique() if "lineup_index" in lineups_df.columns else (
        lineups_df["lineup_id"].nunique() if "lineup_id" in lineups_df.columns else 1
    )
    _logger.info("%d lineups generated for %s", n_lineups, date_dash)

    # Step 7: Score lineups
    if "player_name" not in actuals_df.columns:
        for c in ("name", "dg_name", "player"):
            if c in actuals_df.columns:
                actuals_df = actuals_df.rename(columns={c: "player_name"})
                break

    scored = lineups_df.merge(
        actuals_df[["player_name", "actual_fp"]].drop_duplicates(subset="player_name"),
        on="player_name",
        how="left",
        suffixes=("", "_actual"),
    )
    if "actual_fp_actual" in scored.columns:
        scored["actual_fp"] = scored["actual_fp_actual"].fillna(scored.get("actual_fp", 0.0))
        scored.drop(columns=["actual_fp_actual"], inplace=True)
    scored["actual_fp"] = pd.to_numeric(scored["actual_fp"], errors="coerce").fillna(0.0)

    if "lineup_index" not in scored.columns:
        if "lineup_id" in scored.columns:
            scored["lineup_index"] = scored["lineup_id"]
        else:
            scored["lineup_index"] = 0

    # ── Aggregate lineup-level metrics ────────────────────────────────────
    # Ensure columns exist for ranking even if missing from pool
    for _col, _default in [("gpp_score", 0.0), ("ceil", 0.0), ("own_pct", 0.0)]:
        if _col not in scored.columns:
            scored[_col] = _default

    summary = (
        scored.groupby("lineup_index")
        .agg(
            total_actual=("actual_fp", "sum"),
            total_proj=("proj", "sum"),
            total_salary=("salary", "sum"),
            total_gpp_score=("gpp_score", "sum"),
            total_ceil=("ceil", "sum"),
            avg_own_pct=("own_pct", "mean"),
        )
        .reset_index()
    )
    summary["diff"] = summary["total_actual"] - summary["total_proj"]
    summary = summary.sort_values("total_actual", ascending=False).reset_index(drop=True)

    # ── Ricky SE ranking (non-destructive layer on top) ─────────────────
    summary = rank_lineups_for_se(
        summary, w_gpp=ricky_w_gpp, w_ceil=ricky_w_ceil, w_own=ricky_w_own,
    )

    beat_proj_pct = 0.0
    if len(summary) > 0:
        beat_proj_pct = float((summary["total_actual"] >= summary["total_proj"]).mean() * 100)

    chash = _config_hash(sandbox_overrides, archetype=archetype)

    return {
        "timestamp": datetime.now().isoformat(),
        "date": str(selected_date),
        "preset": preset_name,
        "archetype": archetype,
        "sport": sport,
        "num_lineups": len(summary),
        "avg_actual": round(float(summary["total_actual"].mean()), 2) if len(summary) else 0,
        "avg_proj": round(float(summary["total_proj"].mean()), 2) if len(summary) else 0,
        "best": round(float(summary["total_actual"].max()), 2) if len(summary) else 0,
        "beat_proj_pct": round(beat_proj_pct, 1),
        "config_hash": chash,
        "summary_df": summary,
        "player_df": scored,
    }


# ---------------------------------------------------------------------------
# Config Panel (v2 — grouped knobs)
# ---------------------------------------------------------------------------

def _slider_default(preset_name: str, key: str, fallback: Any) -> Any:
    """Look up the default value for a config key.

    Uses merge_config(preset) as the source of truth so that alias
    mappings (e.g. preset ``low_own_threshold`` → ``GPP_LOW_OWN_THRESHOLD``)
    are resolved the same way the optimizer would resolve them.
    """
    preset = CONTEST_PRESETS[preset_name]
    merged = merge_config(preset)
    val = merged.get(key)
    if val is not None:
        return val
    return fallback


def _render_config_panel(preset_name: str) -> Dict[str, Any]:
    """Render the 4-group config panel. Returns ONLY keys the user changed."""
    sk = _sandbox_config_key(preset_name)
    if sk not in st.session_state:
        st.session_state[sk] = {}
    overrides: Dict[str, Any] = st.session_state[sk]

    def _sl(label: str, key: str, mn: float, mx: float, step: float, fallback: Any, fmt: str = "%.2f") -> Any:
        """Render a slider. Only store in overrides if user changed from preset default."""
        preset_default = _slider_default(preset_name, key, fallback)
        current = overrides.get(key, preset_default)
        if isinstance(current, (int, float)):
            current = max(mn, min(mx, current))
        if isinstance(mn, int) and isinstance(mx, int) and isinstance(step, int):
            val = st.slider(label, min_value=mn, max_value=mx, value=int(current), step=step, key=f"sl_{preset_name}_{key}")
        else:
            val = st.slider(label, min_value=float(mn), max_value=float(mx), value=float(current), step=float(step), format=fmt, key=f"sl_{preset_name}_{key}")
        # Only store override if user moved slider away from preset default
        clamped_default = max(mn, min(mx, preset_default)) if isinstance(preset_default, (int, float)) else preset_default
        if val != clamped_default:
            overrides[key] = val
        elif key in overrides:
            del overrides[key]
        return val

    # Group 1: Core Weights (expanded by default)
    with st.expander("Core Weights", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            _sl("Proj Weight", "GPP_PROJ_WEIGHT", 0.0, 1.0, 0.05, 0.30)
            _sl("Upside Weight", "GPP_UPSIDE_WEIGHT", 0.0, 1.0, 0.05, 0.30)
        with c2:
            _sl("Boom Weight", "GPP_BOOM_WEIGHT", 0.0, 1.0, 0.05, 0.35)
            _sl("Own Penalty Strength", "GPP_OWN_PENALTY_STRENGTH", 0.0, 3.0, 0.1, 1.0, fmt="%.1f")

    # Group 2: Edge Signals
    with st.expander("Edge Signals"):
        c1, c2 = st.columns(2)
        with c1:
            _sl("Smash Weight", "GPP_SMASH_WEIGHT", 0.0, 0.50, 0.05, 0.15)
            _sl("DVP Weight", "GPP_DVP_WEIGHT", 0.0, 0.50, 0.01, 0.12)
            _sl("Pace Env Weight", "GPP_PACE_ENV_WEIGHT", 0.0, 0.50, 0.01, 0.10)
            _sl("Form Weight", "GPP_FORM_WEIGHT", 0.0, 0.50, 0.01, 0.08)
        with c2:
            _sl("Bust Penalty", "GPP_BUST_PENALTY", 0.0, 0.50, 0.05, 0.10)
            _sl("Spread Penalty", "GPP_SPREAD_PENALTY_WEIGHT", 0.0, 0.50, 0.01, 0.08)
            _sl("Catalyst Weight", "GPP_CATALYST_WEIGHT", 0.0, 0.50, 0.05, 0.05)
            _sl("Efficiency Weight", "GPP_EFFICIENCY_WEIGHT", 0.0, 0.50, 0.05, 0.05)

    # Group 3: Ownership Edge
    with st.expander("Ownership Edge"):
        c1, c2 = st.columns(2)
        with c1:
            _sl("Own Weight", "OWN_WEIGHT", 0.0, 1.0, 0.05, 0.0)
            _sl("Leverage Weight", "GPP_LEVERAGE_WEIGHT", 0.0, 0.50, 0.05, 0.05)
            _sl("Min Low Own Players", "GPP_MIN_LOW_OWN_PLAYERS", 0, 4, 1, 1)
        with c2:
            _sl("Low Own Threshold", "GPP_LOW_OWN_THRESHOLD", 0.0, 0.50, 0.05, 0.10)
            _sl("Min Player Minutes", "MIN_PLAYER_MINUTES", 0, 30, 1, 0)

    # Group 4: Structure (collapsed)
    with st.expander("Structure"):
        c1, c2 = st.columns(2)
        with c1:
            _sl("Num Lineups", "NUM_LINEUPS", 1, 50, 1, 10)
            _sl("Min Salary Used", "MIN_SALARY_USED", 45000, 50000, 500, 49000)
        with c2:
            _sl("Max Exposure", "MAX_EXPOSURE", 0.1, 1.0, 0.05, 0.6)
            _sl("Min Uniques", "MIN_UNIQUES", 0, 5, 1, 0)

    st.session_state[sk] = overrides
    return overrides


# ---------------------------------------------------------------------------
# Batch Run
# ---------------------------------------------------------------------------

def _run_batch(
    sport: str,
    preset_name: str,
    sandbox_overrides: Dict[str, Any],
    dates: List[date],
    archetype: str = "Default",
    *,
    ricky_w_gpp: Optional[float] = None,
    ricky_w_ceil: Optional[float] = None,
    ricky_w_own: Optional[float] = None,
    profile_name: str = "",
) -> Dict[str, Any]:
    """Run the pipeline for every date. Returns a batch record."""
    runs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    progress = st.progress(0, text="Starting batch run...")

    for i, d in enumerate(dates):
        progress.progress(
            (i + 1) / len(dates),
            text=f"Running {d.strftime('%Y-%m-%d')} ({i + 1}/{len(dates)})",
        )
        try:
            run = _run_pipeline(
                    sport, d, preset_name, sandbox_overrides,
                    archetype=archetype,
                    ricky_w_gpp=ricky_w_gpp, ricky_w_ceil=ricky_w_ceil,
                    ricky_w_own=ricky_w_own,
                )
            runs.append(run)
            # Also store in the flat run log
            if "sim_lab_runs" not in st.session_state:
                st.session_state["sim_lab_runs"] = []
            st.session_state["sim_lab_runs"].append(run)
        except Exception as exc:
            errors.append({"date": str(d), "error": str(exc)})

    progress.empty()

    chash = _config_hash(sandbox_overrides, archetype=archetype)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine batch number
    batches = st.session_state.get("sim_lab_batches", [])
    batch_number = len(batches) + 1

    avg_actual = round(mean(r["avg_actual"] for r in runs), 2) if runs else 0
    avg_proj = round(mean(r["avg_proj"] for r in runs), 2) if runs else 0
    beat_proj_pct = round(mean(r["beat_proj_pct"] for r in runs), 1) if runs else 0

    # Config label includes archetype for non-Default
    label = f"Run {batch_number}"
    if archetype != "Default":
        label = f"Run {batch_number} ({archetype})"

    return {
        "batch_id": f"{chash}_{timestamp}",
        "preset": preset_name,
        "archetype": archetype,
        "config_hash": chash,
        "config_label": label,
        "profile_name": profile_name,
        "overrides": sandbox_overrides.copy(),
        "runs": runs,
        "errors": errors,
        "avg_actual": avg_actual,
        "avg_proj": avg_proj,
        "beat_proj_pct": beat_proj_pct,
    }


# ---------------------------------------------------------------------------
# Promote Config
# ---------------------------------------------------------------------------

def _render_promote_config(
    preset_name: str,
    sandbox_overrides: Dict[str, Any],
    ricky_weights: Dict[str, float],
    archetype_name: str = "Default",
) -> None:
    """Render the Promote Config panel after a validated batch run.

    Snapshots the current config + Ricky weights + validation stats from
    the latest batch and saves it as a selectable promoted profile.
    Uses the archetype name (not the preset key) for the default config
    key and display name.
    """
    batches: List[Dict[str, Any]] = st.session_state.get("sim_lab_batches", [])
    if not batches:
        return

    latest = batches[-1]
    runs = latest.get("runs", [])
    if not runs:
        return

    st.markdown("---")
    st.markdown("#### Promote Config")
    st.caption(
        "Save this config as a selectable profile on the Optimizer and Lab pages. "
        "Existing profiles with the same key will be replaced."
    )

    # Compute validation stats from the latest batch
    avg_diffs = [r["avg_actual"] - r["avg_proj"] for r in runs]
    avg_diff = round(sum(avg_diffs) / len(avg_diffs), 2) if avg_diffs else 0
    beat_pct = round(sum(r["beat_proj_pct"] for r in runs) / len(runs), 1) if runs else 0
    dates_tested = sorted(set(r["date"] for r in runs))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Avg Diff (Actual − Proj)", f"{avg_diff:+.2f} FP")
        st.metric("Beat Proj %", f"{beat_pct:.1f}%")
    with c2:
        st.metric("Sample Size", f"{len(runs)} slates")
        st.metric("Dates Tested", f"{len(dates_tested)}")

    # Config key + display name default to archetype name when one is active,
    # otherwise fall back to preset name.  This gives promoted configs
    # human-readable names like "Stars & Scrubs Ceiling" instead of
    # "GPP_MAIN_V1".
    if archetype_name and archetype_name != "Default":
        _base_label = archetype_name  # e.g. "Stars & Scrubs Ceiling"
    else:
        _base_label = preset_name     # e.g. "GPP Main"
    _default_key = _base_label.upper().replace(" ", "_").replace("&", "AND") + "_V1"
    _default_display = _base_label

    config_key = st.text_input(
        "Config Key",
        value=_default_key,
        key="promote_config_key",
    )
    config_display = st.text_input(
        "Display Name",
        value=_default_display,
        key="promote_config_display",
    )
    config_desc = st.text_input(
        "Description (optional)",
        value="",
        key="promote_config_desc",
    )

    if st.button("Promote Config", type="primary", key="promote_config_btn"):
        from yak_core.promoted_configs import promote_config
        entry = promote_config(
            key=config_key.strip(),
            display_name=config_display.strip(),
            base_preset=preset_name,
            overrides=dict(sandbox_overrides),
            ricky_weights=dict(ricky_weights),
            validation_stats={
                "avg_diff": avg_diff,
                "beat_proj_pct": beat_pct,
                "sample_size": len(runs),
                "dates_tested": dates_tested,
            },
            description=config_desc.strip(),
        )
        st.success(f"Config promoted as **{entry['key']}** — available in Optimizer and Lab dropdowns.")

    # Show existing promoted configs
    from yak_core.promoted_configs import list_promoted
    promoted = list_promoted()
    if promoted:
        with st.expander(f"Promoted Configs ({len(promoted)})"):
            for pc in promoted:
                vs = pc.get("validation_stats", {})
                _stats = (
                    f"Avg Diff: {vs.get('avg_diff', '?'):+.2f} FP | "
                    f"Beat%: {vs.get('beat_proj_pct', '?')}% | "
                    f"N={vs.get('sample_size', '?')} slates"
                ) if vs else "No validation stats"
                st.markdown(
                    f"**{pc['key']}** — {pc.get('display_name', '')}  \n"
                    f"{pc.get('description', '')}  \n"
                    f"_{_stats}_  \n"
                    f"Base: {pc['base_preset']} | Promoted: {pc.get('promoted_at', '?')[:10]}"
                )
                st.markdown("---")


# ---------------------------------------------------------------------------
# Trend Chart (Chart.js — dark mode)
# ---------------------------------------------------------------------------

def _render_trend_chart() -> None:
    """Render trend charts.

    1. **Session chart** (per-date detail) — only when batches exist in session.
    2. **Persistent chart** — reads from ``batch_history.parquet`` so it
       survives page refreshes.  X = batch run time, Y = avg actual FP.
       Baseline is drawn as a dashed reference line.
    """
    # ---- Session per-date chart (when available) ----
    batches: List[Dict[str, Any]] = st.session_state.get("sim_lab_batches", [])
    if batches:
        st.subheader("Config Comparison — Avg Actual FP by Date")

        datasets_js = []
        for i, batch in enumerate(batches):
            color = _BATCH_COLORS[i % len(_BATCH_COLORS)]
            sorted_runs = sorted(batch["runs"], key=lambda r: r["date"])
            data_points = [{"x": run["date"], "y": run["avg_actual"]} for run in sorted_runs]

            label = f"{batch['config_label']} ({batch['config_hash']})"
            datasets_js.append({
                "label": label,
                "data": data_points,
                "borderColor": color,
                "backgroundColor": color,
                "tension": 0.2,
                "pointRadius": 4,
                "pointHoverRadius": 6,
                "fill": False,
            })

        datasets_json = json.dumps(datasets_js, separators=(",", ":"))
        chart_html = f"""
<div style="width:100%;height:400px;background:#0f1117;border-radius:8px;padding:12px;">
<canvas id="trendChart"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7"></script>
<script>
(function(){{
const ds={datasets_json};
const ctx=document.getElementById('trendChart').getContext('2d');
new Chart(ctx,{{
  type:'line',
  data:{{datasets:ds}},
  options:{{
    responsive:true,
    maintainAspectRatio:false,
    plugins:{{
      legend:{{
        display:true,
        labels:{{color:'#ccc',font:{{size:12}}}}
      }},
      tooltip:{{
        mode:'index',
        intersect:false,
        callbacks:{{
          label:function(ctx){{
            return ctx.dataset.label+': '+ctx.parsed.y.toFixed(1)+' FP';
          }}
        }}
      }}
    }},
    scales:{{
      x:{{
        type:'category',
        labels:[...new Set(ds.flatMap(d=>d.data.map(p=>p.x)))].sort(),
        title:{{display:true,text:'Date',color:'#ccc'}},
        grid:{{color:'rgba(255,255,255,0.06)'}},
        ticks:{{color:'#aaa',maxRotation:45}}
      }},
      y:{{
        title:{{display:true,text:'Avg Actual FP',color:'#ccc'}},
        grid:{{color:'rgba(255,255,255,0.06)'}},
        ticks:{{color:'#aaa'}}
      }}
    }},
    interaction:{{
      mode:'nearest',
      axis:'x',
      intersect:false,
    }}
  }}
}});
}})();
</script>
"""
        components.html(chart_html, height=440, scrolling=False)


def _ema(values: list[float], alpha: float = 0.35) -> list[float]:
    """Compute exponential moving average over a list of floats."""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def _render_persistent_trend(preset_name: str) -> None:
    """Persistent EMA cloud chart from batch_history — survives restarts.

    Renders two EMA clouds:

    * **Green band** (Main Config) — the promoted baseline, drawn as a flat
      reference band spanning the full X axis (dashed lines, filled between
      avg and best).
    * **Blue band** (Sim Lab) — every non-baseline batch run, EMA-smoothed
      (solid lines, filled between avg and best).

    When the blue cloud rises above the green band, the tweaks are winning.
    Colors match the catch-rate charts: green = production, blue = experimental.
    """
    history = _load_batch_history()
    if history.empty:
        return

    for col, default in [("is_baseline", False), ("removed", False),
                          ("config_label", ""), ("best_slate", 0.0),
                          ("overrides_json", "{}"), ("profile_name", "")]:
        if col not in history.columns:
            history[col] = default

    df = history[(history["preset"] == preset_name) & (~history["removed"])].copy()
    if df.empty:
        return

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Classify: baseline = Main Config reference, everything else = Sim Lab
    main_df = df[df["is_baseline"] == True].reset_index(drop=True)   # noqa: E712
    lab_df = df[df["is_baseline"] != True].reset_index(drop=True)    # noqa: E712

    if len(main_df) < 1 and len(lab_df) < 1:
        return

    st.subheader("Performance Trend")

    datasets_js: list[dict] = []

    def _ts_label(ts: str) -> str:
        try:
            return pd.to_datetime(ts).strftime("%m/%d %I:%M%p")
        except Exception:
            return str(ts)[:16]

    # Build shared X-axis labels from ALL non-removed rows (order preserved)
    unique_labels = list(dict.fromkeys(
        _ts_label(row["timestamp"]) for _, row in df.iterrows()
    ))

    # ---- Main Config: flat reference band (dashed, green) ----
    if not main_df.empty:
        bl = main_df.iloc[-1]  # latest baseline
        avg_val = round(float(bl.get("avg_actual", 0)), 1)
        best_val = round(float(bl.get("best_slate", 0)), 1)

        best_pts = [{"x": lbl, "y": best_val} for lbl in unique_labels]
        avg_pts = [{"x": lbl, "y": avg_val} for lbl in unique_labels]

        best_idx = len(datasets_js)
        datasets_js.append({
            "label": f"Main Config Best ({best_val})",
            "data": best_pts,
            "borderColor": "#22c55e",
            "backgroundColor": "transparent",
            "borderDash": [6, 3],
            "tension": 0,
            "pointRadius": 0,
            "borderWidth": 2,
            "fill": False,
        })
        datasets_js.append({
            "label": f"Main Config Avg ({avg_val})",
            "data": avg_pts,
            "borderColor": "#16a34a",
            "backgroundColor": "rgba(34, 197, 94, 0.10)",
            "borderDash": [6, 3],
            "tension": 0,
            "pointRadius": 0,
            "borderWidth": 2,
            "fill": f"{best_idx}",
        })

    # ---- Sim Lab: EMA cloud (solid, blue) ----
    if not lab_df.empty:
        ts_labels = [_ts_label(r["timestamp"]) for _, r in lab_df.iterrows()]
        avg_raw = [float(r.get("avg_actual", 0)) for _, r in lab_df.iterrows()]
        best_raw = [float(r.get("best_slate", 0)) for _, r in lab_df.iterrows()]

        avg_ema = _ema(avg_raw)
        best_ema = _ema(best_raw)

        best_pts = [{"x": ts_labels[i], "y": round(best_ema[i], 1)} for i in range(len(ts_labels))]
        avg_pts = [{"x": ts_labels[i], "y": round(avg_ema[i], 1)} for i in range(len(ts_labels))]

        best_idx = len(datasets_js)
        datasets_js.append({
            "label": "Sim Lab Best",
            "data": best_pts,
            "borderColor": "#60a5fa",
            "backgroundColor": "transparent",
            "tension": 0.3,
            "pointRadius": 3,
            "pointHoverRadius": 5,
            "borderWidth": 2,
            "fill": False,
        })
        datasets_js.append({
            "label": "Sim Lab Avg",
            "data": avg_pts,
            "borderColor": "#3b82f6",
            "backgroundColor": "rgba(59, 130, 246, 0.12)",
            "tension": 0.3,
            "pointRadius": 3,
            "pointHoverRadius": 5,
            "borderWidth": 2,
            "fill": f"{best_idx}",
        })

    if not datasets_js:
        return

    datasets_json = json.dumps(datasets_js, separators=(",", ":"))
    labels_json = json.dumps(unique_labels, separators=(",", ":"))

    chart_html = f"""
<div style="width:100%;height:400px;background:#0f1117;border-radius:8px;padding:12px;">
<canvas id="emaCloud"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7"></script>
<script>
(function(){{
const ds={datasets_json};
const labels={labels_json};
ds.forEach(function(d){{
  if(typeof d.fill==='string' && !isNaN(d.fill))
    d.fill={{target:parseInt(d.fill),above:d.backgroundColor,below:'transparent'}};
}});
const ctx=document.getElementById('emaCloud').getContext('2d');
new Chart(ctx,{{
  type:'line',
  data:{{labels:labels,datasets:ds}},
  options:{{
    responsive:true,
    maintainAspectRatio:false,
    plugins:{{
      legend:{{display:true,labels:{{color:'#ccc',font:{{size:12}}}}}},
      tooltip:{{
        mode:'index',intersect:false,
        callbacks:{{label:function(c){{return c.dataset.label+': '+c.parsed.y.toFixed(1)+' FP';}}}}
      }},
      filler:{{propagate:true}}
    }},
    scales:{{
      x:{{type:'category',title:{{display:true,text:'Batch Run',color:'#ccc'}},grid:{{color:'rgba(255,255,255,0.06)'}},ticks:{{color:'#aaa',maxRotation:45}}}},
      y:{{title:{{display:true,text:'Actual FP (EMA)',color:'#ccc'}},grid:{{color:'rgba(255,255,255,0.06)'}},ticks:{{color:'#aaa'}}}}
    }},
    interaction:{{mode:'nearest',axis:'x',intersect:false}}
  }}
}});
}})();
</script>
"""
    components.html(chart_html, height=440, scrolling=False)


# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------

def _render_comparison_table(preset_name: str) -> None:
    """Render a persistent comparison table with baseline pinning, deltas, and checkboxes.

    Reads from ``batch_history.parquet`` — everything persists across sessions.
    Active baseline is pinned at the top.  Δ Actual and Δ Beat% columns show
    each row's difference vs the baseline.  Checkboxes let the user soft-delete
    rows (they stay in the file for trend tracking but are hidden from view).
    A "Promote as Baseline" button lets the user promote any row.
    """
    history = _load_batch_history()
    if history.empty:
        return

    # Ensure new columns on legacy data
    for col, default in [("is_baseline", False), ("removed", False),
                          ("config_label", ""), ("overrides_json", "{}"),
                          ("profile_name", "")]:
        if col not in history.columns:
            history[col] = default

    # Filter to current preset, exclude soft-deleted
    mask = (history["preset"] == preset_name) & (~history["removed"])
    df = history[mask].copy()
    if df.empty:
        return

    st.subheader("Batch Comparison")

    # Sort: baseline first, then newest-first
    df["_sort_key"] = df["is_baseline"].astype(int) * -1  # baseline = -1 → top
    if "timestamp" in df.columns:
        df = df.sort_values(["_sort_key", "timestamp"], ascending=[True, False])
    df = df.reset_index(drop=True)

    # Compute deltas against active baseline
    bl_row = df[df["is_baseline"] == True]  # noqa: E712
    bl_actual = bl_row["avg_actual"].iloc[0] if not bl_row.empty else None
    bl_beat = bl_row["beat_proj_pct"].iloc[0] if not bl_row.empty else None

    # Build display rows
    display_rows = []
    for _, row in df.iterrows():
        label = row.get("config_label", "") or row.get("config_hash", "")
        if row.get("is_baseline"):
            label = f"\u2693 {label}" if label else "\u2693 Baseline"

        ts = ""
        if "timestamp" in row.index and pd.notna(row["timestamp"]):
            try:
                ts = pd.to_datetime(row["timestamp"]).strftime("%m/%d %I:%M %p")
            except Exception:
                ts = str(row["timestamp"])[:16]

        d_actual = ""
        d_beat = ""
        if bl_actual is not None and not row.get("is_baseline"):
            diff_a = row["avg_actual"] - bl_actual
            d_actual = f"{diff_a:+.1f}"
            if bl_beat is not None:
                diff_b = row["beat_proj_pct"] - bl_beat
                d_beat = f"{diff_b:+.0f}%"

        profile_lbl = row.get("profile_name", "") or ""
        if profile_lbl and profile_lbl in NAMED_PROFILES:
            profile_lbl = NAMED_PROFILES[profile_lbl].get("display_name", profile_lbl)
        elif profile_lbl:
            from yak_core.promoted_configs import get_promoted
            _pc_meta = get_promoted(profile_lbl)
            if _pc_meta:
                profile_lbl = _pc_meta.get("display_name", profile_lbl)

        display_rows.append({
            "_ts": row["timestamp"],  # hidden key for actions
            "Run": label,
            "Profile": profile_lbl,
            "When": ts,
            "Archetype": row.get("archetype", "Default"),
            "Dates": int(row.get("num_dates", 0)),
            "Avg Actual": row["avg_actual"],
            "Avg Proj": row.get("avg_proj", 0.0),
            "\u0394 Actual": d_actual,
            "Beat %": f"{row['beat_proj_pct']:.0f}%",
            "\u0394 Beat%": d_beat,
            "Baseline": bool(row.get("is_baseline")),
        })

    display_df = pd.DataFrame(display_rows)

    # --- Checkboxes to remove rows ---
    with st.form(key="sim_lab_comparison_form"):
        # Show the table using st.dataframe (read-only) above the action buttons
        show_cols = [c for c in display_df.columns if c != "_ts"]
        st.dataframe(
            display_df[show_cols].style.format(
                {"Avg Actual": "{:.1f}", "Avg Proj": "{:.1f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Multiselect for removal (show labels + timestamps)
        remove_options = {}
        for _, r in display_df.iterrows():
            key = r["_ts"]
            lbl = f"{r['Run']}  ({r['When']})"
            remove_options[lbl] = key

        selected_remove = st.multiselect(
            "Select rows to remove",
            options=list(remove_options.keys()),
            key="sim_lab_remove_rows",
        )

        # Promote selector — only non-baseline rows
        non_bl = display_df[~display_df["Baseline"]]
        promote_options = {"(none)": None}
        for _, r in non_bl.iterrows():
            lbl = f"{r['Run']}  ({r['When']})"
            promote_options[lbl] = r["_ts"]

        selected_promote = st.selectbox(
            "Promote as Baseline",
            options=list(promote_options.keys()),
            key="sim_lab_promote_baseline",
        )

        submitted = st.form_submit_button("Apply Changes", use_container_width=True)

    if submitted:
        changed = False
        # Handle removals
        if selected_remove:
            ts_to_remove = [remove_options[lbl] for lbl in selected_remove]
            _remove_history_rows(ts_to_remove)
            changed = True

        # Handle promotion
        promote_ts = promote_options.get(selected_promote)
        if promote_ts is not None:
            _promote_baseline(promote_ts, preset_name)
            changed = True

        if changed:
            st.rerun()


# ---------------------------------------------------------------------------
# Ricky SE Shortlist + Per-Slate Detail
# ---------------------------------------------------------------------------

def _render_ricky_shortlist() -> None:
    """Render the Ricky SE Shortlist from the latest batch's runs."""
    batches: List[Dict[str, Any]] = st.session_state.get("sim_lab_batches", [])
    if not batches:
        return

    latest_batch = batches[-1]
    runs = latest_batch.get("runs", [])
    if not runs:
        return

    st.subheader("Ricky SE Shortlist")
    st.caption(
        "Top-ranked lineups per slate — SE Core / SE Spicy / SE Alt. "
        "Ranking uses GPP score, ceiling, and ownership leverage."
    )

    # Collect shortlisted lineups across all dates in this batch
    shortlist_rows: List[Dict[str, Any]] = []
    for run in runs:
        run_summary = run.get("summary_df")
        if run_summary is None or run_summary.empty:
            continue
        tagged = run_summary[run_summary["ricky_tag"] != ""].copy()
        if tagged.empty:
            continue
        tagged["date"] = run["date"]
        tagged["preset"] = run["preset"]
        tagged["archetype"] = run.get("archetype", "Default")
        shortlist_rows.append(tagged)

    if not shortlist_rows:
        st.caption("No lineups tagged in this batch.")
        return

    shortlist_df = pd.concat(shortlist_rows, ignore_index=True)

    # Display columns
    display_cols = ["date", "ricky_tag", "ricky_rank", "ricky_score"]
    for c in ["total_gpp_score", "total_ceil", "total_proj", "total_actual", "avg_own_pct", "total_salary", "diff"]:
        if c in shortlist_df.columns:
            display_cols.append(c)
    display = shortlist_df[[c for c in display_cols if c in shortlist_df.columns]].copy()

    col_rename = {
        "date": "Date",
        "ricky_tag": "Tag",
        "ricky_rank": "Rank",
        "ricky_score": "Score",
        "total_gpp_score": "GPP Score",
        "total_ceil": "Ceiling",
        "total_proj": "Proj",
        "total_actual": "Actual",
        "avg_own_pct": "Avg Own%",
        "total_salary": "Salary",
        "diff": "Diff",
    }
    display = display.rename(columns=col_rename)

    fmt = {
        "Score": "{:.4f}",
        "GPP Score": "{:.1f}",
        "Ceiling": "{:.1f}",
        "Proj": "{:.1f}",
        "Actual": "{:.1f}",
        "Avg Own%": "{:.1f}",
        "Salary": "${:,.0f}",
        "Diff": "{:+.1f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in display.columns}

    st.dataframe(
        display.style.format(fmt),
        use_container_width=True,
        hide_index=True,
    )


def _render_per_slate_detail() -> None:
    """Render expandable per-slate lineup tables with ricky_rank + ricky_tag."""
    batches: List[Dict[str, Any]] = st.session_state.get("sim_lab_batches", [])
    if not batches:
        return

    latest_batch = batches[-1]
    runs = latest_batch.get("runs", [])
    if not runs:
        return

    with st.expander("Per-Slate Lineup Detail", expanded=False):
        date_options = [r["date"] for r in runs]
        if not date_options:
            return
        selected_date = st.selectbox(
            "Select Date",
            options=date_options,
            key="sim_lab_slate_detail_date",
        )
        run = next((r for r in runs if r["date"] == selected_date), None)
        if run is None:
            return

        run_summary = run.get("summary_df")
        if run_summary is None or run_summary.empty:
            st.caption("No lineup data for this date.")
            return

        # Sort by ricky_rank for display
        display = run_summary.sort_values("ricky_rank").reset_index(drop=True).copy()

        show_cols = ["lineup_index", "ricky_tag", "ricky_rank", "ricky_score"]
        for c in ["total_gpp_score", "total_ceil", "total_proj", "total_actual",
                  "avg_own_pct", "total_salary", "diff"]:
            if c in display.columns:
                show_cols.append(c)
        display = display[[c for c in show_cols if c in display.columns]]

        col_rename = {
            "lineup_index": "#",
            "ricky_tag": "Tag",
            "ricky_rank": "Rank",
            "ricky_score": "Score",
            "total_gpp_score": "GPP Score",
            "total_ceil": "Ceiling",
            "total_proj": "Proj",
            "total_actual": "Actual",
            "avg_own_pct": "Avg Own%",
            "total_salary": "Salary",
            "diff": "Diff",
        }
        display = display.rename(columns=col_rename)

        fmt = {
            "Score": "{:.4f}",
            "GPP Score": "{:.1f}",
            "Ceiling": "{:.1f}",
            "Proj": "{:.1f}",
            "Actual": "{:.1f}",
            "Avg Own%": "{:.1f}",
            "Salary": "${:,.0f}",
            "Diff": "{:+.1f}",
        }
        fmt = {k: v for k, v in fmt.items() if k in display.columns}

        st.dataframe(
            display.style.format(fmt),
            use_container_width=True,
            hide_index=True,
        )


def _build_ranked_lineups_csv() -> Optional[pd.DataFrame]:
    """Build a combined CSV of all ranked lineups across the latest batch.

    Returns None if no batch data is available.
    """
    batches: List[Dict[str, Any]] = st.session_state.get("sim_lab_batches", [])
    if not batches:
        return None

    latest_batch = batches[-1]
    runs = latest_batch.get("runs", [])
    if not runs:
        return None

    all_rows: List[pd.DataFrame] = []
    for run in runs:
        run_summary = run.get("summary_df")
        if run_summary is None or run_summary.empty:
            continue
        chunk = run_summary.copy()
        chunk["date"] = run["date"]
        chunk["preset"] = run["preset"]
        chunk["archetype"] = run.get("archetype", "Default")
        chunk["config_hash"] = run.get("config_hash", "")
        all_rows.append(chunk)

    if not all_rows:
        return None

    combined = pd.concat(all_rows, ignore_index=True)

    # Reorder columns for export
    lead_cols = ["date", "preset", "archetype", "config_hash",
                 "lineup_index", "ricky_rank", "ricky_tag", "ricky_score"]
    data_cols = ["total_gpp_score", "total_ceil", "total_proj",
                 "total_actual", "avg_own_pct", "total_salary", "diff"]
    ordered = [c for c in lead_cols + data_cols if c in combined.columns]
    extra = [c for c in combined.columns if c not in ordered]
    return combined[ordered + extra]


def _render_download_button() -> None:
    """Render a 'Download Ricky Ranked Lineups' button."""
    csv_df = _build_ranked_lineups_csv()
    if csv_df is None:
        return

    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="\U0001f4e5 Download Ricky Ranked Lineups (CSV)",
        data=csv_bytes,
        file_name="ricky_ranked_lineups.csv",
        mime="text/csv",
        key="sim_lab_download_ricky_csv",
    )


# ---------------------------------------------------------------------------
# Calibration Nudge Guidance
# ---------------------------------------------------------------------------

# Path constants for calibration data files
_CALIB_FEEDBACK_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration_feedback"
_SLATE_ERRORS_PATH = _CALIB_FEEDBACK_DIR / "nba" / "slate_errors.json"
_RECALIB_BACKTEST_PATH = _CALIB_FEEDBACK_DIR / "recalibrated_backtest.json"

# Map profile_key → contest field value in recalibrated_backtest.json
_PROFILE_TO_BACKTEST_CONTEST: dict[str, str] = {
    "classic_gpp_main":  "gpp_main",
    "classic_gpp_20max": "gpp_main",
    "classic_gpp_se":    "gpp_main",
    "classic_cash":      "cash_main",
    "showdown_gpp":      "showdown",
    "showdown_cash":     "showdown",
}

# Players per lineup for each profile (used for ownership_sum calculation)
_PROFILE_PLAYERS_PER_LINEUP: dict[str, int] = {
    "classic_gpp_main":  8,
    "classic_gpp_20max": 8,
    "classic_gpp_se":    8,
    "classic_cash":      8,
    "showdown_gpp":      6,
    "showdown_cash":     6,
}


def _load_slate_errors() -> dict:
    """Load slate_errors.json; return {} on any error."""
    try:
        if _SLATE_ERRORS_PATH.is_file():
            return json.loads(_SLATE_ERRORS_PATH.read_text())
    except Exception:
        pass
    return {}


def _load_recalib_backtest() -> list:
    """Load recalibrated_backtest.json slates list; return [] on any error."""
    try:
        if _RECALIB_BACKTEST_PATH.is_file():
            data = json.loads(_RECALIB_BACKTEST_PATH.read_text())
            return data.get("slates", [])
    except Exception:
        pass
    return []


def _compute_nudge_metrics(
    batch: Dict[str, Any],
    profile_key: str,
) -> Dict[str, Optional[float]]:
    """Compute the nudge metric values for a completed batch.

    Returns a dict of metric_name → float | None (None = no data available).
    """
    runs = batch.get("runs", [])
    batch_dates = {r["date"] for r in runs}

    # ── 1. MAE and Correlation (slate_errors.json) ────────────────────────
    slate_errors = _load_slate_errors()
    mae_vals: list = []
    corr_vals: list = []
    for date_key, entry in slate_errors.items():
        if date_key in batch_dates:
            overall = entry.get("overall", {})
            if "mae" in overall:
                mae_vals.append(float(overall["mae"]))
            if "correlation" in overall:
                corr_vals.append(float(overall["correlation"]))

    metrics: Dict[str, Optional[float]] = {
        "mae": round(float(mean(mae_vals)), 3) if mae_vals else None,
        "correlation": round(float(mean(corr_vals)), 4) if corr_vals else None,
    }

    # ── 2. Bias (recalibrated_backtest.json) ─────────────────────────────
    backtest_slates = _load_recalib_backtest()
    contest_filter = _PROFILE_TO_BACKTEST_CONTEST.get(profile_key, "")
    bias_vals: list = []
    for entry in backtest_slates:
        if entry.get("date") in batch_dates and entry.get("contest") == contest_filter:
            if "corrected_bias" in entry:
                bias_vals.append(float(entry["corrected_bias"]))
    metrics["bias"] = round(float(mean(bias_vals)), 3) if bias_vals else None

    # ── 3. Avg Score (from batch run avg_actual) ──────────────────────────
    actuals = [r["avg_actual"] for r in runs if "avg_actual" in r]
    metrics["avg_score"] = round(float(mean(actuals)), 2) if actuals else None

    # ── 4. Ownership Sum ─────────────────────────────────────────────────
    players_per_lu = _PROFILE_PLAYERS_PER_LINEUP.get(profile_key, 8)
    own_sum_vals: list = []
    for run in runs:
        sdf = run.get("summary_df")
        if sdf is not None and not sdf.empty and "avg_own_pct" in sdf.columns:
            # avg_own_pct is per-player mean; multiply by roster size to get sum
            lu_own_sums = sdf["avg_own_pct"].dropna() * players_per_lu
            if not lu_own_sums.empty:
                own_sum_vals.append(float(lu_own_sums.mean()))
    metrics["ownership_sum"] = round(float(mean(own_sum_vals)), 2) if own_sum_vals else None

    # ── 5. Top-1% Hit Rate — requires contest results (not available yet) ─
    metrics["top_1pct_rate"] = None

    # ── 6. Cash Rate — requires contest results (cash configs only) ────────
    metrics["cash_rate"] = None

    # ── 7. Ricky Top-3 Lift % ──────────────────────────────────────────────
    # How much the Ricky-tagged top-3 lineups outperform the pool average.
    lift_vals: list = []
    hit_vals: list = []  # for ricky_top3_hit (metric 8)
    for run in runs:
        sdf = run.get("summary_df")
        if sdf is None or sdf.empty or "ricky_tag" not in sdf.columns:
            continue
        if "total_actual" not in sdf.columns:
            continue
        tagged = sdf[sdf["ricky_tag"] != ""]
        if tagged.empty:
            continue
        pool_avg = float(sdf["total_actual"].mean())
        top3_avg = float(tagged["total_actual"].mean())
        if pool_avg > 0:
            lift_vals.append(((top3_avg - pool_avg) / pool_avg) * 100.0)
        # Hit: did any Ricky pick land in the actual top 5?
        actual_top5_indices = set(sdf.nlargest(5, "total_actual")["lineup_index"].tolist())
        tagged_indices = set(tagged["lineup_index"].tolist())
        hit_vals.append(1.0 if tagged_indices & actual_top5_indices else 0.0)

    metrics["ricky_top3_lift"] = round(float(mean(lift_vals)), 2) if lift_vals else None

    # ── 8. Ricky Top-3 Hit Rate ────────────────────────────────────────────
    # Fraction of slates where ≥1 Ricky pick is in the actual top 5.
    metrics["ricky_top3_hit"] = round(float(mean(hit_vals)), 4) if hit_vals else None

    # ── 9. Ricky Rank Correlation (Spearman) ────────────────────────────────
    # Spearman rank correlation between ricky_rank and actual finish position.
    # A continuous 0-to-1 signal that stress-tests the ranker across the full
    # pool, unlike the binary ricky_top3_hit metric.
    rank_corr_vals: list = []
    for run in runs:
        sdf = run.get("summary_df")
        if sdf is None or sdf.empty or "ricky_rank" not in sdf.columns:
            continue
        if "total_actual" not in sdf.columns:
            continue
        # Need at least 4 lineups for a meaningful correlation
        if len(sdf) < 4:
            continue
        # Actual finish rank: rank 1 = highest total_actual
        actual_rank = sdf["total_actual"].rank(ascending=False, method="average")
        ricky_r = sdf["ricky_rank"].astype(float)
        # scipy-free Spearman: correlation of the two rank vectors
        n = len(sdf)
        d_sq = ((ricky_r.values - actual_rank.values) ** 2).sum()
        if n > 1:
            rho = 1.0 - (6.0 * d_sq) / (n * (n * n - 1.0))
            rank_corr_vals.append(float(rho))
    metrics["ricky_rank_corr"] = (
        round(float(mean(rank_corr_vals)), 4) if rank_corr_vals else None
    )

    # ── 10. Lineup Diversity (20-max only) — unique top-3 salary cores ────
    if profile_key == "classic_gpp_20max":
        unique_cores: set = set()
        for run in runs:
            player_df = run.get("player_df")
            if player_df is None or player_df.empty:
                continue
            if "lineup_index" not in player_df.columns:
                continue
            for lu_idx, group in player_df.groupby("lineup_index"):
                if "salary" in group.columns:
                    top3 = (
                        group.nlargest(3, "salary")["player_name"].tolist()
                        if "player_name" in group.columns
                        else []
                    )
                    if top3:
                        unique_cores.add(frozenset(top3))
        metrics["lineup_diversity_min_cores"] = float(len(unique_cores)) if unique_cores else None
    else:
        metrics["lineup_diversity_min_cores"] = None

    return metrics


def _fmt_nudge_value(metric_name: str, value: float) -> str:
    """Format a metric value for display in the nudge table."""
    if metric_name in ("top_1pct_rate", "cash_rate", "ricky_top3_hit"):
        return f"{value:.1%}"
    if metric_name in ("mae", "bias", "correlation", "ricky_rank_corr"):
        return f"{value:.3f}"
    if metric_name == "lineup_diversity_min_cores":
        return f"{int(value)} cores"
    if metric_name == "ricky_top3_lift":
        return f"{value:+.1f}%"
    return f"{value:.1f}"


def _render_nudge_guidance(
    batch: Dict[str, Any],
    sport: str,
    run_dates: List[date],
    preset_name: str,
    sandbox_overrides: Dict[str, Any],
    ricky_weights: Dict[str, float],
    archetype_name: str = "Default",
) -> None:
    """Render the Calibration Nudge Guidance expander for a completed batch.

    Shows a prescriptive metrics vs targets table with exact config.py
    parameters, computed deltas, and one-click Apply buttons (RIG-8).
    Includes a Re-run batch button.
    """
    from utils.calibration_targets import (
        CALIBRATION_TARGETS,
        METRIC_LABELS,
        evaluate_metric,
        get_target_display,
    )
    from utils.nudge_params import get_nudge_suggestions

    batches = st.session_state.get("sim_lab_batches", [])
    if not batches:
        return

    profile_key = batch.get("profile_name", "") or ""
    if not profile_key or profile_key not in CALIBRATION_TARGETS:
        return

    # Resolve preset defaults (merged config) for "current value" lookup
    preset_defaults: Dict[str, Any] = {}
    try:
        from yak_core.config import CONTEST_PRESETS, merge_config
        if preset_name in CONTEST_PRESETS:
            preset_defaults = merge_config(CONTEST_PRESETS[preset_name])
    except Exception:
        pass

    with st.expander("🎯 Calibration Nudge Guidance", expanded=False):
        st.caption(
            f"Prescriptive nudges for **{profile_key}** — "
            "click **Apply** to update the slider to the suggested value, "
            "then re-run the batch to validate."
        )

        metrics = _compute_nudge_metrics(batch, profile_key)
        targets = CALIBRATION_TARGETS[profile_key]

        # ── Compute off-target metrics for conflict detection ─────────
        off_target_metrics: Dict[str, str] = {}
        for _m_name, (_m_lo, _m_hi) in targets.items():
            _m_val = metrics.get(_m_name)
            if _m_val is None:
                continue
            if _m_val < _m_lo:
                off_target_metrics[_m_name] = "low"
            elif _m_val > _m_hi:
                off_target_metrics[_m_name] = "high"

        # ── Column header ────────────────────────────────────────────────
        _COL_W = [2.2, 1.2, 1.4, 0.6, 2.4, 1.2, 1.2, 0.8]
        hdr = st.columns(_COL_W)
        hdr[0].markdown("**Metric**")
        hdr[1].markdown("**Your Batch**")
        hdr[2].markdown("**Target Range**")
        hdr[3].markdown("**Status**")
        hdr[4].markdown("**Parameter**")
        hdr[5].markdown("**Current**")
        hdr[6].markdown("**Suggested**")
        hdr[7].markdown("**Apply**")
        st.divider()

        any_rows = False
        for metric_name, (lo, hi) in targets.items():
            label = METRIC_LABELS.get(metric_name, metric_name)
            value = metrics.get(metric_name)
            target_str = get_target_display(metric_name, profile_key)

            if value is None:
                row = st.columns(_COL_W)
                row[0].write(label)
                row[1].write("—")
                row[2].write(target_str)
                row[3].write("⚪")
                # No suggestion possible without data
                any_rows = True
                continue

            _status, dot, _nudge = evaluate_metric(metric_name, value, profile_key)
            value_str = _fmt_nudge_value(metric_name, value)

            # Get prescriptive suggestions for off-target metrics
            suggestions = get_nudge_suggestions(
                metric_name=metric_name,
                batch_value=value,
                lo=lo,
                hi=hi,
                current_overrides=sandbox_overrides,
                preset_defaults=preset_defaults,
                ricky_weights=ricky_weights,
                off_target_metrics=off_target_metrics,
            )

            if not suggestions:
                # On target or no mapping — simple status row
                row = st.columns(_COL_W)
                row[0].write(label)
                row[1].write(value_str)
                row[2].write(target_str)
                row[3].write(dot)
                any_rows = True
                continue

            # One sub-row per suggestion; metric info only on the first sub-row
            for i, sug in enumerate(suggestions):
                row = st.columns(_COL_W)
                if i == 0:
                    row[0].write(label)
                    row[1].write(value_str)
                    row[2].write(target_str)
                    row[3].write(dot)
                else:
                    row[0].write("")
                    row[1].write("")
                    row[2].write("")
                    row[3].write("")

                param = sug["param"]
                cur_val = sug["current_value"]
                sug_val = sug["suggested_value"]
                has_slider = sug["has_slider"]

                # Format param display: inline code + description as caption
                row[4].markdown(
                    f"`{param}`  \n"
                    f"<span style='font-size:0.75rem;color:#aaa'>{sug['description']}</span>",
                    unsafe_allow_html=True,
                )
                row[5].write(str(cur_val))

                if sug.get("clamped"):
                    row[6].caption("(guardrail)")

                if sug_val != cur_val:
                    delta = sug_val - cur_val
                    delta_str = f"{delta:+.0f}" if isinstance(sug_val, int) else f"{delta:+.2g}"
                    row[6].markdown(
                        f"**{sug_val}**",
                        help=f"Delta: {delta_str}",
                    )
                    storage = sug.get("storage", "sandbox")
                    apply_key = f"nudge_apply_{preset_name}_{metric_name}_{param}_{i}"
                    if row[7].button("Apply", key=apply_key, type="primary"):
                        if storage == "ricky_weights":
                            # Write to Ricky Ranking weights dict
                            _rk = f"sim_lab_ricky_weights_{preset_name}"
                            if _rk not in st.session_state:
                                st.session_state[_rk] = {}
                            st.session_state[_rk][param] = sug_val
                            if has_slider:
                                slider_key = f"sl_ricky_{param.replace('w_', '')}_{preset_name}"
                                st.session_state.pop(slider_key, None)
                        else:
                            # Write to sandbox config dict
                            sk = _sandbox_config_key(preset_name)
                            if sk not in st.session_state:
                                st.session_state[sk] = {}
                            st.session_state[sk][param] = sug_val
                            if has_slider:
                                slider_key = f"sl_{preset_name}_{param}"
                                st.session_state.pop(slider_key, None)
                        # Persist to disk so it survives page refreshes
                        _save_slider_state(
                            preset_name,
                            dict(st.session_state.get(_sandbox_config_key(preset_name), {})),
                            dict(st.session_state.get(f"sim_lab_ricky_weights_{preset_name}", {})),
                        )
                        st.toast(f"✅ {param} → {sug_val}")
                        st.rerun()
                    if sug.get("warning"):
                        row[7].caption(f"⚠️ {sug['warning']}")
                else:
                    row[6].write(str(sug_val))

                any_rows = True

        if not any_rows:
            st.caption("No metrics defined for this profile.")

        # ── Reset to Defaults + Re-run batch buttons ─────────────────────
        st.divider()
        col_reset, col_rerun = st.columns(2)
        with col_reset:
            if st.button("↩️ Reset to Defaults", key="nudge_reset_defaults", use_container_width=True):
                # Clear sandbox overrides
                st.session_state[_sandbox_config_key(preset_name)] = {}
                # Reset Ricky weights to defaults
                st.session_state[f"sim_lab_ricky_weights_{preset_name}"] = {
                    "w_gpp": 1.0, "w_ceil": 0.8, "w_own": 0.3,
                }
                # Pop slider widget keys so they re-render with defaults
                _keys_to_pop = [
                    k for k in list(st.session_state.keys())
                    if k.startswith(f"sl_{preset_name}_") or
                       (k.startswith("sl_ricky_") and k.endswith(f"_{preset_name}"))
                ]
                for k in _keys_to_pop:
                    st.session_state.pop(k, None)
                # Persist cleared state
                _save_slider_state(preset_name, {}, {"w_gpp": 1.0, "w_ceil": 0.8, "w_own": 0.3})
                st.toast("↩️ All parameters reset to defaults")
                st.rerun()
        with col_rerun:
            if st.button(
                "🔄 Re-run batch with current settings",
                key="nudge_rerun_batch",
                use_container_width=True,
            ):
                with st.spinner("Re-running batch..."):
                    new_batch = _run_batch(
                        sport,
                        preset_name,
                        sandbox_overrides,
                        run_dates,
                        archetype=archetype_name,
                        ricky_w_gpp=ricky_weights.get("w_gpp"),
                        ricky_w_ceil=ricky_weights.get("w_ceil"),
                        ricky_w_own=ricky_weights.get("w_own"),
                        profile_name=profile_key,
                    )
                    new_batch["overrides"] = dict(sandbox_overrides)
                    if "sim_lab_batches" not in st.session_state:
                        st.session_state["sim_lab_batches"] = []
                    st.session_state["sim_lab_batches"].append(new_batch)
                    _append_batch_history(new_batch)
                    _save_slider_state(preset_name, sandbox_overrides, ricky_weights)
                    st.success(
                        f"Re-run complete: {len(new_batch['runs'])} slates | "
                        f"Avg Actual: {new_batch['avg_actual']:.1f} FP"
                    )
                    st.rerun()


# ---------------------------------------------------------------------------
# Sim Lab Report (read-only analysis of CSV exports)
# ---------------------------------------------------------------------------

def _render_sim_lab_report() -> None:
    """Render a read-only report from Sim Lab CSV exports."""
    rank_summary, tag_summary, lineups_df = summarize_sim_lab(_HISTORY_DIR)

    if rank_summary is None:
        st.caption("No Sim Lab exports found yet — run a batch and download the CSV.")
        return

    # ── Rank Bucket Table ────────────────────────────────────────────────
    st.markdown("**Lineup Performance by Ricky Rank Bucket**")
    _fmt = {}
    for c in ["Avg Diff", "Med Diff"]:
        if c in rank_summary.columns:
            _fmt[c] = "{:+.1f}"
    for c in ["Avg Proj", "Avg Actual", "Avg GPP"]:
        if c in rank_summary.columns:
            _fmt[c] = "{:.1f}"
    st.dataframe(
        rank_summary.style.format(_fmt),
        use_container_width=True,
    )

    # ── Rank Bucket Bar Chart ────────────────────────────────────────────
    if "Avg Diff" in rank_summary.columns:
        chart_df = rank_summary[["Avg Diff"]].copy()
        if "Med Diff" in rank_summary.columns:
            chart_df["Med Diff"] = rank_summary["Med Diff"]
        st.bar_chart(chart_df)

    # ── Tag Summary Table ────────────────────────────────────────────────
    if tag_summary is not None and not tag_summary.empty:
        st.markdown("**SE Tag Performance by Date**")
        _tfmt = {}
        for c in ["Avg Diff", "Med Diff"]:
            if c in tag_summary.columns:
                _tfmt[c] = "{:+.1f}"
        for c in ["Avg Proj", "Avg Actual", "Avg Own%"]:
            if c in tag_summary.columns:
                _tfmt[c] = "{:.1f}"
        st.dataframe(
            tag_summary.style.format(_tfmt),
            use_container_width=True,
        )

    # ── Ownership vs Diff Scatter ────────────────────────────────────────
    if (lineups_df is not None
            and not lineups_df.empty
            and "avg_own_pct" in lineups_df.columns
            and "diff" in lineups_df.columns):
        st.markdown("**Ownership vs Diff**")
        _tag_col = "ricky_tag" if "ricky_tag" in lineups_df.columns else None

        scatter = lineups_df[["avg_own_pct", "diff"]].copy()
        if _tag_col:
            scatter["Tag"] = lineups_df[_tag_col].fillna("").replace("", "Untagged")
        else:
            scatter["Tag"] = "Untagged"
        scatter = scatter.rename(columns={"avg_own_pct": "Avg Own%", "diff": "Diff"})

        st.scatter_chart(
            scatter,
            x="Avg Own%",
            y="Diff",
            color="Tag",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Run Log
# ---------------------------------------------------------------------------

def _render_run_log() -> None:
    """Render the run log expander with per-run detail."""
    runs: List[Dict[str, Any]] = st.session_state.get("sim_lab_runs", [])
    if not runs:
        return

    with st.expander("Run Log", expanded=False):
        c1, c2 = st.columns([8, 2])
        with c2:
            if st.button("Clear Runs", key="sim_lab_clear_runs"):
                st.session_state["sim_lab_runs"] = []
                st.session_state["sim_lab_batches"] = []
                st.rerun()

        log_rows = []
        for i, r in enumerate(runs):
            log_rows.append({
                "#": i + 1,
                "Date": r["date"],
                "Preset": r["preset"],
                "Lineups": r["num_lineups"],
                "Avg Actual": r["avg_actual"],
                "Avg Proj": r["avg_proj"],
                "Best": r["best"],
                "Config": r["config_hash"],
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _apply_archetype(preset_name: str, archetype_name: str) -> None:
    """Seed slider overrides from an archetype when the user switches.

    Only writes to the sandbox config dict stored in session_state;
    the main optimizer config is never touched.
    """
    arch = _NBA_GPP_ARCHETYPES.get(archetype_name)
    if not arch:
        return

    sk = _sandbox_config_key(preset_name)
    if archetype_name == "Default":
        # Clear overrides — sliders revert to preset defaults
        st.session_state[sk] = {}
    else:
        # Seed overrides with archetype values
        st.session_state[sk] = dict(arch["overrides"])


def _apply_named_profile(profile_key: str) -> None:
    """Apply a named profile: set preset, seed slider overrides, set Ricky weights.

    Only writes to session_state sandbox config and Ricky weight keys.
    Never touches the main optimizer config or CONTEST_PRESETS.
    """
    profile = NAMED_PROFILES.get(profile_key)
    if not profile:
        # Check promoted configs
        from yak_core.promoted_configs import get_promoted_as_named_profile
        profile = get_promoted_as_named_profile(profile_key)
    if not profile:
        return
    preset_name = profile["base_preset"]
    sk = _sandbox_config_key(preset_name)
    st.session_state[sk] = dict(profile["overrides"])

    # Set Ricky weights
    ricky_key = f"sim_lab_ricky_weights_{preset_name}"
    st.session_state[ricky_key] = dict(profile["ricky_weights"])


def _render_auto_calibrate(
    preset_name: str,
    sandbox_overrides: Dict[str, Any],
    ricky_weights: Dict[str, float],
) -> None:
    """Render the Auto-Calibrate section in Sim Lab.

    Uses Optuna TPE to optimize all build + Ricky ranking parameters
    simultaneously, maximizing SE Core actual FP across RG archive slates.
    """
    st.markdown("---")
    st.subheader("Auto-Calibrate")
    st.caption(
        "Optimize ALL parameters simultaneously to maximize SE Core actual FP "
        "across historical slates. Replaces the manual nudge-apply loop."
    )

    # Settings in expander
    with st.expander("Settings"):
        n_trials = st.slider(
            "Optimization trials", 30, 120, 60, step=10,
            key="autocal_trials",
        )
        dates_per = st.slider(
            "Dates per trial (subsample)", 3, 8, 5,
            key="autocal_dates_per",
        )

    # Discover available dates from RG archive
    from yak_core.auto_calibrate import scan_rg_dates

    available_dates = scan_rg_dates()

    # Exclude today (games may be in progress)
    _today = date.today()
    available_dates = [d for d in available_dates if d != _today]

    if len(available_dates) < 3:
        st.warning(
            f"Need at least 3 historical slates in data/rg_archive/nba/. "
            f"Found {len(available_dates)}."
        )
        return

    st.caption(
        f"{len(available_dates)} historical slates available "
        f"({available_dates[-1]} to {available_dates[0]})"
    )

    # Run button
    if st.button("Auto-Calibrate", type="primary", key="auto_calibrate_btn"):
        progress = st.progress(0, text="Starting auto-calibration...")

        def _on_progress(trial_num: int, total: int, best_fp: float) -> None:
            progress.progress(
                trial_num / total,
                text=f"Trial {trial_num}/{total} — best SE Core: {best_fp:.1f} FP",
            )

        from yak_core.auto_calibrate import run_auto_calibration

        try:
            result = run_auto_calibration(
                preset_name=preset_name,
                dates=available_dates,
                current_ricky_weights=ricky_weights,
                current_overrides=sandbox_overrides,
                n_trials=n_trials,
                dates_per_trial=dates_per,
                progress_callback=_on_progress,
            )
            progress.empty()
            st.session_state["autocal_result"] = result
        except Exception as exc:
            progress.empty()
            st.error(f"Auto-calibration failed: {exc}")
            return

    # Display results
    result = st.session_state.get("autocal_result")
    if result is None:
        return

    # ── Summary metrics ──────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Baseline SE Core", f"{result.baseline_score:.1f} FP")
    with c2:
        st.metric(
            "Optimized SE Core",
            f"{result.best_score:.1f} FP",
            delta=f"{result.improvement_fp:+.1f} FP",
        )
    with c3:
        st.metric("Improvement", f"{result.improvement_pct:+.1f}%")

    # ── Parameter comparison table ───────────────────────────────────
    st.markdown("#### Parameter Changes")
    rows = []
    all_new = {**result.best_params, **result.best_ricky_weights}
    for key, new_val in all_new.items():
        if key in result.best_ricky_weights:
            old_val = ricky_weights.get(key, "—")
        else:
            old_val = sandbox_overrides.get(
                key, _slider_default(preset_name, key, "—"),
            )
        delta = ""
        if isinstance(old_val, (int, float)):
            delta = f"{new_val - float(old_val):+.2f}"
        rows.append({
            "Parameter": key,
            "Current": old_val,
            "Optimized": new_val,
            "Delta": delta or "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Per-date breakdown ───────────────────────────────────────────
    with st.expander("Per-Date Breakdown"):
        date_rows = []
        for base, opt in zip(result.baseline_per_date, result.per_date_results):
            b_fp = base["se_core_actual"]
            o_fp = opt["se_core_actual"]
            date_rows.append({
                "Date": base["date"],
                "Baseline FP": f"{b_fp:.1f}" if b_fp else "—",
                "Optimized FP": f"{o_fp:.1f}" if o_fp else "—",
                "Delta": f"{o_fp - b_fp:+.1f}" if (b_fp and o_fp) else "—",
            })
        st.dataframe(
            pd.DataFrame(date_rows), hide_index=True, use_container_width=True,
        )

    # ── Accept / Reject ──────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "Accept — Apply to Config", type="primary", key="autocal_accept",
        ):
            # Write optimized params into sandbox overrides
            sk = _sandbox_config_key(preset_name)
            if sk not in st.session_state:
                st.session_state[sk] = {}
            for key, val in result.best_params.items():
                st.session_state[sk][key] = val
            # Write Ricky weights
            ricky_key = f"sim_lab_ricky_weights_{preset_name}"
            st.session_state[ricky_key] = dict(result.best_ricky_weights)
            # Persist to disk
            _save_slider_state(
                preset_name,
                st.session_state[sk],
                result.best_ricky_weights,
            )
            st.success(
                "Optimized parameters applied to config sliders and Ricky weights."
            )
            st.rerun()
    with c2:
        if st.button("Reject — Keep Current", key="autocal_reject"):
            del st.session_state["autocal_result"]
            st.rerun()


def render_sim_lab(sport: str) -> None:
    """Render the Sim Lab tab."""
    st.header("\U0001f52c Sim Lab")

    is_pga = sport != "NBA"

    if is_pga:
        # PGA: single contest-type dropdown
        _pga_display = st.selectbox(
            "Contest type", PGA_CONTEST_TYPES,
            key="sim_lab_pga_contest_type",
        )
        preset_name = PGA_DISPLAY_TO_PRESET.get(_pga_display, _pga_display)
        _profile_key_internal: str | None = None
        _active_profile = None
        # Detect change → reset archetype
        _prev_ct = st.session_state.get("_sim_lab_prev_contest_type", "")
        _cur_ct = _pga_display
    else:
        # NBA: two-level dropdown — Game Style → Contest Type
        col_style, col_ct = st.columns([2, 3])
        with col_style:
            _game_style = st.selectbox(
                "Game Style", NBA_GAME_STYLES,
                key="sim_lab_game_style",
            )
        # Reset contest type when game style changes
        _prev_style = st.session_state.get("_sim_lab_prev_game_style", "")
        if _game_style != _prev_style:
            st.session_state["_sim_lab_prev_game_style"] = _game_style
            st.session_state.pop("sim_lab_contest_type", None)
        _ct_options = NBA_CONTEST_TYPES_BY_STYLE[_game_style]
        with col_ct:
            _contest_display = st.selectbox(
                "Contest Type", _ct_options,
                key="sim_lab_contest_type",
            )
        _profile_key_internal = CONTEST_PROFILE_KEY_MAP[(_game_style, _contest_display)]
        preset_name = PROFILE_KEY_TO_PRESET[_profile_key_internal]

        # Auto-wire the named profile (hidden from UI)
        _named_key = PROFILE_KEY_TO_NAMED.get(_profile_key_internal)
        _active_profile = None
        if _named_key and _named_key in NAMED_PROFILES:
            _active_profile = NAMED_PROFILES[_named_key]

        _cur_ct = _contest_display
        _prev_ct = st.session_state.get("_sim_lab_prev_contest_type", "")

    # Detect contest type change → reset archetype, seed sliders
    if _cur_ct != _prev_ct:
        st.session_state["_sim_lab_prev_contest_type"] = _cur_ct
        st.session_state["sim_lab_archetype"] = "Default"
        # Seed sandbox with profile defaults on contest type change
        if not is_pga and _named_key:
            _apply_named_profile(_named_key)
        if _prev_ct:  # skip initial render
            st.rerun()
    else:
        # Normal render (same contest type): restore from disk if session empty
        sk = _sandbox_config_key(preset_name)
        _rk = f"sim_lab_ricky_weights_{preset_name}"
        if sk not in st.session_state:
            _disk_ovr, _disk_rw = _load_slider_state(preset_name)
            if _disk_ovr:
                st.session_state[sk] = _disk_ovr
            elif not is_pga and _named_key:
                _apply_named_profile(_named_key)
            if _disk_rw and _rk not in st.session_state:
                st.session_state[_rk] = _disk_rw

    # --- Archetype selector (NBA GPP presets only) ---
    archetype_name = "Default"
    if sport == "NBA" and preset_name in _ARCHETYPE_ELIGIBLE_PRESETS:
        archetype_name = st.selectbox(
            "GPP Archetype",
            options=_NBA_GPP_ARCHETYPE_NAMES,
            key="sim_lab_archetype",
            help=_NBA_GPP_ARCHETYPES.get(
                st.session_state.get("sim_lab_archetype", "Default"), {}
            ).get("description", ""),
        )

        # Detect archetype change and seed sliders
        prev_arch = st.session_state.get("_sim_lab_prev_archetype", "Default")
        if archetype_name != prev_arch:
            _apply_archetype(preset_name, archetype_name)
            st.session_state["_sim_lab_prev_archetype"] = archetype_name
            st.rerun()

    # Config panel (4 groups) — sliders read from sandbox overrides
    sandbox_overrides = _render_config_panel(preset_name)

    # --- Ricky Ranking Weights (local to Sim Lab, per-contest-type) ---
    _ricky_key = f"sim_lab_ricky_weights_{preset_name}"
    if _ricky_key not in st.session_state:
        st.session_state[_ricky_key] = {
            "w_gpp": RICKY_W_GPP, "w_ceil": RICKY_W_CEIL, "w_own": RICKY_W_OWN,
        }
    with st.expander("Ricky Ranking Weights"):
        _rw = st.session_state[_ricky_key]
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            _rw["w_gpp"] = st.slider(
                "GPP Score", 0.0, 2.0, float(_rw["w_gpp"]), 0.05,
                key=f"sl_ricky_gpp_{preset_name}", format="%.2f",
            )
        with rc2:
            _rw["w_ceil"] = st.slider(
                "Ceiling", 0.0, 2.0, float(_rw["w_ceil"]), 0.05,
                key=f"sl_ricky_ceil_{preset_name}", format="%.2f",
            )
        with rc3:
            _rw["w_own"] = st.slider(
                "Own Penalty", 0.0, 2.0, float(_rw["w_own"]), 0.05,
                key=f"sl_ricky_own_{preset_name}", format="%.2f",
            )
        st.session_state[_ricky_key] = _rw

        _rerank_clicked = st.button(
            "\U0001f504 Re-rank Lineups", key="sim_lab_rerank",
            use_container_width=True,
        )

    _ricky_weights = st.session_state[_ricky_key]

    if sport == "NBA":
        # --- NBA: Batch run across RG dates ---
        rg_dates = _scan_rg_dates()
        _run_dates: List[date] = []

        if rg_dates:
            st.caption(f"{len(rg_dates)} RG archive dates available")

            # Date selection: Run All vs Select Dates
            _date_mode = st.radio(
                "Date selection",
                ["Run All", "Select Dates"],
                horizontal=True,
                key="sim_lab_date_mode",
            )
            # Date guard: today's games may still be in progress, so
            # actuals will be incomplete.  Auto-exclude from batch runs.
            _today_date = date.today()
            _safe_dates = [d for d in rg_dates if d != _today_date]
            _excluded_today = len(_safe_dates) < len(rg_dates)
            if _excluded_today:
                st.caption(
                    f"\u26a0\ufe0f Today ({_today_date.strftime('%Y-%m-%d')}) excluded from batch "
                    "(games may be in progress). Use the Optimizer for today's lineups."
                )

            if _date_mode == "Select Dates":
                _selected_dates = st.multiselect(
                    "Dates",
                    options=[d.strftime("%Y-%m-%d") for d in _safe_dates],
                    default=[d.strftime("%Y-%m-%d") for d in _safe_dates],
                    key="sim_lab_selected_dates",
                )
                _run_dates = [
                    d for d in _safe_dates
                    if d.strftime("%Y-%m-%d") in _selected_dates
                ]
            else:
                _run_dates = _safe_dates

            if st.button(
                "\U0001f504 Run Batch", use_container_width=True,
                key="sim_lab_batch_run",
                disabled=len(_run_dates) == 0 if _date_mode == "Select Dates" else False,
            ):
                # ── Auto-run baseline (main config, no overrides) on first batch ──
                # Only if there is no active baseline for this preset yet.
                active_bl = _get_active_baseline(preset_name)
                if active_bl is None:
                    with st.spinner("Running baseline (main config)..."):
                        _bl_batch = _run_batch(
                            sport, preset_name, {}, _run_dates,
                            archetype="Default",
                            ricky_w_gpp=_ricky_weights["w_gpp"],
                            ricky_w_ceil=_ricky_weights["w_ceil"],
                            ricky_w_own=_ricky_weights["w_own"],
                            profile_name=_profile_key_internal or "",
                        )
                        _bl_batch["config_label"] = "Baseline (main config)"
                        _bl_batch["overrides"] = {}

                        # Also add to session batches so trend chart sees it
                        if "sim_lab_batches" not in st.session_state:
                            st.session_state["sim_lab_batches"] = []
                        st.session_state["sim_lab_batches"].insert(0, _bl_batch)

                        # Persist as baseline
                        _append_batch_history(_bl_batch, is_baseline=True)

                # ── Run the user's tweaked config ────────────────────────────
                with st.spinner("Running batch..."):
                    batch = _run_batch(
                        sport, preset_name, sandbox_overrides, _run_dates,
                        archetype=archetype_name,
                        ricky_w_gpp=_ricky_weights["w_gpp"],
                        ricky_w_ceil=_ricky_weights["w_ceil"],
                        ricky_w_own=_ricky_weights["w_own"],
                        profile_name=_profile_key_internal or "",
                    )
                    batch["overrides"] = dict(sandbox_overrides)

                    if "sim_lab_batches" not in st.session_state:
                        st.session_state["sim_lab_batches"] = []
                    st.session_state["sim_lab_batches"].append(batch)

                    # Persist to disk + GitHub
                    _append_batch_history(batch)
                    _save_slider_state(preset_name, sandbox_overrides, _ricky_weights)

                    n_ok = len(batch["runs"])
                    n_err = len(batch["errors"])
                    st.success(
                        f"Batch complete: {n_ok} slates processed, "
                        f"{n_err} errors | Avg Actual: {batch['avg_actual']:.1f} FP"
                    )
                    if batch["errors"]:
                        with st.expander("Batch Errors"):
                            for err in batch["errors"]:
                                st.warning(f"{err['date']}: {err['error']}")

            # ── Manual baseline promotion button (outside batch run) ────────
            # Rendered in the comparison table form below.
        else:
            st.warning("No RG archive files found in data/rg_archive/nba/")

        # --- Re-rank existing batch when slider button clicked ---
        if _rerank_clicked:
            batches = st.session_state.get("sim_lab_batches", [])
            if batches:
                latest = batches[-1]
                for run in latest.get("runs", []):
                    sdf = run.get("summary_df")
                    if sdf is not None and not sdf.empty:
                        # Drop old ranking cols so rank_lineups_for_se re-creates them
                        for _drop_col in ("ricky_score", "ricky_rank", "ricky_tag"):
                            if _drop_col in sdf.columns:
                                sdf.drop(columns=[_drop_col], inplace=True)
                        run["summary_df"] = rank_lineups_for_se(
                            sdf,
                            w_gpp=_ricky_weights["w_gpp"],
                            w_ceil=_ricky_weights["w_ceil"],
                            w_own=_ricky_weights["w_own"],
                        )
                _save_slider_state(preset_name, sandbox_overrides, _ricky_weights)
                st.toast("Lineups re-ranked with updated weights")

        # Ricky SE Shortlist (tagged lineups from latest batch)
        _render_ricky_shortlist()

        # Per-slate lineup detail with ricky_rank/tag
        _render_per_slate_detail()

        # Download CSV of all ranked lineups
        _render_download_button()

        # Calibration Nudge Guidance (after batch results)
        _batches_now = st.session_state.get("sim_lab_batches", [])
        if _batches_now:
            _render_nudge_guidance(
                _batches_now[-1],
                sport,
                _run_dates,
                preset_name,
                sandbox_overrides,
                _ricky_weights,
                archetype_name,
            )

        # Auto-Calibrate (Optuna optimization — below nudge guidance)
        _render_auto_calibrate(preset_name, sandbox_overrides, _ricky_weights)

        # Promote Config button
        _render_promote_config(preset_name, sandbox_overrides, _ricky_weights, archetype_name)

        # Trend chart — session per-date detail (when batches exist in memory)
        _render_trend_chart()

        # Persistent performance trend — survives restarts, shows baseline ref line
        _render_persistent_trend(preset_name)

        # Comparison table (persistent, with baseline pinning + checkboxes)
        _render_comparison_table(preset_name)

        # Persistent history (survives restarts)
        _render_history_table()

    else:
        # --- PGA: single date flow (no RG archive) ---
        selected_date = st.date_input(
            "Date",
            value=date.today() - timedelta(days=1),
            key="sim_lab_date_pga",
        )

        if st.button("\U0001f504 Fetch & Build", use_container_width=True, key="sim_lab_pga_run"):
            with st.spinner("Fetching pool, building lineups, scoring..."):
                try:
                    run = _run_pipeline(sport, selected_date, preset_name, sandbox_overrides)
                    if "sim_lab_runs" not in st.session_state:
                        st.session_state["sim_lab_runs"] = []
                    st.session_state["sim_lab_runs"].append(run)
                    st.session_state["sim_lab_latest_pga_run"] = run
                except ValueError as exc:
                    st.warning(str(exc))
                except RuntimeError as exc:
                    st.error(f"API error: {exc}")
                except Exception as exc:
                    msg = str(exc)
                    if "in progress" in msg.lower() or "not final" in msg.lower():
                        st.warning("Games may still be in progress. Actuals may be incomplete.")
                    else:
                        st.error(f"Pipeline error: {msg}")

        # Display latest PGA run results
        latest_pga = st.session_state.get("sim_lab_latest_pga_run")
        if latest_pga:
            summary = latest_pga["summary_df"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Actual FP", f"{latest_pga['avg_actual']:.1f}")
            k2.metric("Avg Proj FP", f"{latest_pga['avg_proj']:.1f}")
            k3.metric("Best Lineup", f"{latest_pga['best']:.1f}")
            k4.metric("Beat Proj %", f"{latest_pga['beat_proj_pct']:.0f}%")

            st.subheader("Lineup Scores")
            display = summary.copy()
            display.index = display.index + 1
            display.index.name = "#"
            display = display[["total_actual", "total_proj", "diff", "total_salary"]]
            display.columns = ["Total Actual", "Total Proj", "Diff", "Salary"]
            st.dataframe(
                display.style.format(
                    {"Total Actual": "{:.1f}", "Total Proj": "{:.1f}", "Diff": "{:+.1f}", "Salary": "${:,.0f}"}
                ),
                use_container_width=True,
            )

    # Sim Lab Report (read-only analysis of CSV exports)
    with st.expander("Sim Lab Report", expanded=False):
        _render_sim_lab_report()

    # Run log (always visible at bottom)
    _render_run_log()
