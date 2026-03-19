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

from yak_core.config import CONTEST_PRESETS, DEFAULT_CONFIG, merge_config
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
# Batch history persistence
# ---------------------------------------------------------------------------

def _append_batch_history(batch: Dict[str, Any]) -> None:
    """Append a batch summary row to the persistent history file.

    Writes to data/sim_lab/batch_history.parquet and syncs to GitHub
    so the history survives Streamlit Cloud restarts.
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
        "num_dates": len(successful_runs) + len(batch.get("errors", [])),
        "num_lineups": successful_runs[0].get("num_lineups", 0) if successful_runs else 0,
        "avg_actual": batch.get("avg_actual", 0.0),
        "avg_proj": batch.get("avg_proj", 0.0),
        "best_slate": best_slate_fp,
        "worst_slate": worst_slate_fp,
        "beat_proj_pct": batch.get("beat_proj_pct", 0.0),
        "errors": len(batch.get("errors", [])),
    }

    new_df = pd.DataFrame([row])

    try:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        if _HISTORY_FILE.is_file():
            existing = pd.read_parquet(_HISTORY_FILE)
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


def _render_history_table() -> None:
    """Render the persistent batch history table."""
    history = _load_batch_history()
    if history.empty:
        st.caption("No batch history yet — run a batch to start tracking.")
        return

    st.subheader("Batch History")

    # Sort newest first
    if "timestamp" in history.columns:
        history = history.sort_values("timestamp", ascending=False).reset_index(drop=True)

    display = history.copy()

    # Format timestamp for readability
    if "timestamp" in display.columns:
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%m/%d %I:%M %p")

    col_rename = {
        "timestamp": "When",
        "preset": "Preset",
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

    # Step 6: Build lineups
    prepped = prepare_pool(edge_df, cfg)

    is_showdown = "showdown" in preset_name.lower()
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
        "overrides": sandbox_overrides.copy(),
        "runs": runs,
        "errors": errors,
        "avg_actual": avg_actual,
        "avg_proj": avg_proj,
        "beat_proj_pct": beat_proj_pct,
    }


# ---------------------------------------------------------------------------
# Trend Chart (Chart.js — dark mode)
# ---------------------------------------------------------------------------

def _render_trend_chart() -> None:
    """Render a Chart.js line chart: X = date, Y = avg actual FP, one line per batch."""
    batches: List[Dict[str, Any]] = st.session_state.get("sim_lab_batches", [])
    if not batches:
        return

    st.subheader("Config Comparison — Avg Actual FP by Date")

    # Build datasets for Chart.js
    datasets_js = []
    for i, batch in enumerate(batches):
        color = _BATCH_COLORS[i % len(_BATCH_COLORS)]
        # Sort runs by date
        sorted_runs = sorted(batch["runs"], key=lambda r: r["date"])
        data_points = []
        for run in sorted_runs:
            data_points.append({"x": run["date"], "y": run["avg_actual"]})

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


# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------

def _render_comparison_table() -> None:
    """Render a summary table comparing all batch runs."""
    batches: List[Dict[str, Any]] = st.session_state.get("sim_lab_batches", [])
    if not batches:
        return

    st.subheader("Batch Comparison")
    rows = []
    for batch in batches:
        successful_runs = batch["runs"]
        best_slate = ""
        worst_slate = ""
        if successful_runs:
            best_run = max(successful_runs, key=lambda r: r["avg_actual"])
            worst_run = min(successful_runs, key=lambda r: r["avg_actual"])
            best_slate = f"{best_run['date']} ({best_run['avg_actual']:.1f})"
            worst_slate = f"{worst_run['date']} ({worst_run['avg_actual']:.1f})"

        rows.append({
            "Run": batch["config_label"],
            "Archetype": batch.get("archetype", "Default"),
            "Config Hash": batch["config_hash"],
            "Avg Actual": batch["avg_actual"],
            "Avg Proj": batch["avg_proj"],
            "Best Slate": best_slate,
            "Worst Slate": worst_slate,
            "Beat Proj %": f"{batch['beat_proj_pct']:.0f}%",
            "Errors": len(batch["errors"]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({"Avg Actual": "{:.1f}", "Avg Proj": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )


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


def render_sim_lab(sport: str) -> None:
    """Render the Sim Lab tab."""
    st.header("\U0001f52c Sim Lab")

    # Contest preset selector
    presets = _NBA_PRESETS if sport == "NBA" else _PGA_PRESETS
    preset_name = st.selectbox(
        "Contest Preset",
        options=presets,
        key="sim_lab_preset",
    )

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

        if rg_dates:
            st.caption(f"{len(rg_dates)} RG archive dates available")

            # Date selection: Run All vs Select Dates
            _date_mode = st.radio(
                "Date selection",
                ["Run All", "Select Dates"],
                horizontal=True,
                key="sim_lab_date_mode",
            )
            if _date_mode == "Select Dates":
                _selected_dates = st.multiselect(
                    "Dates",
                    options=[d.strftime("%Y-%m-%d") for d in rg_dates],
                    default=[d.strftime("%Y-%m-%d") for d in rg_dates],
                    key="sim_lab_selected_dates",
                )
                _run_dates = [
                    d for d in rg_dates
                    if d.strftime("%Y-%m-%d") in _selected_dates
                ]
            else:
                _run_dates = rg_dates

            if st.button(
                "\U0001f504 Run Batch", use_container_width=True,
                key="sim_lab_batch_run",
                disabled=len(_run_dates) == 0 if _date_mode == "Select Dates" else False,
            ):
                with st.spinner("Running batch..."):
                    batch = _run_batch(
                        sport, preset_name, sandbox_overrides, _run_dates,
                        archetype=archetype_name,
                        ricky_w_gpp=_ricky_weights["w_gpp"],
                        ricky_w_ceil=_ricky_weights["w_ceil"],
                        ricky_w_own=_ricky_weights["w_own"],
                    )

                    if "sim_lab_batches" not in st.session_state:
                        st.session_state["sim_lab_batches"] = []
                    st.session_state["sim_lab_batches"].append(batch)

                    # Persist to disk + GitHub
                    _append_batch_history(batch)

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
                st.toast("Lineups re-ranked with updated weights")

        # Ricky SE Shortlist (tagged lineups from latest batch)
        _render_ricky_shortlist()

        # Per-slate lineup detail with ricky_rank/tag
        _render_per_slate_detail()

        # Download CSV of all ranked lineups
        _render_download_button()

        # Trend chart (if batches exist)
        _render_trend_chart()

        # Comparison table
        _render_comparison_table()

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

    # Run log (always visible at bottom)
    _render_run_log()
