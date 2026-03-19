"""Sim Lab v2 — Config Tuning + Batch Replay.

Pick a contest preset, adjust knobs across 4 tuning groups,
batch-run all available RG archive dates, and compare configs
via trend chart + summary table.  Session-state only, no persistence.
"""
from __future__ import annotations

import hashlib
import json
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NBA_PRESETS = ["GPP Main", "GPP Early", "GPP Late", "Showdown", "Cash Main", "Cash Game"]
_PGA_PRESETS = ["PGA GPP", "PGA Cash", "PGA Showdown"]

_BATCH_COLORS = ["#4dabf7", "#00ff87", "#ffa726", "#ef5350", "#ab47bc", "#26c6da", "#d4e157", "#ff7043"]


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


def _config_hash(overrides: Dict[str, Any]) -> str:
    return hashlib.md5(
        json.dumps(overrides, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


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
    st.info(
        f"RG merge: {n_merged}/{len(pool)} players matched "
        f"({n_missing} unmatched) | {len(rg)} rows in CSV | "
        f"FPTS range {rg_fpts_range}"
    )
    if n_merged == 0:
        st.warning(
            "No players matched from RG file. Projections will use "
            "YakOS model values instead of RotoGrinders."
        )
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
            st.warning(f"No RG archive file for {date_dash}. Using Tank01 projections.")

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
            st.info(f"Auto-ran {_n_sims} player sims — sim columns populated")
        except Exception as _sim_err:
            st.warning(f"Auto-sim failed ({_sim_err}), continuing with fallback estimates")

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
    st.write(f"**{n_lineups} lineups generated**")

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

    summary = (
        scored.groupby("lineup_index")
        .agg(
            total_actual=("actual_fp", "sum"),
            total_proj=("proj", "sum"),
            total_salary=("salary", "sum"),
        )
        .reset_index()
    )
    summary["diff"] = summary["total_actual"] - summary["total_proj"]
    summary = summary.sort_values("total_actual", ascending=False).reset_index(drop=True)

    beat_proj_pct = 0.0
    if len(summary) > 0:
        beat_proj_pct = float((summary["total_actual"] >= summary["total_proj"]).mean() * 100)

    chash = _config_hash(sandbox_overrides)

    return {
        "timestamp": datetime.now().isoformat(),
        "date": str(selected_date),
        "preset": preset_name,
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
            run = _run_pipeline(sport, d, preset_name, sandbox_overrides)
            runs.append(run)
            # Also store in the flat run log
            if "sim_lab_runs" not in st.session_state:
                st.session_state["sim_lab_runs"] = []
            st.session_state["sim_lab_runs"].append(run)
        except Exception as exc:
            errors.append({"date": str(d), "error": str(exc)})

    progress.empty()

    chash = _config_hash(sandbox_overrides)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine batch number
    batches = st.session_state.get("sim_lab_batches", [])
    batch_number = len(batches) + 1

    avg_actual = round(mean(r["avg_actual"] for r in runs), 2) if runs else 0
    avg_proj = round(mean(r["avg_proj"] for r in runs), 2) if runs else 0
    beat_proj_pct = round(mean(r["beat_proj_pct"] for r in runs), 1) if runs else 0

    return {
        "batch_id": f"{chash}_{timestamp}",
        "preset": preset_name,
        "config_hash": chash,
        "config_label": f"Run {batch_number}",
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

    # Config panel (4 groups)
    sandbox_overrides = _render_config_panel(preset_name)

    if sport == "NBA":
        # --- NBA: Batch run across all RG dates ---
        rg_dates = _scan_rg_dates()

        if rg_dates:
            st.caption(f"{len(rg_dates)} RG archive dates available")

            if st.button("\U0001f504 Run All Dates", use_container_width=True, key="sim_lab_batch_run"):
                with st.spinner("Running batch..."):
                    batch = _run_batch(sport, preset_name, sandbox_overrides, rg_dates)

                    if "sim_lab_batches" not in st.session_state:
                        st.session_state["sim_lab_batches"] = []
                    st.session_state["sim_lab_batches"].append(batch)

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

        # Trend chart (if batches exist)
        _render_trend_chart()

        # Comparison table
        _render_comparison_table()

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
