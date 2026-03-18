"""Sim Lab — contest replay sandbox.

Fetch pool → build lineups → score against actuals → display results.
Single file, no wizards, no persistence.  Session-state only.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from yak_core.config import CONTEST_PRESETS, merge_config
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
    # Normalise columns the optimizer expects
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

    # Find event closest to (and <=) the selected date
    target = slate_date.replace("-", "") if slate_date else ""
    chosen = events.iloc[0]  # default: most recent
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
# Scatter Plot (Chart.js)
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
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    sport: str,
    selected_date: date,
    preset_name: str,
    sandbox_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the full fetch → build → score pipeline. Returns a run dict."""
    date_key = selected_date.strftime("%Y%m%d")
    date_dash = selected_date.strftime("%Y-%m-%d")
    preset = CONTEST_PRESETS[preset_name]
    cfg = merge_config(preset)
    cfg.update(sandbox_overrides)

    # Ensure NUM_LINEUPS is set from sandbox or preset default
    if "NUM_LINEUPS" not in cfg or cfg["NUM_LINEUPS"] <= 0:
        cfg["NUM_LINEUPS"] = preset.get("default_lineups", 10)

    # ── Step 1: Fetch pool ────────────────────────────────────────────
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

    # ── Step 2: Merge RG projections (NBA only) ──────────────────────
    if sport == "NBA":
        rg_path = _RG_ARCHIVE_DIR / f"rg_{date_dash}.csv"
        if rg_path.is_file():
            pool_df = _merge_rg_csv(pool_df, rg_path)
        else:
            st.warning(f"No RG archive file for {date_dash}. Using Tank01 projections.")

    # ── Step 3: Auto-run Monte Carlo sims (if sim columns missing) ───
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

    # ── Step 4: Compute edge ──────────────────────────────────────────
    edge_df = compute_edge_metrics(pool_df, calibration_state=None, sport=sport)

    # ── Step 5: Fetch actuals ─────────────────────────────────────────
    if sport == "NBA":
        actuals_df = fetch_actuals_from_api(date_key, cfg)
    else:
        actuals_df = _fetch_pga_actuals(api_key, date_dash)

    # ── Step 6: Build lineups ─────────────────────────────────────────
    prepped = prepare_pool(edge_df, cfg)

    is_showdown = "showdown" in preset_name.lower()
    if is_showdown:
        lineups_df, _ = build_showdown_lineups(prepped, cfg)
    else:
        lineups_df, _ = build_multiple_lineups_with_exposure(prepped, cfg)

    if lineups_df is None or lineups_df.empty:
        raise ValueError("Optimizer returned no lineups. Try adjusting config.")

    # ── Step 7: Score lineups ─────────────────────────────────────────
    # Normalise name columns for merge
    if "player_name" not in actuals_df.columns:
        # PGA actuals may use different column name
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
    # Use _actual column if merge created one, else use actual_fp
    if "actual_fp_actual" in scored.columns:
        scored["actual_fp"] = scored["actual_fp_actual"].fillna(scored.get("actual_fp", 0.0))
        scored.drop(columns=["actual_fp_actual"], inplace=True)
    scored["actual_fp"] = pd.to_numeric(scored["actual_fp"], errors="coerce").fillna(0.0)

    # Ensure lineup_index exists
    if "lineup_index" not in scored.columns:
        if "lineup_id" in scored.columns:
            scored["lineup_index"] = scored["lineup_id"]
        else:
            scored["lineup_index"] = 0

    # Per-lineup summary
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

    # ── Build run record ────────────────────────────────────────────
    beat_proj_pct = 0.0
    if len(summary) > 0:
        beat_proj_pct = float((summary["total_actual"] >= summary["total_proj"]).mean() * 100)

    config_hash = hashlib.md5(
        json.dumps(sandbox_overrides, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]

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
        "config_hash": config_hash,
        "summary_df": summary,
        "player_df": scored,
    }


# ---------------------------------------------------------------------------
# Config expander
# ---------------------------------------------------------------------------

def _render_config_expander(preset_name: str) -> Dict[str, Any]:
    """Render sandbox config sliders. Returns the current overrides dict."""
    preset = CONTEST_PRESETS[preset_name]
    sk = _sandbox_config_key(preset_name)

    if sk not in st.session_state:
        st.session_state[sk] = {}
    overrides: Dict[str, Any] = st.session_state[sk]

    with st.expander("Config \u2699\ufe0f", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            nl = st.slider(
                "Lineups",
                min_value=1,
                max_value=20,
                value=overrides.get("NUM_LINEUPS", preset.get("default_lineups", 10)),
                key=f"sim_nl_{preset_name}",
            )
            overrides["NUM_LINEUPS"] = nl

            me = st.slider(
                "Max Exposure",
                min_value=0.1,
                max_value=1.0,
                value=overrides.get("MAX_EXPOSURE", preset.get("default_max_exposure", 0.6)),
                step=0.05,
                key=f"sim_me_{preset_name}",
            )
            overrides["MAX_EXPOSURE"] = me

        with c2:
            ms = st.slider(
                "Min Salary",
                min_value=0,
                max_value=50000,
                value=overrides.get("MIN_SALARY_USED", preset.get("min_salary", 49000)),
                step=500,
                key=f"sim_ms_{preset_name}",
            )
            overrides["MIN_SALARY_USED"] = ms

            un = st.slider(
                "Min Uniques",
                min_value=0,
                max_value=5,
                value=overrides.get("MIN_UNIQUES", 0),
                key=f"sim_un_{preset_name}",
            )
            overrides["MIN_UNIQUES"] = un

    st.session_state[sk] = overrides
    return overrides


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _render_results(run: Dict[str, Any]) -> None:
    """Display KPIs, lineup table, and scatter plot for a completed run."""
    summary = run["summary_df"]
    player_df = run["player_df"]

    # ── KPIs ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Actual FP", f"{run['avg_actual']:.1f}")
    k2.metric("Avg Proj FP", f"{run['avg_proj']:.1f}")
    k3.metric("Best Lineup", f"{run['best']:.1f}")
    k4.metric("Beat Proj %", f"{run['beat_proj_pct']:.0f}%")

    # ── Two-column layout: table + scatter ────────────────────────────
    col_left, col_right = st.columns([6, 4])

    with col_left:
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

    with col_right:
        st.subheader("Proj vs Actual")
        # De-dup players for scatter (one dot per player)
        scatter_df = (
            player_df.drop_duplicates(subset="player_name")
            .dropna(subset=["proj", "actual_fp"])
        )
        html = _build_scatter_html(scatter_df)
        components.html(html, height=500, scrolling=False)


# ---------------------------------------------------------------------------
# Run Log
# ---------------------------------------------------------------------------

def _render_run_log() -> Optional[int]:
    """Render the run log expander. Returns index of selected run or None."""
    runs: List[Dict[str, Any]] = st.session_state.get("sim_lab_runs", [])
    if not runs:
        return None

    with st.expander("Run Log", expanded=False):
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

        # Allow re-display of a past run
        if len(runs) > 1:
            sel = st.selectbox(
                "Re-display run",
                options=list(range(len(runs))),
                format_func=lambda i: f"Run {i+1}: {runs[i]['date']} / {runs[i]['preset']} ({runs[i]['config_hash']})",
                key="sim_lab_run_select",
            )
            return sel
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_sim_lab(sport: str) -> None:
    """Render the Sim Lab tab."""
    st.header("\U0001f52c Sim Lab")

    # ── Controls row ──────────────────────────────────────────────────
    presets = _NBA_PRESETS if sport == "NBA" else _PGA_PRESETS

    c_date, c_preset, c_btn = st.columns([2, 3, 2])

    with c_date:
        if sport == "NBA":
            rg_dates = _scan_rg_dates()
            if rg_dates:
                selected_date = st.selectbox(
                    "Date (RG Archive)",
                    options=rg_dates,
                    format_func=lambda d: d.strftime("%Y-%m-%d"),
                    key="sim_lab_date_nba",
                )
            else:
                st.warning("No RG archive files found in data/rg_archive/nba/")
                selected_date = st.date_input(
                    "Date",
                    value=date.today() - timedelta(days=1),
                    key="sim_lab_date",
                )
        else:
            selected_date = st.date_input(
                "Date",
                value=date.today() - timedelta(days=1),
                key="sim_lab_date",
            )

    with c_preset:
        preset_name = st.selectbox(
            "Contest Preset",
            options=presets,
            key="sim_lab_preset",
        )

    sandbox_overrides = _render_config_expander(preset_name)

    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("\U0001f504 Fetch & Build", use_container_width=True, key="sim_lab_run")

    # ── Execute pipeline ──────────────────────────────────────────────
    if run_clicked:
        with st.spinner("Fetching pool, building lineups, scoring\u2026"):
            try:
                run = _run_pipeline(sport, selected_date, preset_name, sandbox_overrides)

                # Store in run log
                if "sim_lab_runs" not in st.session_state:
                    st.session_state["sim_lab_runs"] = []
                st.session_state["sim_lab_runs"].append(run)
                st.session_state["sim_lab_latest_run"] = run

            except ValueError as exc:
                st.warning(str(exc))
                return
            except RuntimeError as exc:
                st.error(f"API error: {exc}")
                return
            except Exception as exc:
                msg = str(exc)
                if "in progress" in msg.lower() or "not final" in msg.lower():
                    st.warning("Games may still be in progress. Actuals may be incomplete.")
                else:
                    st.error(f"Pipeline error: {msg}")
                return

    # ── Display results ───────────────────────────────────────────────
    latest = st.session_state.get("sim_lab_latest_run")
    if latest:
        _render_results(latest)

    # ── Run log ───────────────────────────────────────────────────────
    selected_idx = _render_run_log()
    runs = st.session_state.get("sim_lab_runs", [])
    if selected_idx is not None and 0 <= selected_idx < len(runs):
        st.divider()
        st.caption(f"Showing Run {selected_idx + 1}")
        _render_results(runs[selected_idx])
