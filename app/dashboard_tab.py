"""Tab 4: Dashboard (admin only).

Visual command center: calibration trends, signal lift, contest band tracking,
breakout identification, and operational controls.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Data Loading Helpers ─────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    """Load a JSON file, returning empty dict on failure."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _load_all_dashboard_data(sport: str) -> Dict[str, Any]:
    """Load every data source once at the top of the render cycle."""
    sport_lower = sport.lower()

    # Calibration slate errors (per-sport)
    nba_errors = _load_json(REPO_ROOT / "data" / "calibration_feedback" / "nba" / "slate_errors.json")
    pga_errors = _load_json(REPO_ROOT / "data" / "calibration_feedback" / "pga" / "slate_errors.json")
    slate_errors = nba_errors if sport_lower == "nba" else pga_errors

    # Signal data
    signal_history = _load_json(REPO_ROOT / "data" / "edge_feedback" / "signal_history.json")
    signal_weights = _load_json(REPO_ROOT / "data" / "edge_feedback" / "signal_weights.json")

    # Contest results
    contest_history = _load_json(REPO_ROOT / "data" / "contest_results" / "history.json")

    # Breakout profile
    breakout_profile = _load_json(REPO_ROOT / "data" / "sim_sandbox" / "breakout_profile.json")

    # Breakout accuracy (persisted by nightly calibration)
    breakout_accuracy = _load_json(REPO_ROOT / "data" / "calibration_feedback" / "breakout_accuracy.json")

    # Recalibrated backtest (all slates re-projected through current corrections)
    recal_backtest = _load_json(REPO_ROOT / "data" / "calibration_feedback" / "recalibrated_backtest.json")

    # RG baseline MAE data (standalone file + inline in slate_errors)
    rg_baseline = _load_json(REPO_ROOT / "data" / "calibration_feedback" / "rg_baseline.json")

    return {
        "slate_errors": slate_errors,
        "signal_history": signal_history,
        "signal_weights": signal_weights,
        "contest_history": contest_history,
        "breakout_profile": breakout_profile,
        "breakout_accuracy": breakout_accuracy,
        "recal_backtest": recal_backtest,
        "rg_baseline": rg_baseline,
        "sport": sport_lower,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Render
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard_tab(sport: str) -> None:
    """Render the Dashboard tab."""
    from app.data_loader import load_calibration_data, load_signal_history, published_dir, load_fresh_meta

    data = _load_all_dashboard_data(sport)

    # ── Section 1: System Health ──────────────────────────────────────────
    try:
        _render_system_health(data)
    except Exception as e:
        st.error(f"System Health error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Section 2: Calibration Trend ──────────────────────────────────────
    st.markdown("---")
    try:
        _render_calibration_trend(data)
    except Exception as e:
        st.error(f"Calibration Trend error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Section 3: Breakout Identification ────────────────────────────────
    st.markdown("---")
    try:
        _render_breakout_identification(data, sport)
    except Exception as e:
        st.error(f"Breakout Identification error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Section 4: Contest Band Tracking ──────────────────────────────────
    st.markdown("---")
    try:
        _render_contest_band_tracking(data)
    except Exception as e:
        st.error(f"Contest Band Tracking error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Section 5: Signal Accuracy Trend ──────────────────────────────────
    st.markdown("---")
    try:
        _render_signal_accuracy_trend(data)
    except Exception as e:
        st.error(f"Signal Accuracy Trend error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Section 6: Published Data Status ──────────────────────────────────
    st.markdown("---")
    try:
        _render_published_data_status(sport)
    except Exception as e:
        st.error(f"Published Data Status error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Maintenance Tools ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Maintenance Tools", expanded=False):
        try:
            _render_post_slate_feedback(sport)
        except Exception as e:
            st.error(f"Post-Slate Feedback error: {e}\n```\n{traceback.format_exc()}\n```")
        st.markdown("---")
        try:
            _render_recalibrate_from_archive(sport)
        except Exception as e:
            st.error(f"Recalibrate from Archive error: {e}\n```\n{traceback.format_exc()}\n```")


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: System Health
# ══════════════════════════════════════════════════════════════════════════════

def _render_system_health(data: Dict[str, Any]) -> None:
    st.markdown("### System Health")

    slate_errors = data["slate_errors"]
    signal_weights = data["signal_weights"]
    recal = data.get("recal_backtest", {})
    sport = data["sport"]

    sorted_dates = sorted(slate_errors.keys())

    # Projection Accuracy: prefer recalibrated backtest corrected_mae
    recal_summary = recal.get("summary", {}).get(sport, {})
    recal_mae = recal_summary.get("corrected_mae")
    recal_improvement = recal_summary.get("improvement")

    # Fallback to rolling MAE if no recalibrated data
    recent_mae = recal_mae
    mae_delta = recal_improvement
    if recent_mae is None and len(sorted_dates) >= 2:
        recent_5 = sorted_dates[-5:]
        prior_5 = sorted_dates[-10:-5] if len(sorted_dates) >= 10 else sorted_dates[:max(1, len(sorted_dates) - 5)]
        recent_maes = [slate_errors[d].get("overall", {}).get("mae", 0) for d in recent_5 if slate_errors[d].get("overall", {}).get("mae") is not None]
        prior_maes = [slate_errors[d].get("overall", {}).get("mae", 0) for d in prior_5 if slate_errors[d].get("overall", {}).get("mae") is not None]
        if recent_maes:
            recent_mae = sum(recent_maes) / len(recent_maes)
        prior_mae = sum(prior_maes) / len(prior_maes) if prior_maes else None
        if recent_mae is not None and prior_mae is not None:
            mae_delta = recent_mae - prior_mae

    # Signal weighted hit rate
    sig_stats = signal_weights.get("signal_stats", {})
    weighted_rates = [s.get("weighted_hit_rate", 0) for s in sig_stats.values() if s.get("weighted_hit_rate")]
    overall_hit_rate = sum(weighted_rates) / len(weighted_rates) if weighted_rates else None

    # Slates calibrated
    n_slates = len(sorted_dates)
    # Delta: slates added in last 7 days
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    slates_this_week = sum(1 for d in sorted_dates if d >= cutoff)

    c1, c2, c3 = st.columns(3)
    with c1:
        if recent_mae is not None:
            if recal_mae is not None:
                # Show corrected MAE with raw MAE comparison
                # improvement > 0 means corrections helped (lower MAE = better)
                raw_mae_val = recal_summary.get("raw_mae")
                if raw_mae_val is not None and recal_improvement is not None:
                    if recal_improvement > 0:
                        delta_str = f"-{abs(recal_improvement):.2f} vs raw ({raw_mae_val:.1f})"
                    else:
                        delta_str = f"+{abs(recal_improvement):.2f} vs raw ({raw_mae_val:.1f})"
                else:
                    delta_str = None
                st.metric(
                    "Model MAE (recalibrated)",
                    f"{recent_mae:.2f}",
                    delta=delta_str,
                    delta_color="inverse",  # negative delta = green (lower MAE is better)
                )
            else:
                st.metric(
                    "Projection Accuracy (MAE)",
                    f"{recent_mae:.2f}",
                    delta=f"{mae_delta:+.2f}" if mae_delta is not None else None,
                    delta_color="inverse",
                )
        else:
            st.metric("Model MAE (recalibrated)", "N/A")
    with c2:
        if overall_hit_rate is not None:
            st.metric(
                "Signal Hit Rate",
                f"{overall_hit_rate:.1%}",
                delta=f"target: 35%",
                delta_color="off",
            )
        else:
            st.metric("Signal Hit Rate", "N/A")
    with c3:
        st.metric(
            "Slates Calibrated",
            n_slates,
            delta=f"+{slates_this_week} this week" if slates_this_week else None,
            delta_color="normal",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Calibration Trend
# ══════════════════════════════════════════════════════════════════════════════

def _render_calibration_trend(data: Dict[str, Any]) -> None:
    st.markdown("### Projection Accuracy Over Time")

    slate_errors = data["slate_errors"]
    sport = data["sport"]
    recal = data.get("recal_backtest", {})
    rg_baseline = data.get("rg_baseline", {})

    if not slate_errors and not recal.get("slates"):
        st.info("No calibration data available.")
        return

    # Build rows for the YakOS MAE line from slate_errors
    chart_rows = []
    for d in sorted(slate_errors.keys()):
        overall = slate_errors[d].get("overall", {})
        mae = overall.get("mae")
        if mae is not None:
            chart_rows.append({"date": d, "MAE": mae, "Series": "YakOS MAE"})

        # RG MAE from slate_errors (injected during calibration)
        rg_mae = slate_errors[d].get("rg_mae")
        if rg_mae is not None:
            chart_rows.append({"date": d, "MAE": rg_mae, "Series": "RG Baseline MAE"})

    # Also pull from standalone rg_baseline.json for dates not in slate_errors
    for d, rg_data in sorted(rg_baseline.items()):
        if d not in slate_errors and rg_data.get("rg_mae") is not None:
            chart_rows.append({"date": d, "MAE": rg_data["rg_mae"], "Series": "RG Baseline MAE"})

    if not chart_rows:
        st.info("No MAE data available.")
        return

    df = pd.DataFrame(chart_rows)
    df["date"] = pd.to_datetime(df["date"])

    target_mae = 6.0 if sport == "nba" else 25.0

    # Determine which series are present
    series_present = df["Series"].unique().tolist()
    domain = []
    colors = []
    if "YakOS MAE" in series_present:
        domain.append("YakOS MAE")
        colors.append("#4C78A8")
    if "RG Baseline MAE" in series_present:
        domain.append("RG Baseline MAE")
        colors.append("#F5A623")

    color_scale = alt.Scale(domain=domain, range=colors)

    # MAE lines
    mae_lines = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("MAE:Q", title="MAE", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=color_scale),
        tooltip=["date:T", "Series:N", "MAE:Q"],
    )

    # Target line
    target_df = pd.DataFrame([{"target": target_mae}])
    target_rule = alt.Chart(target_df).mark_rule(
        color="green", strokeDash=[6, 3], strokeWidth=2
    ).encode(
        y="target:Q",
    )

    target_label = alt.Chart(target_df).mark_text(
        align="left", dx=5, dy=-8, color="green", fontSize=11
    ).encode(
        y="target:Q",
        text=alt.value(f"Target: {target_mae}"),
    )

    chart = (mae_lines + target_rule + target_label).properties(
        height=300,
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # Show footnote
    st.caption("YakOS MAE = model projections vs actuals. RG Baseline MAE = raw RotoGrinders projections vs actuals.")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Breakout Identification
# ══════════════════════════════════════════════════════════════════════════════

def _render_breakout_identification(data: Dict[str, Any], sport: str = "NBA") -> None:
    st.markdown("### Breakout Identification")

    left, right = st.columns(2)

    # ── Left: Signal Lift Chart ───────────────────────────────────────────
    with left:
        st.markdown("**Signal Lift**")
        breakout = data["breakout_profile"]
        signals = breakout.get("signals", {})

        if not signals:
            st.info("No breakout signal data.")
        else:
            lift_rows = []
            for sig, lift in signals.items():
                lift_rows.append({
                    "signal": sig,
                    "lift": lift,
                    "works": "Above baseline" if lift > 1.0 else "Below baseline",
                })

            lift_df = pd.DataFrame(lift_rows)

            bars = alt.Chart(lift_df).mark_bar().encode(
                x=alt.X("lift:Q", title="Lift (1.0 = baseline)", scale=alt.Scale(domain=[0.5, 1.5])),
                y=alt.Y("signal:N", sort="-x", title=""),
                color=alt.Color(
                    "works:N",
                    scale=alt.Scale(domain=["Above baseline", "Below baseline"], range=["#2ca02c", "#d62728"]),
                    legend=None,
                ),
                tooltip=["signal:N", "lift:Q"],
            )

            baseline = alt.Chart(pd.DataFrame([{"x": 1.0}])).mark_rule(
                color="black", strokeWidth=2
            ).encode(x="x:Q")

            chart = (bars + baseline).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

            n_slates = breakout.get("n_slates", 0)
            n_breakouts = breakout.get("n_breakouts", 0)
            st.caption(f"{n_breakouts} breakouts across {n_slates} slates")

    # ── Right: Breakout Precision / Recall ────────────────────────────────
    with right:
        st.markdown("**Breakout Precision / Recall**")
        ba_history = data.get("breakout_accuracy", {})

        # Compute rolling precision/recall from all entries for the current sport
        sport_entries = [
            v for v in ba_history.values()
            if isinstance(v, dict) and v.get("sport", "NBA").upper() == sport.upper()
        ]

        if sport_entries:
            total_pred = sum(e.get("n_predicted", 0) for e in sport_entries)
            total_actual = sum(e.get("n_actual", 0) for e in sport_entries)
            total_correct = sum(e.get("n_correct", 0) for e in sport_entries)
            precision = total_correct / total_pred if total_pred > 0 else None
            recall = total_correct / total_actual if total_actual > 0 else None

            if precision is not None:
                prec_pct = precision * 100
                prec_delta = prec_pct - 60.0  # target = 60%
                st.metric(
                    "Precision",
                    f"{prec_pct:.1f}%",
                    delta=f"{prec_delta:+.1f}% vs 60% target",
                    delta_color="normal",
                )
            else:
                st.metric("Precision", "N/A")

            if recall is not None:
                rec_pct = recall * 100
                rec_delta = rec_pct - 50.0  # target = 50%
                st.metric(
                    "Recall",
                    f"{rec_pct:.1f}%",
                    delta=f"{rec_delta:+.1f}% vs 50% target",
                    delta_color="normal",
                )
            else:
                st.metric("Recall", "N/A")

            st.caption(f"Across {len(sport_entries)} calibrated slates")
        else:
            st.metric("Precision", "N/A")
            st.metric("Recall", "N/A")
            st.caption("Run nightly calibration to populate")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Contest Band Tracking
# ══════════════════════════════════════════════════════════════════════════════

def _render_contest_band_tracking(data: Dict[str, Any]) -> None:
    st.markdown("### Contest Band Tracking")

    contest_history = data["contest_history"]
    if not contest_history:
        st.info("No contest results recorded yet.")
        return

    # Separate GPP and cash contests — they have fundamentally different
    # score distributions and should not be plotted on the same chart.
    gpp_rows = []
    cash_rows = []
    for key, entry in contest_history.items():
        d = entry.get("slate_date", key.split("_")[0])
        cash_line = entry.get("cash_line", 0)
        winning = entry.get("winning_score", 0)
        scores = entry.get("scores", {})
        best = scores.get("best", 0)
        avg = scores.get("avg", 0)
        ctype = entry.get("contest_type", "gpp").lower()

        row = {"date": d, "Cash Line": cash_line, "Winning Score": winning}
        if best and best > 0:
            row["Best Lineup"] = best
        if avg and avg > 0:
            row["Avg Lineup"] = avg

        if ctype in ("cash", "50/50", "double_up"):
            cash_rows.append(row)
        else:
            gpp_rows.append(row)

    color_scale = alt.Scale(
        domain=["Cash Line", "Winning Score", "Best Lineup", "Avg Lineup"],
        range=["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd"],
    )

    def _band_chart(rows: list, title: str) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        value_cols = [c for c in ["Cash Line", "Winning Score", "Best Lineup", "Avg Lineup"] if c in df.columns]
        melted = df.melt(id_vars=["date"], value_vars=value_cols, var_name="Metric", value_name="Score")
        melted = melted[melted["Score"] > 0]
        if melted.empty:
            return
        chart = alt.Chart(melted).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(zero=False)),
            color=alt.Color("Metric:N", scale=color_scale),
            tooltip=["date:T", "Metric:N", "Score:Q"],
        ).properties(height=250, title=title).interactive()
        st.altair_chart(chart, use_container_width=True)

    _band_chart(gpp_rows, "GPP Contests")
    _band_chart(cash_rows, "Cash Contests")

    # Compact contest results table below
    table_rows = []
    for key, entry in sorted(contest_history.items(), reverse=True):
        sc = entry.get("scores", {})
        table_rows.append({
            "date": entry.get("slate_date", ""),
            "type": entry.get("contest_type", ""),
            "cash_line": entry.get("cash_line", 0),
            "winning": entry.get("winning_score", 0),
            "entries": entry.get("num_entries", 0),
            "best": sc.get("best", 0),
            "avg": sc.get("avg", 0),
            "cash_rate": sc.get("cash_rate", ""),
        })
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=200)


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Signal Accuracy Trend
# ══════════════════════════════════════════════════════════════════════════════

def _render_signal_accuracy_trend(data: Dict[str, Any]) -> None:
    st.markdown("### Signal Hit Rates Over Time")

    signal_history = data["signal_history"]
    if not signal_history:
        st.info("No signal history data.")
        return

    rows = []
    for d in sorted(signal_history.keys()):
        entry = signal_history[d]
        signals = entry.get("signals", {})
        for sig_name, sig_data in signals.items():
            hr = sig_data.get("hit_rate")
            if hr is not None:
                rows.append({"date": d, "signal": sig_name, "hit_rate": hr})

    if not rows:
        st.info("No hit rate data available.")
        return

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("hit_rate:Q", title="Hit Rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("signal:N", title="Signal"),
        tooltip=["date:T", "signal:N", "hit_rate:Q"],
    ).properties(height=300).interactive()

    st.altair_chart(chart, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Section 6a: Published Data Status (kept from original)
# ══════════════════════════════════════════════════════════════════════════════

def _render_published_data_status(sport: str) -> None:
    from app.data_loader import published_dir, load_fresh_meta

    st.markdown("### Published Data Status")

    for s in ["NBA", "PGA"]:
        meta = load_fresh_meta(s)
        if meta:
            p_dir = published_dir(s)
            lineup_files = list(p_dir.glob("*_lineups.parquet"))
            n_lineups = 0
            for lf in lineup_files:
                try:
                    ldf = pd.read_parquet(lf)
                    if "lineup_index" in ldf.columns:
                        n_lineups += ldf["lineup_index"].nunique()
                except Exception:
                    pass

            st.markdown(f"**{s}:** {meta.get('date', '?')} | {meta.get('pool_size', '?')} players | "
                        f"Source: {meta.get('proj_source', 'N/A')} | {n_lineups} lineups | "
                        f"{len(lineup_files)} contest(s)")
        else:
            st.caption(f"{s}: No published data")


# ══════════════════════════════════════════════════════════════════════════════
# Section 6b: Post-Slate Feedback (kept from original)
# ══════════════════════════════════════════════════════════════════════════════

def _render_post_slate_feedback(sport: str) -> None:
    st.markdown("### Post-Slate Feedback")

    feedback_date = st.date_input(
        "Slate date", value=date.today(), key=f"dash_fb_date_{sport}"
    )

    if st.button("Run Post-Slate", key=f"dash_postslate_{sport}"):
        with st.spinner("Running post-slate analysis..."):
            try:
                result = _run_post_slate(sport, str(feedback_date))
                if result.get("status") == "ok":
                    st.success(f"Post-slate complete: {result.get('message', '')}")
                    if result.get("calibration_update"):
                        st.json(result["calibration_update"])
                    # Display minutes calibration stats if available
                    mins_overall = result.get("minutes_stats", {})
                    if mins_overall.get("min_mae") is not None:
                        st.markdown(
                            f"**Minutes Accuracy:** MAE={mins_overall['min_mae']}, "
                            f"Corr={mins_overall.get('min_correlation', 'N/A')}, "
                            f"Bias={mins_overall.get('min_mean_error', 0):+.1f}"
                        )
                else:
                    st.warning(result.get("message", "No actuals available for this date."))
            except Exception as e:
                st.error(f"Post-slate error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Historical Backfill Section (PRESERVED — has user-active forms)
# ══════════════════════════════════════════════════════════════════════════════

def _render_historical_backfill(sport: str) -> None:
    """PGA and NBA backfill controls."""
    st.markdown("### Historical Backfill")
    st.caption("Calibrate against historical slates to build the curve faster.")

    if sport.upper() == "PGA":
        _render_pga_backfill()



def _render_recalibrate_from_archive(sport: str) -> None:
    """Recalibrate from Archive UI (sport-agnostic)."""
    st.markdown("### Recalibrate from Archive")
    st.caption(
        "Rebuild correction factors from archived slate parquets "
        "(uses YakOS projections vs actuals — not Tank01)."
    )
    archive_dir = REPO_ROOT / "data" / "slate_archive"
    parquets = sorted(archive_dir.glob("*.parquet")) if archive_dir.exists() else []

    # Separate NBA vs PGA archives
    nba_archives = [p for p in parquets if "pga" not in p.name.lower()]
    pga_archives = [p for p in parquets if "pga" in p.name.lower()]

    target = nba_archives if sport.upper() != "PGA" else pga_archives
    sport_label = "NBA" if sport.upper() != "PGA" else "PGA"
    st.info(f"Found **{len(target)}** {sport_label} archived slates in `data/slate_archive/`.")

    if st.button(
        f"Recalibrate {sport_label} from Archive",
        key="recalibrate_archive",
        type="primary",
    ):
        _recalibrate_from_archive(target, sport=sport_label)


def _render_pga_backfill() -> None:
    """PGA Historical Events backfill UI."""
    st.markdown("**PGA Historical Events**")

    dg_key = _resolve_datagolf_key()
    if not dg_key:
        st.warning("DATAGOLF_API_KEY not found in secrets or environment.")
        return

    if st.button("Load Events", key="pga_load_events"):
        with st.spinner("Fetching events from DataGolf..."):
            try:
                from yak_core.datagolf import DataGolfClient
                from yak_core.pga_calibration import get_pga_event_list
                dg = DataGolfClient(dg_key)
                events = get_pga_event_list(dg)
                if events.empty:
                    st.warning("No PGA events found.")
                    return
                st.session_state["pga_events"] = events
            except Exception as e:
                st.error(f"Failed to load events: {e}")
                return

    events = st.session_state.get("pga_events")
    if events is None or (isinstance(events, pd.DataFrame) and events.empty):
        return

    display_cols = [c for c in ["event_name", "date", "calendar_year", "event_id",
                                "dk_salaries", "dk_ownerships"] if c in events.columns]
    edit_df = events[display_cols].copy()
    edit_df.insert(0, "select", False)

    edited = st.data_editor(edit_df, use_container_width=True, hide_index=True,
                            key="pga_event_editor")
    selected = edited[edited["select"]].copy()

    if st.button("Run PGA Backfill", key="pga_run_backfill") and not selected.empty:
        from yak_core.datagolf import DataGolfClient
        from yak_core.pga_calibration import calibrate_pga_event
        from yak_core.outcome_logger import log_slate_outcomes

        dg = DataGolfClient(dg_key)
        progress = st.progress(0.0)
        results = []
        n = len(selected)

        for i, (_, row) in enumerate(selected.iterrows()):
            eid = int(row["event_id"])
            yr = int(row.get("calendar_year", 2025))
            evt_date = str(row.get("date", f"{yr}-{eid:03d}"))
            evt_name = row.get("event_name", f"Event {eid}")

            try:
                cal = calibrate_pga_event(dg, eid, yr, slate_date=evt_date)
                status = "error" if "error" in cal else "ok"
                mae = cal.get("calibration", {}).get("overall", {}).get("mae", 0) if status == "ok" else 0
                n_players = cal.get("n_players_calibrated", 0)

                # Log outcomes if calibration succeeded
                if status == "ok" and n_players > 0:
                    try:
                        from yak_core.pga_calibration import fetch_pga_actuals, _build_pool_from_actuals_and_preds
                        actuals = fetch_pga_actuals(dg, eid, yr)
                        pool = _build_pool_from_actuals_and_preds(dg, actuals, eid, yr)
                        if not pool.empty and "actual_fp" in pool.columns:
                            log_slate_outcomes(evt_date, pool, sport="PGA")
                    except Exception:
                        pass

                results.append({
                    "event": evt_name, "date": evt_date,
                    "MAE": round(mae, 2) if mae else "N/A",
                    "n_players": n_players, "status": status,
                    "detail": cal.get("error", ""),
                })
            except Exception as e:
                results.append({
                    "event": evt_name, "date": evt_date,
                    "MAE": "N/A", "n_players": 0,
                    "status": "error", "detail": str(e),
                })

            progress.progress((i + 1) / n)
            time.sleep(1)

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)




# ── Recalibrate from Archive ────────────────────────────────────────────────

def _inject_rg_mae(slate_date: str, pool_df: pd.DataFrame, sport: str = "NBA") -> None:
    """Compute RG baseline MAE and store it in slate_errors + rg_baseline.json.

    Looks for an RG archive file matching the slate date. If found, merges
    RG projections with actuals from pool_df and stores the MAE.
    """
    if sport.lower() != "nba":
        return  # Only NBA for now

    rg_dir = REPO_ROOT / "data" / "rg_archive" / "nba"
    rg_path = rg_dir / f"rg_{slate_date}.csv"
    if not rg_path.exists():
        return

    if "actual_fp" not in pool_df.columns:
        return

    try:
        rg = pd.read_csv(rg_path)
        rg.columns = [c.strip().upper() for c in rg.columns]
        if "FPTS" not in rg.columns or "PLAYER" not in rg.columns:
            return

        rg_clean = pd.DataFrame()
        rg_clean["player_name"] = rg["PLAYER"].astype(str).str.strip().str.replace('"', '')
        rg_clean["rg_proj"] = pd.to_numeric(rg["FPTS"], errors="coerce")
        rg_clean["_key"] = rg_clean["player_name"].str.strip().str.lower()

        actuals = pool_df[["player_name", "actual_fp"]].dropna(subset=["actual_fp"]).copy()
        actuals["_key"] = actuals["player_name"].astype(str).str.strip().str.lower()

        merged = rg_clean.merge(actuals[["_key", "actual_fp"]], on="_key", how="inner")
        if len(merged) < 10:
            return

        rg_mae = float((merged["rg_proj"] - merged["actual_fp"]).abs().mean())

        # Inject into slate_errors.json
        errors_path = REPO_ROOT / "data" / "calibration_feedback" / "nba" / "slate_errors.json"
        if errors_path.exists():
            import json as _json
            with open(errors_path) as f:
                errors = _json.load(f)
            if slate_date in errors:
                errors[slate_date]["rg_mae"] = round(rg_mae, 2)
                with open(errors_path, "w") as f:
                    _json.dump(errors, f, indent=2)

        # Also update standalone rg_baseline.json
        baseline_path = REPO_ROOT / "data" / "calibration_feedback" / "rg_baseline.json"
        baseline = {}
        if baseline_path.exists():
            import json as _json
            with open(baseline_path) as f:
                baseline = _json.load(f)
        baseline[slate_date] = {
            "rg_mae": round(rg_mae, 2),
            "rg_bias": round(float((merged["rg_proj"] - merged["actual_fp"]).mean()), 2),
            "n_matched": len(merged),
        }
        with open(baseline_path, "w") as f:
            import json as _json
            _json.dump(baseline, f, indent=2)

    except Exception:
        pass  # Non-critical — don't break calibration flow


def _recalibrate_from_archive(archive_files: list, sport: str = "NBA") -> None:
    """Clear existing calibration and rebuild from archived slate parquets.

    Each archive parquet contains YakOS projections (``proj``) and actual
    fantasy points (``actual_fp``), so this bypasses Tank01 entirely and
    recalibrates using YakOS's own projection accuracy.
    """
    from yak_core.calibration_feedback import (
        clear_calibration_history,
        record_slate_errors,
        get_calibration_summary,
    )

    if not archive_files:
        st.warning("No archive files found for this sport.")
        return

    # 1. Clear existing calibration so we rebuild from scratch
    clear_calibration_history(sport=sport.upper())
    st.info(f"Cleared existing {sport} calibration history. Rebuilding...")

    progress = st.progress(0.0)
    status_text = st.empty()
    results = []
    n = len(archive_files)

    for i, fpath in enumerate(archive_files):
        fname = fpath.name if hasattr(fpath, "name") else os.path.basename(str(fpath))
        # Extract slate date from filename (e.g. "2026-02-05_gpp_main.parquet" → "2026-02-05")
        slate_date = fname.split("_")[0]
        status_text.text(f"Processing {fname} ({i + 1}/{n})...")

        try:
            df = pd.read_parquet(fpath)

            # Validate required columns
            required = {"player_name", "pos", "salary", "proj", "actual_fp"}
            missing = required - set(df.columns)
            if missing:
                results.append({
                    "file": fname, "date": slate_date, "status": "skipped",
                    "detail": f"Missing columns: {missing}", "n_players": 0,
                    "MAE": "N/A", "correlation": "N/A",
                })
                progress.progress((i + 1) / n)
                continue

            # Filter to players who actually played
            df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")
            df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
            valid = df[df["actual_fp"].notna() & (df["actual_fp"] > 0) & df["proj"].notna()].copy()

            if valid.empty:
                results.append({
                    "file": fname, "date": slate_date, "status": "skipped",
                    "detail": "No valid proj/actual pairs", "n_players": 0,
                    "MAE": "N/A", "correlation": "N/A",
                })
                progress.progress((i + 1) / n)
                continue

            # Record errors — this appends to history and recomputes corrections
            cal_result = record_slate_errors(slate_date, valid, sport=sport.upper())

            # Inject RG baseline MAE if we have an RG archive for this date
            _inject_rg_mae(slate_date, valid, sport=sport)

            if "error" in cal_result:
                results.append({
                    "file": fname, "date": slate_date, "status": "rejected",
                    "detail": cal_result["error"], "n_players": len(valid),
                    "MAE": "N/A", "correlation": "N/A",
                })
            else:
                ov = cal_result.get("overall", {})
                results.append({
                    "file": fname, "date": slate_date, "status": "ok",
                    "detail": "",
                    "n_players": ov.get("n_players", len(valid)),
                    "MAE": round(ov.get("mae", 0), 2),
                    "correlation": round(ov.get("correlation", 0), 4),
                })

        except Exception as e:
            results.append({
                "file": fname, "date": slate_date, "status": "error",
                "detail": str(e)[:120], "n_players": 0,
                "MAE": "N/A", "correlation": "N/A",
            })

        progress.progress((i + 1) / n)

    status_text.text("Recalibration complete.")

    # Show results table
    if results:
        res_df = pd.DataFrame(results)
        ok_count = (res_df["status"] == "ok").sum()
        st.success(f"Successfully processed {ok_count}/{n} slates.")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Show updated calibration summary
    try:
        summary = get_calibration_summary(sport=sport.upper())
        st.markdown("#### Updated Calibration")
        st.json(summary)
    except Exception:
        pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_rapidapi_key() -> str:
    """Resolve RapidAPI key from secrets or environment."""
    key = os.environ.get("RAPIDAPI_KEY") or os.environ.get("TANK01_RAPIDAPI_KEY", "")
    if not key:
        try:
            key = st.secrets.get("RAPIDAPI_KEY", "")
        except Exception:
            pass
    return key


def _resolve_datagolf_key() -> str:
    """Resolve DataGolf API key from secrets or environment."""
    key = os.environ.get("DATAGOLF_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("DATAGOLF_API_KEY", "")
        except Exception:
            pass
    return key


def _run_post_slate(sport: str, slate_date: str) -> Dict[str, Any]:
    """Run post-slate feedback loop.

    Attempts to fetch actuals and record slate errors for calibration.
    Since post_slate.py doesn't exist as a standalone script, we wire
    the core components directly.
    """
    from app.data_loader import published_dir

    out_dir = published_dir(sport)
    pool_path = out_dir / "slate_pool.parquet"

    if not pool_path.exists():
        return {"status": "error", "message": f"No pool found for {sport}"}

    pool = pd.read_parquet(pool_path)

    # Try to fetch actuals
    try:
        if sport.upper() == "NBA":
            from yak_core.live import fetch_actuals_from_api
            api_key = _resolve_rapidapi_key()
            if not api_key:
                return {"status": "error", "message": "Missing RAPIDAPI_KEY for fetching actuals"}

            actuals = fetch_actuals_from_api(slate_date, {"RAPIDAPI_KEY": api_key})
            if actuals.empty:
                return {"status": "error", "message": f"No actuals available for {slate_date}"}

            # Merge actuals into pool — drop the placeholder actual_fp
            # column first (set to NaN by fetch_live_dfs) so pandas doesn't
            # create actual_fp_x / actual_fp_y suffix columns.
            if "actual_fp" in pool.columns:
                pool = pool.drop(columns=["actual_fp"])
            if "mp_actual" in pool.columns:
                pool = pool.drop(columns=["mp_actual"])

            # Merge both actual_fp and mp_actual from actuals
            merge_cols = ["player_name", "actual_fp"]
            if "mp_actual" in actuals.columns:
                merge_cols.append("mp_actual")
            pool_with_actuals = pool.merge(
                actuals[merge_cols],
                on="player_name",
                how="left",
            )
        else:
            return {"status": "error", "message": "PGA post-slate actuals not yet implemented"}
    except Exception as e:
        return {"status": "error", "message": f"Could not fetch actuals: {e}"}

    # Record slate errors
    try:
        from yak_core.calibration_feedback import record_slate_errors, get_calibration_summary

        has_actual = pool_with_actuals["actual_fp"].notna().sum()
        if has_actual == 0:
            return {"status": "error", "message": "No actuals matched to pool players"}

        slate_record = record_slate_errors(slate_date, pool_with_actuals, sport=sport.upper())

        # Inject RG baseline MAE if we have an RG file for this date
        _inject_rg_mae(slate_date, pool_with_actuals, sport=sport)

        summary = get_calibration_summary(sport=sport.upper())
        result = {
            "status": "ok",
            "message": f"Recorded errors for {has_actual} players",
            "calibration_update": summary,
        }
        # Pass through minutes stats from the slate record if available
        mins_overall = slate_record.get("overall", {})
        if "min_mae" in mins_overall:
            result["minutes_stats"] = {
                k: mins_overall[k]
                for k in ("min_mae", "min_rmse", "min_correlation", "min_mean_error", "min_n_players")
                if k in mins_overall
            }
        return result
    except Exception as e:
        return {"status": "error", "message": f"Calibration update failed: {e}"}
