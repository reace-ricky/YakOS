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

    return {
        "slate_errors": slate_errors,
        "signal_history": signal_history,
        "signal_weights": signal_weights,
        "contest_history": contest_history,
        "breakout_profile": breakout_profile,
        "breakout_accuracy": breakout_accuracy,
        "recal_backtest": recal_backtest,
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

    # ── Post-Slate Feedback ───────────────────────────────────────────────
    st.markdown("---")
    try:
        _render_post_slate_feedback(sport)
    except Exception as e:
        st.error(f"Post-Slate Feedback error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Contest Results (entry form) ──────────────────────────────────────
    st.markdown("---")
    try:
        _render_contest_results(sport)
    except Exception as e:
        st.error(f"Contest Results error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Historical Backfill ───────────────────────────────────────────────
    st.markdown("---")
    try:
        _render_historical_backfill(sport)
    except Exception as e:
        st.error(f"Historical Backfill error: {e}\n```\n{traceback.format_exc()}\n```")


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

    if not slate_errors and not recal.get("slates"):
        st.info("No calibration data available.")
        return

    # Build rows for the as-run MAE line from slate_errors
    chart_rows = []
    for d in sorted(slate_errors.keys()):
        overall = slate_errors[d].get("overall", {})
        mae = overall.get("mae")
        if mae is not None:
            chart_rows.append({"date": d, "MAE": mae, "Series": "As-Run MAE"})

    # Build rows for the recalibrated MAE line from recal_backtest slates
    recal_slates = recal.get("slates", [])
    for s in recal_slates:
        if s.get("sport", "").lower() == sport and s.get("corrected_mae") is not None:
            chart_rows.append({"date": s["date"], "MAE": s["corrected_mae"], "Series": "Recalibrated MAE"})

    if not chart_rows:
        st.info("No MAE data available.")
        return

    df = pd.DataFrame(chart_rows)
    df["date"] = pd.to_datetime(df["date"])

    target_mae = 6.0 if sport == "nba" else 25.0

    color_scale = alt.Scale(
        domain=["As-Run MAE", "Recalibrated MAE"],
        range=["#4C78A8", "#E45756"],
    )

    # Dual MAE lines
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
    st.caption("As-Run = accuracy at time of slate. Recalibrated = accuracy with current model corrections applied retroactively.")


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

    feedback_date = st.text_input("Slate date", value=date.today().isoformat(), key=f"dash_fb_date_{sport}")

    if st.button("Run Post-Slate", key=f"dash_postslate_{sport}"):
        with st.spinner("Running post-slate analysis..."):
            try:
                result = _run_post_slate(sport, feedback_date)
                if result.get("status") == "ok":
                    st.success(f"Post-slate complete: {result.get('message', '')}")
                    if result.get("calibration_update"):
                        st.json(result["calibration_update"])
                else:
                    st.warning(result.get("message", "No actuals available for this date."))
            except Exception as e:
                st.error(f"Post-slate error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Contest Results Section (PRESERVED — has user-active forms)
# ══════════════════════════════════════════════════════════════════════════════

def _render_contest_results(sport: str) -> None:
    """Contest band entry form and history table."""
    st.markdown("### Contest Results")

    c1, c2 = st.columns(2)
    with c1:
        cr_date = st.date_input("Contest date", value=date.today(), key=f"cr_date_{sport}")
    with c2:
        cr_type = st.selectbox("Contest type", ["GPP", "Cash", "Showdown"], key=f"cr_type_{sport}")

    c3, c4 = st.columns(2)
    with c3:
        cash_line = st.number_input("Cash Line", min_value=0.0, step=0.5, key=f"cr_cash_{sport}")
        top_10 = st.number_input("Top 10% Score", min_value=0.0, step=0.5, key=f"cr_top10_{sport}")
    with c4:
        winning = st.number_input("Winning Score", min_value=0.0, step=0.5, key=f"cr_win_{sport}")
        entries = st.number_input("Entry Count", min_value=0, step=1, key=f"cr_entries_{sport}")

    notes = st.text_input("Notes", key=f"cr_notes_{sport}")

    if st.button("Save Contest Result", key=f"cr_save_{sport}"):
        if cash_line <= 0:
            st.warning("Cash line must be > 0")
            return

        from yak_core.contest_calibration import (
            ContestResult, save_contest_result, score_vs_bands,
        )
        from app.data_loader import published_dir

        cr = ContestResult(
            slate_date=str(cr_date),
            contest_type=cr_type.lower(),
            cash_line=cash_line,
            top_15_score=top_10,
            top_1_score=0,
            winning_score=winning,
            num_entries=entries,
            notes=notes,
        )

        # Check for published lineups to auto-score
        scores = None
        p_dir = published_dir(sport)
        lineup_files = list(p_dir.glob("*_lineups.parquet")) if p_dir.exists() else []
        if lineup_files:
            try:
                all_actuals = []
                for lf in lineup_files:
                    ldf = pd.read_parquet(lf)
                    if "actual_fp" in ldf.columns and "lineup_index" in ldf.columns:
                        lu_totals = ldf.groupby("lineup_index")["actual_fp"].sum()
                        all_actuals.extend(lu_totals.dropna().tolist())
                if all_actuals:
                    scores = score_vs_bands(all_actuals, cr)
            except Exception:
                pass

        save_contest_result(cr, scores=scores)
        st.success(f"Saved contest result for {cr_date} ({cr_type})")
        if scores and scores.get("n_lineups", 0) > 0:
            st.json(scores)


# ══════════════════════════════════════════════════════════════════════════════
# Historical Backfill Section (PRESERVED — has user-active forms)
# ══════════════════════════════════════════════════════════════════════════════

def _render_historical_backfill(sport: str) -> None:
    """PGA and NBA backfill controls."""
    st.markdown("### Historical Backfill")
    st.caption("Calibrate against historical slates to build the curve faster.")

    if sport.upper() == "PGA":
        _render_pga_backfill()
    else:
        _render_nba_backfill()


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


def _render_nba_backfill() -> None:
    """NBA Historical Slates backfill UI."""
    st.markdown("**NBA Historical Slates**")

    rapid_key = _resolve_rapidapi_key()
    if not rapid_key:
        st.warning("RAPIDAPI_KEY not found in secrets or environment.")
        return

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start date", value=date.today() - timedelta(days=30),
                              key="nba_bf_start")
    with c2:
        end = st.date_input("End date", value=date.today() - timedelta(days=1),
                            key="nba_bf_end")

    if st.button("Run NBA Backfill", key="nba_run_backfill"):
        _run_nba_backfill(rapid_key, start, end)


def _run_nba_backfill(api_key: str, start_date: date, end_date: date) -> None:
    """Execute NBA backfill for a date range."""
    from yak_core.live import fetch_live_dfs, fetch_actuals_from_api
    from yak_core.calibration_feedback import record_slate_errors
    from yak_core.outcome_logger import log_slate_outcomes

    cfg = {"RAPIDAPI_KEY": api_key}
    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d)
        d += timedelta(days=1)

    if not dates:
        st.warning("No dates in range.")
        return

    progress = st.progress(0.0)
    status_text = st.empty()
    results = []

    for i, d in enumerate(dates):
        date_str = d.isoformat()
        date_key = d.strftime("%Y%m%d")
        status_text.text(f"Processing {date_str}...")

        try:
            # 1. Fetch projections (salary + proj)
            proj_df = fetch_live_dfs(date_key, cfg)
            if proj_df is None or proj_df.empty:
                results.append({"date": date_str, "status": "skipped", "detail": "No DFS data"})
                progress.progress((i + 1) / len(dates))
                time.sleep(1)
                continue

            # 2. Fetch actuals
            actuals = fetch_actuals_from_api(date_key, cfg)
            if actuals is None or actuals.empty:
                results.append({"date": date_str, "status": "skipped", "detail": "No actuals"})
                progress.progress((i + 1) / len(dates))
                time.sleep(1)
                continue

            # 3. Merge projections + actuals
            pool = proj_df.copy()
            act_map = actuals.set_index("player_name")["actual_fp"].to_dict()
            pool["actual_fp"] = pool["player_name"].map(act_map)
            matched = pool["actual_fp"].notna().sum()

            if matched == 0:
                results.append({"date": date_str, "status": "skipped", "detail": "No matches"})
                progress.progress((i + 1) / len(dates))
                time.sleep(1)
                continue

            # Ensure required columns
            if "pos" not in pool.columns:
                pool["pos"] = "G"
            pool["salary"] = pd.to_numeric(pool.get("salary", 0), errors="coerce").fillna(0)
            pool["proj"] = pd.to_numeric(pool.get("proj", 0), errors="coerce").fillna(0)
            pool["actual_fp"] = pd.to_numeric(pool["actual_fp"], errors="coerce")

            valid = pool[
                pool["actual_fp"].notna() & (pool["actual_fp"] > 0)
                & pool["proj"].notna() & pool["salary"].notna()
            ].copy()

            if valid.empty:
                results.append({"date": date_str, "status": "skipped", "detail": "No valid data"})
                progress.progress((i + 1) / len(dates))
                time.sleep(1)
                continue

            # 4. Compute edge metrics for richer outcome logging
            try:
                from yak_core.edge import compute_edge_metrics
                edge_df = compute_edge_metrics(valid, sport="NBA")
                edge_cols = ["player_name", "edge_score", "edge_label", "smash_prob",
                             "bust_prob", "leverage", "ceil", "floor"]
                edge_cols = [c for c in edge_cols if c in edge_df.columns]
                if "player_name" in edge_cols:
                    valid = valid.merge(
                        edge_df[edge_cols].drop_duplicates("player_name"),
                        on="player_name", how="left", suffixes=("", "_edge"),
                    )
                    for col in ["edge_score", "edge_label", "smash_prob", "bust_prob",
                                "leverage", "ceil", "floor"]:
                        ecol = f"{col}_edge"
                        if ecol in valid.columns:
                            valid[col] = valid[ecol].combine_first(valid.get(col, pd.Series()))
                            valid.drop(columns=[ecol], inplace=True)
            except Exception:
                pass

            # 5. Compute ownership estimate
            try:
                from yak_core.ownership import salary_rank_ownership
                valid = salary_rank_ownership(valid, col="own_pct")
            except Exception:
                if "own_pct" not in valid.columns:
                    valid["own_pct"] = 0.0

            # 6. Record calibration errors
            cal_result = record_slate_errors(date_str, valid, sport="NBA")

            # 7. Log outcomes
            outcomes = log_slate_outcomes(date_str, valid, sport="NBA")

            mae = cal_result.get("overall", {}).get("mae", 0) if isinstance(cal_result, dict) else 0
            corr = cal_result.get("overall", {}).get("correlation", 0) if isinstance(cal_result, dict) else 0

            results.append({
                "date": date_str,
                "MAE": round(mae, 2) if mae else "N/A",
                "correlation": round(corr, 3) if corr else "N/A",
                "n_players": len(valid),
                "status": "ok",
                "detail": "",
            })

        except Exception as e:
            err_msg = str(e)
            # Skip gracefully for no-games dates
            if "NoGamesScheduled" in err_msg or "no games" in err_msg.lower():
                results.append({"date": date_str, "status": "skipped", "detail": "No games"})
            else:
                results.append({"date": date_str, "status": "error", "detail": err_msg[:100]})

        progress.progress((i + 1) / len(dates))
        time.sleep(1)

    status_text.text("Backfill complete.")
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


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
            pool_with_actuals = pool.merge(
                actuals[["player_name", "actual_fp"]],
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

        record_slate_errors(slate_date, pool_with_actuals, sport=sport.upper())

        summary = get_calibration_summary(sport=sport.upper())
        return {
            "status": "ok",
            "message": f"Recorded errors for {has_actual} players",
            "calibration_update": summary,
        }
    except Exception as e:
        return {"status": "error", "message": f"Calibration update failed: {e}"}
