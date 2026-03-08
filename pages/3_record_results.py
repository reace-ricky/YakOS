"""Record Results — feed contest outcomes into the calibration + edge feedback loops.

Two input modes:
  1. Screenshot upload (OCR) — upload a RotoGrinders contest result screenshot
  2. Manual entry — paste player names + actual FP

Both feed into:
  - calibration_feedback.py (projection error correction by position + salary tier)
  - edge_feedback.py (signal hit-rate tracking → auto-tuned weights)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import get_slate_state  # noqa: E402
from yak_core.context import get_lab_analysis  # noqa: E402
from yak_core.result_recorder import (  # noqa: E402
    merge_actuals_into_pool,
    actuals_from_ocr,
    record_contest_results,
    get_feedback_status,
)
from yak_core.calibration_feedback import (  # noqa: E402
    get_calibration_summary,
    clear_calibration_history,
)


def main() -> None:
    st.title("Record Results")
    st.caption(
        "Feed completed contest results into the calibration engine. "
        "Each recorded slate tightens projection accuracy and reweights edge signals."
    )

    slate = get_slate_state()
    _analysis = get_lab_analysis()
    pool: pd.DataFrame = _analysis["pool"] if not _analysis["pool"].empty else (
        slate.player_pool if slate.player_pool is not None else pd.DataFrame()
    )

    # ── Feedback System Status ────────────────────────────────────────
    status = get_feedback_status(
        store=st.session_state.get("_calibration_store")
    )

    col_cal, col_edge = st.columns(2)
    with col_cal:
        n_cal = status["total_slates_calibration"]
        _cal_icon = "🟢" if status["calibration_ready"] else ("🟡" if n_cal > 0 else "⚪")
        st.metric("Calibration", f"{n_cal} slates", delta=f"{'Ready' if status['calibration_ready'] else 'Building' if n_cal > 0 else 'No data'}")
    with col_edge:
        n_edge = status["total_slates_edge"]
        _edge_icon = "🟢" if status["edge_ready"] else ("🟡" if n_edge > 0 else "⚪")
        st.metric("Edge Feedback", f"{n_edge} slates", delta=f"{'Ready' if status['edge_ready'] else 'Building' if n_edge > 0 else 'No data'}")

    st.divider()

    # ── Input Mode ────────────────────────────────────────────────────
    input_mode = st.radio(
        "Input method",
        ["Screenshot (OCR)", "Manual Entry"],
        horizontal=True,
        key="_rr_input_mode",
    )

    # Slate date
    _default_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if slate.slate_date:
        _default_date = slate.slate_date
    slate_date = st.text_input(
        "Slate date (YYYY-MM-DD)",
        value=_default_date,
        key="_rr_slate_date",
    )

    contest_type = st.selectbox(
        "Contest type",
        ["GPP", "Cash", "Showdown"],
        key="_rr_contest_type",
    )

    actuals_df = pd.DataFrame()

    if input_mode == "Screenshot (OCR)":
        st.markdown("Upload a RotoGrinders contest result screenshot.")
        uploaded = st.file_uploader(
            "Contest screenshot",
            type=["png", "jpg", "jpeg"],
            key="_rr_upload",
        )
        if uploaded is not None:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                from yak_core.contest_ocr import extract_contest_result
                with st.spinner("Extracting results from screenshot..."):
                    result = extract_contest_result(tmp_path)

                if result.players:
                    actuals_df = actuals_from_ocr(result)
                    st.success(
                        f"Extracted {len(result.players)} players "
                        f"({result.total_points:.1f} total FP, "
                        f"confidence: {result.confidence:.0%})"
                    )
                    # Show extracted data
                    st.dataframe(
                        actuals_df[["player_name", "actual_fp", "salary", "pos"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Auto-fill slate date from OCR
                    if result.slate_date and not slate_date:
                        st.session_state["_rr_slate_date"] = result.slate_date
                else:
                    st.warning("Could not extract players from screenshot. Try manual entry.")
            except Exception as e:
                st.error(f"OCR extraction failed: {e}")
            finally:
                os.unlink(tmp_path)

    else:  # Manual Entry
        st.markdown("Paste player results below (one per line: `Player Name, Actual FP`)")
        manual_text = st.text_area(
            "Player results",
            height=200,
            key="_rr_manual",
            placeholder="LeBron James, 52.5\nNikola Jokic, 61.0\nDe'Aaron Fox, 36.5",
        )
        if manual_text.strip():
            rows = []
            for line in manual_text.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        rows.append({
                            "player_name": parts[0],
                            "actual_fp": float(parts[1]),
                        })
                    except ValueError:
                        continue
            if rows:
                actuals_df = pd.DataFrame(rows)
                st.info(f"Parsed {len(actuals_df)} players")

    st.divider()

    # ── Record Button ─────────────────────────────────────────────────
    _has_actuals = not actuals_df.empty
    _has_pool = not pool.empty

    if not _has_pool:
        st.warning("No pool loaded. Load a slate in **The Lab** first so we can compare projections to actuals.")

    if st.button(
        "Record Results",
        type="primary",
        disabled=not (_has_actuals and _has_pool),
        key="_rr_record",
    ):
        with st.spinner("Recording results into calibration + edge feedback..."):
            summary = record_contest_results(
                slate_date=slate_date,
                pool_df=pool,
                actuals_df=actuals_df,
                contest_type=contest_type,
                store=st.session_state.get("_calibration_store"),
            )

        if "error" in summary:
            st.error(f"Recording failed: {summary['error']}")
        else:
            st.success(
                f"Recorded {summary['players_matched']}/{summary['players_total']} "
                f"players matched"
            )

            # Show calibration summary
            cal = summary.get("calibration", {})
            if cal and "error" not in cal:
                overall = cal.get("overall", {})
                if overall:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE", f"{overall.get('mae', 0):.2f}")
                    c2.metric("RMSE", f"{overall.get('rmse', 0):.2f}")
                    c3.metric("Correlation", f"{overall.get('correlation', 0):.4f}")

            # Show edge feedback summary
            edge = summary.get("edge_feedback", {})
            if edge and "error" not in edge:
                signals = edge.get("signals", {})
                if signals:
                    sig_rows = []
                    for sig_name, sig_data in signals.items():
                        sig_rows.append({
                            "Signal": sig_name.replace("_", " ").title(),
                            "Flagged": sig_data.get("n_flagged", 0),
                            "Hit": sig_data.get("n_hit", 0),
                            "Hit Rate": f"{sig_data['hit_rate']:.0%}" if sig_data.get("hit_rate") is not None else "—",
                        })
                    if sig_rows:
                        st.subheader("Edge Signal Outcomes")
                        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

            # Show updated weights
            updated_w = summary.get("updated_weights", {})
            if updated_w:
                st.subheader("Updated Signal Weights")
                w_rows = [{"Signal": k.replace("_", " ").title(), "Weight": f"{v:.1%}"} for k, v in updated_w.items()]
                st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

    # ── Calibration History ───────────────────────────────────────────
    with st.expander("Calibration History", expanded=False):
        cal_summary = get_calibration_summary(
            store=st.session_state.get("_calibration_store")
        )
        if cal_summary.get("n_slates", 0) == 0:
            st.info("No calibration data yet. Record your first contest result above.")
        else:
            st.markdown(f"**{cal_summary['n_slates']} slates** recorded")
            if cal_summary.get("avg_mae"):
                st.markdown(f"Average MAE: **{cal_summary['avg_mae']:.2f}** FP")
            if cal_summary.get("latest_mae"):
                st.markdown(f"Latest MAE: **{cal_summary['latest_mae']:.2f}** FP")

            # Position corrections
            pos_corr = cal_summary.get("position_corrections", [])
            if pos_corr:
                st.caption("Position corrections (applied to future projections)")
                st.dataframe(pd.DataFrame(pos_corr), use_container_width=True, hide_index=True)

            # Tier corrections
            tier_corr = cal_summary.get("tier_corrections", [])
            if tier_corr:
                st.caption("Salary tier corrections")
                st.dataframe(pd.DataFrame(tier_corr), use_container_width=True, hide_index=True)

            if st.button("Clear calibration history", key="_rr_clear_cal"):
                clear_calibration_history(
                    store=st.session_state.get("_calibration_store")
                )
                st.info("Calibration history cleared.")
                st.rerun()

    # ── Edge Feedback History ─────────────────────────────────────────
    with st.expander("Edge Feedback History", expanded=False):
        edge_summary = get_feedback_status(
            store=st.session_state.get("_calibration_store")
        ).get("edge_feedback", {})

        if edge_summary.get("n_slates", 0) == 0:
            st.info("No edge feedback data yet. Record your first contest result above.")
        else:
            st.markdown(f"**{edge_summary['n_slates']} slates** tracked")
            signals = edge_summary.get("signals", [])
            if signals:
                st.dataframe(
                    pd.DataFrame(signals)[["signal", "description", "hit_rate", "weight", "total_calls"]],
                    use_container_width=True,
                    hide_index=True,
                )


main()
