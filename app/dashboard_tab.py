"""Tab 4: Dashboard (admin only).

Displays calibration health, signal accuracy, published data status,
and post-slate feedback controls.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


def render_dashboard_tab(sport: str) -> None:
    """Render the Dashboard tab."""
    from app.data_loader import load_calibration_data, load_signal_history, published_dir, load_fresh_meta

    # ═══════════════════════════════════════════════════
    # Calibration Health
    # ═══════════════════════════════════════════════════
    st.markdown("### Calibration Health")

    try:
        from yak_core.calibration_feedback import get_calibration_summary
        summary = get_calibration_summary(sport=sport.upper())
    except Exception as e:
        st.warning(f"Could not load calibration data: {e}")
        summary = {"status": "no_data", "n_slates": 0}

    if summary.get("status") == "no_data":
        st.info("No calibration data recorded yet.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Slates tracked", summary.get("n_slates", 0))
        with c2:
            bias = summary.get("overall_bias", 0)
            st.metric("Overall bias", f"{bias:+.2f}" if bias else "0.00")
        with c3:
            avg_mae = summary.get("avg_mae")
            st.metric("Avg MAE", f"{avg_mae:.2f}" if avg_mae else "N/A")

        dates = summary.get("dates", [])
        if dates:
            st.caption(f"Date range: {dates[-1]} → {dates[0]}")

        # Position corrections
        pos_corr = summary.get("position_corrections", [])
        if pos_corr:
            st.markdown("**Corrections by position:**")
            st.dataframe(pd.DataFrame(pos_corr), use_container_width=True, hide_index=True)

        # Salary tier corrections
        tier_corr = summary.get("tier_corrections", [])
        if tier_corr:
            st.markdown("**Corrections by salary tier:**")
            st.dataframe(pd.DataFrame(tier_corr), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════
    # Signal Accuracy
    # ═══════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Signal Accuracy")

    signal_history = load_signal_history()
    if not signal_history:
        st.info("No signal history data found.")
    else:
        slates = signal_history.get("slates", [])
        if isinstance(slates, list) and slates:
            recent = slates[-10:]
            rows = []
            for s in recent:
                rows.append({
                    "date": s.get("date", ""),
                    "sport": s.get("sport", ""),
                    "signals_flagged": s.get("n_flagged", s.get("signals_flagged", 0)),
                    "hit_rate": s.get("hit_rate", 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Per-signal breakdown
        by_signal = signal_history.get("by_signal", {})
        if by_signal:
            st.markdown("**Per-signal breakdown:**")
            sig_rows = []
            for sig_name, sig_data in by_signal.items():
                sig_rows.append({
                    "signal": sig_name,
                    "n_flagged": sig_data.get("n_flagged", 0),
                    "avg_actual": round(sig_data.get("avg_actual", 0), 1),
                    "avg_proj": round(sig_data.get("avg_proj", 0), 1),
                    "hit_rate": round(sig_data.get("hit_rate", 0), 2),
                })
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════
    # Published Data Status
    # ═══════════════════════════════════════════════════
    st.markdown("---")
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

    # ═══════════════════════════════════════════════════
    # Post-Slate Feedback
    # ═══════════════════════════════════════════════════
    st.markdown("---")
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
            import os
            api_key = os.environ.get("RAPIDAPI_KEY") or os.environ.get("TANK01_RAPIDAPI_KEY", "")
            if not api_key:
                try:
                    api_key = st.secrets.get("RAPIDAPI_KEY", "")
                except Exception:
                    pass
            if not api_key:
                return {"status": "error", "message": "Missing RAPIDAPI_KEY for fetching actuals"}

            actuals = fetch_actuals_from_api(slate_date, api_key)
            if actuals.empty:
                return {"status": "error", "message": f"No actuals available for {slate_date}"}

            # Merge actuals into pool
            pool_with_actuals = pool.merge(
                actuals[["player_name", "actual_fp"]].rename(columns={"actual_fp": "actual"}),
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

        has_actual = pool_with_actuals["actual"].notna().sum()
        if has_actual == 0:
            return {"status": "error", "message": "No actuals matched to pool players"}

        record_slate_errors(pool_with_actuals, slate_date=slate_date, sport=sport.upper())

        summary = get_calibration_summary(sport=sport.upper())
        return {
            "status": "ok",
            "message": f"Recorded errors for {has_actual} players",
            "calibration_update": summary,
        }
    except Exception as e:
        return {"status": "error", "message": f"Calibration update failed: {e}"}
