"""Tuning Lab — unified config tuning UI with run history and hit-rate tracking.

Design:
- Single page with contest-type selector at the top.
- Sliders bound to the *active config* for that contest type.
- Ceiling Hunter is the fixed projection baseline for all GPP presets (always ON,
  not shown as a slider — labelled as a status chip).
- Apply → snapshot sliders into a run_history row, mark as active.
- Reset to active → revert sliders to the active config.
- Results chart and run history table (per contest type).
- Legacy Sim Lab report and Ems lines chart moved to a collapsed expander.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from yak_core.config import (
    CONTEST_PRESETS,
    NAMED_PROFILES,
    merge_config,
    is_gpp_preset,
    CEILING_HUNTER_CAL_PROFILE,
)
from yak_core.tuning_lab import (
    TuningRunRow,
    TuningLabStore,
    compute_hit_rates_across_slates,
    snapshot_params,
    TRACKED_PARAM_KEYS,
    CASH_LINE_BY_CONTEST,
    MIN_LINEUPS_FOR_HISTORY,
)
from utils.constants import (
    NBA_GAME_STYLES,
    NBA_CONTEST_TYPES_BY_STYLE,
    CONTEST_PROFILE_KEY_MAP,
    PROFILE_KEY_TO_PRESET,
    PROFILE_KEY_TO_NAMED,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contest type ↔ preset mapping (shared with auto_calibrate)
# ---------------------------------------------------------------------------

_CONTEST_TYPES = ["SE GPP", "MME GPP", "Cash", "Showdown GPP", "Showdown Cash"]

_CONTEST_TYPE_TO_PRESET: Dict[str, str] = {
    "SE GPP":       "GPP Main",
    "MME GPP":      "GPP Early",
    "Cash":         "Cash Main",
    "Showdown GPP": "Showdown",
    "Showdown Cash": "Cash Game",
}

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

_STORE: Optional[TuningLabStore] = None


def _get_store() -> TuningLabStore:
    global _STORE
    if _STORE is None:
        _STORE = TuningLabStore()
    return _STORE


def _ephemeral_slider_key(contest_type: str) -> str:
    return f"tuning_lab_ephemeral_{contest_type}"


def _get_ephemeral(contest_type: str) -> Dict[str, Any]:
    """Return the ephemeral (unapplied) slider state for a contest type."""
    return dict(st.session_state.get(_ephemeral_slider_key(contest_type), {}))


def _set_ephemeral(contest_type: str, overrides: Dict[str, Any]) -> None:
    st.session_state[_ephemeral_slider_key(contest_type)] = dict(overrides)


def _seed_ephemeral_from_active(contest_type: str, preset_name: str) -> None:
    """Seed ephemeral sliders from the active config (or preset defaults)."""
    store = _get_store()
    active = store.get_active_config(contest_type)
    if active is not None:
        overrides = {}
        ricky_weights = {}
        for k in TRACKED_PARAM_KEYS:
            v = active.get(k)
            if v is not None and pd.notna(v):
                if k in ("w_gpp", "w_ceil", "w_own"):
                    ricky_weights[k] = float(v)
                else:
                    overrides[k] = float(v)
        _set_ephemeral(contest_type, {"overrides": overrides, "ricky_weights": ricky_weights})
    else:
        # Fall back to preset defaults
        preset = CONTEST_PRESETS.get(preset_name, {})
        cfg = merge_config(preset)
        overrides = {k: cfg.get(k, 0.0) for k in TRACKED_PARAM_KEYS
                     if k not in ("w_gpp", "w_ceil", "w_own")}
        ricky_weights = {"w_gpp": 0.0, "w_ceil": 1.0, "w_own": 0.15}
        _set_ephemeral(contest_type, {"overrides": overrides, "ricky_weights": ricky_weights})


# ---------------------------------------------------------------------------
# Slider helpers
# ---------------------------------------------------------------------------

_SLIDER_SPECS: Dict[str, tuple] = {
    # (label, min, max, step, format)
    "GPP_PROJ_WEIGHT":          ("Proj Weight",         0.0,  1.0,  0.05, "%.2f"),
    "GPP_UPSIDE_WEIGHT":        ("Upside Weight",       0.0,  1.0,  0.05, "%.2f"),
    "GPP_BOOM_WEIGHT":          ("Boom Weight",         0.0,  1.0,  0.05, "%.2f"),
    "GPP_BOOM_SPREAD_WEIGHT":   ("Boom Spread",         0.0,  0.50, 0.05, "%.2f"),
    "GPP_SMASH_WEIGHT":         ("Smash Weight",        0.0,  0.50, 0.05, "%.2f"),
    "GPP_SNIPER_WEIGHT":        ("Sniper Weight",       0.0,  0.50, 0.05, "%.2f"),
    "GPP_EFFICIENCY_WEIGHT":    ("Efficiency Weight",   0.0,  0.30, 0.05, "%.2f"),
    "GPP_OWN_PENALTY_STRENGTH": ("Own Penalty",         0.0,  3.0,  0.10, "%.1f"),
    "GPP_BUST_PENALTY":         ("Bust Penalty",        0.0,  0.50, 0.05, "%.2f"),
    "GPP_LEVERAGE_WEIGHT":      ("Leverage Weight",     0.0,  0.30, 0.05, "%.2f"),
    "CASH_FLOOR_WEIGHT":        ("Floor Weight",        0.0,  1.0,  0.05, "%.2f"),
    "MAX_EXPOSURE":             ("Max Exposure",        0.10, 1.0,  0.05, "%.2f"),
    "OWN_WEIGHT":               ("Ownership Weight",    0.0,  0.50, 0.05, "%.2f"),
    "w_gpp":                    ("Ricky GPP Wt",        0.0,  2.0,  0.10, "%.2f"),
    "w_ceil":                   ("Ricky Ceil Wt",       0.0,  2.0,  0.10, "%.2f"),
    "w_own":                    ("Ricky Own Wt",        0.0,  2.0,  0.10, "%.2f"),
}

_SLIDER_GROUPS: Dict[str, List[str]] = {
    "Core Weights": [
        "GPP_PROJ_WEIGHT", "GPP_UPSIDE_WEIGHT", "GPP_BOOM_WEIGHT",
        "GPP_OWN_PENALTY_STRENGTH",
    ],
    "Edge Signals": [
        "GPP_SMASH_WEIGHT", "GPP_EFFICIENCY_WEIGHT", "GPP_BUST_PENALTY",
        "GPP_LEVERAGE_WEIGHT", "GPP_BOOM_SPREAD_WEIGHT", "GPP_SNIPER_WEIGHT",
    ],
    "Ownership & Exposure": [
        "OWN_WEIGHT", "MAX_EXPOSURE", "CASH_FLOOR_WEIGHT",
    ],
    "Ricky Ranking": [
        "w_gpp", "w_ceil", "w_own",
    ],
}


def _get_slider_default(preset_name: str, key: str) -> float:
    """Return the preset default for a slider key."""
    preset = CONTEST_PRESETS.get(preset_name, {})
    cfg = merge_config(preset)
    return float(cfg.get(key, 0.0))


def _render_slider_groups(
    contest_type: str,
    preset_name: str,
    ephemeral: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Render all slider groups. Returns (optimizer_overrides, ricky_weights)."""
    overrides: Dict[str, Any] = dict(ephemeral.get("overrides", {}))
    ricky: Dict[str, float] = dict(ephemeral.get("ricky_weights", {}))

    for group_label, keys in _SLIDER_GROUPS.items():
        with st.expander(group_label, expanded=(group_label == "Core Weights")):
            cols = st.columns(2)
            for i, k in enumerate(keys):
                spec = _SLIDER_SPECS.get(k)
                if spec is None:
                    continue
                label, mn, mx, step, fmt = spec
                default = _get_slider_default(preset_name, k)
                if k in ("w_gpp", "w_ceil", "w_own"):
                    current = float(ricky.get(k, default))
                else:
                    current = float(overrides.get(k, default))
                current = max(float(mn), min(float(mx), current))
                with cols[i % 2]:
                    val = st.slider(
                        label,
                        min_value=float(mn),
                        max_value=float(mx),
                        value=current,
                        step=float(step),
                        format=fmt,
                        key=f"tl_{contest_type}_{k}",
                    )
                if k in ("w_gpp", "w_ceil", "w_own"):
                    ricky[k] = val
                else:
                    overrides[k] = val

    return overrides, ricky


# ---------------------------------------------------------------------------
# Run history table + chart
# ---------------------------------------------------------------------------

def _render_run_history_table(contest_type: str, store: TuningLabStore) -> None:
    """Render the run history table for a contest type."""
    df = store.load_for_contest_type(contest_type)
    if df.empty:
        st.caption("No applied runs yet — adjust sliders and click Apply.")
        return

    st.subheader("Run History")

    # Sort newest first
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    display = df.copy()
    if "timestamp" in display.columns:
        display["When"] = pd.to_datetime(display["timestamp"]).dt.strftime("%m/%d %I:%M %p")
    display["Active"] = display.get("is_active", False).apply(
        lambda x: "✅" if x else ""
    )
    display["Source"] = display.get("source", "batch")

    # Format hit rates
    for col, label in [("hit_rate_cash", "Cash%"), ("hit_rate_top5", "Top5%"), ("hit_rate_top1", "Top1%")]:
        display[label] = display.get(col, pd.Series([None] * len(display))).apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
        )

    show_cols = ["Active", "When", "Source", "label", "num_lineups", "num_slates",
                 "Cash%", "Top5%", "Top1%", "avg_actual_fp"]
    show_cols = [c for c in show_cols if c in display.columns]
    rename = {
        "label": "Label",
        "num_lineups": "Lineups",
        "num_slates": "Slates",
        "avg_actual_fp": "Avg Actual FP",
    }
    display = display[show_cols].rename(columns=rename)

    # Highlight active row
    def _highlight(row: pd.Series):
        if row.get("Active", "") == "✅":
            return ["background-color: rgba(0, 255, 135, 0.10)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style.apply(_highlight, axis=1),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Avg Actual FP": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def _render_results_chart(contest_type: str, store: TuningLabStore) -> None:
    """Render the hit-rate trend chart for applied runs."""
    df = store.load_for_contest_type(contest_type)
    if df.empty:
        return

    # Only rows with at least one hit-rate result
    result_cols = ["hit_rate_cash", "hit_rate_top5", "hit_rate_top1"]
    has_results = df[result_cols].notna().any(axis=1)
    df = df[has_results].sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        st.caption("Run results will appear here after slates complete.")
        return

    st.subheader("Hit Rate Trend")
    chart_df = pd.DataFrame({
        "Run": range(1, len(df) + 1),
        "Cash%": df["hit_rate_cash"].fillna(0),
        "Top5%": df["hit_rate_top5"].fillna(0),
        "Top1%": df["hit_rate_top1"].fillna(0),
    }).set_index("Run")

    st.line_chart(chart_df, use_container_width=True)
    st.caption(f"X = applied run #, Y = % of lineups hitting each band ({contest_type})")


# ---------------------------------------------------------------------------
# Main Tuning Lab renderer
# ---------------------------------------------------------------------------

def render_tuning_lab(sport: str = "NBA") -> None:
    """Render the Tuning Lab tab."""
    st.header("🎛️ Tuning Lab")
    st.caption(
        "Adjust optimizer weights on top of the Ceiling Hunter projection baseline. "
        "Only *Applied* runs update the active config and appear in history."
    )

    if sport != "NBA":
        st.info("Tuning Lab is NBA-only for now. Use Sim Lab for PGA.")
        return

    # ── Top controls ──────────────────────────────────────────────────────
    col_ct, col_slate = st.columns([2, 3])
    with col_ct:
        contest_type = st.selectbox(
            "Contest Type",
            _CONTEST_TYPES,
            key="tuning_lab_contest_type",
        )
    preset_name = _CONTEST_TYPE_TO_PRESET.get(contest_type, "GPP Main")

    # Ceiling Hunter status chip (always ON for GPP)
    if is_gpp_preset(preset_name):
        st.markdown(
            '<span style="background:#1a4a1a;color:#00c851;padding:3px 10px;'
            'border-radius:12px;font-size:0.8rem;font-weight:600;">'
            '🏹 Ceiling Hunter ON (fixed GPP baseline)</span>',
            unsafe_allow_html=True,
        )

    # Seed ephemeral sliders on first render (or when contest type changes)
    _ct_prev_key = "_tuning_lab_prev_ct"
    if st.session_state.get(_ct_prev_key) != contest_type:
        st.session_state[_ct_prev_key] = contest_type
        _seed_ephemeral_from_active(contest_type, preset_name)

    ephemeral = _get_ephemeral(contest_type)

    # ── Sliders ────────────────────────────────────────────────────────────
    overrides, ricky_weights = _render_slider_groups(contest_type, preset_name, ephemeral)
    _set_ephemeral(contest_type, {"overrides": overrides, "ricky_weights": ricky_weights})

    # ── Apply / Reset buttons ──────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        apply_clicked = st.button(
            "✅ Apply", type="primary", use_container_width=True,
            key="tuning_lab_apply",
        )
    with btn_col2:
        reset_clicked = st.button(
            "↩️ Reset to Active", use_container_width=True,
            key="tuning_lab_reset",
        )

    if reset_clicked:
        _seed_ephemeral_from_active(contest_type, preset_name)
        st.rerun()

    if apply_clicked:
        _apply_config(contest_type, preset_name, overrides, ricky_weights)
        st.rerun()

    # ── Run history and chart ─────────────────────────────────────────────
    store = _get_store()
    tab_hist, tab_chart, tab_player = st.tabs(
        ["📋 Run History", "📈 Hit Rate Chart", "🔍 Player Diagnostics"]
    )
    with tab_hist:
        _render_run_history_table(contest_type, store)
    with tab_chart:
        _render_results_chart(contest_type, store)
    with tab_player:
        _render_player_diagnostics(contest_type)


def _apply_config(
    contest_type: str,
    preset_name: str,
    optimizer_overrides: Dict[str, Any],
    ricky_weights: Dict[str, float],
    source: str = "batch",
    label: str = "",
    per_date_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Snapshot current sliders into a run_history row and mark as active.

    Parameters
    ----------
    per_date_results : list, optional
        If provided, hit rates are computed immediately.  Otherwise they remain
        None and can be filled later via ``store.update_results()``.
    """
    store = _get_store()
    row = TuningRunRow.from_params(
        contest_type=contest_type,
        source=source,
        preset_name=preset_name,
        optimizer_overrides=optimizer_overrides,
        ricky_weights=ricky_weights,
        label=label,
    )

    if per_date_results:
        hit_rates = compute_hit_rates_across_slates(per_date_results, contest_type)
        row.hit_rate_cash = hit_rates["hit_rate_cash"]
        row.hit_rate_top5 = hit_rates["hit_rate_top5"]
        row.hit_rate_top1 = hit_rates["hit_rate_top1"]

        total_lineups = sum(
            len(r.get("summary_df", pd.DataFrame())) for r in per_date_results
            if r.get("summary_df") is not None
        )
        row.num_lineups = total_lineups
        row.num_slates = len(per_date_results)

        actuals_all = []
        for r in per_date_results:
            sdf = r.get("summary_df")
            if sdf is not None and "total_actual" in sdf.columns:
                actuals_all.extend(
                    pd.to_numeric(sdf["total_actual"], errors="coerce").dropna().tolist()
                )
        if actuals_all:
            row.avg_actual_fp = round(sum(actuals_all) / len(actuals_all), 1)

    row.is_active = True
    store.append(row)
    store.set_active(row.run_id, contest_type)

    _has_rates = (
        row.hit_rate_cash is not None
        and row.hit_rate_top5 is not None
        and row.hit_rate_top1 is not None
    )
    st.success(
        f"✅ Config applied for {contest_type} "
        f"(run #{row.run_id}). "
        + (f"Hit cash: {row.hit_rate_cash:.1f}%, "
           f"top5: {row.hit_rate_top5:.1f}%, "
           f"top1: {row.hit_rate_top1:.1f}%"
           if _has_rates else "Results will be filled after slates complete.")
    )


# ---------------------------------------------------------------------------
# Auto-cal "Apply" integration
# ---------------------------------------------------------------------------

def apply_auto_cal_result(
    contest_type: str,
    preset_name: str,
    best_params: Dict[str, Any],
    best_ricky_weights: Dict[str, float],
    per_date_results: Optional[List[Dict[str, Any]]] = None,
    label: str = "",
) -> None:
    """Apply an auto-calibration result to the run history.

    Called from the Sim Lab after a successful auto-cal run.
    """
    _apply_config(
        contest_type=contest_type,
        preset_name=preset_name,
        optimizer_overrides=best_params,
        ricky_weights=best_ricky_weights,
        source="auto_cal",
        label=label,
        per_date_results=per_date_results,
    )
    # Also seed the ephemeral sliders with the new best params
    _set_ephemeral(contest_type, {
        "overrides": best_params,
        "ricky_weights": best_ricky_weights,
    })


# ---------------------------------------------------------------------------
# Player diagnostics (lightweight)
# ---------------------------------------------------------------------------

def _render_player_diagnostics(contest_type: str) -> None:
    """Render the per-player diagnostics table for a slate."""
    st.caption(
        "Upload a player pool CSV and (optionally) actuals CSV to inspect "
        "Ceiling Hunter projections + active optimizer config."
    )
    pool_file = st.file_uploader(
        "Player pool CSV (with proj, ceil, floor)", type="csv",
        key=f"tl_pool_csv_{contest_type}",
    )
    act_file = st.file_uploader(
        "Actuals CSV (optional, with actual_fp)", type="csv",
        key=f"tl_act_csv_{contest_type}",
    )

    if pool_file is None:
        return

    try:
        pool_df = pd.read_csv(pool_file)
        pool_df.columns = [c.strip() for c in pool_df.columns]
    except Exception as exc:
        st.error(f"Failed to read pool CSV: {exc}")
        return

    actuals_df = None
    if act_file is not None:
        try:
            actuals_df = pd.read_csv(act_file)
            actuals_df.columns = [c.strip() for c in actuals_df.columns]
        except Exception as exc:
            st.warning(f"Failed to read actuals CSV: {exc}")

    from yak_core.tuning_lab import build_player_diagnostic_table
    diag = build_player_diagnostic_table(pool_df, actuals_df, contest_type)

    if diag.empty:
        st.warning("No diagnostic data to display.")
        return

    st.dataframe(
        diag,
        hide_index=True,
        use_container_width=True,
        column_config={
            "salary": st.column_config.NumberColumn("Salary", format="$%d"),
            "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "ceil": st.column_config.NumberColumn("Ceil", format="%.1f"),
            "floor": st.column_config.NumberColumn("Floor", format="%.1f"),
            "ownership_proj": st.column_config.NumberColumn("Own%", format="%.1f%%"),
            "fp_actual": st.column_config.NumberColumn("Actual", format="%.1f"),
            "delta_proj_actual": st.column_config.NumberColumn("Δ Proj", format="%+.1f"),
            "delta_ceil_actual": st.column_config.NumberColumn("Δ Ceil", format="%+.1f"),
        },
    )
    st.download_button(
        "📥 Download diagnostics CSV",
        diag.to_csv(index=False),
        file_name=f"tuning_lab_player_diag_{contest_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
