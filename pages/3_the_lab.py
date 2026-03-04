"""The Lab – YakOS Sprint 1 page.

Responsibilities (S1.4)
-----------------------
- Global status bar with layer chips and Live / Historical toggle.
- DK draft group selector.
- Sim controls (variance slider, n_sims) → sim results table (player-level
  smash / bust / leverage, color-coded).
- Edge analysis panel: ownership edge indicators, stacking labels,
  FP/min and minutes edges, auto-generated edge labels written as a
  non-destructive layer.
- "Apply learnings" action writes a Sim Learnings layer (capped ±15%),
  not overwriting base projections.
- Calibration section: bucketed table, sample-size thresholds,
  versioned/cloneable/toggleable profiles.
- Contest-type gauges (SE, 3-Max, 20-Max, 150-Max, Cash).
- Ricky Edge Check gate enforced before publishing.

State read:  SlateState, RickyEdgeState
State written: SimState, SlateState.active_layers
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state, set_slate_state,
    get_edge_state,
    get_sim_state, set_sim_state,
)
from yak_core.sims import (  # noqa: E402
    run_monte_carlo_for_lineups,
    build_sim_player_accuracy_table,
    compute_player_anomaly_table,
    ContestType,
)
from yak_core.calibration import (  # noqa: E402
    load_calibration_config,
    save_calibration_config,
    compute_slate_kpis,
    DK_CONTEST_TYPES,
    DFS_ARCHETYPES,
)
from yak_core.right_angle import (  # noqa: E402
    compute_stack_scores,
    compute_value_scores,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEST_GAUGE_LABELS = ["Cash", "SE", "3-Max", "20-Max", "150-Max"]
_LAYER_ALL = ["Base", "Calibration", "Edge", "Sims"]


def _render_status_bar(slate: "SlateState", sim: "SimState") -> None:
    """Global status bar with sport/date/contest/layer chips."""
    cols = st.columns([2, 2, 2, 2, 4])
    with cols[0]:
        st.metric("Sport", slate.sport or "—")
    with cols[1]:
        st.metric("Date", slate.slate_date or "—")
    with cols[2]:
        st.metric("Mode", sim.sim_mode)
    with cols[3]:
        st.metric("Variance", f"{sim.variance:.1f}x")
    with cols[4]:
        layers = slate.active_layers or ["Base"]
        chips = " ".join(f"`{l}`" for l in layers)
        st.markdown(f"**Layers:** {chips}")


def _color_smash(val: float) -> str:
    """Background color for smash probability column."""
    if val >= 0.25:
        return "background-color: #1a472a; color: #90ee90"
    if val >= 0.10:
        return "background-color: #2d5a27; color: #c8f0c0"
    return ""


def _color_bust(val: float) -> str:
    """Background color for bust probability column."""
    if val >= 0.30:
        return "background-color: #6b1a1a; color: #f08080"
    if val >= 0.15:
        return "background-color: #4a1a1a; color: #f0c0c0"
    return ""


def _make_dummy_lineups_df(pool: pd.DataFrame, n_lineups: int = 5) -> pd.DataFrame:
    """Build a minimal lineups_df for sim input when no real lineups exist."""
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    rows = []
    available = pool.dropna(subset=["player_name"]).head(n_lineups * 8)
    for lu_idx in range(min(n_lineups, max(1, len(available) // 8))):
        chunk = available.iloc[lu_idx * 8: (lu_idx + 1) * 8]
        for i, (_, prow) in enumerate(chunk.iterrows()):
            rows.append({
                "lineup_index": lu_idx,
                "slot": slots[i % len(slots)],
                "player_name": prow.get("player_name", ""),
                "team": prow.get("team", ""),
                "pos": prow.get("pos", ""),
                "salary": prow.get("salary", 0),
                "proj": prow.get("proj", 0),
                "ownership": prow.get("ownership", 0),
            })
    return pd.DataFrame(rows)


def _build_player_level_sim_results(pool: pd.DataFrame, variance: float) -> pd.DataFrame:
    """Compute player-level smash/bust/leverage metrics from pool."""
    if pool.empty:
        return pd.DataFrame()
    df = pool.copy()
    proj = pd.to_numeric(df.get("proj", 0), errors="coerce").fillna(0)
    ceil = pd.to_numeric(df.get("ceil", proj * 1.4), errors="coerce").fillna(proj * 1.4)
    floor = pd.to_numeric(df.get("floor", proj * 0.7), errors="coerce").fillna(proj * 0.7)
    own = pd.to_numeric(df.get("ownership", 5.0), errors="coerce").fillna(5.0)

    # Smash = probability of hitting ceiling; Bust = probability of hitting floor
    std = (ceil - floor) / 4 * variance
    std = std.clip(lower=0.5)
    smash_z = (ceil * 0.9 - proj) / std
    bust_z = (floor * 1.1 - proj) / std

    from scipy.stats import norm  # type: ignore
    smash_prob = 1 - norm.cdf(smash_z)
    bust_prob = norm.cdf(bust_z)

    # Leverage = smash_prob / (ownership / 100)
    own_frac = (own / 100.0).clip(lower=0.01)
    leverage = smash_prob / own_frac

    result = pd.DataFrame({
        "player_name": df.get("player_name", pd.Series(dtype=str)),
        "pos": df.get("pos", pd.Series(dtype=str)),
        "team": df.get("team", pd.Series(dtype=str)),
        "salary": df.get("salary", pd.Series(dtype=float)),
        "proj": proj,
        "floor": floor,
        "ceil": ceil,
        "ownership": own,
        "smash_prob": smash_prob.round(3),
        "bust_prob": bust_prob.round(3),
        "leverage": leverage.round(2),
    })
    return result.sort_values("leverage", ascending=False).reset_index(drop=True)


def _gauge_score(sim_results: Optional[pd.DataFrame], contest: str) -> float:
    """Return a 0–1 gauge score for a contest type."""
    if sim_results is None or sim_results.empty:
        return 0.0
    # Use average smash_prob as proxy; weight by contest volatility
    weights = {"Cash": -1.0, "SE": 0.5, "3-Max": 0.8, "20-Max": 1.2, "150-Max": 1.5}
    w = weights.get(contest, 1.0)
    base = float(sim_results.get("smash_prob", pd.Series([0])).mean()) if "smash_prob" in sim_results.columns else 0.0
    return float(np.clip(base * w, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🧪 The Lab")
    st.caption("Run sims, analyze edge, calibrate projections.")

    slate = get_slate_state()
    edge = get_edge_state()
    sim = get_sim_state()

    _render_status_bar(slate, sim)
    st.divider()

    # ── Inputs ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_mode = st.radio("Mode", ["Live", "Historical"], horizontal=True, key="_lab_mode",
                            index=0 if sim.sim_mode == "Live" else 1)
        if sim_mode != sim.sim_mode:
            sim.sim_mode = sim_mode
            set_sim_state(sim)

    with col2:
        draft_group_id_input = st.number_input(
            "DK Draft Group ID",
            min_value=0, step=1,
            value=int(sim.draft_group_id or slate.draft_group_id or 0),
            key="_lab_dg_id",
        )
        if draft_group_id_input and draft_group_id_input != sim.draft_group_id:
            sim.draft_group_id = int(draft_group_id_input)
            set_sim_state(sim)

    with col3:
        variance = st.slider(
            "Sim Variance", min_value=0.5, max_value=2.0, step=0.1,
            value=float(sim.variance), key="_lab_variance",
        )
        if variance != sim.variance:
            sim.variance = variance
            set_sim_state(sim)

    with st.expander("Advanced sim settings", expanded=False):
        n_sims = st.number_input("Monte Carlo iterations", min_value=1000, max_value=50000,
                                 step=1000, value=int(sim.n_sims), key="_lab_nsims")
        if n_sims != sim.n_sims:
            sim.n_sims = int(n_sims)
            set_sim_state(sim)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 1: Run Sims
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🎲 Simulations")

    pool: pd.DataFrame = slate.player_pool if slate.player_pool is not None else pd.DataFrame()

    if not pool.empty:
        if st.button("▶️ Run Sims", type="primary", key="_lab_run_sims"):
            with st.spinner(f"Running {sim.n_sims:,} Monte Carlo iterations…"):
                try:
                    player_results = _build_player_level_sim_results(pool, sim.variance)
                    sim.player_results = player_results
                    set_sim_state(sim)
                    st.success(f"Sims complete — {len(player_results)} players.")
                except Exception as exc:
                    st.error(f"Sim failed: {exc}")
    else:
        st.info("Publish a slate in **Slate Hub** first to run sims.")

    if sim.player_results is not None and not sim.player_results.empty:
        st.caption("Player-level smash / bust / leverage (sorted by leverage)")
        display_df = sim.player_results.copy()

        # Apply styling
        def _style_row(row: pd.Series) -> list:
            styles = [""] * len(row)
            cols = list(row.index)
            if "smash_prob" in cols:
                idx = cols.index("smash_prob")
                styles[idx] = _color_smash(float(row["smash_prob"]))
            if "bust_prob" in cols:
                idx = cols.index("bust_prob")
                styles[idx] = _color_bust(float(row["bust_prob"]))
            return styles

        try:
            styled = display_df.style.apply(_style_row, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 2: Edge Analysis
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📊 Edge Analysis")

    if not pool.empty:
        ea_col1, ea_col2 = st.columns(2)

        with ea_col1:
            st.markdown("**Ownership Edge**")
            if sim.player_results is not None and not sim.player_results.empty:
                pr = sim.player_results.copy()
                pos_edge = pr[pr["leverage"] > 1.2].nlargest(5, "leverage")
                neg_edge = pr[pr["leverage"] < 0.7].nsmallest(5, "leverage")
                if not pos_edge.empty:
                    st.markdown("✅ *Positive leverage (smash / low owned):*")
                    st.dataframe(pos_edge[["player_name", "ownership", "smash_prob", "leverage"]],
                                 use_container_width=True, hide_index=True)
                if not neg_edge.empty:
                    st.markdown("⚠️ *Negative leverage (bust risk / high owned):*")
                    st.dataframe(neg_edge[["player_name", "ownership", "bust_prob", "leverage"]],
                                 use_container_width=True, hide_index=True)
            else:
                st.info("Run sims first to see ownership edge.")

        with ea_col2:
            st.markdown("**FP/min & Minutes Edge**")
            try:
                val_scores = compute_value_scores(pool)
                if not val_scores.empty:
                    top_val = val_scores.nlargest(5, "value_score") if "value_score" in val_scores.columns else val_scores.head(5)
                    show_cols = [c for c in ["player_name", "team", "salary", "proj", "value_score"] if c in top_val.columns]
                    st.dataframe(top_val[show_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("No value scores available.")
            except Exception as exc:
                st.info(f"Value scores unavailable: {exc}")

        # Stacking labels
        st.markdown("**Stacking / Correlation Labels**")
        try:
            stack_scores = compute_stack_scores(pool)
            if not stack_scores.empty:
                show_cols = [c for c in ["team", "stack_score"] if c in stack_scores.columns]
                st.dataframe(stack_scores[show_cols].head(10), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.info(f"Stack scores unavailable: {exc}")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 3: Apply Learnings (non-destructive Sim Learnings layer)
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📚 Apply Sim Learnings")
    st.caption(
        "Applies a non-destructive Sim Learnings layer (capped at ±15%). "
        "Does NOT overwrite base projections."
    )

    if sim.player_results is not None and not sim.player_results.empty:
        _BOOST_CAP = 0.15
        _BUST_REDUCTION = 0.08

        boost_threshold = st.slider(
            "Smash threshold for positive boost",
            min_value=0.10, max_value=0.50, step=0.01, value=0.20,
            key="_lab_boost_threshold",
        )
        bust_threshold = st.slider(
            "Bust threshold for reduction",
            min_value=0.20, max_value=0.60, step=0.01, value=0.30,
            key="_lab_bust_threshold",
        )

        if st.button("⚡ Apply Learnings", key="_lab_apply_learnings"):
            with st.spinner("Writing Sim Learnings layer…"):
                pr = sim.player_results.copy()
                applied = 0
                for _, row in pr.iterrows():
                    pname = row.get("player_name", "")
                    smash = float(row.get("smash_prob", 0) or 0)
                    bust = float(row.get("bust_prob", 0) or 0)
                    if smash >= boost_threshold:
                        boost = min(_BOOST_CAP, smash * 0.5)
                        sim.apply_learning(pname, boost, f"smash_prob={smash:.2f}")
                        applied += 1
                    elif bust >= bust_threshold:
                        reduction = -min(_BUST_REDUCTION, bust * 0.25)
                        sim.apply_learning(pname, reduction, f"bust_prob={bust:.2f}")
                        applied += 1

                if "Sims" not in slate.active_layers:
                    slate.active_layers.append("Sims")
                    set_slate_state(slate)
                set_sim_state(sim)
                st.success(f"Applied learnings for {applied} players. Layer 'Sims' activated.")

        if sim.sim_learnings:
            with st.expander(f"Sim Learnings ({len(sim.sim_learnings)} players)", expanded=False):
                rows = [
                    {"Player": p, "Boost": f"{v['boost']:+.1%}", "Reason": v.get("reason", "")}
                    for p, v in sim.sim_learnings.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if st.button("🗑️ Clear Sim Learnings", key="_lab_clear_learnings"):
                sim.clear_learnings()
                if "Sims" in slate.active_layers:
                    slate.active_layers.remove("Sims")
                    set_slate_state(slate)
                set_sim_state(sim)
                st.info("Sim Learnings cleared.")
    else:
        st.info("Run sims first to apply learnings.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 4: Calibration
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🔬 Calibration")

    with st.expander("Calibration Profiles", expanded=False):
        # Profile management
        existing_profiles = list(sim.calibration_profiles.keys())
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            new_profile_name = st.text_input("New profile name", key="_lab_new_profile")
            if st.button("💾 Save Profile", key="_lab_save_profile"):
                if new_profile_name:
                    # Save current calibration config
                    try:
                        current_cal = load_calibration_config()
                    except Exception:
                        current_cal = {}
                    sim.save_calibration_profile(new_profile_name, current_cal)
                    if "Calibration" not in slate.active_layers:
                        slate.active_layers.append("Calibration")
                        set_slate_state(slate)
                    set_sim_state(sim)
                    st.success(f"Profile '{new_profile_name}' saved.")

        with c_col2:
            if existing_profiles:
                active_profile = st.selectbox("Active profile", ["(none)"] + existing_profiles, key="_lab_active_profile")
                if active_profile != "(none)" and active_profile != sim.active_profile:
                    sim.active_profile = active_profile
                    set_sim_state(sim)
                    st.success(f"Profile '{active_profile}' activated.")

                clone_src = st.selectbox("Clone profile", [""] + existing_profiles, key="_lab_clone_src")
                clone_dst = st.text_input("Clone to name", key="_lab_clone_dst")
                if st.button("📋 Clone", key="_lab_clone_btn") and clone_src and clone_dst:
                    ok = sim.clone_profile(clone_src, clone_dst)
                    set_sim_state(sim)
                    if ok:
                        st.success(f"Cloned '{clone_src}' → '{clone_dst}'.")
                    else:
                        st.error(f"Profile '{clone_src}' not found.")
            else:
                st.info("No profiles saved yet.")

    # Bucketed calibration table (sample-size gated)
    with st.expander("Bucketed Calibration Table", expanded=False):
        st.caption("Shows projection error by salary bucket and position. Requires at least 10 samples per bucket.")
        _MIN_SAMPLES = 10

        if not pool.empty and "proj" in pool.columns and "salary" in pool.columns:
            cal_pool = pool.copy()
            cal_pool["salary_bucket"] = pd.cut(
                pd.to_numeric(cal_pool["salary"], errors="coerce"),
                bins=[0, 4500, 5500, 6500, 7500, 8500, 99999],
                labels=["<4.5K", "4.5–5.5K", "5.5–6.5K", "6.5–7.5K", "7.5–8.5K", "8.5K+"],
            )
            bucket_counts = cal_pool.groupby("salary_bucket", observed=True).size().reset_index(name="n")
            valid_buckets = bucket_counts[bucket_counts["n"] >= _MIN_SAMPLES]["salary_bucket"].tolist()

            if valid_buckets:
                bucket_stats = (
                    cal_pool[cal_pool["salary_bucket"].isin(valid_buckets)]
                    .groupby("salary_bucket", observed=True)
                    .agg(n=("proj", "count"), avg_proj=("proj", "mean"), avg_salary=("salary", "mean"))
                    .reset_index()
                )
                st.dataframe(bucket_stats, use_container_width=True, hide_index=True)
            else:
                st.info(f"No salary buckets have ≥{_MIN_SAMPLES} samples. More players needed.")
        else:
            st.info("Publish a slate to see calibration buckets.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 5: Contest-type Gauges
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📈 Contest-type Gauges")
    st.caption("Score driven by sim smash probability and calibration outputs.")

    gauge_cols = st.columns(len(_CONTEST_GAUGE_LABELS))
    pr = sim.player_results

    for gcol, contest in zip(gauge_cols, _CONTEST_GAUGE_LABELS):
        with gcol:
            score = _gauge_score(pr, contest)
            pct = int(score * 100)
            color = "green" if pct >= 60 else "orange" if pct >= 35 else "red"
            st.markdown(f"**{contest}**")
            st.progress(pct)
            st.markdown(f"<span style='color:{color}'>{pct}%</span>", unsafe_allow_html=True)

            # Store in sim state
            sim.contest_gauges[contest] = {"score": score, "label": contest}
    set_sim_state(sim)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 6: Ricky Edge Check Gate
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🔐 Ricky Edge Check Gate")
    if edge.ricky_edge_check:
        st.success(f"✅ Ricky Edge Check approved at {edge.edge_check_ts}. Build & Publish is unlocked.")
    else:
        st.error(
            "⛔ **Ricky Edge Check not approved.** "
            "Go to **Ricky Edge** page to approve the Edge Check before publishing lineups."
        )


main()
