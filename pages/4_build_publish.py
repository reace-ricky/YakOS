"""Build & Publish – YakOS Sprint 1 page.

Responsibilities (S1.5)
-----------------------
- Read roster rules directly from SlateState (Classic / Showdown).
- Support floor / median / ceiling build modes per contest type.
- Exposure management (min / max per player for MME).
- Simple contest selection advisor.
- Build lineups and support DK CSV export.
- "Publish to Edge Share" action per contest type.
- S1.7 late-swap suggestions per contest type using pre-baked GTD rules.

State read:  SlateState, RickyEdgeState (edge_check gate), SimState
State written: LineupSetState
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state,
    get_edge_state,
    get_sim_state,
    get_lineup_state, set_lineup_state,
)
from yak_core.lineups import (  # noqa: E402
    build_multiple_lineups_with_exposure,
    to_dk_upload_format,
    build_showdown_lineups,
    to_dk_showdown_upload_format,
)
from yak_core.calibration import apply_archetype, DFS_ARCHETYPES  # noqa: E402
from yak_core.config import CONTEST_PRESETS, CONTEST_PRESET_LABELS  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BUILD_MODE_COLS = {
    "floor": "floor",
    "median": "proj",
    "ceiling": "ceil",
}
_CONTEST_TO_BUILD_MODE = {
    "GPP - 150 Max": "ceiling",
    "GPP - 20 Max": "ceiling",
    "Single Entry / 3-Max": "median",
    "50/50 / Double-Up": "floor",
    "Showdown": "ceiling",
}


def _render_status_bar(slate: "SlateState", edge: "RickyEdgeState") -> None:
    cols = st.columns([2, 2, 2, 2, 4])
    with cols[0]:
        st.metric("Sport", slate.sport or "—")
    with cols[1]:
        st.metric("Date", slate.slate_date or "—")
    with cols[2]:
        st.metric("Contest", slate.contest_type or "—")
    with cols[3]:
        check = "✅" if edge.ricky_edge_check else "⛔"
        st.metric("Edge Check", check)
    with cols[4]:
        if slate.active_layers:
            chips = " ".join(f"`{l}`" for l in slate.active_layers)
            st.markdown(f"**Layers:** {chips}")


def _apply_sim_learnings(pool: pd.DataFrame, sim: "SimState") -> pd.DataFrame:
    """Apply non-destructive Sim Learnings boosts to effective_proj column."""
    pool = pool.copy()
    if "proj" not in pool.columns:
        return pool
    pool["effective_proj"] = pool["proj"].copy()
    for pname, learning in sim.sim_learnings.items():
        mask = pool.get("player_name", pd.Series(dtype=str)) == pname
        if mask.any():
            boost = float(learning.get("boost", 0))
            pool.loc[mask, "effective_proj"] = pool.loc[mask, "effective_proj"] * (1 + boost)
    return pool


def _get_proj_col(pool: pd.DataFrame, build_mode: str) -> str:
    """Return the best available projection column for the build mode."""
    desired = _BUILD_MODE_COLS.get(build_mode, "proj")
    if desired in pool.columns:
        return desired
    return "proj"


def _build_lineups(
    pool: pd.DataFrame,
    num_lineups: int,
    max_exposure: float,
    min_exposure: float,
    min_salary: int,
    proj_col: str,
    archetype: str,
    slate: "SlateState",
    lock_names: list,
    exclude_names: list,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build lineups using the appropriate engine (Classic / Showdown)."""
    try:
        cfg = {
            "NUM_LINEUPS": num_lineups,
            "SALARY_CAP": slate.salary_cap,
            "MAX_EXPOSURE": max_exposure,
            "MIN_SALARY_USED": min_salary,
            "LOCK": lock_names or [],
            "EXCLUDE": exclude_names or [],
            "PROJ_COL": proj_col,
        }
        if slate.is_showdown:
            lineups_df, expo_df = build_showdown_lineups(pool, cfg)
        else:
            opt_pool = apply_archetype(pool.copy(), archetype)
            lineups_df, expo_df = build_multiple_lineups_with_exposure(opt_pool, cfg)
        return lineups_df, expo_df
    except Exception as exc:
        st.error(f"Optimizer error: {exc}")
        return None, None


def _late_swap_suggestions(
    pool: pd.DataFrame,
    lineups_df: Optional[pd.DataFrame],
    injury_updates: list,
) -> list[dict]:
    """Generate late-swap candidates using pre-baked GTD rules.

    Rules:
    - OUT → pivot to highest-value replacement at same position
    - Limited / GTD → reduce exposure suggestion
    """
    suggestions: list[dict] = []
    if lineups_df is None or lineups_df.empty:
        return suggestions
    if not injury_updates:
        return suggestions

    player_pool_map: dict = {}
    if not pool.empty and "player_name" in pool.columns:
        for _, row in pool.iterrows():
            pname = str(row.get("player_name", ""))
            player_pool_map[pname] = row.to_dict()

    for update in injury_updates:
        pname = str(update.get("player_name", ""))
        status = str(update.get("status", "")).upper()

        if not pname:
            continue

        # Check if player is in lineups
        in_lineups = False
        if "player_name" in lineups_df.columns:
            in_lineups = pname in lineups_df["player_name"].values
        if not in_lineups:
            continue

        if status in ("OUT", "IR", "O"):
            # Find replacement: same position, highest proj, not in lineups
            player_row = player_pool_map.get(pname, {})
            pos = player_row.get("pos", "")
            current_salary = float(player_row.get("salary", 0) or 0)

            in_lineup_players = set(lineups_df["player_name"].tolist()) if "player_name" in lineups_df.columns else set()
            candidates = [
                row for _, row in pool.iterrows()
                if (
                    str(row.get("pos", "")) == pos
                    and str(row.get("player_name", "")) != pname
                    and str(row.get("player_name", "")) not in in_lineup_players
                    and abs(float(row.get("salary", 0) or 0) - current_salary) <= 1500
                    and str(row.get("status", "")).upper() not in ("OUT", "IR", "O")
                )
            ]
            if candidates:
                best = max(candidates, key=lambda r: float(r.get("proj", 0) or 0))
                suggestions.append({
                    "action": "PIVOT",
                    "out_player": pname,
                    "in_player": best.get("player_name", ""),
                    "pos": pos,
                    "salary_delta": int(float(best.get("salary", 0) or 0) - current_salary),
                    "reason": f"{pname} is {status}",
                })
            else:
                suggestions.append({
                    "action": "PIVOT",
                    "out_player": pname,
                    "in_player": "(no replacement found)",
                    "pos": pos,
                    "salary_delta": 0,
                    "reason": f"{pname} is {status}",
                })

        elif status in ("GTD", "Q", "QUESTIONABLE", "DOUBTFUL", "LIMITED"):
            suggestions.append({
                "action": "REDUCE_EXPOSURE",
                "out_player": pname,
                "in_player": "",
                "pos": player_pool_map.get(pname, {}).get("pos", ""),
                "salary_delta": 0,
                "reason": f"{pname} is {status} – reduce exposure",
            })

    return suggestions


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🏗️ Build & Publish")
    st.caption("Configure builds, optimize lineups, export CSVs, and publish to Edge Share.")

    slate = get_slate_state()
    edge = get_edge_state()
    sim = get_sim_state()
    lu_state = get_lineup_state()

    _render_status_bar(slate, edge)
    st.divider()

    # ── Edge Check Gate ───────────────────────────────────────────────────
    if not edge.ricky_edge_check:
        st.error(
            "⛔ **Ricky Edge Check not approved.** "
            "Complete the Edge Check on the **Ricky Edge** page before building lineups."
        )
        st.stop()

    if not slate.is_ready():
        st.warning("⚠️ No slate published. Go to **Slate Hub** and publish a slate first.")
        st.stop()

    pool: pd.DataFrame = slate.player_pool.copy()
    pool = _apply_sim_learnings(pool, sim)

    # ─────────────────────────────────────────────────────────────────────
    # Section 1: Contest Selection Advisor
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🎯 Contest Selection Advisor")

    gauge_summary = sim.contest_gauges
    if gauge_summary:
        advisor_rows = []
        for label in CONTEST_PRESET_LABELS:
            preset = CONTEST_PRESETS.get(label, {})
            # Map preset label → gauge label
            _label_map = {
                "Cash Game": "Cash",
                "Single Entry": "SE",
                "3-Max Tournament": "3-Max",
                "20-Max GPP": "20-Max",
                "MME (150-Max)": "150-Max",
                "Showdown": "20-Max",
            }
            gauge_label = _label_map.get(label, "SE")
            gauge = gauge_summary.get(gauge_label, {})
            score = float(gauge.get("score", 0))
            rec = "✅ Good" if score >= 0.60 else "⚠️ OK" if score >= 0.35 else "❌ Weak"
            advisor_rows.append({
                "Contest": label,
                "Build Mode": _CONTEST_TO_BUILD_MODE.get(label, "median"),
                "Default Lineups": preset.get("default_lineups", 1),
                "Sim Score": f"{int(score * 100)}%",
                "Recommendation": rec,
            })
        st.dataframe(pd.DataFrame(advisor_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run sims in **The Lab** for contest recommendations.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 2: Build Controls
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("⚙️ Build Config")

    col1, col2 = st.columns(2)
    with col1:
        contest_label = st.selectbox("Contest Type", CONTEST_PRESET_LABELS, key="_bp_contest")
        preset = CONTEST_PRESETS.get(contest_label, {})
        default_mode = _CONTEST_TO_BUILD_MODE.get(contest_label, "median")
        build_mode = st.selectbox(
            "Build Mode",
            ["floor", "median", "ceiling"],
            index=["floor", "median", "ceiling"].index(default_mode),
            key="_bp_build_mode",
        )
        archetype = st.selectbox(
            "Archetype",
            list(DFS_ARCHETYPES.keys()),
            index=list(DFS_ARCHETYPES.keys()).index(preset.get("archetype", "Balanced"))
            if preset.get("archetype", "Balanced") in DFS_ARCHETYPES else 0,
            key="_bp_archetype",
        )

    with col2:
        num_lineups = st.number_input(
            "# Lineups", min_value=1, max_value=150,
            value=int(preset.get("default_lineups", 1)),
            key="_bp_num_lineups",
        )
        max_exp = st.slider(
            "Max Exposure", min_value=0.1, max_value=1.0, step=0.05,
            value=float(preset.get("default_max_exposure", 0.5)),
            key="_bp_max_exp",
        )
        min_salary = st.number_input(
            "Min Salary Used", min_value=40000, max_value=50000, step=100,
            value=int(preset.get("min_salary", 48000)),
            key="_bp_min_salary",
        )

    with st.expander("Lock / Exclude Players", expanded=False):
        player_names = sorted(pool["player_name"].dropna().tolist()) if "player_name" in pool.columns else []
        lock_names = st.multiselect("Lock (in every lineup)", player_names, key="_bp_lock")
        exclude_names = st.multiselect("Exclude", player_names, key="_bp_exclude")

    st.caption(
        f"**Roster:** {slate.roster_slots}  |  "
        f"**Cap:** ${slate.salary_cap:,}  |  "
        f"**Type:** {slate.contest_type}"
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 3: Build Lineups
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🔨 Build Lineups")

    proj_col = _get_proj_col(pool, build_mode)

    if st.button("▶️ Build Lineups", type="primary", key="_bp_build"):
        with st.spinner(f"Building {num_lineups} {contest_label} lineups…"):
            lineups_df, expo_df = _build_lineups(
                pool,
                num_lineups=int(num_lineups),
                max_exposure=float(max_exp),
                min_exposure=0.0,
                min_salary=int(min_salary),
                proj_col=proj_col,
                archetype=str(archetype),
                slate=slate,
                lock_names=list(lock_names),
                exclude_names=list(exclude_names),
            )
            if lineups_df is not None:
                lu_state.set_lineups(
                    contest_label,
                    lineups_df,
                    {
                        "build_mode": build_mode,
                        "num_lineups": num_lineups,
                        "max_exposure": max_exp,
                        "min_salary": min_salary,
                        "archetype": archetype,
                        "proj_col": proj_col,
                    },
                )
                if expo_df is not None:
                    lu_state.exposures[contest_label] = expo_df
                set_lineup_state(lu_state)
                st.success(f"Built {num_lineups} lineups for **{contest_label}**.")

    # ── Show lineups ──────────────────────────────────────────────────────
    built_labels = [lbl for lbl, df in lu_state.lineups.items() if df is not None and not df.empty]
    if built_labels:
        view_label = st.selectbox("View lineups for", built_labels, key="_bp_view_label")
        view_df = lu_state.lineups.get(view_label)

        if view_df is not None and not view_df.empty:
            st.caption(f"{len(view_df['lineup_index'].unique()) if 'lineup_index' in view_df.columns else '?'} lineups")
            st.dataframe(view_df, use_container_width=True, hide_index=True)

            # Exposure view
            expo_df = lu_state.exposures.get(view_label)
            if expo_df is not None and not expo_df.empty:
                with st.expander("Player Exposures", expanded=False):
                    st.dataframe(expo_df, use_container_width=True, hide_index=True)

            st.divider()

            # ── DK CSV Export ─────────────────────────────────────────────
            st.subheader("📥 DK CSV Export")
            if st.button("📊 Prepare DK CSV", key="_bp_prep_csv"):
                try:
                    if slate.is_showdown:
                        csv_df = to_dk_showdown_upload_format(view_df)
                    else:
                        csv_df = to_dk_upload_format(view_df)
                    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
                    fname = f"yakos_{view_label.replace(' ', '_').lower()}_{slate.slate_date}.csv"
                    st.download_button(
                        label="⬇️ Download DK Upload CSV",
                        data=csv_bytes,
                        file_name=fname,
                        mime="text/csv",
                        key="_bp_download_csv",
                    )
                except Exception as exc:
                    st.error(f"CSV export failed: {exc}")

            # ── Publish to Edge Share ─────────────────────────────────────
            st.divider()
            st.subheader("📤 Publish to Edge Share")
            if st.button(f"✅ Publish {view_label} to Edge Share", type="primary", key="_bp_publish"):
                _ts = datetime.now(timezone.utc).isoformat()
                lu_state.publish(view_label, _ts)
                set_lineup_state(lu_state)
                st.success(f"✅ **{view_label}** published to Edge Share at {_ts}")
                st.balloons()

    else:
        st.info("Build lineups above to see results and export options.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 4: Late Swap Suggestions (S1.7)
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("⚡ Late Swap Suggestions")
    st.caption("Pre-baked GTD rules: OUT → pivot, GTD/Limited → reduce exposure.")

    injury_updates = st.session_state.get("_hub_injury_updates", [])
    if not injury_updates:
        st.info("No injury updates loaded. Use **Slate Hub → Refresh Injuries** to fetch updates.")
    elif built_labels:
        swap_label = st.selectbox("Contest for swap suggestions", built_labels, key="_bp_swap_label")
        swap_df = lu_state.lineups.get(swap_label)
        suggestions = _late_swap_suggestions(pool, swap_df, injury_updates)

        if suggestions:
            st.warning(f"⚠️ {len(suggestions)} swap suggestion(s) for **{swap_label}**:")
            st.dataframe(pd.DataFrame(suggestions), use_container_width=True, hide_index=True)
        else:
            st.success("✅ No late-swap actions needed for this contest.")
    else:
        st.info("Build lineups first to generate late-swap suggestions.")


main()
