"""Ricky Edge – YakOS Sprint 1 page.

Responsibilities
----------------
- Tag players as core / secondary / value / punt / fade with conviction
  levels (1–5).
- Tag game environments (pace, totals, game stack targets).
- Define two- and three-man stacks.
- Support free-text slate notes entry.
- All tags, stacks, labels, and notes are persisted into RickyEdgeState only.

State written: RickyEdgeState exclusively.
State read:    SlateState.player_pool (for the player picker).
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from yak_core.state import (  # noqa: E402
    get_slate_state,
    get_edge_state,
    set_edge_state,
)
from yak_core.right_angle import (  # noqa: E402
    compute_game_environment_cards,
    compute_tiered_stack_alerts,
)
# Shared slate context and lab analysis helpers (factored into yak_core.context
# so The Lab, Ricky Edge, and Build & Publish all read the same data).
from yak_core.context import get_slate_context, get_lab_analysis  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLAYER_TAGS = ["core", "secondary", "value", "punt", "fade"]
_TAG_COLORS = {
    "core": "🟢",
    "secondary": "🔵",
    "value": "🟡",
    "punt": "⚪",
    "fade": "🔴",
}
_CONV_LABELS = {1: "1 – Low", 2: "2", 3: "3 – Mid", 4: "4", 5: "5 – High"}


def _render_status_bar(slate: "SlateState", edge: "RickyEdgeState") -> None:
    cols = st.columns([2, 2, 2, 2, 4])
    with cols[0]:
        st.metric("Sport", slate.sport or "—")
    with cols[1]:
        st.metric("Date", slate.slate_date or "—")
    with cols[2]:
        st.metric("Contest", slate.contest_type or "—")
    with cols[3]:
        check_icon = "✅" if edge.ricky_edge_check else "⛔"
        st.metric("Edge Check", check_icon)
    with cols[4]:
        if slate.active_layers:
            chips = " ".join(f"`{l}`" for l in slate.active_layers)
            st.markdown(f"**Layers:** {chips}")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🎯 Ricky Edge")
    st.caption("Tag players, environments, and stacks for this slate.")

    # Nav order: Slate Hub → The Lab → Ricky Edge → Build & Publish → Friends / Edge Share
    slate = get_slate_state()
    edge = get_edge_state()
    _render_status_bar(slate, edge)

    if not slate.is_ready():
        st.warning("⚠️ No slate published yet. Go to **Slate Hub** and publish a slate first.")

    # Use get_lab_analysis() as the default data source for player pool and sim metrics.
    # Falls back to the base player pool from SlateState when sims have not been run.
    _analysis = get_lab_analysis()
    pool: pd.DataFrame = _analysis["pool"] if not _analysis["pool"].empty else (
        slate.player_pool if slate.player_pool is not None else pd.DataFrame()
    )
    player_names = sorted(pool["player_name"].dropna().tolist()) if not pool.empty and "player_name" in pool.columns else []

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 1: Slate Notes
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📝 Slate Notes")
    notes = st.text_area(
        "Notes for this slate",
        value=edge.slate_notes,
        height=100,
        key="_re_notes",
        placeholder="e.g. 'Heavy rain in Chicago – avoid pass-catchers. LAL pace game is a smash.'",
    )
    if notes != edge.slate_notes:
        edge.slate_notes = notes
        set_edge_state(edge)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 2: Player Tagging
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🏷️ Player Tags")

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.caption("Add a tag for a player")
        if player_names:
            pick_player = st.selectbox("Player", [""] + player_names, key="_re_pick_player")
        else:
            pick_player = st.text_input("Player name", key="_re_pick_player_text")

        pick_tag = st.selectbox(
            "Tag",
            _PLAYER_TAGS,
            format_func=lambda t: f"{_TAG_COLORS.get(t, '')} {t.title()}",
            key="_re_pick_tag",
        )
        pick_conviction = st.select_slider(
            "Conviction", options=[1, 2, 3, 4, 5],
            format_func=lambda v: _CONV_LABELS[v],
            value=3,
            key="_re_pick_conv",
        )

        col_add, col_rm = st.columns(2)
        with col_add:
            if st.button("➕ Tag", key="_re_add_tag") and pick_player:
                edge.tag_player(pick_player, pick_tag, pick_conviction)
                set_edge_state(edge)
                st.success(f"Tagged {pick_player} → {pick_tag} ({pick_conviction})")
        with col_rm:
            if st.button("➖ Remove", key="_re_rm_tag") and pick_player:
                edge.remove_tag(pick_player)
                set_edge_state(edge)
                st.info(f"Removed tag for {pick_player}")

    with col_right:
        st.caption("Current tags")
        if edge.player_tags:
            rows = [
                {
                    "Player": p,
                    "Tag": f"{_TAG_COLORS.get(v['tag'], '')} {v['tag'].title()}",
                    "Conviction": _CONV_LABELS.get(v["conviction"], str(v["conviction"])),
                }
                for p, v in sorted(edge.player_tags.items())
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Quick summary per tag
            for tag in _PLAYER_TAGS:
                tagged = edge.get_tagged(tag)
                if tagged:
                    st.markdown(f"**{_TAG_COLORS.get(tag, '')} {tag.title()}**: {', '.join(tagged)}")
        else:
            st.info("No players tagged yet.")

    # Auto-generate tags from pool signals
    if not pool.empty and st.button("⚡ Auto-suggest tags from pool", key="_re_auto_tag"):
        with st.spinner("Analyzing pool signals…"):
            try:
                suggestions = []
                if "proj" in pool.columns and "salary" in pool.columns:
                    top_proj = pool.nlargest(3, "proj")["player_name"].tolist()
                    high_value = pool.copy()
                    high_value["_vfp"] = high_value["proj"] / (high_value["salary"] / 1000.0)
                    value_plays = high_value.nlargest(5, "_vfp")
                    low_salary = value_plays[value_plays["salary"] < 6500]["player_name"].tolist()

                    for p in top_proj:
                        if p not in edge.player_tags:
                            suggestions.append((p, "core", 3))
                    for p in low_salary:
                        if p not in edge.player_tags:
                            suggestions.append((p, "value", 2))

                if suggestions:
                    for p, tag, conv in suggestions:
                        edge.tag_player(p, tag, conv)
                    set_edge_state(edge)
                    st.success(f"Auto-tagged {len(suggestions)} players.")
                else:
                    st.info("No auto-tag suggestions (pool may have minimal signal).")
            except Exception as exc:
                st.error(f"Auto-tag failed: {exc}")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 3: Game Environment Tags
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🌡️ Game Environment Tags")

    games = []
    if not pool.empty and "team" in pool.columns and "opponent" in pool.columns:
        seen = set()
        for _, row in pool.iterrows():
            t, o = row.get("team", ""), row.get("opponent", "")
            if t and o:
                key = tuple(sorted([t, o]))
                if key not in seen:
                    seen.add(key)
                    games.append(f"{t} @ {o}")

    if games:
        with st.expander("Tag game environments", expanded=False):
            for game_key in games:
                parts = game_key.split(" @ ")
                home = parts[1] if len(parts) > 1 else game_key
                away = parts[0] if len(parts) > 0 else game_key
                st.markdown(f"**{game_key}**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    pace = st.selectbox(
                        "Pace",
                        ["neutral", "fast", "slow"],
                        key=f"_re_pace_{game_key}",
                        index=["neutral", "fast", "slow"].index(
                            edge.game_tags.get(game_key, {}).get("pace", "neutral")
                        ),
                    )
                with c2:
                    total = st.number_input(
                        "O/U Total",
                        min_value=0.0, max_value=300.0, step=0.5,
                        value=float(edge.game_tags.get(game_key, {}).get("total", 220.0)),
                        key=f"_re_total_{game_key}",
                    )
                with c3:
                    environment = st.selectbox(
                        "Environment",
                        ["neutral", "smash", "avoid"],
                        key=f"_re_env_{game_key}",
                        index=["neutral", "smash", "avoid"].index(
                            edge.game_tags.get(game_key, {}).get("environment", "neutral")
                        ),
                    )
                with c4:
                    stack_target = st.checkbox(
                        "Stack target",
                        value=edge.game_tags.get(game_key, {}).get("stack_target", False),
                        key=f"_re_stack_tgt_{game_key}",
                    )

                edge.game_tags[game_key] = {
                    "pace": pace,
                    "total": total,
                    "environment": environment,
                    "stack_target": stack_target,
                }
            set_edge_state(edge)
    else:
        st.info("No games available. Publish a slate first to tag game environments.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 4: Stack Definitions
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📚 Stack Definitions")

    with st.expander("Add a new stack", expanded=not bool(edge.stacks)):
        teams = sorted(pool["team"].dropna().unique().tolist()) if not pool.empty and "team" in pool.columns else []
        stack_team = st.selectbox("Team", [""] + teams, key="_re_stack_team")

        # Players from that team
        if stack_team and not pool.empty and "team" in pool.columns:
            team_players = sorted(
                pool[pool["team"] == stack_team]["player_name"].dropna().tolist()
            )
        else:
            team_players = player_names

        stack_players = st.multiselect("Players (2–3)", team_players, key="_re_stack_players")
        stack_rationale = st.text_input("Rationale (optional)", key="_re_stack_rationale")

        if st.button("➕ Add Stack", key="_re_add_stack"):
            if stack_team and len(stack_players) >= 2:
                edge.add_stack(stack_team, stack_players, stack_rationale)
                set_edge_state(edge)
                st.success(f"Stack added: {stack_team} – {', '.join(stack_players)}")
            else:
                st.warning("Select a team and at least 2 players.")

    if edge.stacks:
        st.caption(f"{len(edge.stacks)} stack(s) defined")
        rows = []
        for i, s in enumerate(edge.stacks):
            rows.append({
                "#": i + 1,
                "Team": s.get("team", ""),
                "Players": ", ".join(s.get("players", [])),
                "Rationale": s.get("rationale", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear all stacks", key="_re_clear_stacks"):
            edge.stacks = []
            set_edge_state(edge)
            st.info("All stacks cleared.")
    else:
        st.info("No stacks defined yet.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 5: Edge Labels (auto-generated)
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🏷️ Edge Labels")

    if st.button("⚡ Generate Edge Labels", key="_re_gen_labels"):
        with st.spinner("Generating edge labels…"):
            labels: list[str] = []

            # Ownership edge labels
            for p, v in edge.player_tags.items():
                if v["tag"] == "core" and v["conviction"] >= 4:
                    labels.append(f"SMASH: {p} (conviction {v['conviction']})")
                elif v["tag"] == "fade":
                    labels.append(f"FADE: {p} – ownership trap")
                elif v["tag"] == "value" and v["conviction"] >= 3:
                    labels.append(f"VALUE: {p} – low-owned upside play")

            # Stack labels
            for s in edge.stacks:
                labels.append(f"STACK: {s['team']} ({', '.join(s.get('players', [])[:3])})")

            # Game environment labels
            for gk, gt in edge.game_tags.items():
                if gt.get("environment") == "smash":
                    labels.append(f"SMASH GAME: {gk} (O/U {gt.get('total', '?')})")
                elif gt.get("pace") == "fast":
                    labels.append(f"PACE GAME: {gk}")
                elif gt.get("stack_target"):
                    labels.append(f"STACK TARGET: {gk}")

            # Auto-suggestions from pool signals
            if not pool.empty:
                try:
                    env_cards = compute_game_environment_cards(pool)
                    if env_cards:
                        for card in env_cards[:3]:
                            lbl = card.get("label", "")
                            if lbl:
                                labels.append(f"AUTO: {lbl}")
                except Exception:
                    pass

                try:
                    stack_alerts = compute_tiered_stack_alerts(pool)
                    if isinstance(stack_alerts, pd.DataFrame) and not stack_alerts.empty:
                        top = stack_alerts.head(3)
                        for _, row in top.iterrows():
                            team = row.get("team", "")
                            score = row.get("stack_score", "")
                            if team:
                                labels.append(f"STACK ALERT: {team} (score {score:.1f})" if score else f"STACK ALERT: {team}")
                except Exception:
                    pass

            edge.edge_labels = labels
            set_edge_state(edge)
            st.success(f"Generated {len(labels)} edge labels.")

    if edge.edge_labels:
        for lbl in edge.edge_labels:
            st.markdown(f"- {lbl}")
    else:
        st.info("No edge labels yet. Click **Generate Edge Labels**.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Section 6: Ricky Edge Check Gate
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🔐 Ricky Edge Check Gate")
    st.caption(
        "This gate must be approved before lineups can be published in Build & Publish. "
        "Approve only when you are satisfied with the tags, stacks, and edge labels."
    )

    if edge.ricky_edge_check:
        st.success(f"✅ Ricky Edge Check approved at {edge.edge_check_ts} UTC")
        if st.button("🔓 Revoke Edge Check", key="_re_revoke"):
            edge.revoke_edge_check()
            set_edge_state(edge)
            st.warning("Edge Check revoked.")
    else:
        st.error("⛔ Edge Check not approved")
        can_approve = bool(edge.player_tags) or bool(edge.stacks) or bool(edge.edge_labels)
        if not can_approve:
            st.info("Tag at least one player or define a stack before approving.")

        if st.button("✅ Approve Ricky Edge Check", type="primary", key="_re_approve", disabled=not can_approve):
            _ts = datetime.now(timezone.utc).isoformat()
            edge.approve_edge_check(_ts)
            set_edge_state(edge)
            st.success(f"✅ Ricky Edge Check approved at {_ts} UTC")
            st.balloons()


main()
