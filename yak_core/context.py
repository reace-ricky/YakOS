"""yak_core.context – shared slate context helpers.

Centralises all "slate context" loading so pages don't re-derive the same
fields independently.  Import these helpers instead of calling
``get_slate_state()`` / ``get_sim_state()`` directly in multiple pages.

Usage
-----
    from yak_core.context import get_slate_context, get_lab_analysis

    ctx = get_slate_context()
    # ctx.sport, ctx.slate_date, ctx.contest_type, ctx.draft_group_id, ...

    analysis = get_lab_analysis()
    # analysis["pool"]         – player pool DataFrame with sim metrics
    # analysis["player_results"] – per-player smash/bust/leverage table
    # analysis["sim_learnings"]  – applied learnings dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# SlateContext – lightweight snapshot of SlateState fields used across pages
# ---------------------------------------------------------------------------

@dataclass
class SlateContext:
    """Snapshot of the shared slate configuration fields.

    Fields mirror the most-read attributes of ``SlateState``; use this
    instead of importing the full state object in page code.
    """

    sport: str = "NBA"
    site: str = "DK"
    slate_date: str = ""
    contest_id: Optional[int] = None
    draft_group_id: Optional[int] = None
    game_type_id: Optional[int] = None
    contest_name: str = ""
    contest_type: str = "Classic"
    is_showdown: bool = False
    roster_slots: list = field(default_factory=lambda: ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"])
    salary_cap: int = 50000
    player_pool: Optional[pd.DataFrame] = None
    active_layers: list = field(default_factory=lambda: ["Base"])
    published: bool = False


def get_slate_context() -> SlateContext:
    """Return a :class:`SlateContext` populated from the current ``SlateState``.

    This is the canonical way for pages (The Lab, Ricky Edge, Build & Publish)
    to read shared slate configuration without duplicating the
    ``get_slate_state()`` call pattern.

    Returns
    -------
    SlateContext
        Snapshot of the current slate fields.  When no slate has been
        published, the context is returned with its default (empty) values.
    """
    try:
        # Lazy import so this module is usable outside Streamlit (tests, etc.)
        from yak_core.state import get_slate_state  # noqa: PLC0415

        state = get_slate_state()
        return SlateContext(
            sport=state.sport,
            site=state.site,
            slate_date=state.slate_date,
            contest_id=state.contest_id,
            draft_group_id=state.draft_group_id,
            game_type_id=state.game_type_id,
            contest_name=state.contest_name,
            contest_type=state.contest_type,
            is_showdown=state.is_showdown,
            roster_slots=list(state.roster_slots),
            salary_cap=state.salary_cap,
            player_pool=state.player_pool,
            active_layers=list(state.active_layers),
            published=state.published,
        )
    except Exception:
        return SlateContext()


# ---------------------------------------------------------------------------
# LabAnalysis – finalized player pool + metrics produced by The Lab
# ---------------------------------------------------------------------------

def get_lab_analysis() -> Dict[str, Any]:
    """Return the finalized player pool and sim metrics from The Lab.

    This is the canonical data source for pages that consume The Lab's
    output (primarily Ricky Edge and Build & Publish).  It merges the
    base player pool from ``SlateState`` with the per-player sim results
    stored in ``SimState``, applying any active Sim Learnings adjustments.

    Returns
    -------
    dict with keys:
        ``pool``             – player pool DataFrame (base + sim annotations)
        ``player_results``   – per-player smash / bust / leverage table
                               (empty DataFrame if sims have not been run)
        ``sim_learnings``    – dict of ``{player_name: {boost, reason}}``
                               (empty dict if no learnings applied)
        ``variance``         – variance multiplier used in the last sim run
        ``n_sims``           – Monte Carlo iterations used in the last run
    """
    result: Dict[str, Any] = {
        "pool": pd.DataFrame(),
        "player_results": pd.DataFrame(),
        "sim_learnings": {},
        "variance": 1.0,
        "n_sims": 10000,
        "edge_df": pd.DataFrame(),
    }

    try:
        from yak_core.state import get_slate_state, get_sim_state  # noqa: PLC0415

        slate = get_slate_state()
        sim = get_sim_state()

        pool = slate.player_pool if slate.player_pool is not None else pd.DataFrame()

        # Merge sim results into pool when available
        if sim.player_results is not None and not sim.player_results.empty:
            sim_cols = ["player_name", "smash_prob", "bust_prob", "leverage"]
            available = [c for c in sim_cols if c in sim.player_results.columns]
            if not pool.empty and "player_name" in pool.columns and len(available) > 1:
                # Deduplicate sim results by player_name before merge to
                # prevent row multiplication (Showdown pools can produce
                # duplicate sim entries for CPT/FLEX variants).
                _sim_merge = sim.player_results[available].drop_duplicates(
                    subset=["player_name"], keep="first"
                )
                pool = pool.merge(_sim_merge, on="player_name", how="left")

        result["pool"] = pool
        result["player_results"] = sim.player_results if sim.player_results is not None else pd.DataFrame()
        result["sim_learnings"] = dict(sim.sim_learnings) if sim.sim_learnings else {}
        result["variance"] = float(sim.variance)
        result["n_sims"] = int(sim.n_sims)
        result["edge_df"] = slate.edge_df if slate.edge_df is not None else pd.DataFrame()

    except Exception:
        pass

    return result
