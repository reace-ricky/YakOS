"""yak_core.rickys_take -- Deterministic voice generation for Ricky's Take.

Generates three sections for the Edge Analysis tab:
  1. Last Night -- recap of previous slate hits/misses in Ricky's voice
  2. Tonight's Edges -- data-driven callouts about the current slate
  3. Bust Call -- one bold prediction for the biggest underperformer

All text is template-based with data slots. No LLM calls.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# 1. LAST NIGHT -- recap generation
# ---------------------------------------------------------------------------

# Hit templates: player outperformed projection
_HIT_TEMPLATES = [
    "{name} went off for {actual:.0f} against a supposedly tough matchup — the model had that one at {proj:.0f}, close enough.",
    "{name} dropped {actual:.0f} on a {proj:.0f} projection. Model liked it, reality loved it.",
    "{name} smashed at {actual:.0f}. Projected {proj:.0f}. The kind of night that pays for the whole week.",
    "{name} crushed with {actual:.0f} actual on a {proj:.0f} line. Easy money for anyone who was paying attention.",
]

# Miss templates: player underperformed projection
_MISS_TEMPLATES = [
    "{name} owners got cooked. {actual:.0f} actual on a {proj:.0f} projection. Ugly.",
    "{name} bricked hard — {actual:.0f} on a {proj:.0f} line. The field learned that one the expensive way.",
    "{name} at {actual:.0f} actual vs {proj:.0f} projected. Sometimes the chalk just crumbles.",
    "{name} laid an egg. {actual:.0f} fantasy points on a {proj:.0f} projection. Pain.",
]

# Pattern templates (optional third sentence)
_PATTERN_TEMPLATES = [
    "Road favorites went {record} on the night. Worth tracking.",
    "Minutes killed. The back end of the rotation got nothing across the board.",
    "High-salary plays went {hit_rate} on the night. Chalk held.",
    "Chalk crumbled across the board. Only {hit_count} of the top-{total} salaries cleared projection.",
]


def _pick_template(templates: list, seed: int) -> str:
    """Deterministic template selection based on a numeric seed."""
    return templates[seed % len(templates)]


def generate_last_night(recap: Optional[Dict[str, Any]]) -> Optional[str]:
    """Generate a 2-3 sentence recap paragraph in Ricky's voice.

    Parameters
    ----------
    recap : dict or None
        Output of slate_recap.get_previous_slate_recap(). Contains players
        list with player_name, projected, actual, delta, label, salary.

    Returns
    -------
    str or None
        Recap paragraph, or None if no data available.
    """
    if recap is None:
        return None

    players = recap.get("players", [])
    if not players:
        return None

    # Find top hits (biggest positive delta) and top misses (biggest negative delta)
    hits = sorted(
        [p for p in players if p["delta"] > 0 and p["salary"] >= 4000],
        key=lambda p: p["delta"],
        reverse=True,
    )
    misses = sorted(
        [p for p in players if p["delta"] < 0 and p["salary"] >= 4000],
        key=lambda p: p["delta"],
    )

    if not hits and not misses:
        return None

    sentences = []

    # Best hit
    if hits:
        h = hits[0]
        seed = int(h["actual"] * 10) + int(h["projected"] * 10)
        tmpl = _pick_template(_HIT_TEMPLATES, seed)
        sentences.append(
            tmpl.format(name=h["player_name"], actual=h["actual"], proj=h["projected"])
        )

    # Worst miss
    if misses:
        m = misses[0]
        seed = int(abs(m["actual"]) * 10) + int(m["projected"] * 10)
        tmpl = _pick_template(_MISS_TEMPLATES, seed)
        sentences.append(
            tmpl.format(name=m["player_name"], actual=m["actual"], proj=m["projected"])
        )

    # Optional pattern sentence -- high-salary performance summary
    summary = recap.get("summary", {})
    if summary and len(sentences) < 3:
        top_players = [p for p in players if p["salary"] >= 6000]
        if top_players:
            top_hits = sum(1 for p in top_players if p["delta"] >= 0)
            total = len(top_players)
            if top_hits >= total * 0.7:
                sentences.append(
                    f"High-salary plays went {top_hits}-of-{total} on the night. Chalk held."
                )
            elif top_hits <= total * 0.3:
                sentences.append(
                    f"Chalk crumbled across the board. Only {top_hits} of the top-{total} salaries cleared projection."
                )

    return " ".join(sentences) if sentences else None


# ---------------------------------------------------------------------------
# 2. TONIGHT'S EDGES -- forward-looking callouts
# ---------------------------------------------------------------------------

def _find_salary_mismatches(pool: pd.DataFrame) -> List[str]:
    """Find players where projection implies much more value than salary."""
    required = {"player_name", "salary", "proj"}
    if not required.issubset(pool.columns):
        return []

    df = pool[pool["proj"] > 5].copy()
    if df.empty:
        return []

    # Implied value: proj points / (salary / 1000)
    df["pts_per_1k"] = df["proj"] / (df["salary"] / 1000)
    median_pts_per_1k = df["pts_per_1k"].median()

    if median_pts_per_1k <= 0:
        return []

    # Implied fair salary from projection
    df["implied_salary"] = (df["proj"] / median_pts_per_1k) * 1000
    df["underpriced_pct"] = (df["implied_salary"] - df["salary"]) / df["salary"]

    # Players underpriced by >15%
    mispriced = df[df["underpriced_pct"] > 0.15].nlargest(3, "underpriced_pct")

    callouts = []
    for _, row in mispriced.iterrows():
        name = row["player_name"]
        sal = int(row["salary"])
        proj = row["proj"]
        pts_1k = row["pts_per_1k"]
        callouts.append(
            f"{name} at ${sal:,} is the kind of mispricing you see maybe once a week. {proj:.0f} projected at {pts_1k:.1f} pts/$1K. That's free money."
        )
    return callouts


def _find_ownership_traps(pool: pd.DataFrame) -> List[str]:
    """Find high-owned players with red flags."""
    required = {"player_name", "salary", "proj", "ownership"}
    if not required.issubset(pool.columns):
        return []

    df = pool[pool["ownership"] > 0].copy()
    if df.empty:
        return []

    # Top 5 by ownership
    top_owned = df.nlargest(5, "ownership")

    callouts = []
    for _, row in top_owned.iterrows():
        name = row["player_name"]
        own = row["ownership"]
        sal = int(row["salary"])
        flags = []

        # Bad DvP matchup (high rank = bad)
        if "dvp_rank" in df.columns and pd.notna(row.get("dvp_rank")):
            dvp = row["dvp_rank"]
            if dvp >= 25:
                rank_label = 31 - int(dvp)  # rank 30 → "top-1", rank 25 → "top-6"
                flags.append(f"faces a top-{rank_label} defense at the position")

        # Blowout risk
        if "blowout_risk" in df.columns and row.get("blowout_risk", 0) > 0.5:
            flags.append("blowout risk could cap minutes")

        # Big spread (wrong side -- underdog getting blown out)
        if "spread" in df.columns and pd.notna(row.get("spread")):
            spread = row["spread"]
            if spread > 5:
                flags.append(f"+{spread:.1f} spread means garbage time risk")

        if flags:
            flag_str = " and ".join(flags)
            callouts.append(
                f"{own:.0f}% of the field is on {name} tonight. {flag_str.capitalize()}. Trap."
            )

    return callouts[:2]  # Max 2 ownership traps


def _find_contrarian_windows(pool: pd.DataFrame) -> List[str]:
    """Find low-owned players with strong situational edges."""
    required = {"player_name", "salary", "proj", "ownership"}
    if not required.issubset(pool.columns):
        return []

    df = pool[(pool["ownership"] > 0) & (pool["proj"] > 10)].copy()
    if df.empty:
        return []

    # Bottom quartile by ownership
    own_25th = df["ownership"].quantile(0.25)
    low_owned = df[df["ownership"] <= max(own_25th, 5)].copy()

    if low_owned.empty:
        return []

    # Score contrarian value: good DvP + high projection relative to salary
    low_owned["pts_per_1k"] = low_owned["proj"] / (low_owned["salary"] / 1000)

    callouts = []
    for _, row in low_owned.nlargest(5, "pts_per_1k").iterrows():
        name = row["player_name"]
        own = row["ownership"]
        sal = int(row["salary"])
        proj = row["proj"]

        reason_parts = []

        # Good DvP (low rank = good matchup)
        if "dvp_rank" in pool.columns and pd.notna(row.get("dvp_rank")):
            dvp = row["dvp_rank"]
            if dvp <= 8:
                reason_parts.append(
                    f"opponent gives up the {_ordinal(int(dvp))}-most FP to {row.get('pos', 'the position')}"
                )

        # Pace-up / high total environment
        if "over_under" in pool.columns and pd.notna(row.get("over_under")):
            ou = row["over_under"]
            if ou >= 230:
                reason_parts.append(f"{ou:.0f} O/U game environment")

        if reason_parts:
            reason = " and ".join(reason_parts)
            callouts.append(
                f"Nobody's looking at {name} at {own:.0f}% owned but {reason}. Sneaky."
            )
        elif row["pts_per_1k"] > 5.0:
            callouts.append(
                f"{name} at ${sal:,} with {proj:.0f} projected and only {own:.0f}% owned. The field is sleeping."
            )

    return callouts[:2]  # Max 2


def _find_game_environment_edges(pool: pd.DataFrame) -> List[str]:
    """Find game environment edges: high totals, pace advantages."""
    if "over_under" not in pool.columns or "player_name" not in pool.columns:
        return []

    df = pool[pool["over_under"].notna() & (pool["proj"] > 15)].copy()
    if df.empty:
        return []

    # Find games with highest over/under
    if "game_id" in df.columns:
        game_ous = df.groupby("game_id")["over_under"].first().nlargest(1)
        if not game_ous.empty:
            top_game = game_ous.index[0]
            ou = game_ous.iloc[0]
            if ou >= 232:
                game_players = df[df["game_id"] == top_game]
                teams = game_players["team"].unique() if "team" in game_players.columns else []
                if len(teams) >= 2:
                    return [
                        f"{teams[0]}-{teams[1]} has a {ou:.0f} total. Pace and points. Stack territory."
                    ]
    return []


def generate_tonights_edges(pool: pd.DataFrame) -> List[str]:
    """Generate 3-5 data-driven callouts about tonight's slate.

    Parameters
    ----------
    pool : pd.DataFrame
        The current slate pool with salary, proj, ownership, etc.

    Returns
    -------
    list of str
        Each string is a 1-2 sentence callout in Ricky's voice.
    """
    if pool.empty:
        return []

    callouts = []
    callouts.extend(_find_salary_mismatches(pool))
    callouts.extend(_find_ownership_traps(pool))
    callouts.extend(_find_contrarian_windows(pool))
    callouts.extend(_find_game_environment_edges(pool))

    # Cap at 5 callouts, prioritize variety (already ordered by type)
    return callouts[:5]


# ---------------------------------------------------------------------------
# 3. BUST CALL -- one player, named, bold prediction
# ---------------------------------------------------------------------------

def generate_bust_call(
    pool: pd.DataFrame,
    fade_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Pick the one player most likely to massively underperform.

    Scoring factors:
      - High ownership (bad GPP outcome if bust)
      - Bad DvP matchup (high dvp_rank)
      - High salary relative to recent form (rolling avg)
      - Large positive spread (wrong side of blowout)
      - Blowout risk

    Parameters
    ----------
    pool : pd.DataFrame
        Current slate pool.
    fade_candidates : list of dict, optional
        Pre-classified fade candidates from edge_analysis.

    Returns
    -------
    dict or None
        {"name": str, "salary": int, "explanation": str} or None.
    """
    if pool.empty:
        return None

    # Work from the pool to score bust risk — need at least salary + proj + ownership
    required = {"player_name", "salary", "proj", "ownership"}
    if not required.issubset(pool.columns):
        return None

    # Only consider relevant players (meaningful salary and ownership)
    df = pool[(pool["salary"] >= 5000) & (pool["ownership"] > 3) & (pool["proj"] > 10)].copy()
    if df.empty:
        # Fallback: try fade_candidates
        return _bust_from_fades(fade_candidates)

    df["bust_score"] = 0.0

    # Factor 1: Ownership (higher = worse GPP outcome if bust)
    own_max = df["ownership"].max()
    if own_max > 0:
        df["bust_score"] += (df["ownership"] / own_max) * 30

    # Factor 2: Bad DvP matchup (high rank = tough defense)
    if "dvp_rank" in df.columns:
        dvp = df["dvp_rank"].fillna(15)
        df["bust_score"] += (dvp / 30) * 25

    # Factor 3: Salary overpriced vs recent form
    if "rolling_fp_5" in df.columns:
        recent = df["rolling_fp_5"].fillna(df["proj"])
        salary_implied = df["salary"] / 1000 * df["proj"].median() / (df["salary"].median() / 1000)
        # If recent avg is well below projection, higher bust risk
        form_gap = df["proj"] - recent
        form_gap_norm = form_gap / df["proj"].clip(lower=1)
        df["bust_score"] += form_gap_norm.clip(lower=0) * 20

    # Factor 4: Blowout risk (wrong side of spread)
    if "spread" in df.columns:
        spread = df["spread"].fillna(0)
        # Positive spread = underdog = garbage time risk
        df["bust_score"] += (spread.clip(lower=0) / 10) * 15

    if "blowout_risk" in df.columns:
        df["bust_score"] += df["blowout_risk"].fillna(0) * 10

    # Pick the top bust candidate
    bust = df.nlargest(1, "bust_score").iloc[0]
    name = bust["player_name"]
    sal = int(bust["salary"])
    own = bust["ownership"]
    proj = bust["proj"]

    # Build explanation
    reasons = []
    if "dvp_rank" in df.columns and pd.notna(bust.get("dvp_rank")) and bust["dvp_rank"] >= 20:
        rank_label = 31 - int(bust["dvp_rank"])  # rank 30 → "top-1", rank 25 → "top-6"
        reasons.append(f"top-{rank_label} defense at the position")
    if "spread" in df.columns and pd.notna(bust.get("spread")) and bust["spread"] > 4:
        reasons.append(f"+{bust['spread']:.1f} underdog spread")
    if "rolling_fp_5" in df.columns and pd.notna(bust.get("rolling_fp_5")):
        recent = bust["rolling_fp_5"]
        if recent < proj * 0.85:
            reasons.append(f"averaging {recent:.0f} over his last 5")
    if "blowout_risk" in df.columns and bust.get("blowout_risk", 0) > 0.4:
        reasons.append("blowout game script risk")

    if reasons:
        reason_str = ", ".join(reasons)
        explanation = f"{own:.0f}% of the field is about to find out. {reason_str.capitalize()}."
    else:
        explanation = f"{own:.0f}% of the field is about to find out. The price is wrong."

    return {"name": name, "salary": sal, "explanation": explanation}


def _bust_from_fades(
    fade_candidates: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Fallback: pick bust from pre-classified fade_candidates."""
    if not fade_candidates:
        return None

    # Pick the fade with highest salary (most painful bust)
    relevant = [f for f in fade_candidates if f.get("salary", 0) >= 4000]
    if not relevant:
        return None

    bust = max(relevant, key=lambda f: f.get("salary", 0))
    name = bust.get("player_name", "Unknown")
    sal = int(bust.get("salary", 0))
    own = bust.get("ownership", 0)

    return {
        "name": name,
        "salary": sal,
        "explanation": f"{own:.0f}% owned and the model hates it. Don't be a hero.",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """Convert integer to ordinal string (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
