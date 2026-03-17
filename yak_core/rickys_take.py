"""yak_core.rickys_take -- Deterministic voice generation for Ricky's Take.

Generates three sections for the Edge Analysis tab:
  1. Last Night -- recap of previous slate hits/misses in Ricky's voice
  2. Tonight's Edges -- data-driven callouts about the current slate
  3. Bust Call -- one bold prediction for the biggest underperformer

All text is template-based with data slots. No LLM calls.

RICKY'S VOICE:
  Ex-hedge-fund guy who walked away from the suit. Too independent, too fast,
  too willing to cut through bullshit instead of following process. The C-suite
  hated his vibe because he didn't suck up. Now he's at the coffee shop in a
  hoodie, building lineups against a scoreboard that can't be spun.

  He doesn't respect consensus. When 30% of the field is on someone, he sees
  herd behavior — the same groupthink that filled conference rooms with
  nodding heads. He trusts the work, not the narrative. The scoreboard is the
  only honest meeting he's ever attended.
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
    "{name} went off for {actual:.0f} on a {proj:.0f} projection. The model did the work. The scoreboard agreed.",
    "{name} dropped {actual:.0f} against a {proj:.0f} line. That's what happens when you trust the data instead of the narrative.",
    "{name} crushed with {actual:.0f} actual on {proj:.0f} projected. No committee needed to see that one coming.",
    "{name} smashed at {actual:.0f}. Projected {proj:.0f}. The kind of edge that doesn't survive a group chat — which is exactly why it hit.",
]

# Miss templates: player underperformed projection
_MISS_TEMPLATES = [
    "{name} owners got cooked. {actual:.0f} actual on a {proj:.0f} line. The field followed each other right off a cliff.",
    "{name} bricked at {actual:.0f} on a {proj:.0f} projection. Consensus pick, consensus loss. Seen this movie before.",
    "{name} laid an egg — {actual:.0f} on a {proj:.0f} line. Half the field was on it because the other half was on it. That's not analysis, that's a herd.",
    "{name} at {actual:.0f} vs {proj:.0f} projected. Popular pick. Popular outcome. The scoreboard doesn't grade on a curve.",
]

# Pattern templates (optional third sentence about the slate)
_PATTERN_TEMPLATES_CHALK_HELD = [
    "High-salary chalk went {hit_count}-of-{total}. Sometimes the obvious play is the right play — just don't confuse that with doing the work.",
    "Top salaries went {hit_count}-of-{total}. Chalk held, but that's not the norm. Don't get comfortable.",
]

_PATTERN_TEMPLATES_CHALK_CRUMBLED = [
    "Only {hit_count} of the top-{total} salaries cleared projection. The field loves paying up for names. The scoreboard doesn't care about names.",
    "Chalk crumbled — {hit_count}-of-{total} top salaries hit. Everyone nodded along on the same picks. Same result as every bad meeting.",
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
                seed = top_hits + total
                tmpl = _pick_template(_PATTERN_TEMPLATES_CHALK_HELD, seed)
                sentences.append(tmpl.format(hit_count=top_hits, total=total))
            elif top_hits <= total * 0.3:
                seed = top_hits + total
                tmpl = _pick_template(_PATTERN_TEMPLATES_CHALK_CRUMBLED, seed)
                sentences.append(tmpl.format(hit_count=top_hits, total=total))

    return " ".join(sentences) if sentences else None


# ---------------------------------------------------------------------------
# 2. TONIGHT'S EDGES -- forward-looking callouts
# ---------------------------------------------------------------------------

def _find_salary_mismatches(pool: pd.DataFrame, mentioned: set) -> List[str]:
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

    # Players underpriced by >15%, skip already-mentioned
    candidates = df[(df["underpriced_pct"] > 0.15) & (~df["player_name"].isin(mentioned))]
    mispriced = candidates.nlargest(3, "underpriced_pct")

    _MISMATCH_TEMPLATES = [
        "{name} at ${sal} is mispriced. {proj:.0f} projected, {pts_1k:.1f} pts/$1K. The kind of inefficiency that used to pay my bonus. Now it pays lineups.",
        "{name} at ${sal} with {proj:.0f} projected. {pts_1k:.1f} pts/$1K. The market is wrong on this one and the market doesn't self-correct before lock.",
        "{name} at ${sal} projects to {proj:.0f}. That's {pts_1k:.1f} pts/$1K — a pricing gap the field won't notice until the scoreboard posts.",
    ]

    callouts = []
    for i, (_, row) in enumerate(mispriced.iterrows()):
        name = row["player_name"]
        sal = f"{int(row['salary']):,}"
        proj = row["proj"]
        pts_1k = row["pts_per_1k"]
        tmpl = _MISMATCH_TEMPLATES[i % len(_MISMATCH_TEMPLATES)]
        callouts.append(tmpl.format(name=name, sal=sal, proj=proj, pts_1k=pts_1k))
        mentioned.add(name)
    return callouts


def _find_ownership_traps(pool: pd.DataFrame, mentioned: set) -> List[str]:
    """Find high-owned players with red flags."""
    own_col = None
    for c in ("ownership", "own_pct", "POWN"):
        if c in pool.columns:
            own_col = c
            break
    if own_col is None or "player_name" not in pool.columns or "salary" not in pool.columns:
        return []

    df = pool[pool[own_col] > 0].copy()
    if df.empty:
        return []

    # Top 5 by ownership, skip already-mentioned
    top_owned = df[~df["player_name"].isin(mentioned)].nlargest(5, own_col)

    _TRAP_TEMPLATES = [
        "{own:.0f}% of the field is on {name} tonight. Same herd energy. {flags}. The scoreboard doesn't care how popular the pick was.",
        "{own:.0f}% of the field lined up behind {name}. {flags}. That's not conviction, that's a crowded trade. I've watched those unwind.",
        "{name} at {own:.0f}% owned. {flags}. Everyone's nodding along on this one. I've sat in enough rooms to know what that means.",
    ]

    callouts = []
    for i, (_, row) in enumerate(top_owned.iterrows()):
        name = row["player_name"]
        own = row[own_col]
        sal = int(row["salary"])
        flags = []

        # Bad DvP matchup (high rank = bad)
        if "dvp_rank" in df.columns and pd.notna(row.get("dvp_rank")):
            dvp = row["dvp_rank"]
            if dvp >= 25:
                rank_label = 31 - int(dvp)
                flags.append(f"top-{rank_label} defense at the position")

        # Blowout risk
        if "blowout_risk" in df.columns and row.get("blowout_risk", 0) > 0.5:
            flags.append("blowout risk could cap minutes")

        # Big spread (wrong side -- underdog getting blown out)
        if "spread" in df.columns and pd.notna(row.get("spread")):
            spread = row["spread"]
            if spread > 5:
                flags.append(f"+{spread:.1f} spread, garbage time incoming")

        if flags:
            flag_str = ", ".join(flags).capitalize()
            tmpl = _TRAP_TEMPLATES[i % len(_TRAP_TEMPLATES)]
            callouts.append(tmpl.format(name=name, own=own, flags=flag_str))
            mentioned.add(name)

    return callouts[:2]  # Max 2 ownership traps


def _find_contrarian_windows(pool: pd.DataFrame, mentioned: set) -> List[str]:
    """Find low-owned players with strong situational edges."""
    own_col = None
    for c in ("ownership", "own_pct", "POWN"):
        if c in pool.columns:
            own_col = c
            break
    if own_col is None or "player_name" not in pool.columns:
        return []

    df = pool[(pool[own_col] > 0) & (pool["proj"] > 10)].copy()
    if df.empty:
        return []

    # Bottom quartile by ownership, skip already-mentioned
    own_25th = df[own_col].quantile(0.25)
    low_owned = df[(df[own_col] <= max(own_25th, 5)) & (~df["player_name"].isin(mentioned))].copy()

    if low_owned.empty:
        return []

    # Score contrarian value: good DvP + high projection relative to salary
    low_owned["pts_per_1k"] = low_owned["proj"] / (low_owned["salary"] / 1000)

    _CONTRARIAN_TEMPLATES = [
        "{name} at {own:.0f}% owned. The field didn't do the work — {reason}. This is the kind of edge that doesn't survive a committee.",
        "{name} sitting at {own:.0f}% owned. {reason}. Nobody in the room bothered to look. That's the whole edge.",
        "{name}, {own:.0f}% owned. {reason}. The field is too busy copying each other's homework to notice.",
    ]

    _CONTRARIAN_FALLBACK = [
        "{name} at ${sal} with {proj:.0f} projected and only {own:.0f}% owned. The field is asleep. I've made a career on the other side of that trade.",
        "{name} at ${sal}, {proj:.0f} projected, {own:.0f}% owned. When nobody's looking, that's when you look harder.",
    ]

    _LOTTO_TEMPLATES = [
        "GPP lotto tickets: {names}. Low owned, high environment, mispriced. These are the kind of names that show up in winning lineups and nobody saw it coming.",
        "GPP lotto shelf: {names}. The field walked right past them. One of these hits and your lineup looks like genius.",
        "Lotto plays: {names}. Cheap, ignored, and sitting in the right game environment. That's the whole formula.",
    ]

    # Collect qualified contrarian picks
    lead_callout = None  # First strong pick gets its own bullet
    lotto_names = []     # Extras get bundled into a lotto bullet

    for i, (_, row) in enumerate(low_owned.nlargest(5, "pts_per_1k").iterrows()):
        name = row["player_name"]
        own = row[own_col]
        sal = f"{int(row['salary']):,}"
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

        qualified = bool(reason_parts) or row["pts_per_1k"] > 5.0
        if not qualified:
            continue

        # First pick gets a full callout; the rest become lotto picks
        if lead_callout is None:
            if reason_parts:
                reason = " and ".join(reason_parts)
                tmpl = _CONTRARIAN_TEMPLATES[i % len(_CONTRARIAN_TEMPLATES)]
                lead_callout = tmpl.format(name=name, own=own, reason=reason)
            else:
                tmpl = _CONTRARIAN_FALLBACK[i % len(_CONTRARIAN_FALLBACK)]
                lead_callout = tmpl.format(name=name, sal=sal, proj=proj, own=own)
            mentioned.add(name)
        else:
            lotto_names.append(f"{name} ({own:.0f}%)")
            mentioned.add(name)

    callouts = []
    if lead_callout:
        callouts.append(lead_callout)
    if lotto_names:
        import hashlib as _hl
        _seed = int(_hl.md5(",".join(lotto_names).encode()).hexdigest()[:8], 16)
        tmpl = _LOTTO_TEMPLATES[_seed % len(_LOTTO_TEMPLATES)]
        callouts.append(tmpl.format(names=", ".join(lotto_names)))

    return callouts[:2]  # Max 2 (1 lead + 1 lotto bundle)


def _find_game_environment_edges(pool: pd.DataFrame, mentioned: set = None) -> List[str]:
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
                        f"{teams[0]}-{teams[1]} has a {ou:.0f} total. Pace and points. This is where the math lives tonight."
                    ]
    return []


def generate_tonights_edges(pool: pd.DataFrame) -> List[str]:
    """Generate 3-5 data-driven callouts about tonight's slate.

    Each callout covers a different player — no name repeats. The sub-generators
    run in order (mismatches → traps → contrarian → environment) and share a
    ``mentioned`` set so later generators skip players already called out.

    Players whose projection is heavily cascade-inflated (injury_bump_fp >= 40%
    of original_proj) are excluded — we don't hype a number the model doesn't
    actually believe.

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

    # ── Strip cascade-inflated players before generating callouts ──
    # If the injury bump is >= 40% of the original (pre-cascade) projection,
    # the number is mostly cascade math, not a real expectation.
    _pool = pool.copy()
    if "injury_bump_fp" in _pool.columns and "original_proj" in _pool.columns:
        _bump = pd.to_numeric(_pool["injury_bump_fp"], errors="coerce").fillna(0)
        _orig = pd.to_numeric(_pool["original_proj"], errors="coerce").fillna(0)
        _base = _orig.where(_orig > 0, _pool["proj"] if "proj" in _pool.columns else 1)
        _cascade_ratio = _bump / _base.clip(lower=1)
        _inflated = (_cascade_ratio >= 0.40) & (_bump > 0)
        n_removed = _inflated.sum()
        if n_removed > 0:
            _pool = _pool[~_inflated].reset_index(drop=True)
            print(f"[generate_tonights_edges] Excluded {n_removed} cascade-inflated player(s) from callouts")

    mentioned: set = set()  # player names already used in a callout
    callouts = []
    callouts.extend(_find_salary_mismatches(_pool, mentioned))
    callouts.extend(_find_ownership_traps(_pool, mentioned))
    callouts.extend(_find_contrarian_windows(_pool, mentioned))
    callouts.extend(_find_game_environment_edges(_pool, mentioned))

    # Cap at 5 callouts, prioritize variety (already ordered by type)
    return callouts[:5]


# ---------------------------------------------------------------------------
# 3. BUST CALL -- one player, named, bold prediction
# ---------------------------------------------------------------------------

_BUST_EXPLANATIONS_HIGH_OWN = [
    "{own:.0f}% of the field is about to learn a lesson. {reasons}. I've watched this exact setup blow up a hundred portfolios.",
    "{own:.0f}% of the field lined up for this. {reasons}. The consensus was wrong in the boardroom and it's wrong here.",
    "{own:.0f}% owned. {reasons}. Everyone agreed on this pick — and that's exactly the problem.",
]

_BUST_EXPLANATIONS_LOW_OWN = [
    "{own:.0f}% owned, but the price tag is a trap. {reasons}. The salary says star, the situation says sit.",
    "{own:.0f}% owned — doesn't matter. {reasons}. Projections don't match reality here and the scoreboard won't either.",
    "Only {own:.0f}% of the field is on this, but the ones who are will regret it. {reasons}. Bad spot, bad price.",
]

_BUST_FALLBACK_HIGH_OWN = [
    "{own:.0f}% owned and the numbers don't support it. Popularity isn't an edge. Never was.",
    "The field loves {name} tonight at {own:.0f}%. The data doesn't. I'll take the data.",
]

_BUST_FALLBACK_LOW_OWN = [
    "{own:.0f}% owned — low exposure, but still a bad bet. The salary is doing the selling, not the data.",
    "Only {own:.0f}% of the field bit on {name}, but the price tag still doesn't add up. Pass.",
]


def generate_bust_call(
    pool: pd.DataFrame,
    fade_candidates: Optional[List[Dict[str, Any]]] = None,
    positive_tier_names: Optional[set] = None,
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
    positive_tier_names : set of str, optional
        Player names classified as core/leverage/value. Excluded from bust
        candidacy to prevent contradictions with tier classification.

    Returns
    -------
    dict or None
        {"name": str, "salary": int, "explanation": str} or None.
    """
    if pool.empty:
        return None

    own_col = None
    for c in ("ownership", "own_pct", "POWN"):
        if c in pool.columns:
            own_col = c
            break

    # Work from the pool to score bust risk — need at least salary + proj + ownership
    required = {"player_name", "salary", "proj"}
    if not required.issubset(pool.columns) or own_col is None:
        return _bust_from_fades(fade_candidates)

    # Only consider relevant players (meaningful salary and ownership)
    df = pool[(pool["salary"] >= 5000) & (pool[own_col] > 3) & (pool["proj"] > 10)].copy()
    if df.empty:
        return _bust_from_fades(fade_candidates)

    df["bust_score"] = 0.0

    # Factor 1: Ownership (higher = worse GPP outcome if bust)
    own_max = df[own_col].max()
    if own_max > 0:
        df["bust_score"] += (df[own_col] / own_max) * 30

    # Factor 2: Bad DvP matchup (high rank = tough defense)
    if "dvp_rank" in df.columns:
        dvp = df["dvp_rank"].fillna(15)
        df["bust_score"] += (dvp / 30) * 25

    # Factor 3: Salary overpriced vs recent form
    if "rolling_fp_5" in df.columns:
        recent = df["rolling_fp_5"].fillna(df["proj"])
        form_gap = df["proj"] - recent
        form_gap_norm = form_gap / df["proj"].clip(lower=1)
        df["bust_score"] += form_gap_norm.clip(lower=0) * 20

    # Factor 4: Blowout risk (wrong side of spread)
    if "spread" in df.columns:
        spread = df["spread"].fillna(0)
        df["bust_score"] += (spread.clip(lower=0) / 10) * 15

    if "blowout_risk" in df.columns:
        df["bust_score"] += df["blowout_risk"].fillna(0) * 10

    # Align bust call with tier classifier: boost fades, penalize non-fades
    if fade_candidates:
        fade_names = {fc.get("player_name", "") for fc in fade_candidates}
        df.loc[df["player_name"].isin(fade_names), "bust_score"] += 15
        df.loc[~df["player_name"].isin(fade_names), "bust_score"] -= 10

    # Exclude positive-tier players from bust candidacy
    if positive_tier_names:
        df = df[~df["player_name"].isin(positive_tier_names)]
        if df.empty:
            return _bust_from_fades(fade_candidates)

    # Pick the top bust candidate
    bust = df.nlargest(1, "bust_score").iloc[0]
    name = bust["player_name"]
    sal = int(bust["salary"])
    own = bust[own_col]
    proj = bust["proj"]

    # Build explanation
    reasons = []
    if "dvp_rank" in df.columns and pd.notna(bust.get("dvp_rank")) and bust["dvp_rank"] >= 20:
        rank_label = 31 - int(bust["dvp_rank"])
        reasons.append(f"top-{rank_label} defense at the position")
    if "spread" in df.columns and pd.notna(bust.get("spread")) and bust["spread"] > 4:
        reasons.append(f"+{bust['spread']:.1f} spread")
    if "rolling_fp_5" in df.columns and pd.notna(bust.get("rolling_fp_5")):
        recent = bust["rolling_fp_5"]
        if recent < proj * 0.85:
            reasons.append(f"averaging {recent:.0f} over his last 5")
    if "blowout_risk" in df.columns and bust.get("blowout_risk", 0) > 0.4:
        reasons.append("blowout game script risk")

    # Pick a template — tier by ownership so language matches reality
    seed = sal + int(own * 10)
    high_own = own > 15
    if reasons:
        reason_str = ", ".join(reasons).capitalize()
        templates = _BUST_EXPLANATIONS_HIGH_OWN if high_own else _BUST_EXPLANATIONS_LOW_OWN
        tmpl = _pick_template(templates, seed)
        explanation = tmpl.format(name=name, own=own, reasons=reason_str)
    else:
        templates = _BUST_FALLBACK_HIGH_OWN if high_own else _BUST_FALLBACK_LOW_OWN
        tmpl = _pick_template(templates, seed)
        explanation = tmpl.format(name=name, own=own)

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
    own = bust.get("ownership", bust.get("own_pct", 0))

    return {
        "name": name,
        "salary": sal,
        "explanation": f"The model doesn't like it at {own:.0f}% owned. The field followed each other into this one. Don't follow them.",
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
