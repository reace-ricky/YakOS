"""yak_core.rickys_take -- Deterministic voice generation for Ricky's Take.

Generates three sections for the Edge Analysis tab:
  1. Last Night -- recap of previous slate hits/misses in Ricky's voice
  2. Tonight's Edges -- data-driven callouts about the current slate
  3. Bust Call -- one bold prediction for the biggest underperformer

All text is template-based with data slots. No LLM calls.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Template selection helpers -- date-rotated deterministic pick
# ---------------------------------------------------------------------------

def _pick_template(templates: list, player_name: str, category: str) -> str:
    """Deterministic, date-rotated template selection keyed by player name."""
    key = f"{player_name}:{category}:{date.today().toordinal()}"
    return templates[hash(key) % len(templates)]


def _pick_template_by_key(templates: list, key: str, category: str) -> str:
    """Deterministic, date-rotated template selection keyed by an arbitrary string."""
    composite = f"{key}:{category}:{date.today().toordinal()}"
    return templates[hash(composite) % len(templates)]


# ---------------------------------------------------------------------------
# 1. LAST NIGHT -- recap generation
# ---------------------------------------------------------------------------

# Hit templates: player outperformed projection (target: 10)
_HIT_TEMPLATES = [
    "{name} went off for {actual:.0f} on a {proj:.0f} projection. The model did the work. The scoreboard agreed.",
    "{name} dropped {actual:.0f} against a {proj:.0f} line. That's what happens when you trust the data instead of the narrative.",
    "{name} crushed with {actual:.0f} actual on {proj:.0f} projected. No committee needed to see that one coming.",
    "{name} smashed at {actual:.0f}. Projected {proj:.0f}. The kind of edge that doesn't survive a group chat — which is exactly why it hit.",
    "{name} went off for {actual:.0f} on a {proj:.0f} line. Math doesn't care about your feelings. It just cashes.",
    "{name}: {actual:.0f} actual, {proj:.0f} projected. The edge was sitting in the open. Most people were looking somewhere else.",
    "{name} for {actual:.0f}. Projection said {proj:.0f}. Nothing fancy. Just followed the number.",
    "{name} at {actual:.0f} on a {proj:.0f} line. The data was clear. The field was busy arguing about something else.",
    "{name}: {actual:.0f} actual vs {proj:.0f} projected. Edges don't announce themselves. You have to do the work.",
    "{name} hit {actual:.0f} against {proj:.0f}. Clean read, clean result. That's the whole process.",
]

# Miss templates: player underperformed projection (target: 10)
_MISS_TEMPLATES = [
    "{name} owners got cooked. {actual:.0f} actual on a {proj:.0f} line. The field followed each other right off a cliff.",
    "{name} bricked at {actual:.0f} on a {proj:.0f} projection. Consensus pick, consensus loss. Seen this movie before.",
    "{name} laid an egg — {actual:.0f} on a {proj:.0f} line. Half the field was on it because the other half was on it. That's not analysis, that's a herd.",
    "{name} at {actual:.0f} vs {proj:.0f} projected. Popular pick. Popular outcome. The scoreboard doesn't grade on a curve.",
    "{name} at {actual:.0f} vs {proj:.0f}. Popular pick, popular result. The crowd's batting average hasn't changed.",
    "{name} bricked — {actual:.0f} on a {proj:.0f} line. Everyone agreed on this one. That was the tell.",
    "{name}: {actual:.0f} actual on {proj:.0f} projected. Consensus has a losing record. Adding to the sample.",
    "{name}: {actual:.0f} on a {proj:.0f} line. The popular answer is rarely the profitable one.",
    "{name} at {actual:.0f} against {proj:.0f}. Ownership was high. Output was not. Correlation isn't always your friend.",
    "{name} went for {actual:.0f} on {proj:.0f} projected. The field copied each other's homework. Same grade.",
]

# Pattern templates: chalk held (target: 6)
_PATTERN_TEMPLATES_CHALK_HELD = [
    "High-salary chalk went {hit_count}-of-{total}. Sometimes the obvious play is the right play — just don't confuse that with doing the work.",
    "Top salaries went {hit_count}-of-{total}. Chalk held, but that's not the norm. Don't get comfortable.",
    "Chalk went {hit_count}-of-{total}. Even a broken clock. Don't let one good night turn into lazy process.",
    "Top salaries cleared at {hit_count}-of-{total}. The field got lucky. Luck doesn't compound.",
    "{hit_count}-of-{total} top salaries hit. Chalk held tonight. Regression is patient — it'll collect eventually.",
    "High-priced chalk: {hit_count}-of-{total}. Credit where it's due. But one night doesn't change the base rates.",
]

# Pattern templates: chalk crumbled (target: 6)
_PATTERN_TEMPLATES_CHALK_CRUMBLED = [
    "Only {hit_count} of the top-{total} salaries cleared projection. The field loves paying up for names. The scoreboard doesn't care about names.",
    "Chalk crumbled — {hit_count}-of-{total} top salaries hit. Everyone nodded along on the same picks. Same result as every bad meeting.",
    "Top salaries went {hit_count}-of-{total}. The consensus tax came due.",
    "{hit_count}-of-{total} chalk plays hit. Expensive and wrong. The field's favorite combination.",
    "Chalk at {hit_count}-of-{total}. Paying up for comfort is still paying up. The scoreboard doesn't offer refunds.",
    "Only {hit_count} of {total} top salaries delivered. The field paid retail for names and got wholesale results.",
]


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
        tmpl = _pick_template(_HIT_TEMPLATES, h["player_name"], "hit")
        sentences.append(
            tmpl.format(name=h["player_name"], actual=h["actual"], proj=h["projected"])
        )

    # Worst miss
    if misses:
        m = misses[0]
        tmpl = _pick_template(_MISS_TEMPLATES, m["player_name"], "miss")
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
            chalk_key = f"{top_hits}:{total}"
            if top_hits >= total * 0.7:
                tmpl = _pick_template_by_key(_PATTERN_TEMPLATES_CHALK_HELD, chalk_key, "chalk_held")
                sentences.append(tmpl.format(hit_count=top_hits, total=total))
            elif top_hits <= total * 0.3:
                tmpl = _pick_template_by_key(_PATTERN_TEMPLATES_CHALK_CRUMBLED, chalk_key, "chalk_crumbled")
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

    # Salary Mismatch templates (target: 10)
    _MISMATCH_TEMPLATES = [
        "{name} at ${sal} is mispriced. {proj:.0f} projected, {pts_1k:.1f} pts/$1K. The kind of inefficiency that doesn't survive lock.",
        "{name} at ${sal} with {proj:.0f} projected. {pts_1k:.1f} pts/$1K. The market is wrong on this one and the market doesn't self-correct before lock.",
        "{name} at ${sal} projects to {proj:.0f}. That's {pts_1k:.1f} pts/$1K — a pricing gap the field won't notice until the scoreboard posts.",
        "{name} at ${sal} is mispriced. {proj:.0f} projected, {pts_1k:.1f} pts/$1K. This won't stay open long.",
        "${sal} for {name}. {proj:.0f} projected. {pts_1k:.1f} per $1K. The pricing gap closes when the scoreboard posts, not before.",
        "{name}, ${sal}, {proj:.0f} projected. {pts_1k:.1f} pts/$1K. Someone made a mistake on the pricing. Don't correct them — exploit it.",
        "{name} at ${sal}. {proj:.0f} projected, {pts_1k:.1f} pts/$1K. The number is right there. Most people won't bother to check.",
        "${sal} for {name} with {proj:.0f} projected. {pts_1k:.1f} pts/$1K. Mispriced. The math is straightforward.",
        "{name}: ${sal}, {proj:.0f} projected, {pts_1k:.1f} pts/$1K. Price is wrong. That's the whole thesis.",
        "{name} at ${sal} projecting {proj:.0f}. {pts_1k:.1f} per $1K. This is the kind of gap that wins slates. Quietly.",
    ]

    callouts = []
    for _, row in mispriced.iterrows():
        name = row["player_name"]
        sal = f"{int(row['salary']):,}"
        proj = row["proj"]
        pts_1k = row["pts_per_1k"]
        tmpl = _pick_template(_MISMATCH_TEMPLATES, name, "mismatch")
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

    # Ownership Trap templates (target: 10)
    _TRAP_TEMPLATES = [
        "{own:.0f}% of the field is on {name} tonight. {flags}. The scoreboard doesn't care how popular the pick was.",
        "{own:.0f}% of the field lined up behind {name}. {flags}. That's not conviction, that's a crowded trade.",
        "{name} at {own:.0f}% owned. {flags}. When everyone agrees, that's usually the wrong answer.",
        "{own:.0f}% on {name}. {flags}. Popularity and profitability have a negative correlation in GPPs.",
        "{name}: {own:.0f}% owned. {flags}. The field is betting this one the same way. That's the risk.",
        "{own:.0f}% ownership on {name}. {flags}. A crowded position with a thin margin. Bad math.",
        "{name} at {own:.0f}% owned tonight. {flags}. When the whole room agrees, check the exits.",
        "{own:.0f}% of the field lined up for {name}. {flags}. The consensus loved this one. Consensus has a losing record.",
        "{name}, {own:.0f}% owned. {flags}. Everyone's on the same side of this. That's not an edge — it's exposure.",
        "{own:.0f}% on {name}. {flags}. This is what a crowded trade looks like. Same mechanics every time.",
    ]

    callouts = []
    for _, row in top_owned.iterrows():
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
            tmpl = _pick_template(_TRAP_TEMPLATES, name, "trap")
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

    # Contrarian Main templates (target: 10)
    _CONTRARIAN_TEMPLATES = [
        "{name} at {own:.0f}% owned. The field didn't do the work — {reason}. This is the kind of edge that doesn't show up in a group chat.",
        "{name} sitting at {own:.0f}% owned. {reason}. Nobody bothered to look. That's the whole edge.",
        "{name}, {own:.0f}% owned. {reason}. The field is too busy copying each other's homework to notice.",
        "{name}: {own:.0f}% owned. {reason}. Low ownership, high setup. That's the formula.",
        "{name} at {own:.0f}%. {reason}. The field is looking somewhere else. Good.",
        "{own:.0f}% on {name}. {reason}. The best edges are the ones nobody's talking about.",
        "{name}, {own:.0f}% owned. {reason}. Contrarian isn't a strategy — it's a side effect of doing better work.",
        "{name} at {own:.0f}% ownership. {reason}. The field went with the popular name. This is the unpopular math.",
        "{own:.0f}% owned. {name}. {reason}. The crowd went left. The data says right.",
        "{name}: {own:.0f}%. {reason}. Sometimes the best play is the one nobody's making.",
    ]

    # Contrarian Fallback templates (target: 8)
    _CONTRARIAN_FALLBACK = [
        "{name} at ${sal} with {proj:.0f} projected and only {own:.0f}% owned. The field is asleep.",
        "{name} at ${sal}, {proj:.0f} projected, {own:.0f}% owned. When nobody's looking, that's when you look harder.",
        "{name}: ${sal}, {proj:.0f} projected, {own:.0f}% owned. The pricing says value. The ownership says ignored. Both are useful.",
        "${sal} for {name}. {proj:.0f} projected at {own:.0f}% owned. Under the radar for no good reason.",
        "{name} at ${sal} projects to {proj:.0f} with {own:.0f}% ownership. The field walked right past this.",
        "{name}, ${sal}, {proj:.0f} projected. {own:.0f}% owned. Overlooked and underpriced. That's the sweet spot.",
        "{name}: {proj:.0f} projected, ${sal}, {own:.0f}% owned. The numbers work. The field didn't bother to check.",
        "{name} at ${sal}, {proj:.0f} projected, {own:.0f}% owned. Low traffic, good odds. I'll take it.",
    ]

    # Lotto templates (target: 8)
    _LOTTO_TEMPLATES = [
        "GPP lotto tickets: {names}. Low owned, high environment, mispriced. One of these hits and your lineup separates.",
        "GPP lotto shelf: {names}. The field walked right past them. One hit changes the whole slate.",
        "Lotto plays: {names}. Cheap, ignored, and sitting in the right game environment. That's the whole formula.",
        "Deep-roster lotto: {names}. Low ownership, right situation. The kind of names that show up in winning lineups.",
        "Lotto tier: {names}. The field doesn't want them. The math disagrees. That's where edges live.",
        "GPP lotto: {names}. Ignored by the field. Supported by the numbers. Pick your side.",
        "Lotto window: {names}. Low owned, high ceiling environment. These are the asymmetric plays.",
        "Lotto candidates: {names}. The field priced them out of conversation. The scoreboard might price them back in.",
    ]

    # Collect qualified contrarian picks
    lead_callout = None  # First strong pick gets its own bullet
    lotto_names = []     # Extras get bundled into a lotto bullet

    for _, row in low_owned.nlargest(5, "pts_per_1k").iterrows():
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
                tmpl = _pick_template(_CONTRARIAN_TEMPLATES, name, "contrarian")
                lead_callout = tmpl.format(name=name, own=own, reason=reason)
            else:
                tmpl = _pick_template(_CONTRARIAN_FALLBACK, name, "contrarian_fb")
                lead_callout = tmpl.format(name=name, sal=sal, proj=proj, own=own)
            mentioned.add(name)
        else:
            lotto_names.append(f"{name} ({own:.0f}%)")
            mentioned.add(name)

    callouts = []
    if lead_callout:
        callouts.append(lead_callout)
    if lotto_names:
        lotto_key = ",".join(lotto_names)
        tmpl = _pick_template_by_key(_LOTTO_TEMPLATES, lotto_key, "lotto")
        callouts.append(tmpl.format(names=", ".join(lotto_names)))

    return callouts[:2]  # Max 2 (1 lead + 1 lotto bundle)


def _find_game_environment_edges(pool: pd.DataFrame, mentioned: set = None) -> List[str]:
    """Find game environment edges: high totals, pace advantages."""
    if "over_under" not in pool.columns or "player_name" not in pool.columns:
        return []

    df = pool[pool["over_under"].notna() & (pool["proj"] > 15)].copy()
    if df.empty:
        return []

    # Game Environment templates (target: 6)
    _GAME_ENV_TEMPLATES = [
        "{t1}-{t2} has a {ou:.0f} total. Pace and points. This is where the math lives tonight.",
        "{t1}-{t2}: {ou:.0f} total. High-volume environment. Stack here or explain why not.",
        "{ou:.0f} total on {t1}-{t2}. The environment is doing the heavy lifting. Lean into it.",
        "{t1}-{t2} at {ou:.0f}. Pace means possessions. Possessions mean opportunity. Simple math.",
        "{t1}-{t2} game sits at {ou:.0f}. High totals correlate with high ceilings. The data is clear.",
        "{ou:.0f} total in {t1}-{t2}. Volume is the best predictor of upside. This game has it.",
    ]

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
                    game_key = f"{teams[0]}:{teams[1]}:{ou}"
                    tmpl = _pick_template_by_key(_GAME_ENV_TEMPLATES, game_key, "game_env")
                    return [tmpl.format(t1=teams[0], t2=teams[1], ou=ou)]
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

    # ── Filter OUT / IR / Suspended players ──
    # These players can't play — never generate callouts about them.
    _OUT_STATUSES = {"OUT", "IR", "SUSPENDED"}
    _pool = pool.copy()
    if "status" in _pool.columns:
        _before = len(_pool)
        _pool = _pool[
            ~_pool["status"].fillna("").str.strip().str.upper().isin(_OUT_STATUSES)
        ].reset_index(drop=True)
        _removed = _before - len(_pool)
        if _removed:
            print(f"[generate_tonights_edges] Excluded {_removed} OUT/IR/Suspended player(s) from callouts")

    # ── Strip cascade-inflated players before generating callouts ──
    # If the injury bump is >= 40% of the original (pre-cascade) projection,
    # the number is mostly cascade math, not a real expectation.
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

# Bust: high ownership + specific reasons (target: 10)
_BUST_EXPLANATIONS_HIGH_OWN = [
    "{own:.0f}% of the field lined up for this. {reasons}. When everyone agrees, I get nervous. I've learned to trust that instinct.",
    "{own:.0f}% owned. {reasons}. Everyone agreed on this pick — and that's exactly the problem.",
    "{own:.0f}% of the room is on {name}. {reasons}. Popular doesn't mean right. Never has.",
    "{own:.0f}% ownership. {reasons}. Crowded trade. Same mechanics every time.",
    "{name} at {own:.0f}% owned. {reasons}. The field went all-in on narrative. The data tells a different story.",
    "{own:.0f}% of the field on {name}. {reasons}. High ownership with red flags is the worst combination in GPPs.",
    "{own:.0f}% owned. {reasons}. The popular side of this trade has a bad risk profile. I'll pass.",
    "{name}: {own:.0f}% owned. {reasons}. When the whole room leans one direction, check what they're not seeing.",
    "{own:.0f}% ownership on {name}. {reasons}. The consensus pick with consensus blind spots.",
    "{own:.0f}% of the field likes {name}. {reasons}. I've seen this setup before. Different venue, same outcome.",
]

# Bust: low ownership + specific reasons (target: 8)
_BUST_EXPLANATIONS_LOW_OWN = [
    "{own:.0f}% owned, but the price tag is a trap. {reasons}. The salary says star, the situation says sit.",
    "{own:.0f}% owned — doesn't matter. {reasons}. Projections don't match reality here and the scoreboard won't either.",
    "Only {own:.0f}% of the field is on this, but the ones who are will regret it. {reasons}. Bad spot, bad price.",
    "{name} at {own:.0f}% owned. {reasons}. Low ownership doesn't make it a good bet when the situation is this bad.",
    "{own:.0f}% ownership. {reasons}. The price tag is doing the selling. The matchup isn't buying.",
    "{name}: {own:.0f}% owned. {reasons}. Not many people are on this. The few who are have the wrong read.",
    "Only {own:.0f}% on {name}. {reasons}. Low owned for a reason, but the reason isn't what the field thinks.",
    "{own:.0f}% owned. {reasons}. The salary looks right until you check the context. Then it doesn't.",
]

# Bust: high ownership fallback — no specific reasons (target: 6)
_BUST_FALLBACK_HIGH_OWN = [
    "{own:.0f}% owned and the numbers don't support it. Popularity isn't an edge. Never was.",
    "The field loves {name} tonight at {own:.0f}%. The data doesn't. I'll take the data.",
    "{name} at {own:.0f}% owned. The field is confident. The numbers are not. I know which one I trust.",
    "{own:.0f}% ownership on {name}. High conviction from the field, low conviction from the model. Mismatch.",
    "{name}: {own:.0f}% owned. The crowd says yes. The math says no. This isn't a close call.",
    "The field piled into {name} at {own:.0f}%. The data doesn't agree. I've seen enough to trust the data.",
]

# Bust: low ownership fallback — no specific reasons (target: 6)
_BUST_FALLBACK_LOW_OWN = [
    "{own:.0f}% owned — low exposure, but still a bad bet. The salary is doing the selling, not the data.",
    "Only {own:.0f}% of the field bit on {name}, but the price tag still doesn't add up. Pass.",
    "{name} at {own:.0f}% owned. Low traffic doesn't mean value. Sometimes a quiet trade is just a bad trade.",
    "{own:.0f}% on {name}. The field mostly avoided this. They got it right.",
    "{name}: {own:.0f}% owned. Low ownership for a reason. The math confirmed it.",
    "Only {own:.0f}% on {name}. Even the field got this one right. The situation doesn't add up.",
]

# Bust: fade fallback — from pre-classified fade candidates (target: 6)
_BUST_FADE_FALLBACK = [
    "The model doesn't like it at {own:.0f}% owned. The field followed each other into this one. Don't follow them.",
    "{own:.0f}% owned and the model is fading it. The data says pass. I agree.",
    "Fade at {own:.0f}% owned. The field piled in. The model disagrees. I trust the model.",
    "{own:.0f}% ownership. The model flagged this as a fade. Not every popular pick is a good pick.",
    "The model says fade at {own:.0f}% owned. When the crowd and the model disagree, I side with the model.",
    "{own:.0f}% owned. Model fade. The field can have this one.",
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

    # Filter OUT / IR / Suspended players — can't bust if you can't play
    _OUT_STATUSES = {"OUT", "IR", "SUSPENDED"}
    _work = pool.copy()
    if "status" in _work.columns:
        _before = len(_work)
        _work = _work[
            ~_work["status"].fillna("").str.strip().str.upper().isin(_OUT_STATUSES)
        ].reset_index(drop=True)
        _removed = _before - len(_work)
        if _removed:
            print(f"[generate_bust_call] Excluded {_removed} OUT/IR/Suspended player(s) from bust candidates")

    # Only consider relevant players (meaningful salary and ownership)
    df = _work[(_work["salary"] >= 5000) & (_work[own_col] > 3) & (_work["proj"] > 10)].copy()
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
    high_own = own > 15
    if reasons:
        reason_str = ", ".join(reasons).capitalize()
        templates = _BUST_EXPLANATIONS_HIGH_OWN if high_own else _BUST_EXPLANATIONS_LOW_OWN
        tmpl = _pick_template(templates, name, "bust_reasons")
        explanation = tmpl.format(name=name, own=own, reasons=reason_str)
    else:
        templates = _BUST_FALLBACK_HIGH_OWN if high_own else _BUST_FALLBACK_LOW_OWN
        tmpl = _pick_template(templates, name, "bust_fallback")
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

    tmpl = _pick_template_by_key(_BUST_FADE_FALLBACK, name, "bust_fade")
    return {
        "name": name,
        "salary": sal,
        "explanation": tmpl.format(own=own),
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
