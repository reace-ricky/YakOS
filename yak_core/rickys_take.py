"""yak_core.rickys_take -- Deterministic voice generation for Ricky's Take.

Generates three sections for the Edge Analysis tab:
  1. Last Night -- recap of previous slate hits/misses in Ricky's voice
  2. Tonight's Edges -- data-driven callouts about the current slate
  3. Bust Call -- one bold prediction for the biggest underperformer

All text is template-based with data slots. No LLM calls.

RICKY'S VOICE -- RIGHT ANGLE RICKY:
  Ex-Wall Street quant who got sick of bureaucrats, committees, and corporate
  politics. Quit the Street and now grinds DFS on his own. Sharp, cynical about
  institutions, occasionally self-deprecating. Speaks like someone who's sat
  through a thousand bad meetings and now refuses to sit through one more. He
  doesn't do hype. He trusts the math. He thinks consensus is a disease.

  Wall Street background: References to "the desk", "the floor", "my old shop",
  "trading book", "risk committee", "P&L", "alpha", "edge decay", "the bid".
  Anti-bureaucracy: Hates committees, group consensus, alignment meetings.
  Quant mindset: Trusts models, data, distributions, variance.
  DFS grinder: "the field", chalk, ownership leverage, GPP variance.
  Tone: Dry, confident, occasionally funny. Not mean-spirited.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Template selection -- date-seeded hash for daily rotation
# ---------------------------------------------------------------------------

def _pick_template(templates: list, player_name: str, category: str) -> str:
    """Pick a template using player name + date so it rotates daily.

    Different players get different templates on the same night.
    The same player gets a different template on different nights.
    """
    day = date.today().toordinal()
    seed = hash(f"{player_name}:{category}:{day}") % len(templates)
    return templates[seed]


def _pick_template_by_key(templates: list, key: str, category: str) -> str:
    """Pick a template using an arbitrary key + date for non-player contexts."""
    day = date.today().toordinal()
    seed = hash(f"{key}:{category}:{day}") % len(templates)
    return templates[seed]


# ---------------------------------------------------------------------------
# 1. LAST NIGHT -- recap generation
# ---------------------------------------------------------------------------

# Hit templates: player outperformed projection (10)
_HIT_TEMPLATES = [
    "{name} went off for {actual:.0f} on a {proj:.0f} projection. The model did the work. The results page agreed.",
    "{name} dropped {actual:.0f} against a {proj:.0f} line. That's what happens when you trust the data instead of the narrative.",
    "{name} crushed with {actual:.0f} actual on {proj:.0f} projected. No committee needed to see that one coming.",
    "{name} smashed at {actual:.0f}. Projected {proj:.0f}. The kind of edge that doesn't survive a group chat — which is exactly why it hit.",
    "{name} posted {actual:.0f} on a {proj:.0f} line. On the desk we called that alpha. Here it's just called doing your homework.",
    "{name} at {actual:.0f} vs {proj:.0f} projected. I ran tighter risk controls than most of this field runs their lineups, and even I'm impressed.",
    "{name} for {actual:.0f}. Line was {proj:.0f}. The model flagged it. The consensus ignored it. The P&L didn't.",
    "{name} delivered {actual:.0f} against {proj:.0f} projected. Quiet edge, loud outcome. My favorite kind.",
    "{name}: {actual:.0f} actual, {proj:.0f} projected. While the crowd was busy debating narratives, the math was busy being right.",
    "{name} cashed at {actual:.0f} on a {proj:.0f} projection. I left a seven-figure desk to find edges like this at 2am. Worth it.",
]

# Miss templates: player underperformed projection (10)
_MISS_TEMPLATES = [
    "{name} owners got cooked. {actual:.0f} actual on a {proj:.0f} line. The crowd followed each other right off a cliff.",
    "{name} bricked at {actual:.0f} on a {proj:.0f} projection. Consensus pick, consensus loss. Seen this movie before.",
    "{name} laid an egg — {actual:.0f} on a {proj:.0f} line. Half the field was on it because the other half was on it. That's not analysis, that's a herd.",
    "{name} at {actual:.0f} vs {proj:.0f} projected. Popular pick. Popular outcome. The bottom line doesn't grade on a curve.",
    "{name} put up {actual:.0f} on a {proj:.0f} line. The room agreed on this one. The room was wrong. Happens every time.",
    "{name}: {actual:.0f} actual against {proj:.0f}. On the floor we called this a crowded trade going sideways. In DFS they just call it Tuesday.",
    "{name} at {actual:.0f} vs {proj:.0f}. That's the kind of loss that fills a risk committee meeting. Except there's no committee here — just the cash register.",
    "{name} flamed out at {actual:.0f} on {proj:.0f} projected. Everyone was aligned. That's the first red flag I ever learned on the desk.",
    "{name} at {actual:.0f}. Line was {proj:.0f}. The group chat loved this pick. The final column didn't.",
    "{name}: {actual:.0f} on a {proj:.0f} projection. Chalk chasers learned a lesson. They'll forget it by tomorrow.",
]

# Pattern templates: chalk held (6)
_PATTERN_TEMPLATES_CHALK_HELD = [
    "High-salary chalk went {hit_count}-of-{total}. Sometimes the obvious play is the right play — just don't confuse that with doing the work.",
    "Top salaries went {hit_count}-of-{total}. Chalk held, but that's not the norm. Don't get comfortable.",
    "Chalk hit {hit_count}-of-{total} last night. Even a stopped clock. The edge is knowing when chalk is the play vs. when it's a trap.",
    "Top salaries cleared at {hit_count}-of-{total}. I'll acknowledge when the consensus is right. I just won't bet on it being right twice in a row.",
    "The premium names went {hit_count}-of-{total}. Nights like this make people overfit to chalk. That's when the contrarian window opens.",
    "Chalk cashed {hit_count}-of-{total}. My old PM would've called this a trending trade. The trend breaks when the crowd gets comfortable.",
]

# Pattern templates: chalk crumbled (6)
_PATTERN_TEMPLATES_CHALK_CRUMBLED = [
    "Only {hit_count} of the top-{total} salaries cleared projection. The crowd loves paying up for names. The end of day column doesn't care about names.",
    "Chalk crumbled — {hit_count}-of-{total} top salaries hit. Everyone nodded along on the same picks. Same result as every bad meeting.",
    "Top salaries went {hit_count}-of-{total}. A crowded trade that unwound. I've watched this exact movie on the desk a hundred times.",
    "Just {hit_count}-of-{total} chalk plays cleared. The committee loved every one of them. The final column didn't.",
    "Chalk at {hit_count}-of-{total}. When everyone agrees, the trade is already over. Same on Wall Street, same on DraftKings.",
    "{hit_count}-of-{total} top salaries hit their line. The herd ran together. The herd lost together. Position sizing over consensus — always.",
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
            if top_hits >= total * 0.7:
                tmpl = _pick_template_by_key(
                    _PATTERN_TEMPLATES_CHALK_HELD,
                    f"chalk:{top_hits}:{total}",
                    "chalk_held",
                )
                sentences.append(tmpl.format(hit_count=top_hits, total=total))
            elif top_hits <= total * 0.3:
                tmpl = _pick_template_by_key(
                    _PATTERN_TEMPLATES_CHALK_CRUMBLED,
                    f"chalk:{top_hits}:{total}",
                    "chalk_crumbled",
                )
                sentences.append(tmpl.format(hit_count=top_hits, total=total))

    return " ".join(sentences) if sentences else None


# ---------------------------------------------------------------------------
# 2. TONIGHT'S EDGES -- forward-looking callouts
# ---------------------------------------------------------------------------

# Salary mismatch templates (10)
_MISMATCH_TEMPLATES = [
    "{name} at ${sal} is mispriced. {proj:.0f} projected, {pts_1k:.1f} pts/$1K. The kind of inefficiency that used to pay my bonus. Now it pays lineups.",
    "{name} at ${sal} with {proj:.0f} projected. {pts_1k:.1f} pts/$1K. The market is wrong on this one and the market doesn't self-correct before lock.",
    "{name} at ${sal} projects to {proj:.0f}. That's {pts_1k:.1f} pts/$1K — a pricing gap the crowd won't notice until the results post.",
    "{name}, ${sal}. Projecting {proj:.0f} at {pts_1k:.1f} pts/$1K. This is the kind of mispricing that would've had my old PM on the phone in ten seconds.",
    "${sal} for {name}? With {proj:.0f} projected and {pts_1k:.1f} pts/$1K? If this were a trade, I'd be sizing up. It's a lineup slot. Same logic applies.",
    "{name} at ${sal}, {proj:.0f} projected, {pts_1k:.1f} pts/$1K. Left a seven-figure desk because I got tired of committee calls at 6am. Now I'm here finding edges the committee would've debated for a week.",
    "{name} at ${sal} with a {proj:.0f} projection. {pts_1k:.1f} pts/$1K. Mispriced. Moving on.",
    "{name}: ${sal}, {proj:.0f} projected, {pts_1k:.1f} per $1K. The book says this is an asymmetric spot. I trust the book.",
    "{name} at ${sal}. {proj:.0f} projected at {pts_1k:.1f} pts/$1K. What's the crowd seeing that the data isn't? Nothing. They're just not looking.",
    "{name}, ${sal}, {proj:.0f} projected. {pts_1k:.1f} pts per $1K. On the desk we'd call this a clean entry. In DFS I call it a lock.",
]

# Ownership trap templates (10)
_TRAP_TEMPLATES = [
    "{own:.0f}% of the field is on {name} tonight. Same herd energy. {flags}. The bottom line doesn't care how popular the pick was.",
    "{own:.0f}% of the field lined up behind {name}. {flags}. That's not conviction, that's a crowded trade. I've watched those unwind.",
    "{name} at {own:.0f}% owned. {flags}. Everyone's nodding along on this one. I've sat in enough rooms to know what that means.",
    "{name} at {own:.0f}% — the consensus trade. {flags}. On the desk, when everyone's on the same side, someone's about to get carried out.",
    "{own:.0f}% ownership on {name}. {flags}. The alignment meeting agreed. The cash register will disagree.",
    "{name}, {own:.0f}% owned. {flags}. I could be wrong. I've been wrong before. But not as often as the consensus.",
    "{own:.0f}% of the room loves {name}. {flags}. When I was on the desk, we had a word for trades where everyone agreed: overexposed.",
    "{name} at {own:.0f}% owned tonight. {flags}. The group chat is all-in. That's the signal, not the edge.",
    "{own:.0f}% on {name}. {flags}. Crowded trades don't blow up every time. But when they do, they take the whole room with them.",
    "{name}: {own:.0f}% owned, {flags}. In my old shop this was the trade the intern pitched because everyone else pitched it first. Same energy.",
]

# Contrarian main templates (10)
_CONTRARIAN_TEMPLATES = [
    "{name} at {own:.0f}% owned. The crowd didn't do the work — {reason}. This is the kind of edge that doesn't survive a committee.",
    "{name} sitting at {own:.0f}% owned. {reason}. Nobody in the room bothered to look. That's the whole edge.",
    "{name}, {own:.0f}% owned. {reason}. The chalk chasers are too busy copying each other's homework to notice.",
    "{name} at {own:.0f}%. {reason}. On the desk we called this an empty room trade. The best P&L always came from the empty room.",
    "{own:.0f}% owned — {name}. {reason}. I ran a book with tighter risk controls than this field runs their lineups, and this clears every filter.",
    "{name}, barely {own:.0f}% owned. {reason}. The consensus missed it. The model didn't. That's why I trust the model.",
    "{name} at {own:.0f}%. {reason}. Half my career was finding what the room overlooked. Old habits.",
    "{own:.0f}% on {name}. {reason}. The crowd is looking somewhere else. Good. That's where the alpha lives — in the gap between attention and reality.",
    "{name} at {own:.0f}% owned. {reason}. When nobody's bidding, that's when you step in. Same principle on the desk, same principle here.",
    "{name}: {own:.0f}% owned, {reason}. The stakeholder call would've talked themselves out of this by now. Advantage: no stakeholder call.",
]

# Contrarian fallback templates (8)
_CONTRARIAN_FALLBACK = [
    "{name} at ${sal} with {proj:.0f} projected and only {own:.0f}% owned. The crowd is asleep. I've made a career on the other side of that trade.",
    "{name} at ${sal}, {proj:.0f} projected, {own:.0f}% owned. When nobody's looking, that's when you look harder.",
    "{name}: ${sal}, {proj:.0f} projected, {own:.0f}% owned. This is the kind of play that wins GPPs and nobody puts it in their recap.",
    "{name} at ${sal} with {proj:.0f} projected. {own:.0f}% of the field walked right past this. Their loss.",
    "{name}, ${sal}, {proj:.0f} projected, {own:.0f}% owned. Every quant knows — the best edge is the one nobody else is running.",
    "{name} for ${sal}. {proj:.0f} projected at {own:.0f}% owned. I've seen worse risk/reward ratios get approved by a committee of twelve. This one doesn't need a committee.",
    "{name} at ${sal}. {proj:.0f} projected, {own:.0f}% owned. The market is giving this away. I'm not going to argue with free money.",
    "${sal} for {name}, {proj:.0f} projected, {own:.0f}% owned. If the field doesn't want the edge, I'll take it.",
]

# Lotto templates (8)
_LOTTO_TEMPLATES = [
    "GPP lotto tickets: {names}. Low owned, high environment, mispriced. These are the kind of names that show up in winning lineups and nobody saw it coming.",
    "GPP lotto shelf: {names}. The crowd walked right past them. One of these hits and your lineup looks like genius.",
    "Lotto plays: {names}. Cheap, ignored, and sitting in the right game environment. That's the whole formula.",
    "Asymmetric bets: {names}. Small cost, massive upside if they connect. That's how you build a trading book. Same applies here.",
    "Long-shot shelf: {names}. Low ownership, high ceiling, the right game script. You don't need all of them to hit. You need one.",
    "Value tail: {names}. The crowd is fighting over chalk while these names sit unclaimed. Some of my best P&L came from the names nobody wanted.",
    "Lotto window: {names}. If even one of these connects, the crowd's chalk-heavy lineups are in trouble. Variance is the GPP player's best friend.",
    "GPP dart throws: {names}. Low cost, low owned, high environment. My old risk committee would hate the position. Good thing they're not here.",
]

# Game environment templates (6)
_GAME_ENV_TEMPLATES = [
    "{matchup} has a {ou:.0f} total. Pace and points. This is where the math lives tonight.",
    "{matchup} — {ou:.0f} over/under. The environment is screaming. Stack candidates live here.",
    "{matchup} at {ou:.0f} total. When I ran models on the desk, environment was the first filter. Nothing's changed.",
    "{matchup}: {ou:.0f} O/U. High pace, high total, high ceiling. The distribution is wide open and that's exactly what GPP lineups need.",
    "{matchup} with a {ou:.0f} total. This game is the engine room of the slate. Build around it or explain why you didn't.",
    "{matchup} at {ou:.0f}. Points environment. If you're not looking here first, you're not looking at the right things.",
]


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
                    matchup = f"{teams[0]}-{teams[1]}"
                    tmpl = _pick_template_by_key(
                        _GAME_ENV_TEMPLATES, matchup, "game_env"
                    )
                    return [tmpl.format(matchup=matchup, ou=ou)]
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

# Bust high ownership + reasons (10)
_BUST_EXPLANATIONS_HIGH_OWN = [
    "{own:.0f}% of the field is about to learn a lesson. {reasons}. I've watched this exact setup blow up a hundred portfolios.",
    "{own:.0f}% of the field lined up for this. {reasons}. The consensus was wrong in the boardroom and it's wrong here.",
    "{own:.0f}% owned. {reasons}. Everyone agreed on this pick — and that's exactly the problem.",
    "{own:.0f}% of the room is on this. {reasons}. On the desk we called it concentration risk. In DFS the crowd calls it chalk. Same outcome.",
    "{own:.0f}% owned tonight. {reasons}. When the whole floor is on one side, somebody's getting carried out. I've seen it enough times.",
    "{reasons}. And {own:.0f}% of the field signed up for it. I sat through enough alignment meetings to know what happens when everyone agrees.",
    "{own:.0f}% ownership. {reasons}. The risk committee would've flagged this in five minutes. The crowd doesn't have a risk committee.",
    "{own:.0f}% of the field. {reasons}. This has all the hallmarks of a position that looked great in the pitch deck and terrible on the P&L.",
    "{reasons}. At {own:.0f}% owned, the damage is priced in for the whole room. That's not a bet — that's a liability.",
    "{own:.0f}% ownership with {reasons}. I ran tighter books than this field runs their lineups. This doesn't pass the filter.",
]

# Bust low ownership + reasons (8)
_BUST_EXPLANATIONS_LOW_OWN = [
    "{own:.0f}% owned, but the price tag is a trap. {reasons}. The salary says star, the situation says sit.",
    "{own:.0f}% owned — doesn't matter. {reasons}. Projections don't match reality here and the results won't either.",
    "Only {own:.0f}% of the field is on this, but the ones who are will regret it. {reasons}. Bad spot, bad price.",
    "{own:.0f}% owned. Low exposure doesn't mean low risk. {reasons}. The salary is doing the marketing, not the data.",
    "{reasons}. At {own:.0f}% owned it's a quiet mistake, but it's still a mistake. I don't need the crowd to be wrong for me to be right.",
    "{own:.0f}% ownership. {reasons}. Small position, sure. But a bad trade at any size is still a bad trade.",
    "{own:.0f}% owned — the crowd mostly avoided this one. The ones who didn't: {reasons}. Pass.",
    "Only {own:.0f}% of the field, but {reasons}. Sometimes low ownership means the crowd got it right by accident.",
]

# Bust high ownership fallback (6)
_BUST_FALLBACK_HIGH_OWN = [
    "{own:.0f}% owned and the numbers don't support it. Popularity isn't an edge. Never was.",
    "The field loves {name} tonight at {own:.0f}%. The data doesn't. I'll take the data.",
    "{own:.0f}% of the crowd on {name}. When my old PM saw positioning like this, he'd start hedging. There's no hedge in DFS — just a fade.",
    "{name} at {own:.0f}% owned. The group chat is excited. The model is not. I know which one I trust.",
    "{own:.0f}% on {name}. The herd is moving. I've spent a career stepping in front of the herd when the math says to.",
    "{name} at {own:.0f}% owned and the data is shrugging. The consensus is loud but the signal is quiet. I'll follow the signal.",
]

# Bust low ownership fallback (6)
_BUST_FALLBACK_LOW_OWN = [
    "{own:.0f}% owned — low exposure, but still a bad bet. The salary is doing the selling, not the data.",
    "Only {own:.0f}% of the field bit on {name}, but the price tag still doesn't add up. Pass.",
    "{name} at {own:.0f}% owned. Not many takers, and for good reason. The model agrees with the fade here.",
    "{own:.0f}% on {name}. Low owned for a reason the crowd may have stumbled into by accident. Even a broken clock.",
    "{name}, {own:.0f}% owned. Quiet fade. The salary pulled a few people in but the situation is waving them off.",
    "Only {own:.0f}% on {name}. The crowd accidentally got this one right. The price tag is the only thing selling this spot.",
]

# Bust fade fallback — for _bust_from_fades (6)
_BUST_FADE_FALLBACK = [
    "The model doesn't like it at {own:.0f}% owned. The crowd followed each other into this one. Don't follow them.",
    "{own:.0f}% owned and the model says fade. The consensus lined up. The math didn't. I know which one ages better.",
    "At {own:.0f}% owned, the crowd is in. The model is out. I've been on enough wrong sides of consensus to know which matters.",
    "{own:.0f}% of the field is here. The model waved them off. The crowd didn't listen. They rarely do.",
    "The model flagged this as a fade at {own:.0f}% owned. The field ignored it. Same story, different slate.",
    "{own:.0f}% owned. The model's fade signal is clear. When the data says no and the crowd says yes, I take the data every time.",
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

    tmpl = _pick_template_by_key(_BUST_FADE_FALLBACK, name, "bust_fade_fb")
    explanation = tmpl.format(name=name, own=own)

    return {
        "name": name,
        "salary": sal,
        "explanation": explanation,
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
