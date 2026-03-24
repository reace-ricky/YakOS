"""yak_core.rickys_take -- Deterministic voice generation for The Board.

Generates supplementary content rendered below The Board's confidence-gated
player cards (see yak_core/board.py for the primary Board logic):
  1. Last Slate -- recap of previous slate hits/misses in Ricky's voice
  2. Edge Callouts -- data-driven callouts about the current slate
  3. Bust Call -- one bold prediction for the biggest underperformer

All text is template-based with data slots. No LLM calls.

Rotation system
~~~~~~~~~~~~~~~
Templates are selected deterministically via a seeded hash so that:
  - The same slate+player always produces the same output (reproducible).
  - No two callouts in the same post use the same template from a given
    category (intra-post dedup via ``_TemplateRotator``).
  - Across consecutive days the seed shifts so phrasing feels fresh. With
    pool sizes of 20-30 per category, the same exact phrase won't repeat
    for 3-4+ days even for the same player.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Template rotation engine -- deterministic, dedup-aware, multi-day spread
# ---------------------------------------------------------------------------

class _TemplateRotator:
    """Tracks templates used within a single post to guarantee no repeats.

    Seed formula: hash(slate_date + player_name + category + day_ordinal)
    where day_ordinal shifts the pick daily so phrases rotate across days.
    """

    def __init__(self, slate_date: Optional[str] = None) -> None:
        self._used: Dict[str, Set[int]] = {}  # category -> set of used indices
        self._slate_date = slate_date or date.today().isoformat()

    def _seed(self, key: str, category: str) -> int:
        """Deterministic seed from key + category + slate date."""
        raw = f"{key}:{category}:{self._slate_date}"
        return int(hashlib.sha256(raw.encode()).hexdigest(), 16)

    def pick(self, templates: list, key: str, category: str) -> str:
        """Pick a template that hasn't been used in this post for *category*."""
        n = len(templates)
        if n == 0:
            return ""
        used = self._used.setdefault(category, set())
        seed = self._seed(key, category)

        # Walk through candidates in deterministic order until we find one
        # that hasn't been used yet in this post.
        for offset in range(n):
            idx = (seed + offset) % n
            if idx not in used:
                used.add(idx)
                return templates[idx]

        # All used — reset and return the first pick (shouldn't happen with
        # pool sizes >> callouts per post).
        used.clear()
        idx = seed % n
        used.add(idx)
        return templates[idx]


# Module-level rotator instance. Re-created per Streamlit page render because
# the module is re-imported each time via the edge_tab lazy import.  For safety
# we expose a reset helper that edge_tab can call.
_rotator = _TemplateRotator()


def reset_rotator(slate_date: Optional[str] = None) -> None:
    """Reset the post-level rotator (call once per render cycle)."""
    global _rotator
    _rotator = _TemplateRotator(slate_date)


def _pick_template(templates: list, player_name: str, category: str) -> str:
    """Deterministic, dedup-aware template selection keyed by player name."""
    return _rotator.pick(templates, player_name, category)


def _pick_template_by_key(templates: list, key: str, category: str) -> str:
    """Deterministic, dedup-aware template selection keyed by an arbitrary string."""
    return _rotator.pick(templates, key, category)


# ---------------------------------------------------------------------------
# 1. LAST NIGHT -- recap generation
# ---------------------------------------------------------------------------

# ── Board Call hit templates (smirky, first-person) ──
_BOARD_CORE_HIT = [
    "Put {name} on the board as a core play. Went for {actual:.0f} on a {proj:.0f} line. That's the whole process.",
    "{name} was core. {actual:.0f} actual, {proj:.0f} projected. Called it, cashed it.",
    "Core play {name}: {actual:.0f} on {proj:.0f}. The board doesn't miss on the anchors.",
    "{name}: core lock. {actual:.0f} on a {proj:.0f} line. The model did the heavy lifting. I just pressed publish.",
    "Called {name} core. {actual:.0f} actual, {proj:.0f} projected. Not a debate — a data point.",
    "{name} was the anchor. {actual:.0f} on {proj:.0f}. Sometimes the obvious call is the right one. This was one of those times.",
    "Core: {name}. {actual:.0f} on {proj:.0f}. Anchors hold or they don't. This one held.",
    "{name} — core, {actual:.0f} actual, {proj:.0f} projected. I don't celebrate singles. But I log them.",
    "Put {name} in the core bucket. {actual:.0f} on a {proj:.0f} line. That's what conviction looks like.",
    "{name} anchored the board. {actual:.0f} on {proj:.0f}. The field debated. I didn't.",
]

_BOARD_VALUE_HIT = [
    "Value play {name} at ${sal} went for {actual:.0f}. That's a {tier} call that printed.",
    "{name} at ${sal} — called it as a value target. Dropped {actual:.0f}. You're welcome.",
    "Put {name} on the board at ${sal}. Value play. Went for {actual:.0f}. Math works.",
    "{name} at ${sal}: value tier. {actual:.0f} actual. The cheap plays only look cheap in hindsight.",
    "Value call: {name} at ${sal}. Went for {actual:.0f}. The price was wrong. I noticed.",
    "{name}, ${sal}, {actual:.0f} FP. Value play. The field spent up. I spent smart.",
    "Called {name} as value at ${sal}. Dropped {actual:.0f}. Salary inefficiency, exploited.",
    "{name}: ${sal}, {actual:.0f} actual. Value play that cashed. The scoreboard validates, not the salary tag.",
    "Value target {name} at ${sal} delivered {actual:.0f}. That's the kind of edge the field ignores until it costs them.",
    "${sal} for {name}. Went for {actual:.0f}. The board said value. The box score confirmed.",
]

_BOARD_LOTTO_HIT = [
    "Lotto pick {name} at ${sal}. Went for {actual:.0f}. Nobody was looking. I was.",
    "{name} — the lotto play — dropped {actual:.0f} at ${sal}. Cheap, ignored, cashed.",
    "Called {name} as a lotto ticket at ${sal}. Hit for {actual:.0f}. That's the edge.",
    "{name} at ${sal}. Lotto tier. {actual:.0f} actual. The field didn't bother scrolling down.",
    "Lotto play {name} at ${sal}: {actual:.0f}. The kind of pick that separates lineups. Quietly.",
    "{name}, ${sal}, {actual:.0f}. Lotto. The field spent the salary elsewhere. Their loss.",
    "Called {name} as a deep-roster lotto. ${sal}. Went for {actual:.0f}. Asymmetric outcomes — that's the whole game.",
    "Lotto ticket: {name} at ${sal}. {actual:.0f} actual. The field priced this out of conversation. The scoreboard priced it back in.",
    "{name} at ${sal} — lotto tier. Dropped {actual:.0f}. Nobody talks about these picks until they win.",
    "${sal}. {name}. Lotto play. {actual:.0f} FP. The field walked right past it.",
]

_BOARD_FADE_HIT = [
    "Called the fade on {name}. Went for {actual:.0f} on a {proj:.0f} line. The field got cooked.",
    "Faded {name}. {actual:.0f} on {proj:.0f}. Everyone who followed the herd paid for it.",
    "{name} was the fade call. {actual:.0f} actual. The crowd had it wrong. Again.",
    "Fade: {name}. {actual:.0f} on a {proj:.0f} line. Said don't. They didn't listen. They never do.",
    "Called {name} as a fade. {actual:.0f} actual, {proj:.0f} projected. The consensus had the wrong read.",
    "{name} — faded. {actual:.0f} on {proj:.0f}. The crowd loaded up. The scoreboard unloaded them.",
    "Faded {name}. {actual:.0f} actual. The field chased the name. I read the numbers.",
    "{name}: fade call. {actual:.0f} on {proj:.0f}. The crowd went in. I stepped aside. Math wins.",
    "Called the fade. {name}: {actual:.0f} on {proj:.0f}. Popular and wrong. The two go together more than people admit.",
    "{name} was the fade. {actual:.0f} actual. Everyone else saw the name. I saw the context.",
]

_BOARD_SUMMARY_GOOD = [
    "The board went {hit_count}-of-{total} on named calls last night.",
    "{hit_count} of {total} board calls cleared. Process.",
    "Board: {hit_count} for {total}. Not perfect. Not supposed to be. But the edge is there.",
    "{hit_count}-of-{total} on the board. Good night. Same process tomorrow.",
    "The board cleared {hit_count} of {total}. Math works. Moving on.",
    "Board hit rate: {hit_count}/{total}. Results follow process. They did last night.",
    "{hit_count} out of {total} board calls landed. The model earns its keep.",
    "The board went {hit_count}-of-{total}. Clean slate. Clean results.",
]

_BOARD_SUMMARY_BAD = [
    "Board went {hit_count}-of-{total}. Not the night. Back at it.",
    "{hit_count} of {total} board calls hit. Process doesn't change.",
    "Board: {hit_count} for {total}. Bad night. Process still right. Outcomes are noisy.",
    "{hit_count}-of-{total}. Rough one. Math doesn't bat 1.000. But it bats better than instinct.",
    "The board went {hit_count} of {total}. Variance collected. Process doesn't flinch.",
    "{hit_count} of {total}. Not the night. The process doesn't care about one bad sample.",
    "Board took an L — {hit_count} of {total}. Variance happens. The model adjusts. So do I.",
    "{hit_count}-of-{total} on board calls. Off night. Same approach tomorrow.",
]

# ── SE lineup pick templates ──
_SE_PICK_HIT = [
    "Called {name} at ${sal}. Dropped {actual:.0f}. You're welcome.",
    "{name} — {actual:.0f} on a {proj:.0f} line. {tag} pick. Called it.",
    "Put {name} on the board at ${sal}. Went for {actual:.0f}. That's not luck, that's process.",
    "{name}: {actual:.0f} actual, {proj:.0f} projected. I said play this. The scoreboard agreed.",
    "{tag} pick {name} went off — {actual:.0f} on {proj:.0f}. Told you.",
    "{name} at ${sal} for {actual:.0f}. {tag} lineup cashed. Moving on.",
    "{name}: ${sal}. {actual:.0f} actual. {tag} pick. The model said go. I went.",
    "SE pick {name} delivered — {actual:.0f} on {proj:.0f}. The number was right. The field was elsewhere.",
    "{name} at ${sal}, {actual:.0f} FP. {tag} tier. Locked and loaded. Results followed.",
    "Called {name} in the {tag} slot. ${sal}. {actual:.0f} actual. That's the process at work.",
    "{tag}: {name} at ${sal}. {actual:.0f} on a {proj:.0f} line. Clean read.",
    "{name} — {tag} pick. {actual:.0f} on {proj:.0f}. I don't guess. I calculate.",
    "SE: {name}, ${sal}, {actual:.0f} FP. {tag} play. The model called it. I published it.",
    "{name} at ${sal}. {tag} pick. {actual:.0f} actual on {proj:.0f} projected. That's edge.",
    "Locked {name} in the {tag} slot at ${sal}. {actual:.0f}. The data was clear. The result was clearer.",
]

_SE_PICK_MISS = [
    "{name} bricked — {actual:.0f} on {proj:.0f}. Bad beat. Process was right, outcome wasn't.",
    "{name}: {actual:.0f} on a {proj:.0f} line. Didn't hit. Math doesn't bat 1.000.",
    "{name} went for {actual:.0f} on {proj:.0f}. Miss. The model was right about the spot. The player wasn't.",
    "{name}: {actual:.0f} actual, {proj:.0f} projected. Variance. Not changing the approach over one data point.",
    "Miss on {name} — {actual:.0f} on a {proj:.0f} line. Bad outcome. Right process. I've made peace with that math.",
    "{name} at {actual:.0f} on {proj:.0f}. Bricked. The spot was right. The result wasn't. That's variance.",
    "{name}: {actual:.0f} on {proj:.0f}. Didn't cash. Same call tomorrow in the same spot. The math doesn't change.",
    "Took an L on {name} — {actual:.0f} on a {proj:.0f} line. Process was sound. Outcome was noise.",
]

_SE_LINEUP_SUMMARY_GOOD = [
    "SE lineups went {hit_count}-of-{total} on the night. The board delivered.",
    "{hit_count} of {total} SE lineups cleared. Not bad for a Thursday.",
    "The picks went {hit_count}-for-{total}. Process works. Results follow.",
    "SE results: {hit_count} of {total}. Clean night. Same process tomorrow.",
    "{hit_count}-of-{total} SE picks hit. The model keeps earning its keep.",
    "SE went {hit_count} for {total}. Good night at the board.",
    "{hit_count} of {total} SE picks landed. Process. Patience. Results.",
    "SE board: {hit_count}/{total}. The data did the work. I just followed it.",
]

_SE_LINEUP_SUMMARY_BAD = [
    "SE lineups went {hit_count}-of-{total}. Rough night. Back at it.",
    "{hit_count} of {total} cleared. Not the night. Process doesn't change.",
    "SE went {hit_count} for {total}. Off night. The model recalibrates. So do I.",
    "{hit_count}-of-{total} on SE picks. Variance night. Same approach tomorrow.",
    "SE picks: {hit_count} of {total}. Not the results. Still the process. Moving on.",
    "{hit_count} of {total} SE lineups hit. Bad sample. The math doesn't panic over one night.",
    "SE board: {hit_count}/{total}. Rough slate. Process stays the same.",
    "{hit_count}-of-{total}. SE took an L. One night doesn't rewrite the system.",
]

# ── Generic fallback templates (when no SE archive exists) ──
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
    "{name}: {actual:.0f} on a {proj:.0f} line. The model saw it. The field didn't. That's the gap.",
    "{name} went for {actual:.0f}. Projected {proj:.0f}. Straightforward edge. Straightforward result.",
    "{name} delivered {actual:.0f} on a {proj:.0f} projection. Not flashy. Just correct.",
    "{name}: {actual:.0f} actual, {proj:.0f} projected. The number was right there for anyone willing to look.",
    "{name} hit for {actual:.0f} on a {proj:.0f} line. The field ignored it. The scoreboard didn't.",
    "{name} dropped {actual:.0f} against {proj:.0f}. The kind of pick that looks obvious in hindsight. It wasn't.",
    "{name} for {actual:.0f} on {proj:.0f}. I followed the data. The data was right. End of story.",
    "{name}: {actual:.0f} on {proj:.0f}. No debate. Just the number.",
    "{name} went for {actual:.0f} against a {proj:.0f} line. The model doesn't get nervous. Neither do I.",
    "{name}: {actual:.0f} actual. {proj:.0f} projected. Edge identified, edge exploited. That's how this works.",
]

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
    "{name}: {actual:.0f} on {proj:.0f}. The crowd's pick. The crowd's result. Nobody learned anything.",
    "{name} at {actual:.0f} vs {proj:.0f}. When everyone's on the same side, the other side is usually right.",
    "{name} bricked — {actual:.0f} on {proj:.0f}. The narrative sold it. The box score told the truth.",
    "{name}: {actual:.0f} actual, {proj:.0f} projected. The field followed the name. The name didn't deliver.",
    "{name} went for {actual:.0f} on a {proj:.0f} line. Group-think premium: paid in full.",
    "{name} at {actual:.0f}. Projected {proj:.0f}. Popular. Confident. Wrong.",
    "{name}: {actual:.0f} on {proj:.0f}. The crowd was loud about this one. Loud doesn't mean right.",
    "{name} laid down {actual:.0f} against a {proj:.0f} line. The consensus case sounded great. The scoreboard disagreed.",
    "{name}: {actual:.0f} vs {proj:.0f}. The field went all-in. The results went the other way.",
    "{name} at {actual:.0f} on a {proj:.0f} projection. The most popular pick on the board. The least productive too.",
]

_PATTERN_TEMPLATES_CHALK_HELD = [
    "High-salary chalk went {hit_count}-of-{total}. Sometimes the obvious play is the right play — just don't confuse that with doing the work.",
    "Top salaries went {hit_count}-of-{total}. Chalk held, but that's not the norm. Don't get comfortable.",
    "Chalk went {hit_count}-of-{total}. Even a broken clock. Don't let one good night turn into lazy process.",
    "Top salaries cleared at {hit_count}-of-{total}. The field got lucky. Luck doesn't compound.",
    "{hit_count}-of-{total} top salaries hit. Chalk held this slate. Regression is patient — it'll collect eventually.",
    "High-priced chalk: {hit_count}-of-{total}. Credit where it's due. But one night doesn't change the base rates.",
    "Chalk cleared {hit_count} of {total}. Good for them. The question is whether it changes your approach. It shouldn't.",
    "Top salaries: {hit_count}-of-{total}. Chalk night. Happens. Don't confuse correlation with causation.",
    "{hit_count} of {total} expensive plays hit. Chalk works until it doesn't. The base rates haven't changed.",
    "Chalk went {hit_count}-of-{total}. One good night doesn't make paying up a system. But the results were real.",
    "High-salary plays: {hit_count}/{total}. Chalk held. File it. Don't build a strategy around one sample.",
    "Top-tier chalk: {hit_count} of {total} cleared. Nice night for the lazy process. It won't last.",
]

_PATTERN_TEMPLATES_CHALK_CRUMBLED = [
    "Only {hit_count} of the top-{total} salaries cleared projection. The field loves paying up for names. The scoreboard doesn't care about names.",
    "Chalk crumbled — {hit_count}-of-{total} top salaries hit. Everyone nodded along on the same picks. Same result as every bad meeting.",
    "Top salaries went {hit_count}-of-{total}. The consensus tax came due.",
    "{hit_count}-of-{total} chalk plays hit. Expensive and wrong. The field's favorite combination.",
    "Chalk at {hit_count}-of-{total}. Paying up for comfort is still paying up. The scoreboard doesn't offer refunds.",
    "Only {hit_count} of {total} top salaries delivered. The field paid retail for names and got wholesale results.",
    "Top salaries: {hit_count} of {total}. Chalk crumbled. The field paid up for familiarity and got burned for it.",
    "{hit_count}-of-{total} chalk plays cleared. The expensive names cost exactly what they're worth — too much.",
    "Chalk went {hit_count} of {total}. The field's instinct was to pay up. The scoreboard's instinct was to humble them.",
    "Only {hit_count} of {total} high-priced plays hit. The field confused expensive with good. Different things.",
    "Top-{total} salaries: {hit_count} cleared. Chalk failed. The field's most confident picks were its worst.",
    "{hit_count} of {total} chalk plays delivered. The premium names collected the salary. They didn't return the value.",
]


def _load_previous_archive(sport: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Load the most recent SE picks + edge_analysis archive (up to 7 days).

    Searches two locations in priority order:
      1. data/lineup_archive/ — dedicated archive files
      2. data/published/{sport}/ — the most-recently published edge analysis
         (only used when its slate date is strictly before today)

    Returns (se_picks_df, edge_analysis_dict) — either can be None.
    """
    import json
    from pathlib import Path
    from .config import YAKOS_ROOT

    archive_dir = Path(YAKOS_ROOT) / "data" / "lineup_archive"
    published_dir = Path(YAKOS_ROOT) / "data" / "published" / sport.lower()

    se_picks: Optional[pd.DataFrame] = None
    edge_analysis: Optional[Dict[str, Any]] = None
    archive_ea_date: Optional[date] = None
    today = date.today()

    # --- Source 1: lineup_archive directory ---
    if archive_dir.exists():
        for days_back in range(1, 8):
            check = today - timedelta(days=days_back)
            date_str = check.isoformat()

            # SE picks
            if se_picks is None:
                for f in sorted(archive_dir.glob(f"{date_str}_*_se_picks.parquet")):
                    try:
                        df = pd.read_parquet(f)
                        if "player_name" in df.columns:
                            se_picks = df
                            break
                    except Exception:
                        continue

            # Edge analysis (Board calls)
            if edge_analysis is None:
                ea_path = archive_dir / f"{date_str}_{sport.lower()}_edge_analysis.json"
                if ea_path.exists():
                    try:
                        edge_analysis = json.loads(ea_path.read_text())
                        archive_ea_date = check
                    except Exception:
                        pass

            if se_picks is not None and edge_analysis is not None:
                break

    # --- Source 2: published edge analysis ---
    # Use the published edge analysis when its slate date is strictly before
    # today AND more recent than any archive edge analysis found above.
    if published_dir.exists():
        meta_path = published_dir / "slate_meta.json"
        ea_path = published_dir / "edge_analysis.json"
        if meta_path.exists() and ea_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                pub_date_str = meta.get("date", "")
                pub_date = date.fromisoformat(pub_date_str) if pub_date_str else None
                if pub_date and pub_date < today:
                    if archive_ea_date is None or pub_date > archive_ea_date:
                        edge_analysis = json.loads(ea_path.read_text())
            except Exception:
                pass

    return se_picks, edge_analysis


def _build_board_recap(
    edge_analysis: Dict[str, Any],
    actuals_map: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Score the Board's named calls against actuals. Return 1-3 sentences."""
    sentences: List[str] = []
    total_calls = 0
    hit_calls = 0

    # Collect the best hit across all tiers
    best_hit: Optional[Dict[str, Any]] = None
    best_hit_tier: str = ""
    best_fade: Optional[Dict[str, Any]] = None

    for tier, templates in [
        ("core_plays", _BOARD_CORE_HIT),
        ("leverage_plays", _BOARD_VALUE_HIT),
        ("value_plays", _BOARD_VALUE_HIT),
    ]:
        for p in edge_analysis.get(tier, []):
            name = p.get("player_name", "")
            if name not in actuals_map:
                continue
            a = actuals_map[name]
            total_calls += 1
            if a["delta"] > 0:
                hit_calls += 1
                if best_hit is None or a["delta"] > best_hit["delta"]:
                    best_hit = {**a, "name": name, "tier": tier, "templates": templates}
                    best_hit_tier = tier

    # Fade candidates: hit = player UNDERPERFORMED (delta < 0)
    for p in edge_analysis.get("fade_candidates", []):
        name = p.get("player_name", "")
        if name not in actuals_map:
            continue
        a = actuals_map[name]
        total_calls += 1
        if a["delta"] < 0:  # fade was correct
            hit_calls += 1
            if best_fade is None or a["delta"] < best_fade["delta"]:
                best_fade = {**a, "name": name}

    # Build sentences: best hit callout
    if best_hit:
        h = best_hit
        sal = f"{h['salary']:,}" if h.get("salary") else "?"
        if best_hit_tier == "core_plays":
            tmpl = _pick_template(_BOARD_CORE_HIT, h["name"], "board_core")
            sentences.append(tmpl.format(
                name=h["name"], actual=h["actual"], proj=h["projected"],
            ))
        else:
            tmpl = _pick_template(_BOARD_VALUE_HIT, h["name"], "board_value")
            sentences.append(tmpl.format(
                name=h["name"], actual=h["actual"], proj=h["projected"],
                sal=sal, tier=best_hit_tier.replace("_plays", ""),
            ))

    # Fade callout
    if best_fade and len(sentences) < 2:
        f = best_fade
        tmpl = _pick_template(_BOARD_FADE_HIT, f["name"], "board_fade")
        sentences.append(tmpl.format(
            name=f["name"], actual=f["actual"], proj=f["projected"],
        ))

    # Board summary
    if total_calls > 0 and len(sentences) < 3:
        if hit_calls >= total_calls * 0.5:
            tmpl = _pick_template_by_key(_BOARD_SUMMARY_GOOD, f"{hit_calls}:{total_calls}", "board_good")
        else:
            tmpl = _pick_template_by_key(_BOARD_SUMMARY_BAD, f"{hit_calls}:{total_calls}", "board_bad")
        sentences.append(tmpl.format(hit_count=hit_calls, total=total_calls))

    return sentences


def generate_last_night(
    recap: Optional[Dict[str, Any]],
    sport: str = "nba",
) -> Optional[str]:
    """Generate a smirky recap highlighting Ricky's Board calls and SE picks.

    Priority order:
      1. Board calls (core/leverage/value/lotto/fade) from archived edge_analysis
      2. SE lineup picks from archived parquets
      3. Generic hit/miss fallback

    Parameters
    ----------
    recap : dict or None
        Output of slate_recap.get_previous_slate_recap().
    sport : str
        Sport code for loading archives.
    """
    if recap is None:
        return None

    players = recap.get("players", [])
    if not players:
        return None

    actuals_map = {p["player_name"]: p for p in players}
    se_picks, prev_edge = _load_previous_archive(sport)

    sentences: List[str] = []

    # ── Board calls recap (highest priority) ──
    if prev_edge:
        sentences.extend(_build_board_recap(prev_edge, actuals_map))

    # ── SE lineup picks ──
    if se_picks is not None and not se_picks.empty and "player_name" in se_picks.columns:
        se_names = se_picks["player_name"].unique().tolist()
        se_tags = {}
        if "ricky_tag" in se_picks.columns:
            for _, row in se_picks.drop_duplicates("player_name").iterrows():
                se_tags[row["player_name"]] = row["ricky_tag"]

        se_hits = []
        for name in se_names:
            if name not in actuals_map:
                continue
            p = actuals_map[name]
            tag = se_tags.get(name, "SE")
            sal = f"{p['salary']:,}" if p.get("salary") else "?"
            if p["delta"] > 0:
                se_hits.append({"name": name, "tag": tag, "sal": sal, **p})

        se_hits.sort(key=lambda x: x["delta"], reverse=True)
        # Add best SE hit if not redundant with board recap
        mentioned = {s.split()[0] for s in sentences}  # rough dedup
        for h in se_hits[:1]:
            if len(sentences) >= 3:
                break
            tmpl = _pick_template(_SE_PICK_HIT, h["name"], "se_hit")
            sentences.append(tmpl.format(
                name=h["name"], actual=h["actual"], proj=h["projected"],
                tag=h["tag"], sal=h["sal"],
            ))

    if sentences:
        return " ".join(sentences[:3])

    # ── Fallback: generic hit/miss recap ──
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

    if hits:
        h = hits[0]
        tmpl = _pick_template(_HIT_TEMPLATES, h["player_name"], "hit")
        sentences.append(
            tmpl.format(name=h["player_name"], actual=h["actual"], proj=h["projected"])
        )

    if misses:
        m = misses[0]
        tmpl = _pick_template(_MISS_TEMPLATES, m["player_name"], "miss")
        sentences.append(
            tmpl.format(name=m["player_name"], actual=m["actual"], proj=m["projected"])
        )

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

    # Salary Mismatch templates (25)
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
        "{name}, ${sal}. {proj:.0f} projected at {pts_1k:.1f} pts/$1K. The salary doesn't match the projection. I trust the projection.",
        "${sal} for {name}. {proj:.0f} projected. {pts_1k:.1f} per $1K. The field priced this by name. The model priced it by output.",
        "{name} at ${sal}. {proj:.0f} projected, {pts_1k:.1f} pts/$1K. Salary is a sticker price. Projection is the appraisal. They disagree.",
        "{name}: {proj:.0f} projected at ${sal}. {pts_1k:.1f} per $1K. Pricing inefficiency. The field doesn't run this math.",
        "${sal}. {name}. {proj:.0f} projected. {pts_1k:.1f} pts/$1K. The salary says one thing. The data says another. I'll take the data.",
        "{name} at ${sal}, {proj:.0f} projected. {pts_1k:.1f} per $1K. Underpriced relative to output. That's the definition of value.",
        "{name}: ${sal}, {proj:.0f} projected, {pts_1k:.1f} pts/$1K. The gap between price and production is the edge. This one has a gap.",
        "${sal} for {name} projecting {proj:.0f}. {pts_1k:.1f} pts/$1K. The market mispriced this. Markets do that.",
        "{name} at ${sal}. {proj:.0f} projected. {pts_1k:.1f} per $1K. This is the salary equivalent of finding money on the sidewalk.",
        "{name}: ${sal}. {proj:.0f} projected. {pts_1k:.1f} pts/$1K. Mismatch. The model sees it. The pricing doesn't reflect it.",
        "${sal} for {name}. {proj:.0f} FP. {pts_1k:.1f} per $1K. The price is lagging the projection. That's an edge.",
        "{name} at ${sal} with {proj:.0f} projected. {pts_1k:.1f} per $1K. When the price and the production diverge, bet on production.",
        "{name}, ${sal}, {pts_1k:.1f} pts/$1K. {proj:.0f} projected. The salary was set before the situation was clear. The situation is clear now.",
        "{name} at ${sal}. {proj:.0f} projected. {pts_1k:.1f} per $1K. The pricing algorithm missed this. The model didn't.",
        "{name}: ${sal}, {proj:.0f} projected. {pts_1k:.1f} pts per $1K. Below market rate. That's the whole play.",
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

    # Ownership Trap templates (25)
    _TRAP_TEMPLATES = [
        "{own:.0f}% of the field is on {name} this slate. {flags}. The scoreboard doesn't care how popular the pick was.",
        "{own:.0f}% of the field lined up behind {name}. {flags}. That's not conviction, that's a crowded trade.",
        "{name} at {own:.0f}% owned. {flags}. When everyone agrees, that's usually the wrong answer.",
        "{own:.0f}% on {name}. {flags}. Popularity and profitability have a negative correlation in GPPs.",
        "{name}: {own:.0f}% owned. {flags}. The field is betting this one the same way. That's the risk.",
        "{own:.0f}% ownership on {name}. {flags}. A crowded position with a thin margin. Bad math.",
        "{name} at {own:.0f}% owned this slate. {flags}. When the whole room agrees, check the exits.",
        "{own:.0f}% of the field lined up for {name}. {flags}. The consensus loved this one. Consensus has a losing record.",
        "{name}, {own:.0f}% owned. {flags}. Everyone's on the same side of this. That's not an edge — it's exposure.",
        "{own:.0f}% on {name}. {flags}. This is what a crowded trade looks like. Same mechanics every time.",
        "{name}: {own:.0f}% owned. {flags}. High traffic, low edge. The field is selling each other the same ticket.",
        "{own:.0f}% ownership. {name}. {flags}. The crowd is confident. The data is not. I know which I trust.",
        "{name} at {own:.0f}% owned. {flags}. This isn't conviction — it's a popularity contest. GPPs don't reward popular.",
        "{own:.0f}% on {name}. {flags}. The field loves certainty. The scoreboard loves variance. Pick a side.",
        "{name}: {own:.0f}%. {flags}. The consensus case sounds great until the flags start waving.",
        "{own:.0f}% of the field piled into {name}. {flags}. The risk isn't in the player. It's in the crowd.",
        "{name}, {own:.0f}% owned. {flags}. Same pick, same lineup, same result. The field never learns.",
        "{own:.0f}% ownership on {name}. {flags}. High ownership + red flags = negative expected leverage.",
        "{name} at {own:.0f}%. {flags}. The field went with the comfortable pick. Comfort has a cost.",
        "{own:.0f}% on {name} with flags. {flags}. The crowd sees the upside. The model sees the context.",
        "{name}: {own:.0f}% owned. {flags}. Crowded trade with structural risk. The math doesn't support the hype.",
        "{own:.0f}% on {name}. {flags}. Everyone's on this for the same reason. That reason doesn't account for the risk.",
        "{name} at {own:.0f}% owned. {flags}. The field followed the narrative. The data wrote a different one.",
        "{own:.0f}% ownership. {name}. {flags}. When the whole field is in, the downside is all yours.",
        "{name}: {own:.0f}%. {flags}. Popular and flagged. The field sees one. The model sees both.",
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

    # Contrarian Main templates (25)
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
        "{name} at {own:.0f}% owned. {reason}. Low traffic means high upside when it hits. The setup is there.",
        "{own:.0f}% on {name}. {reason}. The field scrolled past this. That's the opportunity.",
        "{name}: {own:.0f}% owned. {reason}. The quiet side of the board. Where the money actually is.",
        "{name} at {own:.0f}%. {reason}. Nobody's arguing about this name. That's exactly why I like it.",
        "{own:.0f}% ownership on {name}. {reason}. The field is loud about the wrong names. This is the right one.",
        "{name}, {own:.0f}% owned. {reason}. When the field ignores a good spot, that's not a mistake — it's a gift.",
        "{name}: {own:.0f}%. {reason}. The popular picks get the attention. The profitable ones get the results.",
        "{own:.0f}% on {name}. {reason}. The data supports it. The field hasn't noticed. Both can be true.",
        "{name} at {own:.0f}% owned. {reason}. Underowned with a structural edge. That's the sweet spot.",
        "{name}: {own:.0f}%. {reason}. The field didn't scroll this far down. Their loss.",
        "{own:.0f}% ownership. {name}. {reason}. The gap between what the field thinks and what the data says — that's leverage.",
        "{name} at {own:.0f}% owned. {reason}. Low owned for the wrong reasons. The right reasons say play.",
        "{name}, {own:.0f}% owned. {reason}. The model flagged this. The field didn't. I trust the model.",
        "{own:.0f}% on {name}. {reason}. Edge plus anonymity. The field can't fade what it doesn't see.",
        "{name}: {own:.0f}%. {reason}. The unpopular math. The profitable math. Same thing.",
    ]

    # Contrarian Fallback templates (20)
    _CONTRARIAN_FALLBACK = [
        "{name} at ${sal} with {proj:.0f} projected and only {own:.0f}% owned. The field is asleep.",
        "{name} at ${sal}, {proj:.0f} projected, {own:.0f}% owned. When nobody's looking, that's when you look harder.",
        "{name}: ${sal}, {proj:.0f} projected, {own:.0f}% owned. The pricing says value. The ownership says ignored. Both are useful.",
        "${sal} for {name}. {proj:.0f} projected at {own:.0f}% owned. Under the radar for no good reason.",
        "{name} at ${sal} projects to {proj:.0f} with {own:.0f}% ownership. The field walked right past this.",
        "{name}, ${sal}, {proj:.0f} projected. {own:.0f}% owned. Overlooked and underpriced. That's the sweet spot.",
        "{name}: {proj:.0f} projected, ${sal}, {own:.0f}% owned. The numbers work. The field didn't bother to check.",
        "{name} at ${sal}, {proj:.0f} projected, {own:.0f}% owned. Low traffic, good odds. I'll take it.",
        "${sal}. {name}. {proj:.0f} projected. {own:.0f}% owned. The field is busy elsewhere. I'm busy here.",
        "{name}: ${sal}, {proj:.0f} FP, {own:.0f}% owned. Cheap, productive, ignored. The trifecta.",
        "{name} at ${sal}. {proj:.0f} projected at {own:.0f}% owned. The field missed this. The model didn't.",
        "${sal} for {name}. {proj:.0f} projected. {own:.0f}% ownership. Below the radar. Above the line.",
        "{name}, ${sal}. {proj:.0f} projected, {own:.0f}% owned. Not the popular pick. That's the point.",
        "{name} at ${sal}, {proj:.0f} projected, {own:.0f}% owned. The quiet plays are often the loudest scorers.",
        "${sal}. {name}. {proj:.0f} FP. {own:.0f}% owned. The field spent their attention elsewhere.",
        "{name}: ${sal}. {proj:.0f} projected. {own:.0f}% owned. Underowned by negligence, not by design.",
        "{name} at ${sal} with {proj:.0f} projected. {own:.0f}% owned. The gap between price and attention is the edge.",
        "${sal} for {name}. {proj:.0f} projected. {own:.0f}% owned. Nobody's talking about this. That's not an accident.",
        "{name}: ${sal}, {proj:.0f} projected, {own:.0f}% owned. Ignored doesn't mean bad. Sometimes it means opportunity.",
        "{name} at ${sal}. {proj:.0f} projected, {own:.0f}% owned. The field doesn't do the math on this tier. I do.",
    ]

    # Lotto templates (20)
    _LOTTO_TEMPLATES = [
        "GPP lotto tickets: {names}. Low owned, high environment, mispriced. One of these hits and your lineup separates.",
        "GPP lotto shelf: {names}. The field walked right past them. One hit changes the whole slate.",
        "Lotto plays: {names}. Cheap, ignored, and sitting in the right game environment. That's the whole formula.",
        "Deep-roster lotto: {names}. Low ownership, right situation. The kind of names that show up in winning lineups.",
        "Lotto tier: {names}. The field doesn't want them. The math disagrees. That's where edges live.",
        "GPP lotto: {names}. Ignored by the field. Supported by the numbers. Pick your side.",
        "Lotto window: {names}. Low owned, high ceiling environment. These are the asymmetric plays.",
        "Lotto candidates: {names}. The field priced them out of conversation. The scoreboard might price them back in.",
        "GPP lotto shelf: {names}. Cheap names, right spots. The kind of plays that turn lineups into winners.",
        "Lotto picks: {names}. Low salary, low ownership, high upside. The field didn't bother. I did.",
        "Lotto tier: {names}. Nobody's rostering these for the right reasons. The math says they should.",
        "GPP lotto: {names}. The bottom of the salary sheet. The top of the value chart. That's the play.",
        "Deep roster: {names}. Low owned, right situation. The field doesn't win slates with popular names.",
        "Lotto plays: {names}. Cheap, overlooked, and in the right game. That's asymmetric risk.",
        "Lotto shelf: {names}. The field is busy fighting over the expensive names. These are the quiet winners.",
        "GPP lotto candidates: {names}. The field ignored the bottom of the board. The bottom of the board didn't ignore the math.",
        "Lotto window: {names}. Low ownership, high environment. The kind of names that don't show up in articles. They show up in winning lineups.",
        "Lotto tier: {names}. The field scrolled past these. The model flagged them. I published them.",
        "GPP lotto: {names}. Cheap shots at ceiling games. One connects and the slate flips.",
        "Deep-roster lotto picks: {names}. Under the radar, inside the right spots. That's the formula.",
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

    # Game Environment templates (20)
    _GAME_ENV_TEMPLATES = [
        "{t1}-{t2} has a {ou:.0f} total. Pace and points. This is where the math lives this slate.",
        "{t1}-{t2}: {ou:.0f} total. High-volume environment. Stack here or explain why not.",
        "{ou:.0f} total on {t1}-{t2}. The environment is doing the heavy lifting. Lean into it.",
        "{t1}-{t2} at {ou:.0f}. Pace means possessions. Possessions mean opportunity. Simple math.",
        "{t1}-{t2} game sits at {ou:.0f}. High totals correlate with high ceilings. The data is clear.",
        "{ou:.0f} total in {t1}-{t2}. Volume is the best predictor of upside. This game has it.",
        "{t1}-{t2}: {ou:.0f} total. When the pace is there, the points follow. The pace is there.",
        "{ou:.0f} on {t1}-{t2}. High game total. High opportunity. The math is uncomplicated.",
        "{t1}-{t2} at a {ou:.0f} total. This is the game to stack. The environment does the work.",
        "{ou:.0f} total. {t1}-{t2}. Pace, possessions, points. The correlation holds.",
        "{t1}-{t2}: {ou:.0f}. The highest total on the board. That's not a coincidence — it's an opportunity.",
        "{ou:.0f} total in {t1}-{t2}. More possessions, more fantasy points. The relationship hasn't changed.",
        "{t1}-{t2} game at {ou:.0f}. High-volume spot. The field stacks the popular game. This is the right one.",
        "{ou:.0f} on {t1}-{t2}. The game environment is doing the heavy lifting. Let it.",
        "{t1}-{t2}: {ou:.0f} total. Ceiling games start with ceiling environments. This one qualifies.",
        "{ou:.0f} total for {t1}-{t2}. The math says pace. The pace says points. Follow the chain.",
        "{t1}-{t2} at {ou:.0f}. Fast game, high total. The model loves this environment. So do I.",
        "{ou:.0f} on {t1}-{t2}. Game environment edge. The field may stack elsewhere. The data says here.",
        "{t1}-{t2}: {ou:.0f}. When the total is this high, you don't need to overthink the exposure.",
        "{ou:.0f} total. {t1}-{t2}. The highest-ceiling environment on the slate. Act accordingly.",
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
    """Generate 3-5 data-driven callouts about the current slate.

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

# Bust: high ownership + specific reasons (25)
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
    "{name} at {own:.0f}% owned. {reasons}. The field's favorite pick with the field's blind spot.",
    "{own:.0f}% ownership. {reasons}. The crowd is in. The flags are up. I'm out.",
    "{name}: {own:.0f}%. {reasons}. Popular + problematic = the fade of the night.",
    "{own:.0f}% on {name}. {reasons}. The field rushed to consensus. Consensus rushed to the wrong answer.",
    "{name} at {own:.0f}% owned. {reasons}. The whole room agreed. The data didn't get a vote.",
    "{own:.0f}% ownership on {name}. {reasons}. High conviction from the crowd, low support from the model. I'll take the model.",
    "{name}: {own:.0f}%. {reasons}. The popular pick with the unpopular matchup. The scoreboard doesn't care about popularity.",
    "{own:.0f}% of the field is on {name}. {reasons}. The louder the consensus, the quieter the edge.",
    "{name} at {own:.0f}% owned. {reasons}. When this many people agree, somebody missed something.",
    "{own:.0f}% ownership. {reasons}. The field is all-in. The context says fold.",
    "{name}: {own:.0f}% owned. {reasons}. Crowded + flagged. The field will learn this lesson again tonight.",
    "{own:.0f}% on {name}. {reasons}. The consensus tax is about to collect.",
    "{name} at {own:.0f}% owned. {reasons}. Popular and exposed. The field's worst combination.",
    "{own:.0f}% ownership on {name}. {reasons}. I've faded this setup a hundred times. The results haven't changed.",
    "{name}: {own:.0f}%. {reasons}. The crowd went all-in. The red flags went unread.",
]

# Bust: low ownership + specific reasons (20)
_BUST_EXPLANATIONS_LOW_OWN = [
    "{own:.0f}% owned, but the price tag is a trap. {reasons}. The salary says star, the situation says sit.",
    "{own:.0f}% owned — doesn't matter. {reasons}. Projections don't match reality here and the scoreboard won't either.",
    "Only {own:.0f}% of the field is on this, but the ones who are will regret it. {reasons}. Bad spot, bad price.",
    "{name} at {own:.0f}% owned. {reasons}. Low ownership doesn't make it a good bet when the situation is this bad.",
    "{own:.0f}% ownership. {reasons}. The price tag is doing the selling. The matchup isn't buying.",
    "{name}: {own:.0f}% owned. {reasons}. Not many people are on this. The few who are have the wrong read.",
    "Only {own:.0f}% on {name}. {reasons}. Low owned for a reason, but the reason isn't what the field thinks.",
    "{own:.0f}% owned. {reasons}. The salary looks right until you check the context. Then it doesn't.",
    "{name} at {own:.0f}% owned. {reasons}. Low traffic doesn't mean safe. The situation is the risk.",
    "{own:.0f}% ownership on {name}. {reasons}. The field mostly passed. The ones who didn't will wish they had.",
    "Only {own:.0f}% on {name}. {reasons}. Low ownership doesn't fix a broken situation.",
    "{name}: {own:.0f}%. {reasons}. Not popular. Not good. Two separate problems, same player.",
    "{own:.0f}% on {name}. {reasons}. The salary attracted the brave few. The context should have scared them off.",
    "{name} at {own:.0f}% owned. {reasons}. The field mostly avoided this. Credit where it's due. Now avoid it too.",
    "{own:.0f}% owned. {reasons}. Low ownership is a feature when the spot is bad. This spot is bad.",
    "Only {own:.0f}% on {name}. {reasons}. The crowd didn't bite. The crowd is right this time.",
    "{name}: {own:.0f}% owned. {reasons}. Cheap ownership doesn't make a bad matchup good.",
    "{own:.0f}% ownership. {reasons}. The price drew some interest. The situation should have killed it.",
    "{name} at {own:.0f}%. {reasons}. Not many are in. The ones who are should get out.",
    "Only {own:.0f}% ownership on {name}. {reasons}. The low ownership is correct. The context confirms it.",
]

# Bust: high ownership fallback — no specific reasons (15)
_BUST_FALLBACK_HIGH_OWN = [
    "{own:.0f}% owned and the numbers don't support it. Popularity isn't an edge. Never was.",
    "The field loves {name} at {own:.0f}% this slate. The data doesn't. I'll take the data.",
    "{name} at {own:.0f}% owned. The field is confident. The numbers are not. I know which one I trust.",
    "{own:.0f}% ownership on {name}. High conviction from the field, low conviction from the model. Mismatch.",
    "{name}: {own:.0f}% owned. The crowd says yes. The math says no. This isn't a close call.",
    "The field piled into {name} at {own:.0f}%. The data doesn't agree. I've seen enough to trust the data.",
    "{own:.0f}% on {name}. The field went all-in. The model didn't. I follow the model.",
    "{name} at {own:.0f}% owned. Popular and overpriced. The field's favorite combination.",
    "{own:.0f}% ownership. The field loves this. The model doesn't. Different methodologies, different conclusions.",
    "{name}: {own:.0f}%. The crowd consensus isn't backed by model consensus. That's the mismatch.",
    "The field loaded up on {name} at {own:.0f}%. The model says pass. The math is clear.",
    "{own:.0f}% on {name}. High traffic. Low conviction from the model. I'll take the quiet side.",
    "{name} at {own:.0f}% owned. The field's confidence exceeds the model's. When that happens, I trust the model.",
    "{own:.0f}% ownership on {name}. The crowd's conviction is strong. The data's conviction is stronger — in the other direction.",
    "{name}: {own:.0f}%. Popular pick. Unpopular model score. The model doesn't have feelings.",
]

# Bust: low ownership fallback — no specific reasons (15)
_BUST_FALLBACK_LOW_OWN = [
    "{own:.0f}% owned — low exposure, but still a bad bet. The salary is doing the selling, not the data.",
    "Only {own:.0f}% of the field bit on {name}, but the price tag still doesn't add up. Pass.",
    "{name} at {own:.0f}% owned. Low traffic doesn't mean value. Sometimes a quiet trade is just a bad trade.",
    "{own:.0f}% on {name}. The field mostly avoided this. They got it right.",
    "{name}: {own:.0f}% owned. Low ownership for a reason. The math confirmed it.",
    "Only {own:.0f}% on {name}. Even the field got this one right. The situation doesn't add up.",
    "{name} at {own:.0f}% owned. The field passed. The model agrees. Pass.",
    "{own:.0f}% ownership on {name}. Not popular. Not a good play either. Both things are true.",
    "Only {own:.0f}% on {name}. The field saw through it. So did the model.",
    "{name}: {own:.0f}% owned. Low ownership. Low model score. Both signals align.",
    "{own:.0f}% on {name}. The field didn't want it. The data doesn't either.",
    "{name} at {own:.0f}% owned. Quiet ownership, quiet projection. Nothing to chase here.",
    "Only {own:.0f}% on {name}. The field was right to pass. The model confirms it.",
    "{own:.0f}% ownership. {name}. Not enough field support, not enough model support. Easy pass.",
    "{name}: {own:.0f}%. The field and the model agree: pass. When both agree, listen.",
]

# Bust: fade fallback — from pre-classified fade candidates (15)
_BUST_FADE_FALLBACK = [
    "The model doesn't like it at {own:.0f}% owned. The field followed each other into this one. Don't follow them.",
    "{own:.0f}% owned and the model is fading it. The data says pass. I agree.",
    "Fade at {own:.0f}% owned. The field piled in. The model disagrees. I trust the model.",
    "{own:.0f}% ownership. The model flagged this as a fade. Not every popular pick is a good pick.",
    "The model says fade at {own:.0f}% owned. When the crowd and the model disagree, I side with the model.",
    "{own:.0f}% owned. Model fade. The field can have this one.",
    "Faded at {own:.0f}% owned. The model saw something the field didn't. I trust the model.",
    "{own:.0f}% ownership. Fade flag from the model. The data doesn't support the play.",
    "Model fade at {own:.0f}% owned. The field went in. I'm stepping aside.",
    "{own:.0f}% owned. The model flagged it. The model has a better track record than the crowd.",
    "Fade at {own:.0f}% ownership. The model's conviction is clear. So is mine.",
    "{own:.0f}% owned. Model says fade. I don't argue with the model.",
    "The model faded this at {own:.0f}% owned. When the data speaks, I listen.",
    "{own:.0f}% ownership. Fade. The model and I agree. The field can disagree all they want.",
    "Model fade. {own:.0f}% owned. The crowd went one way. The data went the other. I follow the data.",
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
