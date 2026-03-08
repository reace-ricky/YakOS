"""OCR extraction for RotoGrinders contest result screenshots.

Handles two known layouts:
  1. Compact view  – smaller image, player table fills most of the frame
  2. Leaderboard / View Lineup – wider image with sidebar leaderboard + Lineup Trends

Both layouts share:
  - Top bar: contest name, entry fee, entries count
  - Rank + green "Winning" badge near top-right
  - Player table: Pos | Team | Player | Salary | Field | Game | Points
  - Totals row at bottom: total salary, total field%, total points

Dependencies: pytesseract, opencv-python, Pillow (all available in the env).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pytesseract


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerRow:
    pos: str = ""
    team: str = ""
    player_name: str = ""
    salary: int = 0
    field_pct: float = 0.0
    game: str = ""
    points: float = 0.0


@dataclass
class ContestResult:
    """Extracted contest result from a screenshot."""
    contest_name: str = ""
    entry_fee: float = 0.0
    field_size: int = 0
    rank: int = 0
    winnings: float = 0.0
    total_salary: int = 0
    total_field_pct: float = 0.0
    total_points: float = 0.0
    players: list[PlayerRow] = field(default_factory=list)
    slate_date: str = ""        # e.g. "03/05/2026"
    slate_time: str = ""        # e.g. "7:00 pm ET"
    slate_type: str = ""        # e.g. "Main (6 games)", "Night (2 games)"
    # Lineup Trends (from RotoGrinders sidebar)
    low_owned_player: bool | None = None   # True=check, False=X, None=unknown
    team_stack: bool | None = None
    game_stack: bool | None = None
    team_stacks_count: int = 0              # from profile info card (0 = unknown)
    game_stacks_count: int = 0
    raw_text: str = ""          # full OCR dump for debugging
    confidence: float = 0.0     # 0-1 how confident we are in the extraction

    @property
    def finish_pct(self) -> float:
        if self.field_size > 0 and self.rank > 0:
            return round((self.rank / self.field_size) * 100, 2)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "contest_name": self.contest_name,
            "entry_fee": self.entry_fee,
            "field_size": self.field_size,
            "rank": self.rank,
            "winnings": self.winnings,
            "total_salary": self.total_salary,
            "total_field_pct": self.total_field_pct,
            "total_points": self.total_points,
            "finish_pct": self.finish_pct,
            "slate_date": self.slate_date,
            "slate_time": self.slate_time,
            "slate_type": self.slate_type,
            "low_owned_player": self.low_owned_player,
            "team_stack": self.team_stack,
            "game_stack": self.game_stack,
            "team_stacks_count": self.team_stacks_count,
            "game_stacks_count": self.game_stacks_count,
            "num_players": len(self.players),
            "players": [
                {
                    "pos": p.pos, "player_name": p.player_name,
                    "salary": p.salary, "field_pct": p.field_pct,
                    "points": p.points, "team": p.team, "game": p.game,
                }
                for p in self.players
            ],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_POS_TOKENS = {"PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"}
_POS_ALIASES = {
    "pg": "PG", "sc": "SG", "sg": "SG", "sf": "SF", "pf": "PF",
    "c": "C", "g": "G", "f": "F", "util": "UTIL", "ut": "UTIL",
    "ul": "UTIL", "utl": "UTIL", "uti": "UTIL",
}


def _extract_lineup_trends(img: np.ndarray) -> dict[str, bool | None]:
    """Detect Lineup Trends check/X icons via color detection.

    The RotoGrinders sidebar shows three trend indicators:
      - Low-owned Player  (top,    y_pct ~0.30-0.36)
      - Team Stack         (middle, y_pct ~0.48-0.54)
      - Game Stack         (bottom, y_pct ~0.62-0.68)

    Green icon = True (check), Red icon = False (X).
    Returns dict with keys 'low_owned_player', 'team_stack', 'game_stack'.
    """
    h, w = img.shape[:2]

    # Crop the trends sidebar: ~27-48% width, 25-75% height
    crop = img[int(h * 0.25): int(h * 0.75), int(w * 0.27): int(w * 0.48)]
    ch = crop.shape[0]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Green check: hue 35-90, sat > 50, val > 50
    green_mask = cv2.inRange(hsv, (35, 50, 50), (90, 255, 255))
    # Red X: hue 0-10 or 170-180, sat > 50, val > 50
    red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) | \
               cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))

    def _find_icons(mask: np.ndarray) -> list[float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        y_pcts = []
        for c in contours:
            _, y, bw, bh = cv2.boundingRect(c)
            if 5 < bw < 40 and 5 < bh < 40:
                y_pcts.append((y + bh / 2) / ch)
        return y_pcts

    green_ys = _find_icons(green_mask)
    red_ys = _find_icons(red_mask)

    # Map y_pct bands to trend labels
    bands = [
        ("low_owned_player", 0.26, 0.42),
        ("team_stack", 0.42, 0.58),
        ("game_stack", 0.58, 0.74),
    ]

    result: dict[str, bool | None] = {
        "low_owned_player": None,
        "team_stack": None,
        "game_stack": None,
    }

    for label, lo, hi in bands:
        if any(lo <= y <= hi for y in green_ys):
            result[label] = True
        elif any(lo <= y <= hi for y in red_ys):
            result[label] = False

    return result


def _extract_info_card(img: np.ndarray) -> dict:
    """Extract data from the profile/Lineups view info card.

    This card appears in the RotoGrinders user profile layout and contains:
      - Rank (blue circle badge)
      - Winning (green pill)
      - Duplicates (blue circle)
      - Team Stacks count (blue circle)
      - Game Stacks count (blue circle)

    Returns dict with optional keys: winning, team_stacks, game_stacks.
    """
    h, w = img.shape[:2]
    # Info card: ~27-46% width, ~20-70% height
    crop = img[int(h * 0.20): int(h * 0.70), int(w * 0.27): int(w * 0.46)]
    scaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(scaled)

    result: dict = {}

    # Look for dollar amounts in the card text (Winning value)
    money_matches = re.findall(r"\$(\d[\d,]*)", text)
    for m in money_matches:
        val = float(m.replace(",", ""))
        if val >= 100:  # Reasonable winning amount
            result["winning"] = val
            break

    # Look for stack counts via blue circle badge OCR
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ch = crop.shape[0]

    blue_badges = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if 10 < bw < 40 and 10 < bh < 40:
            badge = crop[max(0, y - 3): y + bh + 3, max(0, x - 3): x + bw + 3]
            badge_s = cv2.resize(badge, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            t = pytesseract.image_to_string(
                badge_s, config="--psm 10 -c tessedit_char_whitelist=0123456789"
            ).strip()
            y_pct = y / ch
            if t.isdigit():
                blue_badges.append((y_pct, int(t)))

    # Map by vertical position: Team Stacks ~0.55-0.70, Game Stacks ~0.70-0.85
    for y_pct, val in blue_badges:
        if 0.45 <= y_pct <= 0.65:
            result["team_stacks"] = val
        elif 0.65 <= y_pct <= 0.85:
            result["game_stacks"] = val

    return result


def _parse_money(s: str) -> float:
    """'$50,000' -> 50000.0"""
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_green_badge(img: np.ndarray) -> str:
    """Find the green 'Winning: $X' badge via HSV color filtering."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 60), (90, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    badges = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > 500 and bw > 50 and 15 < bh < 80:
            badges.append((x, y, bw, bh, bw * bh))
    badges.sort(key=lambda b: b[4], reverse=True)

    for x, y, bw, bh, _ in badges[:3]:
        pad = 5
        crop = img[max(0, y - pad): y + bh + pad, max(0, x - pad): x + bw + pad]
        scaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(scaled, config="--psm 7").strip()
        if "winning" in text.lower() or "$" in text:
            return text
    return ""


def _extract_rank(img: np.ndarray) -> int:
    """Extract rank from the right-half of the image, upper region."""
    h, w = img.shape[:2]
    for top_pct in (0.16, 0.18, 0.14, 0.20):
        crop = img[int(h * top_pct): int(h * (top_pct + 0.06)), int(w * 0.4):]
        scaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        text = pytesseract.image_to_string(inv, config="--psm 6").strip()
        m = re.search(r"Rank[:\s]*(\d+)", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    # Fallback: search full text
    return 0


def _parse_header(text: str, result: ContestResult) -> None:
    """Parse contest header line for name, entry fee, field size, slate info."""
    # Contest: "$8 NBA $200K Friday 8's [$50K to 1st] (29411 entries)"
    m = re.search(
        r"\$(\d+)[\.\s]*(.+?)\((\d+)\s*entr",
        text, re.IGNORECASE,
    )
    if m:
        result.entry_fee = float(m.group(1))
        result.contest_name = f"${m.group(1)} {m.group(2).strip()}"
        result.field_size = int(m.group(3))
    else:
        # Try without entry fee
        m2 = re.search(r"\$0[\.\s]*(.+?)\((\d+)\s*entr", text, re.IGNORECASE)
        if m2:
            result.entry_fee = 0.0
            result.contest_name = f"$0 {m2.group(1).strip()}"
            result.field_size = int(m2.group(2))

    # Slate date: "03/05/2026" or similar
    date_m = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    if date_m:
        result.slate_date = date_m.group(1)

    # Slate time: "7:00 pm ET" or "10:30 pm ET"
    time_m = re.search(r"(\d{1,2}:\d{2}\s*[ap]m\s*ET)", text, re.IGNORECASE)
    if time_m:
        result.slate_time = time_m.group(1)

    # Slate type: "Main (6 games)" or "Night (2 games)"
    type_m = re.search(r"(Main|Night|Early|Late)\s*\((\d+)\s*games?\)", text, re.IGNORECASE)
    if type_m:
        result.slate_type = f"{type_m.group(1)} ({type_m.group(2)} games)"


def _parse_player_line(line: str) -> Optional[PlayerRow]:
    """Try to extract a player row from an OCR line.

    Lines look roughly like:
        PG  <logo noise>  De'Aaron Fox  $6,800  6.8%  LAC 112 @ SAS 116  36.50
    """
    # Normalize common OCR noise
    line = line.replace("«", "").replace("—", " ").replace("–", " ").replace("=", " ")
    tokens = line.split()
    if len(tokens) < 4:
        return None

    # Find position token
    pos = ""
    pos_idx = -1
    for i, tok in enumerate(tokens[:3]):
        normalized = tok.lower().rstrip(".,;:")
        if normalized in _POS_ALIASES:
            pos = _POS_ALIASES[normalized]
            pos_idx = i
            break
        if tok.upper() in _POS_TOKENS:
            pos = tok.upper()
            pos_idx = i
            break

    if not pos:
        return None

    # Find salary ($X,XXX)
    salary = 0
    salary_idx = -1
    for i, tok in enumerate(tokens):
        sm = re.match(r"\$?([\d,]+)$", tok.replace("$", "$"))
        if tok.startswith("$") and sm:
            val = int(sm.group(1).replace(",", ""))
            if 2000 <= val <= 20000:  # DK salary range
                salary = val
                salary_idx = i
                break

    if salary_idx < 0:
        return None

    # Find points (last float-like token)
    points = 0.0
    points_idx = -1
    for i in range(len(tokens) - 1, max(salary_idx, 0), -1):
        pm = re.match(r"(\d{1,3}\.\d{2})$", tokens[i])
        if pm:
            points = float(pm.group(1))
            points_idx = i
            break

    # Find field% (X.X% or XX.X%)
    field_pct = 0.0
    for i, tok in enumerate(tokens):
        fm = re.match(r"(\d{1,3}\.\d)%?$", tok.rstrip("%"))
        if fm and i > salary_idx:
            field_pct = float(fm.group(1))
            break

    # Player name: tokens between pos and salary (skip logo noise)
    name_tokens = []
    for tok in tokens[pos_idx + 1: salary_idx]:
        # Skip single-char noise, emoji artifacts, symbol-only tokens
        cleaned = re.sub(r"[^A-Za-z'\-\.]", "", tok)
        if len(cleaned) >= 2:
            name_tokens.append(cleaned)

    player_name = " ".join(name_tokens) if name_tokens else ""

    # Game string: tokens between field% and points that contain "@"
    game = ""
    if salary_idx >= 0 and points_idx >= 0:
        game_tokens = tokens[salary_idx + 1: points_idx]
        game_parts = [t for t in game_tokens if "@" in t or re.match(r"[A-Z]{2,3}", t) or re.match(r"\d{2,3}", t)]
        if game_parts:
            game = " ".join(game_parts)

    row = PlayerRow(
        pos=pos,
        player_name=player_name,
        salary=salary,
        field_pct=field_pct,
        points=points,
        game=game,
    )
    return row


def _parse_totals(text: str, result: ContestResult) -> None:
    """Extract totals row: $49,700  71.8%  345.25"""
    # Look for a line with total salary ($4X,XXX or $5X,XXX) and total points (XXX.XX)
    for line in text.split("\n"):
        sal_m = re.search(r"\$([\d]{2},\d{3})", line)
        pts_m = re.search(r"(\d{2,3}\.\d{2})\s*$", line.strip())
        fld_m = re.search(r"(\d{1,3}\.\d)%", line)
        if sal_m and pts_m:
            result.total_salary = int(sal_m.group(1).replace(",", ""))
            result.total_points = float(pts_m.group(1))
            if fld_m:
                result.total_field_pct = float(fld_m.group(1))
            return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_columnar_players(img: np.ndarray) -> tuple[list[PlayerRow], int, float, float]:
    """Extract players from screenshots where OCR reads column-by-column.

    Tries multiple crop regions to handle both narrow (compact) and wide
    (leaderboard) layouts.  Returns (players, total_salary, total_field_pct,
    total_points).
    """
    h, w = img.shape[:2]

    # Try multiple left-crop positions: 0% for compact, ~42% for wide leaderboard
    best_text = ""
    best_sections: dict[str, list[str]] = {}
    col_headers = {"Pos", "Team", "Player", "Salary", "Field", "Game", "Points"}

    for left_pct in (0.42, 0.0, 0.15, 0.30):
        scale = 1.5 if left_pct > 0.1 else 2.0
        crop = img[int(h * 0.25): int(h * 0.95), int(w * left_pct):]
        scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(scaled)

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        sections: dict[str, list[str]] = {}
        current_section = None
        for line in lines:
            if line in col_headers:
                current_section = line
                sections[current_section] = []
            elif current_section is not None:
                sections[current_section] = sections.get(current_section, [])
                sections[current_section].append(line)

        if "Player" in sections and "Salary" in sections:
            player_vals = [v for v in sections["Player"] if len(v) > 2]
            if len(player_vals) >= 6:
                best_text = text
                best_sections = sections
                break

    if not best_sections:
        return [], 0, 0.0, 0.0

    sections = best_sections

    # Parse each column, limiting to 8 entries (DK classic lineup size)
    pos_list = []
    for v in sections.get("Pos", []):
        tok = v.strip().upper()
        if tok in _POS_TOKENS or tok in {k.upper() for k in _POS_ALIASES}:
            pos_list.append(_POS_ALIASES.get(tok.lower(), tok))
    # Cap at 8
    pos_list = pos_list[:8]

    player_names = [v for v in sections.get("Player", []) if len(v) > 2][:8]

    salaries_raw = sections.get("Salary", [])
    salaries = []
    total_salary = 0
    for v in salaries_raw:
        m = re.search(r"\$?([\d,]+)", v)
        if m:
            val = int(m.group(1).replace(",", ""))
            if 2000 <= val <= 20000:
                salaries.append(val)
            elif 40000 <= val <= 60000:
                total_salary = val
    salaries = salaries[:8]

    fields_raw = sections.get("Field", [])
    fields = []
    total_field_pct = 0.0
    for v in fields_raw:
        m = re.search(r"([\d.]+)%?", v)
        if m:
            val = float(m.group(1))
            if val < 100:
                fields.append(val)
            else:
                total_field_pct = val
    fields = fields[:8]

    games_raw = sections.get("Game", [])
    games = [v for v in games_raw if "@" in v][:8]

    points_raw = sections.get("Points", [])
    points = []
    total_points = 0.0
    for v in points_raw:
        m = re.search(r"([\d.]+)", v.replace(",", ""))
        if m:
            val = float(m.group(1))
            if val < 200:
                points.append(val)
            elif val >= 200:
                total_points = val
    points = points[:8]

    # Zip into player rows
    n = len(player_names)
    players = []
    for i in range(n):
        p = PlayerRow(
            pos=pos_list[i] if i < len(pos_list) else "",
            player_name=player_names[i],
            salary=salaries[i] if i < len(salaries) else 0,
            field_pct=fields[i] if i < len(fields) else 0.0,
            game=games[i] if i < len(games) else "",
            points=points[i] if i < len(points) else 0.0,
        )
        players.append(p)

    return players, total_salary, total_field_pct, total_points


def extract_contest_result(image_path: str) -> ContestResult:
    """Extract contest result data from a RotoGrinders screenshot.

    Parameters
    ----------
    image_path : str
        Path to the screenshot image file.

    Returns
    -------
    ContestResult
        Extracted data with a confidence score (0-1).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    result = ContestResult()

    # 1. Full-page OCR
    full_text = pytesseract.image_to_string(img)
    result.raw_text = full_text

    # 2. Header parsing
    _parse_header(full_text, result)

    # 3. Rank (dedicated crop + inversion)
    result.rank = _extract_rank(img)
    # Fallback: search full text
    if result.rank == 0:
        rm = re.search(r"Rank[:\s]*(\d+)", full_text, re.IGNORECASE)
        if rm:
            result.rank = int(rm.group(1))

    # 4. Winnings (green badge detection)
    badge_text = _extract_green_badge(img)
    wm = re.search(r"\$?([\d,]+)", badge_text)
    if wm:
        result.winnings = _parse_money(wm.group(0))

    # 5a. Lineup Trends (check/X icon detection)
    trends = _extract_lineup_trends(img)
    result.low_owned_player = trends["low_owned_player"]
    result.team_stack = trends["team_stack"]
    result.game_stack = trends["game_stack"]

    # 5b. Info card (profile/Lineups view) — fills winnings + stack counts
    info_card = _extract_info_card(img)
    if result.winnings == 0 and "winning" in info_card:
        result.winnings = info_card["winning"]
    if info_card.get("team_stacks"):
        result.team_stacks_count = info_card["team_stacks"]
        if result.team_stack is None:
            result.team_stack = info_card["team_stacks"] > 0
    if info_card.get("game_stacks"):
        result.game_stacks_count = info_card["game_stacks"]
        if result.game_stack is None:
            result.game_stack = info_card["game_stacks"] > 0

    # 6. Columnar extraction (primary — works for both narrow and wide layouts)
    col_players, col_sal, col_fld, col_pts = _extract_columnar_players(img)
    if len(col_players) >= 6:
        result.players = col_players
        if col_sal > 0:
            result.total_salary = col_sal
        if col_fld > 0:
            result.total_field_pct = col_fld
        if col_pts > 0:
            result.total_points = col_pts
    else:
        # Fallback A: row-based parsing on right-half crop (handles profile/
        # Lineups view where info card overlaps the left side)
        h, w = img.shape[:2]
        for left_pct in (0.45, 0.40, 0.35, 0.0):
            crop = img[int(h * 0.20): int(h * 0.95), int(w * left_pct):]
            scaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            crop_text = pytesseract.image_to_string(scaled)
            crop_players: list[PlayerRow] = []
            for line in crop_text.split("\n"):
                player = _parse_player_line(line)
                if player and player.salary > 0:
                    crop_players.append(player)
            if len(crop_players) >= 6:
                result.players = crop_players
                _parse_totals(crop_text, result)
                break
        else:
            # Fallback B: row-based parsing on full-page text
            for line in full_text.split("\n"):
                player = _parse_player_line(line)
                if player and player.salary > 0:
                    result.players.append(player)

    # 7. Totals (fill in anything not yet found)
    if result.total_points == 0 or result.total_salary == 0:
        _parse_totals(full_text, result)

    # 9. Confidence score
    checks = [
        bool(result.contest_name),
        result.field_size > 0,
        result.rank > 0,
        result.winnings > 0,
        result.total_points > 0,
        len(result.players) >= 6,
    ]
    result.confidence = sum(checks) / len(checks)

    return result
