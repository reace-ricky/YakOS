"""Shared contest-type constants used by Optimizer, The Lab, and Sim Lab.

Two-level taxonomy:
  1. Game Style  — Classic | Showdown
  2. Contest Type — depends on Game Style

Internal profile keys follow the pattern ``classic_gpp_main``, ``showdown_gpp``,
etc. and are used consistently across persistence, ownership variance, and
named-profile look-ups.
"""

from __future__ import annotations

# ── NBA Game Styles (top-level dropdown) ──────────────────────────
NBA_GAME_STYLES: list[str] = ["Classic", "Showdown"]

# ── NBA Contest Types per Game Style (second dropdown) ───────────
NBA_CONTEST_TYPES_BY_STYLE: dict[str, list[str]] = {
    "Classic": ["GPP Main", "GPP 20-Max", "GPP Single Entry", "Cash"],
    "Showdown": ["Showdown GPP", "Showdown Cash"],
}

# ── (game_style, contest_type) → internal profile key ────────────
CONTEST_PROFILE_KEY_MAP: dict[tuple[str, str], str] = {
    ("Classic", "GPP Main"):        "classic_gpp_main",
    ("Classic", "GPP 20-Max"):      "classic_gpp_20max",
    ("Classic", "GPP Single Entry"): "classic_gpp_se",
    ("Classic", "Cash"):            "classic_cash",
    ("Showdown", "Showdown GPP"):   "showdown_gpp",
    ("Showdown", "Showdown Cash"):  "showdown_cash",
}

# ── profile_key → CONTEST_PRESETS key (from yak_core/config.py) ──
PROFILE_KEY_TO_PRESET: dict[str, str] = {
    "classic_gpp_main":  "GPP Main",
    "classic_gpp_20max": "GPP Early",
    "classic_gpp_se":    "GPP SE",
    "classic_cash":      "Cash Main",
    "showdown_gpp":      "Showdown",
    "showdown_cash":     "Cash Game",
}

# ── profile_key → NAMED_PROFILES key (or None) ───────────────────
PROFILE_KEY_TO_NAMED: dict[str, str | None] = {
    "classic_gpp_main":  "GPP_MAIN_V1",
    "classic_gpp_20max": "GPP_20MAX_V1",
    "classic_gpp_se":    "GPP_SE_V1",
    "classic_cash":      "CASH_MAIN_V1",
    "showdown_gpp":      "SD_GPP_V1",
    "showdown_cash":     "SD_CASH_V1",
}

# All valid NBA profile keys (flat list for validation / persistence)
NBA_PROFILE_KEYS: list[str] = list(CONTEST_PROFILE_KEY_MAP.values())

# ── PGA contest types (unchanged) ────────────────────────────────
PGA_CONTEST_TYPES: list[str] = [
    "PGA · Tournament GPP",
    "PGA · Cash (50-50)",
    "PGA · Showdown GPP",
]

# PGA display name → CONTEST_PRESETS key
PGA_DISPLAY_TO_PRESET: dict[str, str] = {
    "PGA · Tournament GPP": "PGA GPP",
    "PGA · Cash (50-50)":   "PGA Cash",
    "PGA · Showdown GPP":   "PGA Showdown",
}
