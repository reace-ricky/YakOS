"""yak_core.name_utils -- Player name normalization for cross-source matching."""

from __future__ import annotations

import re
import unicodedata


def normalize_player_name(name: str) -> str:
    """Normalize a player name for fuzzy matching across data sources.

    Steps:
      1. NFD decompose + strip accent marks (Mn category)
      2. Lowercase, strip outer whitespace
      3. Remove periods, apostrophes, hyphens
      4. Collapse multiple spaces
      5. Normalize common suffix variants (Jr, Sr, II, III, IV)
    """
    # NFD decompose → strip accents
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Lowercase
    name = name.lower().strip()
    # Remove periods, apostrophes, hyphens
    name = re.sub(r"[.\-']", "", name)
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    # Normalize common suffixes (strip trailing dot, ensure single space before)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)\.?$", r" \1", name)
    return name
