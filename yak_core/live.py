"""yak_core.live -- fetch live slate data from Tank01 RapidAPI."""
import os
import requests
import pandas as pd
from typing import Any, Dict, List, Optional
from .config import YAKOS_ROOT

_TANK01_HOST = "tank01-fantasy-stats.p.rapidapi.com"
# Known keys under which Tank01 getNBADFS may nest the player list in its body dict.
_TANK01_DFS_PLAYER_KEYS = ("DraftKings", "DK", "dk", "draftkings", "players", "playerList")


# Canonical status mapping from raw Tank01 values
_STATUS_MAP = {
    "ACTIVE": "Active",
    "": "Active",
    "OUT": "OUT",
    "IR": "IR",
    "INJURED RESERVE": "IR",
    "INJ": "IR",
    "SUSPENDED": "Suspended",
    "SUSP": "Suspended",
    "G-LEAGUE": "G-League",
    "G_LEAGUE": "G-League",
    "GLEAGUE": "G-League",
    "DND": "OUT",
    "O": "OUT",
    "QUESTIONABLE": "Questionable",
    "Q": "Questionable",
    "GTD": "GTD",
    "GAME TIME DECISION": "GTD",
    "DAY-TO-DAY": "GTD",
    "PROBABLE": "Probable",
    "P": "Probable",
}
