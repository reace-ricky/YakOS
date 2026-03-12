"""yak_core.weather -- Fetch tournament weather for PGA DFS analysis.

Uses wttr.in (free, no API key) to get multi-day forecasts for the
tournament location. Falls back gracefully if unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests


def fetch_tournament_weather(lat: float, lon: float, days: int = 4) -> Dict[str, Any]:
    """Fetch weather forecast for tournament location.

    Returns dict with:
      current  : {temp_f, wind_mph, wind_dir, humidity, conditions}
      daily    : [{date, high_f, low_f, wind_mph, wind_dir, precip_chance, conditions}]
      wind_summary   : str
      scoring_impact : str
    """
    try:
        url = f"https://wttr.in/{lat},{lon}?format=j1"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "YakOS/1.0"})
        resp.raise_for_status()
        data = resp.json()

        # Current conditions
        cur = data.get("current_condition", [{}])[0]
        current = {
            "temp_f": int(cur.get("temp_F", 0)),
            "wind_mph": int(cur.get("windspeedMiles", 0)),
            "wind_dir": cur.get("winddir16Point", ""),
            "humidity": int(cur.get("humidity", 0)),
            "conditions": cur.get("weatherDesc", [{}])[0].get("value", ""),
        }

        # Daily forecasts
        daily: List[Dict[str, Any]] = []
        for day in data.get("weather", [])[:days]:
            hourly = day.get("hourly", [])
            avg_wind = sum(int(h.get("windspeedMiles", 0)) for h in hourly) / max(len(hourly), 1)
            max_precip = max((int(h.get("chanceofrain", 0)) for h in hourly), default=0)
            dirs = [h.get("winddir16Point", "") for h in hourly if h.get("winddir16Point")]
            # Pick midday conditions (index 4 of 8 hourly = ~noon)
            midday = hourly[4] if len(hourly) > 4 else (hourly[0] if hourly else {})
            daily.append({
                "date": day.get("date", ""),
                "high_f": int(day.get("maxtempF", 0)),
                "low_f": int(day.get("mintempF", 0)),
                "wind_mph": round(avg_wind),
                "wind_dir": dirs[len(dirs) // 2] if dirs else "",
                "precip_chance": max_precip,
                "conditions": midday.get("weatherDesc", [{}])[0].get("value", "")
                if isinstance(midday.get("weatherDesc"), list) else "",
            })

        return {
            "current": current,
            "daily": daily,
            "wind_summary": _wind_summary(daily),
            "scoring_impact": _scoring_impact(daily),
        }
    except Exception as e:
        print(f"[weather] Failed to fetch: {e}")
        return {"current": {}, "daily": [], "wind_summary": "", "scoring_impact": ""}


def _wind_summary(daily: List[Dict]) -> str:
    if not daily:
        return ""
    winds = [d["wind_mph"] for d in daily]
    mn, mx = min(winds), max(winds)
    dirs = [d["wind_dir"] for d in daily if d["wind_dir"]]
    primary = max(set(dirs), key=dirs.count) if dirs else ""
    avg = sum(winds) / len(winds)
    intensity = "Light" if avg < 8 else ("Moderate" if avg < 15 else "Strong")
    return f"{intensity} {mn}-{mx} mph {primary} winds expected"


def _scoring_impact(daily: List[Dict]) -> str:
    if not daily:
        return ""
    avg_wind = sum(d["wind_mph"] for d in daily) / len(daily)
    max_precip = max((d["precip_chance"] for d in daily), default=0)
    parts = []
    if avg_wind >= 15:
        parts.append("Strong winds favor ball-strikers and low-flight players")
    elif avg_wind >= 10:
        parts.append("Moderate wind — accuracy off the tee matters more than distance")
    else:
        parts.append("Calm conditions — scoring should be low, putting premium increases")
    if max_precip >= 50:
        parts.append("Rain likely — soft conditions favor aggressive approach play")
    elif max_precip >= 25:
        parts.append("Chance of rain — monitor for softer greens")
    return ". ".join(parts)
