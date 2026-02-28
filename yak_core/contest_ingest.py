"""
yak_core/contest_ingest.py
Parse DraftKings contest CSVs and merge real ownership
into the optimizer player pool.
"""

import os, re, glob
import pandas as pd
from typing import Dict, List


def load_dk_player_pool(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "name" in df.columns and "player_id" in df.columns:
        df = df.rename(columns={"player_id": "dk_id"})
    elif "name" in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "dk_id"})
    elif "name" in df.columns:
        df["dk_id"] = range(len(df))
    else:
        raise ValueError(f"Cannot parse DK pool CSV: got {list(df.columns)}")
    if "salary" not in df.columns:
        for c in df.columns:
            if "sal" in c.lower():
                df = df.rename(columns={c: "salary"})
                break
    for col in ["fpts", "pown", "salary"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    print(f"[contest_ingest] Loaded DK pool: {len(df)} players from {os.path.basename(path)}")
    return df


def _parse_dk_lineup_string(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    out = []
    for chunk in s.split("|"):
        chunk = chunk.strip()
        chunk = re.sub(r"^[A-Z/]+\s+", "", chunk)
        chunk = re.sub(r"\s*\(\$[\d,]+\)", "", chunk).strip()
        if chunk:
            out.append(chunk)
    return out


def load_dk_contest_results(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    lineup_col = None
    for c in raw.columns:
        if c.lower() in ("lineup", "lineups"):
            lineup_col = c
            break
    if lineup_col and "Points" in raw.columns:
        entries = raw.dropna(subset=[lineup_col])
        n = len(entries)
        if n == 0:
            return pd.DataFrame(columns=["name", "ownership_pct"])
        pc: Dict[str, int] = {}
        for _, row in entries.iterrows():
            for p in _parse_dk_lineup_string(row[lineup_col]):
                pc[p] = pc.get(p, 0) + 1
        r = pd.DataFrame(
            [{"name": k, "ownership_pct": round(v / n * 100, 2)} for k, v in pc.items()]
        ).sort_values("ownership_pct", ascending=False).reset_index(drop=True)
        print(f"[contest_ingest] Parsed contest: {len(r)} players from {n} entries")
        return r
    nc = oc = None
    for c in raw.columns:
        cl = c.lower()
        if cl in ("name", "player", "player_name"):
            nc = c
        elif "own" in cl:
            oc = c
    if nc and oc:
        r = raw[[nc, oc]].copy()
        r.columns = ["name", "ownership_pct"]
        r["ownership_pct"] = pd.to_numeric(
            r["ownership_pct"].astype(str).str.replace("%", ""),
            errors="coerce",
        ).fillna(0)
        print(f"[contest_ingest] Loaded ownership CSV: {len(r)} players")
        return r
    raise ValueError(f"Cannot parse {os.path.basename(path)}: cols={list(raw.columns)}")


def _fuzzy_name_map(pool_names, dk_names):
    def _n(s):
        s = s.lower().strip()
        s = re.sub(r"[.'`]", "", s)
        s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
        return re.sub(r"\s+", " ", s)
    dk_lookup = {_n(n): n for n in dk_names}
    return {pn: dk_lookup[_n(pn)] for pn in pool_names if _n(pn) in dk_lookup}


def merge_ownership_into_pool(pool_df, dk_ownership_df, pool_name_col="player"):
    out = pool_df.copy()
    if "ownership_pct" in dk_ownership_df.columns:
        own_col, name_col = "ownership_pct", "name"
    elif "pown" in dk_ownership_df.columns:
        own_col = "pown"
        name_col = "name" if "name" in dk_ownership_df.columns else dk_ownership_df.columns[0]
    else:
        own_col = None
        for c in dk_ownership_df.columns:
            if "own" in c.lower():
                own_col = c
                break
        name_col = "name" if "name" in dk_ownership_df.columns else dk_ownership_df.columns[0]
    if own_col is None:
        print("[contest_ingest] Warning: no ownership column found")
        out["dk_ownership"] = 0.0
        return out
    nm = _fuzzy_name_map(out[pool_name_col].tolist(), dk_ownership_df[name_col].tolist())
    lk = dict(zip(dk_ownership_df[name_col], dk_ownership_df[own_col]))
    out["dk_ownership"] = out[pool_name_col].map(
        lambda pn: lk.get(nm.get(pn, pn), 0.0)
    )
    m = (out["dk_ownership"] > 0).sum()
    print(f"[contest_ingest] Merged: {m}/{len(out)} matched ({m/len(out)*100:.0f}%)")
    return out


def scan_dk_csvs(directory: str) -> Dict[str, List[str]]:
    result = {"player_pool": [], "contest_results": [], "unknown": []}
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return result
    for p in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        try:
            h = pd.read_csv(p, nrows=2)
            cols = {c.lower().strip() for c in h.columns}
            if "player_id" in cols and "name" in cols:
                result["player_pool"].append(p)
            elif "rank" in cols and ("lineup" in cols or "lineups" in cols):
                result["contest_results"].append(p)
            elif "entryid" in cols or "entry_id" in cols:
                result["contest_results"].append(p)
            elif any("own" in c for c in cols) and any(c in cols for c in ("name", "player")):
                result["contest_results"].append(p)
            else:
                result["unknown"].append(p)
        except Exception:
            result["unknown"].append(p)
    pp = len(result["player_pool"])
    cr = len(result["contest_results"])
    uk = len(result["unknown"])
    print(f"[contest_ingest] Scanned {directory}: {pp} pool, {cr} contest, {uk} unknown ({pp+cr+uk} total)")
    return result
