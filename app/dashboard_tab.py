"""Tab 4: Dashboard – Smash / Bust Explorer.

Interactive scatter explorer showing historical YakOS signal outcomes
(caught vs missed smashes/busts) plus live slate players.  Replaces
the old summary-chart dashboard.  Maintenance tools preserved at bottom.
"""
from __future__ import annotations

import glob as _glob
import html as _html
import json
import os
import time
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Data Loading ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    """Load a JSON file, returning empty dict on failure."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=300)
def _load_explorer_data(sport: str) -> list:
    """Load historical archive + live slate into a flat list of dicts."""
    sport_lower = sport.lower()
    archive_dir = REPO_ROOT / "data" / "slate_archive"
    records: list = []

    # ── Historical archives ──────────────────────────────────────────────
    pattern = "*_gpp_main.parquet"
    if sport_lower == "pga":
        pattern = "*_pga_gpp.parquet"

    parquets = sorted(archive_dir.glob(pattern)) if archive_dir.exists() else []
    for pq in parquets:
        try:
            df = pd.read_parquet(pq)
        except Exception:
            continue

        # Must have actuals
        if "actual_fp" not in df.columns:
            continue
        df = df[df["actual_fp"].notna()].copy()
        if df.empty:
            continue

        for _, row in df.iterrows():
            proj = float(row.get("proj", 0) or 0)
            actual = float(row.get("actual_fp", 0) or 0)
            diff = round(actual - proj, 2)

            # Category
            if diff <= -12:
                cat = "B"
            elif diff >= 12:
                cat = "S"
            else:
                cat = "N"

            # YakOS signals
            smash_prob = float(row.get("smash_prob", 0) or 0)
            bust_prob = float(row.get("bust_prob", 0) or 0)
            breakout = float(row.get("breakout_score", 0) or 0)
            edge_label = str(row.get("edge_label", "") or "")
            edge_score = float(row.get("edge_score", 0) or 0)
            leverage = float(row.get("leverage", 0) or 0)
            ceil_mag = float(row.get("ceil_magnitude", 0) or 0)
            pop_cat = float(row.get("pop_catalyst_score", 0) or 0)
            pop_tag = str(row.get("pop_catalyst_tag", "") or "")
            own = float(row.get("ownership", 0) or row.get("own_proj", 0) or 0)
            value = float(row.get("value", 0) or 0)

            # Caught / missed flags
            is_anchor = "Anchor" in edge_label or "Core" in edge_label
            smash_flag = smash_prob > 0.25 or breakout > 30 or is_anchor
            bust_flag = bust_prob > 0.25 or "Fade" in edge_label

            records.append({
                "n": str(row.get("player_name", "")),
                "t": str(row.get("team", "")),
                "o": str(row.get("opp", row.get("team", ""))),
                "p": str(row.get("pos", "")),
                "d": str(row.get("slate_date", "")),
                "sal": int(row.get("salary", 0) or 0),
                "cat": cat,
                "proj": round(proj, 1),
                "act": round(actual, 1),
                "diff": diff,
                "own": round(own, 1),
                "es": round(edge_score, 3),
                "el": edge_label,
                "sp": round(smash_prob, 3),
                "bp": round(bust_prob, 3),
                "br": round(breakout, 1),
                "lev": round(leverage, 1),
                "cm": round(ceil_mag, 3),
                "pc": round(pop_cat, 3),
                "pt": pop_tag,
                "val": round(value, 2),
                "sf": smash_flag,
                "bf": bust_flag,
            })

    # ── Live slate (no actuals yet) ──────────────────────────────────────
    live_path = REPO_ROOT / "data" / "published" / sport_lower / "signals.parquet"
    if live_path.exists():
        try:
            ldf = pd.read_parquet(live_path)
            for _, row in ldf.iterrows():
                records.append({
                    "n": str(row.get("player_name", "")),
                    "t": str(row.get("team", "")),
                    "o": str(row.get("opp", "")),
                    "p": str(row.get("pos", "")),
                    "d": "LIVE",
                    "sal": int(row.get("salary", 0) or 0),
                    "cat": "L",
                    "proj": round(float(row.get("proj", 0) or 0), 1),
                    "act": 0,
                    "diff": 0,
                    "own": round(float(row.get("ownership", 0) or 0), 1),
                    "es": round(float(row.get("edge_score", 0) or 0), 3),
                    "el": str(row.get("edge_label", "") or ""),
                    "sp": round(float(row.get("smash_prob", 0) or 0), 3),
                    "bp": round(float(row.get("bust_prob", 0) or 0), 3),
                    "br": round(float(row.get("breakout_score", 0) or 0), 1),
                    "lev": round(float(row.get("leverage", 0) or 0), 1),
                    "cm": round(float(row.get("ceil_magnitude", 0) or 0), 3),
                    "pc": round(float(row.get("pop_catalyst_score", 0) or 0), 3),
                    "pt": str(row.get("pop_catalyst_tag", "") or ""),
                    "val": round(float(row.get("value", 0) or 0), 2),
                    "sf": False,
                    "bf": False,
                })
        except Exception:
            pass

    return records


# ── Catch Rate Line Chart Builder ────────────────────────────────────────────


def _compute_catch_rates(data: list) -> list:
    """Compute per-slate original and current-config catch rates.

    Returns a list of dicts with per-date stats for the line chart.
    Uses the same *data* list that feeds the scatter explorer.
    """
    from collections import defaultdict

    by_date: dict = defaultdict(list)
    for r in data:
        if r["cat"] == "L" or r["d"] == "LIVE":
            continue
        by_date[r["d"]].append(r)

    results = []
    for dt in sorted(by_date):
        players = by_date[dt]
        n = len(players)

        smashes = [r for r in players if r["diff"] >= 12]
        busts = [r for r in players if r["diff"] <= -12]
        n_s = len(smashes)
        n_b = len(busts)
        if n_s == 0 and n_b == 0:
            continue

        # ── Original catch logic (dashboard scatter categories) ──
        def _orig_smash(r: dict) -> bool:
            return r["sf"]

        def _orig_bust(r: dict) -> bool:
            return r["bf"]

        # ── Current / fixed catch logic ──
        # Lower smash threshold to 0.15, include Value label
        def _curr_smash(r: dict) -> bool:
            if r["sf"]:
                return True
            return r["sp"] > 0.15 or "Value" in r["el"]

        # Lower bust threshold to 0.15
        def _curr_bust(r: dict) -> bool:
            if r["bf"]:
                return True
            return r["bp"] > 0.15

        os = sum(1 for r in smashes if _orig_smash(r))
        ob = sum(1 for r in busts if _orig_bust(r))
        fs = sum(1 for r in smashes if _curr_smash(r))
        fb = sum(1 for r in busts if _curr_bust(r))

        # Short date label for chart (MM/DD)
        parts = dt.split("-")
        short = f"{parts[1]}/{parts[2]}" if len(parts) == 3 else dt

        results.append({
            "d": short,
            "ns": n_s,
            "nb": n_b,
            "os": round(os / n_s * 100, 1) if n_s else 0,
            "ob": round(ob / n_b * 100, 1) if n_b else 0,
            "fs": round(fs / n_s * 100, 1) if n_s else 0,
            "fb": round(fb / n_b * 100, 1) if n_b else 0,
        })

    return results


def _build_catch_rate_html(rates: list) -> str:
    """Return self-contained HTML/JS for the catch-rate line chart."""
    if not rates:
        return ""

    data_json = json.dumps(rates, separators=(",", ":"))

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f1117;color:#c8ccd4;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;padding:16px 20px}}
.hdr{{display:flex;align-items:center;gap:16px;margin-bottom:12px;flex-wrap:wrap}}
.hdr h2{{font-size:14px;color:#e2e5ec;font-weight:600}}
.stats{{display:flex;gap:16px;margin-left:auto;flex-wrap:wrap}}
.st{{text-align:center;background:#161921;border:1px solid #2a2d38;border-radius:6px;padding:6px 12px;min-width:100px}}
.st .sl{{font-size:8px;color:#6b7280;text-transform:uppercase;letter-spacing:.4px}}
.st .sv{{font-size:16px;font-weight:700}}
.st .sd{{font-size:9px;color:#4b5060}}
.green{{color:#22c55e}}.red{{color:#ef4444}}.amber{{color:#f59e0b}}.blue{{color:#3b82f6}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:8px}}
.cw{{position:relative;height:260px}}
.cw h3{{font-size:11px;color:#6b7280;margin-bottom:4px}}
.note{{font-size:9px;color:#3b3f4a;line-height:1.5}}
</style>
</head>
<body>
<div class="hdr">
  <h2>Catch Rate Trend</h2>
  <div class="stats" id="hdr"></div>
</div>
<div class="charts">
  <div><h3>Smash Catch Rate by Slate</h3><div class="cw"><canvas id="sc"></canvas></div></div>
  <div><h3>Bust Catch Rate by Slate</h3><div class="cw"><canvas id="bc"></canvas></div></div>
</div>
<div class="note">
  Original: sp&gt;0.25 | breakout&gt;30 | Anchor/Core &middot; bp&gt;0.25 | Fade &nbsp;&nbsp;|&nbsp;&nbsp;
  Current: sp&gt;0.15 | Value label added &middot; bp&gt;0.15
</div>
<script>
const R=''' + data_json + ''';
const L=R.map(r=>r.d);
const tS=R.reduce((s,r)=>s+r.ns,0),tB=R.reduce((s,r)=>s+r.nb,0);
const tOS=R.reduce((s,r)=>s+Math.round(r.ns*r.os/100),0);
const tFS=R.reduce((s,r)=>s+Math.round(r.ns*r.fs/100),0);
const tOB=R.reduce((s,r)=>s+Math.round(r.nb*r.ob/100),0);
const tFB=R.reduce((s,r)=>s+Math.round(r.nb*r.fb/100),0);
document.getElementById("hdr").innerHTML=
  '<div class="st"><div class="sl">Orig Smash</div><div class="sv green">'+Math.round(tOS/tS*100)+'%</div><div class="sd">'+tOS+'/'+tS+'</div></div>'+
  '<div class="st"><div class="sl">Curr Smash</div><div class="sv blue">'+Math.round(tFS/tS*100)+'%</div><div class="sd">'+tFS+'/'+tS+' &middot; +'+Math.round((tFS-tOS)/tS*100)+'pp</div></div>'+
  '<div class="st"><div class="sl">Orig Bust</div><div class="sv red">'+Math.round(tOB/tB*100)+'%</div><div class="sd">'+tOB+'/'+tB+'</div></div>'+
  '<div class="st"><div class="sl">Curr Bust</div><div class="sv amber">'+Math.round(tFB/tB*100)+'%</div><div class="sd">'+tFB+'/'+tB+' &middot; +'+Math.round((tFB-tOB)/tB*100)+'pp</div></div>';
const G='#1a1d28',T='#3b3f4a';
function mkOpts(isS){return{responsive:true,maintainAspectRatio:false,animation:{duration:300},interaction:{mode:'index',intersect:false},plugins:{legend:{position:'top',labels:{color:'#6b7280',font:{size:9},boxWidth:10,padding:10,usePointStyle:true}},tooltip:{backgroundColor:'#1a1d28ee',borderColor:'#2a2d38',borderWidth:1,titleColor:'#e2e5ec',bodyColor:'#c8ccd4',titleFont:{size:10},bodyFont:{size:9},callbacks:{afterTitle:ctx=>{const d=R[ctx[0].dataIndex];return isS?d.ns+' smashes':d.nb+' busts'},label:ctx=>' '+ctx.dataset.label+': '+ctx.parsed.y.toFixed(1)+'%'}}},scales:{x:{grid:{color:G},ticks:{color:T,font:{size:8},maxRotation:45}},y:{min:0,max:100,grid:{color:G},ticks:{color:T,font:{size:8},callback:v=>v+'%'}}}}}
new Chart(document.getElementById('sc'),{type:'line',data:{labels:L,datasets:[{label:'Original',data:R.map(r=>r.os),borderColor:'#22c55e',borderWidth:1.5,pointRadius:2.5,tension:.3,borderDash:[5,3]},{label:'Current',data:R.map(r=>r.fs),borderColor:'#3b82f6',borderWidth:2,pointRadius:3,tension:.3}]},options:mkOpts(true)});
new Chart(document.getElementById('bc'),{type:'line',data:{labels:L,datasets:[{label:'Original',data:R.map(r=>r.ob),borderColor:'#ef4444',borderWidth:1.5,pointRadius:2.5,tension:.3,borderDash:[5,3]},{label:'Current',data:R.map(r=>r.fb),borderColor:'#f59e0b',borderWidth:2,pointRadius:3,tension:.3}]},options:mkOpts(false)});
</script>
</body>
</html>'''


# ── Explorer HTML Builder ────────────────────────────────────────────────────

def _build_explorer_html(data: list) -> str:
    """Return self-contained HTML/JS for the scatter explorer."""
    # Clean NaN values
    for r in data:
        for k, v in r.items():
            if isinstance(v, float) and v != v:
                r[k] = 0

    data_json = json.dumps(data, separators=(",", ":"))

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f1117;color:#c8ccd4;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;overflow:hidden;height:100vh}}
.wrap{{display:grid;grid-template-columns:230px 1fr;grid-template-rows:auto 1fr auto;height:100vh}}
.header{{grid-column:1/3;background:#161921;border-bottom:1px solid #2a2d38;padding:8px 16px;display:flex;align-items:center;gap:16px}}
.header h1{{font-size:14px;color:#e2e5ec;font-weight:600;white-space:nowrap}}
.header .cs{{display:flex;gap:14px;margin-left:auto}}
.cs-item{{text-align:center}}
.cs-item .csl{{font-size:9px;color:#6b7280;text-transform:uppercase;letter-spacing:.4px}}
.cs-item .csv{{font-size:16px;font-weight:700}}
.cs-item .css{{font-size:9px;color:#4b5060}}
.green{{color:#22c55e}}.red{{color:#ef4444}}.amber{{color:#f59e0b}}
.sidebar{{grid-row:2/4;background:#161921;border-right:1px solid #2a2d38;padding:12px;overflow-y:auto}}
.sidebar h3{{font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}}
.ct{{display:flex;align-items:center;gap:7px;padding:4px 6px;border-radius:4px;cursor:pointer;user-select:none;border:1px solid transparent;margin-bottom:3px}}
.ct:hover{{background:#1e2130}}.ct.on{{border-color:#2a2d38;background:#1a1d28}}
.ct .dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
.ct .ctl{{font-size:11px;flex:1}}.ct .ctc{{font-size:10px;color:#4b5060;font-family:monospace}}
.divider{{height:1px;background:#2a2d38;margin:10px 0}}
.asel{{margin-bottom:10px}}
.asel label{{font-size:9px;color:#6b7280;text-transform:uppercase;letter-spacing:.4px;display:block;margin-bottom:2px}}
.asel select{{width:100%;background:#1e2130;border:1px solid #2a2d38;color:#c8ccd4;padding:4px 6px;border-radius:4px;font-size:11px;cursor:pointer}}
.sl{{margin-bottom:10px}}
.sl .slh{{display:flex;justify-content:space-between;margin-bottom:3px}}
.sl .sll{{font-size:11px;color:#9ca3af}}.sl .slv{{font-size:10px;color:#4b5060;font-family:monospace}}
.sl input[type=range]{{width:100%;height:3px;-webkit-appearance:none;background:#2a2d38;border-radius:2px;outline:none}}
.sl input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:11px;height:11px;background:#4f6df0;border-radius:50%;cursor:pointer;border:2px solid #0f1117}}
.rbtn{{width:100%;padding:4px;background:#1e2130;border:1px solid #2a2d38;color:#6b7280;border-radius:4px;cursor:pointer;font-size:10px;margin-bottom:10px}}
.rbtn:hover{{border-color:#4f6df0;color:#4f6df0}}
.chart-area{{position:relative;padding:6px 10px}}
.chart-area canvas{{width:100%!important;height:100%!important}}
.bottom{{grid-column:2;background:#161921;border-top:1px solid #2a2d38;padding:6px 16px;display:flex;align-items:center;gap:24px;font-size:11px}}
.sb .sbl{{font-size:8px;color:#4b5060;text-transform:uppercase;letter-spacing:.3px}}
.sb .sbv{{font-size:13px;font-weight:600;color:#e2e5ec}}
.tip{{position:absolute;background:#1a1d28ee;border:1px solid #2a2d38;border-radius:6px;padding:8px 10px;font-size:10px;pointer-events:none;z-index:100;min-width:200px;box-shadow:0 4px 12px rgba(0,0,0,.5)}}
.tip .tn{{font-size:12px;font-weight:600;margin-bottom:2px}}
.tip .ts{{color:#4b5060;margin-bottom:5px;font-size:9px}}
.tip .tr{{display:flex;justify-content:space-between;gap:14px;line-height:1.6}}
.tip .tl{{color:#6b7280}}.tip .tv{{color:#c8ccd4;font-family:monospace;font-size:10px}}
.tip .tag{{display:inline-block;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600;margin-left:6px}}
</style>
</head>
<body>
<div class="wrap">
<div class="header">
  <h1>Smash / Bust Explorer</h1>
  <div class="cs" id="hdr"></div>
</div>
<div class="sidebar">
  <h3>Categories</h3>
  <div id="cats"></div>
  <div class="divider"></div>
  <h3>Axes</h3>
  <div class="asel"><label>X Axis</label><select id="xA" onchange="rebuild()"></select></div>
  <div class="asel"><label>Y Axis</label><select id="yA" onchange="rebuild()"></select></div>
  <div class="divider"></div>
  <h3>Filters</h3>
  <button class="rbtn" onclick="resetF()">Reset Filters</button>
  <div id="sls"></div>
</div>
<div class="chart-area">
  <canvas id="sc"></canvas>
  <div class="tip" id="tip" style="display:none"></div>
</div>
<div class="bottom" id="bot"></div>
</div>
<script>
const D=''' + data_json + ''';
const CATS=[
  {{id:"sc",l:"Smash Caught",c:"#22c55e",t:r=>r.cat==="S"&&r.sf,rad:6}},
  {{id:"sm",l:"Smash Missed",c:"#86efac",t:r=>r.cat==="S"&&!r.sf,rad:6,bdr:"#22c55e"}},
  {{id:"bc",l:"Bust Caught",c:"#ef4444",t:r=>r.cat==="B"&&r.bf,rad:6}},
  {{id:"bm",l:"Bust Missed",c:"#fca5a5",t:r=>r.cat==="B"&&!r.bf,rad:6,bdr:"#ef4444"}},
  {{id:"lv",l:"Live",c:"#3b82f6",t:r=>r.cat==="L",rad:5}},
];
let show={{}};CATS.forEach(c=>show[c.id]=true);
const F={{
  diff:{{l:"Actual vs Proj",f:v=>(v>=0?"+":"")+v.toFixed(1)}},
  sal:{{l:"Salary",f:v=>"$"+v.toLocaleString()}},
  own:{{l:"Ownership %",f:v=>v.toFixed(1)+"%"}},
  proj:{{l:"Projection",f:v=>v.toFixed(1)}},
  act:{{l:"Actual FPTS",f:v=>v.toFixed(1)}},
  es:{{l:"Edge Score",f:v=>v.toFixed(3)}},
  sp:{{l:"Smash Prob",f:v=>v.toFixed(3)}},
  bp:{{l:"Bust Prob",f:v=>v.toFixed(3)}},
  br:{{l:"Breakout",f:v=>v.toFixed(1)}},
  lev:{{l:"Leverage",f:v=>v.toFixed(1)}},
  cm:{{l:"Ceil Magnitude",f:v=>v.toFixed(3)}},
  pc:{{l:"Pop Catalyst",f:v=>v.toFixed(3)}},
  val:{{l:"Value",f:v=>v.toFixed(2)}},
}};
const SL=[
  {{f:"es",l:"Edge Score",mn:0,mx:.7,st:.01}},
  {{f:"sp",l:"Smash Prob",mn:0,mx:1,st:.01}},
  {{f:"bp",l:"Bust Prob",mn:0,mx:.5,st:.01}},
  {{f:"br",l:"Breakout",mn:0,mx:55,st:1}},
  {{f:"sal",l:"Salary",mn:0,mx:17000,st:500}},
  {{f:"own",l:"Ownership %",mn:0,mx:70,st:1}},
];
let flt={{}};
// Build cats
const cd=document.getElementById("cats");
CATS.forEach(c=>{{
  const n=D.filter(c.t).length;
  const e=document.createElement("div");e.className="ct on";e.dataset.id=c.id;
  const sty=c.bdr?"border:2px dashed "+c.bdr+";background:transparent":"background:"+c.c;
  e.innerHTML='<div class="dot" style="'+sty+'"></div><span class="ctl">'+c.l+'</span><span class="ctc">'+n+'</span>';
  e.onclick=(ev)=>{{if(ev.shiftKey){{show[c.id]=!show[c.id];e.classList.toggle("on");}}else{{const solo=Object.values(show).filter(v=>v).length===1&&show[c.id];CATS.forEach(x=>{{show[x.id]=solo;document.querySelector('[data-id="'+x.id+'"]').classList.toggle("on",solo);}});if(!solo){{show[c.id]=true;e.classList.add("on");}}}}rebuild();}};
  cd.appendChild(e);
}});
// Axes
const ak=Object.keys(F);
["xA","yA"].forEach((id,i)=>{{
  const s=document.getElementById(id);
  ak.forEach(f=>{{const o=document.createElement("option");o.value=f;o.textContent=F[f].l;s.appendChild(o);}});
  s.value=i===0?"es":"diff";
}});
// Sliders
const sd=document.getElementById("sls");
SL.forEach(s=>{{
  flt[s.f]=s.mn;
  const d=document.createElement("div");d.className="sl";
  d.innerHTML='<div class="slh"><span class="sll">'+s.l+'</span><span class="slv" id="sv-'+s.f+'">\\u2265 '+s.mn+'</span></div><input type="range" min="'+s.mn+'" max="'+s.mx+'" step="'+s.st+'" value="'+s.mn+'" id="sl-'+s.f+'">';
  sd.appendChild(d);
  d.querySelector("input").addEventListener("input",function(){{flt[s.f]=parseFloat(this.value);document.getElementById("sv-"+s.f).textContent="\\u2265 "+this.value;rebuild();}});
}});
function resetF(){{SL.forEach(s=>{{document.getElementById("sl-"+s.f).value=s.mn;flt[s.f]=s.mn;document.getElementById("sv-"+s.f).textContent="\\u2265 "+s.mn;}});rebuild();}}
let chart=null;
function rebuild(){{
  const xf=document.getElementById("xA").value,yf=document.getElementById("yA").value;
  const fd=D.filter(r=>{{for(const f in flt)if((r[f]||0)<flt[f])return false;return true;}});
  const ds=[];
  ["lv","bm","bc","sm","sc"].forEach(cid=>{{
    const cat=CATS.find(c=>c.id===cid);if(!show[cid])return;
    const pts=fd.filter(cat.t);if(!pts.length)return;
    const isO=cid!=="n",isM=cid.endsWith("m"),isL=cid==="lv";
    ds.push({{
      label:cat.l,data:pts.map(r=>({{x:r[xf]||0,y:r[yf]||0,r:r}})),
      backgroundColor:isM?cat.c+"60":cat.c+(isO?"99":"30"),
      borderColor:isM?(cat.bdr||cat.c):cat.c,borderWidth:isM?2:isO?1:.5,
      borderDash:isM?[3,3]:[],pointRadius:cat.rad,pointHoverRadius:cat.rad+3,
      pointStyle:isM?"rectRot":isL?"triangle":"circle",order:isO?1:3,
    }});
  }});
  if(chart)chart.destroy();
  chart=new Chart(document.getElementById("sc"),{{
    type:"scatter",data:{{datasets:ds}},
    options:{{
      responsive:true,maintainAspectRatio:false,animation:{{duration:0}},
      plugins:{{legend:{{display:true,position:"top",labels:{{color:"#6b7280",font:{{size:10}},boxWidth:10,padding:12,usePointStyle:true}}}},tooltip:{{enabled:false}}}},
      scales:{{
        x:{{title:{{display:true,text:F[xf].l,color:"#6b7280",font:{{size:10}}}},grid:{{color:"#1a1d28"}},ticks:{{color:"#3b3f4a",font:{{size:9}}}}}},
        y:{{title:{{display:true,text:F[yf].l,color:"#6b7280",font:{{size:10}}}},grid:{{color:"#1a1d28"}},ticks:{{color:"#3b3f4a",font:{{size:9}}}}}}
      }},
      onHover:(e,els)=>{{
        const tip=document.getElementById("tip");
        if(!els.length){{tip.style.display="none";return;}}
        const r=els[0].element.$context.raw.r;
        const ci=CATS.find(c=>c.t(r));const cc=ci?ci.c:"#6b7280";
        const caught=r.cat==="S"?r.sf:r.cat==="B"?r.bf:null;
        const ctag=caught===true?'<span class="tag" style="background:#22c55e30;color:#22c55e">CAUGHT</span>':caught===false?'<span class="tag" style="background:#f59e0b30;color:#f59e0b">MISSED</span>':r.cat==="L"?'<span class="tag" style="background:#3b82f630;color:#3b82f6">LIVE</span>':"";
        tip.innerHTML='<div class="tn" style="color:'+cc+'">'+r.n+" "+ctag+"</div>"+
          '<div class="ts">'+r.t+" vs "+r.o+" · "+r.d+" · "+r.p+"</div>"+
          '<div class="tr"><span class="tl">Salary</span><span class="tv">$'+(r.sal||0).toLocaleString()+"</span></div>"+
          '<div class="tr"><span class="tl">Proj → Act</span><span class="tv">'+r.proj.toFixed(1)+" → "+r.act.toFixed(1)+"</span></div>"+
          '<div class="tr"><span class="tl">Diff</span><span class="tv" style="color:'+(r.diff>=0?"#22c55e":"#ef4444")+'">'+(r.diff>=0?"+":"")+r.diff.toFixed(1)+"</span></div>"+
          '<div class="tr"><span class="tl">Edge</span><span class="tv">'+r.es.toFixed(3)+" "+r.el+"</span></div>"+
          '<div class="tr"><span class="tl">Smash P</span><span class="tv">'+r.sp.toFixed(3)+"</span></div>"+
          '<div class="tr"><span class="tl">Bust P</span><span class="tv">'+r.bp.toFixed(3)+"</span></div>"+
          '<div class="tr"><span class="tl">Breakout</span><span class="tv">'+r.br.toFixed(1)+"</span></div>"+
          '<div class="tr"><span class="tl">Own%</span><span class="tv">'+r.own.toFixed(1)+"%</span></div>"+
          '<div class="tr"><span class="tl">Leverage</span><span class="tv">'+r.lev.toFixed(1)+"</span></div>"+
          (r.pt?'<div class="tr"><span class="tl">Pop Tag</span><span class="tv">'+r.pt+"</span></div>":"");
        const rect=document.getElementById("sc").getBoundingClientRect();
        let lx=e.native.clientX-rect.left+14,ly=e.native.clientY-rect.top-16;
        tip.style.display="block";tip.style.left=lx+"px";tip.style.top=ly+"px";
        const tr2=tip.getBoundingClientRect();
        if(tr2.right>rect.right)tip.style.left=(lx-tr2.width-20)+"px";
        if(tr2.bottom>rect.bottom)tip.style.top=(ly-tr2.height+12)+"px";
      }}
    }}
  }});
  // Header stats
  const sm=fd.filter(r=>r.cat==="S"),bu=fd.filter(r=>r.cat==="B");
  const sc2=sm.filter(r=>r.sf).length,bc2=bu.filter(r=>r.bf).length;
  document.getElementById("hdr").innerHTML=
    '<div class="cs-item"><div class="csl">Smash Catch</div><div class="csv green">'+(sm.length?Math.round(sc2/sm.length*100):0)+'%</div><div class="css">'+sc2+"/"+sm.length+"</div></div>"+
    '<div class="cs-item"><div class="csl">Missed</div><div class="csv amber">'+(sm.length-sc2)+'</div></div>'+
    '<div class="cs-item"><div class="csl">Bust Catch</div><div class="csv red">'+(bu.length?Math.round(bc2/bu.length*100):0)+'%</div><div class="css">'+bc2+"/"+bu.length+"</div></div>"+
    '<div class="cs-item"><div class="csl">Missed</div><div class="csv amber">'+(bu.length-bc2)+'</div></div>';
  // Bottom
  document.getElementById("bot").innerHTML=
    '<div class="sb"><span class="sbl">Filtered</span><span class="sbv">'+fd.length+"/"+D.length+"</span></div>"+
    '<div class="sb"><span class="sbl">Smash Rate</span><span class="sbv green">'+(fd.length?(sm.length/fd.length*100).toFixed(1):"0")+"%</span></div>"+
    '<div class="sb"><span class="sbl">Bust Rate</span><span class="sbv red">'+(fd.length?(bu.length/fd.length*100).toFixed(1):"0")+"%</span></div>"+
    '<div class="sb"><span class="sbl">Avg Smash</span><span class="sbv green">+'+(sm.length?(sm.reduce((s,r)=>s+r.diff,0)/sm.length).toFixed(1):"0")+"</span></div>"+
    '<div class="sb"><span class="sbl">Avg Bust</span><span class="sbv red">'+(bu.length?(bu.reduce((s,r)=>s+r.diff,0)/bu.length).toFixed(1):"0")+"</span></div>";
}}
rebuild();
</script>
</body>
</html>'''


# ══════════════════════════════════════════════════════════════════════════════
# Main Render
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard_tab(sport: str) -> None:
    """Render the Dashboard tab — Smash/Bust Explorer + Maintenance Tools."""

    # ── Load archive data ─────────────────────────────────────────────────
    data = _load_explorer_data(sport)

    if not data:
        st.info(
            "No archived slate data found.  Run slates and post-slate feedback "
            "to populate the dashboard."
        )

    # ── Catch Rate Line Chart ─────────────────────────────────────────────
    if data:
        rates = _compute_catch_rates(data)
        if rates:
            cr_html = _build_catch_rate_html(rates)
            components.html(cr_html, height=380, scrolling=False)

        n_hist = sum(1 for r in data if r["cat"] != "L")
        n_live = sum(1 for r in data if r["cat"] == "L")
        slates = len(set(r["d"] for r in data if r["cat"] != "L" and r["d"] != "LIVE"))
        st.caption(
            f"{n_hist} historical players across {slates} archived slates"
            + (f" · {n_live} live players" if n_live else "")
        )

    # ── Scatter Explorer (hosted standalone) ───────────────────────
    st.markdown(
        "[Open Smash / Bust Explorer ↗](https://www.perplexity.ai/computer/a/yakos-smash-bust-explorer-Ir3N20l9RcCPKV0fnhxhhA)",
        help="Interactive scatter plot — click categories to isolate, shift+click to toggle, use filters to drill in.",
    )

    # ── Ricky's Hot Box ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔥 Ricky's Hot Box")
    with st.expander("Projection Accuracy Report", expanded=False):
        try:
            _render_projection_accuracy_report(sport)
        except Exception as e:
            st.error(f"Projection Accuracy Report error: {e}\n```\n{traceback.format_exc()}\n```")

    # ── Maintenance Tools ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Maintenance Tools", expanded=False):
        try:
            _render_post_slate_feedback(sport)
        except Exception as e:
            st.error(f"Post-Slate Feedback error: {e}\n```\n{traceback.format_exc()}\n```")
        st.markdown("---")
        try:
            _render_recalibrate_from_archive(sport)
        except Exception as e:
            st.error(f"Recalibrate from Archive error: {e}\n```\n{traceback.format_exc()}\n```")


def _render_projection_accuracy_report(sport: str) -> None:
    """Render the Projection Accuracy Report — Raw RG vs Config-Adjusted side-by-side.

    Read-only. Does not modify any projection logic or calibration settings.
    """
    from yak_core.projection_accuracy import (
        OUTLIER_MAE_DELTA,
        compute_config_adjusted_accuracy,
        load_accuracy_report,
        build_summary_df,
        build_per_date_df,
    )

    st.caption(
        "Compares raw RotoGrinders projections against Ceiling Hunter config-adjusted "
        "projections. Green = config improved accuracy. Red = config degraded accuracy."
    )

    col_run, col_cached = st.columns([1, 3])
    with col_run:
        run_clicked = st.button(
            "▶ Run Accuracy Report",
            key=f"proj_acc_run_{sport}",
            help="Scans all available RG archive dates against actuals on demand.",
        )

    if run_clicked:
        with st.spinner("Computing accuracy metrics across all dates…"):
            report = compute_config_adjusted_accuracy(sport=sport)
        if "error" in report:
            st.warning(f"Could not compute report: {report['error']}")
            return
        n = report.get("n_dates", 0)
        st.success(f"Report updated — {n} date{'s' if n != 1 else ''} analyzed.")
    else:
        report = load_accuracy_report()
        if report is None:
            st.info("No report computed yet. Click **▶ Run Accuracy Report** to generate.")
            return

    if not report.get("summary"):
        st.warning("Report is empty — no dates with both RG projections and actuals found.")
        return

    n_dates = report.get("n_dates", 0)
    st.caption(f"Based on {n_dates} date(s) with matched RG projections + actuals.")

    # ── Summary table ────────────────────────────────────────────────────
    st.markdown("#### Summary")
    summary_df = build_summary_df(report)

    def _highlight_delta(row: pd.Series) -> list[str]:
        delta = row["Delta"]
        metric = row["Metric"]
        # For MAE and Bias: lower adj = better → green if delta < 0
        # For all others: higher adj = better → green if delta > 0
        if metric in ("MAE (avg)", "Bias (avg)"):
            good = delta < 0
        else:
            good = delta > 0
        color = "color: #2ecc71" if good else ("color: #e74c3c" if delta != 0 else "")
        return ["", "", "", color]

    styled = summary_df.style.apply(_highlight_delta, axis=1).format(
        {
            "Raw RG": "{:.3f}",
            "Config-Adjusted": "{:.3f}",
            "Delta": "{:+.3f}",
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Per-date breakdown ────────────────────────────────────────────────
    with st.expander("Per-date breakdown", expanded=False):
        per_date_df = build_per_date_df(report)
        if per_date_df.empty:
            st.caption("No per-date data available.")
        else:
            # Highlight MAE delta column using the same threshold as the stored outlier flag
            def _highlight_per_date(row: pd.Series) -> list[str]:
                styles = [""] * len(row)
                try:
                    delta_idx = per_date_df.columns.get_loc("MAE Δ")
                    delta_val = row.iloc[delta_idx]
                    if isinstance(delta_val, (int, float)) and not pd.isna(delta_val):
                        if delta_val < -OUTLIER_MAE_DELTA:
                            styles[delta_idx] = "color: #2ecc71"
                        elif delta_val > OUTLIER_MAE_DELTA:
                            styles[delta_idx] = "color: #e74c3c"
                except Exception:
                    pass
                return styles

            fmt: Dict[str, str] = {}
            for col in ["RG MAE", "Adj MAE", "MAE Δ", "RG Bias", "Adj Bias", "RG Corr", "Adj Corr", "RG Top-20%", "Adj Top-20%"]:
                if col in per_date_df.columns:
                    fmt[col] = "{:.3f}"

            try:
                styled_pd = per_date_df.style.apply(_highlight_per_date, axis=1).format(
                    fmt, na_rep=""
                )
                st.dataframe(styled_pd, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(per_date_df, use_container_width=True, hide_index=True)

        # Outlier callouts
        per_date = report.get("per_date", {})
        outliers = [
            (d, v["mae_delta"])
            for d, v in per_date.items()
            if v.get("is_outlier")
        ]
        if outliers:
            st.markdown("**Outlier dates** (config MAE δ ≥ 1.0):")
            for d, delta in sorted(outliers, key=lambda x: abs(x[1]), reverse=True):
                icon = "🟢" if delta < 0 else "🔴"
                st.caption(f"{icon} {d}: MAE Δ = {delta:+.3f}")



    st.markdown("### Post-Slate Feedback")

    feedback_date = st.date_input(
        "Slate date", value=date.today(), key=f"dash_fb_date_{sport}"
    )

    if st.button("Run Post-Slate", key=f"dash_postslate_{sport}"):
        with st.spinner("Running post-slate analysis..."):
            try:
                result = _run_post_slate(sport, str(feedback_date))
                if result.get("status") == "ok":
                    st.success(f"Post-slate complete: {result.get('message', '')}")
                    if result.get("calibration_update"):
                        st.json(result["calibration_update"])
                    # Display minutes calibration stats if available
                    mins_overall = result.get("minutes_stats", {})
                    if mins_overall.get("min_mae") is not None:
                        st.markdown(
                            f"**Minutes Accuracy:** MAE={mins_overall['min_mae']}, "
                            f"Corr={mins_overall.get('min_correlation', 'N/A')}, "
                            f"Bias={mins_overall.get('min_mean_error', 0):+.1f}"
                        )
                else:
                    st.warning(result.get("message", "No actuals available for this date."))
            except Exception as e:
                st.error(f"Post-slate error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Historical Backfill Section (PRESERVED — has user-active forms)
# ══════════════════════════════════════════════════════════════════════════════

def _render_historical_backfill(sport: str) -> None:
    """PGA and NBA backfill controls."""
    st.markdown("### Historical Backfill")
    st.caption("Calibrate against historical slates to build the curve faster.")

    if sport.upper() == "PGA":
        _render_pga_backfill()



def _render_recalibrate_from_archive(sport: str) -> None:
    """Recalibrate from Archive UI (sport-agnostic)."""
    st.markdown("### Recalibrate from Archive")
    st.caption(
        "Rebuild correction factors from archived slate parquets "
        "(uses YakOS projections vs actuals — not Tank01)."
    )
    archive_dir = REPO_ROOT / "data" / "slate_archive"
    parquets = sorted(archive_dir.glob("*.parquet")) if archive_dir.exists() else []

    # Separate NBA vs PGA archives
    nba_archives = [p for p in parquets if "pga" not in p.name.lower()]
    pga_archives = [p for p in parquets if "pga" in p.name.lower()]

    target = nba_archives if sport.upper() != "PGA" else pga_archives
    sport_label = "NBA" if sport.upper() != "PGA" else "PGA"
    st.info(f"Found **{len(target)}** {sport_label} archived slates in `data/slate_archive/`.")

    if st.button(
        f"Recalibrate {sport_label} from Archive",
        key="recalibrate_archive",
        type="primary",
    ):
        _recalibrate_from_archive(target, sport=sport_label)


def _render_pga_backfill() -> None:
    """PGA Historical Events backfill UI."""
    st.markdown("**PGA Historical Events**")

    dg_key = _resolve_datagolf_key()
    if not dg_key:
        st.warning("DATAGOLF_API_KEY not found in secrets or environment.")
        return

    if st.button("Load Events", key="pga_load_events"):
        with st.spinner("Fetching events from DataGolf..."):
            try:
                from yak_core.datagolf import DataGolfClient
                from yak_core.pga_calibration import get_pga_event_list
                dg = DataGolfClient(dg_key)
                events = get_pga_event_list(dg)
                if events.empty:
                    st.warning("No PGA events found.")
                    return
                st.session_state["pga_events"] = events
            except Exception as e:
                st.error(f"Failed to load events: {e}")
                return

    events = st.session_state.get("pga_events")
    if events is None or (isinstance(events, pd.DataFrame) and events.empty):
        return

    display_cols = [c for c in ["event_name", "date", "calendar_year", "event_id",
                                "dk_salaries", "dk_ownerships"] if c in events.columns]
    edit_df = events[display_cols].copy()
    edit_df.insert(0, "select", False)

    edited = st.data_editor(edit_df, use_container_width=True, hide_index=True,
                            key="pga_event_editor")
    selected = edited[edited["select"]].copy()

    if st.button("Run PGA Backfill", key="pga_run_backfill") and not selected.empty:
        from yak_core.datagolf import DataGolfClient
        from yak_core.pga_calibration import calibrate_pga_event
        from yak_core.outcome_logger import log_slate_outcomes

        dg = DataGolfClient(dg_key)
        progress = st.progress(0.0)
        results = []
        n = len(selected)

        for i, (_, row) in enumerate(selected.iterrows()):
            eid = int(row["event_id"])
            yr = int(row.get("calendar_year", 2025))
            evt_date = str(row.get("date", f"{yr}-{eid:03d}"))
            evt_name = row.get("event_name", f"Event {eid}")

            try:
                cal = calibrate_pga_event(dg, eid, yr, slate_date=evt_date)
                status = "error" if "error" in cal else "ok"
                mae = cal.get("calibration", {}).get("overall", {}).get("mae", 0) if status == "ok" else 0
                n_players = cal.get("n_players_calibrated", 0)

                # Log outcomes if calibration succeeded
                if status == "ok" and n_players > 0:
                    try:
                        from yak_core.pga_calibration import fetch_pga_actuals, _build_pool_from_actuals_and_preds
                        actuals = fetch_pga_actuals(dg, eid, yr)
                        pool = _build_pool_from_actuals_and_preds(dg, actuals, eid, yr)
                        if not pool.empty and "actual_fp" in pool.columns:
                            log_slate_outcomes(evt_date, pool, sport="PGA")
                    except Exception:
                        pass

                results.append({
                    "event": evt_name, "date": evt_date,
                    "MAE": round(mae, 2) if mae else "N/A",
                    "n_players": n_players, "status": status,
                    "detail": cal.get("error", ""),
                })
            except Exception as e:
                results.append({
                    "event": evt_name, "date": evt_date,
                    "MAE": "N/A", "n_players": 0,
                    "status": "error", "detail": str(e),
                })

            progress.progress((i + 1) / n)
            time.sleep(1)

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)




# ── Recalibrate from Archive ────────────────────────────────────────────────

def _inject_rg_mae(slate_date: str, pool_df: pd.DataFrame, sport: str = "NBA") -> None:
    """Compute RG baseline MAE and store it in slate_errors + rg_baseline.json.

    Looks for an RG archive file matching the slate date. If found, merges
    RG projections with actuals from pool_df and stores the MAE.
    """
    sport_lower = sport.lower()
    if sport_lower not in ("nba", "pga"):
        return

    rg_dir = REPO_ROOT / "data" / "rg_archive" / sport_lower
    rg_path = rg_dir / f"rg_{slate_date}.csv"
    if not rg_path.exists():
        return

    if "actual_fp" not in pool_df.columns:
        return

    try:
        rg = pd.read_csv(rg_path)
        rg.columns = [c.strip().upper() for c in rg.columns]
        if "FPTS" not in rg.columns or "PLAYER" not in rg.columns:
            return

        rg_clean = pd.DataFrame()
        rg_clean["player_name"] = rg["PLAYER"].astype(str).str.strip().str.replace('"', '')
        rg_clean["rg_proj"] = pd.to_numeric(rg["FPTS"], errors="coerce")
        rg_clean["_key"] = rg_clean["player_name"].str.strip().str.lower()

        actuals = pool_df[["player_name", "actual_fp"]].dropna(subset=["actual_fp"]).copy()
        actuals["_key"] = actuals["player_name"].astype(str).str.strip().str.lower()

        merged = rg_clean.merge(actuals[["_key", "actual_fp"]], on="_key", how="inner")
        if len(merged) < 10:
            return

        rg_mae = float((merged["rg_proj"] - merged["actual_fp"]).abs().mean())

        # Inject into slate_errors.json
        errors_path = REPO_ROOT / "data" / "calibration_feedback" / sport_lower / "slate_errors.json"
        if errors_path.exists():
            import json as _json
            with open(errors_path) as f:
                errors = _json.load(f)
            if slate_date in errors:
                errors[slate_date]["rg_mae"] = round(rg_mae, 2)
                with open(errors_path, "w") as f:
                    _json.dump(errors, f, indent=2)

        # Also update standalone rg_baseline.json
        baseline_path = REPO_ROOT / "data" / "calibration_feedback" / "rg_baseline.json"
        baseline = {}
        if baseline_path.exists():
            import json as _json
            with open(baseline_path) as f:
                baseline = _json.load(f)
        baseline[slate_date] = {
            "rg_mae": round(rg_mae, 2),
            "rg_bias": round(float((merged["rg_proj"] - merged["actual_fp"]).mean()), 2),
            "n_matched": len(merged),
        }
        with open(baseline_path, "w") as f:
            import json as _json
            _json.dump(baseline, f, indent=2)

    except Exception:
        pass  # Non-critical — don't break calibration flow


def _recalibrate_from_archive(archive_files: list, sport: str = "NBA") -> None:
    """Clear existing calibration and rebuild from archived slate parquets.

    Each archive parquet contains YakOS projections (``proj``) and actual
    fantasy points (``actual_fp``), so this bypasses Tank01 entirely and
    recalibrates using YakOS's own projection accuracy.
    """
    from yak_core.calibration_feedback import (
        clear_calibration_history,
        record_slate_errors,
        get_calibration_summary,
    )

    if not archive_files:
        st.warning("No archive files found for this sport.")
        return

    # 1. Clear existing calibration so we rebuild from scratch
    clear_calibration_history(sport=sport.upper())
    st.info(f"Cleared existing {sport} calibration history. Rebuilding...")

    progress = st.progress(0.0)
    status_text = st.empty()
    results = []
    n = len(archive_files)

    for i, fpath in enumerate(archive_files):
        fname = fpath.name if hasattr(fpath, "name") else os.path.basename(str(fpath))
        # Extract slate date from filename (e.g. "2026-02-05_gpp_main.parquet" → "2026-02-05")
        slate_date = fname.split("_")[0]
        status_text.text(f"Processing {fname} ({i + 1}/{n})...")

        try:
            df = pd.read_parquet(fpath)

            # Validate required columns
            required = {"player_name", "pos", "salary", "proj", "actual_fp"}
            missing = required - set(df.columns)
            if missing:
                results.append({
                    "file": fname, "date": slate_date, "status": "skipped",
                    "detail": f"Missing columns: {missing}", "n_players": 0,
                    "MAE": "N/A", "correlation": "N/A",
                })
                progress.progress((i + 1) / n)
                continue

            # Filter to players who actually played
            df["actual_fp"] = pd.to_numeric(df["actual_fp"], errors="coerce")
            df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
            valid = df[df["actual_fp"].notna() & (df["actual_fp"] > 0) & df["proj"].notna()].copy()

            if valid.empty:
                results.append({
                    "file": fname, "date": slate_date, "status": "skipped",
                    "detail": "No valid proj/actual pairs", "n_players": 0,
                    "MAE": "N/A", "correlation": "N/A",
                })
                progress.progress((i + 1) / n)
                continue

            # Record errors — this appends to history and recomputes corrections
            cal_result = record_slate_errors(slate_date, valid, sport=sport.upper())

            # Inject RG baseline MAE if we have an RG archive for this date
            _inject_rg_mae(slate_date, valid, sport=sport)

            if "error" in cal_result:
                results.append({
                    "file": fname, "date": slate_date, "status": "rejected",
                    "detail": cal_result["error"], "n_players": len(valid),
                    "MAE": "N/A", "correlation": "N/A",
                })
            else:
                ov = cal_result.get("overall", {})
                results.append({
                    "file": fname, "date": slate_date, "status": "ok",
                    "detail": "",
                    "n_players": ov.get("n_players", len(valid)),
                    "MAE": round(ov.get("mae", 0), 2),
                    "correlation": round(ov.get("correlation", 0), 4),
                })

        except Exception as e:
            results.append({
                "file": fname, "date": slate_date, "status": "error",
                "detail": str(e)[:120], "n_players": 0,
                "MAE": "N/A", "correlation": "N/A",
            })

        progress.progress((i + 1) / n)

    status_text.text("Recalibration complete.")

    # Show results table
    if results:
        res_df = pd.DataFrame(results)
        ok_count = (res_df["status"] == "ok").sum()
        st.success(f"Successfully processed {ok_count}/{n} slates.")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Show updated calibration summary
    try:
        summary = get_calibration_summary(sport=sport.upper())
        st.markdown("#### Updated Calibration")
        st.json(summary)
    except Exception:
        pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_rapidapi_key() -> str:
    """Resolve RapidAPI key from secrets or environment."""
    key = os.environ.get("RAPIDAPI_KEY") or os.environ.get("TANK01_RAPIDAPI_KEY", "")
    if not key:
        try:
            key = st.secrets.get("RAPIDAPI_KEY", "")
        except Exception:
            pass
    return key


def _resolve_datagolf_key() -> str:
    """Resolve DataGolf API key from secrets or environment."""
    key = os.environ.get("DATAGOLF_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("DATAGOLF_API_KEY", "")
        except Exception:
            pass
    return key


def _run_post_slate(sport: str, slate_date: str) -> Dict[str, Any]:
    """Run post-slate feedback loop.

    Attempts to fetch actuals and record slate errors for calibration.
    Since post_slate.py doesn't exist as a standalone script, we wire
    the core components directly.
    """
    from app.data_loader import published_dir

    out_dir = published_dir(sport)
    pool_path = out_dir / "slate_pool.parquet"

    if not pool_path.exists():
        return {"status": "error", "message": f"No pool found for {sport}"}

    pool = pd.read_parquet(pool_path)

    # Try to fetch actuals
    try:
        if sport.upper() == "NBA":
            from yak_core.live import fetch_actuals_from_api
            api_key = _resolve_rapidapi_key()
            if not api_key:
                return {"status": "error", "message": "Missing RAPIDAPI_KEY for fetching actuals"}

            actuals = fetch_actuals_from_api(slate_date, {"RAPIDAPI_KEY": api_key})
            if actuals.empty:
                return {"status": "error", "message": f"No actuals available for {slate_date}"}

            # Merge actuals into pool — drop the placeholder actual_fp
            # column first (set to NaN by fetch_live_dfs) so pandas doesn't
            # create actual_fp_x / actual_fp_y suffix columns.
            if "actual_fp" in pool.columns:
                pool = pool.drop(columns=["actual_fp"])
            if "mp_actual" in pool.columns:
                pool = pool.drop(columns=["mp_actual"])

            # Merge both actual_fp and mp_actual from actuals using two-pass name matching
            from yak_core.name_utils import merge_with_normalized_names
            value_cols = ["actual_fp"]
            if "mp_actual" in actuals.columns:
                value_cols.append("mp_actual")
            pool_with_actuals = merge_with_normalized_names(
                pool,
                actuals[["player_name"] + value_cols],
                on="player_name",
                how="left",
                value_cols=value_cols,
            )
        elif sport.upper() == "PGA":
            dg_key = _resolve_datagolf_key()
            if not dg_key:
                return {"status": "error", "message": "Missing DATAGOLF_API_KEY for fetching PGA actuals"}

            from yak_core.datagolf import DataGolfClient
            from yak_core.pga_calibration import get_pga_event_list, fetch_pga_actuals

            dg = DataGolfClient(dg_key)

            # Find the event matching the slate_date
            events = get_pga_event_list(dg)
            if events.empty:
                return {"status": "error", "message": "Could not fetch PGA event list from DataGolf"}

            # Match event to slate_date
            latest = None
            if "date" in events.columns:
                events["_date_str"] = events["date"].astype(str)
                candidates = events[events["_date_str"] <= slate_date]
                if not candidates.empty:
                    latest = candidates.iloc[0]
                else:
                    latest = events.iloc[0]
            else:
                latest = events.iloc[0]

            if latest is None:
                return {"status": "error", "message": "No matching PGA event found"}

            event_id = int(latest.get("event_id", 0))
            year = int(latest.get("calendar_year", date.today().year))
            event_name = str(latest.get("event_name", f"Event {event_id}"))

            # Fetch actuals
            actuals = fetch_pga_actuals(dg, event_id=event_id, year=year)
            if actuals.empty:
                return {"status": "error", "message": f"No PGA actuals available for {event_name}. The event may still be in progress, or your DataGolf plan may not include historical DFS data."}

            # Merge actuals into pool
            if "actual_fp" in pool.columns:
                pool = pool.drop(columns=["actual_fp"])

            # Try merging on dg_id first, then player_name fallback
            if "dg_id" in pool.columns and "dg_id" in actuals.columns:
                act_map = actuals.set_index("dg_id")["actual_fp"].to_dict()
                pool["actual_fp"] = pool["dg_id"].map(act_map)

            if "actual_fp" not in pool.columns or pool["actual_fp"].isna().all():
                if "player_name" in pool.columns and "player_name" in actuals.columns:
                    act_map = actuals.set_index("player_name")["actual_fp"].to_dict()
                    pool["actual_fp"] = pool["player_name"].map(act_map)

            pool_with_actuals = pool
        else:
            return {"status": "error", "message": f"{sport} post-slate actuals not yet implemented"}
    except Exception as e:
        return {"status": "error", "message": f"Could not fetch actuals: {e}"}

    # Record slate errors
    try:
        from yak_core.calibration_feedback import record_slate_errors, get_calibration_summary

        has_actual = pool_with_actuals["actual_fp"].notna().sum()
        if has_actual == 0:
            return {"status": "error", "message": "No actuals matched to pool players"}

        slate_record = record_slate_errors(slate_date, pool_with_actuals, sport=sport.upper())

        # Inject RG baseline MAE if we have an RG file for this date
        _inject_rg_mae(slate_date, pool_with_actuals, sport=sport)

        summary = get_calibration_summary(sport=sport.upper())
        result = {
            "status": "ok",
            "message": f"Recorded errors for {has_actual} players",
            "calibration_update": summary,
        }
        # Pass through minutes stats from the slate record if available
        mins_overall = slate_record.get("overall", {})
        if "min_mae" in mins_overall:
            result["minutes_stats"] = {
                k: mins_overall[k]
                for k in ("min_mae", "min_rmse", "min_correlation", "min_mean_error", "min_n_players")
                if k in mins_overall
            }
        return result
    except Exception as e:
        return {"status": "error", "message": f"Calibration update failed: {e}"}
