"""YakOS Calibration - log optimizer runs and compute cross-slate KPIs."""
import os, datetime, uuid
import pandas as pd
import numpy as np

CALIB_CSV = "calibration_runs_gpp_v2.csv"

CALIB_COLUMNS = [
    "run_id", "slate_date", "contest_type", "sport", "mode",
    "lineup_id", "salary_used", "proj", "actual_fp",
    "avg_proj_own", "max_proj_own",
    "num_players", "timestamp",
]


def log_calibration_run(ldf, cfg, root="."):
    """Append one optimizer run to the calibration CSV."""
    rid = str(uuid.uuid4())[:8]
    ts = datetime.datetime.now().isoformat()
    sl = cfg.get("SLATE_DATE", "unknown")
    ct = cfg.get("CONTEST_TYPE", "gpp")
    sp = cfg.get("SPORT", "NBA")
    md = cfg.get("DATA_MODE", "historical")
    rows = []
    df = ldf.copy()
    if "lineup_id" not in df.columns:
        df["lineup_id"] = 0
    for lid, g in df.groupby("lineup_id"):
        sa = int(g["salary"].sum()) if "salary" in g.columns else 0
        pj = round(g["proj"].sum(), 2) if "proj" in g.columns else 0
        af = round(g["actual_fp"].sum(), 2) if "actual_fp" in g.columns else 0
        if "ownership" in g.columns:
            ow = g["ownership"].astype(float)
        else:
            ow = pd.Series([0.0] * len(g))
        rows.append({
            "run_id": rid, "slate_date": sl,
            "contest_type": ct, "sport": sp,
            "mode": md, "lineup_id": int(lid),
            "salary_used": sa, "proj": pj,
            "actual_fp": af,
            "avg_proj_own": round(ow.mean(), 2),
            "max_proj_own": round(ow.max(), 2),
            "num_players": len(g),
            "timestamp": ts,
        })
    rdf = pd.DataFrame(rows, columns=CALIB_COLUMNS)
    cp = os.path.join(root, CALIB_CSV)
    hdr = not os.path.exists(cp)
    rdf.to_csv(cp, mode="a", header=hdr, index=False)
    print(f"[calib] Logged {len(rdf)} LU (run {rid})")
    return rdf


def load_calibration(root="."):
    """Load the calibration CSV. Falls back to legacy format if v2 not found."""
    cp = os.path.join(root, CALIB_CSV)
    if not os.path.exists(cp):
        # Try legacy CSV as fallback
        legacy = os.path.join(root, "calibration_runs_gpp.csv")
        if os.path.exists(legacy):
            import csv
            rows = []
            with open(legacy) as f:
                reader = csv.reader(f)
                header = next(reader)
                n_cols = len(header)
                for row in reader:
                    if len(row) == n_cols:
                        rows.append(dict(zip(header, row)))
                    elif len(row) == n_cols + 1:
                        # Extra run_id column prepended
                        rows.append({"run_id": row[0], **dict(zip(header, row[1:]))})
            df = pd.DataFrame(rows)
            # Map legacy columns to v2 schema where possible
            col_map = {"projection": "proj", "salary": "salary_used"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            for c in ["proj", "salary_used"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "slate_date" not in df.columns and "timestamp" in df.columns:
                df["slate_date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
            if "run_id" not in df.columns:
                df["run_id"] = range(len(df))
            if "actual_fp" not in df.columns:
                df["actual_fp"] = np.nan
            for c in CALIB_COLUMNS:
                if c not in df.columns:
                    df[c] = np.nan if c not in ["sport","mode","contest_type"] else "unknown"
            return df
        return pd.DataFrame(columns=CALIB_COLUMNS)
    df = pd.read_csv(cp)
    if "slate_date" in df.columns:
        df["slate_date"] = df["slate_date"].astype(str)
    return df


def compute_methodology_kpis(cdf):
    """Compute cross-slate methodology KPIs."""
    e = {"summary": pd.DataFrame(),
         "proj_buckets": pd.DataFrame(),
         "own_bands": pd.DataFrame(),
         "overall": {}}
    if cdf.empty:
        return e
    s = cdf.groupby(["run_id","slate_date","contest_type"]).agg(
        n=("lineup_id","nunique"),
        ap=("proj","mean"),
        aa=("actual_fp","mean"),
        asal=("salary_used","mean"),
        ao=("avg_proj_own","mean"),
    ).reset_index()
    s.columns = ["run_id","slate_date","contest_type",
                  "num_lineups","avg_proj","avg_actual",
                  "avg_salary","avg_own"]
    s["proj_diff"] = s["avg_actual"] - s["avg_proj"]
    d = s["avg_proj"].replace(0, 1)
    s["proj_diff_pct"] = ((s["proj_diff"]/d)*100).round(1)
    b = [0,180,200,220,240,260,999]
    bl = ["<180","180-200","200-220","220-240","240-260","260+"]
    c2 = cdf.copy()
    c2["pb"] = pd.cut(c2["proj"],bins=b,labels=bl)
    pb = c2.groupby("pb",observed=True).agg(
        cnt=("lineup_id","count"),
        ap=("proj","mean"),
        aa=("actual_fp","mean"),
    ).reset_index()
    pb.columns=["proj_bucket","count","avg_proj","avg_actual"]
    pb["diff"] = pb["avg_actual"] - pb["avg_proj"]
    ob2 = [0,10,20,30,40,100]
    ol2 = ["<10%","10-20%","20-30%","30-40%","40%+"]
    c2["ob"] = pd.cut(c2["avg_proj_own"],bins=ob2,labels=ol2)
    ow2 = c2.groupby("ob",observed=True).agg(
        cnt=("lineup_id","count"),
        ao=("avg_proj_own","mean"),
        aa=("actual_fp","mean"),
        ap=("proj","mean"),
    ).reset_index()
    ow2.columns=["own_band","count","avg_own",
                  "avg_actual","avg_proj"]
    ow2["leverage"] = 100 - ow2["avg_own"]
    ov = {
        "total_runs": cdf["run_id"].nunique(),
        "total_lineups": len(cdf),
        "total_slates": cdf["slate_date"].nunique(),
        "avg_proj": round(cdf["proj"].mean(), 1),
        "avg_actual": round(cdf["actual_fp"].mean(),1),
        "avg_own": round(cdf["avg_proj_own"].mean(),1),
    }
    return {"summary":s,"proj_buckets":pb,
            "own_bands":ow2,"overall":ov}


# ============================================================
# Player-Level Calibration CSV
# ============================================================
PLAYER_CALIB_CSV = "calibration_players_v2.csv"
PLAYER_CALIB_COLUMNS = [
    "run_id", "slate_date", "player_name", "team", "pos",
    "salary", "proj", "actual_fp", "ownership",
    "proj_source", "timestamp",
]


def log_player_calibration(pool_df, cfg, root="."):
    """Log player-level proj vs actual data for calibration."""
    rid = str(uuid.uuid4())[:8]
    ts = datetime.datetime.now().isoformat()
    sl = cfg.get("SLATE_DATE", "unknown")
    ps = cfg.get("PROJ_SOURCE", "model")
    rows = []
    for _, r in pool_df.iterrows():
        name = r.get("player_name", r.get("name", ""))
        rows.append({
            "run_id": rid,
            "slate_date": sl,
            "player_name": name,
            "team": r.get("team", ""),
            "pos": r.get("pos", ""),
            "salary": r.get("salary", 0),
            "proj": round(r.get("proj", 0), 2),
            "actual_fp": round(r.get("actual_fp", 0), 2),
            "ownership": round(r.get("ownership", 0), 2),
            "proj_source": ps,
            "timestamp": ts,
        })
    rdf = pd.DataFrame(rows, columns=PLAYER_CALIB_COLUMNS)
    cp = os.path.join(root, PLAYER_CALIB_CSV)
    hdr = not os.path.exists(cp)
    rdf.to_csv(cp, mode="a", header=hdr, index=False)
    print(f"[calib] Logged {len(rdf)} players (run {rid})")
    return rdf


def load_player_calibration(root="."):
    """Load the player-level calibration CSV."""
    cp = os.path.join(root, PLAYER_CALIB_CSV)
    if not os.path.exists(cp):
        return pd.DataFrame(columns=PLAYER_CALIB_COLUMNS)
    df = pd.read_csv(cp)
    if "slate_date" in df.columns:
        df["slate_date"] = df["slate_date"].astype(str)
    return df


# ============================================================
# Enhanced Calibration Analytics
# ============================================================
def compute_accuracy_metrics(proj_series, actual_series):
    """Compute MAE, RMSE, R-squared, and bias for proj vs actual."""
    mask = proj_series.notna() & actual_series.notna()
    p = proj_series[mask].values
    a = actual_series[mask].values
    if len(p) < 2:
        return {"n": len(p), "mae": None, "rmse": None, "r2": None, "bias": None}
    diff = a - p
    mae = round(float(np.mean(np.abs(diff))), 2)
    rmse = round(float(np.sqrt(np.mean(diff ** 2))), 2)
    bias = round(float(np.mean(diff)), 2)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = round(float(1 - ss_res / ss_tot), 3) if ss_tot > 0 else 0.0
    return {"n": len(p), "mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def compute_player_calibration_kpis(pdf):
    """Compute player-level calibration KPIs from player calib CSV."""
    result = {
        "overall": {},
        "by_position": pd.DataFrame(),
        "by_salary_tier": pd.DataFrame(),
        "by_slate": pd.DataFrame(),
        "accuracy": {},
    }
    if pdf.empty or "proj" not in pdf.columns or "actual_fp" not in pdf.columns:
        return result

    # Filter to rows with both proj and actual
    df = pdf[(pdf["proj"] > 0) & (pdf["actual_fp"].notna())].copy()
    if df.empty:
        return result

    # Overall accuracy
    result["accuracy"] = compute_accuracy_metrics(df["proj"], df["actual_fp"])
    result["overall"] = {
        "n_players": len(df),
        "n_slates": df["slate_date"].nunique() if "slate_date" in df.columns else 0,
        "avg_proj": round(df["proj"].mean(), 1),
        "avg_actual": round(df["actual_fp"].mean(), 1),
        "avg_salary": round(df["salary"].mean(), 0) if "salary" in df.columns else 0,
    }
    result["overall"].update(result["accuracy"])

    # By position
    if "pos" in df.columns:
        pos_rows = []
        for pos, grp in df.groupby("pos"):
            m = compute_accuracy_metrics(grp["proj"], grp["actual_fp"])
            pos_rows.append({
                "pos": pos, "n": m["n"],
                "avg_proj": round(grp["proj"].mean(), 1),
                "avg_actual": round(grp["actual_fp"].mean(), 1),
                "mae": m["mae"], "rmse": m["rmse"],
                "r2": m["r2"], "bias": m["bias"],
            })
        result["by_position"] = pd.DataFrame(pos_rows)

    # By salary tier
    if "salary" in df.columns:
        bins = [0, 4000, 5000, 6000, 7000, 8000, 100000]
        labels = ["<4K", "4-5K", "5-6K", "6-7K", "7-8K", "8K+"]
        df["sal_tier"] = pd.cut(df["salary"], bins=bins, labels=labels)
        tier_rows = []
        for tier, grp in df.groupby("sal_tier", observed=True):
            m = compute_accuracy_metrics(grp["proj"], grp["actual_fp"])
            tier_rows.append({
                "salary_tier": str(tier), "n": m["n"],
                "avg_proj": round(grp["proj"].mean(), 1),
                "avg_actual": round(grp["actual_fp"].mean(), 1),
                "avg_salary": round(grp["salary"].mean(), 0),
                "mae": m["mae"], "rmse": m["rmse"],
                "r2": m["r2"], "bias": m["bias"],
            })
        result["by_salary_tier"] = pd.DataFrame(tier_rows)

    # By slate
    if "slate_date" in df.columns:
        slate_rows = []
        for sd, grp in df.groupby("slate_date"):
            m = compute_accuracy_metrics(grp["proj"], grp["actual_fp"])
            slate_rows.append({
                "slate_date": sd, "n": m["n"],
                "avg_proj": round(grp["proj"].mean(), 1),
                "avg_actual": round(grp["actual_fp"].mean(), 1),
                "mae": m["mae"], "rmse": m["rmse"],
                "r2": m["r2"], "bias": m["bias"],
            })
        result["by_slate"] = pd.DataFrame(slate_rows)

    return result


# ================================================================
# Calibration Artifact Loader + Projection Calibrator  (Phase 5)
# ================================================================

_CALIB_ARTIFACT_CACHE = None          # module-level cache


def load_calibration_artifact(path=None, root="."):
    """Load a calibration artifact (JSON) at startup.

    The artifact is a dict with keys like:
      {
        "method": "scalar" | "bins" | "isotonic",
        "params": { ... }    # method-specific parameters
      }

    *scalar*  – {"scale": 1.03}  (multiply all proj by scale)
    *bins*    – {"edges": [0,10,20,30,50], "adjustments": [0.95,0.98,1.0,1.05]}
    *isotonic*– {"x": [...], "y": [...]}  (learned monotonic mapping)

    Returns the dict or None if file is missing.
    """
    global _CALIB_ARTIFACT_CACHE
    if _CALIB_ARTIFACT_CACHE is not None:
        return _CALIB_ARTIFACT_CACHE

    if path is None:
        candidates = [
            os.path.join(root, "calibration", "calibration_artifact.json"),
            os.path.join(root, "calibration_artifact.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break

    if path is None or not os.path.exists(path):
        return None

    import json
    with open(path) as fh:
        _CALIB_ARTIFACT_CACHE = json.load(fh)
    return _CALIB_ARTIFACT_CACHE


def apply_calibration_to_projections(proj_df, artifact=None, root="."):
    """Apply calibration mapping to a projection DataFrame.

    Parameters
    ----------
    proj_df : pd.DataFrame
        Must contain a ``proj`` column with raw projections.
    artifact : dict | None
        If None, attempts to auto-load via load_calibration_artifact().
    root : str
        YakOS project root (used for artifact discovery).

    Returns
    -------
    pd.DataFrame  – same schema, ``proj`` column adjusted in-place copy.
    """
    if artifact is None:
        artifact = load_calibration_artifact(root=root)

    df = proj_df.copy()

    if artifact is None:
        # No artifact found – pass-through (identity calibration)
        return df

    method = artifact.get("method", "scalar")
    params = artifact.get("params", {})

    if "proj" not in df.columns:
        return df

    if method == "scalar":
        scale = params.get("scale", 1.0)
        df["proj"] = df["proj"] * scale

    elif method == "bins":
        edges = params.get("edges", [])
        adjustments = params.get("adjustments", [])
        if edges and adjustments:
            bins = pd.cut(df["proj"], bins=edges, labels=False, include_lowest=True)
            for i, adj in enumerate(adjustments):
                mask = bins == i
                df.loc[mask, "proj"] = df.loc[mask, "proj"] * adj

    elif method == "isotonic":
        xs = params.get("x", [])
        ys = params.get("y", [])
        if xs and ys:
            df["proj"] = np.interp(df["proj"], xs, ys)

    return df
