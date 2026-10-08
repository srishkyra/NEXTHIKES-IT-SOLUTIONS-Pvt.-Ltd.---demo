"""Data preparation, customer aggregation, clustering and scoring for the TellCo xDR extract.

Every function is pure (no Streamlit, no global state) so it can be unit-tested and reused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PIPELINE_VERSION = "1.2.0"  # bump whenever cleaning, clustering or scoring changes; stored with every tracked run
RANDOM_STATE = 42
ROUTER = "Huawei B528S-23A"

APPS = {
    "Social Media": ("Social Media DL (Bytes)", "Social Media UL (Bytes)"),
    "Google": ("Google DL (Bytes)", "Google UL (Bytes)"),
    "Email": ("Email DL (Bytes)", "Email UL (Bytes)"),
    "YouTube": ("Youtube DL (Bytes)", "Youtube UL (Bytes)"),
    "Netflix": ("Netflix DL (Bytes)", "Netflix UL (Bytes)"),
    "Gaming": ("Gaming DL (Bytes)", "Gaming UL (Bytes)"),
    "Other": ("Other DL (Bytes)", "Other UL (Bytes)"),
}
APP_COLS = [f"{a} Total (Bytes)" for a in APPS]
IMPUTE_COLS = [
    "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "HTTP DL (Bytes)", "HTTP UL (Bytes)",
    "Avg RTT DL (ms)", "Avg RTT UL (ms)",
]
ENG_METRICS = ["xDR Sessions", "Total Duration (s)", "Total Data Volume (Bytes)"]
EXP_METRICS = ["Avg TCP Retransmission (Bytes)", "Avg RTT (ms)", "Avg Throughput (kbps)"]
MODEL_FEATURES = ENG_METRICS + EXP_METRICS
REQUIRED_RAW_COLUMNS = [
    "Bearer Id", "Start", "End", "Dur. (ms)", "MSISDN/Number", "Handset Manufacturer", "Handset Type",
    "Avg RTT DL (ms)", "Avg RTT UL (ms)", "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)",
    "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "HTTP DL (Bytes)", "HTTP UL (Bytes)",
    "Total DL (Bytes)", "Total UL (Bytes)",
] + [c for pair in APPS.values() for c in pair]


# ── small statistical helpers ────────────────────────────────────────────────
def iqr_flag(s: pd.Series) -> pd.Series:
    """True where a value lies outside the 1.5×IQR fences."""
    q1, q3 = s.quantile([0.25, 0.75])
    i = q3 - q1
    return (s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)


def gini(x) -> float:
    """Gini coefficient of a non-negative array (0 = equal, 1 = one holder)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def order_clusters(km: KMeans, key) -> tuple[dict, np.ndarray]:
    """Renumber clusters by ``key(centres)`` ascending; returns (old→new map, centres in new order)."""
    scores = pd.Series(key(km.cluster_centers_))
    order = scores.sort_values().index.to_list()
    return {old: new for new, old in enumerate(order)}, km.cluster_centers_[order]


def validation_table(X, k_range=range(2, 11)) -> pd.DataFrame:
    """Inertia, silhouette, Davies–Bouldin and Calinski–Harabasz for each k; elbow k in ``attrs``."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=200)
        lab = km.fit_predict(X)
        rows.append({
            "k": k, "Inertia": km.inertia_,
            "Silhouette": silhouette_score(X, lab, sample_size=min(5000, len(X)), random_state=RANDOM_STATE),
            "Davies-Bouldin": davies_bouldin_score(X, lab),
            "Calinski-Harabasz": calinski_harabasz_score(X, lab),
        })
    t = pd.DataFrame(rows).set_index("k")
    kk = t.index.values.astype(float)
    xn = (kk - kk.min()) / (kk.max() - kk.min())
    yn = (t["Inertia"].values - t["Inertia"].min()) / (t["Inertia"].max() - t["Inertia"].min())
    t.attrs["elbow_k"] = int(kk[np.abs(xn + yn - 1).argmax()])
    return t


# ── data loading and documentation ───────────────────────────────────────────
def read_raw(path: str | Path) -> pd.DataFrame:
    """Read the xDR extract (.xlsx or .csv)."""
    path = Path(path)
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def variable_dictionary(raw: pd.DataFrame) -> pd.DataFrame:
    """Column, pandas dtype, analytic role, non-null count, unique count and the official description."""
    desc = pd.read_csv(Path(__file__).parent / "data" / "field_descriptions.csv")
    def norm(s) -> str:
        return " ".join(str(s).strip().lower().split())

    lookup = {norm(f): d for f, d in zip(desc["Fields"], desc["Description"], strict=True)}

    def role(c: str) -> str:
        if c in ("Bearer Id", "IMSI", "MSISDN/Number", "IMEI"):
            return "Identifier"
        if c in ("Start", "End", "Start ms", "End ms") or c.startswith("Dur."):
            return "Time / duration"
        if c in ("Handset Manufacturer", "Handset Type", "Last Location Name"):
            return "Categorical"
        if any(t in c for t in ("RTT", "TP ", "TCP", "Retrans", "Nb of sec")):
            return "Network performance"
        if any(c.startswith(f"{p} ") for p in ("Social Media", "Google", "Email", "Youtube", "Netflix", "Gaming", "Other")):
            return "Application volume"
        if c.startswith(("HTTP", "Total")):
            return "Traffic volume"
        return "Other"

    rows = []
    for c in raw.columns:
        rows.append({
            "Variable": c, "Data type": str(raw[c].dtype), "Role": role(c),
            "Non-null": int(raw[c].notna().sum()), "Missing %": float(raw[c].isna().mean() * 100),
            "Unique": int(raw[c].nunique()), "Description": lookup.get(norm(c), "—"),
        })
    return pd.DataFrame(rows)


def audit_raw(raw: pd.DataFrame) -> dict:
    """Quality facts measured on the raw extract, before any cleaning."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Source file is missing required column(s): " + ", ".join(missing))
    msisdn_na = raw["MSISDN/Number"].isna()
    span = (pd.to_datetime(raw["End"]) - pd.to_datetime(raw["Start"])).dt.total_seconds()
    dur = raw["Dur. (ms)"]
    ok = dur.notna() & span.notna()
    tot = raw["Total DL (Bytes)"].fillna(0) + raw["Total UL (Bytes)"].fillna(0)
    diff = float((span[ok] - dur[ok]).abs().median())
    return {
        "n_raw": len(raw), "n_cols": raw.shape[1],
        "full_duplicates": int(raw.duplicated().sum()),
        "bearer_repeats": int(raw["Bearer Id"].duplicated().sum()),
        "bearer_blank": int(raw["Bearer Id"].astype(str).str.strip().isin(["", "nan"]).sum()),
        "msisdn_missing": int(msisdn_na.sum()),
        "traffic_share_dropped": float(tot[msisdn_na].sum() / tot.sum()),
        "dur_vs_span_median_abs_diff_s": diff, "dur_label_is_seconds": bool(diff < 2),
        "start_min": raw["Start"].min(), "start_max": raw["Start"].max(),
        "end_min": raw["End"].min(), "end_max": raw["End"].max(),
        "end_days": int(pd.to_datetime(raw["End"]).dt.date.nunique()),
        "share_24h_exact": float((span[ok] == 86400).mean()),
        "share_over_24h": float((span[ok] > 86400).mean()),
        "missing_pct": raw.isna().mean().mul(100).sort_values(ascending=False),
    }


# ── session-level cleaning ───────────────────────────────────────────────────
def clean_sessions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename duration, drop unattributable rows, recode unknown handsets, mean-impute network metrics."""
    df = raw.rename(columns={"Dur. (ms)": "Dur. (s)"}).drop(columns=["Dur. (ms).1"], errors="ignore")
    df = df.dropna(subset=["MSISDN/Number"])
    df = df.dropna(subset=["Total DL (Bytes)", "Total UL (Bytes)", "Dur. (s)"]).copy()
    for c in ["Handset Type", "Handset Manufacturer"]:
        und = df[c].astype(str).str.strip().str.lower() == "undefined"
        df.loc[und, c] = "Unknown"
        df[c] = df[c].fillna("Unknown")

    recon = []
    for label, side, n_apps, tot in [("DL, 6 apps (excl. Other)", 0, 6, "Total DL (Bytes)"),
                                     ("DL, 7 apps (incl. Other)", 0, 7, "Total DL (Bytes)"),
                                     ("UL, 6 apps (excl. Other)", 1, 6, "Total UL (Bytes)"),
                                     ("UL, 7 apps (incl. Other)", 1, 7, "Total UL (Bytes)")]:
        cols = [list(APPS.values())[i][side] for i in range(n_apps)]
        ratio = df[cols].sum(axis=1) / df[tot]
        recon.append({"Comparison": label, "Mean ratio to total": ratio.mean(),
                      "Rows matching exactly (%)": (ratio.round(6) == 1).mean() * 100})

    flags = {c: df[c].isna() for c in IMPUTE_COLS}
    log = []
    for c in IMPUTE_COLS:
        med, mean = df[c].median(), df[c].mean()
        log.append({"Column": c, "Missing": int(flags[c].sum()), "Missing %": flags[c].mean() * 100,
                    "Observed median": med, "Fill value (mean)": mean,
                    "Fill ÷ median": mean / med if med else np.nan})
        df[c] = df[c].fillna(mean)
    for c in ["Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]:
        df[c] = df[c].fillna(df[c].mean())

    for a, (d, u) in APPS.items():
        df[f"{a} Total (Bytes)"] = df[d] + df[u]
    df["TCP (Bytes)"] = df["TCP DL Retrans. Vol (Bytes)"] + df["TCP UL Retrans. Vol (Bytes)"]
    df["RTT (ms)"] = df[["Avg RTT DL (ms)", "Avg RTT UL (ms)"]].mean(axis=1)
    df["Throughput (kbps)"] = df[["Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]].mean(axis=1)
    df["tcp_imputed"] = flags[IMPUTE_COLS[0]] | flags[IMPUTE_COLS[1]]
    df["rtt_imputed"] = flags[IMPUTE_COLS[4]] | flags[IMPUTE_COLS[5]]
    df["Start"], df["End"] = pd.to_datetime(df["Start"]), pd.to_datetime(df["End"])
    return df, {"reconciliation": pd.DataFrame(recon), "imp_log": pd.DataFrame(log)}


def customer_table(df: pd.DataFrame) -> pd.DataFrame:
    """Task 1.1: sessions, duration, DL, UL and per-application volume for each customer."""
    g = df.groupby("MSISDN/Number")
    uo = pd.DataFrame({"xDR Sessions": g.size(), "Total Duration (s)": g["Dur. (s)"].sum(),
                       "Total DL (Bytes)": g["Total DL (Bytes)"].sum(), "Total UL (Bytes)": g["Total UL (Bytes)"].sum()})
    for c in APP_COLS:
        uo[c] = g[c].sum()
    uo["Total Data Volume (Bytes)"] = uo["Total DL (Bytes)"] + uo["Total UL (Bytes)"]
    return uo


def treat_outliers(uo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Replace IQR outliers with the mean of the non-outliers (brief-mandated). Returns (table, log, any-flag)."""
    treat_cols = [c for c in uo.columns if c != "Total Data Volume (Bytes)"]
    ut = uo.copy()
    ut[treat_cols] = ut[treat_cols].astype(float)
    any_flag = pd.Series(False, index=uo.index)
    rows = []
    for c in treat_cols:
        f = iqr_flag(uo[c])
        any_flag |= f
        repl = uo.loc[~f, c].mean()
        val = float(round(repl)) if c == "xDR Sessions" else repl
        ut.loc[f, c] = val
        rows.append({"Variable": c, "Outliers replaced": int(f.sum()), "Outliers %": f.mean() * 100,
                     "Replacement (mean of non-outliers)": val})
    ut["Total Data Volume (Bytes)"] = ut["Total DL (Bytes)"] + ut["Total UL (Bytes)"]
    ut["xDR Sessions"] = ut["xDR Sessions"].astype(int)
    return ut, pd.DataFrame(rows), any_flag


def cluster_engagement(ut: pd.DataFrame, k: int = 3) -> dict:
    """k-means on standardised engagement metrics; clusters ordered low→high; adds Engagement Score."""
    sc = StandardScaler().fit(ut[ENG_METRICS])
    X = sc.transform(ut[ENG_METRICS])
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=300).fit(X)
    remap, centers = order_clusters(km, lambda c: c.sum(axis=1))
    labels = pd.Series(km.labels_, index=ut.index).map(remap)
    score = np.linalg.norm(X - centers[0], axis=1)
    return {"scaler": sc, "X": X, "labels": labels, "centers": centers, "score": pd.Series(score, index=ut.index)}


def experience_table(df: pd.DataFrame) -> pd.DataFrame:
    """Task 3.1: per-customer mean TCP, RTT, throughput, modal handset, and imputation shares."""
    gs = df.groupby("MSISDN/Number")
    ex = pd.DataFrame({"Avg TCP Retransmission (Bytes)": gs["TCP (Bytes)"].mean(),
                       "Avg RTT (ms)": gs["RTT (ms)"].mean(),
                       "Avg Throughput (kbps)": gs["Throughput (kbps)"].mean()})

    def modal(col):
        cnt = df.groupby(["MSISDN/Number", col]).size().reset_index(name="n")
        cnt = cnt.sort_values(["MSISDN/Number", "n", col], ascending=[True, False, True])
        return cnt.drop_duplicates("MSISDN/Number").set_index("MSISDN/Number")[col]

    ex["Most Frequent Handset"] = modal("Handset Type")
    ex["Most Frequent Manufacturer"] = modal("Handset Manufacturer")
    ex["TCP imputed share"] = gs["tcp_imputed"].mean()
    ex["RTT imputed share"] = gs["rtt_imputed"].mean()
    return ex


def treat_experience(ex: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """IQR outliers in the three experience metrics → mean of non-outliers. Returns (table, customers changed)."""
    et = ex.copy()
    flags = pd.DataFrame(index=ex.index)
    for c in EXP_METRICS:
        f = iqr_flag(ex[c])
        flags[c] = f
        et[c] = ex[c].where(~f, ex.loc[~f, c].mean())
    return et, int(flags.any(axis=1).sum())


def cluster_experience(et: pd.DataFrame, k: int = 3) -> dict:
    """k-means on standardised experience metrics; ordered best→worst; adds Experience Score."""
    sc = StandardScaler().fit(et[EXP_METRICS])
    X = sc.transform(et[EXP_METRICS])
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=300).fit(X)
    remap, centers = order_clusters(km, lambda c: c[:, 0] + c[:, 1] - c[:, 2])
    labels = pd.Series(km.labels_, index=et.index).map(remap)
    cent = et.assign(_c=labels).groupby("_c")[EXP_METRICS].mean()
    poor_lat = int(cent.loc[[1, 2], "Avg RTT (ms)"].idxmax())
    names = {0: "Good experience", poor_lat: "High / unmeasured latency", 3 - poor_lat: "Low throughput"}
    score = np.linalg.norm(X - centers[2], axis=1)
    return {"scaler": sc, "X": X, "labels": labels, "centers": centers, "names": names,
            "score": pd.Series(score, index=et.index)}


def satisfaction_table(ut, eng, et, expc) -> dict:
    """Task 4: combine scores, min–max scale, average, then k=2 clustering."""
    comb = ut.copy()
    comb["Engagement Cluster"] = eng["labels"]
    comb["Engagement Score"] = eng["score"]
    comb = comb.join(et[EXP_METRICS + ["Most Frequent Handset", "Most Frequent Manufacturer",
                                       "TCP imputed share", "RTT imputed share"]], how="inner")
    comb["Experience Cluster"] = expc["labels"]
    comb["Experience Score"] = expc["score"]
    # Task 4.2 (assignment definition): satisfaction = average of the engagement and experience scores.
    # Primary definition: min-max scale each score to [0, 1] so neither dominates, then average.
    comb["Satisfaction Score (raw mean)"] = comb[["Engagement Score", "Experience Score"]].mean(axis=1)
    mm = MinMaxScaler().fit(comb[["Engagement Score", "Experience Score"]])
    comb[["Eng Score Scaled", "Exp Score Scaled"]] = mm.transform(comb[["Engagement Score", "Experience Score"]])
    comb["Satisfaction Score"] = comb[["Eng Score Scaled", "Exp Score Scaled"]].mean(axis=1)
    comb["Satisfaction Score (scaled)"] = comb["Satisfaction Score"]  # alias so existing references still work
    sc = StandardScaler().fit(comb[["Engagement Score", "Experience Score"]])
    X = sc.transform(comb[["Engagement Score", "Experience Score"]])
    km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20, max_iter=300).fit(X)
    eng_led = int(pd.DataFrame(km.cluster_centers_).iloc[:, 0].idxmax())
    smap = {eng_led: 0, 1 - eng_led: 1}
    comb["Satisfaction Cluster"] = pd.Series(km.labels_, index=comb.index).map(smap)
    sil = silhouette_score(X, comb["Satisfaction Cluster"], sample_size=min(5000, len(X)), random_state=RANDOM_STATE)
    return {"comb": comb, "mm": mm, "scaler": sc, "km": km, "map": smap, "silhouette": sil}


# ── end-to-end pipeline ──────────────────────────────────────────────────────
def run_pipeline(raw: pd.DataFrame) -> dict:
    """Run Tasks 1–4 end to end and return every table the dashboard and exports need."""
    q = audit_raw(raw)
    df, extra = clean_sessions(raw)
    imp_log = extra.pop("imp_log")
    q.update(extra)
    q["n_sessions"] = len(df)
    q["start_before_window"] = int((df["Start"] < df["End"].min().normalize() - pd.Timedelta(days=1)).sum())
    q["unknown_handset_sessions"] = int((df["Handset Type"] == "Unknown").sum())

    sessions = df[["MSISDN/Number", "Start", "End", "Dur. (s)", "Handset Type", "Handset Manufacturer",
                   "Total DL (Bytes)", "Total UL (Bytes)", "TCP (Bytes)", "RTT (ms)", "Throughput (kbps)",
                   "tcp_imputed", "rtt_imputed"] + APP_COLS].copy()

    uo_raw = customer_table(df)
    ut, treat_log, any_flag = treat_outliers(uo_raw)
    eng = cluster_engagement(ut)
    ut["Engagement Cluster"] = eng["labels"]
    ut["Engagement Score"] = eng["score"]
    long_hold_is_mid = int(eng["centers"][:, 1].argmax()) == 1
    eng_names = {0: "Low engagement",
                 1: "Long-hold, light traffic" if long_hold_is_mid else "Medium engagement",
                 2: "High engagement"}
    val_eng = validation_table(eng["X"])
    km2 = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20, max_iter=300).fit(eng["X"])
    ari_k2_k3 = adjusted_rand_score(eng["labels"], km2.labels_)

    ex_raw = experience_table(df)
    et, n_ex_treated = treat_experience(ex_raw)
    expc = cluster_experience(et)
    et["Experience Cluster"] = expc["labels"]
    et["Experience Score"] = expc["score"]
    val_exp = validation_table(expc["X"])

    sat = satisfaction_table(ut, eng, et, expc)
    comb = sat["comb"]
    sat_names = {0: "Engaged, weaker experience", 1: "Less engaged, better experience"}
    comb["Engagement Tier"] = comb["Engagement Cluster"].map(eng_names)
    comb["Experience Group"] = comb["Experience Cluster"].map(expc["names"])
    comb["Satisfaction Group"] = comb["Satisfaction Cluster"].map(sat_names)
    vmed, smed = comb["Total Data Volume (Bytes)"].median(), comb["Satisfaction Score"].median()
    comb["Traffic Tier"] = np.where(comb["Total Data Volume (Bytes)"] >= vmed, "High traffic", "Low traffic")
    comb["Proxy Score Tier"] = np.where(comb["Satisfaction Score"] >= smed, "High proxy score", "Low proxy score")
    comb.index = comb.index.astype("int64")
    comb.index.name = "MSISDN/Number"

    scoring = {
        "eng_mean": eng["scaler"].mean_, "eng_scale": eng["scaler"].scale_, "eng_ref": eng["centers"][0],
        "exp_mean": expc["scaler"].mean_, "exp_scale": expc["scaler"].scale_, "exp_ref": expc["centers"][2],
        "mm_min": sat["mm"].data_min_, "mm_max": sat["mm"].data_max_,
        "sat_mean": sat["scaler"].mean_, "sat_scale": sat["scaler"].scale_,
        "sat_centers": sat["km"].cluster_centers_, "sat_map": sat["map"],
        "eng_centers": eng["centers"], "exp_centers": expc["centers"],
        "eng_p99": ut[ENG_METRICS].quantile(0.99).values, "exp_p99": et[EXP_METRICS].quantile(0.99).values,
    }
    return {
        "q": q, "imp_log": imp_log, "treat_log": treat_log, "ex_n_treated": n_ex_treated,
        "n_treated": int(any_flag.sum()),
        "treated_traffic_share": float(uo_raw.loc[any_flag, "Total Data Volume (Bytes)"].sum()
                                       / uo_raw["Total Data Volume (Bytes)"].sum()),
        "sessions": sessions, "uo_raw": uo_raw, "ut": ut, "ex_raw": ex_raw, "et": et, "comb": comb,
        "val_eng": val_eng, "val_exp": val_exp, "ari_k2_k3": ari_k2_k3,
        "eng_names": eng_names, "exp_names": expc["names"], "sat_names": sat_names,
        "eng_centers": eng["centers"], "exp_centers": expc["centers"], "sat_sil": sat["silhouette"],
        "scoring": scoring, "long_hold_is_mid": long_hold_is_mid,
        "eng_X": eng["X"], "exp_X": expc["X"],
        "pipeline_version": PIPELINE_VERSION,
        "exp_profile": describe_experience_clusters(et, expc["names"]),
        "fingerprints": {"engagement": label_fingerprint(ut["Engagement Cluster"]),
                         "experience": label_fingerprint(et["Experience Cluster"]),
                         "satisfaction": label_fingerprint(comb["Satisfaction Cluster"])},
    }


def score_new_customer(sc: dict, vals_eng, vals_exp) -> dict:
    """Score one hypothetical customer with the fitted scalers and centroids (same maths as the pipeline)."""
    ze = (np.asarray(vals_eng, float) - sc["eng_mean"]) / sc["eng_scale"]
    zx = (np.asarray(vals_exp, float) - sc["exp_mean"]) / sc["exp_scale"]
    eng = float(np.linalg.norm(ze - sc["eng_ref"]))
    exp = float(np.linalg.norm(zx - sc["exp_ref"]))
    span = np.where(sc["mm_max"] - sc["mm_min"] == 0, 1, sc["mm_max"] - sc["mm_min"])
    eng_s, exp_s = np.clip((np.array([eng, exp]) - sc["mm_min"]) / span, 0, 1)
    zs = (np.array([eng, exp]) - sc["sat_mean"]) / sc["sat_scale"]
    cl = int(np.argmin(((sc["sat_centers"] - zs) ** 2).sum(axis=1)))
    return {"eng": eng, "exp": exp, "sat": float((eng_s + exp_s) / 2), "sat_scaled": float((eng_s + exp_s) / 2),
            "sat_cluster": sc["sat_map"][cl],
            "eng_cluster": int(np.argmin(((sc["eng_centers"] - ze) ** 2).sum(axis=1))),
            "exp_cluster_raw": int(np.argmin(((sc["exp_centers"] - zx) ** 2).sum(axis=1)))}


# ── robustness and sensitivity analyses ──────────────────────────────────────
def seed_stability(P: dict, seeds=(0, 1, 42, 100, 2024)) -> pd.DataFrame:
    """Adjusted Rand index of each seed's clustering against the primary (seed 42) run, for both clusterings."""
    rows = []
    specs = [("Engagement (k=3)", P["eng_X"], P["ut"]["Engagement Cluster"].values),
             ("Experience (k=3)", P["exp_X"], P["et"]["Experience Cluster"].values)]
    for name, X, ref in specs:
        for sd in seeds:
            km = KMeans(n_clusters=3, random_state=sd, n_init=20, max_iter=300).fit(X)
            rows.append({"Clustering": name, "Seed": sd, "Inertia": km.inertia_,
                         "ARI vs primary": adjusted_rand_score(ref, km.labels_),
                         "Smallest cluster %": np.bincount(km.labels_).min() / len(X) * 100})
    return pd.DataFrame(rows)


def weighting_sensitivity(comb: pd.DataFrame) -> dict:
    """How much the satisfaction proxy and its top-10 list move under alternative definitions and weightings."""
    sv = pd.DataFrame({
        "Scaled 50/50 (primary)": comb["Satisfaction Score"],
        "Scaled 60/40 (engagement-heavy)": 0.6 * comb["Eng Score Scaled"] + 0.4 * comb["Exp Score Scaled"],
        "Scaled 40/60 (experience-heavy)": 0.4 * comb["Eng Score Scaled"] + 0.6 * comb["Exp Score Scaled"],
    })
    names = list(sv.columns)
    tops = {n: set(sv[n].nlargest(10).index) for n in names}
    overlap = pd.DataFrame([[len(tops[a] & tops[b]) for b in names] for a in names], index=names, columns=names)
    r_a, r_b = sv[names[0]].rank(ascending=False), sv[names[-1]].rank(ascending=False)
    return {"spearman": sv.corr(method="spearman"), "top10_overlap": overlap,
            "mean_rank_change": float((r_a - r_b).abs().mean()), "max_rank_change": float((r_a - r_b).abs().max()),
            "primary_vs": names[-1]}


def alt_experience_scores(P: dict) -> pd.DataFrame:
    """Two cluster-free experience scores and how closely they track the primary proxy."""
    from scipy import stats
    comb, sc = P["comb"], P["scoring"]
    z = (P["et"].loc[comb.index.astype(float), EXP_METRICS].values - sc["exp_mean"]) / sc["exp_scale"]
    z = pd.DataFrame(z, index=comb.index, columns=EXP_METRICS)
    alts = {"Directional composite (no clusters)": -(z.iloc[:, 0] + z.iloc[:, 1] - z.iloc[:, 2]),
            "Throughput only": z.iloc[:, 2]}
    med = comb["Total Data Volume (Bytes)"].median()
    base = comb["Satisfaction Score (scaled)"]
    base_q = (comb["Total Data Volume (Bytes)"] >= med) & (base < base.median())
    top = set(base.nlargest(1000).index)
    rows = []
    for name, s in alts.items():
        s_sc = (s - s.min()) / (s.max() - s.min())
        sat = 0.5 * comb["Eng Score Scaled"] + 0.5 * s_sc
        qd = (comb["Total Data Volume (Bytes)"] >= med) & (sat < sat.median())
        rows.append({"Alternative experience score": name,
                     "Spearman (experience score)": stats.spearmanr(comb["Experience Score"], s)[0],
                     "Spearman (proxy score)": stats.spearmanr(base, sat)[0],
                     "Top-1000 overlap": len(top & set(sat.nlargest(1000).index)),
                     "Priority-quadrant overlap (%)": (qd & base_q).sum() / base_q.sum() * 100})
    return pd.DataFrame(rows).set_index("Alternative experience score")


def capped_variant(P: dict) -> dict:
    """Robustness: clip (winsorise) outliers instead of replacing them with the mean, then recluster."""
    uo = P["uo_raw"].copy()
    for c in [c for c in uo.columns if c != "Total Data Volume (Bytes)"]:
        q1, q3 = uo[c].quantile([0.25, 0.75])
        i = q3 - q1
        uo[c] = uo[c].clip(q1 - 1.5 * i, q3 + 1.5 * i)
    uo["Total Data Volume (Bytes)"] = uo["Total DL (Bytes)"] + uo["Total UL (Bytes)"]
    res = cluster_engagement(uo)
    ari = adjusted_rand_score(P["ut"]["Engagement Cluster"], res["labels"].reindex(P["ut"].index))
    sizes = res["labels"].value_counts(normalize=True).sort_index() * 100
    return {"ari": float(ari), "sizes": sizes, "gini": gini(uo["Total Data Volume (Bytes)"])}


def manufacturer_app_mix(sessions: pd.DataFrame, manufacturers: list[str]) -> pd.DataFrame:
    """Share of each application's counter within each manufacturer's traffic (rows sum to 100)."""
    sub = sessions[sessions["Handset Manufacturer"].isin(manufacturers)]
    mix = sub.groupby("Handset Manufacturer")[APP_COLS].sum()
    mix = mix.div(mix.sum(axis=1), axis=0) * 100
    mix.columns = list(APPS)
    return mix.loc[[m for m in manufacturers if m in mix.index]]


def describe_experience_clusters(et: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Centroid table in original units with a computed one-line interpretation for each experience cluster."""
    g = et.groupby("Experience Cluster")
    prof = pd.DataFrame({
        "Group": pd.Series(names), "Customers": g.size(), "Share (%)": g.size() / len(et) * 100,
        "Mean TCP retransmission (bytes)": g["Avg TCP Retransmission (Bytes)"].mean(),
        "Mean RTT (ms)": g["Avg RTT (ms)"].mean(),
        "Mean throughput (kbps)": g["Avg Throughput (kbps)"].mean(),
        "RTT fully imputed (%)": g["RTT imputed share"].apply(lambda x: (x == 1).mean() * 100),
        "Router customers (%)": g["Most Frequent Handset"].apply(lambda x: (x == ROUTER).mean() * 100),
    })
    base = {"rtt": et["Avg RTT (ms)"].median(), "tp": et["Avg Throughput (kbps)"].median()}
    notes = []
    for _, r in prof.iterrows():
        tp_x = r["Mean throughput (kbps)"] / base["tp"] if base["tp"] else np.nan
        rt_x = r["Mean RTT (ms)"] / base["rtt"] if base["rtt"] else np.nan
        notes.append(f"Throughput {tp_x:.1f}× and RTT {rt_x:.1f}× the all-customer median; "
                     f"{r['RTT fully imputed (%)']:.0f}% have no measured RTT")
    prof["Interpretation (computed)"] = notes
    return prof


def label_fingerprint(labels: pd.Series) -> str:
    """Short hash of a cluster labelling — identical across runs iff the clustering reproduced exactly."""
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(labels.sort_index().values.astype("int8")).tobytes()).hexdigest()[:12]
