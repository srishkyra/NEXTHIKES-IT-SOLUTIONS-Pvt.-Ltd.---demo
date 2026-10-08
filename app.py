"""
TellCo — User Analytics Dashboard
NextHikes IT Solutions
"""

import hashlib
import importlib.metadata as md
import itertools
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tellco import db as tdb
from tellco import feature_store as tfs
from tellco import modeling as tmod
from tellco import pipeline as tpl
from tellco import tracking as ttrack
from tellco.pipeline import (
    APP_COLS,
    APPS,
    ENG_METRICS,
    EXP_METRICS,
    MODEL_FEATURES,
    RANDOM_STATE,
    ROUTER,
    gini,
    score_new_customer,
)

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + PALETTE
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TellCo — User Analytics Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BERRY = "#6B212C"
WINE = "#78211E"
HOLLY = "#616852"
PINE = "#27363F"
CINNAMON = "#C0763B"
SLATE = "#527783"
MIST = "#A9B4B0"
SAND = "#E8D9C5"
CATEGORY_PALETTE = [BERRY, SLATE, HOLLY, CINNAMON, PINE, MIST, WINE, SAND]
TIER_COLOURS = {0: SLATE, 1: CINNAMON, 2: BERRY}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #F8F9FA; }}
    h1, h2, h3 {{ color: {PINE} !important; }}
    div[data-testid="stMetricValue"] {{ color: {PINE}; font-weight: 700; }}
    div[data-testid="stMetricLabel"] {{ color: #666666; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #F0F0F0; border-radius: 6px 6px 0 0; padding: 8px 14px; font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {PINE}; color: white; }}
    section[data-testid="stSidebar"] {{ background-color: #F8F9FA; }}
    div[role="radiogroup"] {{ gap: 6px; flex-wrap: wrap; }}
    div[role="radiogroup"] > label {{ background: #F0F0F0; border-radius: 6px; padding: 6px 12px; margin: 0; }}
    div[role="radiogroup"] > label:has(input:checked) {{ background: {PINE}; }}
    div[role="radiogroup"] > label:has(input:checked) p {{ color: white !important; font-weight: 700; }}
    div[role="radiogroup"] > label > div:first-child {{ display: none; }}
    .verdict {{ border-left: 6px solid {BERRY}; background: #FFFFFF; padding: 14px 18px;
               border-radius: 6px; margin: 6px 0 14px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

LAYOUT_KWARGS = dict(
    paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
    font=dict(color=PINE, family="sans-serif"),
    title_font=dict(size=16, color=PINE),
    colorway=CATEGORY_PALETTE,
    margin=dict(t=60, b=40, l=40, r=20),
)

# Unique key per chart per run (the script reruns top-to-bottom on every interaction,
# so a fresh counter each run keeps keys stable and collision-free).
_chart_ids = itertools.count()

# Friendly axis / colour-bar labels with units, applied wherever a chart exposes a raw column name.
AXIS_LABELS = {
    "xDR Sessions": "xDR sessions per customer (count)",
    "Total Duration (s)": "Total session duration (seconds)",
    "Total Data Volume (Bytes)": "Total data volume (bytes)",
    "Total DL (Bytes)": "Total download (bytes)", "Total UL (Bytes)": "Total upload (bytes)",
    "Avg TCP Retransmission (Bytes)": "Avg TCP retransmission (bytes)",
    "Avg RTT (ms)": "Avg round-trip time (ms)", "Avg Throughput (kbps)": "Avg throughput (kbps)",
    "Engagement Score": "Engagement score (distance from least-engaged centre, std. units)",
    "Experience Score": "Experience score (distance from worst-experience centre, std. units)",
    "Satisfaction Score": "Satisfaction score (mean of the two min–max-scaled scores, 0–1)",
    "Satisfaction Score (scaled)": "Satisfaction score, min–max scaled (0–1)",
    "count": "Number of customers", "value": "Value", "variable": "Series",
    "Sessions": "xDR sessions (count)", "Total_DL_UL": "Total DL+UL data (bytes)",
}
AXIS_LABELS.update({f"{a} Total (Bytes)": f"{a} data volume (bytes)" for a in APPS})


def _audit(fig):
    """Append a one-line quality record for this chart to the file named in TELLCO_AUDIT (testing aid only)."""
    types = {t.type for t in fig.data}
    cart = bool(types - {"pie", "heatmap", "sunburst", "treemap"})
    axes = {n: (fig.layout[n].title.text if fig.layout[n].title is not None else None) for n in _axis_names(fig)
            if n in ("xaxis", "yaxis") or fig.layout[n].title.text}
    rec = {"section": st.session_state.get("nav"), "title": fig.layout.title.text if fig.layout.title else None,
           "types": sorted(types), "cartesian": cart, "axes": axes, "n_traces": len(fig.data),
           "legend_title": fig.layout.legend.title.text if fig.layout.legend and fig.layout.legend.title else None,
           "showlegend_traces": sum(1 for t in fig.data if t.showlegend is not False)}
    with open(os.environ["TELLCO_AUDIT"], "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")



def _axis_names(fig):
    return [k for k in fig.layout if k.startswith(("xaxis", "yaxis"))]


def polish(fig, x=None, y=None, legend=None):
    """Apply labels, units and legible legends. Explicit ``x``/``y`` override; raw column names are translated."""
    for name, override in (("xaxis", x), ("yaxis", y)):
        if override is not None:
            fig.layout[name].title.text = override
    for name in _axis_names(fig):
        ax = fig.layout[name]
        txt = ax.title.text if ax.title is not None else None
        if txt in AXIS_LABELS:
            ax.title.text = AXIS_LABELS[txt]
        ax.title.font = dict(size=12)
        ax.tickfont = dict(size=11)
    for tr in fig.data:
        if tr.name in AXIS_LABELS and tr.type in ("scatter", "bar", "box", "histogram"):
            tr.name = AXIS_LABELS[tr.name]
    fig.update_layout(legend=dict(font=dict(size=12), itemsizing="constant", bgcolor="rgba(0,0,0,0)"),
                      coloraxis_colorbar=dict(title=dict(font=dict(size=11))))
    if legend is not None:
        fig.update_layout(legend_title_text=legend)
    return fig


def show(fig, height=None, x=None, y=None, legend=None, note=None, cbar=None):
    """Render a Plotly chart with consistent styling, axis labels with units and an optional interpretation line."""
    fig.update_layout(**LAYOUT_KWARGS)
    polish(fig, x, y, legend)
    if cbar:
        fig.update_layout(coloraxis_colorbar_title_text=cbar)
    if height:
        fig.update_layout(height=height)
    if os.getenv("TELLCO_AUDIT"):
        _audit(fig)
    st.plotly_chart(fig, width="stretch", key=f"chart_{next(_chart_ids)}")
    if note:
        st.caption(f"💡 {note}")


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent if (APP_DIR.parent / "pyproject.toml").exists() else APP_DIR
OUT_DIR = Path(os.getenv("TELLCO_OUTPUT_DIR", ROOT / "outputs"))
FS_DIR = Path(os.getenv("TELLCO_FEATURE_STORE", ROOT / "feature_store"))
DOCS_DIR = ROOT / "docs"

# Reference values from the analysis notebook, used ONLY by the verification panel.
NOTEBOOK_REFERENCE = {
    "Sessions kept after cleaning": 148_935,
    "Customers": 106_856,
    "Customers with ≥1 treated variable": 16_033,
    "Engagement cluster sizes (Low / Long-hold / High)": [57_834, 24_654, 24_368],
    "Experience cluster shares % (Good / Low throughput / High-unmeasured latency)": [34.56, 40.38, 25.06],
    "High-traffic / Low-proxy-score share % (scaled variant)": 19.23,
    "Router median throughput, untreated (kbps)": 19571,
    "Kruskal–Wallis ε², top-10 handsets, untreated": 0.328,
}
# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────


def fmt_bytes(x: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(x) < 1000:
            return f"{x:,.1f} {unit}"
        x /= 1000
    return f"{x:,.1f} EB"


@st.cache_data(show_spinner="Reading the xDR export (first load only; later loads use a local cache)...")
def load_raw(path: str, size: int, mtime_ns: int) -> pd.DataFrame:
    """Read the Excel export, with a pickle cache next to it so reloads take seconds."""
    p = Path(path)
    cache = p.with_suffix(".cache.pkl")
    if cache.exists() and cache.stat().st_mtime_ns >= mtime_ns:
        try:
            return pd.read_pickle(cache)
        except Exception:
            pass
    df = pd.read_excel(p)
    try:
        df.to_pickle(cache)
    except OSError:
        pass
    return df


@st.cache_data(show_spinner="Reading uploaded file...")
def load_uploaded(data: bytes) -> pd.DataFrame:
    import io
    return pd.read_excel(io.BytesIO(data))


def find_data_file():
    env = os.getenv("TELLCO_DATA")
    if env and Path(env).exists():
        return Path(env)
    for folder in (ROOT, APP_DIR, ROOT / "data", Path.cwd()):
        hits = sorted(folder.glob("telcom_data*.xlsx"))
        if hits:
            return hits[0]
    return None



@st.cache_resource(show_spinner="Cleaning data, building customer tables and fitting clusters...")
def run_pipeline(_raw: pd.DataFrame, data_key: str) -> dict:
    return tpl.run_pipeline(_raw)


@st.cache_data(show_spinner=False)
def variable_dictionary(_raw: pd.DataFrame, data_key: str) -> pd.DataFrame:
    return tpl.variable_dictionary(_raw)


@st.cache_resource(show_spinner="Running robustness checks (cluster stability across seeds, capped variant)...")
def robustness(_P: dict, data_key: str) -> dict:
    return {"seeds": tpl.seed_stability(_P), "capped": tpl.capped_variant(_P),
            "weights": tpl.weighting_sensitivity(_P["comb"]), "alt_exp": tpl.alt_experience_scores(_P)}


@st.cache_resource(show_spinner="Training and comparing regression models, then computing SHAP (about a minute)...")
def train_models(_comb: pd.DataFrame, data_key: str) -> dict:
    return tmod.train_models(_comb, data_key)


def latest_saved_model():
    """Newest saved model, or (None, None) if there is none or it cannot be loaded in this environment."""
    files = sorted((OUT_DIR / "models").glob("*.joblib")) if (OUT_DIR / "models").exists() else []
    st.session_state["model_load_error"] = ""
    if not files:
        return None, None
    try:
        return tmod.load_model(files[-1]), files[-1].name
    except tmod.IncompatibleModelError as exc:
        st.session_state["model_load_error"] = str(exc)
        return None, None


# ──────────────────────────────────────────────────────────────────────────
# DELIVERY CHECKLIST — every status is derived from files and tools that actually exist
# ──────────────────────────────────────────────────────────────────────────
def delivery_status() -> pd.DataFrame:
    rows = []

    def add(item, ok, evidence, action=""):
        rows.append({"Deliverable": item, "Status": "✅ Complete" if ok else "⏳ Needs your input",
                     "Evidence": evidence, "To complete": "" if ok else action})

    # reusable data preparation code, installable via pip
    try:
        ver = md.version("tellco-analytics")
        installed = True
    except md.PackageNotFoundError:
        ver, installed = "", False
    add("Reusable, pip-installable package", installed and (ROOT / "pyproject.toml").exists(),
        f"`tellco-analytics` {ver} — modules: pipeline, modeling, tracking, feature_store, db, cli", "Run `pip install -e .`")
    reg = tfs.load_registry(FS_DIR)
    add("Reusable feature store", bool(reg),
        f"{len(reg)} versioned snapshot(s); latest has {reg[-1]['rows']:,} customers × {len(reg[-1]['features'])} features" if reg else "",
        "Run `tellco-run --data <file>`")
    runs = ttrack.list_runs(OUT_DIR)
    add("Model deployment tracking (MLflow)", len(runs) > 0,
        f"{len(runs)} tracked run(s) in `outputs/mlflow.db` with parameters, metrics, loss steps and artifacts" if len(runs) else "",
        "Run `tellco-run --data <file>`")
    mylog = OUT_DIR / "mysql_export.json"
    info = json.loads(mylog.read_text()) if mylog.exists() else None
    add("Export to MySQL with SELECT evidence", bool(info and info.get("dialect") == "mysql"),
        f"{info['rows_in_table']:,} rows in `{info['table']}` ({info['server_version']}) — real SELECT output shown on the Export & Tracking page" if info else "",
        "Run `tellco-run --data <file> --mysql USER:PASSWORD@HOST:PORT/DB`")
    decks = sorted(DOCS_DIR.glob("*.pptx"))
    n_slides = 0
    if decks:
        with zipfile.ZipFile(decks[0]) as z:
            n_slides = len([n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")])
    add("Slide deck (≤ 20 slides)", bool(decks) and 0 < n_slides <= 20, f"`{decks[0].name}` — {n_slides} slides" if decks else "", "Add the deck to docs/")
    reports = (sorted(DOCS_DIR.glob("*Report*.md")) + sorted(DOCS_DIR.glob("*report*.md"))
           + sorted(DOCS_DIR.glob("*Report*.docx")))
    add("Written report", bool(reports), f"`{reports[0].name}`" if reports else "", "Add the report to docs/")
    cfg = json.loads((ROOT / "project.json").read_text()) if (ROOT / "project.json").exists() else {}
    url = str(cfg.get("github_repo", "")).strip()
    add("GitHub links (dashboard code + analysis code)", url.startswith("https://github.com/"),
        f"{url}" if url else "", "Push the project and put the URL in `project.json` → `github_repo`")
    shot = DOCS_DIR / "dashboard_screenshot.png"
    add("Dashboard screenshot", shot.exists(), f"`{shot.name}`" if shot.exists() else "",
        "Save a browser screenshot as `docs/dashboard_screenshot.png`")
    return pd.DataFrame(rows)


def explain_roles(roles: pd.Series) -> str:
    return ", ".join(f"{n} {r.lower()}" for r, n in roles.value_counts().items())


# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA + PIPELINE — every stage guarded so a bad file or environment issue
# surfaces a specific, actionable message instead of crashing the app.
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("📡 TellCo Analytics")
st.sidebar.caption("NextHikes IT Solutions · xDR user analytics")

data_path = find_data_file()
raw, data_key = None, ""
try:
    if data_path is not None:
        stt = data_path.stat()
        raw = load_raw(str(data_path), stt.st_size, stt.st_mtime_ns)
        data_key = f"{data_path.name}:{stt.st_size}:{stt.st_mtime_ns}"
    else:
        up = st.sidebar.file_uploader("Upload the xDR export (.xlsx)", type=["xlsx"])
        if up is None:
            st.title("📡 TellCo — User Analytics")
            st.info("Place `telcom_data*.xlsx` next to `app.py`, or upload the file from the sidebar to begin.")
            st.stop()
        content = up.getvalue()
        raw = load_uploaded(content)
        data_key = f"upload:{hashlib.sha1(content).hexdigest()}"
except ImportError:
    st.error("Reading Excel needs **openpyxl**. Run `pip install openpyxl` and restart the app.")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the data file: {exc}")
    st.stop()

try:
    P = run_pipeline(raw, data_key)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

q, comb, S = P["q"], P["comb"], P["sessions"]
uo_raw, ut, ex_raw, et = P["uo_raw"], P["ut"], P["ex_raw"], P["et"]
ENG, EXPN, SATN = P["eng_names"], P["exp_names"], P["sat_names"]
N_CUST = len(comb)

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS (customer-level views; clusters themselves are always fitted on the full base)
# ──────────────────────────────────────────────────────────────────────────
top_mfr = comb["Most Frequent Manufacturer"].value_counts().head(12).index.tolist()
FILTER_DEFAULTS = {"f_mfr": [], "f_eng": list(ENG.values()), "f_exp": list(EXPN.values()),
                   "f_unknown": False, "f_router": False}


def reset_filters():
    for k, v in FILTER_DEFAULTS.items():
        st.session_state[k] = v


for k, v in FILTER_DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.sidebar.markdown("### Filters")
st.sidebar.multiselect("Handset manufacturer (empty = all)", top_mfr, key="f_mfr")
st.sidebar.multiselect("Engagement tier", list(ENG.values()), key="f_eng")
st.sidebar.multiselect("Experience group", list(EXPN.values()), key="f_exp")
st.sidebar.checkbox("Exclude customers with unknown device", key="f_unknown")
st.sidebar.checkbox(f"Exclude fixed-wireless router ({ROUTER})", key="f_router")
st.sidebar.button("↺ Reset filters", on_click=reset_filters)

mask = comb["Engagement Tier"].isin(st.session_state["f_eng"]) & comb["Experience Group"].isin(st.session_state["f_exp"])
if st.session_state["f_mfr"]:
    mask &= comb["Most Frequent Manufacturer"].isin(st.session_state["f_mfr"])
if st.session_state["f_unknown"]:
    mask &= comb["Most Frequent Handset"] != "Unknown"
if st.session_state["f_router"]:
    mask &= comb["Most Frequent Handset"] != ROUTER
cf = comb[mask]
st.sidebar.markdown(f"**{len(cf):,}** of **{N_CUST:,}** customers selected")
CSV_COLS = ["Engagement Score", "Experience Score", "Satisfaction Score", "Engagement Tier", "Experience Group",
            "Satisfaction Group", "Most Frequent Handset"] + MODEL_FEATURES


@st.cache_data(show_spinner=False)
def filtered_csv(_df: pd.DataFrame, key: str) -> bytes:
    return _df[CSV_COLS].reset_index().to_csv(index=False).encode()


_fkey = f"{data_key}|{len(cf)}|{int(cf.index.values.sum()) if len(cf) else 0}"
st.sidebar.download_button("⬇ Download filtered customers (CSV)", filtered_csv(cf, _fkey),
                           "tellco_filtered_customers.csv", "text/csv")
st.sidebar.caption("Filters apply to the Experience handset view, Satisfaction and Customer Explorer tabs. "
                   "Cluster definitions and top-10 lists always use the full base.")

# ──────────────────────────────────────────────────────────────────────────
# DERIVED HEADLINES (all computed — nothing typed in)
# ──────────────────────────────────────────────────────────────────────────
tot_vol = uo_raw["Total Data Volume (Bytes)"]
eng_share = comb["Engagement Cluster"].value_counts(normalize=True).sort_index() * 100
exp_share = comb["Experience Group"].value_counts(normalize=True) * 100
high_traffic_share = float(ut.loc[ut["Engagement Cluster"] == 2, "Total Data Volume (Bytes)"].sum()
                           / ut["Total Data Volume (Bytes)"].sum() * 100)
app_tot = uo_raw[APP_COLS].sum()
app_tot.index = list(APPS)
app_share = app_tot / app_tot.sum() * 100
gini_raw, gini_trt = gini(uo_raw["Total Data Volume (Bytes)"]), gini(ut["Total Data Volume (Bytes)"])
gini_word = "low" if gini_raw < 0.3 else "moderate" if gini_raw < 0.5 else "high"
app_corr = pd.Series({a: uo_raw[c].corr(uo_raw["Total Data Volume (Bytes)"]) for a, c in zip(APPS, APP_COLS, strict=True)})
top_app = app_corr.idxmax()
top_app_share = float((uo_raw[f"{top_app} Total (Bytes)"] / uo_raw["Total Data Volume (Bytes)"]).mean() * 100)
top10_raw = float(np.sort(tot_vol.values)[::-1][: int(N_CUST * 0.10)].sum() / tot_vol.sum() * 100)
router_sessions = int((S["Handset Type"] == ROUTER).sum())
router_cust = int((comb["Most Frequent Handset"] == ROUTER).sum())
latgrp = [k for k, v in EXPN.items() if v == "High / unmeasured latency"][0]
lat_members = comb[comb["Experience Cluster"] == latgrp]
lat_imputed_share = float((lat_members["RTT imputed share"] == 1).mean() * 100)
prio_share_scaled = float(((comb["Traffic Tier"] == "High traffic") & (comb["Satisfaction Score (scaled)"] < comb["Satisfaction Score (scaled)"].median())).mean() * 100)
prio_share = float(((comb["Traffic Tier"] == "High traffic") & (comb["Proxy Score Tier"] == "Low proxy score")).mean() * 100)
data_days = f"{q['end_min']:%d %b} – {q['end_max']:%d %b %Y}"
best_sil = int(P["val_eng"]["Silhouette"].idxmax())

st.title("📡 TellCo — User Analytics Dashboard")
st.caption(f"{q['n_raw']:,} xDR session rows · {q['n_cols']} columns · session end dates {data_days} "
           f"({q['end_days']} days) · all figures computed live from the loaded file")

SECTIONS = [
    "🏁 Executive Summary", "🧹 Data Quality", "📱 User Overview", "📊 Usage & EDA", "📶 Engagement",
    "🛰️ Experience", "😊 Satisfaction", "🧪 Robustness", "🔎 Customer Explorer", "⚖️ Recommendation & Limits",
    "📦 Export & Tracking",
]
# Only the selected section is rendered, so each interaction stays fast (st.tabs would execute all ten).
section = st.radio("Section", SECTIONS, horizontal=True, key="nav", label_visibility="collapsed")
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{N_CUST:,}")
    c2.metric("xDR sessions (kept)", f"{q['n_sessions']:,}")
    c3.metric("Period covered", f"{q['end_days']} days", help=f"Session end dates {data_days}")
    c4.metric("Avg data / customer", f"{tot_vol.mean() / 1e6:,.0f} MB", help="Untreated, over the whole period")
    c5.metric("Total traffic", fmt_bytes(tot_vol.sum()))

    st.markdown(
        f"""<div class="verdict"><b>Recommendation: continue due diligence — do not decide to buy or reject on this data alone.</b><br>
        The file describes <b>{q['end_days']} days</b> of session behaviour. It holds no revenue, cost, churn, ARPU,
        margin, survey or location data, so no valuation, retention or growth claim can be supported. What it does show
        is a large identifiable customer base, a clear heavy-usage segment and a measurable network-throughput gap.</div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("What the data show")
        st.markdown(
            f"- **Base:** {N_CUST:,} customers, {q['n_sessions'] / N_CUST:.2f} sessions each; "
            f"{uo_raw['xDR Sessions'].eq(1).mean() * 100:.0f}% have a single session.\n"
            f"- **Usage:** {app_share.nlargest(2).index[0]} and {app_share.nlargest(2).index[1]} are "
            f"{app_share.nlargest(2).sum():.1f}% of the seven application counters.\n"
            f"- **Engagement:** {ENG[2]} is {eng_share[2]:.1f}% of customers and about {high_traffic_share:.0f}% of treated traffic; "
            f"{ENG[0]} is {eng_share[0]:.1f}%.\n"
            f"- **Concentration:** {gini_word} — Gini {gini_raw:.3f} untreated ({gini_trt:.3f} after mean replacement); "
            f"the top 10% of customers carry {top10_raw:.1f}% of traffic.\n"
            f"- **Experience:** {exp_share.get('Good experience', 0):.1f}% good, {exp_share.get('Low throughput', 0):.1f}% low throughput, "
            f"{exp_share.get('High / unmeasured latency', 0):.1f}% high / unmeasured latency.\n"
            f"- **Devices:** the most common handset ({ROUTER}, {router_sessions / q['n_sessions'] * 100:.1f}% of sessions) is a "
            f"fixed-wireless router, not a phone."
        )
    with right:
        st.subheader("What to treat with caution")
        st.markdown(
            f"- **One-period snapshot:** {q['end_days']} days, with {q['start_before_window']:,} sessions that started before the window.\n"
            f"- **Mean replacement matters:** it alters {P['n_treated']:,} customers ({P['n_treated'] / N_CUST * 100:.1f}%) who carry "
            f"{P['treated_traffic_share'] * 100:.1f}% of raw traffic, and it understates concentration.\n"
            f"- **Latency group is largely unmeasured:** {lat_imputed_share:.0f}% of the “High / unmeasured latency” group has a fully imputed RTT.\n"
            f"- **TCP retransmission is mostly imputed** ({S['tcp_imputed'].mean() * 100:.0f}% of sessions) — rely on throughput instead.\n"
            f"- **Satisfaction is a behavioural proxy,** built from distances to reference clusters — not measured sentiment.\n"
            f"- **{top_app}’s weight is partly arithmetic:** it averages {top_app_share:.0f}% of a customer’s total volume, so its correlation with the total is largely by construction (see Usage & EDA)."
        )

    st.subheader("Coverage of the four analytical areas")
    cov = pd.DataFrame([
        ("User Overview", "Top handsets, manufacturers, per-customer aggregation, EDA, deciles, correlation, PCA",
         f"{N_CUST:,} customers · top manufacturer {comb['Most Frequent Manufacturer'].value_counts().idxmax()}", "📱 User Overview · 📊 Usage & EDA"),
        ("User Engagement", "Top-10 customers per metric, k = 3 tiers, elbow, per-application top users",
         f"{ENG[2]}: {eng_share[2]:.1f}% of customers · elbow k = {P['val_eng'].attrs['elbow_k']}", "📶 Engagement"),
        ("User Experience", "TCP / RTT / throughput per customer, top-bottom-frequent values, handset views, k = 3 groups",
         f"Largest group: {exp_share.idxmax()} ({exp_share.max():.1f}%)", "🛰️ Experience"),
        ("User Satisfaction", "Engagement and experience scores, satisfaction = their average, top 10, k = 2, regression, MySQL, tracking",
         f"Mean satisfaction score {comb['Satisfaction Score'].mean():.2f} · {P['sat_names'][0]}: {(comb['Satisfaction Cluster'] == 0).mean() * 100:.1f}%",
         "😊 Satisfaction · 🧪 Robustness · 📦 Export & Tracking"),
    ], columns=["Area", "What is covered", "Live headline", "Where"])
    st.dataframe(cov, hide_index=True, width="stretch")

    ca, cb = st.columns(2)
    with ca:
        d = pd.DataFrame({"Tier": [ENG[i] for i in range(3)], "Share of customers (%)": [eng_share[i] for i in range(3)]})
        show(px.bar(d, x="Tier", y="Share of customers (%)", color="Tier", title="Engagement tiers (k = 3)",
                    color_discrete_sequence=[TIER_COLOURS[i] for i in range(3)], text_auto=".1f"), x="Engagement tier", y="Customers (% of total)", legend="Tier", note=f"{ENG[2]} is {eng_share[2]:.1f}% of customers but ≈{high_traffic_share:.0f}% of treated traffic — the segment whose retention matters most.")
    with cb:
        d = exp_share.rename_axis("Group").reset_index(name="Share of customers (%)")
        show(px.bar(d, x="Group", y="Share of customers (%)", color="Group", title="Experience groups (k = 3)",
                    color_discrete_sequence=CATEGORY_PALETTE, text_auto=".1f"), x="Experience group", y="Customers (% of total)", legend="Group", note=f"{exp_share.idxmax()} is the largest group ({exp_share.max():.1f}%); only throughput separates the groups robustly.")

    with st.expander("🔍 Verification against the analysis notebook (drift check)"):
        ref = NOTEBOOK_REFERENCE
        eng_sizes = comb["Engagement Cluster"].value_counts().sort_index().tolist()
        exp_vals = [round(float(exp_share.get(n, 0)), 2) for n in ["Good experience", "Low throughput", "High / unmeasured latency"]]
        checks = [
            ("Sessions kept after cleaning", ref["Sessions kept after cleaning"], q["n_sessions"]),
            ("Customers", ref["Customers"], N_CUST),
            ("Customers with ≥1 treated variable", ref["Customers with ≥1 treated variable"], P["n_treated"]),
            ("Engagement cluster sizes (Low / Long-hold / High)",
             ref["Engagement cluster sizes (Low / Long-hold / High)"], eng_sizes),
            ("Experience cluster shares % (Good / Low throughput / High-unmeasured latency)",
             ref["Experience cluster shares % (Good / Low throughput / High-unmeasured latency)"], exp_vals),
            ("High-traffic / Low-proxy-score share % (scaled variant)", ref["High-traffic / Low-proxy-score share % (scaled variant)"], round(prio_share_scaled, 2)),
        ]
        _h = ex_raw[ex_raw["Most Frequent Handset"] != "Unknown"]
        _top = _h["Most Frequent Handset"].value_counts().head(10).index
        _d = _h[_h["Most Frequent Handset"].isin(_top)]
        _g = [x["Avg Throughput (kbps)"].values for _, x in _d.groupby("Most Frequent Handset")]
        _H = stats.kruskal(*_g)[0]
        checks += [
            ("Router median throughput, untreated (kbps)", ref["Router median throughput, untreated (kbps)"],
             int(round(ex_raw.loc[ex_raw["Most Frequent Handset"] == ROUTER, "Avg Throughput (kbps)"].median()))),
            ("Kruskal–Wallis ε², top-10 handsets, untreated", ref["Kruskal–Wallis ε², top-10 handsets, untreated"],
             round((_H - len(_g) + 1) / (len(_d) - len(_g)), 3)),
        ]
        vt = pd.DataFrame([{"Check": n, "Notebook": str(a), "This dashboard": str(b), "Match": "✅" if a == b else "⚠️ differs"}
                           for n, a, b in checks])
        st.dataframe(vt, hide_index=True, width="stretch")
        if (vt["Match"] != "✅").any():
            st.warning("Some figures differ from the notebook — the loaded file or the notebook has changed. "
                       "Everything shown in this dashboard is computed from the loaded file; re-verify the notebook.")
        else:
            st.success("Every reference figure reproduces exactly from the loaded file.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[1]:
    st.subheader("Data quality audit")
    a, b, c, d = st.columns(4)
    a.metric("Raw rows", f"{q['n_raw']:,}")
    b.metric("Rows dropped (no MSISDN)", f"{q['n_raw'] - q['n_sessions']:,}",
             help=f"They carry {q['traffic_share_dropped'] * 100:.2f}% of raw traffic")
    c.metric("Repeated Bearer Ids", f"{q['bearer_repeats']:,}", help="Bearer Id is not a session or customer key")
    d.metric("Unknown device (sessions)", f"{q['unknown_handset_sessions']:,}",
             f"{q['unknown_handset_sessions'] / q['n_sessions'] * 100:.2f}%", delta_color="off")

    issues = pd.DataFrame([
        ("`Dur. (ms)` is mislabelled", f"Median |End−Start − Dur.| = {q['dur_vs_span_median_abs_diff_s']:.2f} s, "
         f"so the column is in {'seconds' if q['dur_label_is_seconds'] else 'unknown units'}", "Renamed `Dur. (s)`; `Dur. (ms).1` dropped as redundant"),
        ("Duration is a bearer-open span", f"{q['share_24h_exact'] * 100:.2f}% of sessions last exactly 24 h; "
         f"{q['share_over_24h'] * 100:.2f}% last longer", "Reported as-is; not read as usage time"),
        ("Rows without MSISDN", f"{q['n_raw'] - q['n_sessions']:,} rows ({q['traffic_share_dropped'] * 100:.2f}% of traffic)",
         "Dropped — cannot be attributed to a customer"),
        ("Bearer Id placeholders / repeats", f"{q['bearer_blank']:,} blank; {q['bearer_repeats']:,} repeated", "Customers counted by MSISDN, never by Bearer Id"),
        ("Unknown device", f"{q['unknown_handset_sessions']:,} sessions (`undefined`/missing)", "Recoded `Unknown`; excluded from handset rankings by default"),
        ("`Total DL` vs application counters", "Application DL counters do not sum to `Total DL` (table below)", "Counters used for app shares only; Total DL+UL used for volume"),
        ("Heavy missingness in network metrics",
         f"TCP {q['missing_pct']['TCP DL Retrans. Vol (Bytes)']:.0f}–{q['missing_pct']['TCP UL Retrans. Vol (Bytes)']:.0f}%, "
         f"RTT {q['missing_pct']['Avg RTT DL (ms)']:.0f}%, HTTP {q['missing_pct']['HTTP DL (Bytes)']:.0f}% missing (raw rows)", "Mean-imputed as the brief requires; imputation flags kept and reported"),
        ("Extreme outliers", f"{P['n_treated']:,} customers ({P['n_treated'] / N_CUST * 100:.1f}%) have ≥1 IQR outlier", "Replaced with the non-outlier mean (brief); untreated results shown alongside"),
    ], columns=["Issue", "Evidence (computed)", "Handling"])
    st.dataframe(issues, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        mp = q["missing_pct"][q["missing_pct"] > 0].head(20).sort_values()
        show(px.bar(mp.rename_axis("Column").reset_index(name="% missing"), x="% missing", y="Column", orientation="h", title="Missing values by column (top 20)"), height=520, x="Missing values (% of raw rows)", y="Column", note=f"{mp.index[-1]} is {mp.iloc[-1]:.0f}% empty — any network-experience conclusion rests largely on imputed values.")
    with right:
        st_day = S.groupby(S["Start"].dt.date).size().rename("Sessions by start date")
        en_day = S.groupby(S["End"].dt.date).size().rename("Sessions by end date")
        cov = pd.concat([st_day, en_day], axis=1).fillna(0).reset_index().rename(columns={"index": "Date"})
        cov.columns = ["Date", "Sessions by start date", "Sessions by end date"]
        show(px.bar(cov.melt("Date", var_name="Series", value_name="Sessions"), x="Date", y="Sessions", color="Series",
                    barmode="group", title="Temporal coverage: starts spread wider than ends"), height=520, x="Calendar date", y="Number of xDR sessions", legend="Timestamp used", note=f"Session ends cover only {q['end_days']} days; starts reach earlier because long sessions opened before the window.")

    left, right = st.columns(2)
    with left:
        dur_h = (S["Dur. (s)"] / 3600).clip(upper=48)
        show(px.histogram(x=dur_h, nbins=96, labels={"x": "Duration (hours, capped at 48)"},
                          title="Session duration — spike at 24 h").update_layout(showlegend=False), y="Number of sessions", note=f"{q['share_24h_exact'] * 100:.0f}% of sessions last exactly 24 h, so duration measures how long a bearer stayed open, not active usage.")
    with right:
        st.markdown("**Application counters vs `Total DL` / `Total UL`**")
        st.dataframe(q["reconciliation"].style.format({"Mean ratio to total": "{:.3f}", "Rows matching exactly (%)": "{:.2f}"}),
                     hide_index=True, width="stretch")
        st.caption("UL reconciles when all seven apps are summed; DL does not — `Total DL` excludes `Other DL` while "
                   "`Total UL` includes `Other UL`. Application shares therefore describe the counters, not total traffic.")

    st.markdown("**Mean imputation log (session level)**")
    st.dataframe(P["imp_log"].style.format({"Missing": "{:,.0f}", "Missing %": "{:.2f}", "Observed median": "{:,.1f}",
                                            "Fill value (mean)": "{:,.1f}", "Fill ÷ median": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.caption("A mean fill that is many times the observed median pulls every imputed customer towards a single, unrepresentative value.")
    st.markdown("**Outlier treatment log (customer level, IQR rule → mean of non-outliers)**")
    st.dataframe(P["treat_log"].style.format({"Outliers replaced": "{:,.0f}", "Outliers %": "{:.2f}",
                                              "Replacement (mean of non-outliers)": "{:,.1f}"}),
                 hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — USER OVERVIEW (Task 1: handsets & manufacturers)
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[2]:
    st.subheader("Handsets and manufacturers")
    excl_unk = st.checkbox("Exclude “Unknown” device from rankings", value=True, key="ov_unknown")
    Sx = S[S["Handset Type"] != "Unknown"] if excl_unk else S
    Sm = S[S["Handset Manufacturer"] != "Unknown"] if excl_unk else S

    top10 = Sx["Handset Type"].value_counts().head(10).rename_axis("Handset").reset_index(name="Sessions")
    top10["Share of sessions (%)"] = top10["Sessions"] / len(S) * 100
    top10["Type"] = np.where(top10["Handset"] == ROUTER, "Fixed-wireless router", "Phone / other")
    mfr = Sm["Handset Manufacturer"].value_counts().head(3).rename_axis("Manufacturer").reset_index(name="Sessions")
    mfr["Share of sessions (%)"] = mfr["Sessions"] / len(S) * 100
    cust_m = comb["Most Frequent Manufacturer"].value_counts()
    mfr["Customers"] = mfr["Manufacturer"].map(cust_m).fillna(0).astype(int)

    left, right = st.columns([3, 2])
    with left:
        show(px.bar(top10, x="Sessions", y="Handset", orientation="h", color="Type", title="Top 10 handsets by xDR sessions",
                    color_discrete_map={"Fixed-wireless router": CINNAMON, "Phone / other": BERRY})
             .update_yaxes(autorange="reversed"), height=460, x="xDR sessions (count)", y="Handset model", legend="Device type", note=(f"{ROUTER} is a home router, not a phone — analyse it as a separate fixed-wireless segment." if ROUTER in top10["Handset"].values else "Check whether any fixed-wireless routers appear among the leading devices."))
    with right:
        show(px.bar(mfr, x="Manufacturer", y="Sessions", text=mfr["Share of sessions (%)"].map("{:.1f}%".format),
                    title="Top 3 manufacturers by sessions", color="Manufacturer", color_discrete_sequence=CATEGORY_PALETTE),
             height=460, x="Handset manufacturer", y="xDR sessions (count)", legend="Manufacturer", note=f"Three brands cover {mfr['Share of sessions (%)'].sum():.1f}% of all sessions, so three partnerships reach almost the whole base.")
    st.dataframe(top10[["Handset", "Sessions", "Share of sessions (%)", "Type"]].style.format(
        {"Sessions": "{:,.0f}", "Share of sessions (%)": "{:.2f}"}), hide_index=True, width="stretch")

    st.markdown("#### Top 5 handsets for each of the top 3 manufacturers")
    cols = st.columns(3)
    per_mfr = {}
    for col, m in zip(cols, mfr["Manufacturer"], strict=False):
        t5 = S[S["Handset Manufacturer"] == m]["Handset Type"].value_counts().head(5).rename_axis("Handset").reset_index(name="Sessions")
        per_mfr[m] = t5
        with col:
            show(px.bar(t5, x="Sessions", y="Handset", orientation="h", title=f"Top 5 {m} handsets by sessions").update_yaxes(autorange="reversed"), x="xDR sessions (count)", y="Handset model", height=330)

    top3_share = mfr["Share of sessions (%)"].sum()
    apple5 = per_mfr.get("Apple", pd.DataFrame({"Handset": []}))["Handset"].tolist()
    st.markdown("#### Interpretation and marketing recommendation")
    st.markdown(
        f"- **Concentration:** the top 3 manufacturers ({', '.join(mfr['Manufacturer'])}) account for {top3_share:.1f}% of all sessions — "
        f"handset-led marketing can cover almost the whole base with three partnerships.\n"
        f"- **The #1 “handset” is not a phone:** {ROUTER} has {router_sessions:,} sessions ({router_sessions / q['n_sessions'] * 100:.1f}%) "
        f"from {router_cust:,} customers. Treat it as a separate fixed-wireless segment with different usage economics.\n"
        + (f"- **Apple’s top 5** are {', '.join(apple5)}. If older generations dominate, that is a hypothesis for upgrade or trade-in offers "
           f"— to be tested against plan and revenue data, which this file does not contain.\n" if apple5 else "")
        + f"- **Data gap:** {q['unknown_handset_sessions'] / q['n_sessions'] * 100:.2f}% of sessions have no usable device identity; "
        f"ask TellCo how device identity is captured before sizing any device campaign."
    )

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — USAGE & EDA (Tasks 1.1 / 1.2)
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[3]:
    st.subheader("Variables and data types (Task 1.2)")
    vdict = variable_dictionary(raw, data_key)
    st.caption(f"{len(vdict)} columns in the raw extract: {explain_roles(vdict['Role'])}. Types are as read from the file; "
               f"descriptions come from the official field list (— = not listed there).")
    role_pick = st.multiselect("Filter by role", sorted(vdict["Role"].unique()), key="vd_role")
    vshow = vdict[vdict["Role"].isin(role_pick)] if role_pick else vdict
    st.dataframe(vshow.style.format({"Non-null": "{:,.0f}", "Missing %": "{:.2f}", "Unique": "{:,.0f}"}),
                 hide_index=True, width="stretch", height=360)
    st.markdown(f"**Why these matter:** the identifiers (`MSISDN/Number`) define the customer; the duration and the seven application "
                f"volume pairs drive engagement and usage; the RTT, throughput and TCP columns drive experience. "
                f"`Dur. (ms)` is stored as a float but is really seconds, and {int((vdict['Missing %'] > 50).sum())} columns are more than half empty.")
    st.divider()

    basis = st.radio("Customer table", ["Treated (outliers → mean, as the brief requires)", "Untreated"], horizontal=True, key="eda_basis")
    T = ut if basis.startswith("Treated") else uo_raw
    st.caption("Per-customer aggregation (Task 1.1): sessions, total duration, DL, UL and per-application volume. "
               "Dispersion is always shown on untreated data.")
    prev = T.head(10).copy()
    prev.index = prev.index.astype("int64").astype(str)
    st.dataframe(prev.style.format("{:,.0f}"), width="stretch")

    st.subheader("Basic metrics & dispersion (non-graphical univariate)")
    ucols = ["xDR Sessions", "Total Duration (s)", "Total DL (Bytes)", "Total UL (Bytes)", "Total Data Volume (Bytes)"] + APP_COLS
    src = uo_raw[ucols]
    disp = pd.DataFrame({"mean": src.mean(), "median": src.median(), "std": src.std(), "variance": src.var(),
                         "min": src.min(), "max": src.max(), "range": src.max() - src.min(),
                         "IQR": src.quantile(.75) - src.quantile(.25), "skew": src.skew(), "kurtosis": src.kurtosis(),
                         "CV": src.std() / src.mean()})
    st.dataframe(disp.style.format("{:,.2f}"), width="stretch")
    st.markdown(
        f"**Reading the table:** every variable has a mean above its median and positive skew "
        f"(volume skew {disp.loc['Total Data Volume (Bytes)', 'skew']:.1f}, sessions skew {disp.loc['xDR Sessions', 'skew']:.1f}), so a few heavy "
        f"customers pull the averages up. The median customer moves {disp.loc['Total Data Volume (Bytes)', 'median'] / 1e6:,.0f} MB versus a "
        f"mean of {disp.loc['Total Data Volume (Bytes)', 'mean'] / 1e6:,.0f} MB. Medians and IQR describe the typical customer better than mean and standard deviation."
    )

    st.subheader("Graphical univariate analysis")
    var = st.selectbox("Variable", ucols, index=4, key="eda_var")
    logx = st.checkbox("log10 scale", value=True, key="eda_log")
    vals = uo_raw[var]
    plot_vals = np.log10(vals.clip(lower=1)) if logx else vals
    left, right = st.columns(2)
    with left:
        show(px.histogram(x=plot_vals, nbins=60, title=f"Distribution of {var} across customers")
             .update_layout(showlegend=False), x=("log10 of " if logx else "") + AXIS_LABELS.get(var, var), y="Number of customers",
             note="Check the tail: when it is long, a few heavy customers pull the mean above the typical customer.")
    with right:
        show(px.box(y=plot_vals, title=f"Spread and outliers of {var}").update_layout(showlegend=False),
             x="All customers", y=("log10 of " if logx else "") + AXIS_LABELS.get(var, var),
             note="Points beyond the whiskers are the outliers that the brief’s mean-replacement rule overwrites.")
    st.caption("Histograms suit these right-skewed counts and volumes (log scale reveals the shape); box plots show the long upper tail of outliers.")

    st.subheader("Application usage")
    left, right = st.columns(2)
    with left:
        d = app_share.sort_values().rename_axis("Application").reset_index(name="Share of counters (%)")
        show(px.bar(d, x="Share of counters (%)", y="Application", orientation="h", text_auto=".1f",
                    title="Share of the seven application counters (DL+UL, untreated)"), x="Share of the seven application counters (%)", y="Application", note=f"{app_share.idxmax()} leads with {app_share.max():.1f}% of the counters; the content apps are small by comparison.")
    with right:
        gshare = (uo_raw["Gaming Total (Bytes)"] / uo_raw["Total Data Volume (Bytes)"]).mean() * 100
        st.markdown(
            f"- **Gaming and Other are {app_share['Gaming'] + app_share['Other']:.1f}%** of the counters; YouTube and Netflix are "
            f"{app_share['YouTube']:.1f}% and {app_share['Netflix']:.1f}%.\n"
            f"- The counters sum to more than total traffic (see Data Quality), so these are shares **of the counters, not of Total Data Volume**.\n"
            f"- Gaming averages **{gshare:.0f}%** of a customer’s `Total Data Volume`, so its near-perfect correlation with the total is largely arithmetic.\n"
            f"- Provenance question for TellCo: how “Gaming” and “Other” are defined, since they have near-identical means and spreads."
        )

    st.subheader("Do manufacturers differ in what they use?")
    mfr3 = S[S["Handset Manufacturer"] != "Unknown"]["Handset Manufacturer"].value_counts().head(3).index.tolist()
    mix = tpl.manufacturer_app_mix(S, mfr3)
    left, right = st.columns([3, 2])
    with left:
        show(px.imshow(mix, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                       title="Share of each manufacturer’s application counters (%)"), height=300, x="Application", y="Handset manufacturer", cbar="Share of counters (%)")
    with right:
        spread = (mix.max() - mix.min())
        st.markdown(f"The application mix is **{'nearly identical' if spread.max() < 5 else 'noticeably different'}** across {', '.join(mfr3)}: the largest gap between manufacturers on any "
                    f"single application is {spread.max():.1f} percentage points ({spread.idxmax()}). Manufacturer explains how *much* customers use, "
                    f"not *what* they use, so application-specific handset marketing has little data support.")
    st.divider()

    st.subheader("Bivariate analysis — each application vs total DL+UL")
    rows = []
    for a, c in zip(APPS, APP_COLS, strict=True):
        vol_t = ut["Total Data Volume (Bytes)"]
        per_sess = np.corrcoef(uo_raw[c] / uo_raw["xDR Sessions"], uo_raw["Total Data Volume (Bytes)"] / uo_raw["xDR Sessions"])[0, 1]
        rows.append({"Application": a, "Pearson (untreated)": uo_raw[c].corr(uo_raw["Total Data Volume (Bytes)"]),
                     "Pearson (treated)": ut[c].corr(vol_t),
                     "Spearman (untreated)": uo_raw[c].corr(uo_raw["Total Data Volume (Bytes)"], method="spearman"),
                     "Pearson per session": per_sess})
    biv = pd.DataFrame(rows).set_index("Application")
    st.dataframe(biv.style.format("{:.3f}"), width="stretch")
    sel_app = st.selectbox("Scatter for application", list(APPS), index=5, key="biv_app")
    samp = uo_raw.sample(min(6000, N_CUST), random_state=RANDOM_STATE)
    show(px.scatter(samp, x=f"{sel_app} Total (Bytes)", y="Total Data Volume (Bytes)", log_x=True, log_y=True, opacity=0.35,
                    title=f"{sel_app} volume vs total data volume (6,000-customer sample, log–log)"), note="Each dot is a customer; a tight upward band means this application tracks total traffic closely.")
    others = biv.drop(index="Gaming")
    st.markdown(
        f"**Interpretation:** Gaming’s relationship with total volume is strong ({biv.loc['Gaming', 'Pearson (untreated)']:.2f}) and survives "
        f"per session ({biv.loc['Gaming', 'Pearson per session']:.2f}); the other applications correlate {others['Pearson (untreated)'].min():.2f}–"
        f"{others['Pearson (untreated)'].max():.2f}, but per session this falls to {others['Pearson per session'].min():.2f}–"
        f"{others['Pearson per session'].max():.2f}. Those moderate correlations mostly reflect that customers with more sessions use more of everything, "
        f"not an app preference."
    )

    st.subheader("Duration deciles (top five classes)")
    dcls = pd.qcut(T["Total Duration (s)"].rank(method="first"), 10, labels=False) + 1
    dec = T.groupby(dcls).agg(Customers=("Total Data Volume (Bytes)", "size"), Total_DL_UL=("Total Data Volume (Bytes)", "sum"),
                              Mean_volume=("Total Data Volume (Bytes)", "mean"), Min_duration=("Total Duration (s)", "min"),
                              Max_duration=("Total Duration (s)", "max"))
    dec["Share of traffic (%)"] = dec["Total_DL_UL"] / dec["Total_DL_UL"].sum() * 100
    dec.index.name = "Duration decile"
    top5 = dec.loc[6:10]
    left, right = st.columns(2)
    with left:
        st.dataframe(top5.style.format({"Customers": "{:,.0f}", "Total_DL_UL": "{:,.0f}", "Mean_volume": "{:,.0f}",
                                        "Min_duration": "{:,.0f}", "Max_duration": "{:,.0f}", "Share of traffic (%)": "{:.2f}"}),
                     width="stretch")
    with right:
        d = dec.reset_index()
        d["Group"] = np.where(d["Duration decile"] >= 6, "Top five deciles", "Bottom five deciles")
        show(px.bar(d, x="Duration decile", y="Total_DL_UL", color="Group", title="Total DL+UL per duration decile",
                    color_discrete_map={"Top five deciles": BERRY, "Bottom five deciles": MIST}), x="Duration decile (1 = shortest total duration, 10 = longest)", y="Total DL+UL data (bytes)", legend="Decile group", note="Compare bar heights to see how traffic changes with total connection time; duration is bearer-open time, not active use.")
    raw_dec = uo_raw.groupby(pd.qcut(uo_raw["Total Duration (s)"].rank(method="first"), 10, labels=False) + 1)["Total Data Volume (Bytes)"].sum()
    st.markdown(f"The top five duration deciles carry **{top5['Share of traffic (%)'].sum():.1f}%** of traffic on the selected table "
                f"({raw_dec.loc[6:10].sum() / raw_dec.sum() * 100:.1f}% untreated). Longer connections go with more data, but duration is a "
                f"bearer-open span (many sessions sit at exactly 24 h), so it is not a clean measure of usage time.")

    st.subheader("Correlation between application volumes")
    cm = T[APP_COLS].corr()
    cm.index = cm.columns = list(APPS)
    show(px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson correlation — application volumes"), x="Application", y="Application", cbar="Pearson r")
    off = cm.values[np.triu_indices(7, 1)]
    st.markdown(f"Pairs correlate **{off.min():.2f}–{off.max():.2f}** on the selected table. {'All are positive and modest' if off.min() > 0 and off.max() < 0.5 else 'Some pairs are strongly related'}: customers who use one "
                f"application more tend to use others more, largely because they have more sessions. "
                f"{'No pair is so redundant that an application can be dropped.' if off.max() < 0.8 else 'At least one pair is highly redundant.'}")

    st.subheader("Principal component analysis")
    Z = StandardScaler().fit_transform(T[APP_COLS])
    pca = PCA(random_state=RANDOM_STATE).fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(x=[f"PC{i + 1}" for i in range(7)], y=evr * 100, name="Explained variance (%)", marker_color=BERRY)
        fig.add_scatter(x=[f"PC{i + 1}" for i in range(7)], y=cum * 100, name="Cumulative (%)", line=dict(color=SLATE))
        fig.update_layout(title="Scree plot: variance explained by each principal component")
        show(fig, x="Principal component", y="Variance explained (%)", legend="Measure",
             note="A gentle slope means the seven application volumes are only weakly redundant.")
    with right:
        load = pd.DataFrame(pca.components_[:3].T, index=list(APPS), columns=["PC1", "PC2", "PC3"])
        show(px.imshow(load, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Loadings (first three components)"), x="Principal component", y="Application", cbar="Loading")
    n80 = int(np.argmax(cum >= 0.8) + 1)
    pc1_top = load["PC1"].abs().sort_values(ascending=False).index[:2].tolist()
    st.markdown(
        f"- **PC1 explains {evr[0] * 100:.1f}%** of variance with loadings of the same sign on every application — a general “overall usage intensity” axis.\n"
        f"- **{n80} components** are needed to reach 80% and {int(np.argmax(cum >= 0.9) + 1)} for 90%: the seven volumes are only weakly redundant, so heavy compression loses information.\n"
        f"- **PC2** separates {load['PC2'].idxmax()} from {load['PC2'].idxmin()}, hinting at distinct usage styles beyond intensity.\n"
        f"- **Largest PC1 loadings:** {pc1_top[0]} and {pc1_top[1]}; use the first two or three components for visualisation rather than as a replacement for the raw volumes."
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — ENGAGEMENT (Task 2)
# ══════════════════════════════════════════════════════════════════════════
def ids(s):
    """Return a Series/DataFrame whose index is the full MSISDN as a digit string (never a float or 3.36E+10)."""
    out = s.copy()
    out.index = pd.Index([f"{int(v)}" for v in out.index], name="MSISDN")
    return out


def with_id_col(df: pd.DataFrame, col: str = "MSISDN/Number") -> pd.DataFrame:
    """Copy of ``df`` with an identifier column rendered as a plain digit string."""
    out = df.copy()
    out[col] = [f"{int(v)}" for v in out[col]]
    return out


if section == SECTIONS[4]:
    st.subheader("Top 10 customers per engagement metric")
    st.caption("Untreated values — after mean replacement, the heaviest customers all sit at the same capped value.")
    metric = st.selectbox("Engagement metric", ENG_METRICS, key="eng_metric")
    t10 = ids(uo_raw[metric].sort_values(ascending=False).head(10)).to_frame()
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(t10.style.format("{:,.0f}"), width="stretch")
    with right:
        show(px.bar(t10.reset_index(), x=metric, y="MSISDN", orientation="h", title=f"Top 10 by {metric}")
             .update_yaxes(autorange="reversed", type="category"), height=360, x=AXIS_LABELS.get(metric, metric), y="Customer (MSISDN)", note="Untreated values: these are the customers carrying the most extreme usage on this metric.")

    st.subheader("Engagement tiers (k-means, k = 3 on standardised metrics)")
    prof = ut.groupby("Engagement Cluster")[ENG_METRICS].agg(["min", "max", "mean", "sum"])
    prof.index = [ENG[i] for i in prof.index]
    flat = prof.copy()
    flat.columns = [f"{m} · {s}" for m, s in flat.columns]
    flat.insert(0, "Customers", ut["Engagement Cluster"].value_counts().sort_index().values)
    flat.insert(1, "Share (%)", flat["Customers"] / N_CUST * 100)
    st.dataframe(flat.style.format("{:,.1f}"), width="stretch")
    st.caption("Non-normalised min / max / mean / total per cluster, on the treated table (the brief’s basis). "
               f"k-means, k = 3, random_state = {RANDOM_STATE}, n_init = 20; cluster fingerprint `{P['fingerprints']['engagement']}`.")

    means = ut.groupby("Engagement Cluster")[ENG_METRICS].mean()
    rel = (means / means.max()).reset_index()
    rel["Tier"] = rel["Engagement Cluster"].map(ENG)
    left, right = st.columns(2)
    with left:
        show(px.bar(rel.melt(["Engagement Cluster", "Tier"], var_name="Metric", value_name="Mean (relative to highest tier)"),
                    x="Metric", y="Mean (relative to highest tier)", color="Tier", barmode="group", title="Tier profile (mean, relative)",
                    color_discrete_sequence=[TIER_COLOURS[i] for i in range(3)]), x="Engagement metric", y="Mean, relative to the highest tier (1 = highest)", legend="Engagement tier", note="Compare bars within each metric: the tallest is the highest tier, and the pattern shows what separates the tiers.")
    with right:
        samp = ut.sample(min(8000, N_CUST), random_state=RANDOM_STATE).copy()
        samp["Tier"] = samp["Engagement Cluster"].map(ENG)
        show(px.scatter(samp, x="Total Duration (s)", y="Total Data Volume (Bytes)", color="Tier", opacity=0.45, log_y=True,
                        title="Tiers on duration vs volume (8,000 sample)", color_discrete_sequence=[TIER_COLOURS[i] for i in range(3)]), legend="Engagement tier", note="Tiers are built from three metrics; where they overlap on one axis, another metric separates them.")
    m0, m1, m2 = means.loc[0], means.loc[1], means.loc[2]
    st.markdown(
        f"- **{ENG[0]}** ({eng_share[0]:.1f}% of customers): {m0['xDR Sessions']:.1f} sessions, {m0['Total Data Volume (Bytes)'] / 1e6:,.0f} MB on average.\n"
        f"- **{ENG[1]}** ({eng_share[1]:.1f}%): sessions and traffic similar to the low tier ({m1['xDR Sessions']:.1f} sessions, "
        f"{m1['Total Data Volume (Bytes)'] / 1e6:,.0f} MB) but bearer-open duration {m1['Total Duration (s)'] / max(m0['Total Duration (s)'], 1):.1f}× longer — "
        f"a different kind of light user, not a middle step.\n"
        f"- **{ENG[2]}** ({eng_share[2]:.1f}%): {m2['xDR Sessions']:.1f} sessions and {m2['Total Data Volume (Bytes)'] / 1e6:,.0f} MB on average, "
        f"about **{high_traffic_share:.0f}%** of treated traffic — the group to protect and upsell."
    )
    if not P["long_hold_is_mid"]:
        st.warning("The middle cluster is not the longest-duration group in this run, so it is labelled “Medium engagement”.")

    st.subheader("How many clusters? Elbow and validation")
    ve = P["val_eng"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Inertia (elbow; lower = tighter)", "Silhouette (higher = better)",
                                                        "Davies-Bouldin (lower = better)", "Calinski-Harabasz (higher = better)"])
    for (rr, cc), col in zip([(1, 1), (1, 2), (2, 1), (2, 2)], ve.columns, strict=False):
        fig.add_scatter(x=ve.index, y=ve[col], mode="lines+markers", row=rr, col=cc, showlegend=False)
        fig.add_vline(x=3, line_dash="dash", line_color=MIST, row=rr, col=cc)
        fig.update_xaxes(title_text="Number of clusters (k)", row=rr, col=cc)
        fig.update_yaxes(title_text=col, row=rr, col=cc)
    fig.update_layout(height=560, title="Choosing k for engagement clustering: four validity measures (dashed line = mandated k = 3)")
    show(fig, note="Where the elbow and the silhouette peak disagree with k = 3, the mandated choice is a business convention, not a statistical optimum.")
    st.markdown(
        f"The chord-distance elbow is at **k = {ve.attrs['elbow_k']}**; silhouette is highest at **k = {best_sil}** "
        f"({ve.loc[best_sil, 'Silhouette']:.3f} vs {ve.loc[3, 'Silhouette']:.3f} at k = 3). The brief mandates k = 3, which is reported above, "
        f"{'but k = ' + str(best_sil) + ' is the statistically cleaner split' if best_sil != 3 else 'and k = 3 is also the best-scoring split'} "
        f"(ARI between the k = 2 and k = 3 partitions: {P['ari_k2_k3']:.3f}). "
        f"Test the number of segments against business outcomes before committing to three."
    )
    with st.expander("Validation table"):
        st.dataframe(ve.style.format({"Inertia": "{:,.1f}", "Silhouette": "{:.4f}", "Davies-Bouldin": "{:.4f}", "Calinski-Harabasz": "{:,.0f}"}),
                     width="stretch")

    st.subheader("Most engaged users and most used applications")
    app_pick = st.selectbox("Application", list(APPS), key="eng_app")
    tu = ids(uo_raw[f"{app_pick} Total (Bytes)"].sort_values(ascending=False).head(10)).to_frame("Total traffic (Bytes)")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Top 10 {app_pick} users**")
        st.dataframe(tu.style.format("{:,.0f}"), width="stretch")
    with right:
        top3 = app_tot.nlargest(3).rename_axis("Application").reset_index(name="Bytes")
        show(px.bar(top3, x="Application", y="Bytes", color="Application", text=top3["Bytes"].map(fmt_bytes),
                    title="Top 3 most used applications (total DL+UL)", color_discrete_sequence=CATEGORY_PALETTE), x="Application", y="Total DL+UL traffic (bytes)", legend="Application", note=f"{', '.join(app_tot.nlargest(2).index)} are the two largest counters ({app_tot.nlargest(2).sum() / app_tot.sum() * 100:.0f}% combined); confirm how they are defined before sizing application-led offers.")
    st.caption(f"Ranking by total traffic: {', '.join(app_tot.nlargest(3).index)} — together {app_tot.nlargest(3).sum() / app_tot.sum() * 100:.1f}% of the application counters.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPERIENCE (Task 3)
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[5]:
    st.subheader("Per-customer experience metrics")
    xb = st.radio("Basis", ["Treated (outliers → mean)", "Untreated"], horizontal=True, key="exp_basis")
    EX = et if xb.startswith("Treated") else ex_raw
    ep = EX.head(10).copy()
    ep.index = ep.index.astype("int64").astype(str)
    st.dataframe(ep[EXP_METRICS + ["Most Frequent Handset"]].style.format({c: "{:,.1f}" for c in EXP_METRICS}), width="stretch")
    st.caption(f"TCP retransmission is the sum of DL and UL; RTT and throughput are DL/UL averages. TCP is imputed in "
               f"{S['tcp_imputed'].mean() * 100:.0f}% of sessions and RTT in {S['rtt_imputed'].mean() * 100:.0f}%.")

    st.subheader("Top, bottom and most frequent 10 values")
    lvl = st.radio("Level", ["Session", "Customer"], horizontal=True, key="ext_level")
    mcol = st.selectbox("Metric", ["TCP retransmission", "RTT", "Throughput"], key="ext_metric")
    if lvl == "Session":
        col = {"TCP retransmission": "TCP (Bytes)", "RTT": "RTT (ms)", "Throughput": "Throughput (kbps)"}[mcol]
        series = S[col]
    else:
        col = {"TCP retransmission": EXP_METRICS[0], "RTT": EXP_METRICS[1], "Throughput": EXP_METRICS[2]}[mcol]
        series = ex_raw[col]
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Top 10**")
    c1.dataframe(series.sort_values(ascending=False).head(10).round(2).reset_index(drop=True).to_frame("Value"), width="stretch")
    c2.markdown("**Bottom 10**")
    c2.dataframe(series.sort_values().head(10).round(2).reset_index(drop=True).to_frame("Value"), width="stretch")
    c3.markdown("**Most frequent 10**")
    vc = series.round(2).value_counts().head(10).rename_axis("Value").reset_index(name="Count")
    c3.dataframe(vc, hide_index=True, width="stretch")
    if mcol != "Throughput":
        st.caption(f"The most frequent {mcol} values are largely the mean used to fill missing data, not real measurements.")

    st.subheader("Throughput and TCP retransmission by handset")
    hb = st.radio("Basis for handset statistics", ["Untreated (as in the notebook)", "Treated (outliers → mean)"],
                  horizontal=True, key="hand_basis")
    V = ex_raw if hb.startswith("Untreated") else et
    V = V[V.index.isin(cf.index.astype(float))]
    d_pool = V[V["Most Frequent Handset"] != "Unknown"]
    if st.session_state["f_router"]:
        d_pool = d_pool[d_pool["Most Frequent Handset"] != ROUTER]
    top_h = d_pool["Most Frequent Handset"].value_counts().head(10).index.tolist()
    dh = d_pool[d_pool["Most Frequent Handset"].isin(top_h)]
    ex_obs = ex_raw.loc[dh.index]
    hg = dh.groupby("Most Frequent Handset")
    tab = pd.DataFrame({
        "Customers": hg.size(),
        "Throughput Q1": hg["Avg Throughput (kbps)"].quantile(.25),
        "Throughput median": hg["Avg Throughput (kbps)"].median(),
        "Throughput Q3": hg["Avg Throughput (kbps)"].quantile(.75),
        "Mean TCP": hg["Avg TCP Retransmission (Bytes)"].mean(),
        "Median TCP (observed only)": ex_obs[ex_obs["TCP imputed share"] == 0].groupby("Most Frequent Handset")["Avg TCP Retransmission (Bytes)"].median(),
        "TCP fully observed (%)": (ex_obs["TCP imputed share"] == 0).groupby(ex_obs["Most Frequent Handset"]).mean() * 100,
    }).sort_values("Throughput median", ascending=False)
    order = tab.index.tolist()
    left, right = st.columns(2)
    with left:
        show(px.box(dh, y="Most Frequent Handset", x="Avg Throughput (kbps)", category_orders={"Most Frequent Handset": order},
                    log_x=True, points=False, title="Distribution of average throughput per handset (log scale)"), height=480, x="Average throughput per customer (kbps, log scale)", y="Handset model (most frequent per customer)", note="Compare medians across handsets; use the sidebar to exclude the fixed-wireless router and see phones alone.")
    with right:
        show(px.bar(tab.reset_index(), x="Mean TCP", y="Most Frequent Handset", orientation="h",
                    category_orders={"Most Frequent Handset": order}, title=f"Average TCP retransmission per handset ({hb.split(' ')[0].lower()}, bytes)"), height=480, x="Mean TCP retransmission per customer (bytes)", y="Handset model", note="Means include imputed values; read them alongside the observed-only medians in the table below.")
    st.dataframe(tab.style.format("{:,.1f}", na_rep="—"), width="stretch")
    groups = [g["Avg Throughput (kbps)"].values for _, g in hg]
    if len(groups) > 1:
        H, p = stats.kruskal(*groups)
        eps2 = (H - len(groups) + 1) / (len(dh) - len(groups))
        size = "negligible" if eps2 < 0.01 else "small" if eps2 < 0.06 else "moderate" if eps2 < 0.14 else "large"
        ptxt = "p < 1e-300" if p == 0 else f"p = {p:.1e}"
        tail = ("The fixed-wireless router drives most of this gap — tick “Exclude router” in the sidebar to see how little "
                "difference remains between phones." if ROUTER in order else
                f"With the router excluded, the handset effect on throughput is {size}.")
        st.markdown(f"**Kruskal–Wallis across {len(groups)} handsets:** H = {H:,.0f}, {ptxt}, ε² = {eps2:.3f} ({size} effect). {tail}")
    top_h_row = tab.iloc[0]
    st.markdown(
        f"**Interpretation:** the highest-median handset is {tab.index[0]} ({top_h_row['Throughput median']:,.0f} kbps) against "
        f"{tab['Throughput median'].iloc[-1]:,.0f} kbps for {tab.index[-1]}. Observed-only TCP medians are shown because means are dominated by the imputed "
        f"value; only {tab['TCP fully observed (%)'].mean():.0f}% of these customers (on average across handsets) have a fully observed TCP measurement, so handset-level TCP conclusions are weak."
    )

    st.subheader("Experience clusters (k-means, k = 3)")
    cs = et.groupby("Experience Cluster")[EXP_METRICS].agg(["min", "max", "mean"])
    cs.columns = [f"{m} · {s}" for m, s in cs.columns]
    cs.insert(0, "Customers", et["Experience Cluster"].value_counts().sort_index().values)
    cs.insert(1, "Share (%)", cs["Customers"] / len(et) * 100)
    cs.insert(2, "RTT fully imputed (%)", et.groupby("Experience Cluster")["RTT imputed share"].apply(lambda s: (s == 1).mean() * 100).values)
    cs.insert(3, "Router customers (%)", et.groupby("Experience Cluster")["Most Frequent Handset"].apply(lambda s: (s == ROUTER).mean() * 100).values)
    cs.index = [EXPN[i] for i in cs.index]
    st.dataframe(cs.style.format("{:,.1f}"), width="stretch")
    st.markdown("**Cluster profile in original units, with computed interpretation**")
    prof = P["exp_profile"].set_index("Group")
    st.dataframe(prof.style.format({"Customers": "{:,.0f}", "Share (%)": "{:.1f}", "Mean TCP retransmission (bytes)": "{:,.0f}",
                                    "Mean RTT (ms)": "{:,.1f}", "Mean throughput (kbps)": "{:,.0f}",
                                    "RTT fully imputed (%)": "{:.0f}", "Router customers (%)": "{:.1f}"}), width="stretch")
    st.caption(f"Reproducibility: k-means, k = 3, random_state = {RANDOM_STATE}, n_init = 20, pipeline v{P['pipeline_version']}. "
               f"Cluster fingerprint `{P['fingerprints']['experience']}` is identical on every run of the same data and code.")
    sm = et.sample(min(8000, len(et)), random_state=RANDOM_STATE).copy()
    sm["Group"] = sm["Experience Cluster"].map(EXPN)
    show(px.scatter(sm, x="Avg RTT (ms)", y="Avg Throughput (kbps)", color="Group", opacity=0.4, log_y=True,
                    title="Experience groups: throughput vs RTT (8,000 sample, treated)"), x="Avg round-trip time (ms)", y="Avg throughput (kbps, log scale)", legend="Experience group", note=f"{lat_imputed_share:.0f}% of the high / unmeasured latency group has no measured RTT, so that group partly reflects missing data.")
    g_good, g_lat, g_tp = [k for k, v in EXPN.items() if v == "Good experience"][0], latgrp, [k for k, v in EXPN.items() if v == "Low throughput"][0]
    st.markdown(
        f"- **Good experience** ({exp_share.get('Good experience', 0):.1f}%): highest throughput; includes "
        f"{cs.loc['Good experience', 'Router customers (%)']:.0f}% router customers, so part of the advantage is device type, not network quality.\n"
        f"- **Low throughput** ({exp_share.get('Low throughput', 0):.1f}%): the plurality group; throughput is the one robust contrast between groups. "
        f"Its causes (plan, device, cell load, location) cannot be identified from three customer-level metrics.\n"
        f"- **High / unmeasured latency** ({exp_share.get('High / unmeasured latency', 0):.1f}%): "
        f"{cs.loc['High / unmeasured latency', 'RTT fully imputed (%)']:.0f}% of the group has a fully imputed RTT, so this group largely reflects missing measurement."
    )
    with st.expander("Cluster validation (experience)"):
        vx = P["val_exp"]
        st.line_chart(vx[["Silhouette"]])
        st.dataframe(vx.style.format({"Inertia": "{:,.1f}", "Silhouette": "{:.4f}", "Davies-Bouldin": "{:.4f}", "Calinski-Harabasz": "{:,.0f}"}), width="stretch")
        st.caption(f"Elbow k = {vx.attrs['elbow_k']}; the brief mandates k = 3. Segments depend on preprocessing, so treat only the throughput split as robust.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — SATISFACTION (Task 4)
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[6]:
    st.warning("**Satisfaction here is a behavioural proxy, not measured satisfaction.** Engagement score = distance from the least-engaged "
               "cluster; experience score = distance from the worst-experience cluster; satisfaction = the average of the two scores after each is min–max scaled to 0–1 so neither dominates (Task 4.2). Alternative weightings are tested in Robustness.")
    if len(cf) == 0:
        st.info("No customers match the current filters.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Customers shown", f"{len(cf):,}")
        b.metric("Mean engagement score", f"{cf['Engagement Score'].mean():.2f}")
        c.metric("Mean experience score", f"{cf['Experience Score'].mean():.2f}")
        d.metric("Mean satisfaction (proxy)", f"{cf['Satisfaction Score'].mean():.3f}")

        left, right = st.columns(2)
        with left:
            show(px.histogram(cf, x="Satisfaction Score", nbins=60, title="Distribution of the satisfaction score (average of engagement and experience scores)"),
                 y="Number of customers", note="The score rises with distance from the least-engaged and worst-experience reference clusters.")
        with right:
            sm = cf.sample(min(6000, len(cf)), random_state=RANDOM_STATE)
            show(px.scatter(sm, x="Engagement Score", y="Experience Score", color="Satisfaction Group", opacity=0.4,
                            title="Engagement vs experience score (k = 2 clusters)"), legend="Satisfaction group", note=f"Mean separation between the two k = 2 groups: {abs(cf.groupby('Satisfaction Group')['Engagement Score'].mean().diff().iloc[-1]):.2f} on engagement vs {abs(cf.groupby('Satisfaction Group')['Experience Score'].mean().diff().iloc[-1]):.2f} on experience (std. units).")

        st.subheader("Top 10 most “satisfied” customers (by proxy)")
        top10s = ids(cf.sort_values("Satisfaction Score", ascending=False).head(10)[
            ["Satisfaction Score", "Engagement Score", "Experience Score", "Engagement Tier", "Experience Group"]])
        st.dataframe(top10s.style.format({"Satisfaction Score": "{:.4f}", "Engagement Score": "{:.3f}", "Experience Score": "{:.3f}"}), width="stretch")
        st.caption("Individual shortlists are not stable: changing the weighting or the experience definition reorders customers substantially. "
                   "Use segments, not names, for decisions.")

        st.subheader("Satisfaction clusters (k-means, k = 2)")
        sc_tab = cf.groupby("Satisfaction Group").agg(Customers=("Satisfaction Score", "size"), Avg_satisfaction=("Satisfaction Score", "mean"),
                                                      Avg_experience_score=("Experience Score", "mean"), Avg_engagement_score=("Engagement Score", "mean"))
        sc_tab["Share (%)"] = sc_tab["Customers"] / len(cf) * 100
        st.dataframe(sc_tab.style.format({"Customers": "{:,.0f}", "Avg_satisfaction": "{:.4f}", "Avg_experience_score": "{:.4f}",
                                          "Avg_engagement_score": "{:.4f}", "Share (%)": "{:.1f}"}), width="stretch")
        st.caption(f"Silhouette for k = 2 on the two scores: {P['sat_sil']:.3f}. The clusters separate engaged-but-weaker-experience customers from "
                   f"less-engaged-but-better-experience ones.")

        st.subheader("Does the experience score measure experience or missing data?")
        e = cf.copy()
        e["RTT status"] = np.select([e["RTT imputed share"] == 1, e["RTT imputed share"] == 0], ["RTT fully imputed", "RTT fully observed"], "RTT mixed")
        rho = stats.spearmanr(e["Experience Score"], e["RTT imputed share"])[0]
        show(px.box(e, x="RTT status", y="Experience Score", points=False, color="RTT status", title="Experience score by RTT measurement status"), x="RTT measurement status", y="Experience score (std. units)", legend="RTT status", note="If scores differ by whether RTT was measured, the experience score partly measures data availability.")
        st.markdown(f"Spearman correlation between experience score and the share of imputed RTT: **{rho:.2f}**. Customers with an unmeasured RTT sit "
                    f"close to the imputed-mean reference point, so a large part of the experience score tracks *whether RTT was measured*.")

        st.subheader("Traffic vs satisfaction-proxy priority matrix")
        pm = pd.crosstab(cf["Traffic Tier"], cf["Proxy Score Tier"], normalize="all") * 100
        left, right = st.columns(2)
        with left:
            show(px.imshow(pm.round(2), text_auto=".2f", color_continuous_scale="Reds", title="Share of selected customers (%)"), x="Satisfaction-score tier", y="Traffic tier", cbar="% of customers", note="Top-left to bottom-right mass shows how closely traffic and the score move together; the off-diagonal cell is the investigate-first segment.")
        with right:
            hl = cf[(cf["Traffic Tier"] == "High traffic") & (cf["Proxy Score Tier"] == "Low proxy score")]
            tr = hl["Total Data Volume (Bytes)"].sum() / cf["Total Data Volume (Bytes)"].sum() * 100 if len(cf) else 0
            st.markdown(
                f"**High-traffic / low-proxy-score:** {len(hl):,} customers ({len(hl) / len(cf) * 100:.2f}% of those shown) carrying "
                f"{tr:.1f}% of their treated traffic.\n\n"
                f"This is a **candidate segment to investigate**, not a confirmed churn risk: "
                f"{(hl['Experience Group'] == 'High / unmeasured latency').mean() * 100:.0f}% of it sits in the high / unmeasured latency group, "
                f"and {(hl['RTT imputed share'] == 1).mean() * 100:.0f}% has a fully imputed RTT."
            )

        st.subheader("Traffic concentration")
        def lorenz(x):
            x = np.sort(np.asarray(x, float))
            cum = np.insert(np.cumsum(x) / x.sum(), 0, 0)
            idx = np.linspace(0, len(cum) - 1, 400).astype(int)
            return idx / (len(cum) - 1), cum[idx]
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MIST), name="Perfect equality")
        for lab, ser, colr in [(f"Untreated (Gini {gini_raw:.3f})", uo_raw["Total Data Volume (Bytes)"], BERRY),
                               (f"Treated (Gini {gini_trt:.3f})", ut["Total Data Volume (Bytes)"], SLATE)]:
            xs, ys = lorenz(ser)
            fig.add_scatter(x=xs, y=ys, mode="lines", name=lab, line=dict(color=colr))
        fig.update_layout(title="Lorenz curve of customer traffic", xaxis_title="Cumulative share of customers (poorest → heaviest users)",
                          yaxis_title="Cumulative share of total traffic")
        show(fig, legend="Treatment", note=f"The further the curve bows below the diagonal, the more traffic sits with few customers (Gini {gini_raw:.2f} untreated).")
        st.caption("Mean replacement understates concentration (lower Gini). Quote untreated figures for any heavy-user question.")

        st.subheader("Regression model of the satisfaction proxy")
        st.markdown("The proxy is a deterministic function of the same behavioural features, so a good fit demonstrates **formula recovery**, "
                    "not predictive skill. A real satisfaction model needs churn, survey or revenue labels.")
        saved_cmp = OUT_DIR / "model_comparison.csv"
        saved_loss = OUT_DIR / "loss_curve.csv"
        live = st.button("▶ Retrain live and compute SHAP", key="train_btn") or st.session_state.get("models_trained")
        if live:
            st.session_state["models_trained"] = True
            M = train_models(comb, data_key)
            res, loss, src = M["results"], M["loss"], "trained live in this session"
        elif saved_cmp.exists() and saved_loss.exists():
            res, loss, M, src = pd.read_csv(saved_cmp, index_col="Model"), pd.read_csv(saved_loss), None, "loaded from the last tracked run"
        else:
            M = train_models(comb, data_key)
            st.session_state["models_trained"] = True
            res, loss, src = M["results"], M["loss"], "trained live in this session"
        st.dataframe(res.style.format({"MAE": "{:.5f}", "RMSE": "{:.5f}", "R² (test)": "{:.4f}", "R² (train)": "{:.4f}",
                                       "Train–test gap": "{:.4f}", "CV R² mean": "{:.4f}", "CV R² std": "{:.4f}", "Fit time (s)": "{:.2f}"}),
                     width="stretch")
        st.caption(f"Results {src}. 80/20 split; CV R² is 5-fold on a 20,000-row subsample of the training set.")
        left, right = st.columns(2)
        with left:
            show(px.line(loss, x="iteration", y=["train_loss", "test_loss"], log_y=True, title="Gradient Boosting loss convergence (½·MSE, log scale)"), x="Boosting iteration", y="Loss (½ · mean squared error, log scale)", legend="Dataset", note=f"Final train loss {loss['train_loss'].iloc[-1]:.2e} vs test loss {loss['test_loss'].iloc[-1]:.2e}: " + ("no sign of overfitting." if loss['test_loss'].iloc[-1] <= 1.5 * loss['train_loss'].iloc[-1] else "test loss is well above train loss — check for overfitting."))
        with right:
            if M is not None:
                imp = M["importance"].sort_values("Permutation")
                show(px.bar(imp.reset_index().melt("index"), x="value", y="index", color="variable", barmode="group", orientation="h",
                            title="Feature importance (permutation vs native)", labels={"index": "", "value": "Importance"}), x="Importance (permutation: score drop; native: share of splits)", y="Feature", legend="Method", note="Methods can disagree; neither shows what drives real customer satisfaction.")
            else:
                st.info("Importance and SHAP are computed on demand — click **Retrain live and compute SHAP** above.")
        gbr = res.loc["Gradient Boosting"]
        st.markdown(f"Gradient Boosting reaches test R² **{gbr['R² (test)']:.4f}** with a train–test gap of {gbr['Train–test gap']:.4f}, "
                    f"but this only shows the score can be reconstructed from its inputs.")
        if M is not None:
            st.subheader("SHAP explanation of the Gradient Boosting model")
            sh = M["shap"]
            left, right = st.columns(2)
            with left:
                ma = sh["mean_abs"].sort_values()
                show(px.bar(ma.rename_axis("Feature").reset_index(name="Mean |SHAP|"), x="Mean |SHAP|", y="Feature", orientation="h",
                            title="Mean absolute SHAP value (average impact on the predicted score)"), x="Mean |SHAP| (average change in predicted score)", y="Feature", note="Larger bars mean the model relies on that feature more to reconstruct the score.")
            with right:
                sv, sd = sh["values"], sh["data"]
                rng = np.random.default_rng(RANDOM_STATE)
                fig = go.Figure()
                for i, f in enumerate(ma.index):
                    pct = sd[f].rank(pct=True).values
                    fig.add_scatter(x=sv[f].values, y=i + rng.uniform(-0.28, 0.28, len(sv)), mode="markers", showlegend=False, name=f,
                                    marker=dict(size=4, color=pct, colorscale="RdBu_r", opacity=0.6,
                                                colorbar=dict(title="Feature value<br>(percentile)") if i == 0 else None),
                                    hovertemplate=f"{f}<br>SHAP %{{x:.3f}}<extra></extra>")
                fig.update_yaxes(tickmode="array", tickvals=list(range(len(ma))), ticktext=list(ma.index))
                fig.update_layout(title="SHAP beeswarm (each dot is a customer)", xaxis_title="SHAP value (impact on predicted score)")
                show(fig, y="Feature (sorted by mean |SHAP|)", note="Red = high feature value, blue = low; dots right of zero push the predicted score up.")
            top1, top2 = ma.index[-1], ma.index[-2]
            st.markdown(f"**{top1}** and **{top2}** carry the most weight. SHAP additivity holds (max reconstruction error "
                        f"{sh['additivity_error']:.1e}), so these attributions fully account for each prediction. They describe how the model "
                        f"reconstructs the *proxy formula*, not what makes customers satisfied.")


# ══════════════════════════════════════════════════════════════════════════
# TAB — ROBUSTNESS & SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[7]:
    st.subheader("How much do the conclusions depend on analyst choices?")
    st.caption("Each check below re-runs part of the pipeline under a different, reasonable choice and measures how much the result moves.")
    R = robustness(P, data_key)

    st.markdown("#### 1 · Cluster stability across random seeds")
    sd = R["seeds"]
    piv = sd.pivot(index="Seed", columns="Clustering", values="ARI vs primary")
    left, right = st.columns(2)
    with left:
        show(px.imshow(piv.round(3), text_auto=".3f", color_continuous_scale="Reds", zmin=0, zmax=1, aspect="auto",
                       title="Adjusted Rand index vs the primary run (1 = identical)"), height=340, x="Clustering", y="Random seed", cbar="ARI", note="Values near 1 mean the same customers land in the same clusters whichever seed is used.")
    with right:
        st.dataframe(sd.style.format({"Inertia": "{:,.1f}", "ARI vs primary": "{:.4f}", "Smallest cluster %": "{:.2f}"}), hide_index=True, width="stretch", height=340)
    worst = sd.loc[sd["ARI vs primary"].idxmin()]
    st.markdown(f"The least stable run ({worst['Clustering']}, seed {int(worst['Seed'])}) still agrees with the primary at ARI **{worst['ARI vs primary']:.3f}**. "
                + ("Cluster membership is reproducible across seeds." if worst["ARI vs primary"] > 0.9 else
                   "Some seeds produce materially different partitions, so treat cluster boundaries as approximate."))

    st.markdown("#### 2 · Mean replacement vs capping (winsorising) of outliers")
    cp = R["capped"]
    prim = (P["ut"]["Engagement Cluster"].value_counts(normalize=True).sort_index() * 100)
    cmp_t = pd.DataFrame({"Mean replacement (primary)": prim.values, "IQR capping (robustness)": cp["sizes"].reindex(range(3)).values},
                         index=[ENG[i] for i in range(3)])
    left, right = st.columns(2)
    with left:
        show(px.bar(cmp_t.reset_index().melt("index"), x="index", y="value", color="variable", barmode="group",
                    title="Engagement tier share (%) under each treatment", labels={"index": "", "value": "% of customers"}), x="Engagement tier", y="Customers (% of total)", legend="Outlier treatment", note="Large differences between bars mean tier sizes depend on how outliers are handled.")
    with right:
        st.metric("Agreement between the two segmentations (ARI)", f"{cp['ari']:.3f}")
        st.metric("Gini of traffic under capping", f"{cp['gini']:.3f}", f"{cp['gini'] - gini_trt:+.3f} vs mean replacement", delta_color="off")
        st.markdown("A low ARI means customers move between tiers when the outlier rule changes: the tiers describe the *treatment* as much as the customers. "
                    "Use tier sizes as indicative only.")

    st.markdown("#### 3 · Alternative satisfaction weightings")
    W = R["weights"]
    left, right = st.columns(2)
    with left:
        show(px.imshow(W["spearman"].round(4), text_auto=".4f", color_continuous_scale="Reds", zmin=0.9, zmax=1,
                       title="Spearman rank correlation between weightings"), height=340, x="Satisfaction definition", y="Satisfaction definition", cbar="Spearman ρ")
    with right:
        show(px.imshow(W["top10_overlap"], text_auto=True, color_continuous_scale="Blues", zmin=0, zmax=10,
                       title="Top-10 list overlap (of 10 customers)"), height=340, x="Satisfaction definition", y="Satisfaction definition", cbar="Shared customers")
    off = W["top10_overlap"].values[np.triu_indices(len(W["top10_overlap"]), 1)]
    st.markdown(f"Rankings are {'highly' if W['spearman'].values.min() > 0.9 else 'only moderately'} correlated overall, but the named top-10 lists share only **{off.min()}–{off.max()} of 10** customers across weightings; "
                f"moving from the primary 50/50 definition to {W['primary_vs']} shifts the average customer by **{W['mean_rank_change']:,.0f}** rank places (max {W['max_rank_change']:,.0f}). "
                f"Individual shortlists are therefore not stable — use segments, not names.")

    st.markdown("#### 4 · Cluster-free experience scores")
    st.dataframe(R["alt_exp"].style.format({"Spearman (experience score)": "{:.3f}", "Spearman (proxy score)": "{:.3f}",
                                            "Top-1000 overlap": "{:,.0f}", "Priority-quadrant overlap (%)": "{:.1f}"}), width="stretch")
    keep_lo, keep_hi = R["alt_exp"]["Priority-quadrant overlap (%)"].min(), R["alt_exp"]["Priority-quadrant overlap (%)"].max()
    st.markdown(f"Replacing the cluster-distance experience score with a direct composite or throughput alone keeps the proxy rank correlation at "
                f"{R['alt_exp']['Spearman (proxy score)'].min():.2f}–{R['alt_exp']['Spearman (proxy score)'].max():.2f}. "
                f"{keep_lo:.0f}–{keep_hi:.0f}% of the priority high-traffic / low-score segment is retained, so roughly "
                f"{100 - keep_hi:.0f}–{100 - keep_lo:.0f}% of its members change with the definition of experience.")

    st.markdown("#### 5 · How many clusters? (experience)")
    vx = P["val_exp"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Inertia (elbow)", "Silhouette ↑", "Davies-Bouldin ↓"])
    for i, c in enumerate(["Inertia", "Silhouette", "Davies-Bouldin"], 1):
        fig.add_scatter(x=vx.index, y=vx[c], mode="lines+markers", row=1, col=i, showlegend=False)
        fig.add_vline(x=3, line_dash="dash", line_color=MIST, row=1, col=i)
        fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=i)
        fig.update_yaxes(title_text=c, row=1, col=i)
    fig.update_layout(height=360, title="Choosing k for experience clustering (dashed line = mandated k = 3)")
    show(fig, note="Experience segments are an exploratory grouping; only the throughput split holds across treatments.")
    st.caption(f"Elbow at k = {vx.attrs['elbow_k']}; silhouette is highest at k = {int(vx['Silhouette'].idxmax())}. The brief mandates k = 3 (dashed line).")

# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[8]:
    st.subheader("Look up a customer")
    msisdn = st.text_input("MSISDN / customer number", key="lookup_id", placeholder="e.g. 33659725664")
    if msisdn.strip():
        if msisdn.strip().isdigit() and int(msisdn) in comb.index:
            row = comb.loc[int(msisdn)]
            pct = float((comb["Satisfaction Score"] < row["Satisfaction Score"]).mean() * 100)
            a, b, c, d = st.columns(4)
            a.metric("Satisfaction (proxy)", f"{row['Satisfaction Score']:.3f}", f"{pct:.0f}th percentile", delta_color="off")
            b.metric("Engagement tier", row["Engagement Tier"])
            c.metric("Experience group", row["Experience Group"])
            d.metric("Handset", row["Most Frequent Handset"])
            st.dataframe(row[ENG_METRICS + EXP_METRICS + ["Engagement Score", "Experience Score", "Satisfaction Group"]].astype(str).to_frame("Value"),
                         width="stretch")
        else:
            st.warning("That MSISDN is not in the cleaned customer table.")

    st.subheader("What-if scorer")
    st.caption("Score a hypothetical customer with the fitted scalers and cluster centres. Values beyond the 99th percentile are clipped to the "
               "range the model has seen, because the mean-replacement treatment makes extremes unrepresentable.")
    sc = P["scoring"]
    med_e, med_x = ut[ENG_METRICS].median().values, et[EXP_METRICS].median().values
    left, right = st.columns(2)
    with left:
        in_s = st.number_input("xDR sessions", 1, int(sc["eng_p99"][0]), int(med_e[0]), key="wi_s")
        in_d = st.number_input("Total duration (s)", 0, int(sc["eng_p99"][1]), int(med_e[1]), key="wi_d")
        in_v = st.number_input("Total data volume (MB)", 0, int(sc["eng_p99"][2] / 1e6), int(med_e[2] / 1e6), key="wi_v")
    with right:
        in_t = st.number_input("Avg TCP retransmission (bytes)", 0, int(sc["exp_p99"][0]), int(med_x[0]), key="wi_t")
        in_r = st.number_input("Avg RTT (ms)", 0, int(sc["exp_p99"][1]), int(med_x[1]), key="wi_r")
        in_p = st.number_input("Avg throughput (kbps)", 0, int(sc["exp_p99"][2]), int(med_x[2]), key="wi_p")
    if st.button("Score this customer", type="primary", key="wi_btn"):
        out = score_new_customer(sc, [in_s, in_d, in_v * 1e6], [in_t, in_r, in_p])
        pctl = float((comb["Satisfaction Score"] < out["sat"]).mean() * 100)
        a, b, c, d = st.columns(4)
        a.metric("Engagement score", f"{out['eng']:.2f}")
        b.metric("Experience score", f"{out['exp']:.2f}")
        c.metric("Satisfaction (proxy)", f"{out['sat']:.3f}", f"{pctl:.0f}th percentile", delta_color="off")
        d.metric("Satisfaction cluster", SATN[out["sat_cluster"]])
        st.caption(f"Nearest engagement tier: {ENG[out['eng_cluster']]}.")
        mdl, mname = latest_saved_model()
        if mdl is None and st.session_state.get("model_load_error"):
            st.info(st.session_state["model_load_error"] + " Train on the Satisfaction page to enable the model prediction here.")
        if mdl is None and st.session_state.get("models_trained"):
            mdl, mname = train_models(comb, data_key)["model"], "model trained this session"
        if mdl is not None:
            pred = float(tmod.predict_satisfaction(mdl, {"xDR Sessions": in_s, "Total Duration (s)": in_d, "Total Data Volume (Bytes)": in_v * 1e6,
                                                        "Avg TCP Retransmission (Bytes)": in_t, "Avg RTT (ms)": in_r, "Avg Throughput (kbps)": in_p})[0])
            st.caption(f"Deployed Gradient Boosting model (`{mname}`) predicts {pred:.3f} for the same inputs.")

    st.subheader("Customer table")
    cols_show = ["Satisfaction Score", "Engagement Tier", "Experience Group", "Satisfaction Group", "Most Frequent Handset"] + ENG_METRICS + EXP_METRICS
    st.dataframe(ids(cf.sort_values("Satisfaction Score", ascending=False).head(500)[cols_show]), width="stretch")
    st.caption(f"Showing the top 500 of {len(cf):,} filtered customers; download the full filtered table from the sidebar.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 9 — RECOMMENDATION & LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[9]:
    st.subheader("Recommendation: continue due diligence")
    st.markdown(
        """<div class="verdict">The behavioural evidence <b>neither supports nor rules out</b> buying TellCo. It identifies questions to put to
        the owners, not a valuation. A purchase decision needs financial data the file does not contain.</div>""",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Where TellCo looks strong")
        st.markdown(
            f"- A large, identifiable base: **{N_CUST:,}** customers.\n"
            f"- A sizeable **{ENG[2]}** group ({eng_share[2]:.1f}% of customers, ≈{high_traffic_share:.0f}% of treated traffic).\n"
            f"- Concentration is {gini_word} (Gini {gini_raw:.2f}; top 10% carry {top10_raw:.1f}% of traffic) {'— not dependent on a thin sliver of customers' if top10_raw < 50 else '— traffic is concentrated in a small share of customers'}.\n"
            f"- **{top_app}** has the strongest relationship with traffic volume (r = {app_corr.max():.2f}; next: {app_corr.drop(top_app).idxmax()}, r = {app_corr.drop(top_app).max():.2f}).\n"
            f"- A distinct fixed-wireless segment ({router_cust:,} router customers) that can be marketed separately."
        )
    with right:
        st.markdown("#### Where it looks weaker")
        st.markdown(
            f"- **{exp_share.get('Low throughput', 0):.1f}%** of customers are in the low-throughput group — the one robust network weakness.\n"
            f"- The latency problem is unproven: {lat_imputed_share:.0f}% of that group has an unmeasured RTT.\n"
            f"- TCP monitoring looks unreliable ({S['tcp_imputed'].mean() * 100:.0f}% of sessions imputed) — a direct question for TellCo’s network team.\n"
            f"- **{prio_share:.1f}%** of customers are high-traffic but low on the proxy score — a segment to investigate, partly an artefact of unmeasured RTT."
        )

    st.markdown("#### Required before a buy / do-not-buy decision")
    st.markdown(
        "1. **Financial data:** revenue, ARPU, cost, margin, churn and customer lifetime value.\n"
        "2. **A longer, evenly sampled window** (several weeks) and the extraction rule that produced this one.\n"
        "3. **Outcome data** (survey, NPS, complaints, churn) to test the proxy and support a real predictive model.\n"
        "4. **Network data:** cell-level, location, plan and device data, plus an explanation of the missing RTT and TCP measurements.\n"
        "5. **Definitions** of the “Gaming” and “Other” counters, the `Total DL` / `Total UL` construction and the device-identity gap.\n"
        "6. **Further analysis:** repeat the segmentation and experience analyses on untreated or capped data, and split every experience result into routers and phones."
    )

    st.markdown("#### Limitations log")
    lim = pd.DataFrame([
        ("Short window", f"Only {q['end_days']} days of session end dates; trends and seasonality cannot be assessed", "Obtain a longer, evenly sampled extract"),
        ("No financial or outcome data", "No revenue, cost, churn, ARPU or survey data — no valuation or retention claim is possible", "Request from TellCo"),
        ("Satisfaction is a proxy", "Built from distances to cluster centres; consistency checks are not validation", "Never present as customer satisfaction"),
        ("Regression recovers a formula", "Target is a deterministic function of its inputs", "Do not quote R² as predictive skill"),
        ("Heavy imputation", f"TCP {S['tcp_imputed'].mean() * 100:.0f}% and RTT {S['rtt_imputed'].mean() * 100:.0f}% of sessions imputed", "Rely on throughput; ask about network monitoring"),
        ("Mean replacement of outliers", f"Alters {P['n_treated']:,} customers carrying {P['treated_traffic_share'] * 100:.0f}% of raw traffic", "Untreated results shown alongside; use them for heavy-user questions"),
        ("Segments depend on preprocessing", "Cluster membership changes with treatment choice; only the throughput split is robust", "Treat other splits as exploratory"),
        ("k = 3 is mandated, not optimal", f"k = {best_sil} scores higher on silhouette", "Validate segment count against business outcomes"),
        ("Duration is not usage time", f"{q['share_24h_exact'] * 100:.0f}% of sessions last exactly 24 h", "Confirm the system rule with TellCo"),
        ("Router mixed with phones", f"{router_sessions / q['n_sessions'] * 100:.1f}% of sessions are a home router", "Analyse routers and phones separately"),
        ("Counter definitions unclear", f"Counters do not reconcile to `Total DL`; Gaming and Other have mean ratio "
         f"{uo_raw['Gaming Total (Bytes)'].mean() / uo_raw['Other Total (Bytes)'].mean():.2f} and std ratio "
         f"{uo_raw['Gaming Total (Bytes)'].std() / uo_raw['Other Total (Bytes)'].std():.2f}", "Ask TellCo for the definitions"),
    ], columns=["Limitation", "Why it matters", "How handled / next step"])
    st.dataframe(lim, hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════
# TAB 10 — EXPORT & TRACKING
# ══════════════════════════════════════════════════════════════════════════
if section == SECTIONS[10]:
    final = tdb.scores_frame(comb)
    st.subheader("Final scored table (Task 4.6)")
    st.dataframe(with_id_col(final.head(20)).style.format({"Engagement Score": "{:.4f}", "Experience Score": "{:.4f}", "Satisfaction Score": "{:.4f}"}),
                 hide_index=True, width="stretch")
    st.download_button("⬇ Download full scored table (CSV)", final.to_csv(index=False).encode(), "tellco_user_scores.csv", "text/csv")

    st.markdown("#### MySQL export")
    mylog = OUT_DIR / "mysql_export.json"
    if mylog.exists():
        info = json.loads(mylog.read_text())
        a, b, c, d = st.columns(4)
        a.metric("Rows in table", f"{info['rows_in_table']:,}")
        b.metric("Table", info["table"])
        c.metric("Server", info["server_version"][:22])
        d.metric("Exported", info["exported_at"].replace("T", " "))
        st.markdown("**Verification query and its output:**")
        st.code(info["select_query"], language="sql")
        st.dataframe(with_id_col(pd.DataFrame(info["select_output"])), hide_index=True, width="stretch")
        shot = DOCS_DIR / "mysql_select_screenshot.png"
        if shot.exists():
            st.image(str(shot), caption="SELECT output screenshot")
    else:
        st.info("No export recorded yet. Run `tellco-run --data <file> --mysql USER:PASSWORD@HOST:PORT/DB`, or use the form below.")
    with st.expander("Export now from this page"):
        with st.form("mysql_form"):
            c1, c2 = st.columns(2)
            host, port = c1.text_input("Host", "localhost"), c2.text_input("Port", "3306")
            user, pw = c1.text_input("User", "root"), c2.text_input("Password", type="password")
            dbn, tbl = c1.text_input("Database", "tellco"), c2.text_input("Table", "user_scores")
            go_sql = st.form_submit_button("Export table")
        if go_sql:
            try:
                res = tdb.export_scores(final, tdb.mysql_url(user, pw, host, port, dbn), tbl, log_path=OUT_DIR / "mysql_export.json")
                st.success(f"Exported {res['rows_in_table']:,} rows to `{dbn}.{tbl}`.")
                st.code(res["select_query"], language="sql")
                st.dataframe(with_id_col(pd.DataFrame(res["select_output"])), hide_index=True, width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Export failed: {exc}")

    st.subheader("Model deployment tracking (Task 4.7)")
    runs = ttrack.list_runs(OUT_DIR)
    log = ttrack.load_run_log(OUT_DIR)
    if log and (log.get("pipeline_version") != P["pipeline_version"] or log["source"]["n_customers"] != N_CUST):
        st.warning(f"The tracked run was produced by pipeline v{log.get('pipeline_version', 'unknown')} on "
                   f"{log['source']['n_customers']:,} customers; this dashboard runs v{P['pipeline_version']} on {N_CUST:,}. "
                   f"Re-run `tellco-run --data <file>` so the metrics below are not stale.")
    if len(runs):
        st.markdown(f"**{len(runs)} tracked run(s)** in MLflow (`{(OUT_DIR / 'mlflow.db').name}`). Launch the full tracking UI with "
                    f"`mlflow ui --backend-store-uri sqlite:///{OUT_DIR.name}/mlflow.db`.")
        show_cols = [c for c in runs.columns if c in ("run_id", "start_time", "end_time", "status") or c.startswith(("metrics.", "params.", "tags.code_version", "tags.model"))]
        show_cols = [c for c in show_cols if not c.startswith(("metrics.train_loss", "metrics.test_loss"))]
        st.dataframe(runs[show_cols].astype(str), hide_index=True, width="stretch")
    if log:
        left, right = st.columns(2)
        with left:
            st.markdown("**Run log** (code version, source, parameters, metrics)")
            st.json(log, expanded=False)
        with right:
            lc = OUT_DIR / "loss_curve.csv"
            if lc.exists():
                show(px.line(pd.read_csv(lc), x="iteration", y=["train_loss", "test_loss"], log_y=True, title="Loss convergence of the tracked MLflow run"),
                     x="Boosting iteration", y="Loss (½ · mean squared error, log scale)", legend="Dataset", height=340)
        artifacts = [p.name for p in sorted(OUT_DIR.rglob("*")) if p.is_file() and p.suffix in (".csv", ".json", ".joblib") and "mlartifacts" not in str(p)]
        st.markdown("**Artifacts:** " + ", ".join(f"`{a}`" for a in artifacts))
        st.download_button("⬇ Download run log (JSON)", json.dumps(log, indent=2).encode(), "run_log.json", "application/json")
    if not len(runs) and not log:
        st.info("No tracked run yet. Run `tellco-run --data <file>` (or train on the Satisfaction page and log it below).")
    if st.session_state.get("models_trained") and st.button("Log the live-trained model as a new MLflow run", key="log_live"):
        M = train_models(comb, data_key)
        mp = tmod.save_model(M["model"], OUT_DIR / "models" / f"{M['run_log']['run_id']}.joblib")
        M["run_log"]["model_path"] = str(mp)
        rid = ttrack.log_run(M["run_log"], M["loss"], M["results"], mp, OUT_DIR)
        st.success(f"Logged MLflow run `{rid}`. Reload the page to see it in the table.")
    mdl, mname = latest_saved_model()
    if mname:
        st.markdown(f"**Deployed model artifact:** `{mname}` — loaded by the what-if scorer.")
    elif st.session_state.get("model_load_error"):
        st.warning(st.session_state["model_load_error"])

    st.subheader("Feature store")
    reg = tfs.load_registry(FS_DIR)
    if reg:
        st.dataframe(pd.DataFrame([{"version": e["version"], "rows": e["rows"], "features": len(e["features"]), "created": e["created"],
                                    "sha256": e["sha256"], "path": e["path"]} for e in reg]), hide_index=True, width="stretch")
        lat = reg[-1]
        st.markdown("**Feature schema (latest)**")
        st.dataframe(pd.DataFrame({"Feature": list(lat["features"]), "Type": list(lat["features"].values())}), hide_index=True, width="stretch")
        if st.checkbox("Preview latest snapshot (first 10 customers)", key="fs_preview"):
            st.dataframe(tfs.load_features(FS_DIR).head(10), width="stretch")
        st.code("from tellco.feature_store import load_features\nX = load_features('feature_store', columns=['Avg RTT (ms)', 'Satisfaction Score'])", language="python")
    else:
        st.info("The feature store is empty. Run `tellco-run --data <file>` to create the first snapshot.")
