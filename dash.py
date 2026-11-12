import io
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import truststore
truststore.inject_into_ssl()


st.set_page_config(page_title="SPEC OSG Benchmark Intelligence", page_icon="📊", layout="wide")

SPEC_HOME = "https://spec.org/cgi-bin/osgresults"
SPEC_CSV_URL = "https://spec.org/cgi-bin/osgresults?conf=all;op=dump;format=csvdump"
USER_AGENT = "Mozilla/5.0 (compatible; SPEC-OSG-Dashboard/1.0; benchmark research)"

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px; border-radius: 10px;}
.small-note {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("SPEC OSG Benchmark Intelligence")
st.caption("Explore published SPEC Open Systems Group results across vendors, processors, systems and benchmark families.")

# --------------------------- ingestion ---------------------------
def parse_delimited_data(content):
    # Historical SPEC records include Windows-1252 characters (e.g. registered marks).
    try:
        raw = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = content.decode("cp1252", errors="replace")
    if re.search(r"<!doctype\s+html|<html\b|<pre\b|<table\b", raw[:4096], re.I):
        raise ValueError("SPEC returned HTML instead of CSV. Use the Download as CSV link on the SPEC results page.")
    # Do not silently skip malformed records: that can hide processor families.
    frame = pd.read_csv(io.StringIO(raw), sep=None, engine="python")
    if len(frame.columns) < 2 or frame.empty:
        raise ValueError("The source contains no usable delimited result records.")
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_spec_dump(url: str, refresh_token: int = 0):
    headers = {"User-Agent": USER_AGENT, "Accept": "text/csv,text/plain,*/*"}
    response = requests.get(url, headers=headers, timeout=90, verify=True)
    response.raise_for_status()
    if not response.content or len(response.content) < 20:
        raise ValueError("SPEC returned an empty response.")
    return parse_delimited_data(response.content), response.url, response.status_code


def clean_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()

def first_matching_column(df, patterns):
    # Prefer specific aliases even when metadata columns appear first.
    for pattern in patterns:
        for col in df.columns:
            if re.search(pattern, clean_name(col)):
                return col
    return None

def to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace(r"[^0-9eE.+-]", "", regex=True),
        errors="coerce"
    )

def processor_details(cpu, system, family):
    def clean(value):
        text = re.sub(r"\((?:R|TM)\)|[\u00ae\u2122]", "", str(value), flags=re.I)
        return re.sub(r"\s+", " ", text).strip()

    cpu, system = clean(cpu), clean(system)
    text = cpu + " " + system
    if family == "AMD EPYC":
        match = re.search(r"\bEPYC\s+(?:Embedded\s+)?([4789][0-9A-Z]{2}[1-6][A-Z]*)\b", text, re.I)
        if match:
            sku = match.group(1).upper()
            return "AMD EPYC " + sku, "EPYC " + sku[0] + "00" + sku[3]
        return (cpu if re.search(r"EPYC", cpu, re.I) else "AMD EPYC (model not identified)"), "EPYC (series not identified)"
    if family == "Intel Xeon":
        match = re.search(
            r"\bXeon\s+(?:(?:CPU|Processor)\s+)?"
            r"((?:(?:Platinum|Gold|Silver|Bronze|Max)\s+)?\d{4,5}[A-Z]*(?:\+|-[A-Z]+)?|"
            r"[DEW]\d?-\d{3,5}[A-Z]*(?:\s+v\d+)?|\d{4}[A-Z]*(?:\s+v\d+)?)", text, re.I)
        if match:
            return "Intel Xeon " + match.group(1), "Not EPYC"
        return (cpu if re.search(r"Xeon", cpu, re.I) else "Intel Xeon (model not identified)"), "Not EPYC"
    return cpu or "Processor not identified", "Not EPYC"


def xeon_series(model, family):
    if family != "Intel Xeon":
        return "Not Xeon"
    text = str(model)
    # Tier names distinguish Scalable parts from newer numeric Xeon series.
    tier = re.search(r"\bXeon\s+(Platinum|Gold|Silver|Bronze|Max)\s+(\d{4})", text, re.I)
    if tier:
        return "Xeon " + tier.group(1).title() + " " + tier.group(2)[:2] + "00 series"
    legacy = re.search(r"\bXeon\s+([DEW]\d?)-(\d{4})(?:[A-Z]*)(?:\s+(v\d+))?", text, re.I)
    if legacy:
        label = "Xeon " + legacy.group(1).upper() + "-" + legacy.group(2)[:2] + "00"
        return label + (" " + legacy.group(3).lower() if legacy.group(3) else "") + " series"
    numeric = re.search(r"\bXeon\s+(\d{4,5})", text, re.I)
    if numeric:
        return "Xeon " + numeric.group(1)[:-2] + "00 series"
    return "Xeon (series not identified)"


def benchmark_type(value):
    name = str(value).lower()
    if re.match(r"c(?:int|fp)\d+", name):
        workload = "integer" if name.startswith("cint") else "floating point"
        mode = "throughput (rate)" if "rate" in name else "speed"
        return f"CPU {workload} {mode}"
    for token, label in [("power", "Power / energy efficiency"), ("jbb", "Java business"),
                         ("jvm", "Java virtual machine"), ("sfs", "File server"),
                         ("laddis", "File server"), ("web", "Web server"),
                         ("virt", "Virtualization"), ("mail", "Mail server"),
                         ("jas", "Java application server"), ("jent", "Java enterprise"),
                         ("jms", "Java messaging")]:
        if token in name:
            return label
    return "Other / unidentified"


def filter_processors(frame, series=(), epyc_models=(), xeon_models=(), intel_series=()):
    # AMD and Intel selections form two comparison cohorts; retain both vendors.
    mask = pd.Series(True, index=frame.index)
    amd = frame["processor_family"].eq("AMD EPYC")
    intel = frame["processor_family"].eq("Intel Xeon")
    if series:
        mask &= ~amd | frame["epyc_series"].isin(series)
    if epyc_models:
        mask &= ~amd | frame["processor_model"].isin(epyc_models)
    if intel_series:
        mask &= ~intel | frame["xeon_series"].isin(intel_series)
    if xeon_models:
        mask &= ~intel | frame["processor_model"].isin(xeon_models)
    return frame.loc[mask]


def performance_summary(frame, metric, statistic):
    # Keep configurations separate and never aggregate across benchmark IDs.
    data = frame.copy()
    data["Sockets"] = data["sockets_numeric"].astype("string").fillna("Unknown")
    data["Cores"] = data["cores_numeric"].astype("string").fillna("Unknown")
    data["value"] = pd.to_numeric(data[metric], errors="coerce")
    data = data[data["value"].gt(0)]
    return data.groupby(["benchmark", "processor_family", "epyc_series", "xeon_series", "processor_model", "Sockets", "Cores"], dropna=False)["value"].agg(
        Records="count", Performance=statistic, Minimum="min", Maximum="max").reset_index()


def normalize_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    # Remove fully empty columns/rows and repeated header rows.
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    if df.empty:
        return df
    # Identify useful fields without assuming one fixed SPEC schema.
    aliases = {
        "benchmark": [r"^benchmark$", r"benchmark name", r"workload", r"^suite$"],
        "vendor": [r"^vendor$", r"hardware vendor", r"manufacturer", r"^company$"],
        "system": [r"system", r"^model$", r"computer"],
        "cpu": [r"^processor$", r"^cpu$", r"^processor (?:name|model|type)$", r"^cpu (?:name|model|type)$", r"^chip (?:name|model|type)$", r"^socket type$", r"processor (?:name|model|type|description)"],
        "cores": [r"^cores?$", r"core count", r"total cores", r"^# cores"],
        "sockets": [r"^sockets?$", r"^chips?$", r"^cpu count$", r"^processors$", r"socket count"],
        "score": [r"^result$", r"^score$", r"performance", r"throughput", r"rate", r"^base"],
        "date": [r"^published$", r"^publication date$", r"date", r"submission"],
        "url": [r"url", r"detail", r"result link", r"^link$"],
    }
    for target, pats in aliases.items():
        col = first_matching_column(df, pats)
        if col and target not in df.columns:
            df[target] = df[col]
    # Infer processor/vendor from all text if explicit columns are absent.
    text_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    combined = df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1) if text_cols else pd.Series("", index=df.index)
    if "vendor" not in df.columns:
        df["vendor"] = combined.apply(lambda x: "AMD" if re.search(r"\bAMD\b|EPYC", x, re.I) else ("Intel" if re.search(r"\bIntel\b|Xeon", x, re.I) else "Other / Unknown"))
    else:
        df["vendor"] = df["vendor"].fillna("").astype(str)
        df.loc[df["vendor"].str.strip().eq(""), "vendor"] = combined[df["vendor"].str.strip().eq("")]
    if "cpu" not in df.columns:
        df["cpu"] = combined
    if "system" not in df.columns:
        df["system"] = combined
    if "benchmark" not in df.columns:
        df["benchmark"] = "OSG result (benchmark not identified)"
    if "cores" not in df.columns:
        df["cores"] = combined.str.extract(r"(?i)\b(\d{1,4})\s*(?:physical\s*)?cores?\b", expand=False)
    df["cores_numeric"] = to_numeric(df["cores"]) if "cores" in df.columns else float("nan")
    if "sockets" not in df.columns:
        df["sockets"] = combined.str.extract(r"(?i)\b([1-8])\s*[- ]?socket", expand=False)
    df["sockets_numeric"] = to_numeric(df["sockets"]) if "sockets" in df.columns else float("nan")
    if "score" in df.columns:
        df["score_numeric"] = to_numeric(df["score"])
    else:
        # Score-like columns vary by benchmark; retain the source fields and choose the first numeric candidate.
        candidates = []
        for c in df.columns:
            if re.search(r"score|result|throughput|rate|ops|SPEC", str(c), re.I):
                vals = to_numeric(df[c])
                if vals.notna().sum() > 0:
                    candidates.append((c, vals))
        if candidates:
            df["score_numeric"] = candidates[0][1]
            df["score"] = df[candidates[0][0]]
        else:
            df["score_numeric"] = float("nan")
            df["score"] = ""
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        df["date_parsed"] = pd.NaT
    # Classify processor family, not just company branding.
    cpu_text = df["cpu"].fillna("").astype(str) + " " + df["system"].fillna("").astype(str)
    # Some exports split processor descriptions across several fields.
    for col in raw.columns:
        if re.search(r"(?:processor|cpu|chip)", clean_name(col)):
            name = str(col).strip().replace("\n", " ")
            if name in df.columns and name not in {"cpu", "system"}:
                cpu_text = cpu_text + " " + df[name].fillna("").astype(str)
    df["processor_family"] = cpu_text.apply(
        lambda x: "AMD EPYC" if re.search(r"EPYC", x, re.I) else
                  ("Intel Xeon" if re.search(r"Xeon", x, re.I) else
                   ("AMD (other)" if re.search(r"\bAMD\b", x, re.I) else
                    ("Intel (other)" if re.search(r"\bIntel\b", x, re.I) else "Other / Unknown")))
    )
    details = [processor_details(cpu, system, family) for cpu, system, family in
               zip(df["cpu"].fillna(""), df["system"].fillna(""), df["processor_family"])]
    df["processor_model"] = [item[0] for item in details]
    df["epyc_series"] = [item[1] for item in details]
    df["xeon_series"] = [xeon_series(model, family) for model, family in zip(df["processor_model"], df["processor_family"])]
    df["benchmark_type"] = df["benchmark"].map(benchmark_type)
    df["performance_per_core"] = df["score_numeric"] / df["cores_numeric"].where(df["cores_numeric"].gt(0))
    df["performance_per_socket"] = df["score_numeric"] / df["sockets_numeric"].where(df["sockets_numeric"].gt(0))
    return df

def prepare_excel_data(frame):
    # Excel has no timezone type. Export UTC values while retaining app timezones.
    result = frame.copy()
    for col in result.columns:
        if isinstance(result[col].dtype, pd.DatetimeTZDtype):
            result[col] = result[col].dt.tz_convert("UTC").dt.tz_localize(None)
        elif result[col].dtype == "object":
            result[col] = result[col].map(
                lambda value: pd.Timestamp(value).tz_convert("UTC").tz_localize(None)
                if isinstance(value, (datetime, pd.Timestamp)) and value.tzinfo is not None
                and value.utcoffset() is not None else value
            )
    return result


def load_upload(upload):
    data = upload.getvalue()
    name = upload.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    if name.endswith(".json"):
        return pd.read_json(io.BytesIO(data))
    try:
        return parse_delimited_data(data)
    except Exception:
        tables = pd.read_html(io.BytesIO(data))
        if not tables:
            raise ValueError("Could not read the uploaded file.")
        return max(tables, key=lambda x: x.shape[0] * max(1, x.shape[1]))

# --------------------------- sidebar ---------------------------
with st.sidebar:
    st.header("Data source")
    source_mode = st.radio("Load benchmark data", ["Fetch from SPEC", "Upload downloaded data"], index=0)
    uploaded = None
    if source_mode == "Upload downloaded data":
        uploaded = st.file_uploader("Upload SPEC CSV / Excel / JSON", type=["csv", "txt", "tsv", "xlsx", "xls", "json", "html"])
    st.markdown("---")
    st.markdown("**Official source**")
    st.link_button("Open SPEC OSG results", SPEC_HOME, use_container_width=True)
    st.caption("SPEC may change its export format or temporarily rate-limit automated requests. Upload the official CSV if direct ingestion fails.")

raw = None
source_label = ""
load_error = None
if source_mode == "Fetch from SPEC":
    with st.spinner("Fetching the SPEC OSG full-database CSV export…"):
        try:
            raw, final_url, status = fetch_spec_dump(SPEC_CSV_URL, int(time.time() // 3600))
            source_label = f"SPEC online export · HTTP {status}"
        except Exception as exc:
            load_error = str(exc)
else:
    if uploaded is not None:
        try:
            raw = load_upload(uploaded)
            source_label = f"Uploaded: {uploaded.name}"
        except Exception as exc:
            load_error = str(exc)

if raw is None:
    st.info("Load the SPEC OSG database to start. The dashboard attempts the official full-database CSV export automatically.")
    if load_error:
        st.error(f"Data ingestion issue: {load_error}")
    st.markdown("""
    **If automatic ingestion is unavailable**
    1. Open the official SPEC OSG results page.
    2. Select **Dump All Records As CSV**.
    3. Save the downloaded file and upload it in the sidebar.
    """)
    st.link_button("Open SPEC OSG results page", SPEC_HOME)
    st.stop()

df = normalize_data(raw)
if df.empty:
    st.error("The source returned no usable records.")
    st.stop()

st.caption(f"Data source: {source_label} · Loaded {len(df):,} rows · {len(raw.columns):,} original fields")

# --------------------------- filters ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters")
def options_for(col, frame=None):
    frame = df if frame is None else frame
    if col not in frame.columns:
        return []
    vals = frame[col].dropna().astype(str).unique().tolist()
    return sorted([v for v in vals if v.strip()])

vendors = options_for("vendor")
# Vendor is the submitting OEM/company, not necessarily the processor maker.
selected_vendors = st.sidebar.multiselect("Vendor / OEM", vendors, default=vendors, key="vendor_filter_all_oems")
families = options_for("processor_family")
selected_families = st.sidebar.multiselect("Processor family", families, default=families)
types = options_for("benchmark_type")
selected_types = st.sidebar.multiselect("Benchmark type", types, key="benchmark_types")
benchmark_pool = df[df["benchmark_type"].isin(selected_types)] if selected_types else df
benchmarks = options_for("benchmark", benchmark_pool)
selected_benchmarks = st.sidebar.multiselect("Benchmark / version", benchmarks, key="benchmark_ids")
st.sidebar.caption("Empty selections include all options. Performance comparisons use one exact benchmark at a time.")

amd_pool = df[df["processor_family"].eq("AMD EPYC")]
selected_series = st.sidebar.multiselect("EPYC processor series", options_for("epyc_series", amd_pool), key="epyc_series_filter")
if selected_series:
    amd_pool = amd_pool[amd_pool["epyc_series"].isin(selected_series)]
selected_epyc = st.sidebar.multiselect("EPYC processor model", options_for("processor_model", amd_pool), key="epyc_model_filter")
intel_pool = df[df["processor_family"].eq("Intel Xeon")]
selected_intel_series = st.sidebar.multiselect("Intel Xeon series", options_for("xeon_series", intel_pool), key="xeon_series_filter")
if selected_intel_series:
    intel_pool = intel_pool[intel_pool["xeon_series"].isin(selected_intel_series)]
selected_xeon = st.sidebar.multiselect("Intel Xeon processor model", options_for("processor_model", intel_pool), key="xeon_model_filter")
st.sidebar.caption("EPYC selections narrow AMD results; Xeon selections narrow Intel results. Use Processor family to view only one family.")
selected_sockets = st.sidebar.multiselect("Socket count", sorted(df["sockets_numeric"].dropna().unique()), key="socket_filter")
selected_cores = st.sidebar.multiselect("Total physical cores", sorted(df["cores_numeric"].dropna().unique()), key="core_filter")
systems = options_for("system")
selected_systems = st.sidebar.multiselect("OEM / system", systems)
search_text = st.sidebar.text_input("Search all fields", placeholder="EPYC 9654, PowerEdge, SPEC...")
min_score = st.sidebar.number_input("Minimum score (optional)", min_value=0.0, value=0.0, step=1.0)
only_comparable = st.sidebar.checkbox("Only rows with numeric score", value=False)

filtered = df.copy()
if selected_vendors and "vendor" in filtered:
    filtered = filtered[filtered["vendor"].astype(str).isin(selected_vendors)]
if selected_families:
    filtered = filtered[filtered["processor_family"].isin(selected_families)]
if selected_types:
    filtered = filtered[filtered["benchmark_type"].isin(selected_types)]
if selected_benchmarks:
    filtered = filtered[filtered["benchmark"].astype(str).isin(selected_benchmarks)]
filtered = filter_processors(filtered, selected_series, selected_epyc, selected_xeon, selected_intel_series)
if selected_sockets:
    filtered = filtered[filtered["sockets_numeric"].isin(selected_sockets)]
if selected_cores:
    filtered = filtered[filtered["cores_numeric"].isin(selected_cores)]
if selected_systems:
    filtered = filtered[filtered["system"].astype(str).isin(selected_systems)]
if search_text.strip():
    mask = filtered.astype(str).apply(lambda col: col.str.contains(search_text, case=False, regex=False, na=False)).any(axis=1)
    filtered = filtered[mask]
if only_comparable:
    filtered = filtered[filtered["score_numeric"].notna()]
if min_score > 0:
    filtered = filtered[filtered["score_numeric"].fillna(-1) >= min_score]

# --------------------------- KPI cards ---------------------------
amd_n = int(filtered["processor_family"].eq("AMD EPYC").sum())
xeon_n = int(filtered["processor_family"].eq("Intel Xeon").sum())
scored = filtered["score_numeric"].dropna()
cores_n = int(filtered["cores_numeric"].notna().sum())
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Filtered results", f"{len(filtered):,}")
k2.metric("AMD EPYC records", f"{amd_n:,}")
k3.metric("Intel Xeon records", f"{xeon_n:,}")
k4.metric("Numeric scores", f"{len(scored):,}")
k5.metric("Records with core count", f"{cores_n:,}")

# --------------------------- tabs ---------------------------
tab_overview, tab_compare, tab_oem, tab_data, tab_export = st.tabs(
    ["Overview", "EPYC vs Xeon", "OEM / System analysis", "Results explorer", "Export"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        vendor_counts = filtered["vendor"].fillna("Unknown").astype(str).value_counts().head(15).rename_axis("Vendor").reset_index(name="Records")
        if not vendor_counts.empty:
            st.subheader("Results by vendor")
            st.plotly_chart(px.bar(vendor_counts, x="Vendor", y="Records", color="Vendor"), use_container_width=True)
    with right:
        family_counts = filtered["processor_family"].value_counts().rename_axis("Processor family").reset_index(name="Records")
        st.subheader("Processor family distribution")
        if not family_counts.empty:
            st.plotly_chart(px.pie(family_counts, names="Processor family", values="Records", hole=.45), use_container_width=True)
    st.subheader("Benchmark coverage")
    bench_counts = filtered["benchmark"].fillna("Unknown").astype(str).value_counts().head(25).rename_axis("Benchmark").reset_index(name="Records")
    if not bench_counts.empty:
        st.plotly_chart(px.bar(bench_counts, x="Records", y="Benchmark", orientation="h"), use_container_width=True)
    if filtered["date_parsed"].notna().any():
        st.subheader("Publication / submission trend")
        date_counts = filtered.dropna(subset=["date_parsed"]).assign(Month=lambda x: x["date_parsed"].dt.to_period("M").astype(str)).groupby("Month").size().reset_index(name="Records")
        st.plotly_chart(px.line(date_counts, x="Month", y="Records", markers=True), use_container_width=True)

with tab_compare:
    st.subheader("Processor performance analysis")
    st.caption("Choose EPYC and Xeon series/models in the sidebar. Each comparison uses one exact benchmark; socket and core configurations remain separate. Published results also vary with memory, software and tuning.")
    comp = filtered[filtered["processor_family"].isin(["AMD EPYC", "Intel Xeon"])].copy()
    if comp.empty:
        st.warning("No EPYC or Xeon results match the current filters.")
    else:
        chosen_benchmark = st.selectbox("Benchmark to analyze", options_for("benchmark", comp), key="analysis_benchmark")
        one = comp[comp["benchmark"].astype(str).eq(chosen_benchmark)].copy()
        controls = st.columns(3)
        metric_labels = {"Reported score": "score_numeric", "Score per core": "performance_per_core", "Score per socket": "performance_per_socket"}
        metric_label = controls[0].selectbox("Performance metric", list(metric_labels), key="analysis_metric")
        statistic_label = controls[1].selectbox("Summary statistic", ["Median", "Mean", "Best"], key="analysis_statistic")
        direction = controls[2].selectbox("Better performance", ["Higher values", "Lower values"], key="analysis_direction")
        higher = direction == "Higher values"
        statistic = {"Median": "median", "Mean": "mean", "Best": "max" if higher else "min"}[statistic_label]
        metric = metric_labels[metric_label]
        summary = performance_summary(one, metric, statistic)
        if summary.empty:
            st.info("No positive numeric measurements are available for this metric. Per-core and per-socket views also require a positive core/socket count.")
        else:
            summary = summary.sort_values("Performance", ascending=not higher).reset_index(drop=True)
            summary["Configuration"] = summary["processor_model"] + " | " + summary["Sockets"] + " sockets | " + summary["Cores"] + " cores"
            st.caption(f"{chosen_benchmark}: {statistic_label.lower()} {metric_label.lower()}. Zero/missing measurements are excluded. For power or non-CPU benchmarks, confirm metric units and direction in the source result.")
            baseline = st.selectbox("Reference configuration", summary["Configuration"].tolist(), key="analysis_reference")
            reference = float(summary.loc[summary["Configuration"].eq(baseline), "Performance"].iloc[0])
            summary["Relative performance"] = summary["Performance"] / reference if higher else reference / summary["Performance"]
            summary["Difference vs reference (%)"] = (summary["Relative performance"] - 1) * 100
            st.caption("Relative performance is 1.0 for the reference; values above 1.0 indicate better performance in the selected direction. Differences describe published configurations, not an isolated CPU effect.")
            a, b = st.columns(2)
            a.metric("Configurations compared", len(summary))
            b.metric("Scored records", int(summary["Records"].sum()))
            st.dataframe(summary.rename(columns={"processor_model": "Processor", "epyc_series": "EPYC series", "xeon_series": "Xeon series", "processor_family": "Family", "benchmark": "Benchmark"}), use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(summary, x="Configuration", y="Performance", color="processor_family",
                                   hover_data=["Records", "Minimum", "Maximum", "Relative performance"],
                                   labels={"Performance": f"{statistic_label} {metric_label.lower()}", "processor_family": "Family"}), use_container_width=True)
            values = pd.to_numeric(one[metric], errors="coerce")
            valid = one.loc[values.gt(0)].copy()
            valid["Measurement"] = values.loc[valid.index]
            valid["Configuration"] = valid["processor_model"] + " | " + valid["sockets_numeric"].astype("string").fillna("Unknown") + " sockets | " + valid["cores_numeric"].astype("string").fillna("Unknown") + " cores"
            st.plotly_chart(px.box(valid, x="Configuration", y="Measurement", color="processor_family", points="outliers",
                                   hover_data=["vendor", "system", "cpu"], labels={"Measurement": metric_label}), use_container_width=True)
            st.download_button("Download performance comparison CSV", summary.to_csv(index=False).encode("utf-8-sig"),
                               file_name="spec_processor_comparison.csv", mime="text/csv")

with tab_oem:
    st.subheader("OEM and server comparison")
    oem = filtered.copy()
    oem["OEM / System"] = oem["system"].fillna("Unknown").astype(str)
    oem["Processor"] = oem["cpu"].fillna("Not identified").astype(str)
    oem_summary = oem.groupby(["OEM / System", "processor_family"], dropna=False).agg(
        Records=("processor_family", "size"),
        Median_score=("score_numeric", "median"),
        Median_cores=("cores_numeric", "median")
    ).reset_index().sort_values("Records", ascending=False)
    st.dataframe(oem_summary, use_container_width=True, hide_index=True)
    if oem["score_numeric"].notna().any():
        st.plotly_chart(px.scatter(oem.dropna(subset=["score_numeric"]), x="cores_numeric", y="score_numeric",
                                   color="processor_family", hover_name="OEM / System",
                                   hover_data=["Processor", "benchmark", "vendor"],
                                   labels={"cores_numeric":"Physical cores", "score_numeric":"Reported score"}),
                        use_container_width=True)
    st.caption("OEM/system grouping uses the system/model field available in the SPEC export. If the selected configuration does not publish an OEM field, the original fields are still shown in the explorer.")

with tab_data:
    st.subheader("Searchable SPEC result records")
    display_cols = list(raw.columns)
    extra_cols = [c for c in ["vendor", "processor_family", "processor_model", "epyc_series", "xeon_series", "benchmark_type", "benchmark", "system", "cpu", "cores_numeric", "sockets_numeric", "score_numeric", "performance_per_core", "performance_per_socket", "date"] if c in filtered.columns]
    display_cols = list(dict.fromkeys([c for c in display_cols if c in filtered.columns] + extra_cols))
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True, height=620)
    st.caption(f"Showing {len(filtered):,} filtered records from {len(df):,} loaded records.")

with tab_export:
    st.subheader("Download filtered results")
    st.caption("Excel timestamps are exported in UTC without timezone metadata.")
    st.write(f"Your current filters select **{len(filtered):,} records**.")
    csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download filtered CSV", csv_bytes, file_name="spec_osg_filtered_results.csv", mime="text/csv", use_container_width=True)
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        prepare_excel_data(filtered).to_excel(writer, sheet_name="Filtered Results", index=False)
        pd.DataFrame({
            "Metric": ["Loaded records", "Filtered records", "AMD EPYC records", "Intel Xeon records", "Records with numeric scores", "Data source"],
            "Value": [len(df), len(filtered), amd_n, xeon_n, len(scored), source_label]
        }).to_excel(writer, sheet_name="Dashboard Summary", index=False)
    st.download_button("Download filtered Excel workbook", excel_buffer.getvalue(),
                       file_name="spec_osg_benchmark_dashboard.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

st.markdown("---")
st.caption("Source: SPEC OSG public results database. This dashboard is an independent analysis tool and is not affiliated with or endorsed by SPEC. Always verify individual result details and run rules on the official SPEC result pages before using comparisons in customer-facing material.")
