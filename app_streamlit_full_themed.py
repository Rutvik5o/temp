import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

############################
# BASIC LLM PLACEHOLDER
############################
# TODO: Replace this function with your real LLM API call
# (e.g., OpenAI, Groq, etc.)
def call_llm(prompt: str) -> str:
    # SIMPLE FALLBACK so app runs even without an API:
    # Just return the last lines of the prompt.
    return "LLM placeholder: connect a real LLM API here to generate a rich answer.\n\nPrompt preview:\n" + prompt[-500:]


############################
# RAG SETUP (STATIC DOCS)
############################

EXPLANATION_DOCS = [
    """
Churn rate is the percentage of customers who stop using a service over a period.
Retention is the percentage of customers who stay over time.
A retention curve usually starts high and then gradually declines as customers leave.
If the curve drops sharply after a certain month, that is a risk point for churn.
""",
    """
Cohort analysis groups customers by the month they started and tracks what fraction remain active in later months.
A cohort heatmap shows retention percentages by cohort and by months since signup.
Rows (cohorts) with faster color fading indicate worse retention.
""",
    """
Segment analysis compares retention between groups such as Contract type, InternetService, or PaymentMethod.
If month-to-month contracts churn faster than two-year contracts, long-term contracts retain better.
Look for lines that drop more steeply in segmented retention charts.
"""
]

@st.cache_resource(show_spinner=False)
def build_rag_index(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(docs)
    nn = NearestNeighbors(n_neighbors=min(3, len(docs)), metric="cosine")
    nn.fit(embs)
    return model, nn, embs

def rag_retrieve(model, nn, embs, docs, query, top_k=3):
    q_emb = model.encode([query])
    _, idx = nn.kneighbors(q_emb, n_neighbors=min(top_k, len(docs)))
    return [docs[i] for i in idx[0]]

model, nn, embs = build_rag_index(EXPLANATION_DOCS)

############################
# STREAMLIT UI + CSS
############################

st.set_page_config(
    page_title='RAG Customer Churn Analyzer',
    layout='wide',
    initial_sidebar_state='expanded'
)

CSS = '''
<style>
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg,#f7fbff 0%,#f0f7ff 35%,#ffffff 100%);
  padding-top: 0.5rem;
}
.header-card {
  background: linear-gradient(90deg, #0f172a 0%, #0ea5a3 100%);
  color: white;
  padding: 14px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.12);
}
.info-card {
  background: #ffffff;
  color: #0f172a;
  border-radius: 10px;
  padding: 12px;
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
  font-size: 14px;
  line-height: 1.4;
}
.small-muted {
  color: #1f2937;
  font-size: 13px;
}
.stButton>button {
  background: linear-gradient(90deg,#0ea5a3,#06b6d4) !important;
  color: white;
  border: none;
  padding: 8px 14px;
  border-radius: 8px;
  font-weight: 600;
}
[data-testid="stDataFrameContainer"] th {
  background-color: #f1f5f9 !important;
  color: #0f172a !important;
}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="header-card"><h2 style="margin:0">RAG‑Based Customer Churn Analyzer</h2>'
    '<div class="small-muted">Upload a churn dataset, ask questions like “What’s the retention trend?”, and get AI‑generated answers plus charts.</div></div>',
    unsafe_allow_html=True
)
st.write("")

############################
# DATA LOAD
############################

col_left, col_right = st.columns([2, 1])
with col_left:
    uploaded = st.file_uploader("Upload CSV (leave empty to use sample)", type=["csv"])
    st.markdown(
        '<div class="small-muted">Accepted: CSV. If empty, sample dataset at /mnt/data/chrundata.csv will be used (if present).</div>',
        unsafe_allow_html=True
    )
with col_right:
    st.markdown(
        '<div class="info-card"><strong>Tips</strong><br>1) Select correct columns in the sidebar.<br>'
        '2) Use filters to segment data.<br>3) Ask a question and click Run analysis.</div>',
        unsafe_allow_html=True
    )

SAMPLE = "/mnt/data/chrundata.csv"
if uploaded is None:
    if os.path.exists(SAMPLE):
        try:
            df = pd.read_csv(SAMPLE)
            st.success("Loaded sample dataset.")
        except Exception:
            st.error("Sample dataset could not be read. Upload a CSV.")
            st.stop()
    else:
        st.error("No sample dataset found and no upload. Please upload your CSV.")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Uploaded file could not be read: {e}")
        st.stop()

############################
# SIDEBAR CONFIG
############################

st.sidebar.header("Configure analysis")
cols = df.columns.tolist()

cust_id_col = st.sidebar.selectbox(
    "Customer ID column", ["(none)"] + cols,
    index=(1 if "customerID" in cols else 0)
)
churn_col = st.sidebar.selectbox(
    "Churn flag column", ["(none)"] + cols,
    index=(cols.index("Churn") + 1 if "Churn" in cols else 0)
)
tenure_col = st.sidebar.selectbox(
    "Tenure (months) column (optional)", ["(none)"] + cols,
    index=(cols.index("tenure") + 1 if "tenure" in cols else 0)
)
signup_col = st.sidebar.selectbox(
    "Signup / start date column (optional)", ["(none)"] + cols
)
last_col = st.sidebar.selectbox(
    "Last-active / end date column (optional)", ["(none)"] + cols
)

st.sidebar.markdown("---")
question_text = st.sidebar.text_input(
    "Ask a churn question",
    value="What’s the retention trend?"
)

candidate_segments = ["Contract", "InternetService", "PaymentMethod", "Payment Method", "Plan"]
possible_segments = [c for c in candidate_segments if c in cols]

st.sidebar.markdown("---")
st.sidebar.header("Segment filters (optional)")
segment_filters = {}
for i, s in enumerate(possible_segments):
    vals = sorted(df[s].dropna().unique().tolist())
    if not vals:
        segment_filters[s] = []
        st.sidebar.write(f"{s}: (no values)")
    else:
        segment_filters[s] = st.sidebar.multiselect(
            f"Filter {s}", options=vals, default=vals
        )

st.sidebar.markdown("---")
run = st.sidebar.button("Run analysis")

############################
# HELPERS
############################

def normalize_churn(series):
    if series.dtype == object:
        return series.map(
            lambda x: 1
            if str(x).strip().lower() in ['yes','y','true','1','t','churn','exited']
            else (
                0 if str(x).strip().lower() in ['no','n','false','0','f','stay','retained']
                else None
            )
        )
    return series

def compute_retention_trend(df2, tenure_col_name):
    if tenure_col_name == "(none)" or tenure_col_name not in df2.columns:
        return None, None
    if not pd.api.types.is_numeric_dtype(df2[tenure_col_name]):
        return None, None
    if len(df2) == 0:
        return None, None

    total = len(df2)
    max_m = int(df2[tenure_col_name].dropna().astype(int).max())
    months = list(range(0, max_m + 1))
    retention = [(m, (df2[tenure_col_name] >= m).sum() / total) for m in months]
    ret_df = pd.DataFrame(retention, columns=["month", "retention_rate"])

    stats = {
        "start": ret_df["retention_rate"].iloc[0],
        "mid_idx": min(6, len(ret_df)-1),
        "mid": ret_df["retention_rate"].iloc[min(6, len(ret_df)-1)],
        "end": ret_df["retention_rate"].iloc[-1],
        "last_month": int(ret_df["month"].iloc[-1]),
    }
    return ret_df, stats

############################
# MAIN ANALYSIS
############################

if run:
    df2 = df.copy()
    for s, selvals in segment_filters.items():
        if selvals:
            df2 = df2[df2[s].isin(selvals)]

    st.markdown('<div class="info-card"><strong>Filtered sample</strong></div>', unsafe_allow_html=True)
    st.dataframe(df2.head())

    churn_key = None
    churn_rate = None
    if churn_col != "(none)":
        df2["_churn_mapped"] = normalize_churn(df2[churn_col])
        churn_key = "_churn_mapped"
        try:
            churn_rate = float(df2[churn_key].dropna().astype(float).mean())
        except Exception:
            churn_rate = None

    c1, c2, c3 = st.columns(3)
    if churn_rate is not None:
        c1.metric("Overall churn rate", f"{churn_rate:.2%}")
    else:
        c1.metric("Overall churn rate", "N/A")
    c2.metric("Total (filtered)", len(df2))
    if tenure_col != "(none)" and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        first_month_ret = (df2[tenure_col] >= 1).sum() / len(df2) if len(df2) > 0 else 0
        c3.metric("Retention @ month 1", f"{first_month_ret:.2%}")
    else:
        c3.metric("Retention @ month 1", "N/A")

    st.markdown("---")
    st.subheader("Visualizations")

    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="info-card"><strong>Churn distribution</strong></div>', unsafe_allow_html=True)
        if churn_key is not None:
            tmp = df2[churn_key].dropna().astype(int)
            if len(tmp) > 0:
                fig = px.pie(
                    values=tmp.value_counts().values,
                    names=tmp.value_counts().index.astype(str),
                    title="Churn (0=retain,1=churn)",
                    hole=0.35
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No churn values after filtering.")
        else:
            st.info("Select a churn column in the sidebar to view distribution.")

    with right:
        st.markdown('<div class="info-card"><strong>Retention trend (tenure-based)</strong></div>', unsafe_allow_html=True)
        ret_df, stats = compute_retention_trend(df2, tenure_col)
        if ret_df is not None:
            fig = px.line(
                ret_df, x="month", y="retention_rate",
                markers=True, title="Retention trend over tenure"
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download retention CSV",
                data=ret_df.to_csv(index=False).encode("utf-8"),
                file_name="retention_by_tenure.csv",
                key="down_ret"
            )
        else:
            st.info("Tenure column not selected or not numeric.")

    ############################
    # RAG‑BASED ANSWER
    ############################
    st.markdown("---")
    st.subheader("RAG‑based answer to your question")

    # Build metrics snippet for the LLM
    metrics_lines = []
    if churn_rate is not None:
        metrics_lines.append(f"Overall churn rate: {churn_rate:.2%}")
    if stats is not None:
        metrics_lines.append(f"Retention start: {stats['start']:.1%}")
        metrics_lines.append(f"Retention mid (month {stats['mid_idx']}): {stats['mid']:.1%}")
        metrics_lines.append(f"Retention end (month {stats['last_month']}): {stats['end']:.1%}")

    metrics_text = "\n".join(metrics_lines) if metrics_lines else "No retention metrics available."

    # Retrieve explanatory context
    rag_context = rag_retrieve(model, nn, embs, EXPLANATION_DOCS, question_text)
    context_text = "\n\n".join(rag_context)

    prompt = f"""
You are a senior customer churn analyst.

Use BOTH the data metrics and the documentation context below to answer the user's question.
Focus on being clear and business‑friendly, in 3–6 sentences.

[DATA METRICS]
{metrics_text}

[DOC CONTEXT]
{context_text}

[USER QUESTION]
{question_text}

[ANSWER]
"""

    answer = call_llm(prompt)
    st.markdown(f"**Question:** {question_text}")
    st.markdown("**Answer (RAG‑based):**")
    st.write(answer)

    ############################
    # COHORT ANALYSIS
    ############################
    st.markdown("---")
    st.subheader("Cohort Analysis")
    if signup_col != "(none)" and last_col != "(none)":
        try:
            df2[signup_col] = pd.to_datetime(df2[signup_col], errors="coerce")
            df2[last_col] = pd.to_datetime(df2[last_col], errors="coerce")
            df2["cohort_month"] = df2[signup_col].dt.to_period("M").dt.to_timestamp()
            df2["activity_month"] = df2[last_col].dt.to_period("M").dt.to_timestamp()

            def month_diff(a, b):
                return (b.year - a.year) * 12 + (b.month - a.month)

            df2["period_number"] = df2.apply(
                lambda r: month_diff(r["cohort_month"], r["activity_month"])
                if pd.notnull(r["cohort_month"]) and pd.notnull(r["activity_month"])
                else np.nan,
                axis=1
            )
            idcol = cust_id_col if cust_id_col != "(none)" else df2.columns[0]
            cohorts = df2.groupby(["cohort_month", "period_number"]).agg(
                n_customers=(idcol, "count")
            ).reset_index()
            cohort_sizes = df2.groupby("cohort_month").agg(
                cohort_size=(idcol, "count")
            ).reset_index()
            cohorts = cohorts.merge(cohort_sizes, on="cohort_month")
            cohorts["retention"] = cohorts["n_customers"] / cohorts["cohort_size"]
            pivot = cohorts.pivot(
                index="cohort_month", columns="period_number", values="retention"
            ).fillna(0)

            heat_z = (pivot.values * 100).round(1)
            heat_x = [int(x) for x in pivot.columns.tolist()]
            heat_y = [str(d.date()) for d in pivot.index.tolist()]

            heatmap = go.Figure(
                data=go.Heatmap(
                    z=heat_z,
                    x=heat_x,
                    y=heat_y,
                    colorscale="YlGnBu",
                    text=heat_z,
                    hovertemplate="Cohort: %{y}<br>Months: %{x}<br>Retention: %{z:.1f}%"
                )
            )
            heatmap.update_layout(
                title="Cohort retention heatmap (percent)",
                xaxis_title="Months since cohort",
                yaxis_title="Cohort month",
                height=500
            )
            st.plotly_chart(heatmap, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to compute cohort pivot: {e}")
    else:
        st.info("Select both Signup and Last-active columns to enable cohort heatmap.")

    ############################
    # SEGMENTED RETENTION
    ############################
    st.markdown("---")
    st.subheader("Segmented retention comparison")
    seg_col = None
    for s in ["Contract", "InternetService", "PaymentMethod"]:
        if s in df2.columns:
            seg_col = s
            break
    if seg_col is not None and tenure_col != "(none)" and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        segs = df2[seg_col].dropna().unique().tolist()[:6]
        fig = go.Figure()
        max_m = int(df2[tenure_col].dropna().astype(int).max())
        for s in segs:
            d = df2[df2[seg_col] == s]
            if len(d) == 0:
                continue
            retention = [(m, (d[tenure_col] >= m).sum() / len(d)) for m in range(0, max_m + 1)]
            rdf = pd.DataFrame(retention, columns=["month", "retention_rate"])
            fig.add_trace(
                go.Scatter(
                    x=rdf["month"],
                    y=rdf["retention_rate"],
                    mode="lines+markers",
                    name=str(s)
                )
            )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            title=f"Retention by {seg_col}",
            xaxis_title="Months",
            yaxis_title="Retention"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Segment column not present or tenure missing — segmented retention unavailable.")

    st.markdown("---")
    csv_all = df2.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered dataset (CSV)",
        data=csv_all,
        file_name="filtered_dataset.csv",
        key="down_all"
    )

    st.markdown(
        '<div class="small-muted">RAG app: metrics + doc context → LLM answer, with interactive charts.</div>',
        unsafe_allow_html=True
    )

else:
    st.info("Select options in the sidebar, type a question, and click Run analysis.")

