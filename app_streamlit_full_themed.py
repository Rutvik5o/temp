import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title='RAG Customer Churn Analyzer (No Dependencies)',
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

# BUILT-IN "RAG" (keyword matching, no external libs)
EXPLANATION_DOCS = {
    "churn": """
Churn rate is the percentage of customers who stop using a service over a period.
High churn (>20%) indicates problems with product fit, pricing, or support.
Low churn (<5%) shows strong customer loyalty.
""",
    "retention": """
Retention shows what fraction of customers stay over time.
Healthy retention stays above 80% for first 6 months, then gradually declines.
Sharp drops indicate specific risk points to investigate.
""",
    "cohort": """
Cohort analysis groups customers by signup month and tracks retention over time.
Newer cohorts (bottom rows) fading faster than older ones = worsening retention.
Diagonal line shows month 0 retention (should be ~100%).
""",
    "segment": """
Segment analysis compares groups like Contract type or PaymentMethod.
Month-to-month contracts usually churn faster than annual contracts.
Focus retention efforts on high-churn segments first.
""",
    "trend": """
Retention trend: look at the line chart. Steady decline = normal.
Sudden drops = investigate that month/cohort/segment.
Flat line = excellent retention.
"""
}

def simple_rag(query):
    query_lower = query.lower()
    matches = []
    for key, text in EXPLANATION_DOCS.items():
        if key in query_lower:
            matches.append(text)
    return "\n\n".join(matches) if matches else "No specific documentation found for this question."

def fake_llm(prompt):
    """Smart template-based "LLM" using RAG context + metrics"""
    lines = prompt.split("\n")
    metrics = ""
    context = ""
    question = ""
    
    for line in lines:
        if "[DATA METRICS]" in line:
            metrics = "\n".join(l for l in lines if any(k in l for k in ["churn", "Retention", "%"]))
        elif "[DOC CONTEXT]" in line:
            context = "\n".join(l.strip() for l in lines if len(l.strip()) > 20 and not line.startswith("["))
        elif "QUESTION]" in line:
            question = lines[lines.index(line)+1].strip()
    
    # Generate answer based on keywords
    answer = f"**Analysis for: {question}**\n\n"
    
    if "retention" in question.lower() or "trend" in question.lower():
        answer += "The retention curve shows customers staying over time. "
        answer += "It starts high and declines gradually, which is normal. "
        answer += "Watch for sharp drops - those are churn risks to investigate."
    elif "churn" in question.lower():
        answer += "Churn rate measures customers leaving. "
        answer += "High churn suggests product or service issues. "
        answer += "Use cohort and segment views to find causes."
    else:
        answer += "Key metrics and charts above show the full picture. "
        answer += "Use filters and column selectors for deeper analysis."
    
    if metrics:
        answer += f"\n\n**Key metrics:**\n{metrics}"
    
    return answer

# Header
st.markdown(
    '<div class="header-card"><h2 style="margin:0">RAG Churn Analyzer (Zero Dependencies)</h2>'
    '<div class="small-muted">Upload CSV, ask questions, get smart answers + charts. No external ML needed.</div></div>',
    unsafe_allow_html=True
)

# Data upload (same as before)
col_left, col_right = st.columns([2, 1])
with col_left:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
with col_right:
    st.markdown('<div class="info-card"><strong>Works instantly</strong><br>No heavy ML installs needed.</div>', unsafe_allow_html=True)

SAMPLE = "/mnt/data/chrundata.csv"
if uploaded is None:
    if os.path.exists(SAMPLE):
        df = pd.read_csv(SAMPLE)
        st.success("Loaded sample dataset.")
    else:
        st.warning("Upload CSV or add sample at /mnt/data/chrundata.csv")
        st.stop()
else:
    df = pd.read_csv(uploaded)

# Sidebar (simplified)
st.sidebar.header("Analysis Setup")
cols = df.columns.tolist()
tenure_col = st.sidebar.selectbox("Tenure column", ["(none)"] + cols, index=(cols.index("tenure")+1 if "tenure" in cols else 0))
churn_col = st.sidebar.selectbox("Churn column", ["(none)"] + cols, index=(cols.index("Churn")+1 if "Churn" in cols else 0))

question = st.sidebar.text_input("Ask about churn:", value="What's the retention trend?")
if st.sidebar.button("Analyze"):

    # Metrics
    st.metric("Total customers", len(df))
    if churn_col != "(none)":
        churn_rate = df[churn_col].astype(str).str.lower().str.contains("yes|true|1|churn").mean()
        st.metric("Churn rate", f"{churn_rate:.1%}")

    # Retention chart
    if tenure_col != "(none)":
        total = len(df)
        max_m = int(df[tenure_col].dropna().max())
        months = list(range(max_m + 1))
        retention = [(df[tenure_col] >= m).sum() / total for m in months]
        ret_df = pd.DataFrame({"month": months, "retention": retention})
        
        fig = px.line(ret_df, x="month", y="retention", markers=True, title="Retention Trend")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # RAG Answer
    st.subheader("🤖 RAG Answer")
    context = simple_rag(question)
    metrics_text = f"Churn: {churn_rate:.1% if 'churn_rate' in locals() else 'N/A'} | Customers: {len(df)}"
    answer = fake_llm(f"[DATA METRICS]\n{metrics_text}\n\n[DOC CONTEXT]\n{context}\n\n[QUESTION]\n{question}")
    st.markdown(answer)
