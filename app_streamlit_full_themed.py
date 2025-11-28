import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title='RAG Churn Analyzer', layout='wide', initial_sidebar_state='expanded')

# CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {background: linear-gradient(180deg,#f7fbff 0%,#f0f7ff 35%,#ffffff 100%);}
.header-card {background: linear-gradient(90deg, #0f172a 0%, #0ea5a3 100%); color: white; padding: 14px; border-radius: 12px; box-shadow: 0 6px 18px rgba(2,6,23,0.12);}
.info-card {background: #ffffff; color: #0f172a; border-radius: 10px; padding: 12px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);}
.small-muted {color: #1f2937; font-size: 13px;}
.stButton>button {background: linear-gradient(90deg,#0ea5a3,#06b6d4) !important; color: white; border: none; padding: 8px 14px; border-radius: 8px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-card"><h2>🤖 RAG Customer Churn Analyzer</h2><div class="small-muted">Ask questions like "What\'s the retention trend?" → Get AI answers + charts. Zero dependencies.</div></div>', unsafe_allow_html=True)

# RAG Knowledge Base (keyword matching)
KNOWLEDGE = {
    "churn": "Churn rate = % customers who leave. High churn (>20%) = product/service issues. Low churn (<5%) = strong loyalty.",
    "retention": "Retention shows % customers who stay over time. Healthy: >80% first 6 months. Sharp drops = investigate that month.",
    "cohort": "Cohort analysis: group by signup month, track retention. Newer cohorts fading faster = worsening retention.",
    "contract": "Month-to-month contracts churn faster than 1/2-year contracts. Focus retention on high-risk segments.",
    "trend": "Retention trend: steady decline = normal. Sudden drops = churn risk point. Flat line = excellent."
}

def rag_answer(question, metrics):
    """Keyword-based RAG + smart templating"""
    q_lower = question.lower()
    context = []
    
    for key, text in KNOWLEDGE.items():
        if key in q_lower:
            context.append(text)
    
    answer = f"**Q: {question}**\n\n"
    
    if any(word in q_lower for word in ["retention", "trend"]):
        answer += "📈 **Retention Analysis:** Customers start strong but decline over time (normal pattern). "
        answer += f"Key metrics: {metrics}. Watch for sharp drops in the chart above."
    elif "churn" in q_lower:
        answer += "🔴 **Churn Analysis:** High churn indicates issues with product fit, pricing, or support. "
        answer += f"Current rate: {metrics}. Use cohort/segment views to find causes."
    else:
        answer += "📊 **Overview:** Check metrics, retention curve, and cohort heatmap for full picture. "
        answer += f"Dataset has {metrics} total customers."
    
    return answer

# Data Upload
col1, col2 = st.columns([2,1])
with col1:
    uploaded_file = st.file_uploader("📁 Upload Churn CSV", type="csv")
with col2:
    st.markdown('<div class="info-card"><strong>✅ Works Instantly</strong><br>No ML installs needed.</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists("/mnt/data/chrundata.csv"):
    df = pd.read_csv("/mnt/data/chrundata.csv")
    st.success("✅ Loaded sample dataset")
else:
    st.error("❌ Upload CSV or add sample at /mnt/data/chrundata.csv")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Configure")
cols = df.columns.tolist()
tenure_col = st.sidebar.selectbox("Tenure column", ["(none)"] + cols)
churn_col = st.sidebar.selectbox("Churn column", ["(none)"] + cols)

question = st.sidebar.text_input("🤔 Ask:", value="What's the retention trend?")
if st.sidebar.button("🚀 Analyze", type="primary"):

    # Metrics
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Customers", len(df))
    
    churn_rate = 0
    if churn_col != "(none)":
        churn_rate = df[churn_col].astype(str).str.lower().str.contains("yes|true|1|churn", na=False).mean()
        col_b.metric("Churn Rate", f"{churn_rate:.1%}")
    
    if tenure_col != "(none)":
        month1_ret = (df[tenure_col] >= 1).mean()
        col_c.metric("Month 1 Retention", f"{month1_ret:.1%}")

    # Retention Chart
    if tenure_col != "(none)":
        total = len(df)
        max_m = int(df[tenure_col].dropna().max())
        months = list(range(max_m + 1))
        retention = [(df[tenure_col] >= m).sum() / total for m in months]
        
        ret_df = pd.DataFrame({"Month": months, "Retention": retention})
        fig = px.line(ret_df, x="Month", y="Retention", markers=True, title="📈 Retention Trend")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # RAG Answer
    st.markdown("---")
    metrics_summary = f"{len(df)} customers, churn {churn_rate:.1%}"
    answer = rag_answer(question, metrics_summary)
    st.markdown(answer)

    # Dataset preview
    st.markdown("---")
    st.dataframe(df.head(), use_container_width=True)

else:
    st.info("👈 Configure columns + ask a question, then click Analyze")
