import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ---------------------------------------------------------------------
# PAGE CONFIG (CLEAN — NO HIDDEN INDENTATION ISSUES)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analyzer — Dark PowerBI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# COLOR PALETTE
# ---------------------------------------------------------------------
PALETTE = px.colors.qualitative.Plotly
ACCENT = "#0ea5a3"

# ---------------------------------------------------------------------
# CSS THEME
# ---------------------------------------------------------------------
CSS = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background: #0A0F1A;
    color: #E6EEF6;
}
.metric-card {
    background: #101623;
    padding: 12px;
    border-radius: 10px;
    border-left: 6px solid #0ea5a3;
}
.metric-sub {
    color: #96A3B0;
    font-size: 13px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
}
.chart-card {
    background: #111827;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 14px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown(
    """
    <div style='background:#0F172A;padding:16px;border-radius:10px;'>
        <h2 style='margin:0;color:white;'>📊 Customer Churn Analyzer — Dark PowerBI</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------------------
uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin")
    st.stop()

df = pd.read_csv(uploaded)

# ---------------------------------------------------------------------
# COLUMN SELECTION
# ---------------------------------------------------------------------
cols = df.columns.tolist()

cust_col = st.sidebar.selectbox("Customer ID", ["(none)"] + cols)
churn_col = st.sidebar.selectbox("Churn Column", ["(none)"] + cols)
tenure_col = st.sidebar.selectbox("Tenure Column", ["(none)"] + cols)

run = st.sidebar.button("Run Analysis")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def normalize_churn(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.lower().str.strip()
    return s.map({"yes":1,"no":0,"true":1,"false":0,"1":1,"0":0})

def apply_layout(fig):
    fig.update_layout(
        paper_bgcolor="#0A0F1A",
        plot_bgcolor="#0A0F1A",
        font_color="#E6EEF6",
        margin=dict(l=20,r=20,t=40,b=120)
    )
    fig.update_xaxes(automargin=True, tickangle=-40)
    fig.update_yaxes(automargin=True)
    return fig

# ---------------------------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------------------------
if run:

    df2 = df.copy()

    # KPI VALUES
    total_customers = len(df2)

    # churn
    if churn_col != "(none)":
        df2["_churn"] = normalize_churn(df2[churn_col])
        churn_rate = df2["_churn"].mean()
    else:
        churn_rate = None

    # tenure
    if tenure_col != "(none)":
        df2[tenure_col] = pd.to_numeric(df2[tenure_col], errors="coerce")
        avg_tenure = df2[tenure_col].mean()
    else:
        avg_tenure = None

    # -----------------------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------------------
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
        <div class='metric-card'>
            <div class='metric-sub'>Total Customers</div>
            <div class='metric-value'>{total_customers}</div>
        </div>
    """, unsafe_allow_html=True)

    churn_val = f"{churn_rate:.1%}" if churn_rate is not None else "N/A"
    c2.markdown(f"""
        <div class='metric-card'>
            <div class='metric-sub'>Churn Rate</div>
            <div class='metric-value'>{churn_val}</div>
        </div>
    """, unsafe_allow_html=True)

    avg_tenure_val = f"{avg_tenure:.1f}" if avg_tenure is not None else "N/A"
    c3.markdown(f"""
        <div class='metric-card'>
            <div class='metric-sub'>Average Tenure</div>
            <div class='metric-value'>{avg_tenure_val}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------------------------------------------
    # BAR CHART — CHURN BY CONTRACT (IF EXISTS)
    # -----------------------------------------------------------------
    if "Contract" in df2.columns and churn_rate is not None:
        st.markdown("<div class='chart-card'><b>Churn by Contract</b></div>", unsafe_allow_html=True)
        temp = df2.groupby("Contract")["_churn"].mean().reset_index()
        fig = px.bar(temp, x="Contract", y="_churn", color="Contract", color_discrete_sequence=PALETTE)
        fig.update_yaxes(title="Churn Rate")
        fig = apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # BAR — PAYMENT METHOD
    # -----------------------------------------------------------------
    if "PaymentMethod" in df2.columns and churn_rate is not None:
        st.markdown("<div class='chart-card'><b>Churn by Payment Method</b></div>", unsafe_allow_html=True)
        temp = df2.groupby("PaymentMethod")["_churn"].mean().reset_index()
        fig = px.bar(temp, x="PaymentMethod", y="_churn", color="PaymentMethod", color_discrete_sequence=PALETTE)
        fig.update_yaxes(title="Churn Rate")
        fig = apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # BAR — INTERNET SERVICE
    # -----------------------------------------------------------------
    if "InternetService" in df2.columns and churn_rate is not None:
        st.markdown("<div class='chart-card'><b>Churn by Internet Service</b></div>", unsafe_allow_html=True)
        temp = df2.groupby("InternetService")["_churn"].mean().reset_index()
        fig = px.bar(temp, x="InternetService", y="_churn", color="InternetService", color_discrete_sequence=PALETTE)
        fig.update_yaxes(title="Churn Rate")
        fig = apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

