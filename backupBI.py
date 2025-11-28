import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import os, io

st.set_page_config(page_title='Customer Churn Analyzer — PowerBI Style', layout='wide',
                   initial_sidebar_state='expanded')

# Enhanced PowerBI-style CSS
CSS = '''
<style>
/* Page background */
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg,#f8fafc 0%,#e2e8f0 50%,#f1f5f9 100%);
  padding-top: 0.5rem;
}
/* PowerBI Header */
.header-card {
  background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #0ea5a3 100%);
  color: white;
  padding: 20px;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(30,41,59,0.3);
}
/* PowerBI Cards */
.metric-card {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 8px 25px rgba(15,23,42,0.1);
  border-left: 5px solid #0ea5a3;
}
.info-card {
  background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
  color: #0f172a;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
  font-size: 14px;
  line-height: 1.5;
}
/* Small text */
.small-muted {
  color: #64748b;
  font-size: 13px;
}
/* Sidebar */
section[data-testid="stSidebar"] .css-1lcbmhc {
  background: linear-gradient(180deg,#ffffff, #f8fafc);
  border-radius: 12px;
  padding: 16px;
}
/* Buttons */
.stButton>button {
  background: linear-gradient(135deg,#0ea5a3,#06b6d4) !important;
  color: white !important;
  border: none !important;
  padding: 10px 20px !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 15px rgba(14,165,163,0.3) !important;
}
/* Dataframe */
[data-testid="stDataFrameContainer"] th {
  background: linear-gradient(135deg, #f1f5f9, #e2e8f0) !important;
  color: #0f172a !important;
  font-weight: 600 !important;
}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

# Enhanced Header
st.markdown('''
<div class="header-card">
  <h1 style="margin:0; font-size:2.2em;">📊 Customer Churn Analyzer</h1>
  <div style="font-size:16px; margin-top:8px;">PowerBI-style analytics • Interactive dashboards • Export-ready insights</div>
</div>
''', unsafe_allow_html=True)
st.markdown("")

# KPI Cards Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h3>Total Customers</h3></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>Churn Rate</h3></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>Month 1 Retention</h3></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h3>Avg Tenure</h3></div>', unsafe_allow_html=True)

# Upload area
col_left, col_right = st.columns([2,1])
with col_left:
    uploaded = st.file_uploader("📁 Upload CSV (or use sample)", type=["csv"], key="uploader_powerbi")
    st.markdown('<div class="small-muted">CSV format • Sample available at /mnt/data/chrundata.csv</div>', unsafe_allow_html=True)
with col_right:
    st.markdown('''
    <div class="info-card">
    <strong>🚀 Quick Start</strong><br>
    1️⃣ Select columns in sidebar<br>
    2️⃣ Apply segment filters<br>
    3️⃣ Click <strong>Run Analysis</strong>
    </div>
    ''', unsafe_allow_html=True)

# Load data (unchanged)
SAMPLE = "/mnt/data/chrundata.csv"
if uploaded is None:
    if os.path.exists(SAMPLE):
        try:
            df = pd.read_csv(SAMPLE)
            st.success("✅ Loaded sample dataset")
        except Exception as e:
            st.error("❌ Sample dataset error. Upload CSV.")
            st.stop()
    else:
        st.error("❌ No sample + no upload. Please upload CSV.")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df):,} rows")
    except Exception as e:
        st.error(f"❌ Upload error: {e}")
        st.stop()

# Sidebar (unchanged structure, enhanced styling)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configure Analysis")
cols = df.columns.tolist()
cust_id_col = st.sidebar.selectbox("👤 Customer ID", ["(none)"] + cols, index=(1 if "customerID" in cols else 0), key="sid_cust_power")
churn_col = st.sidebar.selectbox("❌ Churn Flag", ["(none)"] + cols, index=(cols.index("Churn")+1 if "Churn" in cols else 0), key="sid_churn_power")
tenure_col = st.sidebar.selectbox("📅 Tenure (months)", ["(none)"] + cols, index=(cols.index("tenure")+1 if "tenure" in cols else 0), key="sid_tenure_power")
signup_col = st.sidebar.selectbox("📆 Signup Date", ["(none)"] + cols, key="sid_signup_power")
last_col = st.sidebar.selectbox("📆 Last Active", ["(none)"] + cols, key="sid_last_power")

# Filters
candidate_segments = ["Contract","InternetService","PaymentMethod","Payment Method","Plan"]
possible_segments = [c for c in candidate_segments if c in cols]
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filters")
segment_filters = {}
for i, s in enumerate(possible_segments):
    vals = sorted(df[s].dropna().unique().tolist())
    if vals:
        segment_filters[s] = st.sidebar.multiselect(f"🔧 {s}", options=vals, default=vals[:3], key=f"seg_power_{i}")

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Analysis", key="run_power", type="primary")

# Helper functions (unchanged)
def normalize_churn(series):
    if series.dtype == object:
        return series.map(lambda x: 1 if str(x).strip().lower() in ['yes','y','true','1','t','churn','exited'] else (0 if str(x).strip().lower() in ['no','n','false','0','f','stay','retained'] else None))
    return series

# Main analysis
if run:
    df2 = df.copy()
    for s, selvals in segment_filters.items():
        if selvals:
            df2 = df2[df2[s].isin(selvals)]
    
    st.markdown('<div class="info-card"><strong>📋 Filtered Dataset</strong> • {} rows'.format(len(df2)), unsafe_allow_html=True)
    st.dataframe(df2.head(10), use_container_width=True, height=200)

    # Enhanced KPI Cards
    churn_key = None
    churn_rate = None
    if churn_col != "(none)":
        df2["_churn_mapped"] = normalize_churn(df2[churn_col])
        churn_key = "_churn_mapped"
        churn_rate = float(df2[churn_key].dropna().astype(float).mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Customers", len(df2), delta="Filtered")
    if churn_rate:
        c2.metric("🔴 Churn Rate", f"{churn_rate:.1%}", delta=f"{churn_rate*100:.0f}%")
    else:
        c2.metric("🔴 Churn Rate", "N/A")
    
    if tenure_col != "(none)" and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        first_month_ret = (df2[tenure_col] >= 1).sum() / len(df2)
        avg_tenure = df2[tenure_col].mean()
        c3.metric("🟢 Month 1 Retention", f"{first_month_ret:.1%}", delta=f"+{first_month_ret:.0%}")
        c4.metric("📊 Avg Tenure", f"{avg_tenure:.1f} months", delta=f"{avg_tenure:.0f}mo")
    else:
        c3.metric("🟢 Month 1 Retention", "N/A")
        c4.metric("📊 Avg Tenure", "N/A")

    st.markdown("---")

    # POWERBI-STYLE DASHBOARD
    row1_col1, row1_col2 = st.columns([2,1])
    
    with row1_col1:
        st.markdown('<div class="info-card"><strong>📈 Retention Trend</strong></div>', unsafe_allow_html=True)
        if tenure_col != "(none)" and pd.api.types.is_numeric_dtype(df2[tenure_col]) and len(df2)>0:
            total = len(df2)
            max_m = min(int(df2[tenure_col].dropna().astype(int).max()), 24)
            months = list(range(0, max_m+1))
            retention = [(df2[tenure_col] >= m).sum() / total for m in months]
            ret_df = pd.DataFrame(retention, columns=["retention_rate"])
            ret_df["month"] = months
            
            # PowerBI-style line chart
            fig = px.line(ret_df, x="month", y="retention_rate", 
                         markers=True, line_shape='spline',
                         title="Customer Retention Over Time",
                         color_discrete_sequence=['#0ea5a3'])
            fig.update_traces(line=dict(width=4), marker=dict(size=8))
            fig.update_layout(
                plot_bgcolor='rgba(248,250,252,1)',
                paper_bgcolor='rgba(255,255,255,0)',
                font=dict(size=12),
                height=400,
                showlegend=False,
                yaxis=dict(tickformat=".0%", gridcolor="#e2e8f0")
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
            
            st.download_button("📥 Download Retention Data", 
                             data=ret_df.to_csv(index=False).encode("utf-8"), 
                             file_name="retention_trend.csv")
    
    with row1_col2:
        st.markdown('<div class="info-card"><strong>🍰 Churn Distribution</strong></div>', unsafe_allow_html=True)
        if churn_key is not None:
            tmp = df2[churn_key].dropna().astype(int)
            if len(tmp)>0:
                labels = ['Retained 🟢', 'Churned 🔴']
                fig = px.pie(values=tmp.value_counts().values, 
                           names=[labels[i] for i in tmp.value_counts().index],
                           hole=0.4, title="",
                           color_discrete_sequence=['#10b981', '#ef4444'])
                fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # Cohort Heatmap (Enhanced)
    st.markdown("---")
    st.markdown('<div class="info-card"><h3>🔥 Cohort Retention Heatmap</h3></div>', unsafe_allow_html=True)
    if signup_col != "(none)" and last_col != "(none)":
        try:
            df2[signup_col] = pd.to_datetime(df2[signup_col], errors="coerce")
            df2[last_col] = pd.to_datetime(df2[last_col], errors="coerce")
            df2["cohort_month"] = df2[signup_col].dt.to_period("M").dt.to_timestamp()
            df2["activity_month"] = df2[last_col].dt.to_period("M").dt.to_timestamp()
            
            def month_diff(a,b): 
                return (b.year - a.year) * 12 + (b.month - a.month)
            df2["period_number"] = df2.apply(lambda r: month_diff(r["cohort_month"], r["activity_month"]) if pd.notnull(r["cohort_month"]) and pd.notnull(r["activity_month"]) else np.nan, axis=1)
            
            idcol = cust_id_col if cust_id_col != "(none)" else df2.columns[0]
            cohorts = df2.groupby(["cohort_month","period_number"]).agg(n_customers=(idcol,"count")).reset_index()
            cohort_sizes = df2.groupby("cohort_month").agg(cohort_size=(idcol,"count")).reset_index()
            cohorts = cohorts.merge(cohort_sizes, on="cohort_month")
            cohorts["retention"] = cohorts["n_customers"] / cohorts["cohort_size"]
            pivot = cohorts.pivot(index="cohort_month", columns="period_number", values="retention").fillna(0)
            
            # PowerBI-style heatmap
            heat_z = (pivot.values * 100).round(1)
            heat_x = [int(x) for x in pivot.columns.tolist()]
            heat_y = [str(d.date())[:7] for d in pivot.index.tolist()]
            
            fig = go.Figure(data=go.Heatmap(
                z=heat_z, x=heat_x, y=heat_y, 
                colorscale=[[0, '#fef3c7'], [0.5, '#fbbf24'], [1, '#0ea5a3']],
                text=heat_z, texttemplate="%{text}%%",
                hovertemplate="Cohort: %{y}<br>Months: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
                showscale=True
            ))
            fig.update_layout(
                title="📊 Cohort Analysis (Darker = Better Retention)",
                xaxis_title="Months Since Signup", 
                yaxis_title="Cohort Start Month",
                height=500,
                plot_bgcolor='rgba(248,250,252,1)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.download_button("📥 Download Cohort Data", 
                             data=pivot.reset_index().to_csv(index=False).encode("utf-8"), 
                             file_name="cohort_analysis.csv")
        except Exception as e:
            st.error(f"Cohort error: {e}")
    else:
        st.info("🔧 Select Signup + Last Active columns for cohort analysis")

    # Segmented Analysis (Enhanced)
    st.markdown("---")
    st.markdown('<div class="info-card"><h3>🎯 Retention by Segment</h3></div>', unsafe_allow_html=True)
    seg_col = None
    for s in ["Contract","InternetService","PaymentMethod"]:
        if s in df2.columns:
            seg_col = s; break
            
    if seg_col and tenure_col != "(none)" and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        segs = df2[seg_col].dropna().unique().tolist()[:5]
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        max_m = min(int(df2[tenure_col].dropna().max()), 24)
        
        colors = px.colors.qualitative.Set1[:len(segs)]
        for i, s in enumerate(segs):
            d = df2[df2[seg_col] == s]
            if len(d) > 0:
                retention = [(d[tenure_col] >= m).sum() / len(d) for m in range(0, max_m+1)]
                fig.add_trace(
                    go.Scatter(x=list(range(max_m+1)), y=retention,
                             mode='lines+markers', name=str(s),
                             line=dict(color=colors[i], width=3),
                             marker=dict(size=6)),
                    secondary_y=False
                )
        
        fig.update_layout(
            title=f"Retention Curves by {seg_col}",
            xaxis_title="Months", yaxis_title="Retention Rate",
            height=500, plot_bgcolor='rgba(248,250,252,1)',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        fig.update_yaxes(tickformat=".0%", secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("---")
    csv_all = df2.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Full Dataset", data=csv_all, file_name="churn_analysis_full.csv", type="secondary")

    st.markdown('<div class="small-muted">✨ PowerBI-style dashboard • Interactive charts • Export-ready</div>', unsafe_allow_html=True)

else:
    st.info("👈 Configure columns + filters → Click **Run Analysis** 🚀")
