import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, io

st.set_page_config(page_title='Customer Churn Analyzer — PowerBI Style', layout='wide',
                   initial_sidebar_state='expanded')

# -----------------------------------------------------------------------------
# Styles (PowerBI-inspired)
# -----------------------------------------------------------------------------
CSS = r'''
<style>
:root{--accent:#0ea5a3;--accent-2:#06b6d4;--card-bg:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);}
/* App background */
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg,#f8fafc 0%,#eef2ff 45%,#f1f5f9 100%);
  padding-top: 0.6rem;
  padding-bottom:1rem;
}
/* Header */
.header-card {
  background: linear-gradient(135deg,#0f172a 0%, #1f2937 40%, var(--accent) 100%);
  color: white;
  padding: 18px;
  border-radius: 14px;
  box-shadow: 0 12px 30px rgba(2,6,23,0.25);
}
.header-sub {
  color: rgba(255,255,255,0.9);
  opacity:0.95;
}
/* Metric cards */
.metric-card {
  background: var(--card-bg);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 18px rgba(15,23,42,0.06);
  border-left: 6px solid var(--accent);
}
.metric-value {font-size:1.45rem; font-weight:700;}
.metric-sub {color:#475569}
/* Info card */
.info-card {background:linear-gradient(180deg,#ffffff,#f8fafc); border-radius:10px; padding:12px; box-shadow:0 6px 18px rgba(15,23,42,0.05);}
.small-muted {color:#64748b; font-size:13px}
/* Sidebar */
section[data-testid="stSidebar"] .css-1lcbmhc{ background: linear-gradient(180deg,#ffffff, #f8fafc); border-radius:12px; padding:12px}
/* Buttons */
.stButton>button{
  background: linear-gradient(90deg,var(--accent),var(--accent-2));
  color:white; border:none; padding:8px 18px; border-radius:10px; font-weight:600; box-shadow:0 6px 18px rgba(14,165,163,0.18);
}
/* Dataframe header */
[data-testid="stDataFrameContainer"] th{ background:linear-gradient(135deg,#eef2ff,#e6f4f2)!important; color:#0f172a!important; font-weight:600}

/* Responsive tweaks */
@media (max-width:900px){ .header-card h1{font-size:1.4rem;} }
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('''
<div class="header-card">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
    <div>
      <h1 style="margin:0;font-size:1.8rem;">📊 Customer Churn Analyzer</h1>
      <div class="header-sub" style="font-size:0.95rem;margin-top:6px;">PowerBI-style — interactive, export-ready dashboards</div>
    </div>
    <div style="text-align:right; font-size:0.9rem; color:rgba(255,255,255,0.95);">
      <div>Made with ❤️ · Streamlit + Plotly</div>
    </div>
  </div>
</div>
''', unsafe_allow_html=True)

st.markdown("")

# -----------------------------------------------------------------------------
# Left upload / right quickstart
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([2,1])
with col_left:
    uploaded = st.file_uploader("📁 Upload CSV (or use sample)", type=["csv"], key="uploader_powerbi")
    st.markdown('<div class="small-muted">CSV format • sample path: /mnt/data/churndata.csv (if available)</div>', unsafe_allow_html=True)
with col_right:
    st.markdown('''
    <div class="info-card">
    <strong>🚀 Quick Start</strong><br>
    1️⃣ Upload CSV or use sample<br>
    2️⃣ Select ID, churn & tenure columns<br>
    3️⃣ Apply segments & click <strong>Run Analysis</strong>
    </div>
    ''', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
SAMPLE = "/mnt/data/churndata.csv"
if uploaded is None:
    if os.path.exists(SAMPLE):
        try:
            df = pd.read_csv(SAMPLE)
            st.success("✅ Loaded sample dataset")
        except Exception as e:
            st.error("❌ Sample dataset error. Please upload a CSV file.")
            st.stop()
    else:
        st.info("No sample dataset found. Please upload your CSV to get started.")
        df = pd.DataFrame()
else:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df):,} rows")
    except Exception as e:
        st.error(f"❌ Upload error: {e}")
        st.stop()

if df.empty:
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: configuration
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configure Analysis")
cols = df.columns.tolist()

# safe helper for default index
def safe_index(lst, value, offset=0):
    try:
        return lst.index(value) + offset
    except ValueError:
        return 0

cust_id_col = st.sidebar.selectbox("👤 Customer ID", ["(none)"] + cols, index=safe_index(cols, "customerID", 1), key="sid_cust_power")
churn_col = st.sidebar.selectbox("❌ Churn Flag", ["(none)"] + cols, index=safe_index(cols, "Churn", 1), key="sid_churn_power")
tenure_col = st.sidebar.selectbox("📅 Tenure (months)", ["(none)"] + cols, index=safe_index(cols, "tenure", 1), key="sid_tenure_power")
signup_col = st.sidebar.selectbox("📆 Signup Date", ["(none)"] + cols, index=safe_index(cols, "signup_date", 1), key="sid_signup_power")
last_col = st.sidebar.selectbox("📆 Last Active", ["(none)"] + cols, index=safe_index(cols, "last_active", 1), key="sid_last_power")

# Filters (auto-detect useful categorical columns)
candidate_segments = ["Contract","InternetService","PaymentMethod","Plan","subscription_type","Segment"]
possible_segments = [c for c in candidate_segments if c in cols]
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filters")
segment_filters = {}
for i, s in enumerate(possible_segments):
    vals = sorted(df[s].dropna().unique().tolist())
    if vals:
        # choose no more than 6 defaults to avoid accidental overfiltering
        default_vals = vals[:min(3, len(vals))]
        segment_filters[s] = st.sidebar.multiselect(f"🔧 {s}", options=vals, default=default_vals, key=f"seg_power_{i}")

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Analysis", key="run_power", type="primary")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def normalize_churn(series):
    """Map common churn values to 0/1. Returns numeric series with NaN for unmapped."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        s = series.astype(str).str.strip().str.lower()
        return s.map({
            'yes':1,'y':1,'true':1,'1':1,'t':1,'churn':1,'exited':1,'cancelled':1,'cancel':1,
            'no':0,'n':0,'false':0,'0':0,'f':0,'stay':0,'retained':0,'active':0
        }).astype(float)
    # numeric types
    return pd.to_numeric(series, errors='coerce')

# -----------------------------------------------------------------------------
# Run analysis
# -----------------------------------------------------------------------------
if run:
    df2 = df.copy()

    # apply segment filters
    for s, selvals in segment_filters.items():
        if selvals:
            df2 = df2[df2[s].isin(selvals)]

    st.markdown('<div class="info-card"><strong>📋 Filtered Dataset</strong> • {} rows</div>'.format(len(df2)), unsafe_allow_html=True)
    st.dataframe(df2.head(10), use_container_width=True, height=220)

    # churn mapping
    churn_key = None
    churn_rate = None
    if churn_col != "(none)":
        df2["_churn_mapped"] = normalize_churn(df2[churn_col])
        churn_key = "_churn_mapped"
        if df2[churn_key].dropna().shape[0] > 0:
            churn_rate = float(df2[churn_key].dropna().astype(float).mean())

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    total_customers = len(df2)
    avg_tenure = None
    month1_ret = None

    with c1:
        st.markdown('<div class="metric-card"><div class="metric-sub">👥 Total Customers</div><div class="metric-value">{:,}</div></div>'.format(total_customers), unsafe_allow_html=True)
    with c2:
        if churn_rate is not None:
            st.markdown('<div class="metric-card"><div class="metric-sub">🔴 Churn Rate</div><div class="metric-value">{:.1%}</div></div>'.format(churn_rate), unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="metric-sub">🔴 Churn Rate</div><div class="metric-value">N/A</div></div>', unsafe_allow_html=True)
    with c3:
        if tenure_col != "(none)" and pd.api.types.is_numeric_dtype(df2[tenure_col]):
            month1_ret = (df2[tenure_col] >= 1).sum() / max(len(df2),1)
            st.markdown('<div class="metric-card"><div class="metric-sub">🟢 Month 1 Retention</div><div class="metric-value">{:.1%}</div></div>'.format(month1_ret), unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="metric-sub">🟢 Month 1 Retention</div><div class="metric-value">N/A</div></div>', unsafe_allow_html=True)
    with c4:
        if tenure_col != "(none)" and pd.api.types.is_numeric_dtype(df2[tenure_col]):
            avg_tenure = df2[tenure_col].mean()
            st.markdown('<div class="metric-card"><div class="metric-sub">📊 Avg Tenure</div><div class="metric-value">{:.1f} mo</div></div>'.format(avg_tenure), unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="metric-sub">📊 Avg Tenure</div><div class="metric-value">N/A</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Row: Retention trend + churn distribution + new signups
    r1c1, r1c2, r1c3 = st.columns([2,1,1])

    # Retention Trend (line)
    with r1c1:
        st.markdown('<div class="info-card"><strong>📈 Retention Trend</strong></div>', unsafe_allow_html=True)
        if tenure_col != "(none)" and pd.api.types.is_numeric_dtype(df2[tenure_col]) and len(df2)>0:
            total = len(df2)
            max_m = int(min(df2[tenure_col].dropna().astype(int).max(), 36))
            months = list(range(0, max_m+1))
            retention = [(df2[tenure_col] >= m).sum() / max(total,1) for m in months]
            ret_df = pd.DataFrame({'month':months,'retention_rate':retention})

            fig = px.line(ret_df, x='month', y='retention_rate', markers=True, title='Customer Retention Over Time')
            fig.update_traces(line=dict(width=4, color='#0ea5a3'), marker=dict(size=7))
            fig.update_layout(plot_bgcolor='rgba(248,250,252,1)', height=420, yaxis=dict(tickformat='%.0%'))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("📥 Download Retention Data", data=ret_df.to_csv(index=False).encode('utf-8'), file_name='retention_trend.csv')
        else:
            st.info("Provide a numeric tenure column to compute retention trend.")

    # Churn distribution (pie) + New signups (bar)
    with r1c2:
        st.markdown('<div class="info-card"><strong>🍰 Churn Distribution</strong></div>', unsafe_allow_html=True)
        if churn_key is not None and df2[churn_key].dropna().shape[0]>0:
            tmp = df2[churn_key].dropna().astype(int)
            counts = tmp.value_counts().sort_index()
            labels = ['Retained 🟢','Churned 🔴']
            vals = [counts.get(0,0), counts.get(1,0)]
            fig = px.pie(values=vals, names=labels, hole=0.45)
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Churn column not mapped. Select churn flag in sidebar.")

    with r1c3:
        st.markdown('<div class="info-card"><strong>🆕 New Signups by Month</strong></div>', unsafe_allow_html=True)
        if signup_col != "(none)":
            try:
                tmp = pd.to_datetime(df2[signup_col], errors='coerce')
                monthly = tmp.dt.to_period('M').value_counts().sort_index()
                if len(monthly)>0:
                    mdf = monthly.reset_index()
                    mdf.columns = ['month','new_signups']
                    mdf['month'] = mdf['month'].astype(str)
                    fig = px.bar(mdf, x='month', y='new_signups', title='New Signups (by month)')
                    fig.update_layout(height=300, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info('No valid signup dates to plot.')
            except Exception as e:
                st.info('Error parsing signup dates.')
        else:
            st.info('Select Signup Date column to show signups by month.')

    st.markdown('---')

    # Cohort heatmap (improved)
    st.markdown('<div class="info-card"><h3>🔥 Cohort Retention Heatmap</h3></div>', unsafe_allow_html=True)
    if signup_col != "(none)" and last_col != "(none)":
        try:
            df2[signup_col] = pd.to_datetime(df2[signup_col], errors='coerce')
            df2[last_col] = pd.to_datetime(df2[last_col], errors='coerce')
            df2['cohort_month'] = df2[signup_col].dt.to_period('M').dt.to_timestamp()
            df2['activity_month'] = df2[last_col].dt.to_period('M').dt.to_timestamp()

            def month_diff(a,b):
                return (b.year - a.year)*12 + (b.month - a.month)
            df2['period_number'] = df2.apply(lambda r: month_diff(r['cohort_month'], r['activity_month']) if pd.notnull(r['cohort_month']) and pd.notnull(r['activity_month']) else np.nan, axis=1)

            idcol = cust_id_col if cust_id_col != '(none)' else df2.columns[0]
            cohorts = df2.groupby(['cohort_month','period_number']).agg(n_customers=(idcol,'count')).reset_index()
            cohort_sizes = df2.groupby('cohort_month').agg(cohort_size=(idcol,'count')).reset_index()
            cohorts = cohorts.merge(cohort_sizes, on='cohort_month')
            cohorts['retention'] = cohorts['n_customers']/cohorts['cohort_size']
            pivot = cohorts.pivot(index='cohort_month', columns='period_number', values='retention').fillna(0)

            heat_z = (pivot.values*100).round(1)
            heat_x = [int(x) for x in pivot.columns.tolist()]
            heat_y = [str(d.date())[:7] for d in pivot.index.tolist()]

            fig = go.Figure(data=go.Heatmap(z=heat_z, x=heat_x, y=heat_y, colorscale=[[0,'#fff7ed'],[0.25,'#fde68a'],[0.6,'#f97316'],[1,'#0ea5a3']], text=heat_z, texttemplate='%{text}%%', hovertemplate='Cohort: %{y}<br>Months: %{x}<br>Retention: %{z:.1f}%<extra></extra>'))
            fig.update_layout(title='📊 Cohort Analysis (Darker = Better Retention)', xaxis_title='Months Since Signup', yaxis_title='Cohort Start Month', height=520, plot_bgcolor='rgba(248,250,252,1)')
            st.plotly_chart(fig, use_container_width=True)
            st.download_button('📥 Download Cohort Data', data=pivot.reset_index().to_csv(index=False).encode('utf-8'), file_name='cohort_analysis.csv')
        except Exception as e:
            st.error(f'Cohort error: {e}')
    else:
        st.info('🔧 Select Signup + Last Active columns for cohort analysis')

    st.markdown('---')

    # Segmented retention curves
    st.markdown('<div class="info-card"><h3>🎯 Retention by Segment</h3></div>', unsafe_allow_html=True)
    seg_col = None
    for s in ['Contract','InternetService','PaymentMethod','Plan','subscription_type']:
        if s in df2.columns:
            seg_col = s; break

    if seg_col and tenure_col != '(none)' and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        segs = df2[seg_col].dropna().unique().tolist()[:6]
        fig = make_subplots(specs=[[{'secondary_y':False}]])
        max_m = int(min(df2[tenure_col].dropna().max(), 36))
        palette = px.colors.qualitative.Plotly
        for i, s in enumerate(segs):
            d = df2[df2[seg_col]==s]
            if len(d)>0:
                retention = [(d[tenure_col] >= m).sum()/len(d) for m in range(0, max_m+1)]
                fig.add_trace(go.Scatter(x=list(range(max_m+1)), y=retention, mode='lines+markers', name=str(s), line=dict(width=3, color=palette[i % len(palette)]), marker=dict(size=6)))
        fig.update_layout(title=f'Retention Curves by {seg_col}', xaxis_title='Months', yaxis_title='Retention Rate', height=520, plot_bgcolor='rgba(248,250,252,1)', legend=dict(orientation='h', yanchor='bottom', y=-0.2))
        fig.update_yaxes(tickformat='%.0%')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Select a segment column and numeric tenure to see segmented retention.')

    # Funnel summary (simple)
    st.markdown('---')
    st.markdown('<div class="info-card"><h3>🔎 Quick Funnel</h3></div>', unsafe_allow_html=True)
    if churn_key is not None and cust_id_col != '(none)':
        try:
            total = len(df2)
            active = df2[df2[churn_key]==0].shape[0]
            churned = df2[df2[churn_key]==1].shape[0]
            fig = go.Figure(go.Bar(x=['Total','Active','Churned'], y=[total, active, churned], text=[total,active,churned], textposition='auto'))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info('Funnel requires mapped churn column and customer id column (optional).')
    else:
        st.info('Map churn flag to enable funnel and churn summaries.')

    # Export full dataset
    st.markdown('---')
    csv_all = df2.to_csv(index=False).encode('utf-8')
    st.download_button('💾 Download Full Dataset', data=csv_all, file_name='churn_analysis_full.csv')

    st.markdown('<div class="small-muted">✨ PowerBI-style dashboard • Interactive charts • Export-ready</div>', unsafe_allow_html=True)

else:
    st.info('👈 Configure columns + filters → Click **Run Analysis** 🚀')
