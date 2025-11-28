import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ------------------------- PAGE CONFIG (FIXED, NO INDENTATION ERRORS) -------------------------
st.set_page_config(
    page_title='Customer Churn Analyzer — PowerBI Dark',
    layout='wide',
    initial_sidebar_state='expanded'
)


# ------------------------- COLOR PALETTE ---------------------------------
# Professional, colorful palette for all bar/hist/box charts
PALETTE = px.colors.qualitative.Plotly  # easily readable set of colors
ACCENT = '#0ea5a3'

                   initial_sidebar_state='expanded')

# ------------------------- DARK THEME CSS ---------------------------------
CSS = r'''
<style>
:root{--bg:#0b1220;--card:#0f1724;--muted:#9aa4b2;--accent:#0ea5a3;--accent-2:#06b6d4;--glass:rgba(255,255,255,0.03);}
/* App background */
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg,#071019 0%, #071724 40%, #0a1522 100%);
  color: #e6eef6;
  padding-top: 0.6rem;
  padding-bottom: 1rem;
}
/* Header */
.header-card {
  background: linear-gradient(90deg,#071126 0%, #0d2330 40%, var(--accent) 100%);
  color: white;
  padding: 18px;
  border-radius: 10px;
  box-shadow: 0 12px 40px rgba(2,6,23,0.6);
  border: 1px solid rgba(255,255,255,0.03);
}
.header-sub {color: rgba(255,255,255,0.9); opacity:0.95}
/* Sidebar */
section[data-testid="stSidebar"] .css-1lcbmhc{ background: linear-gradient(180deg,#07101a,#0b1420); border-radius:10px; padding:12px; color: #dfe9f2 }
/* Metric cards */
.metric-card {background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:14px; box-shadow: 0 6px 20px rgba(2,8,23,0.6); border-left:6px solid var(--accent);}
.metric-sub {color:var(--muted); font-size:13px}
.metric-value {font-size:1.45rem; font-weight:700; color: #e6f7f2}
.info-card {background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:10px; padding:12px;}
.small-muted {color:var(--muted); font-size:13px}
/* Controls */
.stButton>button{ background: linear-gradient(90deg,var(--accent),var(--accent-2)); color:white; border:none; padding:8px 18px; border-radius:10px; font-weight:600; box-shadow: 0 6px 20px rgba(14,165,163,0.18);}
/* Dataframe header */
[data-testid="stDataFrameContainer"] th{ background:linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important; color:#dff6f0 !important; font-weight:600}

/* Small tweaks */
.chart-card { padding:10px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005)); border-radius:10px; box-shadow: 0 10px 30px rgba(2,6,23,0.6); border: 1px solid rgba(255,255,255,0.02);}

</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------- HEADER -----------------------------------------
st.markdown('''
<div class="header-card">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
    <div>
      <h1 style="margin:0;font-size:1.6rem;">📊 Customer Churn Analyzer — Dark PowerBI</h1>
      <div class="header-sub" style="font-size:0.95rem;margin-top:6px;">Interactive • Clean visuals • Export-ready</div>
    </div>
    <div style="text-align:right; font-size:0.85rem; color:rgba(255,255,255,0.85);">
      <div>Built with Streamlit + Plotly</div>
    </div>
  </div>
</div>
''', unsafe_allow_html=True)

# ------------------------- UPLOAD AREA ------------------------------------
col_left, col_right = st.columns([2,1])
with col_left:
    uploaded = st.file_uploader("📁 Upload CSV (or use sample)", type=["csv"], key="uploader_powerbi")
    st.markdown('<div class="small-muted">CSV format • sample path: /mnt/data/churndata.csv (if available)</div>', unsafe_allow_html=True)
with col_right:
    st.markdown('<div class="info-card"><strong>⚡ Quick Tips</strong><br>• Use "Churn" column with Yes/No<br>• Tenure (months) numeric<br>• MonthlyCharges numeric</div>', unsafe_allow_html=True)

# ------------------------- LOAD DATA -------------------------------------
SAMPLE = "/mnt/data/churndata.csv"
if uploaded is None:
    if os.path.exists(SAMPLE):
        try:
            df = pd.read_csv(SAMPLE)
            st.success("✅ Loaded sample dataset")
        except Exception:
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

# ------------------------- SIDEBAR CONFIG --------------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configure Analysis (Dark)")
cols = df.columns.tolist()

def safe_index(lst, value, offset=0):
    try:
        return lst.index(value) + offset
    except ValueError:
        return 0

cust_id_col = st.sidebar.selectbox("👤 Customer ID", ["(none)"] + cols, index=safe_index(cols, "customerID", 1))
churn_col = st.sidebar.selectbox("❌ Churn Flag", ["(none)"] + cols, index=safe_index(cols, "Churn", 1))
tenure_col = st.sidebar.selectbox("📅 Tenure (months)", ["(none)"] + cols, index=safe_index(cols, "tenure", 1))
signup_col = st.sidebar.selectbox("📆 Signup Date (optional)", ["(none)"] + cols, index=safe_index(cols, "signup_date", 1))
last_col = st.sidebar.selectbox("📆 Last Active (optional)", ["(none)"] + cols, index=safe_index(cols, "last_active", 1))

# auto detect categorical filters
candidate_segments = ["Contract","InternetService","PaymentMethod","PaperlessBilling","Gender","gender","SeniorCitizen"]
possible_segments = [c for c in candidate_segments if c in cols]
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filters")
segment_filters = {}
for i, s in enumerate(possible_segments):
    vals = sorted(df[s].dropna().unique().tolist())
    if vals:
        default_vals = vals[:min(3, len(vals))]
        segment_filters[s] = st.sidebar.multiselect(f"🔧 {s}", options=vals, default=default_vals, key=f"seg_power_{i}")

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Analysis")

# ------------------------- HELPERS ---------------------------------------
def normalize_churn(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    s = series.astype(str).str.strip().str.lower()
    return s.map({'yes':1,'y':1,'true':1,'1':1,'t':1,'churn':1,'no':0,'n':0,'false':0,'0':0,'stay':0,'retained':0}).astype(float)

# plotly dark layout template helper
def dark_plotly_layout(fig, height=420, showlegend=True):
    fig.update_layout(
        template=None,
        paper_bgcolor='rgba(11,18,32,0)',
        plot_bgcolor='rgba(11,18,32,0)',
        font=dict(color='#e6eef6'),
        height=height,
        legend=dict(bgcolor='rgba(255,255,255,0.02)') if showlegend else dict(visible=False)
    )
    # axis style
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor='rgba(255,255,255,0.06)', tickfont=dict(color='#d0e8e2'))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, showline=True, linecolor='rgba(255,255,255,0.06)', tickfont=dict(color='#d0e8e2'))
    return fig

# ------------------------- RUN ANALYSIS ----------------------------------
if run:
    df2 = df.copy()
    # apply filters
    for s, selvals in segment_filters.items():
        if selvals:
            df2 = df2[df2[s].isin(selvals)]

    st.markdown('<div class="info-card"><strong>📋 Filtered Dataset</strong> • {} rows</div>'.format(len(df2)), unsafe_allow_html=True)
    st.dataframe(df2.head(8), use_container_width=True, height=200)

    # normalize churn
    churn_key = None
    churn_rate = None
    if churn_col != "(none)":
        df2['_churn_mapped'] = normalize_churn(df2[churn_col])
        churn_key = '_churn_mapped'
        if df2[churn_key].dropna().shape[0] > 0:
            churn_rate = float(df2[churn_key].dropna().mean())

    # KPIs
    total_customers = len(df2)
    avg_tenure = None
    month1_ret = None
    if tenure_col != "(none)":
        df2[tenure_col] = pd.to_numeric(df2[tenure_col], errors='coerce')
        avg_tenure = df2[tenure_col].dropna().mean()
        month1_ret = (df2[tenure_col] >= 1).sum() / max(total_customers, 1)
    df2['MonthlyCharges'] = pd.to_numeric(df2['MonthlyCharges'], errors='coerce') if 'MonthlyCharges' in df2.columns else pd.Series(np.nan)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        html_k1 = '<div class="metric-card"><div class="metric-sub">👥 Total Customers</div><div class="metric-value">{:,}</div></div>'.format(total_customers)
        st.markdown(html_k1, unsafe_allow_html=True)
    with k2:
        if churn_rate is not None:
            html_k2 = '<div class="metric-card"><div class="metric-sub">🔴 Churn Rate</div><div class="metric-value">{:.1%}</div></div>'.format(churn_rate)
            st.markdown(html_k2, unsafe_allow_html=True)
        else:
            html_k2 = '<div class="metric-card"><div class="metric-sub">🔴 Churn Rate</div><div class="metric-value">N/A</div></div>'
            st.markdown(html_k2, unsafe_allow_html=True)
    with k3:
        avg_tenure_fmt = "N/A" if avg_tenure is None else "{:.1f}".format(avg_tenure)
        html_k3 = '<div class="metric-card"><div class="metric-sub">📅 Avg Tenure</div><div class="metric-value">{}</div></div>'.format(avg_tenure_fmt)
        st.markdown(html_k3, unsafe_allow_html=True)
    with k4:
        avg_monthly = df2['MonthlyCharges'].mean() if 'MonthlyCharges' in df2.columns else None
        avg_monthly_fmt = "N/A" if avg_monthly is None or np.isnan(avg_monthly) else "{:.2f}".format(avg_monthly)
        html_k4 = '<div class="metric-card"><div class="metric-sub">💳 Avg Monthly Charges</div><div class="metric-value">{}</div></div>'.format(avg_monthly_fmt)
        st.markdown(html_k4, unsafe_allow_html=True)

    st.markdown('---', unsafe_allow_html=True)

    # Charts layout: 2-column main
    left_col, right_col = st.columns([2,1])

    # LEFT: Retention + Monthly charges box + Contract churn
    with left_col:
        st.markdown('<div class="chart-card"><strong>📈 Retention (Tenure Survival)</strong></div>', unsafe_allow_html=True)
        if tenure_col != "(none)" and df2[tenure_col].dropna().shape[0] > 0:
            total = len(df2)
            max_m = int(min(df2[tenure_col].dropna().astype(int).max(), 48))
            months = list(range(0, max_m+1))
            retention = [(df2[tenure_col] >= m).sum() / max(total, 1) for m in months]
            ret_df = pd.DataFrame({'month': months, 'retention_rate': retention})
            fig = px.line(ret_df, x='month', y='retention_rate', markers=True, color_discrete_sequence=[ACCENT], color_discrete_sequence=[ACCENT])
            fig.update_traces(line=dict(width=4, color='#0ea5a3'), marker=dict(size=6))
            fig.update_yaxes(tickformat='%')
            fig = dark_plotly_layout(fig, height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Provide numeric tenure column to show retention curve.')

        st.markdown('<div class="chart-card" style="margin-top:12px;"><strong>📦 Monthly Charges by Churn (Box)</strong></div>', unsafe_allow_html=True)
        if churn_key and 'MonthlyCharges' in df2.columns:
            tmp = df2[["MonthlyCharges", churn_key]].dropna()
            tmp[churn_key] = tmp[churn_key].astype(int).map({0:'No',1:'Yes'})
            fig = px.box(tmp, x=churn_key, y='MonthlyCharges', points='all', color_discrete_sequence=PALETTE)
            fig = dark_plotly_layout(fig, height=340)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('MonthlyCharges or Churn mapping missing to plot boxplot.')

        st.markdown('<div class="chart-card" style="margin-top:12px;"><strong>📊 Churn Rate by Contract Type</strong></div>', unsafe_allow_html=True)
        if 'Contract' in df2.columns and churn_key:
            tmp = df2[["Contract", churn_key]].dropna()
            agg = tmp.groupby('Contract').agg(total=('Contract','count'), churned=(churn_key,'sum')).reset_index()
            agg['churn_rate'] = agg['churned'] / agg['total']
            fig = px.bar(agg, x='Contract', y='churn_rate', text=agg['churn_rate'].apply(lambda x: f"{x:.0%}", color_discrete_sequence=PALETTE))
            fig.update_traces(marker_line_width=0)
            fig = dark_plotly_layout(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Contract or Churn column missing.')

    # RIGHT: Distribution pies and feature importance style bars
    with right_col:
        st.markdown('<div class="chart-card"><strong>🍰 Churn Distribution</strong></div>', unsafe_allow_html=True)
        if churn_key:
            counts = df2[churn_key].dropna().astype(int).value_counts().sort_index()
            vals = [counts.get(0,0), counts.get(1,0)]
            labels = ['Retained','Churned']
            fig = px.pie(values=vals, names=labels, hole=0.45, color_discrete_sequence=PALETTE)
            fig.update_traces(textinfo='percent+label', textfont_size=13)
            fig = dark_plotly_layout(fig, height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Map the Churn column to view distribution.')

        st.markdown('<div class="chart-card" style="margin-top:12px;"><strong>🔍 Top Drivers (Service Flags)</strong></div>', unsafe_allow_html=True)
        # Compute churn rates for service flags
        service_cols = [c for c in ['TechSupport','OnlineSecurity','DeviceProtection','OnlineBackup','StreamingTV','StreamingMovies','PhoneService'] if c in df2.columns]
        if service_cols and churn_key:
            rows = []
            for c in service_cols:
                tmp = df2[[c, churn_key]].dropna()
                # map yes/no
                mapped = tmp[c].astype(str).str.strip().str.lower().replace({'yes':1,'no':0,'no phone service':0,'nan':np.nan})
                if mapped.dropna().shape[0] == 0:
                    continue
                grp = pd.DataFrame({'flag': mapped, 'churn': tmp[churn_key]})
                agg = grp.groupby('flag').agg(total=('churn','count'), churned=('churn','sum')).reset_index()
                try:
                    churn_rate_flag = agg.loc[agg['flag']==1, 'churned'].values[0] / agg.loc[agg['flag']==1, 'total'].values[0]
                except Exception:
                    churn_rate_flag = 0
                rows.append({'feature': c, 'churn_rate_if_yes': churn_rate_flag})
            feat_df = pd.DataFrame(rows).sort_values('churn_rate_if_yes', ascending=False)
            if not feat_df.empty:
                fig = px.bar(feat_df, x='churn_rate_if_yes', y='feature', orientation='h', text=feat_df['churn_rate_if_yes'].apply(lambda x: f"{x:.0%}", color_discrete_sequence=PALETTE))
                fig = dark_plotly_layout(fig, height=360, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No usable service flags found.')
        else:
            st.info('Service flag columns or churn mapping missing.')

    st.markdown('---', unsafe_allow_html=True)

    # MORE: Payment method and Internet service
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card"><strong>💳 Churn by Payment Method</strong></div>', unsafe_allow_html=True)
        if 'PaymentMethod' in df2.columns and churn_key:
            tmp = df2[['PaymentMethod', churn_key]].dropna()
            agg = tmp.groupby('PaymentMethod').agg(total=('PaymentMethod','count'), churned=(churn_key,'sum')).reset_index()
            agg['churn_rate'] = agg['churned'] / agg['total']
            fig = px.bar(agg, x='PaymentMethod', y='churn_rate', text=agg['churn_rate'].apply(lambda x: f"{x:.0%}", color_discrete_sequence=PALETTE))
            fig = dark_plotly_layout(fig, height=350)
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('PaymentMethod or Churn missing.')

    with c2:
        st.markdown('<div class="chart-card"><strong>🌐 Churn by Internet Service</strong></div>', unsafe_allow_html=True)
        if 'InternetService' in df2.columns and churn_key:
            tmp = df2[['InternetService', churn_key]].dropna()
            agg = tmp.groupby('InternetService').agg(total=('InternetService','count'), churned=(churn_key,'sum')).reset_index()
            agg['churn_rate'] = agg['churned'] / agg['total']
            fig = px.bar(agg, x='InternetService', y='churn_rate', text=agg['churn_rate'].apply(lambda x: f"{x:.0%}", color_discrete_sequence=PALETTE))
            fig = dark_plotly_layout(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('InternetService or Churn missing.')

    st.markdown('---', unsafe_allow_html=True)

    # Correlation heatmap (robust + colorful)
    st.markdown('<div class="chart-card"><strong>🧭 Numeric Correlation</strong></div>', unsafe_allow_html=True)
    num_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges'] if c in df2.columns]
    if num_cols and churn_key:
        # Coerce numeric, include churn as numeric
        corr_df = df2[num_cols].copy()
        for c in corr_df.columns:
            corr_df[c] = pd.to_numeric(corr_df[c], errors='coerce')
        # ensure churn numeric
        corr_df['_churn_for_corr'] = pd.to_numeric(df2[churn_key], errors='coerce')
        # drop rows with all-NaN
        corr_df = corr_df.dropna(how='all')
        if corr_df.shape[0] >= 2:
            corr = corr_df.corr()
            z = corr.values
            # Use a colorful diverging palette
            fig = go.Figure(data=go.Heatmap(z=z, x=corr.columns, y=corr.index, colorscale='RdYlBu', reversescale=True))
            fig.update_traces(colorbar=dict(title="corr"))
            fig = dark_plotly_layout(fig, height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Not enough numeric rows after coercion to compute correlation.')
    else:
        st.info('Not enough numeric columns for correlation.')

    # Export
    st.markdown('---', unsafe_allow_html=True)
    csv_all = df2.to_csv(index=False).encode('utf-8')
    st.download_button('💾 Download Filtered Dataset', data=csv_all, file_name='churn_analysis_filtered.csv')
    st.markdown('<div class="small-muted">✨ Dark PowerBI-style • Clean • Focused</div>', unsafe_allow_html=True)

else:
    st.info('👈 Configure columns + filters → Click **Run Analysis** 🚀')
