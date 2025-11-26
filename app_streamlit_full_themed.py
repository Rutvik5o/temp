
import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, os

# Page config
st.set_page_config(page_title='Customer Churn Analyzer — Themed', layout='wide',
                   initial_sidebar_state='expanded')

# THEME / STYLES
GRADIENT_CSS = '''
<style>
/* Page background */
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg, #f7fbff 0%, #f0f7ff 35%, #ffffff 100%);
  padding-top: 0rem;
}

/* Header card */
.header-card {
  background: linear-gradient(90deg, #0f172a 0%, #0ea5a3 100%);
  color: white;
  padding: 18px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.12);
}

/* Info cards */
.info-card {
  background: white;
  border-radius: 10px;
  padding: 14px;
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
}

/* Sidebar style adjustments */
.css-1d391kg { padding-top: 0rem; }
section[data-testid="stSidebar"] .css-1lcbmhc {
  background: linear-gradient(180deg,#ffffff, #f8fafc);
  border-radius: 8px;
  padding: 12px;
}
/* Buttons */
.stButton>button {
  background: linear-gradient(90deg,#0ea5a3,#06b6d4);
  color: white;
  border: none;
  padding: 8px 14px;
  border-radius: 8px;
  font-weight: 600;
}
/* Smaller text */
.small-muted { color: #64748b; font-size:13px; }
</style>
'''

st.markdown(GRADIENT_CSS, unsafe_allow_html=True)

# Header
st.markdown('<div class="header-card"><h1 style="margin:0">Customer Churn Analyzer</h1><div class="small-muted">Upload a dataset, choose columns, and explore retention & churn with beautiful interactive charts.</div></div>', unsafe_allow_html=True)
st.write('')  # spacing

# Upload area + quick instructions in a top row
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader('Upload CSV (or leave empty to use sample)', type=['csv'])
    st.markdown('<div class="small-muted">Accepted file: CSV. If no file is uploaded, a sample dataset included with the project will be used (if available).</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="info-card"><strong>Tips</strong><br>1) Select proper columns in the sidebar.<br>2) Use filters to segment analysis.<br>3) Click Run analysis when ready.</div>', unsafe_allow_html=True)

# Load data (sample fallback)
DATA_PATH = r'/mnt/data/chrundata.csv'
if uploaded is None:
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            st.success('Loaded sample dataset.')
        except Exception as e:
            st.error('Sample dataset not found or unreadable. Please upload a CSV.')
            st.stop()
    else:
        st.error('No sample dataset in environment and no file uploaded. Please upload a CSV.')
        st.stop()
else:
    df = pd.read_csv(uploaded)

# Sidebar selectors
st.sidebar.header('Configure analysis')
cols = df.columns.tolist()
cust_id_col = st.sidebar.selectbox('Customer ID column', options=['(none)']+cols, index=cols.index('customerID')+1 if 'customerID' in cols else 0)
churn_col = st.sidebar.selectbox('Churn flag column', options=['(none)']+cols, index=cols.index('Churn')+1 if 'Churn' in cols else 0)
tenure_col = st.sidebar.selectbox('Tenure (months) column (optional)', options=['(none)']+cols, index=cols.index('tenure')+1 if 'tenure' in cols else 0)
signup_col = st.sidebar.selectbox('Signup / start date column (optional)', options=['(none)']+cols, index=0)
last_col = st.sidebar.selectbox('Last-active / end date column (optional)', options=['(none)']+cols, index=0)

st.sidebar.markdown('---')
st.sidebar.header('Segment filters (optional)')
possible_segments = [c for c in ['Contract','InternetService','PaymentMethod','Payment Method','PaymentMethod'] if c in cols]
segment_filters = {}
for s in possible_segments:
    vals = sorted(df[s].dropna().unique().tolist())
    sel = st.sidebar.multiselect(f'Filter {s}', options=vals, default=vals)
    segment_filters[s] = sel

st.sidebar.markdown('---')
run = st.sidebar.button('Run analysis', key='run')

# Utility: map churn values to 0/1
def normalize_churn(series):
    if series.dtype == object:
        return series.map(lambda x: 1 if str(x).strip().lower() in ['yes','y','true','1','t','churn','exited'] else (0 if str(x).strip().lower() in ['no','n','false','0','f','stay','retained'] else None))
    else:
        return series

if run:
    df2 = df.copy()
    # Apply segment filters
    for s, vals in segment_filters.items():
        if vals:
            df2 = df2[df2[s].isin(vals)]
    st.markdown('<div class="info-card"><strong>Filtered dataset</strong></div>', unsafe_allow_html=True)
    st.write(df2.head())

    # Normalize churn column
    churn = churn_col if churn_col != '(none)' else None
    if churn is not None:
        df2['_churn_mapped'] = normalize_churn(df2[churn])
        churn = '_churn_mapped'
        try:
            churn_rate = df2[churn].dropna().astype(float).mean()
        except Exception:
            churn_rate = None
    else:
        churn_rate = None

    # Layout: left column for metrics, right column for main charts
    m1, m2, m3 = st.columns(3)
    if churn_rate is not None:
        m1.metric('Overall churn rate', f'{churn_rate:.2%}')
    else:
        m1.metric('Overall churn rate', 'N/A')
    m2.metric('Total customers (filtered)', len(df2))
    # First-month retention (if tenure exists)
    if tenure_col != '(none)' and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        first_month = (df2[tenure_col] >= 1).sum() / len(df2)
        m3.metric('Retention @ month 1', f'{first_month:.2%}')
    else:
        m3.metric('Retention @ month 1', 'N/A')

    st.markdown('---')
    # Main visualizations area
    st.subheader('Visualizations')

    # 1) Churn distribution / Pie chart
    col_a, col_b = st.columns([1,2])
    with col_a:
        st.markdown('<div class="info-card"><strong>Churn distribution</strong></div>', unsafe_allow_html=True)
        if churn is not None:
            tmp = df2[churn].dropna().astype(int)
            fig = px.pie(values=tmp.value_counts().values, names=tmp.value_counts().index.astype(str), title='Churn (0 = retain, 1 = churn)', hole=0.35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write('Select a churn column in sidebar to view distribution.')

    # 2) Retention / Survival curve (tenure-based)
    with col_b:
        st.markdown('<div class="info-card"><strong>Retention curve (tenure-based)</strong></div>', unsafe_allow_html=True)
        if tenure_col != '(none)' and pd.api.types.is_numeric_dtype(df2[tenure_col]):
            total = len(df2)
            max_m = int(df2[tenure_col].dropna().astype(int).max())
            months = list(range(0, max_m+1))
            retention = [(m, (df2[tenure_col] >= m).sum()/total) for m in months]
            ret_df = pd.DataFrame(retention, columns=['month','retention_rate'])
            fig = px.line(ret_df, x='month', y='retention_rate', markers=True, title='Retention (tenure-based)')
            fig.update_yaxes(tickformat='.0%')
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.download_button('Download retention CSV', data=ret_df.to_csv(index=False).encode('utf-8'), file_name='retention_by_tenure.csv')
        else:
            st.write('Tenure column not selected or not numeric.')

    st.markdown('---')

    # 3) Cohort heatmap if dates provided
    st.subheader('Cohort Analysis (heatmap)')
    if signup_col != '(none)' and last_col != '(none)':
        try:
            df2[signup_col] = pd.to_datetime(df2[signup_col], errors='coerce')
            df2[last_col] = pd.to_datetime(df2[last_col], errors='coerce')
            df2['cohort_month'] = df2[signup_col].dt.to_period('M').dt.to_timestamp()
            df2['activity_month'] = df2[last_col].dt.to_period('M').dt.to_timestamp()
            def month_diff(a,b): return (b.year-a.year)*12 + (b.month-a.month)
            df2['period_number'] = df2.apply(lambda r: month_diff(r['cohort_month'], r['activity_month']) if pd.notnull(r['cohort_month']) and pd.notnull(r['activity_month']) else np.nan, axis=1)
            idcol = cust_id_col if cust_id_col != '(none)' else df2.columns[0]
            cohorts = df2.groupby(['cohort_month','period_number']).agg(n_customers=(idcol,'count')).reset_index()
            cohort_sizes = df2.groupby('cohort_month').agg(cohort_size=(idcol,'count')).reset_index()
            cohorts = cohorts.merge(cohort_sizes, on='cohort_month')
            cohorts['retention'] = cohorts['n_customers']/cohorts['cohort_size']
            pivot = cohorts.pivot(index='cohort_month', columns='period_number', values='retention').fillna(0)
            heat_z = (pivot.values*100).round(1)
            heat_x = [int(x) for x in pivot.columns.tolist()]
            heat_y = [str(d.date()) for d in pivot.index.tolist()]
            heatmap = go.Figure(data=go.Heatmap(z=heat_z, x=heat_x, y=heat_y, colorscale='YlGnBu', text=heat_z, hovertemplate='Cohort: %{y}<br>Months: %{x}<br>Retention: %{z:.1f}%'))
            heatmap.update_layout(title='Cohort retention heatmap (percent)', xaxis_title='Months since cohort', yaxis_title='Cohort month', height=500)
            st.plotly_chart(heatmap, use_container_width=True)
            st.download_button('Download cohort CSV', data=pivot.reset_index().to_csv(index=False).encode('utf-8'), file_name='cohort_pivot.csv')
        except Exception as e:
            st.error(f'Error computing cohort pivot: {e}')
    else:
        st.info('Select both Signup and Last-active columns in the sidebar to compute cohort heatmap.')

    st.markdown('---')
    # 4) Segmented retention comparison
    st.subheader('Segmented retention comparison')
    seg_col = None
    for s in ['Contract','InternetService','PaymentMethod']:
        if s in df2.columns:
            seg_col = s; break
    if seg_col is not None and tenure_col != '(none)' and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        segs = df2[seg_col].dropna().unique().tolist()[:6]
        fig = go.Figure()
        max_m = int(df2[tenure_col].dropna().astype(int).max())
        for s in segs:
            d = df2[df2[seg_col]==s]
            retention = [(m, (d[tenure_col]>=m).sum()/len(d)) for m in range(0, max_m+1)]
            rdf = pd.DataFrame(retention, columns=['month','retention_rate'])
            fig.add_trace(go.Scatter(x=rdf['month'], y=rdf['retention_rate'], mode='lines+markers', name=str(s)))
        fig.update_yaxes(tickformat='.0%')
        fig.update_layout(title=f'Retention by {seg_col}', xaxis_title='Months', yaxis_title='Retention')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Segment column not present or tenure missing — segmented retention unavailable.')

    st.markdown('---')
    st.subheader('Export & Notes')
    st.write('Use the download buttons near each visualization to export CSV results. The filtered dataset can be downloaded below.')
    csv_all = df2.to_csv(index=False).encode('utf-8')
    st.download_button('Download filtered dataset (CSV)', data=csv_all, file_name='filtered_dataset.csv')

    st.markdown('<div class="small-muted">App by: Customer Churn Analyzer — Themed UI | Improvements: gradient header, cards, interactive Plotly charts, export buttons.</div>', unsafe_allow_html=True)
else:
    st.info('Configure options in the sidebar and click "Run analysis" when ready.')
