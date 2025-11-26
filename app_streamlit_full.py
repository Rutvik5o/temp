import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, os
st.set_page_config(page_title='Customer Churn Analyzer — Full', layout='wide')
# Theme / CSS tweaks
st.markdown('''
<style>
.stApp { background-color: #f8fafc; }
.css-1d391kg {padding-top: 0rem;}
</style>
''', unsafe_allow_html=True)
st.title('Customer Churn Analyzer — Full UI')
st.write('Interactive dashboard: upload CSV, pick columns, filter segments, and explore multiple visualizations.')
# Upload
uploaded = st.file_uploader('Upload churn CSV', type=['csv'])
if uploaded is None:
    try:
        df = pd.read_csv(r'/mnt/data/chrundata.csv')
        st.info('Using sample dataset loaded from instructor-provided file.')
    except Exception as e:
        st.error('No sample dataset; please upload a CSV to proceed.')
        st.stop()
else:
    df = pd.read_csv(uploaded)
st.sidebar.header('Column selectors & options')
cols = df.columns.tolist()
cust_id_col = st.sidebar.selectbox('Customer ID column', options=['(none)']+cols, index=cols.index('customerID')+1 if 'customerID' in cols else 1)
churn_col = st.sidebar.selectbox('Churn flag column', options=['(none)']+cols, index=cols.index('Churn')+1 if 'Churn' in cols else 1)
tenure_col = st.sidebar.selectbox('Tenure (months) column (optional)', options=['(none)']+cols, index=cols.index('tenure')+1 if 'tenure' in cols else 1)
signup_col = st.sidebar.selectbox('Signup / start date column (optional)', options=['(none)']+cols, index=1)
last_col = st.sidebar.selectbox('Last-active / end date column (optional)', options=['(none)']+cols, index=1)
st.sidebar.markdown('---')
st.sidebar.header('Segment filters (optional)')
# detect common segment columns
possible_segments = [c for c in ['Contract','InternetService','PaymentMethod','Payment Method','ContractType','Plan'] if c in cols]
seg_choices = {}
for s in possible_segments:
    seg_choices[s] = st.sidebar.multiselect(f'Filter {s}', options=sorted(df[s].dropna().unique()), default=sorted(df[s].dropna().unique()))
st.sidebar.markdown('---')
run_btn = st.sidebar.button('Run analysis')
if run_btn:
    df2 = df.copy()
    # apply filters
    for s,vals in seg_choices.items():
        if vals:
            df2 = df2[df2[s].isin(vals)]
    st.subheader('Dataset overview')
    st.write('Rows after filtering: ', len(df2))
    st.write(df2.head())
    # churn mapping
    churn = churn_col if churn_col!='(none)' else None
    if churn is not None:
        # map Yes/No -> 1/0 if necessary
        if df2[churn].dtype == object:
            df2['_churn_mapped'] = df2[churn].apply(lambda x: 1 if str(x).strip().lower() in ['yes','y','true','1','t'] else (0 if str(x).strip().lower() in ['no','n','false','0','f'] else None))
            churn = '_churn_mapped'
        try:
            churn_rate = df2[churn].dropna().astype(float).mean()
            st.metric('Overall churn rate', f'{churn_rate:.2%}')
        except Exception:
            st.write('Churn column present but could not compute numeric mean.')
    # Visualization 1: churn distribution by segment (if churn exists)
    st.markdown('## Churn distribution')
    if churn is not None:
        fig = px.histogram(df2, x=churn, nbins=3, title='Churn flag distribution', labels={churn:'Churn'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write('No churn column selected — showing tenure distribution if available.')
    # Visualization 2: retention curve (tenure-based)
    st.markdown('## Retention / Survival curve')
    if tenure_col!='(none)' and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        total = len(df2)
        max_m = int(df2[tenure_col].dropna().astype(int).max())
        months = list(range(0, max_m+1))
        retention = [(m, (df2[tenure_col] >= m).sum()/total) for m in months]
        ret_df = pd.DataFrame(retention, columns=['month','retention_rate'])
        fig = px.line(ret_df, x='month', y='retention_rate', markers=True, title='Retention (tenure-based)')
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        st.download_button('Download retention CSV', data=ret_df.to_csv(index=False).encode('utf-8'), file_name='retention_by_tenure.csv')
    else:
        st.write('Tenure column not selected or not numeric.')
    # Visualization 3: Cohort heatmap if signup+last provided
    st.markdown('## Cohort retention heatmap')
    if signup_col!='(none)' and last_col!='(none)':
        try:
            df2[signup_col] = pd.to_datetime(df2[signup_col], errors='coerce')
            df2[last_col] = pd.to_datetime(df2[last_col], errors='coerce')
            df2['cohort_month'] = df2[signup_col].dt.to_period('M').dt.to_timestamp()
            df2['activity_month'] = df2[last_col].dt.to_period('M').dt.to_timestamp()
            def month_diff(a,b): return (b.year-a.year)*12 + (b.month-a.month)
            df2['period_number'] = df2.apply(lambda r: month_diff(r['cohort_month'], r['activity_month']) if pd.notnull(r['cohort_month']) and pd.notnull(r['activity_month']) else np.nan, axis=1)
            idcol = cust_id_col if cust_id_col!='(none)' else df2.columns[0]
            cohorts = df2.groupby(['cohort_month','period_number']).agg(n_customers=(idcol,'count')).reset_index()
            cohort_sizes = df2.groupby('cohort_month').agg(cohort_size=(idcol,'count')).reset_index()
            cohorts = cohorts.merge(cohort_sizes, on='cohort_month')
            cohorts['retention'] = cohorts['n_customers']/cohorts['cohort_size']
            pivot = cohorts.pivot(index='cohort_month', columns='period_number', values='retention').fillna(0)
            # create heatmap with plotly
            heat_z = pivot.values*100  # percent
            heat_x = [int(x) for x in pivot.columns.tolist()]
            heat_y = [str(d.date()) for d in pivot.index.tolist()]
            heatmap = go.Figure(data=go.Heatmap(z=heat_z, x=heat_x, y=heat_y, colorscale='YlGnBu', text=heat_z, hovertemplate='Cohort: %{y}<br>Months: %{x}<br>Retention: %{z:.1f}%'))
            heatmap.update_layout(title='Cohort retention heatmap (percent)', xaxis_title='Months since cohort', yaxis_title='Cohort month')
            st.plotly_chart(heatmap, use_container_width=True)
            st.download_button('Download cohort CSV', data=pivot.reset_index().to_csv(index=False).encode('utf-8'), file_name='cohort_pivot.csv')
        except Exception as e:
            st.write('Error computing cohort pivot:', e)
    else:
        st.write('Signup and Last-active columns not both selected; cohort heatmap unavailable.')
    # Visualization 4: Segmented retention comparison (by Contract or InternetService)
    st.markdown('## Segmented retention comparison')
    seg_col = None
    for s in ['Contract','InternetService','PaymentMethod']:
        if s in df2.columns:
            seg_col = s; break
    if seg_col is not None and tenure_col!='(none)' and pd.api.types.is_numeric_dtype(df2[tenure_col]):
        segs = df2[seg_col].dropna().unique().tolist()[:6]  # limit to first 6 segments
        fig = go.Figure()
        total = len(df2)
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
        st.write('Segment column not available or tenure missing for segmented retention.')
    # Export whole filtered dataset
    st.markdown('---')
    csv_all = df2.to_csv(index=False).encode('utf-8')
    st.download_button('Download filtered dataset (CSV)', data=csv_all, file_name='filtered_dataset.csv')
    # Small conclusions area
    st.markdown('---')
    st.subheader('Auto-generated insights')
    insights = []
    try:
        if churn is not None:
            insights.append(f'Overall churn rate: {churn_rate:.2%}')
        if tenure_col!='(none)' and pd.api.types.is_numeric_dtype(df2[tenure_col]):
            rm = ret_df.loc[ret_df.month==0,'retention_rate'].values[0]
            r3 = ret_df.loc[ret_df.month==3,'retention_rate'].values[0] if 3 in ret_df.month.values else None
            insights.append(f'First-month retention: {rm:.2%}')
            if r3 is not None:
                insights.append(f'Retention at month 3: {r3:.2%}')
    except Exception:
        pass
    for it in insights:
        st.write('- ', it)
else:
    st.write('Select options in the sidebar and click Run analysis.')