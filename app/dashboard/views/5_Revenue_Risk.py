import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from configs.config import PROCESSED_DATA_DIR
from app.dashboard.components.utils import (
    render_topbar, render_page_header, render_empty_state,
    format_currency, section_header, CHART_CONFIG, get_chart_layout
)

render_topbar()
render_page_header(
    "Revenue Risk",
    "Financial exposure analysis and ROI scenario simulator."
)

@st.cache_data
def load_data():
    try: return pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
    except: return None

df = load_data()
if df is None:
    render_empty_state("Run data pipeline first: python -m scripts.generate_data")
    st.stop()

# Compute exposure
df['expected_loss'] = df['annual_revenue'] * df['churn_probability_ground_truth']
df['risk_tier'] = pd.cut(
    df['churn_probability_ground_truth'],
    bins=[0, 0.4, 0.65, 1.01],
    labels=['Protected', 'Medium Risk', 'High Risk'], right=False
)
df['risk_tier'] = df['risk_tier'].astype(str)

tot_arr  = df['annual_revenue'].sum()
exp_loss = df['expected_loss'].sum()
high_df  = df[df['risk_tier'] == 'High Risk']
high_arr = high_df['annual_revenue'].sum()
med_arr  = df[df['risk_tier'] == 'Medium Risk']['annual_revenue'].sum()
prot_arr = df[df['risk_tier'] == 'Protected']['annual_revenue'].sum()

# ── KPIs ────────────────────────────────────────────────────
with st.container(border=True):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total ARR", format_currency(tot_arr))
    k2.metric("Expected Exposure", format_currency(exp_loss), f"{exp_loss/tot_arr:.1%} of Total", delta_color="inverse")
    k3.metric("High-Risk ARR", format_currency(high_arr), f"{len(high_df)} accounts", delta_color="inverse")
    k4.metric("Medium-Risk ARR", format_currency(med_arr), delta_color="off")
    k5.metric("Protected ARR", format_currency(prot_arr), delta_color="normal")

st.html('<div class="iq-spacer-md"></div>')

# ── Breakdowns ──────────────────────────────────────────────
ch1, ch2, ch3 = st.columns(3)

with ch1:
    with st.container(border=True):
        section_header("Exposure by Segment")
        s_rev = df.groupby('segment_name')['expected_loss'].sum().reset_index().sort_values('expected_loss', ascending=True)
        fig1 = px.bar(
            s_rev, x='expected_loss', y='segment_name', orientation='h',
            color_discrete_sequence=["#ef4444"],
            text=s_rev['expected_loss'].apply(lambda v: f"${v/1000:.0f}K")
        )
        fig1.update_traces(textposition='outside')
        fig1.update_layout(**get_chart_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            height=200, margin=dict(l=0,r=50,t=8,b=0)
        ))
        st.plotly_chart(fig1, use_container_width=True, config=CHART_CONFIG)

with ch2:
    with st.container(border=True):
        section_header("Exposure by Plan")
        p_rev = df.groupby('plan_type')['expected_loss'].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=p_rev['plan_type'], values=p_rev['expected_loss'], hole=0.6,
            marker=dict(colors=['#8B7CF6','#A78BFA','#F59E0B'], line=dict(color='#101219',width=2)),
            textinfo='none', hovertemplate='%{label}: $%{value:,.0f}<extra></extra>'
        ))
        fig2.update_layout(**get_chart_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10, color="#A7ACB8")),
            height=200, margin=dict(l=0,r=0,t=8,b=0)
        ))
        st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)

with ch3:
    with st.container(border=True):
        section_header("Exposure by Country")
        c_rev = df.groupby('country')['expected_loss'].sum().reset_index().sort_values('expected_loss', ascending=False).head(5)
        c_rev = c_rev.sort_values('expected_loss', ascending=True)
        fig3 = px.bar(
            c_rev, x='expected_loss', y='country', orientation='h',
            color_discrete_sequence=["#f59e0b"],
            text=c_rev['expected_loss'].apply(lambda v: f"${v/1000:.0f}K")
        )
        fig3.update_traces(textposition='outside')
        fig3.update_layout(**get_chart_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            height=200, margin=dict(l=0,r=50,t=8,b=0)
        ))
        st.plotly_chart(fig3, use_container_width=True, config=CHART_CONFIG)

st.html('<div class="iq-spacer-sm"></div>')

# ── Risk Tiers & Scenario Simulator ──────────────────────────
rt_df = df.groupby('risk_tier').agg(
    accounts=('customer_id', 'count'),
    total_arr=('annual_revenue', 'sum'),
    exp_loss=('expected_loss', 'sum')
).reset_index()

cr1, cr2 = st.columns([1.5, 1])
with cr1:
    with st.container(border=True):
        section_header("ARR vs Expected Loss per Risk Tier")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            name='Total ARR', x=rt_df['risk_tier'], y=rt_df['total_arr'],
            marker_color='#1e293b', text=rt_df['total_arr'].apply(lambda v: f"${v/1e6:.1f}M"), textposition='auto'
        ))
        fig4.add_trace(go.Bar(
            name='Expected Loss', x=rt_df['risk_tier'], y=rt_df['exp_loss'],
            marker_color='#ef4444', text=rt_df['exp_loss'].apply(lambda v: f"${v/1e6:.1f}M"), textposition='auto'
        ))
        fig4.update_layout(**get_chart_layout(
            barmode='group',
            yaxis=dict(showgrid=True, gridcolor='#1e2235', showticklabels=False),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            height=320, margin=dict(l=0,r=0,t=20,b=0)
        ))
        st.plotly_chart(fig4, use_container_width=True, config=CHART_CONFIG)

with cr2:
    with st.container(border=True):
        section_header("ROI Simulator", "Model intervention effectiveness")
        target_thresh = st.slider("Target Risk Threshold", 0.4, 0.9, 0.5, 0.05, key="revrisk_target_thresh")
        success_rate  = st.slider("Expected Success Rate", 0.0, 1.0, 0.2, 0.05, key="revrisk_success_rate")
        cost_per_acc  = st.number_input("Intervention Cost per Account ($)", value=250, step=50, key="revrisk_cost")
        
        target_df = df[df['churn_probability_ground_truth'] >= target_thresh]
        n_intervened = len(target_df)
        cost = n_intervened * cost_per_acc
        saved = target_df['expected_loss'].sum() * success_rate
        net_roi = saved - cost
        roi_pct = (net_roi / cost) if cost > 0 else 0
        
        st.divider()
        st.metric("Net ARR Saved (ROI)", format_currency(net_roi), f"{roi_pct:.1%} return", delta_color="normal" if net_roi > 0 else "inverse")
        s1, s2 = st.columns(2)
        s1.metric("Gross Saved", format_currency(saved))
        s2.metric("Program Cost", format_currency(cost))

