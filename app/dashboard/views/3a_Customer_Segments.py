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
    "Customer Segments",
    "KMeans-derived behavioural clusters revealing revenue concentration, churn risk, and engagement patterns."
)

@st.cache_data
def load_data():
    try: return pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
    except: return None

df = load_data()
if df is None:
    render_empty_state("Run data pipeline first: python -m scripts.generate_data")
    st.stop()

SEG_COLORS = {
    "High Value At Risk": "#EF4444",
    "High Value Loyal":   "#22C55E",
    "Support Heavy":      "#F59E0B",
    "New & Growing":      "#8B7CF6",
    "Low Engagement":     "#697080",
}
SEG_DESCS = {
    "High Value At Risk": "High-revenue accounts with declining engagement or growing support burden. Maximum financial exposure.",
    "High Value Loyal":   "Flagship accounts with healthy usage. Protect and expand — minimal intervention needed.",
    "Support Heavy":      "Disproportionate support load. Risk of frustration-driven churn without intervention.",
    "New & Growing":      "Recently acquired. Still forming product habits. High potential, moderate churn risk.",
    "Low Engagement":     "Dormant or disengaged customers. Passive churn risk — automated re-engagement recommended.",
}

seg = df.groupby('segment_name').agg(
    count=('customer_id','count'),
    total_arr=('annual_revenue','sum'),
    avg_arr=('annual_revenue','mean'),
    avg_churn=('churn_probability_ground_truth','mean'),
    churn_rate=('churned','mean'),
    avg_support=('support_tickets_30d','mean'),
    avg_csat=('csat_score','mean'),
    avg_tenure=('tenure_months','mean'),
    avg_login=('login_frequency','mean'),
    avg_adoption=('feature_adoption_rate','mean'),
).reset_index()
seg['rev_exposure'] = seg['total_arr'] * seg['avg_churn']

# ── KPI Row ──────────────────────────────────────────────────
with st.container(border=True):
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Segments", len(seg))
    k2.metric("Total Customers", f"{len(df):,}")
    highest_risk = seg.loc[seg['avg_churn'].idxmax()]
    k3.metric("Highest Risk Segment", highest_risk['segment_name'], f"{highest_risk['avg_churn']:.1%} avg", delta_color="inverse")
    highest_rev = seg.loc[seg['total_arr'].idxmax()]
    k4.metric("Highest Revenue Segment", highest_rev['segment_name'], format_currency(highest_rev['total_arr']))

st.html('<div class="iq-spacer-md"></div>')

# ── Segment Profile Cards ────────────────────────────────────
section_header("Segment Profiles", "Business-labelled customer clusters from KMeans segmentation")
cols = st.columns(len(seg))
for idx, (_, row) in enumerate(seg.iterrows()):
    seg_name = row['segment_name']
    color    = SEG_COLORS.get(seg_name, "#6b7280")
    desc     = SEG_DESCS.get(seg_name, "")
    churn_c  = "#ef4444" if row['avg_churn']>=0.6 else ("#f59e0b" if row['avg_churn']>=0.4 else "#10b981")
    with cols[idx]:
        with st.container(border=True):
            st.html(f"""
            <div style="border-bottom:1px solid #1e2235;padding-bottom:10px;margin-bottom:10px;">
                <div style="font-size:9px;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.1em;color:{color};margin-bottom:4px;">{seg_name}</div>
                <div style="font-size:22px;font-weight:700;color:#f1f5f9;line-height:1;">{row['count']:,}</div>
                <div style="font-size:10px;color:#4b5563;margin-top:3px;line-height:1.4;">{desc}</div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px;">
                <span style="color:#64748b;">Avg ARR</span>
                <span style="color:#f1f5f9;font-weight:600;">{format_currency(row['avg_arr'])}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px;">
                <span style="color:#64748b;">Churn Risk</span>
                <span style="color:{churn_c};font-weight:600;">{row['avg_churn']:.1%}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px;">
                <span style="color:#64748b;">Churn Rate</span>
                <span style="color:#94a3b8;font-weight:500;">{row['churn_rate']:.1%}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;">
                <span style="color:#64748b;">Avg Tickets/mo</span>
                <span style="color:#94a3b8;font-weight:500;">{row['avg_support']:.1f}</span>
            </div>
            """)

st.html('<div class="iq-spacer-sm"></div>')

# ── Charts ───────────────────────────────────────────────────
ch1, ch2 = st.columns(2)
with ch1:
    with st.container(border=True):
        section_header("Segment Size")
        fig = px.pie(
            seg, values='count', names='segment_name', hole=0.52,
            color='segment_name', color_discrete_map=SEG_COLORS
        )
        fig.update_traces(textinfo='none', hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>',
                          marker=dict(line=dict(color='#0f1117',width=2)))
        fig.update_layout(**get_chart_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
            height=240, margin=dict(l=0,r=0,t=8,b=0)
        ))
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

with ch2:
    with st.container(border=True):
        section_header("Total ARR by Segment")
        arr_seg = seg.sort_values('total_arr', ascending=True)
        fig2 = px.bar(
            arr_seg, x='total_arr', y='segment_name', orientation='h',
            color='segment_name', color_discrete_map=SEG_COLORS,
            text=arr_seg['total_arr'].apply(lambda v: f"${v/1000:.0f}K")
        )
        fig2.update_traces(textposition='outside', hovertemplate='%{y}: $%{x:,.0f}<extra></extra>',
                           marker_line_width=0)
        fig2.update_layout(**get_chart_layout(
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showgrid=False),
            height=240, margin=dict(l=0,r=60,t=8,b=0)
        ))
        st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)

ch3, ch4 = st.columns(2)
with ch3:
    with st.container(border=True):
        section_header("Avg Churn Risk by Segment")
        risk_seg = seg.sort_values('avg_churn', ascending=True)
        fig3 = px.bar(
            risk_seg, x='avg_churn', y='segment_name', orientation='h',
            color='avg_churn',
            color_continuous_scale=[[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']],
            text=risk_seg['avg_churn'].apply(lambda v: f"{v:.0%}")
        )
        fig3.update_traces(textposition='outside', hovertemplate='%{y}: %{x:.1%}<extra></extra>',
                           marker_line_width=0)
        fig3.update_layout(**get_chart_layout(
            coloraxis_showscale=False,
            xaxis=dict(tickformat='.0%', showgrid=False),
            yaxis=dict(showgrid=False),
            height=220, margin=dict(l=0,r=50,t=8,b=0)
        ))
        st.plotly_chart(fig3, use_container_width=True, config=CHART_CONFIG)

with ch4:
    with st.container(border=True):
        section_header("Revenue Exposure by Segment", "Expected loss = ARR × avg churn probability")
        exp_seg = seg.sort_values('rev_exposure', ascending=True)
        fig4 = px.bar(
            exp_seg, x='rev_exposure', y='segment_name', orientation='h',
            color_discrete_sequence=["#f59e0b"],
            text=exp_seg['rev_exposure'].apply(lambda v: f"${v/1000:.0f}K")
        )
        fig4.update_traces(textposition='outside', hovertemplate='%{y}: $%{x:,.0f}<extra></extra>',
                           marker_line_width=0)
        fig4.update_layout(**get_chart_layout(
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showgrid=False),
            height=220, margin=dict(l=0,r=60,t=8,b=0)
        ))
        st.plotly_chart(fig4, use_container_width=True, config=CHART_CONFIG)

# ── Scatter ──────────────────────────────────────────────────
with st.container(border=True):
    section_header("Usage Change vs Churn Risk", f"Sample of {min(1000,len(df)):,} customers — bubble size = ARR")
    sample = df.sample(min(1000, len(df)), random_state=42).copy()
    sample['size_rev'] = sample['annual_revenue'].clip(lower=100)
    fig_sc = px.scatter(
        sample, x='usage_change_30d', y='churn_probability_ground_truth',
        color='segment_name', size='size_rev', size_max=18,
        opacity=0.65, color_discrete_map=SEG_COLORS,
        hover_data=['customer_id', 'plan_type', 'annual_revenue']
    )
    fig_sc.add_hline(y=0.5, line_dash="dash", line_color="#ef4444")
    fig_sc.add_vline(x=0,   line_dash="dash", line_color="#374151")
    fig_sc.update_layout(**get_chart_layout(
        xaxis=dict(title="30-Day Usage Change (%)", gridcolor='#1e2235', showgrid=True),
        yaxis=dict(title="Churn Probability", tickformat='.0%', gridcolor='#1e2235', showgrid=True),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
        height=380, margin=dict(l=0,r=0,t=8,b=0)
    ))
    st.plotly_chart(fig_sc, use_container_width=True, config=CHART_CONFIG)

# ── Segment Explorer ─────────────────────────────────────────
st.html('<div class="iq-spacer-sm"></div>')
with st.container(border=True):
    sel_seg = st.selectbox("Explore Segment", options=sorted(df['segment_name'].unique()), key="segments_explore_seg")
    seg_cust = df[df['segment_name'] == sel_seg].copy()
    section_header(f"Customers in {sel_seg}",
                   SEG_DESCS.get(sel_seg,''),
                   count=len(seg_cust))
    disp = seg_cust[[
        'customer_id','country','plan_type','annual_revenue',
        'churn_probability_ground_truth','usage_change_30d',
        'support_tickets_30d','csat_score','feature_adoption_rate'
    ]].sort_values('churn_probability_ground_truth', ascending=False).copy()
    
    disp['annual_revenue'] = disp['annual_revenue'].apply(lambda x: f"${x:,.0f}")
    disp['churn_probability_ground_truth'] = disp['churn_probability_ground_truth'].apply(lambda x: f"{x:.0%}")
    disp['usage_change_30d'] = disp['usage_change_30d'].apply(lambda x: f"{x:.1f}%")
    disp['support_tickets_30d'] = disp['support_tickets_30d'].astype(int).astype(str)
    disp['csat_score'] = disp['csat_score'].apply(lambda x: f"{x:.1f}")
    disp['feature_adoption_rate'] = disp['feature_adoption_rate'].apply(lambda x: f"{x:.0%}")
    
    disp.columns = ['Customer', 'Country', 'Plan', 'ARR', 'Churn Risk', 'Usage Δ 30d', 'Tickets/mo', 'CSAT', 'Adoption']
    
    st.dataframe(disp, hide_index=True, use_container_width=True, height=350)
