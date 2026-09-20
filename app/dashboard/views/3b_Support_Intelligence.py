import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from configs.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from app.dashboard.components.utils import (
    render_topbar, render_page_header, render_empty_state,
    format_currency, section_header, CHART_CONFIG, get_chart_layout, ICONS
)

render_topbar()
render_page_header(
    "Support Intelligence",
    "NLP analysis of customer support tickets, sentiment trends, and their impact on churn risk."
)

@st.cache_data
def load_data():
    try:
        t = pd.read_csv(RAW_DATA_DIR / "nlp_tickets.csv")
        m = pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
        return t, m
    except: return None, None

t_df, m_df = load_data()
if t_df is None or m_df is None:
    render_empty_state("Run data pipeline first: python -m scripts.generate_data")
    st.stop()

m_risk = m_df[['customer_id', 'churn_probability_ground_truth', 'annual_revenue']].copy()
t_df = t_df.merge(m_risk, on='customer_id', how='left')
t_df['churn_probability_ground_truth'] = t_df['churn_probability_ground_truth'].fillna(0)
t_df['is_high_risk'] = t_df['churn_probability_ground_truth'] >= 0.5

tot_tickets = len(t_df)
pos = len(t_df[t_df['sentiment'] == 'Positive'])
neu = len(t_df[t_df['sentiment'] == 'Neutral'])
neg = len(t_df[t_df['sentiment'] == 'Negative'])
high_risk_tickets = len(t_df[t_df['is_high_risk']])

with st.container(border=True):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Tickets", f"{tot_tickets:,}")
    k2.metric("Negative Sentiment", f"{neg:,}", f"{neg/tot_tickets:.1%}", delta_color="inverse")
    k3.metric("Neutral Sentiment", f"{neu:,}", f"{neu/tot_tickets:.1%}", delta_color="off")
    k4.metric("Positive Sentiment", f"{pos:,}", f"{pos/tot_tickets:.1%}", delta_color="normal")
    k5.metric("High-Risk Accounts", f"{high_risk_tickets:,}", f"{high_risk_tickets/tot_tickets:.1%} of vol", delta_color="inverse")

st.html('<div class="iq-spacer-md"></div>')

ch1, ch2, ch3 = st.columns(3)
with ch1:
    with st.container(border=True):
        section_header("Sentiment Distribution")
        fig1 = go.Figure(go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[pos, neu, neg],
            hole=0.6,
            marker=dict(colors=['#22C55E', '#F59E0B', '#EF4444'], line=dict(color='#101219',width=2)),
            textinfo='none', hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>'
        ))
        fig1.update_layout(**get_chart_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10, color="#A7ACB8")),
            height=200, margin=dict(l=0,r=0,t=8,b=0)
        ))
        st.plotly_chart(fig1, use_container_width=True, config=CHART_CONFIG)

with ch2:
    with st.container(border=True):
        section_header("Issue Categories")
        cats = t_df['issue_category'].value_counts().reset_index()
        fig2 = px.bar(
            cats, x='count', y='issue_category', orientation='h',
            color_discrete_sequence=["#8B7CF6"],
            text=cats['count'].apply(lambda x: f"{x:,}")
        )
        fig2.update_traces(textposition='outside', hovertemplate='%{y}: %{x:,}<extra></extra>')
        fig2.update_layout(**get_chart_layout(
            yaxis=dict(showgrid=False, categoryorder='total ascending', tickfont=dict(color="#A7ACB8")),
            xaxis=dict(showgrid=False, showticklabels=False),
            height=200, margin=dict(l=0,r=30,t=8,b=0)
        ))
        st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)

with ch3:
    with st.container(border=True):
        section_header("Priority Distribution")
        pris = t_df['priority'].value_counts().reset_index()
        fig3 = px.bar(
            pris, x='priority', y='count',
            color='priority', color_discrete_map={"High":"#EF4444", "Medium":"#F59E0B", "Low":"#22C55E"},
            text=pris['count'].apply(lambda x: f"{x:,}")
        )
        fig3.update_traces(textposition='outside', hovertemplate='%{x}: %{y:,}<extra></extra>')
        fig3.update_layout(**get_chart_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, categoryorder='array', categoryarray=['High','Medium','Low']),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=200, margin=dict(l=0,r=0,t=8,b=0)
        ))
        st.plotly_chart(fig3, use_container_width=True, config=CHART_CONFIG)

st.html('<div class="iq-spacer-sm"></div>')

ch4, ch5 = st.columns(2)
with ch4:
    with st.container(border=True):
        section_header("Sentiment vs Churn Risk", "Average churn probability by sentiment")
        s_risk = t_df.groupby('sentiment').agg(avg_churn=('churn_probability_ground_truth','mean')).reset_index()
        fig4 = px.bar(
            s_risk, x='sentiment', y='avg_churn',
            color='sentiment', color_discrete_map={"Negative":"#ef4444", "Neutral":"#f59e0b", "Positive":"#10b981"},
            text=s_risk['avg_churn'].apply(lambda x: f"{x:.1%}")
        )
        fig4.update_traces(textposition='outside')
        fig4.update_layout(**get_chart_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, categoryorder='array', categoryarray=['Negative','Neutral','Positive']),
            yaxis=dict(showgrid=True, gridcolor='#1e2235', tickformat='.0%'),
            height=240, margin=dict(l=0,r=0,t=8,b=0)
        ))
        st.plotly_chart(fig4, use_container_width=True, config=CHART_CONFIG)

with ch5:
    with st.container(border=True):
        section_header("Issue Type vs Churn Risk", "Average churn probability by issue category")
        i_risk = t_df.groupby('issue_category').agg(avg_churn=('churn_probability_ground_truth','mean')).reset_index().sort_values('avg_churn', ascending=False)
        fig5 = px.bar(
            i_risk, x='issue_category', y='avg_churn',
            color='avg_churn', color_continuous_scale=[[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']],
            text=i_risk['avg_churn'].apply(lambda x: f"{x:.0%}")
        )
        fig5.update_traces(textposition='outside')
        fig5.update_layout(**get_chart_layout(
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#1e2235', tickformat='.0%'),
            height=240, margin=dict(l=0,r=0,t=8,b=0)
        ))
        st.plotly_chart(fig5, use_container_width=True, config=CHART_CONFIG)

st.html('<div class="iq-spacer-sm"></div>')

with st.container(border=True):
    section_header("Top Recurring Issues", "High-priority support topics requiring attention")
    iss_df = t_df.groupby('issue_category').agg(
        total=('ticket_id','count'),
        neg=('sentiment', lambda x: (x=='Negative').sum()),
        high_prio=('priority', lambda x: (x=='High').sum()),
        avg_risk=('churn_probability_ground_truth', 'mean')
    ).reset_index()
    iss_df['neg_pct'] = iss_df['neg'] / iss_df['total']
    iss_df = iss_df.sort_values('avg_risk', ascending=False)
    
    disp = iss_df.copy()
    disp['neg_pct'] = disp['neg_pct'].apply(lambda x: f"{x:.0%}")
    disp['avg_risk'] = disp['avg_risk'].apply(lambda x: f"{x:.0%}")
    
    disp.columns = ['Issue Category', 'Total Tickets', 'neg', 'High Priority', 'Avg Churn Risk', 'Negative %']
    
    # We drop the raw 'neg' column since 'Negative %' covers it, or reorder
    disp = disp[['Issue Category', 'Total Tickets', 'Negative %', 'High Priority', 'Avg Churn Risk']]
    
    st.dataframe(disp, hide_index=True, use_container_width=True)
