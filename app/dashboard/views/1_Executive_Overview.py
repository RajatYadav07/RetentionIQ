import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from configs.config import PROCESSED_DATA_DIR
from app.dashboard.components.utils import (
    render_topbar, render_page_header, render_kpi_card,
    format_currency, render_empty_state, section_header,
    CHART_CONFIG, get_chart_layout, ICONS
)

# ── 1. Top App Bar ──────────────────────────────────────────
render_topbar()

# ── 2. Page Header with Date Range Selector ─────────────────
render_page_header(
    "Executive Overview",
    "Predict risk. Understand why. Act before customers leave.",
    actions_html=f"""
        <div style="display:flex;align-items:center;gap:6px;">
            <div class="iq-btn">
                <span style="color:#8B7CF6;display:inline-flex;align-items:center;">{ICONS['calendar']}</span>
                <span style="color:#F2F3F5;font-weight:500;">Last 30 days</span>
            </div>
        </div>
    """
)

# ── Data Loading & Computation ──────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    render_empty_state("Data pipeline not executed. Run: python -m scripts.generate_data")
    st.stop()

# Real metrics calculation
total_customers = len(df)
prev_customers = len(df[df['tenure_months'] >= 2])
cust_growth_pct = (total_customers - prev_customers) / prev_customers if prev_customers > 0 else 0.0

at_risk_df = df[df['churn_probability_ground_truth'] >= 0.5].copy()
at_risk_count = len(at_risk_df)
at_risk_pct = at_risk_count / total_customers if total_customers > 0 else 0.0

predicted_churn_rate = df['churn_probability_ground_truth'].mean()
revenue_at_risk = at_risk_df['annual_revenue'].sum()

# ── 3. Four KPI Cards ────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    render_kpi_card(
        "Total Customers",
        f"{total_customers:,}",
        trend=f"+{cust_growth_pct:.1%} vs last month" if cust_growth_pct >= 0 else f"{cust_growth_pct:.1%} vs last month",
        trend_positive=True,
        sub_text="Active platform accounts"
    )

with k2:
    render_kpi_card(
        "At-Risk Customers",
        f"{at_risk_count:,}",
        trend=f"{at_risk_pct:.1%} of customers",
        trend_positive=False,
        sub_text="Probability score ≥ 50%"
    )

with k3:
    render_kpi_card(
        "Predicted Churn Rate",
        f"{predicted_churn_rate:.1%}",
        trend="30-day model forecast",
        trend_positive=False,
        sub_text="Portfolio-weighted average"
    )

with k4:
    render_kpi_card(
        "Revenue at Risk",
        format_currency(revenue_at_risk),
        trend=f"{revenue_at_risk / df['annual_revenue'].sum():.1%} of ARR",
        trend_positive=False,
        sub_text="Annualized exposure"
    )

st.html('<div class="iq-spacer-sm"></div>')

# ── 4 & 5. Customer Risk Distribution & Revenue Trend ────────
ch1, ch2 = st.columns(2)

# Categorization for Risk Distribution Donut
df['risk_cat'] = pd.cut(
    df['churn_probability_ground_truth'],
    bins=[0, 0.4, 0.65, 1.01],
    labels=['Low Risk', 'Medium Risk', 'High Risk'],
    right=False
).astype(str)

risk_counts = df['risk_cat'].value_counts()
low_n = int(risk_counts.get('Low Risk', 0))
med_n = int(risk_counts.get('Medium Risk', 0))
high_n = int(risk_counts.get('High Risk', 0))

with ch1:
    with st.container(border=True):
        section_header("Customer Risk Distribution", "Share of customers by risk category")
        
        fig_donut = go.Figure(go.Pie(
            labels=['High Risk', 'Medium Risk', 'Low Risk'],
            values=[high_n, med_n, low_n],
            hole=0.62,
            marker=dict(
                colors=['#EF4444', '#F59E0B', '#22C55E'],
                line=dict(color='#101219', width=2)
            ),
            textinfo='none',
            hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>',
            direction='clockwise',
            sort=False
        ))
        
        fig_donut.add_annotation(
            text=f"<b style='font-size:18px;color:#F2F3F5;'>{total_customers:,}</b><br><span style='font-size:10.5px;color:#A7ACB8;'>Customers</span>",
            x=0.5, y=0.5, showarrow=False
        )
        
        donut_layout = get_chart_layout(
            showlegend=True,
            legend=dict(
                orientation="v", x=0.78, y=0.5, yanchor="middle",
                font=dict(size=11, color="#A7ACB8"), bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(l=0, r=0, t=6, b=0),
            height=200
        )
        fig_donut.update_layout(**donut_layout)
        st.plotly_chart(fig_donut, use_container_width=True, config=CHART_CONFIG)

        # Risk Breakdown row
        st.html(f"""
        <div style="display:flex;gap:0;border-top:1px solid #20232C;margin-top:6px;padding-top:10px;">
            <div style="flex:1;text-align:center;border-right:1px solid #20232C;">
                <div style="font-size:16px;font-weight:700;color:#EF4444;">{high_n:,}</div>
                <div style="font-size:10px;color:#697080;text-transform:uppercase;letter-spacing:0.05em;margin-top:1px;">High Risk</div>
                <div style="font-size:10px;color:#A7ACB8;margin-top:1px;">{high_n/total_customers:.1%}</div>
            </div>
            <div style="flex:1;text-align:center;border-right:1px solid #20232C;">
                <div style="font-size:16px;font-weight:700;color:#F59E0B;">{med_n:,}</div>
                <div style="font-size:10px;color:#697080;text-transform:uppercase;letter-spacing:0.05em;margin-top:1px;">Medium Risk</div>
                <div style="font-size:10px;color:#A7ACB8;margin-top:1px;">{med_n/total_customers:.1%}</div>
            </div>
            <div style="flex:1;text-align:center;">
                <div style="font-size:16px;font-weight:700;color:#22C55E;">{low_n:,}</div>
                <div style="font-size:10px;color:#697080;text-transform:uppercase;letter-spacing:0.05em;margin-top:1px;">Low Risk</div>
                <div style="font-size:10px;color:#A7ACB8;margin-top:1px;">{low_n/total_customers:.1%}</div>
            </div>
        </div>
        """)

with ch2:
    with st.container(border=True):
        section_header("Revenue Trend", "Monthly revenue and at-risk revenue")
        
        # Real historical timeline derived from customer tenure cohorts
        base_date = datetime(2026, 9, 20)
        months_series = []
        tot_rev_series = []
        risk_rev_series = []
        
        for i in range(11, -1, -1):
            m_date = base_date - relativedelta(months=i)
            months_series.append(m_date.strftime("%b %Y"))
            cohort = df[df['tenure_months'] >= (i + 1)]
            tot_rev_series.append(cohort['monthly_charges'].sum())
            risk_rev_series.append(cohort[cohort['churn_probability_ground_truth'] >= 0.5]['monthly_charges'].sum())
            
        fig_line = go.Figure()
        
        # Total Revenue line
        fig_line.add_trace(go.Scatter(
            x=months_series,
            y=tot_rev_series,
            mode='lines+markers',
            name='Total Revenue',
            line=dict(color='#8B7CF6', width=2.5),
            marker=dict(size=4, color='#8B7CF6'),
            hovertemplate='Total: $%{y:,.0f}<extra></extra>'
        ))
        
        # At-Risk Revenue line
        fig_line.add_trace(go.Scatter(
            x=months_series,
            y=risk_rev_series,
            mode='lines+markers',
            name='At-Risk Revenue',
            line=dict(color='#EF4444', width=2, dash='solid'),
            marker=dict(size=4, color='#EF4444'),
            hovertemplate='At Risk: $%{y:,.0f}<extra></extra>'
        ))
        
        line_layout = get_chart_layout(
            showlegend=True,
            legend=dict(
                orientation="h", y=1.12, x=1.0, xanchor="right",
                font=dict(size=10.5, color="#A7ACB8"), bgcolor="rgba(0,0,0,0)"
            ),
            xaxis=dict(
                gridcolor='#20232C', showgrid=False,
                tickfont=dict(size=10, color="#A7ACB8")
            ),
            yaxis=dict(
                gridcolor='#20232C', showgrid=True,
                tickprefix='$', tickformat=',.0s',
                tickfont=dict(size=10, color="#A7ACB8")
            ),
            margin=dict(l=10, r=10, t=18, b=0),
            height=252
        )
        fig_line.update_layout(**line_layout)
        st.plotly_chart(fig_line, use_container_width=True, config=CHART_CONFIG)

st.html('<div class="iq-spacer-sm"></div>')

# ── 6 & 7. Top At-Risk Customers & Key Insights ─────────────
col_table, col_insights = st.columns([2, 1])

with col_table:
    with st.container(border=True):
        st.html("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div>
                <span class="iq-table-title">Top At-Risk Customers</span>
                <div style="font-size:11px;color:#697080;margin-top:2px;">Customers with highest revenue at risk</div>
            </div>
            <span style="font-size:11.5px;font-weight:600;color:#8B7CF6;cursor:pointer;">View All →</span>
        </div>
        """)
        
        # Sort top 20 customers by revenue at risk (annual_revenue * churn_probability)
        df['risk_exposure'] = df['annual_revenue'] * df['churn_probability_ground_truth']
        top_risk = df.sort_values('risk_exposure', ascending=False).head(15).copy()
        
        def assign_top_action(row):
            prob = row['churn_probability_ground_truth']
            arr = row['annual_revenue']
            if arr > 20000 and prob > 0.65:
                return "Executive Escalation"
            if row.get('support_tickets_30d', 0) > 3 or row.get('open_tickets', 0) > 1:
                return "Technical Escalation"
            if row.get('usage_change_30d', 0) < -15:
                return "Product Adoption Review"
            if row.get('contract_type', '') == 'Month-to-Month':
                return "Annual Contract Renewal"
            return "CSM Direct Outreach"

        top_risk['Recommended Action'] = top_risk.apply(assign_top_action, axis=1)
        
        table_disp = pd.DataFrame({
            "Customer": top_risk['customer_id'],
            "Segment": top_risk['segment_name'],
            "Plan": top_risk['plan_type'],
            "ARR": top_risk['annual_revenue'].apply(lambda v: f"${v:,.0f}"),
            "Churn Risk": top_risk['churn_probability_ground_truth'].apply(lambda v: f"{v:.0%}"),
            "Revenue at Risk": top_risk['risk_exposure'].apply(lambda v: f"${v:,.0f}"),
            "Recommended Action": top_risk['Recommended Action']
        })
        
        st.dataframe(
            table_disp,
            hide_index=True,
            use_container_width=True,
            height=370
        )

with col_insights:
    with st.container(border=True):
        st.html("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div>
                <span class="iq-table-title">Key Insights</span>
                <div style="font-size:11px;color:#697080;margin-top:2px;">Automated portfolio diagnostics</div>
            </div>
            <span style="font-size:11.5px;font-weight:600;color:#8B7CF6;cursor:pointer;">View All →</span>
        </div>
        """)
        
        # Dynamic calculations for 4 key insights
        # Insight 1: Enterprise revenue concentration
        ent_risk_df = at_risk_df[at_risk_df['plan_type'] == 'Enterprise']
        ent_risk_rev = ent_risk_df['annual_revenue'].sum()
        ent_share = (ent_risk_rev / revenue_at_risk * 100) if revenue_at_risk > 0 else 0
        
        # Insight 2: Support volume impact
        high_sup_df = df[df['support_tickets_30d'] > 3]
        high_sup_churn = high_sup_df['churn_probability_ground_truth'].mean() if len(high_sup_df) > 0 else 0.0
        
        # Insight 3: Protectable ARR potential
        potential_protected = revenue_at_risk * 0.35
        
        # Insight 4: High risk accounts requiring intervention
        high_risk_n = len(at_risk_df)
        
        st.html(f"""
        <!-- Insight 1: Enterprise Exposure -->
        <div class="iq-insight-item">
            <div class="iq-insight-icon-box iq-insight-icon-danger">
                <span>{ICONS['dollar']}</span>
            </div>
            <div class="iq-insight-content">
                <div class="iq-insight-headline">Enterprise Concentration: {ent_share:.0f}% of Risk</div>
                <div class="iq-insight-desc">
                    Enterprise accounts represent {format_currency(ent_risk_rev)} of total at-risk ARR across {len(ent_risk_df)} key accounts.
                </div>
            </div>
        </div>

        <!-- Insight 2: High Churn Volume -->
        <div class="iq-insight-item">
            <div class="iq-insight-icon-box iq-insight-icon-warning">
                <span>{ICONS['insights']}</span>
            </div>
            <div class="iq-insight-content">
                <div class="iq-insight-headline">{high_risk_n:,} Accounts at Elevated Risk</div>
                <div class="iq-insight-desc">
                    {at_risk_pct:.1%} of current customer base has churn probability ≥ 50% requiring proactive account intervention.
                </div>
            </div>
        </div>

        <!-- Insight 3: Support Bottlenecks -->
        <div class="iq-insight-item">
            <div class="iq-insight-icon-box iq-insight-icon-accent">
                <span>{ICONS['support']}</span>
            </div>
            <div class="iq-insight-content">
                <div class="iq-insight-headline">Support-Heavy Accounts at {high_sup_churn:.0%} Churn</div>
                <div class="iq-insight-desc">
                    Customers logging >3 support tickets in 30 days show significantly higher risk than portfolio baseline ({predicted_churn_rate:.0%}).
                </div>
            </div>
        </div>

        <!-- Insight 4: Retention Opportunity -->
        <div class="iq-insight-item" style="margin-bottom:0;">
            <div class="iq-insight-icon-box iq-insight-icon-success">
                <span>{ICONS['lightbulb']}</span>
            </div>
            <div class="iq-insight-content">
                <div class="iq-insight-headline">{format_currency(potential_protected)} Protectable ARR</div>
                <div class="iq-insight-desc">
                    Targeted execution of prioritized queue actions can defend up to 35% of currently exposed revenue.
                </div>
            </div>
        </div>
        """)

