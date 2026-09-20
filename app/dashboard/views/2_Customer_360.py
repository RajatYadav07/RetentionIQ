import streamlit as st
import pandas as pd
import os, sys, logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from configs.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.explainability.explainer import get_customer_explanation
from src.recommendations.rules import generate_recommendations
from app.dashboard.components.utils import (
    render_topbar, render_page_header, render_empty_state,
    format_currency, section_header, CHART_CONFIG, get_chart_layout, ICONS, COLORS
)

logger = logging.getLogger(__name__)
render_topbar()
render_page_header("Customer 360", "Deep-dive into individual customer profiles, ML explanations, and retention strategies.")

@st.cache_data
def load_master():
    try:    return pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
    except: return None

@st.cache_data
def load_tickets():
    try:    return pd.read_csv(RAW_DATA_DIR / "nlp_tickets.csv")
    except: return None

df = load_master()
tickets_df = load_tickets()

if df is None:
    render_empty_state("Data pipeline not executed. Run: python -m scripts.generate_data")
    st.stop()

# ── Customer Search ─────────────────────────────────────────
all_ids = sorted(df['customer_id'].astype(str).unique().tolist())

# Query params or state support
default_idx = 0
if "c360_customer_id" in st.session_state and st.session_state["c360_customer_id"] in all_ids:
    default_idx = all_ids.index(st.session_state["c360_customer_id"]) + 1

with st.container(border=True):
    c1, c2 = st.columns([2, 3])
    with c1:
        selected_id = st.selectbox(
            "Search Customer ID",
            options=[""] + all_ids,
            index=default_idx,
            key="c360_customer_id",
            format_func=lambda x: "Select or type a Customer ID..." if x == "" else x,
        )
    with c2:
        if not selected_id:
            st.html('<div style="padding-top:26px;font-size:12px;color:#4b5563;">Type to search — 5,000 accounts available. Example: C00001</div>')

if not selected_id:
    st.html('<div class="iq-spacer-md"></div>')
    section_header("Top At-Risk Accounts", "Click or search a customer ID to open their full profile", count=8)
    sample = df.nlargest(8, 'churn_probability_ground_truth')[
        ['customer_id', 'segment_name', 'plan_type', 'industry',
         'annual_revenue', 'churn_probability_ground_truth']].copy()
    sample['annual_revenue'] = sample['annual_revenue'].apply(lambda x: f"${x:,.0f}")
    sample['churn_probability_ground_truth'] = sample['churn_probability_ground_truth'].apply(lambda x: f"{x:.0%}")
    sample.columns = ['Customer ID', 'Segment', 'Plan', 'Industry', 'ARR', 'Churn Risk']
    with st.container(border=True):
        st.dataframe(sample, hide_index=True, use_container_width=True)
    st.stop()

# Safe string-normalized lookup
selected_id = str(selected_id).strip()
cust_rows = df[df['customer_id'].astype(str) == selected_id]
if cust_rows.empty:
    render_empty_state(f"Customer '{selected_id}' not found in the dataset.")
    st.stop()

cust = cust_rows.iloc[0]

# ── Load ML Explanation ──────────────────────────────────────
explanation, rec = None, None
with st.spinner("Loading ML explanation..."):
    try:
        explanation = get_customer_explanation(cust_rows.drop(columns=['churned'], errors='ignore'))
        rec = generate_recommendations(cust, explanation['churn_probability'])
    except FileNotFoundError:
        st.error("Model pipeline not found. Run training pipeline first.")
        st.stop()
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        prob = float(cust.get('churn_probability_ground_truth', 0))
        explanation = {'churn_probability': prob, 'top_drivers': [], 'top_protectors': []}
        rec = generate_recommendations(cust, prob)

prob  = explanation['churn_probability']
score = rec.get('priority_score', 0) if rec else 0
risk  = rec.get('risk_level', 'Low').upper() if rec else "UNKNOWN"

risk_color = "#EF4444" if prob >= 0.7 else ("#F59E0B" if prob >= 0.4 else "#22C55E")
risk_badge_cls = "iq-badge-danger" if prob >= 0.7 else ("iq-badge-warning" if prob >= 0.4 else "iq-badge-success")

# ── Profile Header Card ──────────────────────────────────────
initials = "".join(w[0].upper() for w in selected_id.replace("C","").split() if w)[:2] or selected_id[:2].upper()
arr      = float(cust['annual_revenue'])
exp_loss = arr * prob

st.html(f"""
<div class="iq-profile-header">
    <div class="iq-profile-avatar">{initials}</div>
    <div class="iq-profile-info">
        <div class="iq-profile-name">{cust['customer_id']}</div>
        <div class="iq-profile-meta">
            {cust.get('industry','—')} &nbsp;·&nbsp; {cust.get('country','—')}
            &nbsp;·&nbsp; {cust.get('segment_name','—')} &nbsp;·&nbsp; {cust.get('plan_type','—')} Plan
            &nbsp;·&nbsp; {cust.get('tenure_months',0):.0f} months tenure
        </div>
    </div>
    <div class="iq-profile-stats">
        <div class="iq-profile-stat">
            <span class="iq-profile-stat-label">ARR</span>
            <span class="iq-profile-stat-value">{format_currency(arr)}</span>
        </div>
        <div class="iq-profile-stat">
            <span class="iq-profile-stat-label">Churn Risk</span>
            <span class="iq-profile-stat-value" style="color:{risk_color};">{prob:.0%}</span>
        </div>
        <div class="iq-profile-stat">
            <span class="iq-profile-stat-label">Priority Score</span>
            <span class="iq-profile-stat-value">{score:.0f}<span style="font-size:12px;color:#697080;">/100</span></span>
        </div>
        <div class="iq-profile-stat">
            <span class="iq-profile-stat-label">Risk Level</span>
            <span><span class="iq-badge {risk_badge_cls}" style="font-size:12px;padding:3px 10px;">{risk}</span></span>
        </div>
    </div>
</div>
""")

# ── Tabs ────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["Action & Explanation", "Profile & Commercial", "Usage & Engagement", "Support History"])

# ─── Tab 1: Action + SHAP ────────────────────────────────────
with t1:
    a1, a2 = st.columns([1, 1])

    with a1:
        with st.container(border=True):
            section_header("Recommended Action", "Generated by the deterministic rules engine")
            if rec:
                action    = rec.get('recommended_action', '—')
                reason    = rec.get('reason', '—')
                objective = rec.get('expected_objective', '—')
                urgency   = rec.get('urgency', 'Low')
                urg_cls   = {"Critical": "danger", "High": "warning", "Medium": "blue"}.get(urgency, "neutral")
                st.html(f"""
                <div style="background:rgba(139,124,246,0.06);border:1px solid rgba(139,124,246,0.2);
                            border-radius:6px;padding:14px;margin-bottom:12px;">
                    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:8px;">
                        <span style="font-size:13.5px;font-weight:600;color:#F2F3F5;">{action}</span>
                        <span class="iq-badge iq-badge-{urg_cls}">{urgency}</span>
                    </div>
                    <div style="font-size:12px;color:#A7ACB8;margin-bottom:8px;line-height:1.5;">{reason}</div>
                    <div style="font-size:11px;color:#697080;border-top:1px solid #20232C;padding-top:8px;">
                        <strong style="color:#A7ACB8;">Objective:</strong> {objective}
                    </div>
                </div>
                """)
            st.html('<div style="border-top:1px solid #20232C;margin-bottom:10px;"></div>')
            e1, e2 = st.columns(2)
            e1.metric("ARR at Risk", format_currency(arr))
            e2.metric("Expected Loss", format_currency(exp_loss), f"P={prob:.0%}", delta_color="inverse")

    with a2:
        with st.container(border=True):
            section_header("SHAP Feature Contributions", "Top churn drivers for this customer")
            drivers    = explanation.get('top_drivers', [])
            protectors = explanation.get('top_protectors', [])
            if drivers:
                st.html('<div style="font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Risk Drivers</div>')
                for d in drivers:
                    feat  = d['feature'].replace("num__","").replace("cat__","").replace("_"," ").title()
                    impact = d['impact']
                    w = min(int(abs(impact) * 250), 100)
                    st.html(f"""
                    <div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #1e2235;">
                        <span style="font-size:12px;color:#cbd5e1;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{feat}</span>
                        <div style="width:64px;height:4px;background:#1e2235;border-radius:2px;flex-shrink:0;">
                            <div style="width:{w}%;height:100%;background:#ef4444;border-radius:2px;"></div>
                        </div>
                        <span style="font-size:12px;font-weight:600;color:#f87171;min-width:46px;text-align:right;">+{impact:.3f}</span>
                    </div>
                    """)
                if protectors:
                    st.html('<div style="font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin:12px 0 8px;">Protective Factors</div>')
                    for p in protectors:
                        feat   = p['feature'].replace("num__","").replace("cat__","").replace("_"," ").title()
                        impact = p['impact']
                        w = min(int(abs(impact) * 250), 100)
                        st.html(f"""
                        <div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #1e2235;">
                            <span style="font-size:12px;color:#cbd5e1;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{feat}</span>
                            <div style="width:64px;height:4px;background:#1e2235;border-radius:2px;flex-shrink:0;">
                                <div style="width:{w}%;height:100%;background:#10b981;border-radius:2px;"></div>
                            </div>
                            <span style="font-size:12px;font-weight:600;color:#34d399;min-width:46px;text-align:right;">{impact:.3f}</span>
                        </div>
                        """)
            else:
                st.html('<div style="font-size:12px;color:#4b5563;padding:24px 0;text-align:center;">SHAP explanations not available for this customer.</div>')

# ─── Tab 2: Profile ───────────────────────────────────────────
with t2:
    with st.container(border=True):
        section_header("Account Profile")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Customer ID", cust['customer_id'])
        r1c2.metric("Industry",    cust.get('industry', '—'))
        r1c3.metric("Country",     cust.get('country', '—'))
        r1c4.metric("Company Size",cust.get('company_size', '—'))

        st.divider()
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("Plan",          cust.get('plan_type', '—'))
        r2c2.metric("Contract Type", cust.get('contract_type', '—'))
        r2c3.metric("Tenure (mo)",   f"{cust.get('tenure_months', 0):.0f}")
        r2c4.metric("Auto-Renew",    "Yes" if cust.get('auto_renew') else "No")

        st.divider()
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        r3c1.metric("ARR",            format_currency(arr))
        r3c2.metric("Monthly Charges",format_currency(float(cust.get('monthly_charges', 0))))
        r3c3.metric("Discount",       f"{cust.get('discount_percent', 0):.1f}%")
        r3c4.metric("Segment",        cust.get('segment_name', '—'))

# ─── Tab 3: Usage ─────────────────────────────────────────────
with t3:
    with st.container(border=True):
        section_header("Usage & Engagement")
        u1, u2, u3, u4 = st.columns(4)
        u30 = float(cust.get('usage_change_30d', 0))
        u90 = float(cust.get('usage_change_90d', 0))
        u1.metric("30d Usage Change", f"{u30:.1f}%", delta_color="normal" if u30 >= 0 else "inverse")
        u2.metric("90d Usage Change", f"{u90:.1f}%", delta_color="normal" if u90 >= 0 else "inverse")
        u3.metric("Login Frequency",  f"{cust.get('login_frequency', 0):.0f}/mo")
        u4.metric("Feature Adoption", f"{cust.get('feature_adoption_rate', 0):.0%}")
        st.divider()
        u5, u6, u7, u8 = st.columns(4)
        u5.metric("Weekly Active Days",  f"{cust.get('weekly_active_days', 0):.1f}")
        u6.metric("Monthly Active Users",f"{cust.get('monthly_active_users', 0):.0f}")
        u7.metric("Emails Opened",       f"{cust.get('emails_opened', 0):.0f}")
        u8.metric("Emails Clicked",      f"{cust.get('emails_clicked', 0):.0f}")

# ─── Tab 4: Support ───────────────────────────────────────────
with t4:
    with st.container(border=True):
        section_header("Support Intelligence")
        health = float(cust.get('customer_support_health_score', 100))
        sent   = float(cust.get('sentiment_trend', 0))
        sent_label = "Positive" if sent > 0.1 else ("Negative" if sent < -0.1 else "Neutral")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Health Score",    f"{health:.0f}/100")
        s2.metric("Dominant Topic",  str(cust.get('dominant_support_topic', '—')))
        s3.metric("Sentiment",       sent_label)
        s4.metric("CSAT",            f"{cust.get('csat_score', 0):.1f}/5.0")
        st.divider()
        s5, s6, s7, s8 = st.columns(4)
        s5.metric("Tickets 30d",   f"{cust.get('support_tickets_30d', 0):.0f}")
        s6.metric("Tickets 90d",   f"{cust.get('support_tickets_90d', 0):.0f}")
        s7.metric("Open Tickets",  f"{cust.get('open_tickets', 0):.0f}")
        s8.metric("Avg Resolution",f"{cust.get('avg_resolution_hours', 0):.1f}h")
        st.divider()
        s9, s10 = st.columns(2)
        s9.metric("Escalations", f"{cust.get('escalation_count', 0):.0f}")
        s10.metric("NPS Score",  f"{cust.get('nps_score', 0):.1f}")

    if tickets_df is not None:
        cust_tickets = tickets_df[tickets_df['customer_id'] == selected_id].copy()
        if not cust_tickets.empty:
            st.html('<div class="iq-spacer-sm"></div>')
            with st.container(border=True):
                section_header("Support Ticket History", f"{len(cust_tickets)} tickets on record", count=len(cust_tickets))
                cols_to_show = [c for c in ['ticket_id', 'issue_category', 'sentiment', 'priority', 'description'] if c in cust_tickets.columns]
                disp_tickets = cust_tickets[cols_to_show].copy()
                disp_tickets.columns = [c.replace('_', ' ').title() for c in cols_to_show]
                st.dataframe(disp_tickets, hide_index=True, use_container_width=True)
        else:
            st.html(f'<div style="font-size:12px;color:#4b5563;padding:16px 0;">No support tickets on file for {selected_id}.</div>')
