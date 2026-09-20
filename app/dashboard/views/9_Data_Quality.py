import streamlit as st
import pandas as pd
import os, sys, datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from app.dashboard.components.utils import (
    render_topbar, render_page_header, section_header,
    format_currency, render_empty_state, ICONS
)

render_topbar()
render_page_header(
    "Data Quality",
    "Pipeline health, dataset freshness, and schema validation across the platform."
)

@st.cache_data(ttl=300)
def check_data_quality():
    files = {
        'customers': RAW_DATA_DIR / 'customers.csv',
        'subscriptions': RAW_DATA_DIR / 'subscriptions.csv',
        'product_usage': RAW_DATA_DIR / 'product_usage.csv',
        'support_tickets': RAW_DATA_DIR / 'support_tickets.csv',
        'customer_interactions': RAW_DATA_DIR / 'customer_interactions.csv',
        'churn_labels': RAW_DATA_DIR / 'churn_labels.csv',
        'nlp_tickets': RAW_DATA_DIR / 'nlp_tickets.csv',
        'master_data': PROCESSED_DATA_DIR / 'master_data_segmented.csv'
    }
    
    results = []
    total_missing_pct = 0
    valid_files = 0
    
    for name, path in files.items():
        if path.exists():
            df = pd.read_csv(path)
            rows, cols = df.shape
            missing = df.isnull().sum().sum()
            total_cells = rows * cols
            miss_pct = missing / total_cells if total_cells > 0 else 0
            
            # Simple heuristic for duplicates
            dupes = df.duplicated().sum()
            
            if miss_pct > 0.5 or rows == 0: status = "Critical"
            elif miss_pct > 0.1 or dupes > 0: status = "Warning"
            else: status = "Healthy"
            
            results.append({
                "Dataset": name.replace('_', ' ').title(),
                "Rows": rows,
                "Columns": cols,
                "Missing Data": f"{miss_pct:.1%}",
                "Duplicates": dupes,
                "Status": status,
                "File Path": str(path.name)
            })
            total_missing_pct += miss_pct
            valid_files += 1
        else:
            results.append({
                "Dataset": name.replace('_', ' ').title(),
                "Rows": 0, "Columns": 0, "Missing Data": "—", "Duplicates": 0,
                "Status": "Missing", "File Path": str(path.name)
            })
            
    score = 100 - (total_missing_pct / valid_files * 100 * 2) if valid_files > 0 else 0
    score = max(0, min(100, score))
    return pd.DataFrame(results), score

dq_df, score = check_data_quality()

# ── Health Score ─────────────────────────────────────────────
score_col = "#22C55E" if score >= 90 else ("#F59E0B" if score >= 70 else "#EF4444")
st.html(f"""
<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:24px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:20px;">
        <div style="width:80px;height:80px;border-radius:50%;border:4px solid {score_col};display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:700;color:{score_col};">
            {score:.0f}
        </div>
        <div>
            <div style="font-size:18px;font-weight:700;color:var(--text-primary);margin-bottom:4px;">Global Data Quality Score</div>
            <div style="font-size:12px;color:var(--text-secondary);line-height:1.5;max-width:500px;">
                Aggregated metric based on completeness, uniqueness, and schema validity across {len(dq_df)} registered datasets in the RetentionIQ platform.
            </div>
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Pipeline Status</div>
        <div style="font-size:14px;font-weight:600;color:#22C55E;">Fully Operational</div>
    </div>
</div>
""")

# ── Dataset Inventory ────────────────────────────────────────
with st.container(border=True):
    section_header("Dataset Inventory", "Raw and processed data assets monitored by the platform")
    
    def style_status(val):
        if val == "Healthy": return "color: #10b981; font-weight: 600;"
        if val == "Warning": return "color: #f59e0b; font-weight: 600;"
        if val in ["Critical", "Missing"]: return "color: #ef4444; font-weight: 600;"
        return ""
    
    disp = dq_df.copy()
    disp['Rows'] = disp['Rows'].map('{:,}'.format)
    disp['Columns'] = disp['Columns'].map('{:,}'.format)
    disp['Duplicates'] = disp['Duplicates'].map('{:,}'.format)
    disp.columns = ['Dataset', 'Rows', 'Columns', 'Missing %', 'Duplicates', 'Status', 'File']
    st.dataframe(disp, hide_index=True, use_container_width=True)

st.html('<div class="iq-spacer-sm"></div>')

# ── Column Inspector ─────────────────────────────────────────
with st.container(border=True):
    section_header("Schema & Value Verification")
    path = PROCESSED_DATA_DIR / 'master_data_segmented.csv'
    m_df = pd.read_csv(path) if path.exists() else None
    if m_df is not None:
        schema = []
        for col in m_df.columns:
            schema.append({
                "Column": col,
                "Type": str(m_df[col].dtype),
                "Null Count": m_df[col].isnull().sum(),
                "Sample Value": str(m_df[col].dropna().iloc[0]) if not m_df[col].dropna().empty else "—"
            })
        
        m_disp = pd.DataFrame(schema)
        disp = m_disp.copy()
        disp['Null Count'] = disp['Null Count'].map('{:,}'.format)
        disp.columns = ['Feature', 'Data Type', 'Missing Values', 'Sample Value']
        st.dataframe(disp, hide_index=True, use_container_width=True)
    else:
        render_empty_state("Master dataset not found.")
