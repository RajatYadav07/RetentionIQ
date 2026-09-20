import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.dashboard.components.utils import load_css, render_sidebar

# Setup layout & styling
from PIL import Image

logo_path = os.path.join(os.path.dirname(__file__), "assets", "retentioniq_favicon.png")
favicon = Image.open(logo_path) if os.path.exists(logo_path) else "📊"

st.set_page_config(
    page_title="RetentionIQ — Customer Intelligence",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load design system CSS first
load_css()

# ── Page registry ──────────────────────────────────────────
pages_dict = {
    "OVERVIEW": [
        st.Page("views/1_Executive_Overview.py",
                title="Executive Overview", icon=":material/dashboard:")
    ],
    "CUSTOMERS": [
        st.Page("views/2_Customer_360.py",
                title="Customer 360", icon=":material/account_circle:")
    ],
    "INTELLIGENCE": [
        st.Page("views/3_Churn_Intelligence.py",
                title="Churn Intelligence", icon=":material/trending_down:"),
        st.Page("views/3a_Customer_Segments.py",
                title="Customer Segments", icon=":material/group:"),
        st.Page("views/3b_Support_Intelligence.py",
                title="Support Intelligence", icon=":material/headset_mic:"),
    ],
    "REVENUE": [
        st.Page("views/5_Revenue_Risk.py",
                title="Revenue Risk", icon=":material/monetization_on:")
    ],
    "MODEL & DATA": [
        st.Page("views/8_Model_Performance.py",
                title="Model Performance", icon=":material/psychology:"),
        st.Page("views/9_Data_Quality.py",
                title="Data Quality", icon=":material/database:")
    ],
}

# Register pages — hide Streamlit's default nav (position="hidden")
pg = st.navigation(pages_dict, position="hidden")

# Custom sidebar
render_sidebar(pages_dict)

# Run the active page
pg.run()
