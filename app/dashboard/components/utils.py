"""
RetentionIQ — UI Component Library
Premium dark SaaS design system utilities.
"""
import streamlit as st
import os
import pandas as pd
from datetime import datetime


# ============================================================
# DESIGN TOKENS (mirrored from CSS for Python use)
# ============================================================
COLORS = {
    "bg_base":      "#08090D",
    "bg_card":      "#101219",
    "bg_sidebar":   "#0B0D13",
    "bg_input":     "#12151D",
    "border":       "#20232C",
    "accent":       "#8B7CF6",
    "accent_hover": "#9B8CFF",
    "success":      "#22C55E",
    "warning":      "#F59E0B",
    "danger":       "#EF4444",
    "text_primary": "#F2F3F5",
    "text_secondary":"#A7ACB8",
    "text_muted":   "#697080",
}

CHART_LAYOUT = {
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font":          {"family": "Inter, system-ui, sans-serif", "color": "#A7ACB8", "size": 12},
    "margin":        {"l": 0, "r": 0, "t": 28, "b": 0},
    "xaxis":         {"gridcolor": "#20232C", "showgrid": False, "tickfont": {"size": 11, "color": "#A7ACB8"}, "zeroline": False},
    "yaxis":         {"gridcolor": "#20232C", "showgrid": True, "tickfont": {"size": 11, "color": "#A7ACB8"}, "zeroline": False},
    "legend":        {"font": {"size": 11, "color": "#A7ACB8"}, "bgcolor": "rgba(0,0,0,0)"},
    "hoverlabel":    {"bgcolor": "#101219", "bordercolor": "#20232C", "font_size": 12, "font_family": "Inter", "font_color": "#F2F3F5"},
}
CHART_CONFIG = {"displayModeBar": False}


# ============================================================
# CSS LOADER
# ============================================================
def load_css():
    """Inject the design system CSS."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.html(f"<style>{f.read()}</style>")


# ============================================================
# SVG ICONS (Lucide-style, 14x14 by default)
# ============================================================
ICONS = {
    "dashboard":    '<span class="material-symbols-rounded" style="font-size:inherit">dashboard</span>',
    "users":        '<span class="material-symbols-rounded" style="font-size:inherit">group</span>',
    "insights":     '<span class="material-symbols-rounded" style="font-size:inherit">trending_down</span>',
    "grid":         '<span class="material-symbols-rounded" style="font-size:inherit">grid_view</span>',
    "support":      '<span class="material-symbols-rounded" style="font-size:inherit">headset_mic</span>',
    "dollar":       '<span class="material-symbols-rounded" style="font-size:inherit">monetization_on</span>',
    "checklist":    '<span class="material-symbols-rounded" style="font-size:inherit">checklist</span>',
    "lightbulb":    '<span class="material-symbols-rounded" style="font-size:inherit">lightbulb</span>',
    "model":        '<span class="material-symbols-rounded" style="font-size:inherit">psychology</span>',
    "database":     '<span class="material-symbols-rounded" style="font-size:inherit">database</span>',
    "search":       '<span class="material-symbols-rounded" style="font-size:inherit">search</span>',
    "bell":         '<span class="material-symbols-rounded" style="font-size:inherit">notifications</span>',
    "chevron":      '<span class="material-symbols-rounded" style="font-size:inherit">expand_more</span>',
    "download":     '<span class="material-symbols-rounded" style="font-size:inherit">download</span>',
    "calendar":     '<span class="material-symbols-rounded" style="font-size:inherit">calendar_today</span>',
    "arrow_up":     '<span class="material-symbols-rounded" style="font-size:inherit">arrow_upward</span>',
    "arrow_down":   '<span class="material-symbols-rounded" style="font-size:inherit">arrow_downward</span>',
    "minus":        '<span class="material-symbols-rounded" style="font-size:inherit">remove</span>',
    "filter":       '<span class="material-symbols-rounded" style="font-size:inherit">filter_alt</span>',
}


# ============================================================
# SIDEBAR RENDERER
# ============================================================
def render_sidebar(pages_dict):
    """Renders the custom premium RetentionIQ sidebar."""
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    with st.sidebar:
        # Brand
        logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "retentioniq_logo.png")
        img_tag = ""
        if os.path.exists(logo_path):
            import base64
            with open(logo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_tag = f'<img src="data:image/png;base64,{b64}" class="iq-brand-logo" />'
            
        st.html(f"""
        <div class="iq-sidebar-brand">
            <div class="iq-brand-row">
                {img_tag}
                <div class="iq-brand-text">
                    <div class="iq-brand-name">RetentionIQ</div>
                    <div class="iq-brand-sub">Customer Intelligence Platform</div>
                </div>
            </div>
        </div>
        """)

        # Navigation
        for section, pages in pages_dict.items():
            st.html(f'<div class="iq-nav-section">{section}</div>')
            for page in pages:
                st.page_link(page, label=page.title, icon=page.icon)

        # Status Footer (flex footer anchored at bottom)
        st.html(f"""
        <div class="iq-sidebar-status">
            <div class="iq-status-dot"></div>
            <div class="iq-status-text">
                <span class="iq-status-title">Model Active</span>
                <span class="iq-status-sub">Updated {now}</span>
            </div>
        </div>
        """)


# ============================================================
# TOP APP BAR
# ============================================================
def render_topbar(search_placeholder: str = "Search customers, accounts, or insights...",
                  user_name: str = "Rajat Yadav",
                  user_role: str = "Data Scientist",
                  user_initials: str = "RY"):
    """Renders the premium top application bar."""
    
    # Base64 SVGs to avoid any literal text rendering or font issues
    import base64
    search_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'
    bell_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>'
    chevron_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>'
    
    search_b64 = base64.b64encode(search_svg.encode()).decode()
    bell_b64 = base64.b64encode(bell_svg.encode()).decode()
    chevron_b64 = base64.b64encode(chevron_svg.encode()).decode()

    st.html(f"""
    <div class="iq-topbar">
        <div class="iq-topbar-left">
            <div class="iq-search-box">
                <img src="data:image/svg+xml;base64,{search_b64}" width="16" height="16" style="flex-shrink:0;" />
                <span class="iq-search-text">{search_placeholder}</span>
                <span class="iq-search-kbd">Ctrl K</span>
            </div>
        </div>
        <div class="iq-topbar-right">
            <div class="iq-icon-btn" title="Notifications">
                <img src="data:image/svg+xml;base64,{bell_b64}" width="18" height="18" />
            </div>
            <div class="iq-topbar-divider"></div>
            <div class="iq-user-pill">
                <div class="iq-avatar">{user_initials}</div>
                <div class="iq-user-info">
                    <span class="iq-user-name">{user_name}</span>
                    <span class="iq-user-role">{user_role}</span>
                </div>
                <img src="data:image/svg+xml;base64,{chevron_b64}" width="14" height="14" style="margin-left:4px;" />
            </div>
        </div>
    </div>
    """)


# ============================================================
# PAGE HEADER
# ============================================================
def render_page_header(title: str, subtitle: str, actions_html: str = ""):
    """Renders the standard page header."""
    st.html(f"""
    <div class="iq-page-header">
        <div>
            <div class="iq-page-title">{title}</div>
            <div class="iq-page-subtitle">{subtitle}</div>
        </div>
        <div class="iq-header-actions">{actions_html}</div>
    </div>
    """)


# ============================================================
# KPI CARD
# ============================================================
def render_kpi_card(title: str, value: str, trend: str = None,
                    trend_positive: bool = True, sub_text: str = None):
    """Renders a compact KPI card using st.container with metric."""
    with st.container(border=True):
        st.metric(
            label=title,
            value=value,
            delta=trend,
            delta_color="normal" if trend_positive else "inverse"
        )
        if sub_text:
            st.html(f'<div class="iq-kpi-sub">{sub_text}</div>')


# ============================================================
# BADGE
# ============================================================
def risk_badge(value: float) -> str:
    """Returns HTML badge string for a risk value."""
    if value >= 0.65:
        return f'<span class="iq-badge iq-badge-danger">High</span>'
    elif value >= 0.4:
        return f'<span class="iq-badge iq-badge-warning">Medium</span>'
    else:
        return f'<span class="iq-badge iq-badge-success">Low</span>'


def status_badge(status: str) -> str:
    """Returns HTML badge for status strings."""
    s = str(status).lower()
    if s in ("healthy", "pass", "active", "resolved"):
        return f'<span class="iq-badge iq-badge-success">{status}</span>'
    elif s in ("warning", "medium risk", "in progress", "contacted"):
        return f'<span class="iq-badge iq-badge-warning">{status}</span>'
    elif s in ("critical", "fail", "high risk", "error", "missing"):
        return f'<span class="iq-badge iq-badge-danger">{status}</span>'
    else:
        return f'<span class="iq-badge iq-badge-neutral">{status}</span>'


# ============================================================
# SECTION TITLE HELPER
# ============================================================
def section_header(title: str, subtitle: str = "", count: int = None):
    """Renders a compact section title row."""
    count_html = f'<span class="iq-table-count">{count:,}</span>' if count is not None else ""
    st.html(f"""
    <div class="iq-table-header">
        <div>
            <span class="iq-table-title">{title}</span>
            {'<div style="font-size:11px;color:#697080;margin-top:2px">' + subtitle + '</div>' if subtitle else ''}
        </div>
        {count_html}
    </div>
    """)


# ============================================================
# EMPTY STATE
# ============================================================
def render_empty_state(message: str, icon: str = "database"):
    """Renders a professional empty/error state."""
    st.html(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                padding:48px 24px;text-align:center;color:#697080;">
        <div style="width:40px;height:40px;border-radius:8px;background:#101219;border:1px solid #20232C;
                    display:flex;align-items:center;justify-content:center;margin-bottom:12px;color:#697080;">
            {ICONS.get(icon, ICONS['database'])}
        </div>
        <div style="font-size:13px;font-weight:600;color:#A7ACB8;margin-bottom:4px;">No data available</div>
        <div style="font-size:12px;color:#697080;max-width:320px;line-height:1.5;">{message}</div>
    </div>
    """)


# ============================================================
# HELPERS
# ============================================================
def format_currency(val: float) -> str:
    """Formats a float as compact currency string."""
    if val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    elif val >= 1_000:
        return f"${val/1_000:.0f}K"
    return f"${val:,.0f}"


def format_currency_full(val: float) -> str:
    return f"${val:,.0f}"


def style_dataframe_risk(val):
    """DEPRECATED — do not use with st.dataframe(df.style). Kept for legacy compat."""
    return ""


def get_chart_layout(**overrides) -> dict:
    """Returns a base plotly layout dict merged with any overrides."""
    layout = {**CHART_LAYOUT}
    layout.update(overrides)
    return layout
