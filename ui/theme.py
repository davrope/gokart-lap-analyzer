from __future__ import annotations

from typing import Callable

import streamlit as st


HOME_PAGE = "app.py"
UPLOAD_PAGE = "pages/1_Upload_Attempt.py"
ATTEMPT_PAGE = "pages/2_Attempt_Analysis.py"
HISTORY_PAGE = "pages/3_Track_History.py"
FURTHER_PAGE = "pages/4_Further_Analysis.py"


def _go(page: str) -> None:
    try:
        st.switch_page(page)
    except Exception:
        st.info("Page navigation is temporarily unavailable. Use the top navigation buttons.")


def configure_page(page_title: str, page_icon: str = "🏁") -> None:
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_global_theme()


def apply_global_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
  --bg: #080a0d;
  --bg-soft: #0f141a;
  --surface: rgba(255, 255, 255, 0.04);
  --surface-border: rgba(255, 255, 255, 0.12);
  --text-main: #f6f7f9;
  --text-muted: #aab3bf;
  --accent: #ff4d2d;
  --accent-2: #ff9f1a;
}

.stApp {
  background:
    radial-gradient(1300px 600px at 10% -10%, rgba(255, 77, 45, 0.18), transparent 50%),
    radial-gradient(1000px 500px at 110% -20%, rgba(255, 159, 26, 0.15), transparent 45%),
    linear-gradient(155deg, #06080b 0%, #0a0d12 40%, #080a0d 100%);
  color: var(--text-main);
}

.stApp::before {
  content: "";
  position: fixed;
  inset: -20% -20%;
  pointer-events: none;
  background: repeating-linear-gradient(
    35deg,
    transparent,
    transparent 100px,
    rgba(255, 77, 45, 0.035) 100px,
    rgba(255, 77, 45, 0.035) 170px
  );
  z-index: 0;
}

.main .block-container {
  max-width: 1320px;
  padding-top: 1.2rem;
  padding-bottom: 2.4rem;
  position: relative;
  z-index: 1;
}

h1, h2, h3 {
  font-family: "Bebas Neue", sans-serif;
  letter-spacing: 0.04em;
  color: var(--text-main);
}

p, li, label, div, span, [data-testid="stMarkdownContainer"] {
  font-family: "Rajdhani", sans-serif;
}

[data-testid="stSidebarNav"] {
  display: none !important;
}
[data-testid="stSidebar"] {
  display: none !important;
}
[data-testid="collapsedControl"] {
  display: none !important;
}

header[data-testid="stHeader"] {
  background: transparent;
}

div[data-testid="metric-container"] {
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 14px;
  padding: 0.9rem 0.95rem;
}

.app-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.6rem;
  background: rgba(6, 10, 15, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 14px;
  padding: 0.7rem 0.95rem;
  backdrop-filter: blur(8px);
}

.brand-title {
  margin: 0;
  font-family: "Bebas Neue", sans-serif;
  letter-spacing: 0.08em;
  font-size: 1.45rem;
  line-height: 1;
}

.brand-sub {
  color: var(--text-muted);
  font-size: 0.93rem;
  margin-top: 0.1rem;
}

.user-chip {
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 999px;
  padding: 0.3rem 0.75rem;
  color: var(--text-main);
  font-weight: 600;
  font-size: 0.86rem;
  white-space: nowrap;
}

.hero-panel {
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 16px;
  padding: 1.4rem 1.2rem;
  background: linear-gradient(150deg, rgba(255, 77, 45, 0.19), rgba(255, 159, 26, 0.1) 55%, rgba(255, 255, 255, 0.02));
}

.hero-kicker {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.78rem;
  color: #ffd0c4;
  font-weight: 700;
}

.hero-headline {
  margin: 0.5rem 0 0.6rem 0;
  font-size: 2.8rem;
  line-height: 0.95;
}

.hero-copy {
  color: #e7ebf1;
  font-size: 1.08rem;
}

.glass-card {
  background: rgba(10, 14, 19, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 14px;
  padding: 1rem 1rem;
}

.section-title {
  font-family: "Bebas Neue", sans-serif;
  font-size: 2rem;
  letter-spacing: 0.07em;
  margin: 1rem 0 0.6rem 0;
}

.step-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.75rem;
}

.step-item {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-left: 4px solid var(--accent);
  border-radius: 10px;
  padding: 0.85rem 0.8rem;
}

.step-num {
  color: var(--accent-2);
  font-weight: 700;
  font-size: 0.83rem;
  letter-spacing: 0.08em;
}

.step-name {
  color: #ffffff;
  font-size: 1rem;
  font-weight: 700;
  margin-top: 0.2rem;
}

.step-copy {
  color: var(--text-muted);
  margin-top: 0.15rem;
  font-size: 0.95rem;
}

.stButton > button {
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.18);
}

.stTextInput input, .stSelectbox [data-baseweb="select"], .stTextArea textarea {
  background: rgba(255, 255, 255, 0.04) !important;
}

[data-testid="stDataFrame"] {
  border-radius: 10px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_top_nav(
    active_page: str,
    *,
    user_label: str | None = None,
    show_signout: bool = False,
    on_signout: Callable[[], None] | None = None,
    show_links: bool = True,
) -> None:
    user_html = f'<div class="user-chip">{user_label}</div>' if user_label else ""
    st.markdown(
        f"""
<div class="app-topbar">
  <div>
    <p class="brand-title">GoKart Coaching Platform</p>
    <div class="brand-sub">Telemetry intelligence for faster and more consistent lap times</div>
  </div>
  {user_html}
</div>
""",
        unsafe_allow_html=True,
    )

    if not show_links:
        return

    nav_cols = st.columns([1, 1, 1, 1, 0.8], gap="small")
    with nav_cols[0]:
        if st.button("Home", key=f"nav.{active_page}.home", use_container_width=True, type="primary" if active_page == "home" else "secondary"):
            _go(HOME_PAGE)
    with nav_cols[1]:
        if st.button("Upload", key=f"nav.{active_page}.upload", use_container_width=True, type="primary" if active_page == "upload" else "secondary"):
            _go(UPLOAD_PAGE)
    with nav_cols[2]:
        if st.button("Attempt", key=f"nav.{active_page}.attempt", use_container_width=True, type="primary" if active_page == "attempt" else "secondary"):
            _go(ATTEMPT_PAGE)
    with nav_cols[3]:
        if st.button("History", key=f"nav.{active_page}.history", use_container_width=True, type="primary" if active_page == "history" else "secondary"):
            _go(HISTORY_PAGE)
    with nav_cols[4]:
        if show_signout and st.button("Sign out", key=f"nav.{active_page}.signout", use_container_width=True):
            if on_signout is not None:
                on_signout()
            else:
                st.session_state.clear()
                _go(HOME_PAGE)


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="hero-panel">
  <div class="hero-kicker">Session Workspace</div>
  <h1 class="hero-headline">{title}</h1>
  <p class="hero-copy">{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )
