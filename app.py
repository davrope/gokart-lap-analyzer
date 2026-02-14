from __future__ import annotations

import numpy as np
import streamlit as st

from services import (
    bootstrap_auth_session_from_query,
    get_app_config,
    get_attempt_repository,
    get_authenticated_user,
    get_auth_service,
    get_track_repository,
    sign_out_user,
    supabase_configured,
)
from ui import ATTEMPT_PAGE, FURTHER_PAGE, HISTORY_PAGE, UPLOAD_PAGE, configure_page, render_page_header, render_top_nav


def _go(page: str) -> None:
    try:
        st.switch_page(page)
    except Exception:
        st.info("Page navigation is temporarily unavailable. Use the top navigation buttons.")


def _sign_out() -> None:
    sign_out_user()
    _go("app.py")


def _render_google_button(url: str) -> None:
    st.markdown(
        f"""
<a href="{url}" target="_self" style="display:flex;align-items:center;justify-content:center;gap:10px;padding:0.62rem 0.9rem;border:1px solid #dadce0;border-radius:10px;text-decoration:none;background:#fff;color:#1f2937;font-weight:600;">
  <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
    <path d="M17.64 9.2045C17.64 8.56632 17.5827 7.95268 17.4764 7.36359H9V10.8454H13.8436C13.635 11.9704 12.9977 12.9232 12.0418 13.5614V15.8195H14.9509C16.6527 14.2523 17.64 11.9459 17.64 9.2045Z" fill="#4285F4"/>
    <path d="M9 18C11.43 18 13.4673 17.1941 14.9509 15.8195L12.0418 13.5614C11.2359 14.1014 10.2068 14.42 9 14.42C6.65591 14.42 4.67182 12.8364 3.96409 10.7082H0.956818V13.04C2.43273 15.9723 5.46727 18 9 18Z" fill="#34A853"/>
    <path d="M3.96409 10.7082C3.78409 10.1682 3.68182 9.59182 3.68182 9C3.68182 8.40818 3.78409 7.83182 3.96409 7.29182V4.96H0.956818C0.349091 6.17045 0 7.54409 0 9C0 10.4559 0.349091 11.8295 0.956818 13.04L3.96409 10.7082Z" fill="#FBBC05"/>
    <path d="M9 3.58C10.3164 3.58 11.4995 4.03273 12.4295 4.92182L15.0164 2.33491C13.4632 0.878182 11.4259 0 9 0C5.46727 0 2.43273 2.02773 0.956818 4.96L3.96409 7.29182C4.67182 5.16364 6.65591 3.58 9 3.58Z" fill="#EA4335"/>
  </svg>
  Continue with Google
</a>
""",
        unsafe_allow_html=True,
    )


configure_page("GoKart Coaching Platform")

cfg = get_app_config()
if not cfg.multi_attempt_mode:
    st.error("`MULTI_ATTEMPT_MODE` is disabled. Enable it in env/secrets to use this workflow.")
    st.stop()

status_msg = bootstrap_auth_session_from_query()
if status_msg:
    st.success(status_msg)

if not supabase_configured(cfg):
    render_top_nav(active_page="home", show_links=False)
    st.error(
        "Supabase is not configured. Add `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `APP_BASE_URL` in Streamlit secrets or `.env`."
    )
    st.code(
        """
SUPABASE_URL="https://<project>.supabase.co"
SUPABASE_ANON_KEY="<anon-key>"
APP_BASE_URL="https://<your-streamlit-app-url>"
MULTI_ATTEMPT_MODE="true"
""".strip()
    )
    st.stop()

user = get_authenticated_user()

if user is None:
    render_top_nav(active_page="home", show_links=False)
    left, right = st.columns([1.45, 1.0], gap="large", vertical_alignment="top")

    with left:
        st.markdown(
            """
<div class="hero-panel">
  <div class="hero-kicker">GoKart Data Coach</div>
  <h1 class="hero-headline">Find and fix the corners where lap time is leaking.</h1>
  <p class="hero-copy">
    Upload Garmin FIT attempts, detect laps and curves automatically, compare historical progression,
    and train with objective telemetry-driven guidance.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Telemetry", "GPS + Speed")
        m2.metric("Analysis", "Lap + Curve")
        m3.metric("Progress", "Attempt History")

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title" style="font-size:1.55rem;margin-top:0.1rem;">Sign In</p>', unsafe_allow_html=True)
        st.caption("Passwordless access with Supabase auth")

        google_url = ""
        try:
            google_url = get_auth_service().get_google_auth_url()
        except Exception as exc:
            st.warning(f"Google login unavailable: {exc}")

        if google_url:
            _render_google_button(google_url)
            st.caption("If Google fails, verify provider config and redirect URLs in Supabase.")

        st.markdown("##### Email magic link")
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="driver@example.com")
            submitted = st.form_submit_button("Send magic link", type="primary", use_container_width=True)

        if submitted:
            try:
                get_auth_service().send_magic_link(email)
                st.success("Magic link sent. Check your email and return to this page.")
            except Exception as exc:
                st.error(f"Failed to send magic link: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="section-title">What You Can Do</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="small")
    with f1:
        st.markdown(
            """
<div class="step-item">
  <div class="step-num">ATTEMPT ANALYSIS</div>
  <div class="step-name">Inspect one session deeply</div>
  <div class="step-copy">Lap metrics, speed profile, line geometry, and curve-level behavior in one flow.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
<div class="step-item">
  <div class="step-num">TRACK HISTORY</div>
  <div class="step-name">Track progress over time</div>
  <div class="step-copy">Best lap trend, consistency drift, and curve-specific opportunities per attempt.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
<div class="step-item">
  <div class="step-num">COACHING LAYER</div>
  <div class="step-name">Generate structured insights</div>
  <div class="step-copy">A recommendation architecture ready for a future LLM provider integration.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-title">Workflow</p>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="step-grid">
  <div class="step-item">
    <div class="step-num">01</div>
    <div class="step-name">Authenticate</div>
    <div class="step-copy">Email magic link or Google sign-in through Supabase.</div>
  </div>
  <div class="step-item">
    <div class="step-num">02</div>
    <div class="step-name">Upload FIT Attempt</div>
    <div class="step-copy">Attach the attempt to an existing track or create a new one.</div>
  </div>
  <div class="step-item">
    <div class="step-num">03</div>
    <div class="step-name">Review Attempt</div>
    <div class="step-copy">Understand where speed is gained or lost across each lap and curve.</div>
  </div>
  <div class="step-item">
    <div class="step-num">04</div>
    <div class="step-name">Compare History</div>
    <div class="step-copy">Validate improvements and prioritize your next training actions.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

render_top_nav(
    active_page="home",
    user_label=user.email or user.id,
    show_signout=True,
    on_signout=_sign_out,
)

render_page_header(
    "Driver Command Center",
    "Pick your next action: upload fresh telemetry, inspect a specific attempt, or review track progress.",
)

tracks_count = 0
attempts_count = 0
best_lap_s = np.nan

try:
    track_repo = get_track_repository()
    attempt_repo = get_attempt_repository()

    tracks = track_repo.list_tracks(user.id)
    attempts = attempt_repo.list_attempts(user.id)

    tracks_count = len(tracks)
    attempts_count = len(attempts)

    vals = [float(a.get("best_lap_s")) for a in attempts if a.get("best_lap_s") is not None]
    vals = [v for v in vals if np.isfinite(v)]
    if vals:
        best_lap_s = min(vals)
except Exception as exc:
    st.warning(f"Could not load dashboard stats: {exc}")

c1, c2, c3 = st.columns(3)
c1.metric("Tracks", f"{tracks_count}")
c2.metric("Attempts", f"{attempts_count}")
c3.metric("Personal Best", f"{best_lap_s:.2f}s" if np.isfinite(best_lap_s) else "—")

st.markdown('<p class="section-title">Workflow Actions</p>', unsafe_allow_html=True)
w1, w2, w3 = st.columns(3, gap="small")
with w1:
    st.markdown(
        """
<div class="step-item">
  <div class="step-num">STEP 1</div>
  <div class="step-name">Upload Attempt</div>
  <div class="step-copy">Import a FIT file and process it under a specific track profile.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Open Upload Workspace", key="goto_upload", use_container_width=True, type="primary"):
        _go(UPLOAD_PAGE)
with w2:
    st.markdown(
        """
<div class="step-item">
  <div class="step-num">STEP 2</div>
  <div class="step-name">Attempt Analysis</div>
  <div class="step-copy">Review one attempt with overview and advanced curve analytics.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Open Attempt Analysis", key="goto_attempt", use_container_width=True):
        _go(ATTEMPT_PAGE)
with w3:
    st.markdown(
        """
<div class="step-item">
  <div class="step-num">STEP 3</div>
  <div class="step-name">Track History</div>
  <div class="step-copy">Compare progression over sessions and identify coaching priorities.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Open Track History", key="goto_history", use_container_width=True):
        _go(HISTORY_PAGE)

with st.expander("Advanced Single-Session View"):
    st.caption("Standalone advanced diagnostics page from current session context.")
    if st.button("Open Advanced Session Page", use_container_width=True, key="goto_further"):
        _go(FURTHER_PAGE)
