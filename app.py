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


UPLOAD_PAGE = "pages/1_Upload_Attempt.py"
ATTEMPT_PAGE = "pages/2_Attempt_Analysis.py"
HISTORY_PAGE = "pages/3_Track_History.py"
FURTHER_PAGE = "pages/4_Further_Analysis.py"


def _go(page: str) -> None:
    try:
        st.switch_page(page)
    except Exception:
        st.info("Use the sidebar pages menu to navigate.")


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


st.set_page_config(page_title="GoKart Coaching Platform", layout="wide")

cfg = get_app_config()
if not cfg.multi_attempt_mode:
    st.title("GoKart Coaching Platform")
    st.info("`MULTI_ATTEMPT_MODE` is disabled. Enable it in Streamlit secrets or environment to use this workflow.")
    st.stop()

status_msg = bootstrap_auth_session_from_query()
if status_msg:
    st.success(status_msg)

if not supabase_configured(cfg):
    st.error(
        "Supabase is not configured. Add `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `APP_BASE_URL` in Streamlit secrets."
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

st.markdown(
    """
<style>
.hero-wrap {padding: 1.4rem; border-radius: 18px; background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%); border: 1px solid #dbeafe;}
.eyebrow {letter-spacing: .08em; text-transform: uppercase; font-size: .74rem; color: #1d4ed8; font-weight: 700;}
.hero-title {font-size: 2.1rem; line-height: 1.15; margin: .4rem 0 .7rem 0; color: #0f172a; font-weight: 800;}
.hero-copy {font-size: 1.03rem; color: #334155; margin-bottom: 0;}
.card {padding: 1rem; border: 1px solid #e5e7eb; border-radius: 14px; background: #ffffff;}
.minor {color: #64748b; font-size: .92rem;}
</style>
""",
    unsafe_allow_html=True,
)

user = get_authenticated_user()

if user is None:
    left, right = st.columns([1.6, 1.0], vertical_alignment="top")

    with left:
        st.markdown(
            """
<div class="hero-wrap">
  <div class="eyebrow">GoKart Data Coach</div>
  <div class="hero-title">Turn your FIT telemetry into faster, repeatable lap times.</div>
  <p class="hero-copy">
    Upload attempts, detect laps and curves automatically, compare progression by track,
    and focus training on the corners where you lose the most time.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

        f1, f2, f3 = st.columns(3)
        f1.metric("Telemetry", "GPS + Speed")
        f2.metric("Analysis", "Lap + Curve")
        f3.metric("Progress", "Attempt History")

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sign in")
        st.caption("Use passwordless email or Google via Supabase Auth")

        google_url = ""
        try:
            google_url = get_auth_service().get_google_auth_url()
        except Exception as exc:
            st.warning(f"Google login unavailable: {exc}")

        if google_url:
            _render_google_button(google_url)
            st.caption("If Google auth fails, verify Google provider is enabled in Supabase Auth settings.")

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

        st.markdown(
            """
<p class="minor">
After login, you can upload attempts, review detailed telemetry, and track improvements across sessions.
</p>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### What you get")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Attempt Analysis**")
        st.caption("Inspect one session with lap stats, speed profile, and curve-level behavior.")
    with g2:
        st.markdown("**Historical Trends**")
        st.caption("Track best lap evolution, consistency, and curve improvements over time.")
    with g3:
        st.markdown("**Coaching-Ready Insights**")
        st.caption("Structured summaries designed to plug into an LLM recommendation layer later.")

    st.stop()

st.markdown(
    f"""
<div class="hero-wrap">
  <div class="eyebrow">Welcome back</div>
  <div class="hero-title" style="font-size:1.5rem;">{user.email or user.id}</div>
  <p class="hero-copy">Choose a workflow step to upload data, inspect attempts, or compare long-term progression.</p>
</div>
""",
    unsafe_allow_html=True,
)

row1, row2 = st.columns([3, 1])
with row2:
    if st.button("Sign out", use_container_width=True):
        sign_out_user()
        st.rerun()

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

st.subheader("Workflow")
w1, w2, w3 = st.columns(3)
with w1:
    st.markdown("**1) Upload Attempt**")
    st.caption("Upload FIT and attach to an existing or new track.")
    if st.button("Open Upload", key="goto_upload", use_container_width=True, type="primary"):
        _go(UPLOAD_PAGE)
with w2:
    st.markdown("**2) Attempt Analysis**")
    st.caption("Review one attempt with overview and advanced telemetry.")
    if st.button("Open Attempt Analysis", key="goto_attempt", use_container_width=True):
        _go(ATTEMPT_PAGE)
with w3:
    st.markdown("**3) Track History**")
    st.caption("Compare attempts and identify where time is still being lost.")
    if st.button("Open Track History", key="goto_history", use_container_width=True):
        _go(HISTORY_PAGE)

with st.expander("Advanced Single-Session Page"):
    st.caption("Standalone advanced view from the current session context.")
    if st.button("Open Advanced Session Page", use_container_width=True, key="goto_further"):
        _go(FURTHER_PAGE)
