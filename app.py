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


st.set_page_config(page_title="GoKart Coaching Platform", layout="wide")

cfg = get_app_config()
if not cfg.multi_attempt_mode:
    st.title("GoKart Coaching Platform")
    st.info("`MULTI_ATTEMPT_MODE` is disabled. Enable it in Streamlit secrets or environment to use this workflow.")
    st.stop()

st.title("GoKart Coaching Platform")

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
<div style="padding:1rem 1rem 0.5rem 1rem;border:1px solid #e5e7eb;border-radius:16px;background:linear-gradient(140deg,#f8fafc,#eef2ff);">
  <h3 style="margin-top:0;">Analyze, Compare, Improve</h3>
  <p style="margin-bottom:0.6rem;">Upload multiple Garmin FIT attempts per track, visualize lap and curve behavior over time, and track performance improvements with structured coaching insights.</p>
</div>
""",
    unsafe_allow_html=True,
)

user = get_authenticated_user()

if user is None:
    st.subheader("Login")
    st.caption("Passwordless login with Supabase Magic Link")

    with st.form("login_form"):
        email = st.text_input("Email", placeholder="driver@example.com")
        submitted = st.form_submit_button("Send magic link", type="primary")

    if submitted:
        try:
            get_auth_service().send_magic_link(email)
            st.success("Magic link sent. Check your email and return to this page.")
        except Exception as exc:
            st.error(f"Failed to send magic link: {exc}")

    st.stop()

st.success(f"Logged in as {user.email or user.id}")

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
c3.metric("Personal best lap", f"{best_lap_s:.2f}s" if np.isfinite(best_lap_s) else "—")

st.subheader("Workflow")

w1, w2, w3 = st.columns(3)
with w1:
    st.markdown("**1) Upload Attempt**")
    st.caption("Upload FIT file and attach it to a track.")
    if st.button("Go to Upload", key="goto_upload", use_container_width=True, type="primary"):
        _go(UPLOAD_PAGE)
with w2:
    st.markdown("**2) Attempt Analysis**")
    st.caption("Inspect one attempt with overview + advanced telemetry.")
    if st.button("Go to Attempt Analysis", key="goto_attempt", use_container_width=True):
        _go(ATTEMPT_PAGE)
with w3:
    st.markdown("**3) Track History**")
    st.caption("Compare attempts and monitor long-term improvements.")
    if st.button("Go to Track History", key="goto_history", use_container_width=True):
        _go(HISTORY_PAGE)

with st.expander("Advanced (single-session deep dive)"):
    st.caption("Standalone advanced page from current session context.")
    if st.button("Open Advanced Session Page", use_container_width=True, key="goto_further"):
        _go(FURTHER_PAGE)
