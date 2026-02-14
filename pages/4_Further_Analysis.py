from __future__ import annotations

import streamlit as st

from analysis import build_further_analysis
from services import get_authenticated_user, sign_out_user
from ui import configure_page, render_advanced_analysis, render_page_header, render_top_nav


MAIN_PAGE = "app.py"


def _go_to_main_page() -> None:
    try:
        st.switch_page(MAIN_PAGE)
    except Exception:
        st.info("Use the top navigation buttons to return to Home.")


def _sign_out() -> None:
    sign_out_user()
    try:
        st.switch_page("app.py")
    except Exception:
        st.rerun()


configure_page("Further Analysis")

user = get_authenticated_user()
render_top_nav(
    active_page="attempt",
    user_label=(user.email or user.id) if user else None,
    show_signout=user is not None,
    on_signout=_sign_out if user is not None else None,
)
render_page_header(
    "Further Analysis",
    "Standalone advanced diagnostics for the active session context.",
)

context = st.session_state.get("latest_analysis")
if not context:
    st.warning("No analysis found in this session. Run Attempt Analysis or Upload first.")
    if st.button("Back to main page", type="primary"):
        _go_to_main_page()
    st.stop()

analysis = build_further_analysis(context)
render_advanced_analysis(analysis, key_prefix="standalone_further")

if st.button("Back to main page"):
    _go_to_main_page()
