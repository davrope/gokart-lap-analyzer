from __future__ import annotations

import streamlit as st

from analysis import build_further_analysis
from ui import render_advanced_analysis


MAIN_PAGE = "app.py"


def _go_to_main_page() -> None:
    try:
        st.switch_page(MAIN_PAGE)
    except Exception:
        st.info("Open the main page from the sidebar pages menu.")


st.set_page_config(page_title="Further Analysis", layout="wide")
st.title("Further Analysis")

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
