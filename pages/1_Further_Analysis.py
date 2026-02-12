from __future__ import annotations

import plotly.express as px
import streamlit as st

from analysis import build_further_analysis


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
    st.warning("No analysis found in this session. Run the main analysis first.")
    if st.button("Back to main page", type="primary"):
        _go_to_main_page()
    st.stop()

method_name = context.get("method_name", "Unknown method")
params = context.get("params", {})

st.caption(f"Source method: {method_name} • Params: {params}")

analysis = build_further_analysis(context)
lap_overview = analysis.lap_overview
curve_overview = analysis.curve_overview

if lap_overview.empty:
    st.warning("No lap metrics available. Upload a FIT file with valid detected laps on the main page.")
else:
    a, b, c = st.columns(3)
    a.metric("Laps analyzed", f"{len(lap_overview)}")

    if "lap_time_s" in lap_overview.columns and lap_overview["lap_time_s"].notna().any():
        b.metric("Best lap (s)", f"{lap_overview['lap_time_s'].min():.2f}")
    else:
        b.metric("Best lap (s)", "—")

    if "avg_speed_kmh" in lap_overview.columns and lap_overview["avg_speed_kmh"].notna().any():
        c.metric("Avg lap speed (km/h)", f"{lap_overview['avg_speed_kmh'].mean():.2f}")
    else:
        c.metric("Avg lap speed (km/h)", "—")

    st.subheader("Lap-by-lap speed profile")
    if "lap" in lap_overview.columns and "avg_speed_kmh" in lap_overview.columns:
        fig = px.line(
            lap_overview,
            x="lap",
            y="avg_speed_kmh",
            markers=True,
            title="Average speed per lap",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Average speed columns are not available in the current lap metrics.")

    st.subheader("Lap overview")
    st.dataframe(lap_overview, use_container_width=True)

st.subheader("Curve analysis")
st.caption("This section is scaffolded for curve entry/apex/exit speed and time-loss analytics.")
if curve_overview.empty:
    st.info("Curve extraction is not implemented yet. Schema is ready to plug in curve detection outputs.")
else:
    st.dataframe(curve_overview, use_container_width=True)

st.subheader("Recommendations")
for i, rec in enumerate(analysis.recommendations, start=1):
    st.write(f"{i}. {rec}")

if st.button("Back to main page"):
    _go_to_main_page()
