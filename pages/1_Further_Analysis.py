from __future__ import annotations

import numpy as np
import streamlit as st

from analysis import build_further_analysis, generate_insight_summary
from analysis.further_plots import (
    plot_curve_apex_variability,
    plot_curve_geometry,
    plot_curve_time_loss,
    plot_curvature_profile,
    plot_lap_speed_profile,
    plot_speed_distribution,
    plot_track_inliers,
    plot_track_limits_and_best_path,
)


MAIN_PAGE = "app.py"


def _go_to_main_page() -> None:
    try:
        st.switch_page(MAIN_PAGE)
    except Exception:
        st.info("Open the main page from the sidebar pages menu.")


def _fmt_num(value: float, fmt: str) -> str:
    return fmt.format(value) if np.isfinite(value) else "—"


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

for msg in analysis.warnings:
    st.warning(msg)

st.subheader("Session KPIs")
a, b, c, d, e, f = st.columns(6)
a.metric("Laps detected", f"{analysis.laps_detected}")
b.metric("Valid laps", f"{analysis.valid_laps}")
c.metric("Curves detected", f"{analysis.curves_detected}")
d.metric("Boundary curves", f"{analysis.boundary_curves}")
e.metric("Speed threshold", _fmt_num(analysis.speed_threshold_kmh, "{:.2f} km/h"))
f.metric("Ref lap", f"{analysis.reference_lap_id}" if analysis.reference_lap_id is not None else "—")

g, h = st.columns(2)
g.metric("Median track width", _fmt_num(analysis.median_track_width_m, "{:.2f} m"))
h.metric("Track width p90", _fmt_num(analysis.p90_track_width_m, "{:.2f} m"))

lap_overview = analysis.lap_overview
curve_overview = analysis.curve_overview
curve_summary = analysis.curve_summary
reference_path = analysis.reference_path
samples_tagged = analysis.samples_tagged

st.subheader("Lap Performance")
if lap_overview.empty:
    st.warning("No lap metrics available. Upload a FIT file with valid detected laps on the main page.")
else:
    if {"lap", "avg_speed_kmh"}.issubset(lap_overview.columns):
        st.plotly_chart(plot_lap_speed_profile(lap_overview), use_container_width=True)
    st.dataframe(lap_overview, use_container_width=True)

st.subheader("Speed Inlier Detection")
if samples_tagged.empty:
    st.info("No samples available for speed-based inlier analysis.")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_speed_distribution(samples_tagged, analysis.speed_threshold_kmh),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(plot_track_inliers(samples_tagged), use_container_width=True)

st.subheader("Track Limits And Best Path")
if reference_path.empty:
    st.info("Reference path could not be estimated for this session.")
else:
    st.plotly_chart(
        plot_track_limits_and_best_path(reference_path, samples_tagged),
        use_container_width=True,
    )

st.subheader("Curve Detection")
if reference_path.empty or curve_overview.empty:
    st.info("Curve geometry is not available for this session.")
else:
    curves_geometry = (
        curve_overview[["curve_id", "s_start_m", "s_end_m", "s_apex_m", "peak_curvature"]]
        .drop_duplicates()
        .sort_values("curve_id")
        .reset_index(drop=True)
    )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            plot_curvature_profile(reference_path, curve_overview),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            plot_curve_geometry(reference_path, curves_geometry),
            use_container_width=True,
        )

    st.dataframe(curves_geometry, use_container_width=True)

st.subheader("Curve Performance")
if curve_summary.empty:
    st.info("No curve performance metrics available.")
else:
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(plot_curve_time_loss(curve_summary), use_container_width=True)
    with c6:
        st.plotly_chart(plot_curve_apex_variability(curve_summary), use_container_width=True)

    st.write("Curve summary")
    st.dataframe(curve_summary, use_container_width=True)

    st.write("Curve detail (lap-by-lap)")
    st.dataframe(curve_overview, use_container_width=True)

st.subheader("Recommendations")
for i, rec in enumerate(analysis.recommendations, start=1):
    st.write(f"{i}. {rec}")

st.subheader("AI Coaching Summary (Architecture Stub)")
insight = generate_insight_summary(analysis)
st.write(f"**{insight.title}**")
for bullet in insight.bullets:
    st.write(f"- {bullet}")
st.write("Opportunities")
for opp in insight.opportunities:
    st.write(f"- {opp}")
st.caption(f"Summary engine: {insight.engine}")
st.info(
    "Architecture is ready for an LLM-backed provider. "
    "Replace the default heuristic provider with `LLMInsightProvider` in `analysis/insights.py` when you connect a model API."
)

if st.button("Back to main page"):
    _go_to_main_page()
