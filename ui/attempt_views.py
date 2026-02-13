from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from analysis import FurtherAnalysisResult, generate_insight_summary
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
from lap_methods.quality import quality_summary_gps_gate
from viz.plots import animate_track_points, make_plots



def _fmt_num(value: float, fmt: str) -> str:
    return fmt.format(value) if np.isfinite(value) else "—"



def _params_to_dict(params: Any) -> dict[str, Any]:
    if isinstance(params, dict):
        return params
    if is_dataclass(params):
        return asdict(params)
    return {}



def render_overview_analysis(
    result: dict[str, Any],
    method_name: str,
    params: Any,
    *,
    key_prefix: str = "attempt_overview",
) -> None:
    gps = result["gps"]
    pass_idx = result["pass_idx"]
    dt_s = result["dt_s"]
    laps = result["laps"]
    lap_metrics = result["lap_metrics"]
    samples = result["samples"]
    gate_lat = result["gate_lat"]
    gate_lon = result["gate_lon"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GPS points", f"{len(gps)}")
    c2.metric("Sampling interval (approx)", f"{dt_s:.2f}s")
    c3.metric("Detected passes", f"{len(pass_idx)}")
    c4.metric("Detected laps", f"{len(laps)}")

    st.write(f"**Gate center:** lat `{gate_lat:.6f}`, lon `{gate_lon:.6f}`")
    st.caption(f"Method: {method_name} • Params: {_params_to_dict(params)}")

    if laps.empty:
        st.warning("No laps detected. Try adjusting method parameters and reprocessing this attempt.")
        return

    st.subheader("Lap metrics")
    st.dataframe(lap_metrics.sort_values("lap"), use_container_width=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            "Download laps_gps.csv",
            data=lap_metrics.to_csv(index=False).encode("utf-8"),
            file_name="laps_gps.csv",
            mime="text/csv",
            key=f"{key_prefix}.dl_laps",
        )
    with dc2:
        st.download_button(
            "Download samples_with_laps_gps.csv",
            data=samples.to_csv(index=False).encode("utf-8"),
            file_name="samples_with_laps_gps.csv",
            mime="text/csv",
            key=f"{key_prefix}.dl_samples",
        )

    view_col, animate_col, show_col = st.columns(3)
    with view_col:
        track_view = st.radio(
            "Track view",
            ["Cartesian (fast)", "Map background"],
            index=0,
            key=f"{key_prefix}.track_view",
            horizontal=True,
        )
    with animate_col:
        animate = st.checkbox("Animate GPS points", value=False, key=f"{key_prefix}.animate")
    with show_col:
        show_plots = st.checkbox("Show plots", value=True, key=f"{key_prefix}.show_plots")

    if show_plots:
        st.subheader("Plots")
        fig_track, fig_dist, fig_timeline, fig_step, fig_distdomain = make_plots(result, params, track_view=track_view)
        st.plotly_chart(fig_track, use_container_width=True)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.plotly_chart(fig_timeline, use_container_width=True)
        st.plotly_chart(fig_step, use_container_width=True)
        st.plotly_chart(fig_distdomain, use_container_width=True)

    st.subheader("Quality summary (no ground truth)")
    try:
        q = quality_summary_gps_gate(result, params)
    except Exception as exc:
        st.warning(f"Quality summary failed: {exc}")
        q = None

    if q:
        a, b, c, d = st.columns(4)
        a.metric("Laps detected", q.get("laps_detected"))
        b.metric("Lap time CV", f"{q.get('lap_time_cv'):.3f}" if np.isfinite(q.get("lap_time_cv", np.nan)) else "—")
        c.metric("Shape corr (mean)", f"{q.get('shape_corr_mean'):.3f}" if np.isfinite(q.get("shape_corr_mean", np.nan)) else "—")
        d.metric("Boundary spread (m)", f"{q.get('boundary_spread_m'):.1f}" if np.isfinite(q.get("boundary_spread_m", np.nan)) else "—")

        e, f = st.columns(2)
        e.metric("Lap dist CV", f"{q.get('lap_dist_cv'):.3f}" if np.isfinite(q.get("lap_dist_cv", np.nan)) else "—")
        f.metric("Lap dist MAD (m)", f"{q.get('lap_dist_mad_m'):.1f}" if np.isfinite(q.get("lap_dist_mad_m", np.nan)) else "—")

        dfq = pd.DataFrame([q]).T.reset_index()
        dfq.columns = ["metric", "value"]
        st.dataframe(dfq, use_container_width=True)

    if animate:
        fig_anim = animate_track_points(gps, track_view=track_view, fps=20, max_frames=450, tail_points=250)
        st.plotly_chart(fig_anim, use_container_width=True, key=f"{key_prefix}.fig_anim")



def render_advanced_analysis(analysis: FurtherAnalysisResult, *, key_prefix: str = "attempt_advanced") -> None:
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
        st.warning("No lap metrics available for advanced analysis.")
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
            st.plotly_chart(plot_speed_distribution(samples_tagged, analysis.speed_threshold_kmh), use_container_width=True)
        with c2:
            st.plotly_chart(plot_track_inliers(samples_tagged), use_container_width=True)

    st.subheader("Track Limits And Best Path")
    if reference_path.empty:
        st.info("Reference path could not be estimated for this attempt.")
    else:
        st.plotly_chart(plot_track_limits_and_best_path(reference_path, samples_tagged), use_container_width=True)

    st.subheader("Curve Detection")
    if reference_path.empty or curve_overview.empty:
        st.info("Curve geometry is not available for this attempt.")
    else:
        curves_geometry = (
            curve_overview[["curve_id", "s_start_m", "s_end_m", "s_apex_m", "peak_curvature"]]
            .drop_duplicates()
            .sort_values("curve_id")
            .reset_index(drop=True)
        )

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_curvature_profile(reference_path, curve_overview), use_container_width=True)
        with c4:
            st.plotly_chart(plot_curve_geometry(reference_path, curves_geometry), use_container_width=True)

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
