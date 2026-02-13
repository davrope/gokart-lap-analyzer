from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = {
    "accent": "#0B6E4F",
    "accent_2": "#CC3F0C",
    "inlier": "#2A9D8F",
    "outlier": "#9CA3AF",
    "left": "#1D4ED8",
    "right": "#1D4ED8",
    "center": "#111827",
    "best": "#F97316",
    "curve": "#7C3AED",
}


def _style(fig: go.Figure, *, title: str, height: int = 420, x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin=dict(l=16, r=16, t=56, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def plot_lap_speed_profile(lap_overview: pd.DataFrame) -> go.Figure:
    fig = px.line(
        lap_overview,
        x="lap",
        y="avg_speed_kmh",
        markers=True,
        line_shape="linear",
        color_discrete_sequence=[PALETTE["accent"]],
    )
    return _style(fig, title="Average Speed Per Lap", x_title="Lap", y_title="Speed (km/h)")


def plot_speed_distribution(samples_tagged: pd.DataFrame, speed_threshold_kmh: float) -> go.Figure:
    fig = px.histogram(
        samples_tagged,
        x="speed_kmh",
        nbins=40,
        color_discrete_sequence=[PALETTE["accent"]],
        opacity=0.85,
    )
    fig.add_vline(
        x=speed_threshold_kmh,
        line_width=2,
        line_dash="dash",
        line_color=PALETTE["accent_2"],
        annotation_text=f"Threshold {speed_threshold_kmh:.2f} km/h",
        annotation_position="top right",
    )
    return _style(fig, title="Speed Distribution And Inlier Threshold", x_title="Speed (km/h)", y_title="Points")


def plot_track_inliers(samples_tagged: pd.DataFrame) -> go.Figure:
    df = samples_tagged.copy()
    df["label"] = np.where(df["speed_inlier"], "Track-speed inliers", "Speed outliers")

    fig = px.scatter(
        df,
        x="lon",
        y="lat",
        color="label",
        color_discrete_map={
            "Track-speed inliers": PALETTE["inlier"],
            "Speed outliers": PALETTE["outlier"],
        },
        opacity=0.72,
        hover_data=["speed_kmh", "lap"],
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _style(fig, title="Track Inliers vs Outliers (Speed-First)", x_title="Longitude", y_title="Latitude", height=560)


def plot_track_limits_and_best_path(reference_path: pd.DataFrame, samples_tagged: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    outliers = samples_tagged[~samples_tagged["speed_inlier"]]
    inliers = samples_tagged[samples_tagged["speed_inlier"]]

    if not outliers.empty:
        fig.add_trace(
            go.Scattergl(
                x=outliers["lon"],
                y=outliers["lat"],
                mode="markers",
                name="Speed outliers",
                marker=dict(size=5, color=PALETTE["outlier"], opacity=0.35),
            )
        )

    if not inliers.empty:
        fig.add_trace(
            go.Scattergl(
                x=inliers["lon"],
                y=inliers["lat"],
                mode="markers",
                name="Track-speed inliers",
                marker=dict(size=5, color=PALETTE["inlier"], opacity=0.40),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=reference_path["lon_right"],
            y=reference_path["lat_right"],
            mode="lines",
            name="Estimated right limit",
            line=dict(width=2, color=PALETTE["right"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reference_path["lon_left"],
            y=reference_path["lat_left"],
            mode="lines",
            name="Estimated left limit",
            line=dict(width=2, color=PALETTE["left"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reference_path["lon_center"],
            y=reference_path["lat_center"],
            mode="lines",
            name="Reference centerline",
            line=dict(width=2, color=PALETTE["center"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reference_path["lon_best"],
            y=reference_path["lat_best"],
            mode="lines",
            name="Estimated best path",
            line=dict(width=3, color=PALETTE["best"]),
        )
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _style(fig, title="Track Limits And Estimated Best Path", x_title="Longitude", y_title="Latitude", height=620)


def plot_curvature_profile(reference_path: pd.DataFrame, curve_overview: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reference_path["s_m"],
            y=reference_path["curvature"],
            mode="lines",
            name="Curvature",
            line=dict(width=2, color=PALETTE["center"]),
        )
    )

    if not curve_overview.empty:
        curve_apex = (
            curve_overview.groupby("curve_id", as_index=False)
            .agg(s_apex_m=("s_apex_m", "median"), peak_curvature=("peak_curvature", "median"))
            if {"s_apex_m", "peak_curvature"}.issubset(curve_overview.columns)
            else pd.DataFrame()
        )
        if not curve_apex.empty:
            fig.add_trace(
                go.Scatter(
                    x=curve_apex["s_apex_m"],
                    y=curve_apex["peak_curvature"],
                    mode="markers+text",
                    text=curve_apex["curve_id"].astype(str),
                    textposition="top center",
                    name="Detected curve apex",
                    marker=dict(size=9, color=PALETTE["accent_2"]),
                )
            )

    return _style(fig, title="Curvature Along Reference Lap", x_title="Distance along lap (m)", y_title="Curvature")


def plot_curve_geometry(reference_path: pd.DataFrame, curves_geometry: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reference_path["x_center_m"],
            y=reference_path["y_center_m"],
            mode="lines",
            name="Reference centerline",
            line=dict(width=1.2, color="#6B7280"),
        )
    )

    for _, row in curves_geometry.iterrows():
        s0 = float(row["s_start_m"])
        s1 = float(row["s_end_m"])
        cid = int(row["curve_id"])

        if s1 >= s0:
            m = (reference_path["s_m"] >= s0) & (reference_path["s_m"] <= s1)
        else:
            m = (reference_path["s_m"] >= s0) | (reference_path["s_m"] <= s1)

        seg = reference_path[m]
        if seg.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=seg["x_center_m"],
                y=seg["y_center_m"],
                mode="lines",
                name=f"Curve {cid}",
                line=dict(width=3, color=PALETTE["curve"]),
                showlegend=False,
            )
        )

        j = int(np.argmin(np.abs(reference_path["s_m"].to_numpy() - float(row["s_apex_m"]))))
        fig.add_trace(
            go.Scatter(
                x=[float(reference_path.iloc[j]["x_center_m"])],
                y=[float(reference_path.iloc[j]["y_center_m"])],
                mode="markers+text",
                text=[str(cid)],
                textposition="top center",
                marker=dict(size=8, color=PALETTE["accent_2"]),
                name=f"Apex {cid}",
                showlegend=False,
            )
        )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _style(fig, title="Detected Curves On Closed Loop", x_title="Local X (m)", y_title="Local Y (m)", height=560)


def plot_curve_time_loss(curve_summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        curve_summary,
        x=curve_summary["curve_id"].astype(str),
        y="mean_time_loss_s",
        color_discrete_sequence=[PALETTE["accent"]],
    )
    fig.add_hline(y=0.0, line_color="#374151", line_width=1)
    return _style(fig, title="Mean Time Loss vs Best Lap By Curve", x_title="Curve", y_title="Time loss (s)")


def plot_curve_apex_variability(curve_summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        curve_summary,
        x=curve_summary["curve_id"].astype(str),
        y="apex_speed_std_kmh",
        color_discrete_sequence=[PALETTE["accent_2"]],
    )
    return _style(fig, title="Apex Speed Variability By Curve", x_title="Curve", y_title="Std apex speed (km/h)")
