import streamlit as st
import numpy as np
import plotly.graph_objects as go


from lap_methods import get_methods, default_params, params_to_dict
from lap_methods.ui import render_params_form


st.set_page_config(page_title="FIT Lap Analyzer", layout="wide")
st.title("FIT Lap Analyzer")

methods = get_methods()
method_name = st.sidebar.selectbox(
    "Lap detection method",
    list(methods.keys()),
    key="method_selector"
)

params = default_params(method_name)

params = render_params_form(params, title="Method parameters", key_prefix=method_name)
show_plots = st.sidebar.checkbox("Show plots", value=True, key=f"{method_name}.show_plots")


uploaded = st.file_uploader("Upload a Garmin .FIT file", type=["fit"])
if uploaded is None:
    st.info("Upload a .fit file to begin.")
    st.stop()

fit_bytes = uploaded.getvalue()

runner = methods[method_name]["runner"]

try:
    result = runner(fit_bytes, params)
except Exception as e:
    st.error(f"Failed to analyze file: {e}")
    st.stop()

gps = result["gps"]
fast_pts = result["fast_pts"]
gate_lat = result["gate_lat"]
gate_lon = result["gate_lon"]
dist = result["dist"]
pass_idx = result["pass_idx"]
dt_s = result["dt_s"]
laps = result["laps"]
lap_metrics = result["lap_metrics"]
samples = result["samples"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("GPS points", f"{len(gps)}")
c2.metric("Sampling interval (approx)", f"{dt_s:.2f}s")
c3.metric("Detected passes", f"{len(pass_idx)}")
c4.metric("Detected laps", f"{len(laps)}")

st.write(f"**Gate center:** lat `{gate_lat:.6f}`, lon `{gate_lon:.6f}`")
st.caption(f"Method: {method_name} • Params: {params_to_dict(params)}")

if laps.empty:
    st.warning("No laps detected. Try increasing 'Gate proximity' or lowering 'Min lap time'.")
else:
    st.subheader("Lap metrics")
    st.dataframe(lap_metrics.sort_values("lap"), use_container_width=True)

    st.download_button(
        "Download laps_gps.csv",
        data=lap_metrics.to_csv(index=False).encode("utf-8"),
        file_name="laps_gps.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download samples_with_laps_gps.csv",
        data=samples.to_csv(index=False).encode("utf-8"),
        file_name="samples_with_laps_gps.csv",
        mime="text/csv",
    )
if show_plots:
    track_view = st.sidebar.radio(
    "Track view",
    ["Cartesian (fast)", "Map background"],
    index=0,
    key=f"{method_name}.track_view"
    )

    st.subheader("Interactive plots")

    if track_view == "Map background":

        # Compute padded bounds
        lat_min, lat_max = gps["lat"].min(), gps["lat"].max()
        lon_min, lon_max = gps["lon"].min(), gps["lon"].max()

        # Padding as a fraction of the total span
        PAD_FRAC = 0.15  # 15% padding looks good for tracks

        lat_pad = (lat_max - lat_min) * PAD_FRAC
        lon_pad = (lon_max - lon_min) * PAD_FRAC

        bounds = dict(
            west=float(lon_min - lon_pad),
            east=float(lon_max + lon_pad),
            south=float(lat_min - lat_pad),
            north=float(lat_max + lat_pad),
        )

        # MAP VERSION (Scattermapbox)
        fig_track = go.Figure()

        fig_track.add_trace(go.Scattermapbox(
            lat=gps["lat"], lon=gps["lon"],
            mode="lines",
            name="Track",
            line=dict(width=3),
            hoverinfo="skip",
        ))

        fig_track.add_trace(go.Scattermapbox(
            lat=fast_pts["lat"], lon=fast_pts["lon"],
            mode="markers",
            name=f"Speed ≥ {params.speed_for_gate} km/h (gate search)",
            marker=dict(size=6),
            hovertemplate="Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>",
        ))

        fig_track.add_trace(go.Scattermapbox(
            lat=[gate_lat], lon=[gate_lon],
            mode="markers",
            name="Gate center",
            marker=dict(size=14, color="orange"),
            hovertemplate="Gate<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>",
        ))

        if len(pass_idx) > 0:
            fig_track.add_trace(go.Scattermapbox(
                lat=gps.loc[pass_idx, "lat"],
                lon=gps.loc[pass_idx, "lon"],
                mode="markers",
                name="Detected passes",
                marker=dict(size=10, color="red"),
                hovertemplate="PASS<br>%{lat:.6f}, %{lon:.6f}<extra></extra>",
            ))

        center_lat = float(gps["lat"].median())
        center_lon = float(gps["lon"].median())

        fig_track.update_layout(
            title="GPS Track (map background)",
            mapbox=dict(
                style="open-street-map",  # no token needed
                center=dict(lat=center_lat, lon=center_lon),
                zoom=16,
            ),
            height=600,
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="h"),
        )

        try:
            fig_track.update_layout(
                mapbox=dict(
                    style="open-street-map",
                ),
                mapbox_bounds=bounds,
                height=600,
                margin=dict(l=10, r=10, t=60, b=10),
                legend=dict(orientation="h"),
            )

        except Exception:
            pass

    else:
        # CARTESIAN VERSION (lon vs lat)
        fig_track = go.Figure()

        fig_track.add_trace(go.Scatter(
            x=gps["lon"], y=gps["lat"],
            mode="lines",
            name="All points",
            line=dict(width=2),
            hoverinfo="skip",
        ))

        fig_track.add_trace(go.Scatter(
            x=fast_pts["lon"], y=fast_pts["lat"],
            mode="markers",
            name=f"Speed ≥ {params.speed_for_gate} km/h (gate search)",
            marker=dict(size=5),
            hovertemplate="Lon: %{x:.6f}<br>Lat: %{y:.6f}<extra></extra>",
        ))

        fig_track.add_trace(go.Scatter(
            x=[gate_lon], y=[gate_lat],
            mode="markers",
            name="Gate center",
            marker=dict(size=14, color="orange", symbol="star"),
            hovertemplate="Gate<br>Lon: %{x:.6f}<br>Lat: %{y:.6f}<extra></extra>",
        ))

        if len(pass_idx) > 0:
            fig_track.add_trace(go.Scatter(
                x=gps.loc[pass_idx, "lon"],
                y=gps.loc[pass_idx, "lat"],
                mode="markers",
                name="Detected passes",
                marker=dict(size=8, color="red"),
                hovertemplate="PASS<br>Lon: %{x:.6f}<br>Lat: %{y:.6f}<extra></extra>",
            ))

        fig_track.update_layout(
            title="GPS Track (Cartesian)",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            yaxis_scaleanchor="x",  # keep aspect ratio
            height=550,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h"),
        )

    st.plotly_chart(fig_track, use_container_width=True)


    st.subheader("Distance to gate over time")

    fig_dist = go.Figure()

    # Main distance line
    fig_dist.add_trace(go.Scatter(
        x=gps["timestamp"],
        y=dist,
        mode="lines",
        name="Distance to gate",
        line=dict(width=2),
        hovertemplate="Time: %{x}<br>Distance: %{y:.1f} m<extra></extra>",
    ))

    # Detected passes (minima)
    if len(pass_idx) > 0:
        fig_dist.add_trace(go.Scatter(
            x=gps.loc[pass_idx, "timestamp"],
            y=dist[pass_idx],
            mode="markers",
            name="Detected passes",
            marker=dict(size=9, color="orange"),
            hovertemplate="PASS<br>%{x}<br>%{y:.1f} m<extra></extra>",
        ))

    # Threshold line (near_m)
    fig_dist.add_hline(
        y=float(params.near_m),
        line_dash="dash",
        annotation_text=f"near_m = {params.near_m} m",
        annotation_position="top right",
    )

    fig_dist.update_layout(
        title="Distance to gate over time (passes = minima)",
        xaxis_title="Time",
        yaxis_title="Distance (meters)",
        height=320,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig_dist, use_container_width=True)


#These plots are from the previous person using pyplot.

# if show_plots:
#     import matplotlib.pyplot as plt

#     st.subheader("Plots")

#     fig1 = plt.figure(figsize=(7, 6))
#     plt.plot(gps["lon"], gps["lat"], linewidth=1, label="All points")
#     plt.scatter(fast_pts["lon"], fast_pts["lat"], s=6, label=f"Speed ≥ {params.speed_for_gate} km/h (gate search)")
#     plt.scatter([gate_lon], [gate_lat], s=140, color="orange", label="Gate center")
#     plt.xlabel("Longitude"); plt.ylabel("Latitude")
#     plt.title("Track + Gate")
#     plt.axis("equal")
#     plt.legend()
#     st.pyplot(fig1)

#     fig2 = plt.figure(figsize=(12, 3))
#     plt.plot(gps["timestamp"], dist, linewidth=1)
#     if len(pass_idx) > 0:
#         plt.scatter(gps.loc[pass_idx, "timestamp"], dist[pass_idx], s=25, color="orange", label="Detected passes")
#     plt.axhline(float(params.near_m), linestyle="--", label=f"near_m={params.near_m}m")
#     plt.title("Distance to gate over time (passes = minima)")
#     plt.xticks(rotation=25)
#     plt.tight_layout()
#     plt.legend()
#     st.pyplot(fig2)
