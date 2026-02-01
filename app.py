import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd


from lap_methods import get_methods, default_params, params_to_dict
from lap_methods.ui import render_params_form
from lap_methods.quality import quality_summary_gps_gate
from viz.plots import make_plots


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

    #PLOT using the plots module
    track_view = st.sidebar.radio(
        "Track view",
        ["Cartesian (fast)", "Map background"],
        index=0,
        key="track_view"
    )

    if show_plots:
        st.subheader("Plots")

        fig_track, fig_dist = make_plots(result, params, track_view=track_view)

        st.plotly_chart(fig_track, use_container_width=True)
        st.plotly_chart(fig_dist, use_container_width=True)


    st.subheader("Quality summary (no ground truth)")

    # Only compute for this method for now
    # If you add more methods, we can route to the right summary function via registry.
    try:
        q = quality_summary_gps_gate(result, params)
    except Exception as e:
        st.warning(f"Quality summary failed: {e}")
        q = None

    if q:
        # Top KPIs
        a, b, c, d = st.columns(4)
        a.metric("Laps detected", q.get("laps_detected"))
        b.metric("Lap time CV", f"{q.get('lap_time_cv'):.3f}" if np.isfinite(q.get("lap_time_cv", np.nan)) else "—")
        c.metric("Shape corr (mean)", f"{q.get('shape_corr_mean'):.3f}" if np.isfinite(q.get("shape_corr_mean", np.nan)) else "—")
        d.metric("Boundary spread (m)", f"{q.get('boundary_spread_m'):.1f}" if np.isfinite(q.get("boundary_spread_m", np.nan)) else "—")

        # Detail table
        dfq = pd.DataFrame([q]).T.reset_index()
        dfq.columns = ["metric", "value"]
        st.dataframe(dfq, use_container_width=True)

        # Friendly interpretation
        st.caption(
            "Interpretation: lower lap-time CV, lower boundary spread, and lower stability shifts are better. "
            "Higher shape correlation and higher fast-fraction are better."
        )

