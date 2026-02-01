# viz/plots.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _safe_get(result: dict, key: str, default=None):
    return result[key] if isinstance(result, dict) and key in result else default


def _is_finite_number(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _padded_bounds(lat: pd.Series, lon: pd.Series, pad_frac: float = 0.15):
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())

    lat_span = max(1e-12, lat_max - lat_min)
    lon_span = max(1e-12, lon_max - lon_min)

    lat_pad = lat_span * pad_frac
    lon_pad = lon_span * pad_frac

    return dict(
        west=lon_min - lon_pad,
        east=lon_max + lon_pad,
        south=lat_min - lat_pad,
        north=lat_max + lat_pad,
    )


def plot_track(
    gps: pd.DataFrame,
    *,
    track_view: str = "Cartesian (fast)",
    fast_pts: pd.DataFrame | None = None,
    gate_lat: float | None = None,
    gate_lon: float | None = None,
    pass_idx: np.ndarray | list[int] | None = None,
    speed_for_gate: float | None = None,
) -> go.Figure:
    """
    Interactive track plot. If map view, draws on OpenStreetMap tiles.
    Works even if fast_pts/gate/pass_idx are missing.
    """
    fast_pts = fast_pts if isinstance(fast_pts, pd.DataFrame) else None
    pass_idx = np.array(pass_idx, dtype=int) if pass_idx is not None else np.array([], dtype=int)

    have_fast = fast_pts is not None and len(fast_pts) > 0 and {"lat", "lon"}.issubset(fast_pts.columns)
    have_gate = _is_finite_number(gate_lat) and _is_finite_number(gate_lon)
    have_pass = len(pass_idx) > 0 and {"lat", "lon"}.issubset(gps.columns)

    if track_view == "Map background":
        fig = go.Figure()

        fig.add_trace(go.Scattermapbox(
            lat=gps["lat"], lon=gps["lon"],
            mode="lines",
            name="Track",
            line=dict(width=3),
            hoverinfo="skip",
        ))

        if have_fast:
            label = f"Speed ≥ {speed_for_gate} km/h (gate search)" if speed_for_gate is not None else "Fast points"
            fig.add_trace(go.Scattermapbox(
                lat=fast_pts["lat"], lon=fast_pts["lon"],
                mode="markers",
                name=label,
                marker=dict(size=6),
                hovertemplate="Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>",
            ))

        if have_gate:
            fig.add_trace(go.Scattermapbox(
                lat=[float(gate_lat)], lon=[float(gate_lon)],
                mode="markers",
                name="Gate center",
                marker=dict(size=14, color="orange"),
                hovertemplate="Gate<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>",
            ))

        if have_pass:
            fig.add_trace(go.Scattermapbox(
                lat=gps.iloc[pass_idx]["lat"],
                lon=gps.iloc[pass_idx]["lon"],
                mode="markers",
                name="Detected passes",
                marker=dict(size=10, color="red"),
                hovertemplate="PASS<br>%{lat:.6f}, %{lon:.6f}<extra></extra>",
            ))

        bounds = _padded_bounds(gps["lat"], gps["lon"], pad_frac=0.15)

        fig.update_layout(
            title="GPS Track (map background)",
            mapbox=dict(style="open-street-map"),
            height=600,
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="h"),
        )

        # Some Plotly versions support bounds; if not, fall back to center/zoom.
        try:
            fig.update_layout(mapbox_bounds=bounds)
        except Exception:
            center_lat = float(gps["lat"].median())
            center_lon = float(gps["lon"].median())
            fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=16))

        return fig

    # Cartesian view
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gps["lon"], y=gps["lat"],
        mode="lines",
        name="Track",
        line=dict(width=2),
        hoverinfo="skip",
    ))

    if have_fast:
        label = f"Speed ≥ {speed_for_gate} km/h (gate search)" if speed_for_gate is not None else "Fast points"
        fig.add_trace(go.Scatter(
            x=fast_pts["lon"], y=fast_pts["lat"],
            mode="markers",
            name=label,
            marker=dict(size=5),
            hovertemplate="Lon: %{x:.6f}<br>Lat: %{y:.6f}<extra></extra>",
        ))

    if have_gate:
        fig.add_trace(go.Scatter(
            x=[float(gate_lon)], y=[float(gate_lat)],
            mode="markers",
            name="Gate center",
            marker=dict(size=14, color="orange", symbol="star"),
            hovertemplate="Gate<br>Lon: %{x:.6f}<br>Lat: %{y:.6f}<extra></extra>",
        ))

    if have_pass:
        fig.add_trace(go.Scatter(
            x=gps.iloc[pass_idx]["lon"],
            y=gps.iloc[pass_idx]["lat"],
            mode="markers",
            name="Detected passes",
            marker=dict(size=8, color="red"),
            hovertemplate="PASS<br>Lon: %{x:.6f}<br>Lat: %{y:.6f}<extra></extra>",
        ))

    fig.update_layout(
        title="GPS Track (cartesian)",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        yaxis_scaleanchor="x",
        height=550,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h"),
    )
    return fig


def plot_distance_to_gate(
    gps: pd.DataFrame,
    *,
    dist: np.ndarray | None = None,
    pass_idx: np.ndarray | list[int] | None = None,
    near_m: float | None = None,
    title: str = "Distance to gate over time (passes = minima)",
) -> go.Figure:
    """
    Distance-to-gate plot. If dist isn't available, returns an empty figure with a message.
    """
    pass_idx = np.array(pass_idx, dtype=int) if pass_idx is not None else np.array([], dtype=int)

    fig = go.Figure()

    if dist is None or len(dist) == 0 or not np.isfinite(np.nanmax(np.asarray(dist, dtype=float))):
        # Keep the view, but show that data isn't available
        fig.update_layout(
            title=title,
            height=320,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_title="Time",
            yaxis_title="Distance (m)",
            annotations=[dict(
                text="This method does not produce gate-distance data.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14),
            )],
        )
        return fig

    dist = np.asarray(dist, dtype=float)

    fig.add_trace(go.Scatter(
        x=gps["timestamp"],
        y=dist,
        mode="lines",
        name="Distance to gate",
        line=dict(width=2),
        hovertemplate="Time: %{x}<br>Distance: %{y:.1f} m<extra></extra>",
    ))

    if len(pass_idx) > 0:
        fig.add_trace(go.Scatter(
            x=gps.iloc[pass_idx]["timestamp"],
            y=dist[pass_idx],
            mode="markers",
            name="Detected passes",
            marker=dict(size=9, color="orange"),
            hovertemplate="PASS<br>%{x}<br>%{y:.1f} m<extra></extra>",
        ))

    if near_m is not None and _is_finite_number(near_m):
        fig.add_hline(
            y=float(near_m),
            line_dash="dash",
            annotation_text=f"near_m = {near_m} m",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Distance (m)",
        height=320,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h"),
    )
    return fig


def make_plots(result: dict, params, track_view: str):
    gps = _safe_get(result, "gps")
    fast_pts = _safe_get(result, "fast_pts", None)
    gate_lat = _safe_get(result, "gate_lat", None)
    gate_lon = _safe_get(result, "gate_lon", None)
    dist = _safe_get(result, "dist", None)
    pass_idx = _safe_get(result, "pass_idx", None)
    laps = _safe_get(result, "laps", None)

    speed_for_gate = getattr(params, "speed_for_gate", None)
    near_m = getattr(params, "near_m", None)

    fig_track = plot_track(
        gps, track_view=track_view,
        fast_pts=fast_pts, gate_lat=gate_lat, gate_lon=gate_lon,
        pass_idx=pass_idx, speed_for_gate=speed_for_gate
    )

    fig_dist = plot_distance_to_gate(gps, dist=dist, pass_idx=pass_idx, near_m=near_m)

    fig_timeline = plot_lap_timeline(gps, pass_idx=pass_idx, laps=laps)
    fig_step = plot_lap_step(gps, pass_idx=pass_idx)

    fig_distdomain = plot_laps_through_distance(gps, pass_idx=pass_idx, laps=laps)
    return fig_track, fig_dist, fig_timeline, fig_step, fig_distdomain


def plot_lap_timeline(gps: pd.DataFrame, pass_idx, laps: pd.DataFrame | None = None) -> go.Figure:
    """
    Universal view: laps through time (works for all methods).
    Shows boundary lines + lap bands + speed overlay.
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    pass_idx = np.array(pass_idx, dtype=int) if pass_idx is not None else np.array([], dtype=int)

    # Speed line (if present)
    if "speed_kmh" in gps.columns and gps["speed_kmh"].notna().any():
        fig.add_trace(go.Scatter(
            x=gps["timestamp"],
            y=gps["speed_kmh"],
            mode="lines",
            name="Speed (km/h)",
            line=dict(width=2),
            hovertemplate="Time: %{x}<br>Speed: %{y:.1f} km/h<extra></extra>",
        ))
        y_title = "Speed (km/h)"
    else:
        y_title = "Value"
        fig.add_trace(go.Scatter(
            x=gps["timestamp"],
            y=np.zeros(len(gps)),
            mode="lines",
            name="Timeline",
            hoverinfo="skip",
        ))

    # Lap bands from laps DataFrame if available (preferred)
    shapes = []
    annotations = []

    if laps is not None and isinstance(laps, pd.DataFrame) and not laps.empty:
        for _, r in laps.iterrows():
            # translucent band per lap (alternating)
            lap = int(r["lap"])
            fill = "rgba(0,0,0,0.04)" if lap % 2 == 0 else "rgba(0,0,0,0.02)"
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=r["start"], x1=r["end"],
                y0=0, y1=1,
                fillcolor=fill,
                line=dict(width=0),
                layer="below",
            ))
            annotations.append(dict(
                x=r["start"],
                y=1.02,
                xref="x",
                yref="paper",
                text=f"Lap {lap}",
                showarrow=False,
                font=dict(size=11),
            ))

    # Boundary lines (always)
    if len(pass_idx) > 0:
        for i, idx in enumerate(pass_idx):
            ts = gps.iloc[idx]["timestamp"]
            shapes.append(dict(
                type="line",
                xref="x", yref="paper",
                x0=ts, x1=ts,
                y0=0, y1=1,
                line=dict(color="orange", width=2, dash="dot"),
                layer="above",
            ))

    fig.update_layout(
        title="Laps through time (boundaries + lap bands)",
        xaxis_title="Time",
        yaxis_title=y_title,
        height=360,
        margin=dict(l=40, r=40, t=60, b=40),
        shapes=shapes,
        annotations=annotations,
        legend=dict(orientation="h"),
    )
    return fig

def plot_lap_step(gps: pd.DataFrame, pass_idx) -> go.Figure:
    import numpy as np
    import plotly.graph_objects as go

    pass_idx = np.array(pass_idx, dtype=int) if pass_idx is not None else np.array([], dtype=int)

    if len(pass_idx) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Lap index over time",
            height=260,
            annotations=[dict(
                text="Not enough boundaries to build lap index plot.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )]
        )
        return fig

    t = gps.iloc[pass_idx]["timestamp"].reset_index(drop=True)

    # Build staircase from boundaries
    xs = []
    ys = []
    for lap in range(1, len(t)):
        xs.extend([t.iloc[lap-1], t.iloc[lap]])
        ys.extend([lap, lap])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        name="Lap index",
        line=dict(width=3),
        hovertemplate="Time: %{x}<br>Lap: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title="Lap index over time (staircase)",
        xaxis_title="Time",
        yaxis_title="Lap #",
        height=260,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
def cumulative_distance_m(gps: pd.DataFrame) -> np.ndarray:
    """
    Approx cumulative distance from successive lat/lon points using haversine.
    Returns array same length as gps.
    """
    lat = gps["lat"].to_numpy(dtype=float)
    lon = gps["lon"].to_numpy(dtype=float)

    # Haversine in meters (vectorized)
    R = 6371000.0
    lat1 = np.radians(lat[:-1]); lon1 = np.radians(lon[:-1])
    lat2 = np.radians(lat[1:]);  lon2 = np.radians(lon[1:])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    step = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.concatenate([[0.0], np.cumsum(step)])
    return s

def plot_laps_through_distance(gps: pd.DataFrame, pass_idx, laps: pd.DataFrame | None = None) -> go.Figure:
    """
    Universal view: laps through *distance covered*.
    X-axis is cumulative distance (m).
    Shows boundary lines and optional lap bands, with speed overlay if available.
    """
    pass_idx = np.array(pass_idx, dtype=int) if pass_idx is not None else np.array([], dtype=int)

    s = cumulative_distance_m(gps)

    fig = go.Figure()

    # Speed overlay (if available)
    if "speed_kmh" in gps.columns and gps["speed_kmh"].notna().any():
        fig.add_trace(go.Scatter(
            x=s,
            y=gps["speed_kmh"],
            mode="lines",
            name="Speed (km/h)",
            line=dict(width=2),
            hovertemplate="Dist: %{x:.1f} m<br>Speed: %{y:.1f} km/h<extra></extra>",
        ))
        y_title = "Speed (km/h)"
    else:
        fig.add_trace(go.Scatter(
            x=s,
            y=np.zeros(len(gps)),
            mode="lines",
            name="Timeline",
            hoverinfo="skip",
        ))
        y_title = "Value"

    shapes = []
    annotations = []

    # Lap bands (optional)
    if laps is not None and isinstance(laps, pd.DataFrame) and not laps.empty:
        # Convert lap start/end timestamps to distance positions by nearest timestamp
        t = gps["timestamp"].to_numpy()
        for _, r in laps.iterrows():
            lap = int(r["lap"])
            # nearest indices
            i0 = int(np.argmin(np.abs(t - np.datetime64(r["start"]))))
            i1 = int(np.argmin(np.abs(t - np.datetime64(r["end"]))))
            x0, x1 = float(s[i0]), float(s[i1])
            fill = "rgba(0,0,0,0.04)" if lap % 2 == 0 else "rgba(0,0,0,0.02)"
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=x0, x1=x1,
                y0=0, y1=1,
                fillcolor=fill,
                line=dict(width=0),
                layer="below",
            ))
            annotations.append(dict(
                x=x0, y=1.02,
                xref="x", yref="paper",
                text=f"Lap {lap}",
                showarrow=False,
                font=dict(size=11),
            ))

    # Boundary lines
    if len(pass_idx) > 0:
        for idx in pass_idx:
            x = float(s[idx])
            shapes.append(dict(
                type="line",
                xref="x", yref="paper",
                x0=x, x1=x,
                y0=0, y1=1,
                line=dict(color="orange", width=2, dash="dot"),
                layer="above",
            ))

    fig.update_layout(
        title="Laps through distance (boundaries + lap bands)",
        xaxis_title="Cumulative distance (m)",
        yaxis_title=y_title,
        height=360,
        margin=dict(l=40, r=40, t=60, b=40),
        shapes=shapes,
        annotations=annotations,
        legend=dict(orientation="h"),
    )
    return fig
