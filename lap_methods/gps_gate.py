from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from fitparse import FitFile

try:
    from scipy.signal import find_peaks
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


@dataclass
class GpsGateParams:
    expected_laps: int = field(default=15, metadata={"min": 1, "max": 300, "step": 1, "help": "Just for sanity checking"})
    min_lap_s: float = field(default=50.0, metadata={"min": 10.0, "max": 200.0, "step": 1.0, "help": "Minimum seconds between lap passes"})
    near_m: float = field(default=22.0, metadata={"min": 5.0, "max": 60.0, "step": 1.0, "help": "How close (meters) counts as passing the gate"})
    speed_for_gate: float = field(default=18.0, metadata={"min": 0.0, "max": 60.0, "step": 1.0, "help": "Filter slow pit/exit points when choosing gate"})
    cell_size_m: float = field(default=8.0, metadata={"min": 2.0, "max": 25.0, "step": 1.0, "help": "Grid size used to find densest gate region"})


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def fit_to_gps_speed_df_from_bytes(fit_bytes: bytes) -> pd.DataFrame:
    fitfile = FitFile(io.BytesIO(fit_bytes))
    rows = []
    for msg in fitfile.get_messages("record"):
        rows.append({f.name: f.value for f in msg})
    df = pd.DataFrame(rows)

    if "timestamp" not in df.columns:
        raise ValueError("No timestamp found in FIT record messages.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if not {"position_lat", "position_long"}.issubset(df.columns):
        raise ValueError("No position_lat/position_long found in FIT records.")

    df["lat"] = pd.to_numeric(df["position_lat"], errors="coerce").astype("float64") * (180.0 / 2**31)
    df["lon"] = pd.to_numeric(df["position_long"], errors="coerce").astype("float64") * (180.0 / 2**31)

    # speed km/h
    speed_col = None
    if "enhanced_speed" in df.columns and df["enhanced_speed"].notna().any():
        speed_col = "enhanced_speed"
    elif "speed" in df.columns and df["speed"].notna().any():
        speed_col = "speed"
    if speed_col is None:
        raise ValueError("No speed/enhanced_speed found; needed to avoid pits.")
    df["speed_kmh"] = pd.to_numeric(df[speed_col], errors="coerce") * 3.6

    # optional extras
    if "heart_rate" in df.columns:
        df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")

    alt_col = None
    if "enhanced_altitude" in df.columns and df["enhanced_altitude"].notna().any():
        alt_col = "enhanced_altitude"
    elif "altitude" in df.columns and df["altitude"].notna().any():
        alt_col = "altitude"
    if alt_col:
        df["elev_m"] = pd.to_numeric(df[alt_col], errors="coerce")

    keep = ["timestamp", "lat", "lon", "speed_kmh"]
    if "heart_rate" in df.columns: keep.append("heart_rate")
    if "elev_m" in df.columns: keep.append("elev_m")

    return df[keep].dropna(subset=["lat", "lon"]).reset_index(drop=True)


def find_gate_center_from_fast_points(
    gps: pd.DataFrame,
    cell_size_m: float,
    speed_for_gate: float,
) -> Tuple[float, float, pd.DataFrame]:
    fast = gps[gps["speed_kmh"] >= speed_for_gate].copy()
    if len(fast) < 30:
        fast = gps.copy()

    lat = fast["lat"].to_numpy()
    lon = fast["lon"].to_numpy()

    lat0 = float(np.median(lat))
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))

    x = (lon - lon.mean()) * m_per_deg_lon
    y = (lat - lat.mean()) * m_per_deg_lat

    gx = np.floor(x / cell_size_m).astype(int)
    gy = np.floor(y / cell_size_m).astype(int)

    bins = np.stack([gx, gy], axis=1)
    uniq, counts = np.unique(bins, axis=0, return_counts=True)
    bx, by = uniq[np.argmax(counts)]

    mask = (gx == bx) & (gy == by)
    gate_lat = float(np.median(lat[mask]))
    gate_lon = float(np.median(lon[mask]))
    return gate_lat, gate_lon, fast


def detect_passes_by_minima(
    gps: pd.DataFrame,
    gate_lat: float,
    gate_lon: float,
    near_m: float,
    min_lap_s: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    dist = haversine_m(gps["lat"].to_numpy(), gps["lon"].to_numpy(), gate_lat, gate_lon)
    t = gps["timestamp"].to_numpy()

    dt_s = np.median(np.diff(t) / np.timedelta64(1, "s"))
    if not np.isfinite(dt_s) or dt_s <= 0:
        dt_s = 1.0
    min_samples = max(1, int(round(min_lap_s / dt_s)))

    if HAVE_SCIPY:
        prom = max(2.0, 0.12 * near_m)
        idx, _ = find_peaks(-dist, distance=min_samples, prominence=prom)
        idx = idx[dist[idx] <= near_m]
    else:
        idx_list = []
        last = -10**9
        for i in range(1, len(dist) - 1):
            if i - last < min_samples:
                continue
            if dist[i] <= dist[i - 1] and dist[i] <= dist[i + 1] and dist[i] <= near_m:
                idx_list.append(i)
                last = i
        idx = np.array(idx_list, dtype=int)

    return dist, idx, float(dt_s)


def passes_to_laps(gps: pd.DataFrame, pass_idx: np.ndarray) -> pd.DataFrame:
    times = gps.loc[pass_idx, "timestamp"].reset_index(drop=True)
    if len(times) < 2:
        return pd.DataFrame()
    laps = pd.DataFrame({
        "lap": np.arange(1, len(times)),
        "start": times.iloc[:-1].values,
        "end": times.iloc[1:].values,
    })
    laps["lap_time_s"] = (pd.to_datetime(laps["end"]) - pd.to_datetime(laps["start"])).dt.total_seconds()
    return laps


def summarize_laps(gps: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()
    out = []
    for _, r in laps.iterrows():
        seg = gps[(gps["timestamp"] >= r["start"]) & (gps["timestamp"] < r["end"])].copy()
        row = {
            "lap": int(r["lap"]),
            "start": r["start"],
            "end": r["end"],
            "lap_time_s": float(r["lap_time_s"]),
            "avg_speed_kmh": float(seg["speed_kmh"].mean()),
            "max_speed_kmh": float(seg["speed_kmh"].max()),
        }
        if "heart_rate" in seg.columns and seg["heart_rate"].notna().any():
            row["avg_hr"] = float(seg["heart_rate"].mean())
            row["max_hr"] = float(seg["heart_rate"].max())
        if "elev_m" in seg.columns and seg["elev_m"].notna().any():
            row["elev_min"] = float(seg["elev_m"].min())
            row["elev_max"] = float(seg["elev_m"].max())
            row["elev_range"] = row["elev_max"] - row["elev_min"]
        out.append(row)
    return pd.DataFrame(out)


def tag_samples(gps: pd.DataFrame, laps: pd.DataFrame, dist: np.ndarray) -> pd.DataFrame:
    g = gps.copy()
    g["dist_to_gate_m"] = dist
    g["lap"] = pd.NA
    for _, r in laps.iterrows():
        m = (g["timestamp"] >= r["start"]) & (g["timestamp"] < r["end"])
        g.loc[m, "lap"] = int(r["lap"])
    return g


def run(fit_bytes: bytes, params: GpsGateParams) -> Dict[str, object]:
    """
    Standard method entry point.
    Returns a dict with:
      gps, fast_pts, gate_lat, gate_lon, dist, pass_idx, dt_s, laps, lap_metrics, samples
    """
    gps = fit_to_gps_speed_df_from_bytes(fit_bytes)

    gate_lat, gate_lon, fast_pts = find_gate_center_from_fast_points(
        gps,
        cell_size_m=float(params.cell_size_m),
        speed_for_gate=float(params.speed_for_gate),
    )
    dist, pass_idx, dt_s = detect_passes_by_minima(
        gps,
        gate_lat,
        gate_lon,
        near_m=float(params.near_m),
        min_lap_s=float(params.min_lap_s),
    )

    laps = passes_to_laps(gps, pass_idx)
    lap_metrics = summarize_laps(gps, laps)
    samples = tag_samples(gps, laps, dist)

    return {
        "gps": gps,
        "fast_pts": fast_pts,
        "gate_lat": gate_lat,
        "gate_lon": gate_lon,
        "dist": dist,
        "pass_idx": pass_idx,
        "dt_s": dt_s,
        "laps": laps,
        "lap_metrics": lap_metrics,
        "samples": samples,
    }
