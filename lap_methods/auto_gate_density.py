# lap_methods/auto_gate_density.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .gps_gate import (
    fit_to_gps_speed_df_from_bytes,
    haversine_m,
    detect_passes_by_minima,
    passes_to_laps,
    tag_samples,
)

try:
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


@dataclass
class AutoGateDensityParams:
    # Grid resolution for density/track-space
    cell_size_m: float = field(default=6.0, metadata={"min": 2.0, "max": 20.0, "step": 1.0})
    # Density thresholding (auto picks a percentile of cell counts)
    cell_count_percentile: float = field(default=80.0, metadata={"min": 50.0, "max": 98.0, "step": 1.0})

    # Track-speed auto: percentiles over track points
    speed_gate_percentile: float = field(default=70.0, metadata={"min": 40.0, "max": 95.0, "step": 1.0})
    speed_min_floor: float = field(default=8.0, metadata={"min": 0.0, "max": 40.0, "step": 1.0})

    # near_m auto scaling
    near_quantile: float = field(default=0.08, metadata={"min": 0.02, "max": 0.25, "step": 0.01})
    near_scale: float = field(default=1.8, metadata={"min": 1.0, "max": 4.0, "step": 0.1})
    near_bounds_m: Tuple[float, float] = (10.0, 60.0)

    # Lap time bounds (used to estimate min_lap_s)
    lap_s_bounds: Tuple[float, float] = (35.0, 140.0)
    min_lap_s_safety_factor: float = field(default=0.75, metadata={"min": 0.5, "max": 0.95, "step": 0.05})

    # If you still want an "expected laps" estimate:
    estimate_expected_laps: bool = field(default=True)


def _meters_per_deg(lat0: float) -> Tuple[float, float]:
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    return m_per_deg_lat, m_per_deg_lon


def _to_xy_m(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    lat0 = float(np.nanmedian(lat))
    mlat, mlon = _meters_per_deg(lat0)
    x = (lon - float(np.nanmean(lon))) * mlon
    y = (lat - float(np.nanmean(lat))) * mlat
    return x, y, mlat, mlon


def _grid_bins(x: np.ndarray, y: np.ndarray, cell_m: float) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.floor(x / float(cell_m)).astype(int)
    gy = np.floor(y / float(cell_m)).astype(int)
    return gx, gy


def _largest_component_cells(active: np.ndarray, gx: np.ndarray, gy: np.ndarray):
    """
    active: boolean per-point indicating point falls in an active cell.
    Returns mask for points in the largest connected component of active cells (4-neighborhood).
    """
    # Build unique active cells
    cells = np.stack([gx[active], gy[active]], axis=1)
    if len(cells) == 0:
        return active

    uniq = np.unique(cells, axis=0)
    # map cell -> index
    cell_to_idx = { (int(a), int(b)): i for i, (a,b) in enumerate(uniq) }

    # adjacency list
    adj = [[] for _ in range(len(uniq))]
    for i, (cx, cy) in enumerate(uniq):
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            j = cell_to_idx.get((int(cx+dx), int(cy+dy)))
            if j is not None:
                adj[i].append(j)

    # BFS components
    seen = np.zeros(len(uniq), dtype=bool)
    best_comp = []
    for i in range(len(uniq)):
        if seen[i]:
            continue
        q = [i]
        seen[i] = True
        comp = [i]
        while q:
            u = q.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        if len(comp) > len(best_comp):
            best_comp = comp

    best_cells = set(tuple(uniq[i]) for i in best_comp)
    in_best = np.array([(int(a), int(b)) in best_cells for a,b in zip(gx, gy)], dtype=bool)
    return in_best


def track_space_mask(gps: pd.DataFrame, p: AutoGateDensityParams) -> np.ndarray:
    """
    Returns boolean mask of points considered 'track space' using grid density + largest component.
    """
    lat = gps["lat"].to_numpy(dtype=float)
    lon = gps["lon"].to_numpy(dtype=float)
    x, y, *_ = _to_xy_m(lat, lon)
    gx, gy = _grid_bins(x, y, p.cell_size_m)

    # cell counts
    key = gx.astype(np.int64) * 10_000_000 + gy.astype(np.int64)
    uniq, counts = np.unique(key, return_counts=True)
    # threshold by percentile of cell counts
    thr = np.percentile(counts, float(p.cell_count_percentile))
    active_cells = set(uniq[counts >= thr])

    active = np.array([k in active_cells for k in key], dtype=bool)
    # keep only the largest connected component of active cells (usually the loop track)
    active = _largest_component_cells(active, gx, gy)
    return active


def auto_gate_from_density(gps: pd.DataFrame, track_mask: np.ndarray, p: AutoGateDensityParams) -> Tuple[float, float]:
    """
    Gate = densest cell within track_mask.
    """
    g = gps.loc[track_mask].copy()
    if len(g) < 30:
        # fallback
        return (float(np.nanmedian(gps["lat"])), float(np.nanmedian(gps["lon"])))

    lat = g["lat"].to_numpy(dtype=float)
    lon = g["lon"].to_numpy(dtype=float)
    x, y, *_ = _to_xy_m(lat, lon)
    gx, gy = _grid_bins(x, y, p.cell_size_m)

    key = gx.astype(np.int64) * 10_000_000 + gy.astype(np.int64)
    uniq, counts = np.unique(key, return_counts=True)
    best = uniq[np.argmax(counts)]

    m = key == best
    return (float(np.nanmedian(lat[m])), float(np.nanmedian(lon[m])))


def auto_speed_for_gate(gps: pd.DataFrame, track_mask: np.ndarray, p: AutoGateDensityParams) -> float:
    spd = gps.loc[track_mask, "speed_kmh"].to_numpy(dtype=float)
    spd = spd[np.isfinite(spd)]
    if len(spd) == 0:
        return float(p.speed_min_floor)
    return float(max(p.speed_min_floor, np.percentile(spd, float(p.speed_gate_percentile))))


def auto_near_m(gps: pd.DataFrame, gate_lat: float, gate_lon: float, track_mask: np.ndarray, speed_for_gate: float, p: AutoGateDensityParams) -> float:
    g = gps.loc[track_mask].copy()
    # focus on "fast" track points to avoid pit approach
    g = g[g["speed_kmh"] >= float(speed_for_gate)]
    if len(g) < 30:
        g = gps.loc[track_mask].copy()

    d = haversine_m(g["lat"].to_numpy(dtype=float), g["lon"].to_numpy(dtype=float), gate_lat, gate_lon)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float(np.mean(p.near_bounds_m))

    base = float(np.quantile(d, float(p.near_quantile)))
    near = float(base * float(p.near_scale))
    bounds = getattr(p, "near_bounds_m", (10.0, 60.0))

    # Robust parsing in case Streamlit widgets changed the type/shape
    if isinstance(bounds, str):
        # Accept "10,60" or "10 60"
        parts = [x for x in bounds.replace(",", " ").split() if x.strip()]
        vals = []
        for s in parts:
            try:
                vals.append(float(s))
            except Exception:
                pass
        if len(vals) >= 2:
            lo, hi = vals[0], vals[1]
        else:
            lo, hi = 10.0, 60.0
    else:
        try:
            b = list(bounds)
            if len(b) >= 2:
                lo, hi = float(b[0]), float(b[1])
            else:
                lo, hi = 10.0, 60.0
        except Exception:
            lo, hi = 10.0, 60.0

    # Ensure ordering
    if hi < lo:
        lo, hi = hi, lo

    return float(np.clip(near, lo, hi))



def auto_min_lap_s(gps: pd.DataFrame, dist: np.ndarray, near_m: float, p: AutoGateDensityParams) -> float:
    """
    Estimate lap time from spacing between near-minima events.
    Uses a loose peak finding on -dist, then takes median spacing within bounds.
    """
    t = gps["timestamp"].to_numpy()
    if len(t) < 20:
        return float(p.lap_s_bounds[0])

    dt_s = np.median(np.diff(t) / np.timedelta64(1, "s"))
    if not np.isfinite(dt_s) or dt_s <= 0:
        dt_s = 1.0

    # Candidate minima indices
    # We use a relaxed threshold: dist must be <= near_m to count as potential pass
    cand = np.where(dist <= float(near_m))[0]
    if len(cand) < 5:
        return float(p.lap_s_bounds[0])

    if _HAVE_SCIPY:
        # Find peaks in -dist (minima in dist)
        idx, _ = find_peaks(-dist, distance=max(2, int(round(10.0 / dt_s))))
        idx = idx[dist[idx] <= float(near_m)]
    else:
        # Simple local minima
        idx = []
        for i in range(1, len(dist)-1):
            if dist[i] <= dist[i-1] and dist[i] <= dist[i+1] and dist[i] <= float(near_m):
                idx.append(i)
        idx = np.array(idx, dtype=int)

    if len(idx) < 3:
        return float(p.lap_s_bounds[0])

    times = t[idx]
    gaps = np.diff(times) / np.timedelta64(1, "s")
    bounds = getattr(p, "near_bounds_m", (10.0, 60.0))

    # Robust parsing in case Streamlit widgets changed the type/shape
    if isinstance(bounds, str):
        # Accept "10,60" or "10 60"
        parts = [x for x in bounds.replace(",", " ").split() if x.strip()]
        vals = []
        for s in parts:
            try:
                vals.append(float(s))
            except Exception:
                pass
        if len(vals) >= 2:
            lo, hi = vals[0], vals[1]
        else:
            lo, hi = 10.0, 60.0
    else:
        try:
            b = list(bounds)
            if len(b) >= 2:
                lo, hi = float(b[0]), float(b[1])
            else:
                lo, hi = 10.0, 60.0
        except Exception:
            lo, hi = 10.0, 60.0

    # Ensure ordering
    if hi < lo:
        lo, hi = hi, lo



    gaps = gaps[(gaps >= lo) & (gaps <= hi)]
    if len(gaps) == 0:
        return float(lo)

    med = float(np.median(gaps))
    # Use a safety factor so we don’t accidentally block legit laps
    return float(max(lo, min(hi, med * float(p.min_lap_s_safety_factor))))


def estimate_expected_laps(gps: pd.DataFrame, laps: pd.DataFrame) -> int:
    if laps is None or laps.empty:
        return 0
    return int(len(laps))


def run(fit_bytes: bytes, p: AutoGateDensityParams) -> Dict[str, object]:
    gps = fit_to_gps_speed_df_from_bytes(fit_bytes)
    gps = gps.dropna(subset=["timestamp", "lat", "lon", "speed_kmh"]).reset_index(drop=True)

    # 1) Track space
    tmask = track_space_mask(gps, p)

    # 2) Auto gate center
    gate_lat, gate_lon = auto_gate_from_density(gps, tmask, p)

    # 3) Auto speed_for_gate and near_m
    speed_for_gate = auto_speed_for_gate(gps, tmask, p)
    near_m = auto_near_m(gps, gate_lat, gate_lon, tmask, speed_for_gate, p)

    # 4) Build dist + auto min_lap_s
    dist = haversine_m(gps["lat"].to_numpy(dtype=float), gps["lon"].to_numpy(dtype=float), gate_lat, gate_lon)
    min_lap_s = auto_min_lap_s(gps, dist, near_m, p)

    # 5) Detect passes/laps using your trusted minima logic
    dist2, pass_idx, dt_s = detect_passes_by_minima(gps, gate_lat, gate_lon, near_m=near_m, min_lap_s=min_lap_s)
    laps = passes_to_laps(gps, pass_idx)

    # 6) Samples + lap_metrics (minimal; app will enrich)
    samples = tag_samples(gps, laps, dist2)
    lap_metrics = laps.copy() if not laps.empty else pd.DataFrame()

    # fast points for visualization
    fast_pts = gps[gps["speed_kmh"] >= float(speed_for_gate)].copy()

    return {
        "gps": gps,
        "fast_pts": fast_pts,
        "gate_lat": float(gate_lat),
        "gate_lon": float(gate_lon),
        "dist": dist2,
        "pass_idx": pass_idx,
        "dt_s": float(dt_s),
        "laps": laps,
        "lap_metrics": lap_metrics,
        "samples": samples,
        # expose the auto params for display
        "auto_params": {
            "track_cell_size_m": float(p.cell_size_m),
            "speed_for_gate": float(speed_for_gate),
            "near_m": float(near_m),
            "min_lap_s": float(min_lap_s),
            "track_points_frac": float(np.mean(tmask)),
        },
        "track_mask": tmask,  # useful for plots
    }
