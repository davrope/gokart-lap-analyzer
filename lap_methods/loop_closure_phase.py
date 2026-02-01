from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from .gps_gate import fit_to_gps_speed_df_from_bytes, haversine_m, detect_passes_by_minima, passes_to_laps, tag_samples


@dataclass
class LoopClosurePhaseParams:
    # Lap expectations (used as constraints, not strict truth)
    expected_laps: int = field(default=15, metadata={"min": 1, "max": 200, "step": 1})
    min_lap_s: float = field(default=50.0, metadata={"min": 10.0, "max": 240.0, "step": 1.0})
    max_lap_s: float = field(default=120.0, metadata={"min": 20.0, "max": 400.0, "step": 1.0})

    # Loop closure geometry / motion constraints
    closure_radius_m: float = field(default=18.0, metadata={"min": 5.0, "max": 80.0, "step": 1.0, "help": "Near-previous distance to count as closure"})
    heading_tol_deg: float = field(default=35.0, metadata={"min": 5.0, "max": 120.0, "step": 1.0, "help": "Heading similarity tolerance"})
    min_closure_gap_s: float = field(default=20.0, metadata={"min": 5.0, "max": 120.0, "step": 1.0, "help": "Only compare to points at least this many seconds earlier"})

    # Speed filtering to avoid pits/exit
    speed_for_track_kmh: float = field(default=18.0, metadata={"min": 0.0, "max": 80.0, "step": 1.0, "help": "Used to ignore pit/exit slow regions"})
    min_boundary_speed_kmh: float = field(default=15.0, metadata={"min": 0.0, "max": 80.0, "step": 1.0, "help": "Boundary point must be >= this speed"})

    # Gate choice (phase consistency)
    grid_cell_m: float = field(default=8.0, metadata={"min": 2.0, "max": 30.0, "step": 1.0, "help": "Grid size for gate clustering"})
    top_cells: int = field(default=12, metadata={"min": 3, "max": 50, "step": 1, "help": "Try the best N closure clusters as gate candidates"})

    # Once gate chosen, this is the same as your gps_gate method
    near_m: float = field(default=22.0, metadata={"min": 8.0, "max": 80.0, "step": 1.0, "help": "Pass detection threshold around gate"})


# ------------------------------
# Helpers
# ------------------------------
def _meters_per_deg(lat0: float) -> Tuple[float, float]:
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    return m_per_deg_lat, m_per_deg_lon


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def compute_heading_deg(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Rough heading from consecutive points. Returns degrees in [-180, 180].
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    latr = np.radians(lat)
    lonr = np.radians(lon)
    dlon = np.diff(lonr, prepend=lonr[0])
    dlat = np.diff(latr, prepend=latr[0])
    hd = np.degrees(np.arctan2(dlon, dlat))
    # stabilize first element
    if len(hd) > 1:
        hd[0] = hd[1]
    return hd


def loop_closure_score(
    gps: pd.DataFrame,
    radius_m: float,
    heading_tol_deg: float,
    min_gap_s: float,
    speed_for_track_kmh: float,
    sample_stride: int = 2,
) -> np.ndarray:
    """
    For each point i, count how many earlier points j (>= min_gap_s earlier)
    are within radius_m and with similar heading. Uses subsampling for speed.
    """
    lat = gps["lat"].to_numpy(dtype=float)
    lon = gps["lon"].to_numpy(dtype=float)
    t = gps["timestamp"].to_numpy()
    spd = gps["speed_kmh"].to_numpy(dtype=float)

    hd = compute_heading_deg(lat, lon)
    tol = float(heading_tol_deg)

    n = len(gps)
    score = np.zeros(n, dtype=float)

    # pre-filter: only evaluate points likely on track (fast enough)
    valid_i = np.where(spd >= float(speed_for_track_kmh))[0]
    if len(valid_i) == 0:
        return score

    # Use stride for speed
    idx = np.arange(0, n, max(1, int(sample_stride)))

    for i in valid_i:
        # find j where time difference >= min_gap_s
        ti = t[i]
        # binary search lower bound
        # (assuming timestamps are increasing)
        jmax = np.searchsorted(t, ti - np.timedelta64(int(min_gap_s * 1000), "ms"), side="left")
        if jmax <= 10:
            continue

        # compare to subsampled earlier indices
        js = idx[idx < jmax]
        if len(js) == 0:
            continue

        # quick heading filter first
        dh = np.abs(((hd[js] - hd[i] + 180.0) % 360.0) - 180.0)
        js2 = js[dh <= tol]
        if len(js2) == 0:
            continue

        # distance filter
        d = haversine_m(lat[js2], lon[js2], lat[i], lon[i])
        score[i] = float(np.sum(d <= float(radius_m)))

    return score


def grid_top_cells(gps: pd.DataFrame, weight: np.ndarray, cell_m: float, top_k: int) -> list[tuple[float, float]]:
    """
    Bin points into meter-grid cells and return top-k cell centers
    (median lat/lon of points in each cell) weighted by closure score.
    """
    lat = gps["lat"].to_numpy(dtype=float)
    lon = gps["lon"].to_numpy(dtype=float)

    lat0 = float(np.nanmedian(lat))
    mlat, mlon = _meters_per_deg(lat0)

    x = (lon - float(np.nanmean(lon))) * mlon
    y = (lat - float(np.nanmean(lat))) * mlat

    gx = np.floor(x / float(cell_m)).astype(int)
    gy = np.floor(y / float(cell_m)).astype(int)

    # aggregate weights per cell
    key = gx.astype(np.int64) * 10_000_000 + gy.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)

    wsum = np.zeros(len(uniq), dtype=float)
    for i, k in enumerate(inv):
        wsum[k] += float(weight[i])

    # pick top-k
    order = np.argsort(wsum)[::-1]
    centers = []
    for k in order[: max(1, int(top_k))]:
        mask = inv == k
        if np.sum(mask) < 10:
            continue
        centers.append((float(np.nanmedian(lat[mask])), float(np.nanmedian(lon[mask]))))

    # if nothing strong, fallback to median point
    if not centers:
        centers = [(float(np.nanmedian(lat)), float(np.nanmedian(lon)))]
    return centers


def score_gate_candidate(
    gps: pd.DataFrame,
    gate_lat: float,
    gate_lon: float,
    params: LoopClosurePhaseParams,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Given a gate, detect passes and score how "phase-consistent" it is.
    Lower score is better.
    Returns: (score, dist, pass_idx, dt_s)
    """
    dist, pass_idx, dt_s = detect_passes_by_minima(
        gps,
        gate_lat,
        gate_lon,
        near_m=float(params.near_m),
        min_lap_s=float(params.min_lap_s),
    )

    if len(pass_idx) < 3:
        return (1e18, dist, pass_idx, dt_s)

    # speed plausibility at boundaries
    bspd = gps.loc[pass_idx, "speed_kmh"].to_numpy(dtype=float)
    low_speed_frac = float(np.mean(bspd < float(params.min_boundary_speed_kmh)))

    # lap times
    laps = passes_to_laps(gps, pass_idx)
    if laps.empty:
        return (1e18, dist, pass_idx, dt_s)

    lt = laps["lap_time_s"].to_numpy(dtype=float)
    mean = float(np.mean(lt))
    cv = float(np.std(lt, ddof=1) / mean) if len(lt) > 1 and mean > 0 else 1.0

    # boundary clustering in meters
    lat0 = float(np.nanmedian(gps.loc[pass_idx, "lat"]))
    mlat, mlon = _meters_per_deg(lat0)
    bx = (gps.loc[pass_idx, "lon"].to_numpy(dtype=float) - float(np.nanmean(gps.loc[pass_idx, "lon"]))) * mlon
    by = (gps.loc[pass_idx, "lat"].to_numpy(dtype=float) - float(np.nanmean(gps.loc[pass_idx, "lat"]))) * mlat
    spread_m = float(np.sqrt(np.nanvar(bx) + np.nanvar(by)))

    # number of laps plausibility (soft)
    n_laps = len(laps)
    n_err = abs(n_laps - int(params.expected_laps))
    n_pen = 0.15 * float(n_err)

    # final score (tweakable)
    score = 0.02 * spread_m + 2.0 * cv + 10.0 * low_speed_frac + n_pen
    return (score, dist, pass_idx, dt_s)


# ------------------------------
# Runner
# ------------------------------
def run(fit_bytes: bytes, params: LoopClosurePhaseParams) -> Dict[str, object]:
    gps = fit_to_gps_speed_df_from_bytes(fit_bytes)
    gps = gps.dropna(subset=["lat", "lon", "speed_kmh"]).reset_index(drop=True)

    # 1) compute loop-closure score
    score = loop_closure_score(
        gps,
        radius_m=float(params.closure_radius_m),
        heading_tol_deg=float(params.heading_tol_deg),
        min_gap_s=float(params.min_closure_gap_s),
        speed_for_track_kmh=float(params.speed_for_track_kmh),
        sample_stride=2,
    )

    # 2) propose gate candidates from top closure clusters
    # weight by score AND speed to avoid pits
    w = score * (gps["speed_kmh"].to_numpy(dtype=float) >= float(params.speed_for_track_kmh))
    gate_candidates = grid_top_cells(gps, w, cell_m=float(params.grid_cell_m), top_k=int(params.top_cells))

    # 3) pick best gate by phase consistency score
    best = {"score": 1e18, "gate_lat": np.nan, "gate_lon": np.nan, "dist": None, "pass_idx": None, "dt_s": np.nan}
    for (glat, glon) in gate_candidates:
        sc, dist, pass_idx, dt_s = score_gate_candidate(gps, glat, glon, params)
        if sc < best["score"]:
            best = {"score": sc, "gate_lat": glat, "gate_lon": glon, "dist": dist, "pass_idx": pass_idx, "dt_s": dt_s}

    gate_lat = float(best["gate_lat"])
    gate_lon = float(best["gate_lon"])
    dist = np.asarray(best["dist"], dtype=float) if best["dist"] is not None else np.full(len(gps), np.nan)
    pass_idx = np.asarray(best["pass_idx"], dtype=int) if best["pass_idx"] is not None else np.array([], dtype=int)
    dt_s = float(best["dt_s"]) if np.isfinite(best["dt_s"]) else np.nan

    laps = passes_to_laps(gps, pass_idx)
    samples = tag_samples(gps, laps, dist)


    # Build lap metrics like other methods (keep minimal here; your app already enriches)
    if laps.empty:
        lap_metrics = pd.DataFrame()
    else:
        lap_metrics = laps.copy()
        lap_metrics["lap"] = lap_metrics["lap"].astype(int)

    # fast points for visualization: points above track speed
    fast_pts = gps[gps["speed_kmh"] >= float(params.speed_for_track_kmh)].copy()

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
        "closure_score": score,                 # optional debug
        "gate_candidate_count": len(gate_candidates),
        "quality_score_internal": float(best["score"]),
    }
