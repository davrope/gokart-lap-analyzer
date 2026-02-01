from __future__ import annotations

import numpy as np
import pandas as pd


def _meters_per_deg(lat_deg: float):
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_deg))
    return m_per_deg_lat, m_per_deg_lon


def _resample_series(x: np.ndarray, y: np.ndarray, n: int = 200) -> np.ndarray:
    """
    Resample y(x) onto n evenly spaced points across x range using linear interp.
    x must be increasing.
    """
    if len(x) < 2:
        return np.full(n, np.nan)
    x0, x1 = x[0], x[-1]
    if x1 == x0:
        return np.full(n, np.nan)
    xi = np.linspace(x0, x1, n)
    return np.interp(xi, x, y)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return np.nan
    return float(np.dot(a, b) / (da * db))


def quality_summary_gps_gate(result: dict, params) -> dict:
    """
    Unsupervised quality metrics for the GPS-gate method.
    Expects keys from gps_gate.run(): gps, pass_idx, laps, lap_metrics, dist.
    """
    gps: pd.DataFrame = result["gps"]
    pass_idx = result["pass_idx"]
    laps: pd.DataFrame = result["laps"]
    lap_metrics: pd.DataFrame = result["lap_metrics"]
    dist: np.ndarray = result["dist"]

    out = {}

    # -----------------------
    # 1) Periodicity (lap times)
    # -----------------------
    if laps is None or laps.empty:
        out["laps_detected"] = 0
        out["lap_time_mean_s"] = np.nan
        out["lap_time_std_s"] = np.nan
        out["lap_time_cv"] = np.nan
        out["lap_time_mad_s"] = np.nan
    else:
        lt = lap_metrics["lap_time_s"].to_numpy(dtype=float)
        out["laps_detected"] = int(len(lt))
        out["lap_time_mean_s"] = float(np.mean(lt))
        out["lap_time_std_s"] = float(np.std(lt, ddof=1)) if len(lt) > 1 else 0.0
        out["lap_time_cv"] = float(out["lap_time_std_s"] / out["lap_time_mean_s"]) if out["lap_time_mean_s"] else np.nan
        med = float(np.median(lt))
        out["lap_time_mad_s"] = float(np.median(np.abs(lt - med)))

        # -----------------------
    # 1b) Lap distance consistency (requires lap_metrics["lap_distance_m"])
    # -----------------------
    if lap_metrics is not None and not lap_metrics.empty and "lap_distance_m" in lap_metrics.columns:
        ld = lap_metrics["lap_distance_m"].to_numpy(dtype=float)
        out["lap_dist_mean_m"] = float(np.mean(ld))
        out["lap_dist_std_m"] = float(np.std(ld, ddof=1)) if len(ld) > 1 else 0.0
        out["lap_dist_cv"] = float(out["lap_dist_std_m"] / out["lap_dist_mean_m"]) if out["lap_dist_mean_m"] else np.nan
        med_d = float(np.median(ld))
        out["lap_dist_mad_m"] = float(np.median(np.abs(ld - med_d)))
    else:
        out["lap_dist_mean_m"] = np.nan
        out["lap_dist_std_m"] = np.nan
        out["lap_dist_cv"] = np.nan
        out["lap_dist_mad_m"] = np.nan

    # -----------------------
    # 2) Within-lap shape consistency (speed)
    # -----------------------
    # Resample each lap's speed to same length and correlate with template
    shape_corrs = []
    if laps is not None and not laps.empty:
        N = 200
        lap_shapes = []
        for _, r in laps.iterrows():
            seg = gps[(gps["timestamp"] >= r["start"]) & (gps["timestamp"] < r["end"])].copy()
            if len(seg) < 10:
                continue
            t = seg["timestamp"].to_numpy()
            x = (t - t[0]) / np.timedelta64(1, "s")
            y = seg["speed_kmh"].to_numpy(dtype=float)
            y_rs = _resample_series(x, y, n=N)
            if np.all(np.isfinite(y_rs)):
                lap_shapes.append(y_rs)

        if len(lap_shapes) >= 2:
            M = np.vstack(lap_shapes)
            template = np.nanmean(M, axis=0)
            for i in range(M.shape[0]):
                shape_corrs.append(_corr(M[i], template))

    out["shape_corr_mean"] = float(np.nanmean(shape_corrs)) if len(shape_corrs) else np.nan
    out["shape_corr_median"] = float(np.nanmedian(shape_corrs)) if len(shape_corrs) else np.nan
    out["shape_corr_nlaps_used"] = int(np.sum(np.isfinite(shape_corrs))) if len(shape_corrs) else 0

    # -----------------------
    # 3) Boundary plausibility
    # -----------------------
    # A) Are boundary locations clustered?
    if pass_idx is not None and len(pass_idx) > 0:
        b = gps.loc[pass_idx, ["lat", "lon", "speed_kmh"]].copy()
        lat0 = float(np.nanmedian(b["lat"]))
        mlat, mlon = _meters_per_deg(lat0)
        bx = (b["lon"].to_numpy(dtype=float) - float(np.nanmean(b["lon"]))) * mlon
        by = (b["lat"].to_numpy(dtype=float) - float(np.nanmean(b["lat"]))) * mlat

        out["boundary_spread_m"] = float(np.sqrt(np.nanvar(bx) + np.nanvar(by)))  # radial-ish spread
        out["boundary_speed_mean_kmh"] = float(np.nanmean(b["speed_kmh"]))
        out["boundary_speed_min_kmh"] = float(np.nanmin(b["speed_kmh"]))
        out["boundary_fast_frac"] = float(np.mean(b["speed_kmh"].to_numpy(dtype=float) >= float(params.speed_for_gate)))
        out["boundary_dist_mean_m"] = float(np.nanmean(dist[pass_idx]))
        out["boundary_dist_max_m"] = float(np.nanmax(dist[pass_idx]))
    else:
        out["boundary_spread_m"] = np.nan
        out["boundary_speed_mean_kmh"] = np.nan
        out["boundary_speed_min_kmh"] = np.nan
        out["boundary_fast_frac"] = np.nan
        out["boundary_dist_mean_m"] = np.nan
        out["boundary_dist_max_m"] = np.nan

    # -----------------------
    # 4) Stability under perturbation (downsample)
    # -----------------------
    # We compare pass timestamps between full data and a downsampled version (every 2nd point).
    # "Stable" means boundaries shift only slightly.
    stability = {
        "stable_passes_matched": 0,
        "stable_passes_total": int(len(pass_idx)) if pass_idx is not None else 0,
        "stable_median_shift_s": np.nan,
        "stable_p95_shift_s": np.nan,
    }

    if pass_idx is not None and len(pass_idx) >= 3:
        # Downsample
        gps2 = gps.iloc[::2].reset_index(drop=True)

        # Re-use same gate and detection logic: relies on dist minima to gate
        gate_lat = float(result["gate_lat"])
        gate_lon = float(result["gate_lon"])

        # Compute dist for downsampled
        from .gps_gate import haversine_m, detect_passes_by_minima  # method-specific

        dist2 = haversine_m(gps2["lat"].to_numpy(), gps2["lon"].to_numpy(), gate_lat, gate_lon)
        # Re-run minima detection with same near/minlap
        # We call gps_gate.detect_passes_by_minima for consistent behavior:
        # (it recomputes dt_s from gps2 timestamps)
        dist2b, pass_idx2, _dt2 = detect_passes_by_minima(
            gps2, gate_lat, gate_lon, near_m=float(params.near_m), min_lap_s=float(params.min_lap_s)
        )

        t1 = gps.loc[pass_idx, "timestamp"].to_numpy()
        t2 = gps2.loc[pass_idx2, "timestamp"].to_numpy()

        # Match each t1 to nearest t2 within tolerance
        tol_s = 5.0
        shifts = []
        j = 0
        for ti in t1:
            # advance j while next is closer
            while j + 1 < len(t2) and abs((t2[j + 1] - ti) / np.timedelta64(1, "s")) < abs((t2[j] - ti) / np.timedelta64(1, "s")):
                j += 1
            if len(t2) == 0:
                continue
            shift = float((t2[j] - ti) / np.timedelta64(1, "s"))
            if abs(shift) <= tol_s:
                shifts.append(abs(shift))

        if len(shifts) > 0:
            stability["stable_passes_matched"] = int(len(shifts))
            stability["stable_median_shift_s"] = float(np.median(shifts))
            stability["stable_p95_shift_s"] = float(np.percentile(shifts, 95))

    out.update(stability)

    # A simple single “quality score” (optional heuristic)
    # Lower is better for CV/spread/shift; higher is better for shape_corr/fast_frac
    score = 0.0
    if np.isfinite(out.get("lap_time_cv", np.nan)):
        score += 2.0 * out["lap_time_cv"]

    # NEW: lap distance consistency penalty
    if np.isfinite(out.get("lap_dist_cv", np.nan)):
        score += 1.5 * out["lap_dist_cv"]

    if np.isfinite(out.get("boundary_spread_m", np.nan)):
        score += 0.02 * out["boundary_spread_m"]
    if np.isfinite(out.get("stable_median_shift_s", np.nan)):
        score += 0.5 * out["stable_median_shift_s"]
    if np.isfinite(out.get("shape_corr_mean", np.nan)):
        score += 1.0 * (1.0 - out["shape_corr_mean"])
    if np.isfinite(out.get("boundary_fast_frac", np.nan)):
        score += 0.5 * (1.0 - out["boundary_fast_frac"])
    out["quality_score_heuristic"] = float(score) if score != 0.0 else np.nan


    return out
