from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import cKDTree


R_EARTH_M = 6_371_000.0
CURVE_COLUMNS = [
    "lap",
    "curve_id",
    "entry_speed_kmh",
    "apex_speed_kmh",
    "exit_speed_kmh",
    "curve_time_s",
    "time_loss_vs_best_s",
]
CURVE_SUMMARY_COLUMNS = [
    "curve_id",
    "laps",
    "mean_time_loss_s",
    "p90_time_loss_s",
    "apex_speed_mean_kmh",
    "apex_speed_std_kmh",
    "entry_speed_mean_kmh",
    "exit_speed_mean_kmh",
]
REFERENCE_PATH_COLUMNS = [
    "s_m",
    "x_center_m",
    "y_center_m",
    "x_left_m",
    "y_left_m",
    "x_right_m",
    "y_right_m",
    "x_best_m",
    "y_best_m",
    "lat_center",
    "lon_center",
    "lat_left",
    "lon_left",
    "lat_right",
    "lon_right",
    "lat_best",
    "lon_best",
    "curvature",
]


@dataclass
class FurtherAnalysisResult:
    lap_overview: pd.DataFrame
    curve_overview: pd.DataFrame
    curve_summary: pd.DataFrame
    reference_path: pd.DataFrame
    samples_tagged: pd.DataFrame
    recommendations: list[str]
    warnings: list[str]
    speed_threshold_kmh: float
    laps_detected: int
    valid_laps: int
    reference_lap_id: Optional[int]
    curves_detected: int
    boundary_curves: int
    median_track_width_m: float
    p90_track_width_m: float
    fast_laps: list[int]


def _empty_curve_overview() -> pd.DataFrame:
    return pd.DataFrame(columns=CURVE_COLUMNS)


def _empty_curve_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=CURVE_SUMMARY_COLUMNS)


def _empty_reference_path() -> pd.DataFrame:
    return pd.DataFrame(columns=REFERENCE_PATH_COLUMNS)


def _empty_result(message: str) -> FurtherAnalysisResult:
    return FurtherAnalysisResult(
        lap_overview=pd.DataFrame(),
        curve_overview=_empty_curve_overview(),
        curve_summary=_empty_curve_summary(),
        reference_path=_empty_reference_path(),
        samples_tagged=pd.DataFrame(),
        recommendations=[message],
        warnings=[],
        speed_threshold_kmh=float("nan"),
        laps_detected=0,
        valid_laps=0,
        reference_lap_id=None,
        curves_detected=0,
        boundary_curves=0,
        median_track_width_m=float("nan"),
        p90_track_width_m=float("nan"),
        fast_laps=[],
    )


def _otsu_speed_threshold(speed_kmh: np.ndarray, bins: int = 64, min_floor_kmh: float = 8.0) -> float:
    x = np.asarray(speed_kmh, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float(min_floor_kmh)

    lo, hi = np.percentile(x, [1.0, 99.0])
    xc = x[(x >= lo) & (x <= hi)]
    if len(xc) < 10:
        return float(max(min_floor_kmh, np.median(x)))

    hist, edges = np.histogram(xc, bins=bins, range=(float(xc.min()), float(xc.max())))
    p = hist.astype(float) / max(1.0, float(hist.sum()))
    mids = 0.5 * (edges[:-1] + edges[1:])

    w = np.cumsum(p)
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * w - mu) ** 2 / np.maximum(w * (1.0 - w), 1e-12)
    idx = int(np.nanargmax(sigma_b2))

    return float(max(min_floor_kmh, mids[idx]))


def _robust_inlier_mask(values: np.ndarray, zmax: float = 3.5) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-9:
        return np.ones(len(x), dtype=bool)

    z = 0.6745 * (x - med) / mad
    return np.abs(z) <= float(zmax)


def _haversine_segments_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_M * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _to_local_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: Optional[float] = None,
    lon0: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    if lat0 is None:
        lat0 = float(np.nanmedian(lat))
    if lon0 is None:
        lon0 = float(np.nanmean(lon))

    mlat = 111132.0
    mlon = 111320.0 * np.cos(np.radians(lat0))
    x = (lon - lon0) * mlon
    y = (lat - lat0) * mlat
    return x, y, lat0, lon0, mlat, mlon


def _to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    lat0: float,
    lon0: float,
    mlat: float,
    mlon: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat = y / mlat + lat0
    lon = x / mlon + lon0
    return lat, lon


def _lap_arrays(lap_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lap_df = lap_df.sort_values("timestamp")
    lat = lap_df["lat"].to_numpy(dtype=float)
    lon = lap_df["lon"].to_numpy(dtype=float)
    t = lap_df["timestamp"].astype("int64").to_numpy() / 1e9
    v = lap_df["speed_kmh"].to_numpy(dtype=float)

    seg = _haversine_segments_m(lat, lon)
    s = np.concatenate([[0.0], np.cumsum(seg)])

    keep = np.r_[True, np.diff(s) > 1e-6]
    return lat[keep], lon[keep], s[keep], t[keep], v[keep]


def _build_reference_centerline(
    lap_df: pd.DataFrame,
    step_m: float = 1.0,
    smooth_window_pts: int = 51,
) -> Dict[str, Any]:
    lat, lon, s, _, _ = _lap_arrays(lap_df)
    x, y, lat0, lon0, mlat, mlon = _to_local_xy(lat, lon)

    if len(s) < 10 or s[-1] < 30.0:
        raise ValueError("Insufficient lap points for reference centerline")

    s_u = np.arange(0.0, float(s[-1]), float(step_m))
    x_u = np.interp(s_u, s, x)
    y_u = np.interp(s_u, s, y)

    win = min(int(smooth_window_pts), len(s_u) - 1)
    if win % 2 == 0:
        win -= 1
    win = max(9, win)
    if win >= len(s_u):
        win = len(s_u) - 1 if (len(s_u) - 1) % 2 == 1 else len(s_u) - 2
        win = max(9, win)

    x_s = savgol_filter(x_u, window_length=win, polyorder=3, mode="wrap")
    y_s = savgol_filter(y_u, window_length=win, polyorder=3, mode="wrap")

    tx = np.gradient(x_s, step_m)
    ty = np.gradient(y_s, step_m)
    tnorm = np.hypot(tx, ty)
    tx = tx / np.maximum(tnorm, 1e-9)
    ty = ty / np.maximum(tnorm, 1e-9)

    nx = -ty
    ny = tx

    return {
        "s": s_u,
        "x": x_s,
        "y": y_s,
        "tx": tx,
        "ty": ty,
        "nx": nx,
        "ny": ny,
        "L": float(s_u[-1]),
        "lat0": lat0,
        "lon0": lon0,
        "mlat": mlat,
        "mlon": mlon,
        "step_m": float(step_m),
    }


def _map_points_to_reference(points_df: pd.DataFrame, ref: Dict[str, Any]) -> pd.DataFrame:
    x = (points_df["lon"].to_numpy(dtype=float) - ref["lon0"]) * ref["mlon"]
    y = (points_df["lat"].to_numpy(dtype=float) - ref["lat0"]) * ref["mlat"]

    tree = cKDTree(np.c_[ref["x"], ref["y"]])
    dist, idx = tree.query(np.c_[x, y], k=1)

    vx = x - ref["x"][idx]
    vy = y - ref["y"][idx]
    offset = vx * ref["nx"][idx] + vy * ref["ny"][idx]

    out = points_df.copy()
    out["x_m"] = x
    out["y_m"] = y
    out["ref_idx"] = idx.astype(int)
    out["s_ref_m"] = ref["s"][idx]
    out["offset_m"] = offset
    out["dist_to_ref_m"] = dist
    return out


def _periodic_interp(s_target: np.ndarray, s_known: np.ndarray, v_known: np.ndarray, lap_len_m: float) -> np.ndarray:
    s_ext = np.concatenate([s_known - lap_len_m, s_known, s_known + lap_len_m])
    v_ext = np.concatenate([v_known, v_known, v_known])
    order = np.argsort(s_ext)
    return np.interp(s_target, s_ext[order], v_ext[order])


def _estimate_track_limits(
    mapped_df: pd.DataFrame,
    ref: Dict[str, Any],
    s_bin_m: float = 5.0,
    q_low: float = 0.05,
    q_high: float = 0.95,
    min_points: int = 6,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    bins = np.arange(0.0, ref["L"] + s_bin_m, s_bin_m)
    bidx = np.digitize(mapped_df["s_ref_m"].to_numpy(dtype=float), bins) - 1

    rows: list[dict[str, Any]] = []
    for b in range(len(bins) - 1):
        m = bidx == b
        if int(m.sum()) < int(min_points):
            continue

        off = mapped_df.loc[m, "offset_m"].to_numpy(dtype=float)
        ql, qh = np.quantile(off, [q_low, q_high])
        s_mid = 0.5 * (bins[b] + bins[b + 1])

        rows.append(
            {
                "s_mid_m": float(s_mid),
                "offset_right_m": float(ql),
                "offset_left_m": float(qh),
                "width_m": float(qh - ql),
                "n_points": int(m.sum()),
            }
        )

    limits = pd.DataFrame(rows)
    if limits.empty:
        raise ValueError("Track limit estimation failed: no bins with enough points")

    left_full = _periodic_interp(
        ref["s"],
        limits["s_mid_m"].to_numpy(dtype=float),
        limits["offset_left_m"].to_numpy(dtype=float),
        ref["L"],
    )
    right_full = _periodic_interp(
        ref["s"],
        limits["s_mid_m"].to_numpy(dtype=float),
        limits["offset_right_m"].to_numpy(dtype=float),
        ref["L"],
    )

    return limits, left_full, right_full


def _estimate_best_path_offset(
    mapped_valid_laps: pd.DataFrame,
    ref: Dict[str, Any],
    fast_laps: set[int],
    s_bin_m: float = 5.0,
    min_points: int = 5,
) -> np.ndarray:
    df_fast = mapped_valid_laps[mapped_valid_laps["lap"].astype(int).isin(fast_laps)].copy()
    if df_fast.empty:
        return np.zeros_like(ref["s"])

    bins = np.arange(0.0, ref["L"] + s_bin_m, s_bin_m)
    bidx = np.digitize(df_fast["s_ref_m"].to_numpy(dtype=float), bins) - 1

    rows: list[dict[str, float]] = []
    for b in range(len(bins) - 1):
        m = bidx == b
        if int(m.sum()) < int(min_points):
            continue
        s_mid = 0.5 * (bins[b] + bins[b + 1])
        rows.append(
            {
                "s_mid_m": float(s_mid),
                "offset_best_m": float(np.median(df_fast.loc[m, "offset_m"])),
            }
        )

    best_df = pd.DataFrame(rows)
    if best_df.empty:
        return np.zeros_like(ref["s"])

    return _periodic_interp(
        ref["s"],
        best_df["s_mid_m"].to_numpy(dtype=float),
        best_df["offset_best_m"].to_numpy(dtype=float),
        ref["L"],
    )


def _detect_curves_cyclic(
    ref: Dict[str, Any],
    prominence_quantile: float = 60.0,
    min_peak_distance_m: float = 18.0,
    expand_quantile: float = 50.0,
    min_len_m: float = 10.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    s = ref["s"]
    x = ref["x"]
    y = ref["y"]
    step_m = ref["step_m"]
    n = len(s)

    dx = np.gradient(x, step_m)
    dy = np.gradient(y, step_m)
    d2x = np.gradient(dx, step_m)
    d2y = np.gradient(dy, step_m)

    curvature = np.abs(dx * d2y - dy * d2x) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)

    win = min(51, n - 1)
    if win % 2 == 0:
        win -= 1
    win = max(9, win)
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
        win = max(9, win)

    curvature_s = savgol_filter(curvature, window_length=win, polyorder=2, mode="wrap")

    pad = int(max(30, round(min_peak_distance_m * 2.0 / step_m)))
    ext = np.concatenate([curvature_s[-pad:], curvature_s, curvature_s[:pad]])

    prominence = float(np.percentile(curvature_s, prominence_quantile))
    distance_pts = int(max(8, round(min_peak_distance_m / step_m)))
    peaks_ext, _ = find_peaks(ext, prominence=prominence, distance=distance_pts)
    peaks = ((peaks_ext - pad) % n).astype(int)

    peak_order = sorted(set(peaks.tolist()), key=lambda i: float(curvature_s[i]), reverse=True)
    kept: list[int] = []
    for p in peak_order:
        if all(min((p - k) % n, (k - p) % n) > distance_pts for k in kept):
            kept.append(int(p))
    kept = sorted(kept)

    thr_expand = float(np.percentile(curvature_s, expand_quantile))
    rows: list[dict[str, Any]] = []

    for p in kept:
        l = int(p)
        r = int(p)

        for _ in range(n):
            prev = (l - 1) % n
            if curvature_s[prev] < thr_expand:
                break
            l = prev

        for _ in range(n):
            nxt = (r + 1) % n
            if curvature_s[nxt] < thr_expand:
                break
            r = nxt

        seg_len_pts = (r - l) % n
        if seg_len_pts <= 0:
            seg_len_pts += n
        seg_len_m = seg_len_pts * step_m

        if seg_len_m < min_len_m:
            continue

        rows.append(
            {
                "s_start_m": float(s[l]),
                "s_end_m": float(s[r]),
                "s_apex_m": float(s[p]),
                "segment_len_m": float(seg_len_m),
                "peak_curvature": float(curvature_s[p]),
            }
        )

    curves = pd.DataFrame(rows).sort_values("s_apex_m").reset_index(drop=True)
    if not curves.empty:
        curves["curve_id"] = np.arange(1, len(curves) + 1)

    return curves, curvature_s


def _compute_curve_metrics(
    samples_df: pd.DataFrame,
    valid_laps: set[int],
    curves_df: pd.DataFrame,
    reference_lap_id: int,
    reference_lap_len_m: float,
) -> pd.DataFrame:
    valid_samples = samples_df[samples_df["lap"].isin(valid_laps)].copy()
    if valid_samples.empty or curves_df.empty:
        return _empty_curve_overview()

    best_df = valid_samples[valid_samples["lap"] == reference_lap_id]
    _, _, best_s, best_t, _ = _lap_arrays(best_df)

    rows: list[dict[str, Any]] = []
    for lap_id, lap_df in valid_samples.groupby("lap"):
        _, _, s, t, v = _lap_arrays(lap_df)
        if len(s) < 10 or s[-1] <= 0:
            continue

        scale = float(s[-1] / max(reference_lap_len_m, 1e-6))

        for _, curve in curves_df.iterrows():
            s0_ref = float(curve["s_start_m"])
            s1_ref = float(curve["s_end_m"])

            s0 = min(s0_ref * scale, float(s[-1]))
            s1 = min(s1_ref * scale, float(s[-1]))

            if s1 >= s0:
                seg_mask = (s >= s0) & (s <= s1)
                t0 = float(np.interp(s0, s, t))
                t1 = float(np.interp(s1, s, t))
            else:
                seg_mask = (s >= s0) | (s <= s1)
                t0a = float(np.interp(s0, s, t))
                t1a = float(t[-1])
                t0b = float(t[0])
                t1b = float(np.interp(s1, s, t))
                t0 = t0a
                t1 = (t1a - t0a) + (t1b - t0b) + t0a

            curve_time = float(max(0.0, t1 - t0))

            if s1_ref >= s0_ref:
                b0 = float(np.interp(s0_ref, best_s, best_t))
                b1 = float(np.interp(s1_ref, best_s, best_t))
                best_curve_time = b1 - b0
            else:
                b0a = float(np.interp(s0_ref, best_s, best_t))
                b1a = float(best_t[-1])
                b0b = float(best_t[0])
                b1b = float(np.interp(s1_ref, best_s, best_t))
                best_curve_time = (b1a - b0a) + (b1b - b0b)

            apex_speed = float(np.min(v[seg_mask])) if np.any(seg_mask) else float(np.interp(s0, s, v))

            rows.append(
                {
                    "lap": int(lap_id),
                    "curve_id": int(curve["curve_id"]),
                    "entry_speed_kmh": float(np.interp(s0, s, v)),
                    "apex_speed_kmh": apex_speed,
                    "exit_speed_kmh": float(np.interp(s1, s, v)),
                    "curve_time_s": curve_time,
                    "time_loss_vs_best_s": float(curve_time - best_curve_time),
                }
            )

    if not rows:
        return _empty_curve_overview()

    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def _build_lap_overview(lap_metrics: pd.DataFrame, valid_laps: set[int]) -> pd.DataFrame:
    if lap_metrics is None or lap_metrics.empty:
        return pd.DataFrame()

    base = lap_metrics.copy().sort_values("lap").reset_index(drop=True)
    best_lap_time_s = float(base["lap_time_s"].min()) if "lap_time_s" in base.columns else np.nan

    base["lap_time_delta_s"] = base["lap_time_s"] - best_lap_time_s if np.isfinite(best_lap_time_s) else np.nan
    base["speed_rank"] = base["avg_speed_kmh"].rank(method="min", ascending=False) if "avg_speed_kmh" in base.columns else np.nan
    base["is_valid_lap"] = base["lap"].astype(int).isin(valid_laps)

    cols = [
        c
        for c in [
            "lap",
            "lap_time_s",
            "lap_time_delta_s",
            "avg_speed_kmh",
            "max_speed_kmh",
            "lap_distance_m",
            "lap_distance_km",
            "speed_rank",
            "is_valid_lap",
        ]
        if c in base.columns
    ]
    return base[cols]


def _build_curve_summary(curve_metrics: pd.DataFrame) -> pd.DataFrame:
    if curve_metrics is None or curve_metrics.empty:
        return _empty_curve_summary()

    summary = (
        curve_metrics.groupby("curve_id", as_index=False)
        .agg(
            laps=("lap", "nunique"),
            mean_time_loss_s=("time_loss_vs_best_s", "mean"),
            p90_time_loss_s=("time_loss_vs_best_s", lambda x: np.quantile(x, 0.90)),
            apex_speed_mean_kmh=("apex_speed_kmh", "mean"),
            apex_speed_std_kmh=("apex_speed_kmh", "std"),
            entry_speed_mean_kmh=("entry_speed_kmh", "mean"),
            exit_speed_mean_kmh=("exit_speed_kmh", "mean"),
        )
        .sort_values("mean_time_loss_s", ascending=False)
        .reset_index(drop=True)
    )
    return summary[CURVE_SUMMARY_COLUMNS]


def _build_recommendations(lap_overview: pd.DataFrame, curve_summary: pd.DataFrame) -> list[str]:
    recs: list[str] = []

    if not lap_overview.empty and "lap_time_delta_s" in lap_overview.columns:
        spread = float(lap_overview["lap_time_delta_s"].max())
        if spread > 1.5:
            recs.append("Lap-time spread is high; focus on repeatable braking points and exit timing.")

    if not curve_summary.empty:
        costly = curve_summary[curve_summary["mean_time_loss_s"] > 0.2].head(2)
        for _, row in costly.iterrows():
            recs.append(
                f"Curve {int(row['curve_id'])} is a main time-loss area (mean +{row['mean_time_loss_s']:.2f}s vs best)."
            )

        unstable = curve_summary.sort_values("apex_speed_std_kmh", ascending=False).head(1)
        if not unstable.empty:
            r = unstable.iloc[0]
            recs.append(
                f"Curve {int(r['curve_id'])} has high apex-speed variability ({r['apex_speed_std_kmh']:.2f} km/h); prioritize line consistency."
            )

    if not recs:
        recs.append("Pace is relatively stable; next gain likely comes from curve-by-curve exit optimization.")

    return recs


def _build_reference_path_df(
    ref: Dict[str, Any],
    left_offset_full: np.ndarray,
    right_offset_full: np.ndarray,
    best_offset_full: np.ndarray,
    curvature: np.ndarray,
) -> pd.DataFrame:
    x_left = ref["x"] + left_offset_full * ref["nx"]
    y_left = ref["y"] + left_offset_full * ref["ny"]

    x_right = ref["x"] + right_offset_full * ref["nx"]
    y_right = ref["y"] + right_offset_full * ref["ny"]

    x_best = ref["x"] + best_offset_full * ref["nx"]
    y_best = ref["y"] + best_offset_full * ref["ny"]

    lat_center, lon_center = _to_latlon(ref["x"], ref["y"], ref["lat0"], ref["lon0"], ref["mlat"], ref["mlon"])
    lat_left, lon_left = _to_latlon(x_left, y_left, ref["lat0"], ref["lon0"], ref["mlat"], ref["mlon"])
    lat_right, lon_right = _to_latlon(x_right, y_right, ref["lat0"], ref["lon0"], ref["mlat"], ref["mlon"])
    lat_best, lon_best = _to_latlon(x_best, y_best, ref["lat0"], ref["lon0"], ref["mlat"], ref["mlon"])

    out = pd.DataFrame(
        {
            "s_m": ref["s"],
            "x_center_m": ref["x"],
            "y_center_m": ref["y"],
            "x_left_m": x_left,
            "y_left_m": y_left,
            "x_right_m": x_right,
            "y_right_m": y_right,
            "x_best_m": x_best,
            "y_best_m": y_best,
            "lat_center": lat_center,
            "lon_center": lon_center,
            "lat_left": lat_left,
            "lon_left": lon_left,
            "lat_right": lat_right,
            "lon_right": lon_right,
            "lat_best": lat_best,
            "lon_best": lon_best,
            "curvature": curvature,
        }
    )
    return out[REFERENCE_PATH_COLUMNS]


def build_further_analysis(context: Dict[str, Any]) -> FurtherAnalysisResult:
    result = context.get("result", {}) if isinstance(context, dict) else {}
    samples = result.get("samples", pd.DataFrame())
    lap_metrics = result.get("lap_metrics", pd.DataFrame())

    if samples is None or samples.empty:
        return _empty_result("No analysis samples found. Run the main page analysis first.")

    samples = samples.copy()
    needed = {"timestamp", "lat", "lon", "speed_kmh"}
    if not needed.issubset(samples.columns):
        return _empty_result("Samples are missing required GPS/speed columns for further analysis.")

    samples["timestamp"] = pd.to_datetime(samples["timestamp"], errors="coerce")
    samples["speed_kmh"] = pd.to_numeric(samples["speed_kmh"], errors="coerce")
    samples["lap"] = pd.to_numeric(samples.get("lap", np.nan), errors="coerce")
    samples = samples.dropna(subset=["timestamp", "lat", "lon", "speed_kmh"]).reset_index(drop=True)

    if samples.empty:
        return _empty_result("No valid GPS samples after cleaning.")

    speed_thr = _otsu_speed_threshold(samples["speed_kmh"].to_numpy(dtype=float))
    samples["speed_inlier"] = samples["speed_kmh"] >= speed_thr

    laps_detected = int(samples["lap"].nunique(dropna=True))

    warnings: list[str] = []

    if lap_metrics is None or lap_metrics.empty:
        lap_metrics = pd.DataFrame({"lap": sorted(samples["lap"].dropna().astype(int).unique())})
        lap_metrics["is_valid_lap"] = True
        valid_laps = set(lap_metrics["lap"].astype(int).tolist())
        warnings.append("Lap metrics were missing; using lap labels only for further analysis.")
    else:
        lap_metrics = lap_metrics.copy().sort_values("lap").reset_index(drop=True)
        lap_metrics["lap"] = pd.to_numeric(lap_metrics["lap"], errors="coerce").astype("Int64")

        time_ok = _robust_inlier_mask(lap_metrics["lap_time_s"].to_numpy(dtype=float)) if "lap_time_s" in lap_metrics.columns else np.ones(len(lap_metrics), dtype=bool)

        if "lap_distance_m" in lap_metrics.columns and lap_metrics["lap_distance_m"].notna().any():
            dist_ok = _robust_inlier_mask(lap_metrics["lap_distance_m"].to_numpy(dtype=float))
        else:
            dist_ok = np.ones(len(lap_metrics), dtype=bool)

        lap_metrics["is_valid_lap"] = time_ok & dist_ok
        valid_laps = set(lap_metrics.loc[lap_metrics["is_valid_lap"], "lap"].dropna().astype(int).tolist())

    if not valid_laps and laps_detected > 0:
        valid_laps = set(samples["lap"].dropna().astype(int).tolist())
        warnings.append("No valid laps after robust filtering; fell back to all detected laps.")

    lap_overview = _build_lap_overview(lap_metrics, valid_laps)

    curve_overview = _empty_curve_overview()
    curve_summary = _empty_curve_summary()
    reference_path = _empty_reference_path()
    reference_lap_id: Optional[int] = None
    boundary_curves = 0
    curves_detected = 0
    median_track_width_m = float("nan")
    p90_track_width_m = float("nan")
    fast_laps: list[int] = []

    try:
        valid_lap_metrics = lap_metrics[lap_metrics["lap"].astype("Int64").isin(list(valid_laps))].copy()
        if "lap_time_s" in valid_lap_metrics.columns and valid_lap_metrics["lap_time_s"].notna().any():
            valid_lap_metrics = valid_lap_metrics.sort_values("lap_time_s").reset_index(drop=True)
            reference_lap_id = int(valid_lap_metrics.iloc[0]["lap"])
        elif valid_laps:
            reference_lap_id = int(sorted(valid_laps)[0])

        if reference_lap_id is None:
            raise ValueError("No reference lap available.")

        ref_lap_df = samples[samples["lap"] == reference_lap_id].copy()
        ref = _build_reference_centerline(ref_lap_df, step_m=1.0, smooth_window_pts=51)

        track_points = samples[samples["speed_inlier"]].copy()
        mapped_track = _map_points_to_reference(track_points, ref)

        limits_df, left_offset_full, right_offset_full = _estimate_track_limits(
            mapped_track,
            ref,
            s_bin_m=5.0,
            q_low=0.05,
            q_high=0.95,
            min_points=6,
        )

        median_track_width_m = float(limits_df["width_m"].median())
        p90_track_width_m = float(limits_df["width_m"].quantile(0.90))

        mapped_valid = _map_points_to_reference(samples[samples["lap"].isin(valid_laps)].copy(), ref)

        if "lap_time_s" in valid_lap_metrics.columns and valid_lap_metrics["lap_time_s"].notna().any():
            fast_q = float(valid_lap_metrics["lap_time_s"].quantile(0.25))
            fast_laps = sorted(valid_lap_metrics[valid_lap_metrics["lap_time_s"] <= fast_q]["lap"].dropna().astype(int).tolist())
        if not fast_laps and reference_lap_id is not None:
            fast_laps = [int(reference_lap_id)]

        best_offset_full = _estimate_best_path_offset(mapped_valid, ref, set(fast_laps), s_bin_m=5.0, min_points=5)
        best_offset_full = np.clip(best_offset_full, right_offset_full, left_offset_full)

        curves_df, curvature = _detect_curves_cyclic(
            ref,
            prominence_quantile=60.0,
            min_peak_distance_m=18.0,
            expand_quantile=50.0,
            min_len_m=10.0,
        )

        curves_detected = int(len(curves_df))
        if curves_detected > 0:
            boundary_curves = int(
                (
                    (curves_df["s_apex_m"] < 0.12 * ref["L"]) |
                    (curves_df["s_apex_m"] > 0.88 * ref["L"])
                ).sum()
            )

        curve_overview = _compute_curve_metrics(
            samples_df=samples,
            valid_laps=valid_laps,
            curves_df=curves_df,
            reference_lap_id=int(reference_lap_id),
            reference_lap_len_m=float(ref["L"]),
        )
        if not curve_overview.empty and not curves_df.empty:
            curve_overview = curve_overview.merge(
                curves_df[["curve_id", "s_start_m", "s_end_m", "s_apex_m", "peak_curvature"]],
                on="curve_id",
                how="left",
            )
        curve_summary = _build_curve_summary(curve_overview)

        reference_path = _build_reference_path_df(
            ref,
            left_offset_full=left_offset_full,
            right_offset_full=right_offset_full,
            best_offset_full=best_offset_full,
            curvature=curvature,
        )

    except Exception as exc:
        warnings.append(f"Advanced curve pipeline fallback: {exc}")

    recommendations = _build_recommendations(lap_overview, curve_summary)

    return FurtherAnalysisResult(
        lap_overview=lap_overview,
        curve_overview=curve_overview,
        curve_summary=curve_summary,
        reference_path=reference_path,
        samples_tagged=samples,
        recommendations=recommendations,
        warnings=warnings,
        speed_threshold_kmh=float(speed_thr),
        laps_detected=laps_detected,
        valid_laps=int(len(valid_laps)),
        reference_lap_id=reference_lap_id,
        curves_detected=curves_detected,
        boundary_curves=boundary_curves,
        median_track_width_m=median_track_width_m,
        p90_track_width_m=p90_track_width_m,
        fast_laps=fast_laps,
    )
