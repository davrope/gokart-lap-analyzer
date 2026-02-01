from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# We reuse FIT parsing from gps_gate
from .gps_gate import fit_to_gps_speed_df_from_bytes, haversine_m


@dataclass
class DtwTemplateParams:
    # “Hands-off” defaults. UI can show these but you usually won’t touch them.
    min_lap_s: float = field(default=40.0, metadata={"min": 20.0, "max": 180.0, "step": 1.0, "help": "Min lap time considered during period estimate"})
    max_lap_s: float = field(default=120.0, metadata={"min": 30.0, "max": 300.0, "step": 1.0, "help": "Max lap time considered during period estimate"})
    resample_hz: float = field(default=2.0, metadata={"min": 1.0, "max": 10.0, "step": 0.5, "help": "Uniform resample rate (Hz). 2Hz works well for karts"})
    refine_window_s: float = field(default=6.0, metadata={"min": 0.0, "max": 20.0, "step": 1.0, "help": "Boundary refinement window ± seconds"})
    dtw_band_frac: float = field(default=0.08, metadata={"min": 0.0, "max": 0.3, "step": 0.01, "help": "Sakoe-Chiba band as fraction of lap length"})
    use_turn_rate: bool = field(default=True, metadata={"help": "Use turn-rate + speed for template matching"})


# -----------------------------
# Signal helpers
# -----------------------------
def _wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _heading_rad(lat, lon):
    # heading from consecutive points
    lat = np.radians(lat)
    lon = np.radians(lon)
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    # rough bearing in local coords
    return np.arctan2(dlon, dlat)


def _resample_uniform(gps: pd.DataFrame, hz: float) -> pd.DataFrame:
    g = gps.copy()
    g = g.dropna(subset=["timestamp", "lat", "lon", "speed_kmh"]).sort_values("timestamp")
    g = g.reset_index(drop=True)

    t = g["timestamp"]
    t0 = t.iloc[0]
    sec = (t - t0).dt.total_seconds().to_numpy(dtype=float)

    dt = 1.0 / float(hz)
    grid = np.arange(sec[0], sec[-1], dt)

    def interp(col):
        y = g[col].to_numpy(dtype=float)
        return np.interp(grid, sec, y)

    lat = interp("lat")
    lon = interp("lon")
    spd = interp("speed_kmh")

    # turn rate (rad/s)
    hd = _heading_rad(lat, lon)
    # align lengths
    hd = np.concatenate([[hd[0]], hd])
    dh = _wrap_pi(np.diff(hd, prepend=hd[0]))
    turn_rate = dh / dt

    out = pd.DataFrame({
        "t_s": grid,
        "timestamp": (t0 + pd.to_timedelta(grid, unit="s")),
        "lat": lat,
        "lon": lon,
        "speed_kmh": spd,
        "turn_rate": turn_rate,
    })
    return out


def _zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or not np.isfinite(sd):
        return x * 0.0
    return (x - mu) / sd


# -----------------------------
# DTW (multivariate) with band
# -----------------------------
def dtw_distance(X: np.ndarray, Y: np.ndarray, band: int | None = None) -> float:
    """
    X: (n, d), Y: (m, d)
    band: Sakoe-Chiba band half-width in samples (optional)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    m, d2 = Y.shape
    assert d == d2

    INF = 1e18
    dp = np.full((n + 1, m + 1), INF, dtype=float)
    dp[0, 0] = 0.0

    if band is None:
        band = max(n, m)

    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        xi = X[i - 1]
        for j in range(j0, j1 + 1):
            yj = Y[j - 1]
            cost = float(np.sum((xi - yj) ** 2))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(np.sqrt(dp[n, m] / (n + m)))


# -----------------------------
# Period estimate (autocorr)
# -----------------------------
def estimate_lap_period_samples(sig: np.ndarray, min_samp: int, max_samp: int) -> int:
    """
    Return lag (samples) of strongest autocorr peak in [min_samp, max_samp].
    """
    x = np.asarray(sig, dtype=float)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)

    # autocorr via FFT is overkill; direct is fine for a few thousand points
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2:]  # non-negative lags

    lo = max(1, min_samp)
    hi = min(len(ac) - 1, max_samp)
    if hi <= lo:
        return max(1, min_samp)

    # ignore tiny lags; pick max in window
    lag = int(lo + np.argmax(ac[lo:hi + 1]))
    return lag


def _make_features(u: pd.DataFrame, use_turn_rate: bool) -> np.ndarray:
    sp = _zscore(u["speed_kmh"].to_numpy(dtype=float))
    if use_turn_rate:
        tr = _zscore(np.abs(u["turn_rate"].to_numpy(dtype=float)))
        X = np.stack([sp, tr], axis=1)
    else:
        X = sp[:, None]
    return X


def _segments_from_period(u: pd.DataFrame, period_samp: int) -> list[tuple[int, int]]:
    n = len(u)
    segs = []
    i = 0
    while i + period_samp <= n:
        segs.append((i, i + period_samp))
        i += period_samp
    return segs


def _choose_template(u: pd.DataFrame, segs: list[tuple[int, int]], params: DtwTemplateParams) -> int:
    """
    Choose a medoid segment by DTW distance to others (subsampled for speed).
    Return index of segs.
    """
    if len(segs) == 1:
        return 0

    # use up to ~12 segments for template choice
    idxs = np.linspace(0, len(segs) - 1, min(len(segs), 12)).astype(int).tolist()
    feats = []
    for k in idxs:
        a, b = segs[k]
        feats.append(_make_features(u.iloc[a:b], params.use_turn_rate))

    L = len(feats[0])
    band = int(max(1, params.dtw_band_frac * L))

    # pairwise distances
    D = np.zeros((len(idxs), len(idxs)), dtype=float)
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            d = dtw_distance(feats[i], feats[j], band=band)
            D[i, j] = D[j, i] = d

    med = int(np.argmin(D.sum(axis=1)))
    return idxs[med]


def _refine_boundaries(u: pd.DataFrame, segs: list[tuple[int, int]], template: np.ndarray, params: DtwTemplateParams) -> list[int]:
    """
    Refine each boundary between segments by searching ±window around boundary
    to maximize similarity of next lap to template.
    Returns list of boundary sample indices (pass-like events).
    """
    L = template.shape[0]
    band = int(max(1, params.dtw_band_frac * L))
    w = int(round(params.refine_window_s * params.resample_hz))

    boundaries = []
    for (a, b) in segs:
        boundaries.append(a)
    boundaries.append(segs[-1][1])  # end boundary

    refined = [boundaries[0]]
    for k in range(1, len(boundaries) - 1):
        base = boundaries[k]
        best_i = base
        best_d = np.inf

        for shift in range(-w, w + 1):
            i0 = base + shift
            i1 = i0 + L
            if i0 < 0 or i1 > len(u):
                continue
            cand = _make_features(u.iloc[i0:i1], params.use_turn_rate)
            d = dtw_distance(cand, template, band=band)
            if d < best_d:
                best_d = d
                best_i = i0

        refined.append(best_i)

    refined.append(boundaries[-1])
    # ensure strictly increasing
    refined = sorted(set(refined))
    return refined


def passes_to_laps_from_boundaries(u: pd.DataFrame, boundary_idx: list[int]) -> pd.DataFrame:
    if len(boundary_idx) < 2:
        return pd.DataFrame()
    times = u.iloc[boundary_idx]["timestamp"].reset_index(drop=True)
    laps = pd.DataFrame({
        "lap": np.arange(1, len(times)),
        "start": times.iloc[:-1].values,
        "end": times.iloc[1:].values,
    })
    laps["lap_time_s"] = (pd.to_datetime(laps["end"]) - pd.to_datetime(laps["start"])).dt.total_seconds()
    return laps


def summarize_laps(u: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()
    out = []
    for _, r in laps.iterrows():
        seg = u[(u["timestamp"] >= r["start"]) & (u["timestamp"] < r["end"])].copy()
        row = {
            "lap": int(r["lap"]),
            "start": r["start"],
            "end": r["end"],
            "lap_time_s": float(r["lap_time_s"]),
            "avg_speed_kmh": float(seg["speed_kmh"].mean()),
            "max_speed_kmh": float(seg["speed_kmh"].max()),
        }
        out.append(row)
    return pd.DataFrame(out)


def tag_samples(u: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    g = u.copy()
    g["lap"] = pd.NA
    for _, r in laps.iterrows():
        m = (g["timestamp"] >= r["start"]) & (g["timestamp"] < r["end"])
        g.loc[m, "lap"] = int(r["lap"])
    return g


def run(fit_bytes: bytes, params: DtwTemplateParams) -> Dict[str, object]:
    # Parse raw gps
    gps = fit_to_gps_speed_df_from_bytes(fit_bytes)

    # Uniform resample
    u = _resample_uniform(gps, hz=float(params.resample_hz))

    # Signal for period estimation
    sig = _zscore(u["speed_kmh"].to_numpy())
    if params.use_turn_rate:
        sig = sig + 0.7 * _zscore(np.abs(u["turn_rate"].to_numpy()))

    min_samp = int(round(float(params.min_lap_s) * float(params.resample_hz)))
    max_samp = int(round(float(params.max_lap_s) * float(params.resample_hz)))
    period_samp = estimate_lap_period_samples(sig, min_samp=min_samp, max_samp=max_samp)

    # Rough segments
    segs = _segments_from_period(u, period_samp)
    if len(segs) < 2:
        # Not enough data to segment
        laps = pd.DataFrame()
        lap_metrics = pd.DataFrame()
        samples = u.copy()
        return {
            "gps": u,
            "fast_pts": u,            # not used here but keeps schema stable
            "gate_lat": np.nan,       # not used here
            "gate_lon": np.nan,       # not used here
            "dist": np.full(len(u), np.nan),
            "pass_idx": np.array([], dtype=int),
            "dt_s": 1.0 / float(params.resample_hz),
            "laps": laps,
            "lap_metrics": lap_metrics,
            "samples": samples,
            "period_samp": period_samp,
        }

    # Choose template segment
    template_seg_idx = _choose_template(u, segs, params)
    a, b = segs[template_seg_idx]
    template = _make_features(u.iloc[a:b], params.use_turn_rate)

    # Refine boundaries by local DTW search
    boundary_idx = _refine_boundaries(u, segs, template, params)

    # Convert to laps
    laps = passes_to_laps_from_boundaries(u, boundary_idx)
    lap_metrics = summarize_laps(u, laps)
    samples = tag_samples(u, laps)

    # For UI parity: "pass_idx" are boundary indices (start points)
    pass_idx = np.array(boundary_idx[:-1], dtype=int)

    return {
        "gps": u,
        "fast_pts": u,            # placeholder for compatibility
        "gate_lat": np.nan,
        "gate_lon": np.nan,
        "dist": np.full(len(u), np.nan),
        "pass_idx": pass_idx,
        "dt_s": 1.0 / float(params.resample_hz),
        "laps": laps,
        "lap_metrics": lap_metrics,
        "samples": samples,
        "period_samp": period_samp,
    }
