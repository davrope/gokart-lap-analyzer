# lap_methods/spectral_phase.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .gps_gate import fit_to_gps_speed_df_from_bytes  # reuse your FIT parser


@dataclass
class SpectralPhaseParams:
    # Search window for lap period
    min_lap_s: float = field(default=40.0, metadata={"min": 20.0, "max": 180.0, "step": 1.0})
    max_lap_s: float = field(default=120.0, metadata={"min": 30.0, "max": 300.0, "step": 1.0})

    # Resampling (uniform time grid improves spectral stability)
    resample_hz: float = field(default=2.0, metadata={"min": 1.0, "max": 10.0, "step": 0.5})

    # Signal composition
    use_turn_rate: bool = field(default=True, metadata={"help": "Use |turn_rate| alongside speed"})
    turn_weight: float = field(default=0.7, metadata={"min": 0.0, "max": 2.0, "step": 0.1})

    # Phase anchoring search
    n_phase_candidates: int = field(default=16, metadata={"min": 4, "max": 64, "step": 1})
    min_boundary_speed_kmh: float = field(default=15.0, metadata={"min": 0.0, "max": 80.0, "step": 1.0})

    # Filtering around dominant frequency (fractional bandwidth)
    band_frac: float = field(default=0.25, metadata={"min": 0.05, "max": 0.6, "step": 0.05})

    # Boundary detection
    min_lap_gap_s: float = field(default=45.0, metadata={"min": 10.0, "max": 180.0, "step": 1.0})


# -----------------------
# Utilities
# -----------------------
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _meters_per_deg(lat_deg: float) -> Tuple[float, float]:
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_deg))
    return m_per_deg_lat, m_per_deg_lon


def _heading_rad(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.radians(lat)
    lon = np.radians(lon)
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    hd = np.arctan2(dlon, dlat)  # rough bearing in local coords
    return np.concatenate([[hd[0]], hd])


def _resample_uniform(gps: pd.DataFrame, hz: float) -> pd.DataFrame:
    g = gps.dropna(subset=["timestamp", "lat", "lon", "speed_kmh"]).sort_values("timestamp").reset_index(drop=True)
    t0 = g["timestamp"].iloc[0]
    sec = (g["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=float)

    dt = 1.0 / float(hz)
    grid = np.arange(sec[0], sec[-1], dt)

    def interp(col):
        y = g[col].to_numpy(dtype=float)
        return np.interp(grid, sec, y)

    lat = interp("lat")
    lon = interp("lon")
    spd = interp("speed_kmh")

    hd = _heading_rad(lat, lon)
    dh = _wrap_pi(np.diff(hd, prepend=hd[0]))
    turn_rate = dh / dt

    return pd.DataFrame({
        "t_s": grid,
        "timestamp": t0 + pd.to_timedelta(grid, unit="s"),
        "lat": lat,
        "lon": lon,
        "speed_kmh": spd,
        "turn_rate": turn_rate,
    })


def hilbert_analytic(x: np.ndarray) -> np.ndarray:
    """
    Analytic signal via FFT-based Hilbert transform (no scipy needed).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)


def bandpass_fft(x: np.ndarray, fs: float, f0: float, band_frac: float) -> np.ndarray:
    """
    Simple FFT-domain bandpass around f0. Keeps frequencies in [f0*(1-band), f0*(1+band)].
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    lo = max(0.0, f0 * (1.0 - band_frac))
    hi = f0 * (1.0 + band_frac)

    mask = (freqs >= lo) & (freqs <= hi)
    Xf = X * mask
    y = np.fft.irfft(Xf, n=n)
    return y


def dominant_freq(sig: np.ndarray, fs: float, min_lap_s: float, max_lap_s: float) -> float:
    """
    Find dominant frequency (Hz) in band corresponding to [min_lap_s, max_lap_s].
    """
    x = np.asarray(sig, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    X = np.fft.rfft(x)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    f_lo = 1.0 / float(max_lap_s)
    f_hi = 1.0 / float(min_lap_s)

    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return float((f_lo + f_hi) / 2.0)

    k = np.argmax(P[band])
    return float(freqs[band][k])


def detect_phase_crossings(u: pd.DataFrame, phase: np.ndarray, target: float, min_gap_s: float, fs: float) -> np.ndarray:
    """
    Indices where wrapped phase crosses target (with positive direction).
    """
    ph = _wrap_pi(phase)
    d = _wrap_pi(ph - target)

    # Crossing when d changes sign from negative to positive
    cross = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1

    # Enforce min gap
    min_samp = int(round(min_gap_s * fs))
    keep = []
    last = -10**12
    for idx in cross:
        if idx - last >= min_samp:
            keep.append(idx)
            last = idx
    return np.array(keep, dtype=int)


def score_boundaries(u: pd.DataFrame, idx: np.ndarray, min_speed_kmh: float) -> float:
    """
    Lower score is better.
    Combines:
    - boundary spatial spread
    - penalty for low-speed boundaries
    - lap-time variability
    """
    if len(idx) < 3:
        return 1e9

    b = u.iloc[idx][["lat", "lon", "speed_kmh", "timestamp"]].copy()
    # speed penalty
    speed = b["speed_kmh"].to_numpy(dtype=float)
    low_speed_frac = float(np.mean(speed < min_speed_kmh))
    speed_penalty = 10.0 * low_speed_frac

    # spatial spread (meters)
    lat0 = float(np.median(b["lat"]))
    mlat, mlon = _meters_per_deg(lat0)
    x = (b["lon"].to_numpy(dtype=float) - float(np.mean(b["lon"]))) * mlon
    y = (b["lat"].to_numpy(dtype=float) - float(np.mean(b["lat"]))) * mlat
    spread_m = float(np.sqrt(np.var(x) + np.var(y)))

    # lap-time CV from crossings
    t = b["timestamp"].to_numpy()
    lap_s = np.diff(t) / np.timedelta64(1, "s")
    mean = float(np.mean(lap_s))
    std = float(np.std(lap_s, ddof=1)) if len(lap_s) > 1 else 0.0
    cv = float(std / mean) if mean > 0 else 1.0

    return spread_m * 0.02 + cv * 2.0 + speed_penalty


def boundaries_to_laps(u: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    if len(idx) < 2:
        return pd.DataFrame()
    times = u.iloc[idx]["timestamp"].reset_index(drop=True)
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
        seg = u[(u["timestamp"] >= r["start"]) & (u["timestamp"] < r["end"])]
        out.append({
            "lap": int(r["lap"]),
            "start": r["start"],
            "end": r["end"],
            "lap_time_s": float(r["lap_time_s"]),
            "avg_speed_kmh": float(seg["speed_kmh"].mean()),
            "max_speed_kmh": float(seg["speed_kmh"].max()),
        })
    return pd.DataFrame(out)


def tag_samples(u: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    g = u.copy()
    g["lap"] = pd.NA
    for _, r in laps.iterrows():
        m = (g["timestamp"] >= r["start"]) & (g["timestamp"] < r["end"])
        g.loc[m, "lap"] = int(r["lap"])
    return g


# -----------------------
# Runner
# -----------------------
def run(fit_bytes: bytes, params: SpectralPhaseParams) -> Dict[str, object]:
    gps = fit_to_gps_speed_df_from_bytes(fit_bytes)
    u = _resample_uniform(gps, hz=float(params.resample_hz))
    fs = float(params.resample_hz)

    # Build composite periodic signal
    sig = _zscore(u["speed_kmh"].to_numpy(dtype=float))
    if params.use_turn_rate:
        sig = sig + float(params.turn_weight) * _zscore(np.abs(u["turn_rate"].to_numpy(dtype=float)))

    # Find dominant lap frequency
    f0 = dominant_freq(sig, fs=fs, min_lap_s=float(params.min_lap_s), max_lap_s=float(params.max_lap_s))

    # Band-limit around f0 and compute phase
    sig_f = bandpass_fft(sig, fs=fs, f0=f0, band_frac=float(params.band_frac))
    analytic = hilbert_analytic(sig_f)
    phase = np.unwrap(np.angle(analytic))

    # Try multiple phase offsets; pick best by spatial/speed/consistency score
    best = {"score": 1e18, "idx": np.array([], dtype=int), "target": 0.0}

    candidates = np.linspace(-np.pi, np.pi, int(params.n_phase_candidates), endpoint=False)
    for target in candidates:
        idx = detect_phase_crossings(
            u, phase=phase, target=float(target),
            min_gap_s=float(params.min_lap_gap_s), fs=fs
        )
        sc = score_boundaries(u, idx, min_speed_kmh=float(params.min_boundary_speed_kmh))
        if sc < best["score"]:
            best = {"score": sc, "idx": idx, "target": float(target)}

    pass_idx = best["idx"]
    laps = boundaries_to_laps(u, pass_idx)
    lap_metrics = summarize_laps(u, laps)
    samples = tag_samples(u, laps)

    # Keep schema compatible with your app; no gate/dist here
    return {
        "gps": u,
        "fast_pts": u,                # placeholder
        "gate_lat": np.nan,
        "gate_lon": np.nan,
        "dist": np.full(len(u), np.nan),
        "pass_idx": pass_idx,
        "dt_s": 1.0 / fs,
        "laps": laps,
        "lap_metrics": lap_metrics,
        "samples": samples,
        # extras for debugging / plots if you want later
        "f0_hz": f0,
        "phase_target": best["target"],
        "phase_signal": sig_f,
    }
