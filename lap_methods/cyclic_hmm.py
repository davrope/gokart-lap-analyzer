# lap_methods/cyclic_hmm.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .gps_gate import fit_to_gps_speed_df_from_bytes, haversine_m, passes_to_laps, tag_samples


@dataclass
class CyclicHMMParams:
    # Model size
    n_phase_states: int = field(default=16, metadata={"min": 6, "max": 40, "step": 1})
    # PIT state is always 1 extra (state 0)

    # Feature engineering / resampling
    resample_hz: float = field(default=2.0, metadata={"min": 1.0, "max": 10.0, "step": 0.5})
    use_turn_rate: bool = field(default=True)
    use_accel: bool = field(default=True)

    # PIT heuristics (initialization + mild regularization)
    pit_speed_kmh: float = field(default=12.0, metadata={"min": 0.0, "max": 60.0, "step": 1.0})
    speed_for_gate: float = field(default=18.0, metadata={"min": 0.0, "max": 80.0, "step": 1.0})  # for visuals/plots

    # Lap plausibility constraints
    min_lap_s: float = field(default=50.0, metadata={"min": 10.0, "max": 240.0, "step": 1.0})
    expected_laps: int = field(default=15, metadata={"min": 1, "max": 200, "step": 1})

    # Training
    n_iter: int = field(default=25, metadata={"min": 5, "max": 200, "step": 1})
    seed: int = field(default=7, metadata={"min": 0, "max": 10_000, "step": 1})
    cov_floor: float = field(default=1e-3, metadata={"min": 1e-6, "max": 1e-1, "step": 1e-3})

    # Transition strength (higher => more strictly cyclic)
    stay_prob: float = field(default=0.70, metadata={"min": 0.40, "max": 0.95, "step": 0.01})
    step_prob: float = field(default=0.28, metadata={"min": 0.03, "max": 0.55, "step": 0.01})
    skip_prob: float = field(default=0.02, metadata={"min": 0.00, "max": 0.10, "step": 0.01})


# -------------------------
# Resample + features
# -------------------------
def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _heading_rad(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.radians(lat)
    lon = np.radians(lon)
    dlon = np.diff(lon, prepend=lon[0])
    dlat = np.diff(lat, prepend=lat[0])
    hd = np.arctan2(dlon, dlat)
    if len(hd) > 1:
        hd[0] = hd[1]
    return hd


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
    turn_rate = dh / dt  # rad/s

    accel = np.diff(spd, prepend=spd[0]) / dt  # (km/h)/s

    return pd.DataFrame({
        "t_s": grid,
        "timestamp": t0 + pd.to_timedelta(grid, unit="s"),
        "lat": lat,
        "lon": lon,
        "speed_kmh": spd,
        "turn_rate": turn_rate,
        "accel": accel,
    })


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def build_features(u: pd.DataFrame, p: CyclicHMMParams) -> np.ndarray:
    feats = []
    feats.append(_z(u["speed_kmh"].to_numpy(dtype=float)))
    if p.use_turn_rate:
        feats.append(_z(np.abs(u["turn_rate"].to_numpy(dtype=float))))
    if p.use_accel:
        feats.append(_z(u["accel"].to_numpy(dtype=float)))
    X = np.vstack(feats).T
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# -------------------------
# Gaussian HMM (diag cov), constrained transitions
# -------------------------
def _log_gauss_diag(X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    # X: (T,D), mean/var: (D,)
    D = X.shape[1]
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.sum(np.log(2*np.pi*var)) + np.sum((X-mean)**2 / var, axis=1))


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def forward_backward(logB: np.ndarray, logA: np.ndarray, logpi: np.ndarray):
    """
    logB: (T,N) emission log probs
    logA: (N,N) transition log probs
    logpi: (N,) initial log probs
    Returns gamma (T,N), xi_sum (N,N), ll
    """
    T, N = logB.shape
    log_alpha = np.zeros((T, N))
    log_beta = np.zeros((T, N))

    log_alpha[0] = logpi + logB[0]
    for t in range(1, T):
        log_alpha[t] = logB[t] + _logsumexp(log_alpha[t-1][:, None] + logA, axis=0)

    ll = _logsumexp(log_alpha[T-1], axis=0)

    log_beta[T-1] = 0.0
    for t in range(T-2, -1, -1):
        log_beta[t] = _logsumexp(logA + logB[t+1][None, :] + log_beta[t+1][None, :], axis=1)

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)

    # xi summed over time
    xi_sum = np.zeros((N, N), dtype=float)
    for t in range(T-1):
        log_xi = (log_alpha[t][:, None] + logA + logB[t+1][None, :] + log_beta[t+1][None, :]) - ll
        xi_sum += np.exp(log_xi)

    return gamma, xi_sum, float(ll)


def viterbi(logB: np.ndarray, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
    T, N = logB.shape
    dp = np.zeros((T, N))
    back = np.zeros((T, N), dtype=int)

    dp[0] = logpi + logB[0]
    for t in range(1, T):
        scores = dp[t-1][:, None] + logA
        back[t] = np.argmax(scores, axis=0)
        dp[t] = logB[t] + np.max(scores, axis=0)

    path = np.zeros(T, dtype=int)
    path[T-1] = int(np.argmax(dp[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]
    return path


def make_cyclic_transition_matrix(K: int, pit_self: float, stay: float, step: float, skip: float) -> np.ndarray:
    """
    States: 0 = PIT, 1..K = phase states
    PIT mostly self-loop, can go to phase1.
    Phase i can stay, step to i+1, or skip to i+2 (wrap).
    """
    N = K + 1
    A = np.zeros((N, N), dtype=float)

    # PIT (0)
    A[0, 0] = pit_self
    A[0, 1] = 1.0 - pit_self  # enter lap

    # phases
    for i in range(1, K+1):
        A[i, i] = stay
        nxt = 1 if i == K else i + 1
        A[i, nxt] = step
        if skip > 0:
            nxt2 = 1 if i >= K-1 else i + 2
            A[i, nxt2] += skip

        # allow small chance of falling into PIT (handles slowdowns / transitions)
        A[i, 0] = max(1e-6, 1.0 - np.sum(A[i]))

    # normalize
    A = A / A.sum(axis=1, keepdims=True)
    return A


def fit_cyclic_hmm(X: np.ndarray, pit_mask: np.ndarray, p: CyclicHMMParams):
    """
    EM for constrained cyclic HMM with diag Gaussians.
    pit_mask: True when likely pit/transition (speed low) used for init only.
    """
    rng = np.random.default_rng(int(p.seed))
    T, D = X.shape
    K = int(p.n_phase_states)
    N = K + 1

    # Init transition
    A = make_cyclic_transition_matrix(
        K=K,
        pit_self=0.97,
        stay=float(p.stay_prob),
        step=float(p.step_prob),
        skip=float(p.skip_prob),
    )

    # Init pi: start likely PIT or phase1 depending on first seconds
    pi = np.zeros(N, dtype=float)
    pi[0] = 0.7
    pi[1] = 0.3
    pi /= pi.sum()

    # Init emissions
    means = np.zeros((N, D), dtype=float)
    vars_ = np.ones((N, D), dtype=float)

    # PIT state (0): mean/var from pit_mask if enough, else from slowest 20%
    if pit_mask.sum() >= 10:
        Xpit = X[pit_mask]
    else:
        idx = np.argsort(X[:, 0])[: max(10, int(0.2 * T))]
        Xpit = X[idx]
    means[0] = Xpit.mean(axis=0)
    vars_[0] = Xpit.var(axis=0) + float(p.cov_floor)

    # Phase states: take non-pit points and seed means along time (rough phase)
    nonpit = ~pit_mask
    Xnp = X[nonpit]
    if len(Xnp) < max(20, 3*K):
        Xnp = X  # fallback

    # deterministic time-based seeding: split into K chunks
    idx_np = np.where(nonpit)[0]
    if len(idx_np) < 2*K:
        idx_np = np.arange(T)

    chunks = np.array_split(idx_np, K)
    for i in range(1, K+1):
        ci = chunks[i-1]
        if len(ci) == 0:
            ci = rng.integers(0, T, size=20)
        Xi = X[ci]
        means[i] = Xi.mean(axis=0)
        vars_[i] = Xi.var(axis=0) + float(p.cov_floor)

    # EM loop
    for it in range(int(p.n_iter)):
        # emissions logB
        logB = np.zeros((T, N), dtype=float)
        for j in range(N):
            logB[:, j] = _log_gauss_diag(X, means[j], vars_[j])

        logA = np.log(np.maximum(A, 1e-300))
        logpi = np.log(np.maximum(pi, 1e-300))

        gamma, xi_sum, ll = forward_backward(logB, logA, logpi)

        # M-step: update pi
        pi = gamma[0] + 1e-9
        pi /= pi.sum()

        # update A (keep cyclic structure but allow learning within it)
        A_new = xi_sum + 1e-9
        A_new /= A_new.sum(axis=1, keepdims=True)

        # Softly pull A towards cyclic prior to keep phase stable
        A_prior = make_cyclic_transition_matrix(
            K=K,
            pit_self=0.97,
            stay=float(p.stay_prob),
            step=float(p.step_prob),
            skip=float(p.skip_prob),
        )
        A = 0.7 * A_new + 0.3 * A_prior
        A /= A.sum(axis=1, keepdims=True)

        # update emissions
        for j in range(N):
            w = gamma[:, j][:, None]  # (T,1)
            wsum = float(np.sum(w))
            if wsum <= 1e-8:
                continue
            m = np.sum(w * X, axis=0) / wsum
            v = np.sum(w * (X - m) ** 2, axis=0) / wsum
            means[j] = m
            vars_[j] = np.maximum(v, float(p.cov_floor))

    # final decode
    logB = np.zeros((T, N), dtype=float)
    for j in range(N):
        logB[:, j] = _log_gauss_diag(X, means[j], vars_[j])
    logA = np.log(np.maximum(A, 1e-300))
    logpi = np.log(np.maximum(pi, 1e-300))
    path = viterbi(logB, logA, logpi)

    return dict(A=A, pi=pi, means=means, vars=vars_, path=path)


# -------------------------
# Boundary extraction + visual "gate"
# -------------------------
def phase_wrap_pass_idx(path: np.ndarray, K: int, min_gap_samples: int) -> np.ndarray:
    """
    Pass = transition from phase K to phase 1. Ignore PIT (0).
    """
    idx = []
    last = -10**12
    for t in range(1, len(path)):
        if t - last < min_gap_samples:
            continue
        if path[t-1] == K and path[t] == 1:
            idx.append(t)
            last = t
    return np.array(idx, dtype=int)


def derive_visual_gate(gps: pd.DataFrame, pass_idx: np.ndarray) -> Tuple[float, float]:
    if len(pass_idx) == 0:
        return (float(np.nanmedian(gps["lat"])), float(np.nanmedian(gps["lon"])))
    lat = gps.iloc[pass_idx]["lat"].to_numpy(dtype=float)
    lon = gps.iloc[pass_idx]["lon"].to_numpy(dtype=float)
    return (float(np.nanmedian(lat)), float(np.nanmedian(lon)))


# -------------------------
# Runner (method entrypoint)
# -------------------------
def run(fit_bytes: bytes, params: CyclicHMMParams) -> Dict[str, object]:
    gps0 = fit_to_gps_speed_df_from_bytes(fit_bytes)
    u = _resample_uniform(gps0, hz=float(params.resample_hz))
    X = build_features(u, params)

    pit_mask = u["speed_kmh"].to_numpy(dtype=float) < float(params.pit_speed_kmh)

    model = fit_cyclic_hmm(X, pit_mask=pit_mask, p=params)
    path = model["path"]  # 0..K

    fs = float(params.resample_hz)
    min_gap_samples = max(1, int(round(float(params.min_lap_s) * fs)))
    pass_idx = phase_wrap_pass_idx(path, K=int(params.n_phase_states), min_gap_samples=min_gap_samples)

    # Build laps based on pass times, but ignore PIT-only segments (naturally handled by state sequence)
    laps = passes_to_laps(u, pass_idx)

    # Post-hoc "visual gate" for diagnostics/plots
    gate_lat, gate_lon = derive_visual_gate(u, pass_idx)
    dist = haversine_m(u["lat"].to_numpy(dtype=float), u["lon"].to_numpy(dtype=float), gate_lat, gate_lon)

    # fast points for your track plot layer
    fast_pts = u[u["speed_kmh"] >= float(params.speed_for_gate)].copy()

    # Minimal lap metrics; your app can enrich more
    lap_metrics = laps.copy() if not laps.empty else pd.DataFrame()
    samples = tag_samples(u.assign(hmm_state=path), laps, dist)

    return {
        "gps": u,
        "fast_pts": fast_pts,
        "gate_lat": float(gate_lat),
        "gate_lon": float(gate_lon),
        "dist": dist,
        "pass_idx": pass_idx,
        "dt_s": 1.0 / fs,
        "laps": laps,
        "lap_metrics": lap_metrics,
        "samples": samples,
        # debug
        "hmm_states": path,
        "hmm_A": model["A"],
        "hmm_means": model["means"],
        "hmm_vars": model["vars"],
    }
