from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from analysis import build_further_analysis
from lap_methods import default_params, get_methods
from lap_methods.metrics import add_lap_distance_metrics


DEFAULT_METHOD = "GPS Gate (fast-points + distance minima)"



def _build_params(method_name: str, params_json: dict[str, Any] | None = None):
    params = default_params(method_name)
    params_json = params_json or {}
    for k, v in params_json.items():
        if hasattr(params, k):
            setattr(params, k, v)
    return params



def run_attempt_analysis(
    fit_bytes: bytes,
    method_name: str = DEFAULT_METHOD,
    params_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    methods = get_methods()
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")

    params = _build_params(method_name, params_json)
    result = methods[method_name]["runner"](fit_bytes, params)
    result = add_lap_distance_metrics(result)

    context = {
        "method_name": method_name,
        "params": asdict(params),
        "result": result,
    }
    further = build_further_analysis(context)

    return {
        "method_name": method_name,
        "params": asdict(params),
        "result": result,
        "further": further,
    }



def build_attempt_summary_payload(analysis_bundle: dict[str, Any]) -> dict[str, Any]:
    further = analysis_bundle["further"]
    lap_overview = further.lap_overview

    best_lap_s = float("nan")
    avg_lap_s = float("nan")
    lap_time_cv = float("nan")

    if not lap_overview.empty and "lap_time_s" in lap_overview.columns and lap_overview["lap_time_s"].notna().any():
        lap_time = lap_overview["lap_time_s"].to_numpy(dtype=float)
        best_lap_s = float(np.nanmin(lap_time))
        avg_lap_s = float(np.nanmean(lap_time))
        lap_time_cv = float(np.nanstd(lap_time) / max(np.nanmean(lap_time), 1e-9))

    return {
        "status": "processed",
        "processing_error": None,
        "speed_threshold_kmh": float(further.speed_threshold_kmh),
        "laps_detected": int(further.laps_detected),
        "valid_laps": int(further.valid_laps),
        "curves_detected": int(further.curves_detected),
        "boundary_curves": int(further.boundary_curves),
        "median_track_width_m": float(further.median_track_width_m) if np.isfinite(further.median_track_width_m) else None,
        "p90_track_width_m": float(further.p90_track_width_m) if np.isfinite(further.p90_track_width_m) else None,
        "best_lap_s": best_lap_s if np.isfinite(best_lap_s) else None,
        "avg_lap_s": avg_lap_s if np.isfinite(avg_lap_s) else None,
        "lap_time_cv": lap_time_cv if np.isfinite(lap_time_cv) else None,
    }



def build_attempt_lap_rows(attempt_id: str, lap_overview: pd.DataFrame) -> list[dict[str, Any]]:
    if lap_overview is None or lap_overview.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, r in lap_overview.iterrows():
        rows.append(
            {
                "attempt_id": attempt_id,
                "lap": int(r.get("lap")),
                "lap_time_s": _to_float(r.get("lap_time_s")),
                "avg_speed_kmh": _to_float(r.get("avg_speed_kmh")),
                "max_speed_kmh": _to_float(r.get("max_speed_kmh")),
                "lap_distance_m": _to_float(r.get("lap_distance_m")),
                "is_valid_lap": bool(r.get("is_valid_lap")) if "is_valid_lap" in lap_overview.columns else None,
            }
        )
    return rows



def build_attempt_curve_rows(attempt_id: str, curve_overview: pd.DataFrame) -> list[dict[str, Any]]:
    if curve_overview is None or curve_overview.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, r in curve_overview.iterrows():
        rows.append(
            {
                "attempt_id": attempt_id,
                "lap": int(r.get("lap")),
                "curve_id": int(r.get("curve_id")),
                "entry_speed_kmh": _to_float(r.get("entry_speed_kmh")),
                "apex_speed_kmh": _to_float(r.get("apex_speed_kmh")),
                "exit_speed_kmh": _to_float(r.get("exit_speed_kmh")),
                "curve_time_s": _to_float(r.get("curve_time_s")),
                "time_loss_vs_best_s": _to_float(r.get("time_loss_vs_best_s")),
                "s_start_m": _to_float(r.get("s_start_m")),
                "s_end_m": _to_float(r.get("s_end_m")),
                "s_apex_m": _to_float(r.get("s_apex_m")),
                "peak_curvature": _to_float(r.get("peak_curvature")),
            }
        )
    return rows



def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    return f if np.isfinite(f) else None
