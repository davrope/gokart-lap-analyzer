from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrackHistoryBundle:
    attempts: pd.DataFrame
    laps: pd.DataFrame
    curves: pd.DataFrame
    trend: pd.DataFrame
    consistency: pd.DataFrame
    heatmap: pd.DataFrame
    latest_vs_best: dict[str, Any]
    opportunities: pd.DataFrame



def build_track_history(attempts: pd.DataFrame, laps: pd.DataFrame, curves: pd.DataFrame) -> TrackHistoryBundle:
    if attempts is None:
        attempts = pd.DataFrame()
    if laps is None:
        laps = pd.DataFrame()
    if curves is None:
        curves = pd.DataFrame()

    attempts = attempts.copy()
    if not attempts.empty:
        attempts["uploaded_at"] = pd.to_datetime(attempts["uploaded_at"], errors="coerce")
        attempts = attempts.sort_values("uploaded_at")

    trend = attempts[["id", "uploaded_at", "best_lap_s"]].copy() if {"id", "uploaded_at", "best_lap_s"}.issubset(attempts.columns) else pd.DataFrame()
    if not trend.empty:
        trend["rolling_median_best_lap_s"] = trend["best_lap_s"].rolling(window=3, min_periods=1).median()

    consistency = attempts[["id", "uploaded_at", "lap_time_cv"]].copy() if {"id", "uploaded_at", "lap_time_cv"}.issubset(attempts.columns) else pd.DataFrame()

    heatmap = pd.DataFrame()
    if not curves.empty and {"attempt_id", "curve_id", "time_loss_vs_best_s"}.issubset(curves.columns):
        heatmap = (
            curves.groupby(["attempt_id", "curve_id"], as_index=False)["time_loss_vs_best_s"]
            .mean()
            .pivot(index="attempt_id", columns="curve_id", values="time_loss_vs_best_s")
        )

    latest_vs_best: dict[str, Any] = {}
    opportunities = pd.DataFrame()

    if not attempts.empty and {"id", "uploaded_at", "best_lap_s", "avg_lap_s"}.issubset(attempts.columns):
        latest = attempts.sort_values("uploaded_at").iloc[-1]
        best = attempts.loc[attempts["best_lap_s"].astype(float).idxmin()]

        latest_vs_best = {
            "latest_attempt_id": latest["id"],
            "best_attempt_id": best["id"],
            "latest_best_lap_s": _f(latest.get("best_lap_s")),
            "best_best_lap_s": _f(best.get("best_lap_s")),
            "delta_best_lap_s": _f(latest.get("best_lap_s")) - _f(best.get("best_lap_s")),
            "latest_avg_lap_s": _f(latest.get("avg_lap_s")),
            "best_avg_lap_s": _f(best.get("avg_lap_s")),
            "delta_avg_lap_s": _f(latest.get("avg_lap_s")) - _f(best.get("avg_lap_s")),
        }

        if not curves.empty and {"attempt_id", "curve_id", "time_loss_vs_best_s", "apex_speed_kmh"}.issubset(curves.columns):
            recent_curve = (
                curves[curves["attempt_id"] == latest["id"]]
                .groupby("curve_id", as_index=False)
                .agg(
                    mean_time_loss_s=("time_loss_vs_best_s", "mean"),
                    apex_speed_mean_kmh=("apex_speed_kmh", "mean"),
                    apex_speed_std_kmh=("apex_speed_kmh", "std"),
                )
                .sort_values("mean_time_loss_s", ascending=False)
            )
            opportunities = recent_curve.head(5)

    return TrackHistoryBundle(
        attempts=attempts,
        laps=laps,
        curves=curves,
        trend=trend,
        consistency=consistency,
        heatmap=heatmap,
        latest_vs_best=latest_vs_best,
        opportunities=opportunities,
    )



def _f(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")
