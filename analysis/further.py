from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class FurtherAnalysisResult:
    lap_overview: pd.DataFrame
    curve_overview: pd.DataFrame
    recommendations: List[str]


def _build_lap_overview(lap_metrics: pd.DataFrame) -> pd.DataFrame:
    if lap_metrics is None or lap_metrics.empty:
        return pd.DataFrame()

    base = lap_metrics.copy().sort_values("lap").reset_index(drop=True)
    best_lap_time_s = float(base["lap_time_s"].min()) if "lap_time_s" in base.columns else np.nan
    base["lap_time_delta_s"] = base["lap_time_s"] - best_lap_time_s if np.isfinite(best_lap_time_s) else np.nan
    base["speed_rank"] = base["avg_speed_kmh"].rank(method="min", ascending=False) if "avg_speed_kmh" in base.columns else np.nan

    cols = [c for c in ["lap", "lap_time_s", "lap_time_delta_s", "avg_speed_kmh", "max_speed_kmh", "lap_distance_m", "speed_rank"] if c in base.columns]
    return base[cols]


def _build_curve_overview() -> pd.DataFrame:
    # Placeholder schema for future curve extraction, apex speed, and line analysis.
    return pd.DataFrame(
        columns=[
            "lap",
            "curve_id",
            "entry_speed_kmh",
            "apex_speed_kmh",
            "exit_speed_kmh",
            "time_loss_vs_best_s",
        ]
    )


def _build_recommendations(lap_overview: pd.DataFrame) -> List[str]:
    if lap_overview.empty:
        return ["No laps detected yet. Run an analysis with valid laps to unlock recommendations."]

    recs: List[str] = []
    if "lap_time_delta_s" in lap_overview.columns and lap_overview["lap_time_delta_s"].notna().any():
        spread = float(lap_overview["lap_time_delta_s"].max())
        if spread > 1.5:
            recs.append("Consistency opportunity: focus on repeating your best lap line and braking markers.")

    if "avg_speed_kmh" in lap_overview.columns and lap_overview["avg_speed_kmh"].notna().any():
        speed_cv = float(lap_overview["avg_speed_kmh"].std(ddof=0) / max(1e-9, lap_overview["avg_speed_kmh"].mean()))
        if speed_cv > 0.04:
            recs.append("Average speed varies across laps; prioritize cleaner exits to stabilize pace.")

    if not recs:
        recs.append("Pace looks stable. Next step: identify curve-by-curve apex speed differences.")

    return recs


def build_further_analysis(context: Dict[str, Any]) -> FurtherAnalysisResult:
    result = context.get("result", {})
    lap_metrics = result.get("lap_metrics", pd.DataFrame())

    lap_overview = _build_lap_overview(lap_metrics)
    curve_overview = _build_curve_overview()
    recommendations = _build_recommendations(lap_overview)

    return FurtherAnalysisResult(
        lap_overview=lap_overview,
        curve_overview=curve_overview,
        recommendations=recommendations,
    )

