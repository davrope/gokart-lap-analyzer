from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .further import FurtherAnalysisResult


@dataclass
class InsightSummary:
    title: str
    bullets: list[str]
    opportunities: list[str]
    engine: str


class InsightProvider(Protocol):
    def summarize(self, analysis: FurtherAnalysisResult) -> InsightSummary:
        ...


class HeuristicInsightProvider:
    """
    Default local provider.
    This keeps architecture compatible with a future LLM provider.
    """

    def summarize(self, analysis: FurtherAnalysisResult) -> InsightSummary:
        bullets: list[str] = []
        opportunities: list[str] = []

        if analysis.laps_detected:
            bullets.append(f"Detected {analysis.laps_detected} laps, with {analysis.valid_laps} passing robust validity checks.")
        if np.isfinite(analysis.speed_threshold_kmh):
            bullets.append(f"Speed-based inlier threshold used: {analysis.speed_threshold_kmh:.2f} km/h.")
        if np.isfinite(analysis.median_track_width_m):
            bullets.append(f"Estimated median track width: {analysis.median_track_width_m:.2f} m.")
        if analysis.curves_detected:
            bullets.append(f"Detected {analysis.curves_detected} curve segments ({analysis.boundary_curves} near lap boundary).")

        if not analysis.curve_summary.empty:
            costly = analysis.curve_summary.sort_values("mean_time_loss_s", ascending=False).head(2)
            for _, row in costly.iterrows():
                if row["mean_time_loss_s"] > 0.05:
                    opportunities.append(
                        f"Curve {int(row['curve_id'])}: mean loss +{row['mean_time_loss_s']:.2f}s vs best lap."
                    )

            variable = analysis.curve_summary.sort_values("apex_speed_std_kmh", ascending=False).head(1)
            if not variable.empty:
                r = variable.iloc[0]
                opportunities.append(
                    f"Curve {int(r['curve_id'])}: apex speed variability {r['apex_speed_std_kmh']:.2f} km/h; target consistency."
                )

        if not bullets:
            bullets.append("No reliable telemetry summary could be generated from current analysis data.")
        if not opportunities:
            opportunities.append("No standout weak curve identified. Keep iterating on consistent exits.")

        return InsightSummary(
            title="Session Coaching Snapshot",
            bullets=bullets,
            opportunities=opportunities,
            engine="heuristic-local",
        )


class LLMInsightProvider:
    """
    Architecture placeholder for a real LLM integration.

    Future implementation should:
    - Accept a model client (LLM model provider)
    - Build a compact prompt from FurtherAnalysisResult
    - Return InsightSummary in a validated schema
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def summarize(self, analysis: FurtherAnalysisResult) -> InsightSummary:
        raise NotImplementedError(
            "LLM provider is not implemented yet. Use HeuristicInsightProvider for now."
        )


def generate_insight_summary(
    analysis: FurtherAnalysisResult,
    provider: InsightProvider | None = None,
) -> InsightSummary:
    provider = provider or HeuristicInsightProvider()
    return provider.summarize(analysis)


def generate_track_history_summary(
    track_metrics: dict,
    provider: InsightProvider | None = None,
) -> InsightSummary:
    """
    Placeholder entrypoint for future LLM-powered historical coaching summaries.

    For now, this function derives a concise heuristic summary from aggregate
    track metrics while preserving a stable callable interface.
    """
    _ = provider  # reserved for future use with model-backed providers
    bullets: list[str] = []
    opportunities: list[str] = []

    attempts = int(track_metrics.get("attempts_count", 0) or 0)
    curves = int(track_metrics.get("curves_count", 0) or 0)
    best_lap_delta = track_metrics.get("latest_vs_best_delta_s")

    bullets.append(f"Attempts analyzed: {attempts}.")
    bullets.append(f"Curves tracked: {curves}.")

    if isinstance(best_lap_delta, (int, float)) and np.isfinite(float(best_lap_delta)):
        if best_lap_delta <= 0:
            opportunities.append("Latest attempt matched or improved personal best lap pace.")
        else:
            opportunities.append(f"Latest attempt is +{float(best_lap_delta):.2f}s vs personal best lap.")
    else:
        opportunities.append("Insufficient attempt data to compare latest run against personal best.")

    return InsightSummary(
        title="Track History Coaching Snapshot",
        bullets=bullets,
        opportunities=opportunities,
        engine="heuristic-local",
    )
