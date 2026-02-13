from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import build_further_analysis
from lap_methods.gps_gate import GpsGateParams, run as run_gps_gate
from lap_methods.metrics import add_lap_distance_metrics


FIT_PATH_ENV = "FIT_TEST_FILE"


def _resolve_fit_path() -> Path:
    fit_path_raw = os.environ.get(FIT_PATH_ENV, "").strip()
    if not fit_path_raw:
        raise RuntimeError(
            f"Environment variable {FIT_PATH_ENV} is required for FIT-based tests."
        )

    fit_path = Path(fit_path_raw).expanduser()
    if not fit_path.is_absolute():
        fit_path = (Path.cwd() / fit_path).resolve()

    if not fit_path.exists():
        raise FileNotFoundError(
            f"{FIT_PATH_ENV} points to a missing file: {fit_path}"
        )
    if fit_path.suffix.lower() != ".fit":
        raise ValueError(f"{FIT_PATH_ENV} must point to a .fit file: {fit_path}")

    return fit_path


class FitPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fit_path = _resolve_fit_path()
        cls.fit_bytes = cls.fit_path.read_bytes()
        cls.params = GpsGateParams()

    def test_gps_gate_run_contract(self) -> None:
        result = run_gps_gate(self.fit_bytes, self.params)

        expected_keys = {
            "gps",
            "fast_pts",
            "gate_lat",
            "gate_lon",
            "dist",
            "pass_idx",
            "dt_s",
            "laps",
            "lap_metrics",
            "samples",
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

        gps = result["gps"]
        dist = result["dist"]
        pass_idx = result["pass_idx"]
        laps = result["laps"]
        lap_metrics = result["lap_metrics"]
        samples = result["samples"]

        self.assertIsInstance(gps, pd.DataFrame)
        self.assertGreater(len(gps), 0)
        self.assertEqual(len(dist), len(gps))
        self.assertEqual(len(samples), len(gps))
        self.assertGreater(float(result["dt_s"]), 0.0)
        self.assertTrue(np.isfinite(float(result["gate_lat"])))
        self.assertTrue(np.isfinite(float(result["gate_lon"])))

        self.assertTrue(np.all(np.diff(pass_idx) >= 0))
        if len(pass_idx) > 0:
            self.assertGreaterEqual(int(pass_idx.min()), 0)
            self.assertLess(int(pass_idx.max()), len(gps))

        self.assertIsInstance(laps, pd.DataFrame)
        self.assertIsInstance(lap_metrics, pd.DataFrame)
        if not laps.empty:
            self.assertTrue((laps["lap_time_s"] > 0).all())

    def test_distance_metrics_contract(self) -> None:
        result = run_gps_gate(self.fit_bytes, self.params)
        enriched = add_lap_distance_metrics(result)
        lap_metrics = enriched["lap_metrics"]

        if lap_metrics.empty:
            self.skipTest("No laps detected for this FIT file/params.")

        self.assertIn("lap_distance_m", lap_metrics.columns)
        self.assertIn("lap_distance_km", lap_metrics.columns)
        self.assertTrue((lap_metrics["lap_distance_m"] > 0).all())
        self.assertTrue((lap_metrics["lap_distance_km"] > 0).all())

    def test_further_analysis_contract(self) -> None:
        result = run_gps_gate(self.fit_bytes, self.params)
        result = add_lap_distance_metrics(result)
        context = {
            "method_name": "GPS Gate (fast-points + distance minima)",
            "params": {},
            "result": result,
        }

        fa = build_further_analysis(context)
        self.assertIsInstance(fa.lap_overview, pd.DataFrame)
        self.assertIsInstance(fa.curve_overview, pd.DataFrame)
        self.assertIsInstance(fa.curve_summary, pd.DataFrame)
        self.assertIsInstance(fa.reference_path, pd.DataFrame)
        self.assertIsInstance(fa.samples_tagged, pd.DataFrame)
        self.assertIsInstance(fa.recommendations, list)
        self.assertGreater(len(fa.recommendations), 0)
        self.assertIsInstance(fa.warnings, list)
        self.assertGreaterEqual(fa.laps_detected, 0)
        self.assertGreaterEqual(fa.valid_laps, 0)

        if np.isfinite(fa.speed_threshold_kmh):
            self.assertGreater(fa.speed_threshold_kmh, 0.0)

        if not fa.samples_tagged.empty:
            self.assertIn("speed_inlier", fa.samples_tagged.columns)

        expected_curve_cols = [
            "lap",
            "curve_id",
            "entry_speed_kmh",
            "apex_speed_kmh",
            "exit_speed_kmh",
            "time_loss_vs_best_s",
        ]
        self.assertTrue(set(expected_curve_cols).issubset(set(fa.curve_overview.columns)))

        if not fa.reference_path.empty:
            expected_ref_cols = {"s_m", "lat_center", "lon_center", "lat_best", "lon_best", "curvature"}
            self.assertTrue(expected_ref_cols.issubset(set(fa.reference_path.columns)))

        if not fa.lap_overview.empty and "lap_time_delta_s" in fa.lap_overview.columns:
            self.assertGreaterEqual(float(fa.lap_overview["lap_time_delta_s"].min()), 0.0)

    def test_further_analysis_empty_context(self) -> None:
        fa = build_further_analysis({})
        self.assertTrue(fa.lap_overview.empty)
        self.assertTrue(fa.curve_overview.empty)
        self.assertTrue(fa.curve_summary.empty)
        self.assertTrue(fa.reference_path.empty)
        self.assertGreater(len(fa.recommendations), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
