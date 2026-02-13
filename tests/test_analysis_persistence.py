from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from services.analysis_service import (
    build_attempt_curve_rows,
    build_attempt_lap_rows,
    build_attempt_summary_payload,
)


class _FakeFurther:
    def __init__(self) -> None:
        self.lap_overview = pd.DataFrame(
            [
                {
                    "lap": 1,
                    "lap_time_s": 75.0,
                    "avg_speed_kmh": 40.0,
                    "max_speed_kmh": 48.0,
                    "lap_distance_m": 850.0,
                    "is_valid_lap": True,
                }
            ]
        )
        self.curve_overview = pd.DataFrame(
            [
                {
                    "lap": 1,
                    "curve_id": 1,
                    "entry_speed_kmh": 45.0,
                    "apex_speed_kmh": 33.0,
                    "exit_speed_kmh": 42.0,
                    "curve_time_s": 5.4,
                    "time_loss_vs_best_s": 0.2,
                    "s_start_m": 100.0,
                    "s_end_m": 150.0,
                    "s_apex_m": 125.0,
                    "peak_curvature": 0.04,
                }
            ]
        )
        self.speed_threshold_kmh = 21.0
        self.laps_detected = 12
        self.valid_laps = 11
        self.curves_detected = 6
        self.boundary_curves = 1
        self.median_track_width_m = 1.9
        self.p90_track_width_m = 4.2


class AnalysisPersistenceTests(unittest.TestCase):
    def test_summary_payload(self) -> None:
        bundle = {"further": _FakeFurther()}
        p = build_attempt_summary_payload(bundle)
        self.assertEqual(p["status"], "processed")
        self.assertEqual(p["laps_detected"], 12)
        self.assertTrue(np.isfinite(p["best_lap_s"]))

    def test_lap_rows_payload(self) -> None:
        f = _FakeFurther()
        rows = build_attempt_lap_rows("a1", f.lap_overview)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempt_id"], "a1")
        self.assertEqual(rows[0]["lap"], 1)

    def test_curve_rows_payload(self) -> None:
        f = _FakeFurther()
        rows = build_attempt_curve_rows("a1", f.curve_overview)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["curve_id"], 1)
        self.assertTrue(np.isfinite(rows[0]["time_loss_vs_best_s"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
