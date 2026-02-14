from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lap_methods.quality import quality_summary_gps_gate


class QualitySummaryTests(unittest.TestCase):
    def test_quality_summary_accepts_dict_params(self) -> None:
        ts = pd.date_range("2026-01-01", periods=20, freq="1s")
        gps = pd.DataFrame(
            {
                "timestamp": ts,
                "lat": np.linspace(40.0, 40.0002, 20),
                "lon": np.linspace(-3.0, -3.0002, 20),
                "speed_kmh": np.linspace(25.0, 35.0, 20),
            }
        )

        laps = pd.DataFrame({"lap": [1], "start": [ts[0]], "end": [ts[-1]]})
        lap_metrics = pd.DataFrame({"lap": [1], "lap_time_s": [19.0], "lap_distance_m": [300.0]})
        pass_idx = np.array([5, 10], dtype=int)
        dist = np.linspace(0.0, 8.0, 20)

        result = {
            "gps": gps,
            "pass_idx": pass_idx,
            "laps": laps,
            "lap_metrics": lap_metrics,
            "dist": dist,
            "gate_lat": 40.0001,
            "gate_lon": -3.0001,
        }
        params = {"speed_for_gate": 20.0, "near_m": 12.0, "min_lap_s": 20.0}

        out = quality_summary_gps_gate(result, params)
        self.assertIn("boundary_fast_frac", out)
        self.assertTrue(np.isfinite(out["boundary_fast_frac"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
