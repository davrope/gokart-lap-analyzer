from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.further_plots import plot_curve_geometry


class FurtherPlotsTests(unittest.TestCase):
    def test_curve_geometry_wrap_segment_is_split(self) -> None:
        s = np.arange(0.0, 100.0, 1.0)
        ref = pd.DataFrame(
            {
                "s_m": s,
                "x_center_m": s,
                "y_center_m": np.sin(s / 10.0),
            }
        )
        # Wrap around start/finish: 90 -> 10
        curves = pd.DataFrame(
            {
                "curve_id": [1],
                "s_start_m": [90.0],
                "s_end_m": [10.0],
                "s_apex_m": [95.0],
                "peak_curvature": [0.02],
            }
        )

        fig = plot_curve_geometry(ref, curves)

        line_traces = [t for t in fig.data if getattr(t, "mode", None) == "lines"]
        # 1 centerline + 2 split wrapped curve segments
        self.assertEqual(len(line_traces), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
