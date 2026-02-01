import numpy as np
from lap_methods.utils_distance import cumulative_distance_m

def add_lap_distance_metrics(result: dict) -> dict:
    gps = result["gps"]
    laps = result["laps"]
    lap_metrics = result["lap_metrics"]

    if laps is None or laps.empty or lap_metrics is None or lap_metrics.empty:
        return result

    s = cumulative_distance_m(gps)
    t = gps["timestamp"].to_numpy()

    lap_dist = []
    for _, r in laps.iterrows():
        i0 = int(np.argmin(np.abs(t - np.datetime64(r["start"]))))
        i1 = int(np.argmin(np.abs(t - np.datetime64(r["end"]))))
        d = float(s[i1] - s[i0])
        lap_dist.append(d)

    # align by lap order
    lap_metrics = lap_metrics.sort_values("lap").reset_index(drop=True)
    lap_metrics["lap_distance_m"] = lap_dist[:len(lap_metrics)]
    lap_metrics["lap_distance_km"] = lap_metrics["lap_distance_m"] / 1000.0

    result["lap_metrics"] = lap_metrics
    return result
