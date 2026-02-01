# lap_methods/utils_distance.py
from __future__ import annotations
import numpy as np
import pandas as pd

def cumulative_distance_m(gps: pd.DataFrame) -> np.ndarray:
    lat = gps["lat"].to_numpy(dtype=float)
    lon = gps["lon"].to_numpy(dtype=float)

    R = 6371000.0
    lat1 = np.radians(lat[:-1]); lon1 = np.radians(lon[:-1])
    lat2 = np.radians(lat[1:]);  lon2 = np.radians(lon[1:])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    step = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)

    return np.concatenate([[0.0], np.cumsum(step)])
