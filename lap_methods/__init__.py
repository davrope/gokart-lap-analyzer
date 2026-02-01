from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

from .gps_gate import GpsGateParams, run as run_gps_gate
from .dtw_template import DtwTemplateParams, run as run_dtw_template

METHODS = {
    "GPS Gate (fast-points + distance minima)": {
        "params_cls": GpsGateParams,
        "runner": run_gps_gate,
    },
    "DTW Template (self-supervised)": {
    "params_cls": DtwTemplateParams,
    "runner": run_dtw_template,
    },

}

def get_methods() -> Dict[str, Dict[str, Any]]:
    return METHODS

def default_params(method_name: str) -> Any:
    cls = METHODS[method_name]["params_cls"]
    return cls()

def params_to_dict(params: Any) -> Dict[str, Any]:
    return asdict(params)
