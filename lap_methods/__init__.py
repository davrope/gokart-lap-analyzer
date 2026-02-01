from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

from .gps_gate import GpsGateParams, run as run_gps_gate
from .dtw_template import DtwTemplateParams, run as run_dtw_template
from .spectral_phase import SpectralPhaseParams, run as run_spectral_phase
from .loop_closure_phase import LoopClosurePhaseParams, run as run_loop_closure_phase
from .cyclic_hmm import CyclicHMMParams, run as run_cyclic_hmm
from .auto_gate_density import AutoGateDensityParams, run as run_auto_gate_density


METHODS = {
    "GPS Gate (fast-points + distance minima)": {
        "params_cls": GpsGateParams,
        "runner": run_gps_gate,
    },
    "DTW Template (self-supervised)": {
    "params_cls": DtwTemplateParams,
    "runner": run_dtw_template,
    },
    "Spectral Phase (self-supervised)": {
    "params_cls": SpectralPhaseParams,
    "runner": run_spectral_phase,
    },
    "Loop closure + phase consistency": {
    "params_cls": LoopClosurePhaseParams,
    "runner": run_loop_closure_phase,
    },
    "Cyclic HMM (loop phase + pit)": {
    "params_cls": CyclicHMMParams,
    "runner": run_cyclic_hmm,
    },
    "Auto gate (density → auto params)": {
    "params_cls": AutoGateDensityParams,
    "runner": run_auto_gate_density,
    },

}

def get_methods() -> Dict[str, Dict[str, Any]]:
    return METHODS

def default_params(method_name: str) -> Any:
    cls = METHODS[method_name]["params_cls"]
    return cls()

def params_to_dict(params: Any) -> Dict[str, Any]:
    return asdict(params)
