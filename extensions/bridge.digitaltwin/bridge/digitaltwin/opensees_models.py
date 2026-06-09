# Constants, OpenSeesPy import, and shared data models for the bridge dynamic
# analysis stack.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import openseespy.opensees as ops
    _OPS_AVAILABLE = True
except Exception as _ops_err:
    ops = None          # type: ignore[assignment]
    _OPS_AVAILABLE = False
    print(f"[opensees_analyser] OpenSeesPy unavailable ({_ops_err}); "
          "dynamic analysis disabled.")

try:
    from .bridge_config import (
        GRAVITY, DAMPING_RATIO, NEWMARK_GAMMA, NEWMARK_BETA,
        DT_TARGET, MAX_STEPS, MIN_NODE_MASS_KG, MAX_FREQ_FOR_DT_HZ, N_MODES,
        GEOMETRIC_NONLINEARITY_DEFAULT, NEWTON_TOLERANCE, NEWTON_MAX_ITER,
    )
except ImportError:
    from bridge_config import (  # type: ignore[no-redef]
        GRAVITY, DAMPING_RATIO, NEWMARK_GAMMA, NEWMARK_BETA,
        DT_TARGET, MAX_STEPS, MIN_NODE_MASS_KG, MAX_FREQ_FOR_DT_HZ, N_MODES,
        GEOMETRIC_NONLINEARITY_DEFAULT, NEWTON_TOLERANCE, NEWTON_MAX_ITER,
    )

# Keep private aliases so existing imports in opensees_dynamic / opensees_static
# (e.g. "from .opensees_models import _GRAVITY") continue to work unchanged.
_DAMPING_RATIO      = DAMPING_RATIO
_NEWMARK_GAMMA      = NEWMARK_GAMMA
_NEWMARK_BETA       = NEWMARK_BETA
_DT_TARGET          = DT_TARGET
_GRAVITY            = GRAVITY
_MAX_STEPS          = MAX_STEPS
_MIN_NODE_MASS_KG   = MIN_NODE_MASS_KG
_MAX_FREQ_FOR_DT_HZ = MAX_FREQ_FOR_DT_HZ


@dataclass
class CrossingResult:
    # Per-member peak dynamic stress (Pa, absolute value)
    peak_stresses: Dict[int, float]
    # Per-member stress time series, shape (steps_completed+1,), Pa
    stress_histories: Dict[int, np.ndarray]
    # Natural frequencies (Hz), lowest first
    natural_frequencies: List[float]
    # Per-member DAF envelope ratio, max across all members, >= 1.0
    dynamic_amplification_factor: float
    # Time vector (s) matching the stress_histories arrays
    time_vector: np.ndarray
    # Whether OpenSeesPy was available; False => static fallback was used
    is_dynamic: bool = True
    # Convergence diagnostics -- non-converged results must not feed DamageModel
    steps_completed: int = 0    # actual steps run (<= n_steps planned)
    converged: bool = True      # False if time-stepping stopped early


@dataclass
class AnalyserConfig:
    damping_ratio: float = _DAMPING_RATIO
    dt_target:     float = _DT_TARGET
    # Number of eigen modes to extract (0 = skip eigenvalue + time-step refinement)
    n_modes:       int   = N_MODES
    geometric_nonlinearity: bool = GEOMETRIC_NONLINEARITY_DEFAULT
    newton_tolerance: float = NEWTON_TOLERANCE
    newton_max_iter: int = NEWTON_MAX_ITER


