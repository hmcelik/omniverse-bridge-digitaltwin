# Sensor-vs-model residual helpers for live digital-twin validation.
#
# The functions here are pure Python so they can be tested outside Omniverse.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class SensorResidual:
    member_index: int
    measured_microstrain: float
    predicted_microstrain: float
    residual_microstrain: float
    relative_error: float
    status: str


def stress_to_microstrain(stress_pa: float, e_pa: float) -> float:
    if e_pa <= 0.0:
        return 0.0
    return stress_pa / e_pa * 1_000_000.0


def compute_sensor_residuals(
    measured_microstrain: Dict[int, float],
    predicted_stress_pa: Dict[int, float],
    gauged_members: Iterable[int],
    e_pa: float,
    *,
    stale_members: Iterable[int] = (),
    missing_threshold_ue: float = 1e-6,
    drift_threshold: float = 0.35,
    outlier_threshold: float = 1.0,
) -> List[SensorResidual]:
    stale = set(stale_members)
    residuals: List[SensorResidual] = []

    for m_idx in gauged_members:
        predicted = stress_to_microstrain(
            predicted_stress_pa.get(m_idx, 0.0), e_pa
        )
        measured_present = m_idx in measured_microstrain
        measured = float(measured_microstrain.get(m_idx, 0.0))
        residual = measured - predicted
        denom = max(abs(predicted), missing_threshold_ue)
        rel = abs(residual) / denom if measured_present else 0.0

        if not measured_present:
            status = "missing"
        elif m_idx in stale:
            status = "stale"
        elif abs(measured) <= missing_threshold_ue and abs(predicted) > 1.0:
            status = "missing"
        elif rel >= outlier_threshold:
            status = "outlier"
        elif rel >= drift_threshold:
            status = "drift"
        else:
            status = "ok"

        residuals.append(SensorResidual(
            member_index=m_idx,
            measured_microstrain=measured,
            predicted_microstrain=predicted,
            residual_microstrain=residual,
            relative_error=rel,
            status=status,
        ))

    return residuals

