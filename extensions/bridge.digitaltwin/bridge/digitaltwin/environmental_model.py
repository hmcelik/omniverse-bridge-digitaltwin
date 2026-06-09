# Track slow material degradation from environmental exposure.
#
# The model stores accumulated humidity and temperature-cycle exposure, then
# returns reduced stiffness, yield strength, and fatigue limit values. The values
# are empirical estimates for the digital twin; they must be calibrated before
# being used for field decisions.

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


try:
    from .bridge_config import (
        E_MODULUS, YIELD_STRENGTH, FATIGUE_LIMIT_PA,
        ENV_MIN_E_FRACTION, ENV_MIN_YIELD_FRACTION, ENV_MIN_FATIGUE_FRACTION,
        ENV_TEMP_CYCLE_YIELD_RATE, ENV_TEMP_CYCLE_FATIGUE_RATE, ENV_TEMP_CYCLE_E_RATE,
        ENV_HUMIDITY_YIELD_RATE, ENV_HUMIDITY_FATIGUE_RATE, ENV_HUMIDITY_E_RATE,
        ENV_OUTDOOR_MULTIPLIER, ENV_INDOOR_MULTIPLIER,
    )
except ImportError:
    from bridge_config import (  # type: ignore[no-redef]
        E_MODULUS, YIELD_STRENGTH, FATIGUE_LIMIT_PA,
        ENV_MIN_E_FRACTION, ENV_MIN_YIELD_FRACTION, ENV_MIN_FATIGUE_FRACTION,
        ENV_TEMP_CYCLE_YIELD_RATE, ENV_TEMP_CYCLE_FATIGUE_RATE, ENV_TEMP_CYCLE_E_RATE,
        ENV_HUMIDITY_YIELD_RATE, ENV_HUMIDITY_FATIGUE_RATE, ENV_HUMIDITY_E_RATE,
        ENV_OUTDOOR_MULTIPLIER, ENV_INDOOR_MULTIPLIER,
    )

_REF_E_PA             = E_MODULUS
_REF_YIELD_PA         = YIELD_STRENGTH
_REF_FATIGUE_LIM_PA   = FATIGUE_LIMIT_PA

_MIN_E_FRACTION       = ENV_MIN_E_FRACTION
_MIN_YIELD_FRACTION   = ENV_MIN_YIELD_FRACTION
_MIN_FATIGUE_FRACTION = ENV_MIN_FATIGUE_FRACTION

_TEMP_CYCLE_YIELD_RATE   = ENV_TEMP_CYCLE_YIELD_RATE
_TEMP_CYCLE_FATIGUE_RATE = ENV_TEMP_CYCLE_FATIGUE_RATE
_TEMP_CYCLE_E_RATE       = ENV_TEMP_CYCLE_E_RATE

_HUMIDITY_YIELD_RATE   = ENV_HUMIDITY_YIELD_RATE
_HUMIDITY_FATIGUE_RATE = ENV_HUMIDITY_FATIGUE_RATE
_HUMIDITY_E_RATE       = ENV_HUMIDITY_E_RATE

_OUTDOOR_MULTIPLIER = ENV_OUTDOOR_MULTIPLIER
_INDOOR_MULTIPLIER  = ENV_INDOOR_MULTIPLIER


@dataclass
class EnvironmentalState:
    temperature_cycles: int    = 0
    delta_T_avg_C:      float  = 20.0    # degC -- typical diurnal swing
    humidity_hours:     float  = 0.0
    humidity_rh_avg:    float  = 0.55    # 55% RH -- moderate indoor default
    exposure:           str    = "indoor"
    session_start_utc:  str    = ""
    last_updated_utc:   str    = ""


@dataclass
class DegradedProperties:
    E_pa:              float
    yield_pa:          float
    fatigue_limit_pa:  float
    # Fractional reduction from reference (0.0 = no change, 1.0 = total loss)
    E_knockdown:        float
    yield_knockdown:    float
    fatigue_knockdown:  float


class EnvironmentalModel:

    def __init__(
        self,
        base_E_pa:             float = _REF_E_PA,
        base_yield_pa:         float = _REF_YIELD_PA,
        base_fatigue_limit_pa: float = _REF_FATIGUE_LIM_PA,
        state:                 Optional[EnvironmentalState] = None,
    ) -> None:
        self.base_E              = base_E_pa
        self.base_yield          = base_yield_pa
        self.base_fatigue_limit  = base_fatigue_limit_pa
        self.state               = state or EnvironmentalState()
        if not self.state.session_start_utc:
            self.state.session_start_utc = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())


    def get_degraded_properties(self) -> DegradedProperties:
        s = self.state
        exposure_mult = (_OUTDOOR_MULTIPLIER if s.exposure == "outdoor"
                         else _INDOOR_MULTIPLIER)

        # Temperature cycling knock-downs
        temp_load   = s.temperature_cycles * (s.delta_T_avg_C ** 2)
        kd_E_temp   = _TEMP_CYCLE_E_RATE     * temp_load
        kd_Y_temp   = _TEMP_CYCLE_YIELD_RATE * temp_load
        kd_F_temp   = _TEMP_CYCLE_FATIGUE_RATE * temp_load

        # Humidity knock-downs
        hum_load    = s.humidity_hours * s.humidity_rh_avg * exposure_mult
        kd_E_hum    = _HUMIDITY_E_RATE       * hum_load
        kd_Y_hum    = _HUMIDITY_YIELD_RATE   * hum_load
        kd_F_hum    = _HUMIDITY_FATIGUE_RATE * hum_load

        # Total knock-downs (additive, then clamped to physical floors)
        kd_E  = min(kd_E_temp  + kd_E_hum,  1.0 - _MIN_E_FRACTION)
        kd_Y  = min(kd_Y_temp  + kd_Y_hum,  1.0 - _MIN_YIELD_FRACTION)
        kd_F  = min(kd_F_temp  + kd_F_hum,  1.0 - _MIN_FATIGUE_FRACTION)

        E_deg    = self.base_E             * (1.0 - kd_E)
        yield_deg = self.base_yield        * (1.0 - kd_Y)
        fat_deg  = self.base_fatigue_limit * (1.0 - kd_F)

        return DegradedProperties(
            E_pa=E_deg,
            yield_pa=yield_deg,
            fatigue_limit_pa=fat_deg,
            E_knockdown=kd_E,
            yield_knockdown=kd_Y,
            fatigue_knockdown=kd_F,
        )

    def advance_time(self, hours: float = 1.0,
                     n_temp_cycles: int = 0,
                     delta_T_C: float = 20.0) -> None:
        self.state.humidity_hours      += hours
        self.state.temperature_cycles  += n_temp_cycles
        # Running average of DeltaT (weighted by count)
        total = self.state.temperature_cycles
        if total > 0:
            prev_total = total - n_temp_cycles
            self.state.delta_T_avg_C = (
                self.state.delta_T_avg_C * prev_total + delta_T_C * n_temp_cycles
            ) / total
        self.state.last_updated_utc = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def set_exposure(self, exposure: str, humidity_rh: float = 0.55) -> None:
        if exposure not in ("indoor", "outdoor"):
            raise ValueError("exposure must be 'indoor' or 'outdoor'")
        self.state.exposure       = exposure
        self.state.humidity_rh_avg = max(0.0, min(1.0, humidity_rh))


    def to_dict(self) -> dict:
        s = self.state
        return {
            "temperature_cycles": s.temperature_cycles,
            "delta_T_avg_C":      s.delta_T_avg_C,
            "humidity_hours":     s.humidity_hours,
            "humidity_rh_avg":    s.humidity_rh_avg,
            "exposure":           s.exposure,
            "session_start_utc":  s.session_start_utc,
            "last_updated_utc":   s.last_updated_utc,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        base_E_pa:             float = _REF_E_PA,
        base_yield_pa:         float = _REF_YIELD_PA,
        base_fatigue_limit_pa: float = _REF_FATIGUE_LIM_PA,
    ) -> "EnvironmentalModel":
        state = EnvironmentalState(
            temperature_cycles = int(data.get("temperature_cycles", 0)),
            delta_T_avg_C      = float(data.get("delta_T_avg_C", 20.0)),
            humidity_hours     = float(data.get("humidity_hours", 0.0)),
            humidity_rh_avg    = float(data.get("humidity_rh_avg", 0.55)),
            exposure           = str(data.get("exposure", "indoor")),
            session_start_utc  = str(data.get("session_start_utc", "")),
            last_updated_utc   = str(data.get("last_updated_utc", "")),
        )
        return cls(base_E_pa, base_yield_pa, base_fatigue_limit_pa, state)


# Self-test
def run_self_test(verbose: bool = True) -> None:
    if verbose:
        print("--- environmental_model self-test ---")

    model = EnvironmentalModel()

    # 1. Fresh model should return near-reference values
    props = model.get_degraded_properties()
    assert abs(props.E_pa   - _REF_E_PA)    < 1.0, "Fresh E should equal reference"
    assert abs(props.yield_pa - _REF_YIELD_PA) < 1.0, "Fresh yield should equal reference"
    if verbose:
        print("  Fresh model: OK")

    # 2. After significant outdoor exposure, properties should degrade
    model.set_exposure("outdoor", humidity_rh=0.80)
    model.advance_time(hours=5000, n_temp_cycles=200, delta_T_C=25.0)
    props2 = model.get_degraded_properties()
    assert props2.yield_pa < _REF_YIELD_PA * 0.99, "yield should degrade outdoors"
    assert props2.fatigue_limit_pa < _REF_FATIGUE_LIM_PA * 0.99, "fatigue limit should degrade"
    assert props2.yield_knockdown > 0, "yield knockdown should be positive"
    if verbose:
        print(f"  After 5000h outdoor: yield={props2.yield_pa/1e6:.1f} MPa "
              f"({props2.yield_knockdown*100:.1f}% loss), "
              f"fatigue_limit={props2.fatigue_limit_pa/1e6:.1f} MPa "
              f"({props2.fatigue_knockdown*100:.1f}% loss)")
        print("  Degradation after outdoor exposure: OK")

    # 3. Property floors are respected
    # Simulate 200 years of harsh outdoor exposure
    model.advance_time(hours=200 * 8760, n_temp_cycles=200 * 365, delta_T_C=30.0)
    props3 = model.get_degraded_properties()
    assert props3.E_pa    >= _REF_E_PA    * _MIN_E_FRACTION    - 1.0
    assert props3.yield_pa >= _REF_YIELD_PA * _MIN_YIELD_FRACTION - 1.0
    assert props3.fatigue_limit_pa >= _REF_FATIGUE_LIM_PA * _MIN_FATIGUE_FRACTION - 1.0
    if verbose:
        print(f"  Floor limits respected: E>={_MIN_E_FRACTION*100:.0f}%  "
              f"yield>={_MIN_YIELD_FRACTION*100:.0f}%  "
              f"fatigue>={_MIN_FATIGUE_FRACTION*100:.0f}%")

    # 4. Round-trip persistence
    d = model.to_dict()
    model2 = EnvironmentalModel.from_dict(d)
    props4 = model2.get_degraded_properties()
    assert abs(props4.yield_pa - props3.yield_pa) < 1.0, "Persistence round-trip failed"
    if verbose:
        print("  Persistence round-trip: OK")

    if verbose:
        print("SELF-TEST PASSED")


if __name__ == "__main__":
    run_self_test()



