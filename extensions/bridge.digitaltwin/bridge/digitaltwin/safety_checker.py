# Create operator alerts from stress, damage, crack growth, and vehicle limits.
#
# SafetyChecker does not run the structural model. It receives FEM results and the
# current damage state, applies threshold rules, and logs warning/critical alerts
# when a CSV path is configured.

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .damage_model import DamageModel, DamageState
    from .fem_solver import FEMResult
except ImportError:
    from damage_model import DamageModel, DamageState  # type: ignore[no-redef]
    from fem_solver import FEMResult  # type: ignore[no-redef]

try:
    from .bridge_config import YIELD_STRENGTH as _YIELD_DEFAULT, V_MAX_PROTOTYPE as _V_MAX_DEFAULT
except ImportError:
    from bridge_config import YIELD_STRENGTH as _YIELD_DEFAULT, V_MAX_PROTOTYPE as _V_MAX_DEFAULT  # type: ignore[no-redef]

# Alert threshold for disagreement between the fast peak-stress estimate and
# the rainflow + OpenSeesPy accurate estimate.  35% is derived from the
# expected disagreement sources across the prototype's operating envelope:
#   - Moving load vs. fast-path peak-static approximation : up to ~20%
#   - Rainflow cycle distribution vs. single peak cycle   : up to ~15%
#   - Analytical DAF formula vs. Newmark integration      : up to ~10%
# These combine (non-linearly) to ~35% at the 95th percentile of realistic
# crossings.  Values above this indicate a sensor anomaly or model mismatch,
# not normal solver variance.  The threshold should be re-evaluated if the
# fast-path DAF formula is replaced with a physics-based model.
try:
    from .bridge_config import FAST_VS_ACCURATE_THRESHOLD
except ImportError:
    from bridge_config import FAST_VS_ACCURATE_THRESHOLD  # type: ignore[no-redef]

_FAST_VS_ACCURATE_THRESHOLD = FAST_VS_ACCURATE_THRESHOLD


class AlertLevel(IntEnum):
    INFO     = 0
    WARNING  = 1
    CRITICAL = 2


@dataclass
class Alert:
    level: AlertLevel
    message: str
    member_index: Optional[int] = None   # None if vehicle-level alert

    def __str__(self) -> str:
        prefix = {AlertLevel.INFO: "INFO", AlertLevel.WARNING: "WARN",
                  AlertLevel.CRITICAL: "CRIT"}[self.level]
        m = f" [M{self.member_index}]" if self.member_index is not None else ""
        return f"[{prefix}]{m} {self.message}"


@dataclass
class VehicleParams:
    weight_kg: float
    speed_ms: float
    axle_position_frac: float   # 0=left end, 1=right end along span


class SafetyChecker:

    def __init__(
        self,
        yield_strength_pa: float = _YIELD_DEFAULT,
        safe_load_kg: float = 2.0,
        max_safe_speed_undamaged: float = _V_MAX_DEFAULT,   # m/s at D=0
        max_safe_speed_worn: float = 0.4,         # m/s at D=0.7
        log_path: Optional[Path] = None,
    ) -> None:
        self.yield_strength = yield_strength_pa
        self.safe_load_kg = safe_load_kg
        self._v_undamaged = max_safe_speed_undamaged
        self._v_worn = max_safe_speed_worn
        self._log_path = log_path
        self._log_headers_written = False
        # Cached natural frequency (Hz) from opensees_analyser -- updated externally.
        # When set, overrides the linear speed-limit formula.
        self.natural_frequency_hz: Optional[float] = None

    def check(
        self,
        fem: FEMResult,
        damage: DamageModel,
        vehicle: VehicleParams,
        safe_load_kg: Optional[float] = None,
        safe_speed: Optional[float] = None,
        # Optional extras from the dynamic/environmental pipeline
        resonance_detected: bool = False,
        env_yield_knockdown: float = 0.0,       # fraction (0-1)
        fast_vs_accurate_error: Optional[float] = None,  # fraction deviation
        log_alerts: bool = True,
    ) -> List[Alert]:
        eff_load_limit  = safe_load_kg if safe_load_kg is not None else self.safe_load_kg
        eff_speed_limit = safe_speed   if safe_speed   is not None else self._speed_limit(damage)

        alerts: List[Alert] = []

        # --- Per-member structural and fatigue checks ---
        for m_idx, stress in fem.axial_stresses.items():
            ratio = abs(stress) / self.yield_strength
            d = damage.get_damage(m_idx)

            if ratio > 0.9:
                pct = ratio * 100.0
                alerts.append(Alert(
                    AlertLevel.CRITICAL,
                    f"Do not cross -- member {m_idx} at {pct:.0f}% yield",
                    m_idx,
                ))
            elif ratio > 0.7:
                alerts.append(Alert(
                    AlertLevel.WARNING,
                    f"Reduce speed -- high stress on member {m_idx} ({ratio*100:.0f}% yield)",
                    m_idx,
                ))

            if d >= 1.0:
                alerts.append(Alert(
                    AlertLevel.CRITICAL,
                    f"Member {m_idx} has exceeded fatigue life (D={d:.2f})",
                    m_idx,
                ))
            elif d > 0.7:
                pct_life = d * 100.0
                alerts.append(Alert(
                    AlertLevel.WARNING,
                    f"Member {m_idx} approaching fatigue limit ({pct_life:.0f}% life used)",
                    m_idx,
                ))

            crack_ratio = damage.get_crack_ratio(m_idx)
            if crack_ratio > 0.9:
                alerts.append(Alert(
                    AlertLevel.CRITICAL,
                    f"Fracture risk - member {m_idx} crack is "
                    f"{crack_ratio*100:.0f}% of critical size",
                    m_idx,
                ))
            elif crack_ratio > 0.5:
                alerts.append(Alert(
                    AlertLevel.WARNING,
                    f"Crack growth - member {m_idx} crack is "
                    f"{crack_ratio*100:.0f}% of critical size",
                    m_idx,
                ))

        # --- Vehicle-level checks (against dynamic limits) ---
        if vehicle.weight_kg > eff_load_limit:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"Vehicle exceeds current capacity "
                f"({vehicle.weight_kg:.3f} kg > {eff_load_limit:.3f} kg)",
            ))

        if vehicle.speed_ms > eff_speed_limit:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"Over speed limit ({vehicle.speed_ms:.2f} m/s, "
                f"limit {eff_speed_limit:.2f} m/s at current load & damage)",
            ))

        # --- Dynamic / environmental checks (new) ---
        if resonance_detected:
            fn_str = (f" (f_n={self.natural_frequency_hz:.2f} Hz)"
                      if self.natural_frequency_hz else "")
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"Resonance risk -- vehicle speed near bridge natural frequency{fn_str}",
            ))

        if (fast_vs_accurate_error is not None
                and fast_vs_accurate_error > _FAST_VS_ACCURATE_THRESHOLD):
            pct = fast_vs_accurate_error * 100.0
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"Sensor anomaly -- dynamic FEM deviates {pct:.0f}% from fast estimate "
                f"(threshold {_FAST_VS_ACCURATE_THRESHOLD*100:.0f}%); "
                "check sensor calibration",
            ))

        if env_yield_knockdown > 0.05:
            pct = env_yield_knockdown * 100.0
            alerts.append(Alert(
                AlertLevel.INFO,
                f"Material ageing: {pct:.1f}% yield reduction from environmental exposure",
            ))

        alerts.sort(key=lambda a: (-int(a.level), a.member_index is None))
        if log_alerts:
            self._log(alerts, vehicle)
        return alerts

    def _speed_limit(self, damage: DamageModel) -> float:
        _, worst_d = damage.worst_member()
        t = min(1.0, worst_d / 0.7)
        damage_limit = self._v_undamaged + t * (self._v_worn - self._v_undamaged)

        if self.natural_frequency_hz and self.natural_frequency_hz > 0:
            # Resonance avoidance: keep crossing frequency f_c = v/L away from
            # f_n by at least 20%.  f_c < 0.80*f_n  ->  v < 0.80*f_n*L
            # L is baked into V_MAX via the prototype scale; use 0.5 m.
            L = 0.5   # m
            resonance_limit = 0.80 * self.natural_frequency_hz * L
            return min(damage_limit, resonance_limit)

        return damage_limit

    def _log(self, alerts: List[Alert], vehicle: VehicleParams) -> None:
        if not self._log_path:
            return
        loggable = [a for a in alerts if a.level >= AlertLevel.WARNING]
        if not loggable:
            return
        write_header = not self._log_path.exists() or not self._log_headers_written
        with self._log_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(
                    ["timestamp_utc", "level", "member_index",
                     "message", "weight_kg", "speed_ms"]
                )
                self._log_headers_written = True
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for a in loggable:
                writer.writerow([
                    ts,
                    a.level.name,
                    a.member_index if a.member_index is not None else "",
                    a.message,
                    f"{vehicle.weight_kg:.3f}",
                    f"{vehicle.speed_ms:.3f}",
                ])



