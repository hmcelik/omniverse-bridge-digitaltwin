# Miner's rule cumulative fatigue damage tracker for bridge members.
#
# Uses Eurocode 9 detail category 71 S-N curve for aluminium welded joints.
# Damage state persists to JSON between Omniverse sessions.
#
# No Omniverse imports -- runs standalone.

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Eurocode 9 fatigue constants -- sourced from bridge_config
try:
    from .bridge_config import (
        FATIGUE_DETAIL_CATEGORY_PA, FATIGUE_EXPONENT, FATIGUE_LIMIT_PA,
        PARIS_C, PARIS_M, FRACTURE_TOUGHNESS_KIC, CRACK_A0, CRACK_F,
        YIELD_STRENGTH,
    )
except ImportError:
    from bridge_config import (  # type: ignore[no-redef]
        FATIGUE_DETAIL_CATEGORY_PA, FATIGUE_EXPONENT, FATIGUE_LIMIT_PA,
        PARIS_C, PARIS_M, FRACTURE_TOUGHNESS_KIC, CRACK_A0, CRACK_F,
        YIELD_STRENGTH,
    )

_DETAIL_CATEGORY_PA = FATIGUE_DETAIL_CATEGORY_PA
_EXPONENT           = FATIGUE_EXPONENT
_FATIGUE_LIMIT_PA   = FATIGUE_LIMIT_PA
_PARIS_C            = PARIS_C
_PARIS_M            = PARIS_M
_KIC_PA_SQRT_M      = FRACTURE_TOUGHNESS_KIC
_CRACK_A0_M         = CRACK_A0
_CRACK_F            = CRACK_F
_YIELD_STRENGTH_PA  = YIELD_STRENGTH


class DamageState(str, Enum):
    HEALTHY  = "HEALTHY"    # D < 0.3
    WORN     = "WORN"       # 0.3 <= D < 0.7
    WARNING  = "WARNING"    # 0.7 <= D < 1.0
    CRITICAL = "CRITICAL"   # D >= 1.0
    FRACTURE = "FRACTURE"   # crack ratio >= 1.0


def _state_from_d(d: float) -> DamageState:
    if d < 0.3:
        return DamageState.HEALTHY
    if d < 0.7:
        return DamageState.WORN
    if d < 1.0:
        return DamageState.WARNING
    return DamageState.CRITICAL


def _cycles_to_failure(stress_amplitude_pa: float) -> float:
    if stress_amplitude_pa <= _FATIGUE_LIMIT_PA:
        return math.inf
    # N_f such that (detail_cat / stress)^m = N_f / N_ref
    # N_f = N_ref * (DeltaSigma_c / DeltaSigma)^m
    return 2e6 * (_DETAIL_CATEGORY_PA / stress_amplitude_pa) ** _EXPONENT


def _critical_crack_size(max_stress_pa: float) -> float:
    s = abs(max_stress_pa)
    if s <= 0.0:
        return math.inf
    return (_KIC_PA_SQRT_M / (_CRACK_F * s)) ** 2 / math.pi


def _grow_crack_closed_form(a0_m: float, stress_pa: float, n_cycles: float) -> float:
    s = abs(stress_pa)
    n = max(float(n_cycles), 0.0)
    if s <= 0.0 or n <= 0.0:
        return max(a0_m, _CRACK_A0_M)

    a0 = max(a0_m, _CRACK_A0_M)
    p = _PARIS_M / 2.0
    b = _PARIS_C * (_CRACK_F * s * math.sqrt(math.pi)) ** _PARIS_M

    if b <= 0.0:
        return a0
    if abs(p - 1.0) < 1e-12:
        return a0 * math.exp(b * n)

    exponent = 1.0 - p
    term = a0 ** exponent + exponent * b * n
    if term <= 0.0:
        return math.inf
    return term ** (1.0 / exponent)


def _overload_damage_increment(stress_pa: float) -> float:
    ratio = abs(stress_pa) / max(_YIELD_STRENGTH_PA, 1e-9)
    if ratio <= 0.90:
        return 0.0
    if ratio <= 1.0:
        t = (ratio - 0.90) / 0.10
        return 0.05 * t * t
    t = min((ratio - 1.0) / 0.50, 1.0)
    return 0.05 + 0.45 * t * t


class DamageModel:

    def __init__(self, n_members: int, json_path: Optional[Path] = None) -> None:
        self._n = n_members
        self._path = json_path
        self._damage: Dict[int, float] = {i: 0.0 for i in range(n_members)}
        self._pass_count: int = 0
        self._accurate_pass_count: int = 0
        self._total_load_kn_per_pass: List[float] = []
        self._crack_sizes: Dict[int, float] = {
            i: _CRACK_A0_M for i in range(n_members)
        }
        self._max_stress_seen: Dict[int, float] = {i: 0.0 for i in range(n_members)}
        # Buffer of fast-path increments since the last accurate correction.
        # Keyed by member index; cleared after each accurate update.
        self._simple_increments: Dict[int, float] = {}
        if json_path and json_path.exists():
            self.load()

    # Core API
    def _grow_member_crack(self, m_idx: int, stress_pa: float,
                           n_cycles: float) -> None:
        s = abs(stress_pa)
        if s <= 0.0 or n_cycles <= 0:
            return
        self._max_stress_seen[m_idx] = max(
            self._max_stress_seen.get(m_idx, 0.0), s
        )
        current = self._crack_sizes.get(m_idx, _CRACK_A0_M)
        grown = _grow_crack_closed_form(current, s, n_cycles)
        crit = self.get_crack_critical_size(m_idx)
        if math.isfinite(crit):
            # Cap just above critical so ratios can trip fracture alerts while
            # avoiding infinities in persisted JSON after severe overloads.
            grown = min(grown, crit * 1.25)
        self._crack_sizes[m_idx] = max(current, grown)

    def crack_state_snapshot(self) -> dict:
        return {
            "crack_sizes": dict(self._crack_sizes),
            "max_stress_seen": dict(self._max_stress_seen),
        }

    def simple_increment_snapshot(self) -> Dict[int, float]:
        return dict(self._simple_increments)

    def _discard_simple_increments(self, increments: Dict[int, float]) -> None:
        for m_idx, inc in increments.items():
            remaining = self._simple_increments.get(m_idx, 0.0) - float(inc)
            if remaining > 1e-18:
                self._simple_increments[m_idx] = remaining
            else:
                self._simple_increments.pop(m_idx, None)

    def _restore_crack_state(self, snapshot: Optional[dict]) -> None:
        if not snapshot:
            return
        sizes = snapshot.get("crack_sizes", {})
        stresses = snapshot.get("max_stress_seen", {})
        self._crack_sizes = {
            i: max(float(sizes.get(i, _CRACK_A0_M)), _CRACK_A0_M)
            for i in range(self._n)
        }
        self._max_stress_seen = {
            i: max(float(stresses.get(i, 0.0)), 0.0)
            for i in range(self._n)
        }

    def record_pass(self, member_stresses: Dict[int, float],
                    n_cycles: int = 1) -> None:
        self.record_pass_simple(member_stresses, n_cycles=n_cycles)

    def record_pass_simple(self, member_stresses: Dict[int, float],
                           n_cycles: int = 1) -> None:
        for m_idx, stress in member_stresses.items():
            s = abs(stress)
            self._grow_member_crack(m_idx, s, n_cycles)
            nf = _cycles_to_failure(s)
            if math.isfinite(nf) and nf > 0:
                inc = n_cycles / nf
                self._damage[m_idx] = self._damage.get(m_idx, 0.0) + inc
                # Track the fast-path increment so accurate path can correct it
                self._simple_increments[m_idx] = (
                    self._simple_increments.get(m_idx, 0.0) + inc
                )
            overload_inc = _overload_damage_increment(s)
            if overload_inc > 0.0:
                self._damage[m_idx] = self._damage.get(m_idx, 0.0) + overload_inc
                self._simple_increments[m_idx] = (
                    self._simple_increments.get(m_idx, 0.0) + overload_inc
                )
        self._pass_count += 1
        if member_stresses:
            mean_s = sum(abs(v) for v in member_stresses.values()) / len(member_stresses)
            self._total_load_kn_per_pass.append(round(mean_s / 1e3, 4))

    def record_pass_accurate(
        self,
        member_cycles: Dict[int, List[tuple]],
        simple_increments_to_replace: Optional[Dict[int, float]] = None,
        crack_state_to_replace: Optional[dict] = None,
    ) -> None:
        replace = (simple_increments_to_replace
                   if simple_increments_to_replace is not None
                   else dict(self._simple_increments))
        self._restore_crack_state(crack_state_to_replace)

        for m_idx, cycles in member_cycles.items():
            # Compute accurate Miner increment from the full cycle distribution
            acc_inc = 0.0
            for amplitude_pa, _mean_pa, count in cycles:
                self._grow_member_crack(m_idx, abs(amplitude_pa), count)
                nf = _cycles_to_failure(abs(amplitude_pa))
                if math.isfinite(nf) and nf > 0:
                    acc_inc += count / nf

            simple_inc = replace.get(m_idx, 0.0)
            current_d  = self._damage.get(m_idx, 0.0)

            if acc_inc >= simple_inc:
                # Accurate is more damaging than simple estimate -- apply in full.
                correction = acc_inc - simple_inc
                self._damage[m_idx] = current_d + correction
            else:
                # Accurate is less damaging -- smooth correction (never jump down).
                correction = 0.5 * (simple_inc - acc_inc)
                self._damage[m_idx] = max(current_d - correction, 0.0)

        # Clear simple increments buffer for this batch
        if simple_increments_to_replace is None:
            self._simple_increments.clear()
        else:
            self._discard_simple_increments(simple_increments_to_replace)

        self._accurate_pass_count += 1

    def get_damage(self, member_index: int) -> float:
        return self._damage.get(member_index, 0.0)

    def get_state(self, member_index: int) -> DamageState:
        if self.get_crack_ratio(member_index) >= 1.0:
            return DamageState.FRACTURE
        return _state_from_d(self.get_damage(member_index))

    def get_crack_size(self, member_index: int) -> float:
        return self._crack_sizes.get(member_index, _CRACK_A0_M)

    def get_crack_critical_size(self, member_index: int) -> float:
        return _critical_crack_size(self._max_stress_seen.get(member_index, 0.0))

    def get_crack_ratio(self, member_index: int) -> float:
        crit = self.get_crack_critical_size(member_index)
        if not math.isfinite(crit) or crit <= 0.0:
            return 0.0
        return self.get_crack_size(member_index) / crit

    def worst_member(self) -> tuple[int, float]:
        m = max(self._damage, key=lambda k: self._damage[k])
        return m, self._damage[m]

    @property
    def pass_count(self) -> int:
        return self._pass_count

    @property
    def accurate_pass_count(self) -> int:
        return self._accurate_pass_count

    def reset(self) -> None:
        self._damage = {i: 0.0 for i in range(self._n)}
        self._pass_count = 0
        self._accurate_pass_count = 0
        self._total_load_kn_per_pass = []
        self._crack_sizes = {i: _CRACK_A0_M for i in range(self._n)}
        self._max_stress_seen = {i: 0.0 for i in range(self._n)}
        self._simple_increments.clear()
        if self._path:
            self.save()

    # Persistence
    def save(self) -> None:
        if not self._path:
            return
        data = {
            "schema": "bridge_damage_v1",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pass_count": self._pass_count,
            "accurate_pass_count": self._accurate_pass_count,
            "total_load_kn_per_pass": self._total_load_kn_per_pass[-1000:],  # cap log size
            "damage": {str(k): v for k, v in self._damage.items()},
            "crack_state": {
                "crack_sizes_m": {str(k): v for k, v in self._crack_sizes.items()},
                "max_stress_seen_pa": {
                    str(k): v for k, v in self._max_stress_seen.items()
                },
            },
        }
        self._path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if data.get("schema") != "bridge_damage_v1":
                print("[damage_model] Unknown JSON schema -- starting fresh.")
                return
            self._pass_count = int(data.get("pass_count", 0))
            self._accurate_pass_count = int(data.get("accurate_pass_count", 0))
            self._total_load_kn_per_pass = data.get("total_load_kn_per_pass", [])
            raw = data.get("damage", {})
            for k, v in raw.items():
                self._damage[int(k)] = float(v)
            crack_state = data.get("crack_state", {})
            for k, v in crack_state.get("crack_sizes_m", {}).items():
                self._crack_sizes[int(k)] = max(float(v), _CRACK_A0_M)
            for k, v in crack_state.get("max_stress_seen_pa", {}).items():
                self._max_stress_seen[int(k)] = max(float(v), 0.0)
        except Exception as exc:
            print(f"[damage_model] Failed to load {self._path}: {exc}")

    def all_damage(self) -> Dict[int, float]:
        return dict(self._damage)

    def all_crack_ratios(self) -> Dict[int, float]:
        return {i: self.get_crack_ratio(i) for i in range(self._n)}

    def residual_capacity_factor(self) -> float:
        """Return a conservative residual strength factor from damage/cracks."""
        worst_d = max(self._damage.values(), default=0.0)
        worst_crack = max(self.all_crack_ratios().values(), default=0.0)

        # Fatigue damage alone leaves a 40% residual at D >= 1.0, matching the
        # existing residual-strength assumption used by the extension.
        damage_factor = max(0.1, 1.0 - min(worst_d, 1.0) * 0.6)

        # Crack ratio is more brittle: at critical crack size, residual capacity
        # should be near zero even if Miner's D is still numerically small.
        crack_factor = max(0.05, 1.0 - min(worst_crack, 1.0) * 0.95)
        return min(damage_factor, crack_factor)


# Self-test
def run_self_test(verbose: bool = True) -> None:
    import tempfile

    if verbose:
        print("--- damage_model self-test ---")

    # 1. Basic S-N curve
    # At the detail category stress (71 MPa), Nf must be exactly 2x106
    nf_at_detail = _cycles_to_failure(71e6)
    assert abs(nf_at_detail - 2e6) < 1.0, f"Nf at detail cat wrong: {nf_at_detail}"

    # Below fatigue limit -> infinite life
    nf_below = _cycles_to_failure(34e6)
    assert math.isinf(nf_below), "Below fatigue limit should give infinite life"

    # Higher stress -> fewer cycles
    nf_high = _cycles_to_failure(142e6)  # double the stress -> 1/8 cycles
    assert abs(nf_high - 2e6 / 8) < 1.0, f"Nf at 2x stress wrong: {nf_high}"
    if verbose:
        print("  S-N curve: OK")

    # 2. Miner accumulation
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    path.unlink(missing_ok=True)

    model = DamageModel(n_members=3, json_path=path)
    # Apply a stress at the detail category once -> D should be 1/(2e6)
    model.record_pass({0: 71e6, 1: 34e6, 2: 0.0})
    d0 = model.get_damage(0)
    d1 = model.get_damage(1)  # below limit, no damage
    d2 = model.get_damage(2)  # zero stress, no damage
    assert abs(d0 - 1.0 / 2e6) < 1e-15, f"D[0] wrong: {d0}"
    assert d1 == 0.0, f"D[1] should be 0 (below fatigue limit), got {d1}"
    assert d2 == 0.0, f"D[2] should be 0, got {d2}"
    assert model.pass_count == 1
    if verbose:
        print("  Miner accumulation: OK")

    # 3. Persistence round-trip
    model.save()
    model2 = DamageModel(n_members=3, json_path=path)
    assert abs(model2.get_damage(0) - d0) < 1e-20, "Persistence failed"
    assert model2.pass_count == 1
    if verbose:
        print("  JSON persistence: OK")

    # 4. Damage states
    model2._damage[0] = 0.5
    assert model2.get_state(0) == DamageState.WORN
    model2._damage[0] = 0.85
    assert model2.get_state(0) == DamageState.WARNING
    model2._damage[0] = 1.1
    assert model2.get_state(0) == DamageState.CRITICAL
    model2._damage[0] = 0.1
    assert model2.get_state(0) == DamageState.HEALTHY
    if verbose:
        print("  Damage states: OK")

    # 5. Plastic overload penalty
    overload_model = DamageModel(n_members=2)
    overload_model.record_pass_simple({0: 0.95 * _YIELD_STRENGTH_PA}, n_cycles=1)
    overload_model.record_pass_simple({1: 1.20 * _YIELD_STRENGTH_PA}, n_cycles=1)
    assert 0.0 < overload_model.get_damage(0) < 0.05
    assert overload_model.get_damage(1) > 0.10
    if verbose:
        print("  Plastic overload penalty: OK")

    # 6. Paris crack growth
    crack_model = DamageModel(n_members=2)
    crack_model.record_pass_simple({0: 100e6, 1: 60e6}, n_cycles=600_000)
    high_ratio = crack_model.get_crack_ratio(0)
    low_ratio = crack_model.get_crack_ratio(1)
    assert high_ratio >= 1.0, (
        f"100 MPa crack should reach critical size, ratio={high_ratio:.3f}")
    assert low_ratio < 0.5, (
        f"60 MPa crack should grow much slower, ratio={low_ratio:.3f}")
    assert crack_model.get_state(0) == DamageState.FRACTURE
    assert crack_model.residual_capacity_factor() <= 0.06
    crack_only = DamageModel(n_members=1)
    crack_only._crack_sizes[0] = 0.5
    crack_only._max_stress_seen[0] = _KIC_PA_SQRT_M / _CRACK_F / math.sqrt(
        math.pi)
    assert 0.5 < crack_only.residual_capacity_factor() < 0.6
    if verbose:
        print(f"  Paris crack growth: OK "
              f"(100 MPa ratio={high_ratio:.2f}, 60 MPa ratio={low_ratio:.3f})")

    # 7. Crack persistence round-trip
    path.unlink(missing_ok=True)
    crack_model_persist = DamageModel(n_members=1, json_path=path)
    crack_model_persist.record_pass_simple({0: 100e6}, n_cycles=600_000)
    crack_ratio_before = crack_model_persist.get_crack_ratio(0)
    crack_model_persist.save()
    crack_model_loaded = DamageModel(n_members=1, json_path=path)
    assert abs(crack_model_loaded.get_crack_ratio(0) - crack_ratio_before) < 1e-12
    if verbose:
        print("  Crack JSON persistence: OK")

    # 8. Reset
    model2.reset()
    assert model2.get_damage(0) == 0.0
    assert abs(model2.get_crack_size(0) - _CRACK_A0_M) < 1e-15
    assert model2.pass_count == 0
    if verbose:
        print("  Reset: OK")

    path.unlink(missing_ok=True)
    if verbose:
        print("SELF-TEST PASSED")


if __name__ == "__main__":
    run_self_test()


