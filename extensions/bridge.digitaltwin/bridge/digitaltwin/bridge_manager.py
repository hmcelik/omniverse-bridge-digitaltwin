# Background dynamic-analysis worker.
#
# The extension records quick damage estimates immediately after each crossing.
# This manager periodically runs the slower OpenSees analysis, rainflow-counts the
# stress histories, and corrects the accumulated damage when the solve converges.

from __future__ import annotations

import queue
import threading
from typing import Dict, List, Optional

try:
    from .opensees_analyser import OpenSeesAnalyser
    from .damage_model import DamageModel
    from .environmental_model import EnvironmentalModel
    from .sensor_reader import SensorReader, VehiclePass
    from .safety_checker import SafetyChecker
    from .rainflow_counter import count_cycles_per_member
    from .bridge_config import REAL_BRIDGE_LENGTH
except ImportError:
    from opensees_analyser import OpenSeesAnalyser  # type: ignore[no-redef]
    from damage_model import DamageModel  # type: ignore[no-redef]
    from environmental_model import EnvironmentalModel  # type: ignore[no-redef]
    from sensor_reader import SensorReader, VehiclePass  # type: ignore[no-redef]
    from safety_checker import SafetyChecker  # type: ignore[no-redef]
    from rainflow_counter import count_cycles_per_member  # type: ignore[no-redef]
    from bridge_config import REAL_BRIDGE_LENGTH  # type: ignore[no-redef]


# Background dynamic analysis manager
class DynamicAnalysisManager:

    def __init__(
        self,
        damage_model: DamageModel,
        env_model: EnvironmentalModel,
        sensor_reader: SensorReader,
        safety_checker: SafetyChecker,
        passes_per_trigger: int = 10,
    ) -> None:
        self._damage      = damage_model
        self._env         = env_model
        self._sensor      = sensor_reader
        self._safety      = safety_checker
        self._per_trigger = passes_per_trigger

        self._analyser:    Optional[OpenSeesAnalyser] = None
        self._work_queue:  "queue.Queue[dict]" = queue.Queue(maxsize=8)
        self._result_queue: "queue.Queue[dict]" = queue.Queue(maxsize=16)

        self._stop      = threading.Event()
        self._thread:   Optional[threading.Thread] = None
        self._pending_passes: List[dict] = []   # buffer since last trigger

        # Status readable by UI
        self.status: str = "idle"
        self.last_daf:  float = 1.0
        self.last_nat_freq: List[float] = []
        self.passes_since_last_run: int = 0
        self.analysis_mode: str = "unavailable"


    def start(self) -> None:
        self._stop.clear()
        # Pre-arm trigger on brand-new sessions (accurate_pass_count is 0 when
        # the damage model has never had a background solve -- fresh install or
        # after a full reset).  Setting passes_since_last_run to the threshold
        # means the very first enqueued pass will fire the worker immediately,
        # so operators see dynamic results even in very short demo sessions.
        # On subsequent restarts (accurate_pass_count > 0) the counter starts
        # at 0 so the first trigger fires after passes_per_trigger normal passes.
        if self._damage.accurate_pass_count == 0:
            self.passes_since_last_run = self._per_trigger
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="DynAnalysis")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._work_queue.put_nowait({"_stop": True})
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None

    def set_analyser(self, analyser: OpenSeesAnalyser) -> None:
        self._analyser = analyser
        self.analysis_mode = (
            "OpenSees 3D dynamic" if getattr(analyser, "is_3d_frame", False)
            else "OpenSees fallback-capable"
        )


    def enqueue_pass(self, vp: VehiclePass, forces: Dict[int, float],
                     member_area: float, forced: bool = False) -> None:
        member_stresses = {m: abs(f) / member_area for m, f in forces.items()}
        self._pending_passes.append({
            "weight_kg":    vp.weight_kg,
            "speed_ms":     vp.speed_ms,
            "lateral_frac": 0.5,
            "member_stresses_simple": dict(member_stresses),
        })
        self.passes_since_last_run += 1

        should_run = forced or self.passes_since_last_run >= self._per_trigger
        if should_run:
            self._queue_pending(forced=forced)

    def request_immediate(self) -> None:
        self._queue_pending(forced=True)

    def poll_results(self) -> Optional[dict]:
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def _queue_pending(self, forced: bool = False) -> None:
        if self._analyser is None or not self._pending_passes:
            return
        try:
            self._work_queue.put_nowait({
                "passes":  list(self._pending_passes),
                "forced":  forced,
                "crack_state_before_batch": self._damage.crack_state_snapshot(),
                "simple_increments_to_replace": (
                    self._damage.simple_increment_snapshot()
                    if hasattr(self._damage, "simple_increment_snapshot")
                    else {}
                ),
            })
            self._pending_passes.clear()
            self.passes_since_last_run = 0
        except queue.Full:
            pass   # worker is busy; pending passes will accumulate


    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._work_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            if item.get("_stop"):
                break

            passes = item.get("passes", [])
            if not passes or self._analyser is None:
                continue

            self.status = "running"
            try:
                # Use the last pass in the batch as the representative vehicle
                rep = passes[-1]
                env_props = self._env.get_degraded_properties()

                cr = self._analyser.run_crossing(
                    weight_kg     = rep["weight_kg"],
                    speed_ms      = rep["speed_ms"],
                    lateral_frac  = rep["lateral_frac"],
                    bridge_length_m = REAL_BRIDGE_LENGTH,
                    E_override    = env_props.E_pa,
                    yield_override = env_props.yield_pa,
                )

                # Rainflow count all member stress histories
                member_cycles = count_cycles_per_member(cr.stress_histories)

                apply_accurate_damage = bool(cr.is_dynamic and cr.converged)
                if not apply_accurate_damage:
                    print(f"[DynamicAnalysisManager] skipping record_pass_accurate: "
                          f"result not dynamic/converged "
                          f"({cr.steps_completed}/{cr.steps_completed} steps)")

                # Compute fast-vs-accurate divergence for the governing member.
                # Guard on is_dynamic: the static fallback path returns DAF from
                # an analytical formula which always differs from the simple fast
                # path by design (~84% in tests), causing a spurious sensor-anomaly
                # alert.  Only a real dynamic solve is a meaningful comparison.
                # Also guard on converged so a partial result doesn't fire an alert.
                if cr.is_dynamic and cr.converged and cr.peak_stresses:
                    governing     = max(cr.peak_stresses, key=lambda k: cr.peak_stresses[k])
                    simple_stress = rep["member_stresses_simple"].get(governing, 0.0)
                    dyn_stress    = cr.peak_stresses.get(governing, 0.0)
                    error: Optional[float] = (
                        abs(dyn_stress - simple_stress) / max(simple_stress, 1e-6)
                        if simple_stress > 0 else 0.0)
                else:
                    error = None

                self.last_daf      = cr.dynamic_amplification_factor
                self.last_nat_freq = cr.natural_frequencies
                self.analysis_mode = (
                    "OpenSees dynamic" if cr.is_dynamic
                    else "static fallback"
                )

                result = {
                    "natural_frequencies":    cr.natural_frequencies,
                    "daf":                    cr.dynamic_amplification_factor,
                    "is_dynamic":             cr.is_dynamic,
                    "converged":              cr.converged,
                    "steps_completed":        cr.steps_completed,
                    "env_yield_knockdown":    env_props.yield_knockdown,
                    "fast_vs_accurate_error": error,
                    "n_passes_analysed":      len(passes),
                    "analysis_mode":           self.analysis_mode,
                    "member_cycles":           member_cycles,
                    "apply_accurate_damage":   apply_accurate_damage,
                    "simple_increments_to_replace": item.get(
                        "simple_increments_to_replace", {}),
                    "crack_state_before_batch": item.get(
                        "crack_state_before_batch"),
                    "daf_calibration": (
                        (rep["speed_ms"], cr.dynamic_amplification_factor)
                        if apply_accurate_damage
                        and cr.dynamic_amplification_factor > 0
                        else None
                    ),
                }
                try:
                    self._result_queue.put_nowait(result)
                except queue.Full:
                    pass   # main thread not polling fast enough; discard oldest
            except Exception as exc:
                print(f"[DynamicAnalysisManager] worker error: {exc!r}")
            finally:
                self.status = "idle"



