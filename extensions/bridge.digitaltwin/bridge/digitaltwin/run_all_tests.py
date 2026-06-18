# Standalone engineering test runner for bridge.digitaltwin.
#
# Run from this directory with:
# python run_all_tests.py
#
# The runner intentionally avoids extension.py, bridge_ui.py, and omni.* imports
from __future__ import annotations

import importlib
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SkipTest(Exception):
    pass


@dataclass
class TestResult:
    name: str
    status: str
    reason: str = ""


CheckFn = Callable[[], None]


def _import(name: str):
    return importlib.import_module(name)


def _require_opensees() -> None:
    models = _import("opensees_models")
    if not getattr(models, "_OPS_AVAILABLE", False):
        raise SkipTest("OpenSeesPy unavailable in this Python environment")


def _run_subprocess(script_name: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / script_name)],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    output = proc.stdout.strip()
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-12:])
        raise AssertionError(
            f"{script_name} exited {proc.returncode}"
            + (f"\n{tail}" if tail else "")
        )
    return output


def check_fem_2d() -> None:
    _import("fem_solver").run_self_test(verbose=False)


def check_fem_3d() -> None:
    _import("fem_solver").run_3d_frame_self_test(verbose=False)


def check_opensees_beam_3d() -> None:
    _require_opensees()
    ok = _import("opensees_dynamic").run_3d_frame_beam_self_test(verbose=False)
    if not ok:
        raise SkipTest("OpenSees 3D beam self-test reported unavailable")


def check_opensees_geometric_nonlinearity() -> None:
    _require_opensees()
    ok = _import("opensees_dynamic").run_geometric_nonlinearity_self_test(
        verbose=False
    )
    if not ok:
        raise SkipTest("OpenSees nonlinear self-test reported unavailable")


def check_opensees_analyser_pipeline() -> None:
    _require_opensees()
    output = _run_subprocess("opensees_analyser.py")
    if "SELF-TEST PASSED" not in output:
        raise AssertionError("opensees_analyser.py did not report SELF-TEST PASSED")


def check_damage_model() -> None:
    _import("damage_model").run_self_test(verbose=False)


def check_rainflow_counter() -> None:
    _import("rainflow_counter").run_self_test(verbose=False)


def check_environmental_model() -> None:
    _import("environmental_model").run_self_test(verbose=False)


def check_verify_mass() -> None:
    _require_opensees()
    output = _run_subprocess("verify_mass.py")
    if "RESULT:" not in output:
        raise AssertionError("verify_mass.py did not print a result summary")


def check_sensor_traffic_spectrum() -> None:
    _import("sensor_reader").run_self_test()


def check_sensor_weight_multiplier() -> None:
    sensor_reader = _import("sensor_reader")
    reader = sensor_reader.SensorReader(sensor_reader.ConnectionConfig(mode="sim"))
    reader.set_weight_multiplier(10.0)
    assert reader.config.weight_multiplier == 10.0
    reader.set_weight_multiplier(100.0)
    assert reader.config.weight_multiplier == 100.0
    reader.set_weight_multiplier(10.0)

    state = sensor_reader._HWState()
    times = iter([0.00, 0.35, 0.85])
    original_monotonic = sensor_reader.time.monotonic
    sensor_reader.time.monotonic = lambda: next(times)
    try:
        reader._process_frame(
            reader._parse_websocket_json(
                '{"weight": 2, "pressure": 1000, "speed": 1.0}'
            ),
            state,
        )
        reader._process_frame(
            reader._parse_websocket_json(
                '{"weight": 2, "pressure": 0, "speed": 1.0}'
            ),
            state,
        )
        tick = reader.current_tick
        assert tick is not None, tick
        assert tick.weight_kg == 20.0, tick
        assert tick.metadata["raw_weight_kg"] == 2.0, tick.metadata
        assert tick.metadata["weight_multiplier"] == 10.0, tick.metadata

        reader._process_frame(
            reader._parse_websocket_json(
                '{"weight": 0, "pressure": 0, "speed": 1.0}'
            ),
            state,
        )
        vp = reader.latest_pass
        assert vp is not None, vp
        assert vp.weight_kg > 20.0, vp
        assert vp.metadata["raw_weight_kg"] == 2.0, vp.metadata
        assert vp.metadata["weight_multiplier"] == 10.0, vp.metadata
    finally:
        sensor_reader.time.monotonic = original_monotonic


def check_sensor_websocket_parser() -> None:
    sensor_reader = _import("sensor_reader")
    reader = sensor_reader.SensorReader(
        sensor_reader.ConnectionConfig(
            mode="sim",
            gauge_channel_map={0: 1, 1: 5, 2: 9, 3: 13},
        )
    )
    fields = reader._parse_websocket_json(
        '{"weight": 259, "pressure": 1757, "t": 48372, '
        '"speed": 0.87, "crossing": 1, "strain0": 12.5, '
        '"strain1": -7.25, "strain2": 4.5, "strain3": 9.75, '
        '"id": "abc"}'
    )
    assert fields["W"] == "259", fields
    assert fields["P"] == "1", fields
    assert fields["pressure"] == "1757", fields
    assert fields["speed"] == "0.87", fields
    assert fields["t"] == "48372", fields
    assert fields["id"] == "abc", fields
    assert fields["S0"] == "12.5", fields
    assert fields["S1"] == "-7.25", fields
    assert fields["S2"] == "4.5", fields
    assert fields["S3"] == "9.75", fields

    fields = reader._parse_websocket_json(
        '{"weight": 259, "pressure": 1757, "t": 48372, '
        '"speed": 0.87, "strain1": 12.5, '
        '"strain2": -7.25, "strain3": 4.5, "strain4": 9.75, '
        '"id": "abc"}'
    )
    assert fields["S0"] == "12.5", fields
    assert fields["S1"] == "-7.25", fields
    assert fields["S2"] == "4.5", fields
    assert fields["S3"] == "9.75", fields

    fields = reader._parse_websocket_json(
        '{"weight": 259, "pressure": 1757, "t": 48372, '
        '"speed": 0.87, "strain0": 12.5}'
    )
    assert fields["P"] == "1", fields

    state = sensor_reader._HWState()
    times = iter([0.0, 0.10, 0.22])
    original_monotonic = sensor_reader.time.monotonic
    sensor_reader.time.monotonic = lambda: next(times)
    try:
        reader._process_frame(
            reader._parse_websocket_json('{"weight": 0, "pressure": 1757}'),
            state,
        )
        reader._process_frame(
            reader._parse_websocket_json('{"weight": 259, "pressure": 1757}'),
            state,
        )
        assert abs(state.last_speed_ms - 1.4) < 1e-9, state.last_speed_ms
        tick = reader.current_tick
        assert tick is not None, tick
        assert tick.in_transit, tick
        assert tick.position_frac == 0.0, tick
        assert tick.weight_kg == 259.0, tick
        assert tick.metadata.get("phase") == "approach", tick.metadata

        reader._process_frame(
            reader._parse_websocket_json('{"weight": 0, "pressure": 0}'),
            state,
        )
        tick = reader.current_tick
        assert tick is not None, tick
        assert tick.in_transit, tick
        assert tick.speed_ms is not None and abs(tick.speed_ms - 1.4) < 1e-9, tick
        assert 0.0 <= tick.position_frac <= 1.0, tick
        assert tick.weight_kg == 259.0, tick
    finally:
        sensor_reader.time.monotonic = original_monotonic

    reader = sensor_reader.SensorReader(
        sensor_reader.ConnectionConfig(
            mode="sim",
            gauge_channel_map={0: 1, 1: 5, 2: 9, 3: 13},
        )
    )
    state = sensor_reader._HWState()
    times = iter([10.00, 10.05, 10.10, 10.14, 10.35])
    original_monotonic = sensor_reader.time.monotonic
    sensor_reader.time.monotonic = lambda: next(times)
    try:
        reader._process_frame(
            reader._parse_websocket_json(
                '{"weight": 0, "pressure": 1000, "speed": 1.0}'
            ),
            state,
        )
        assert state.contact_start == 10.00, state.contact_start

        reader._process_frame(
            reader._parse_websocket_json(
                '{"weight": 0, "pressure": 0, "speed": 1.0}'
            ),
            state,
        )
        assert reader.current_tick is None, reader.current_tick

        reader._process_frame(
            reader._parse_websocket_json(
                '{"weight": 0, "pressure": 1200, "speed": 1.0}'
            ),
            state,
        )
        assert state.contact_start == 10.00, state.contact_start

        reader._process_frame(
            reader._parse_websocket_json(
                '{"weight": 42, "pressure": 0, "speed": 1.0}'
            ),
            state,
        )
        assert state.peak_weight == 42.0, state.peak_weight
        tick = reader.current_tick
        assert tick is not None, tick
        assert tick.position_frac == 0.0, tick
        assert tick.weight_kg == 42.0, tick
        assert tick.metadata.get("phase") == "approach", tick.metadata

        reader._process_frame(
            reader._parse_websocket_json(
                '{"weight": 0, "pressure": 0, "speed": 1.0}'
            ),
            state,
        )
        tick = reader.current_tick
        assert tick is not None, tick
        assert tick.in_transit, tick
        assert tick.speed_ms == 1.0, tick
        assert tick.weight_kg == 42.0, tick
    finally:
        sensor_reader.time.monotonic = original_monotonic


def check_feedback_payload() -> None:
    sensor_reader = _import("sensor_reader")
    cmd = sensor_reader.FeedbackCommand(
        max_load_kg=4.2,
        safe_speed_ms=0.7,
        advisory="reduce_speed",
        alert_level="WARNING",
        reason="test warning",
        timestamp_utc="2026-06-09T00:00:00Z",
        strain_values=[70.0, -140.0, 210.0, 280.0],
        safe_to_pass=False,
    )
    payload = cmd.payload()
    assert payload["maxLoad"] == 4.2
    assert payload["safeToPass"] == 0
    assert payload["twin1"] == 70.0
    assert payload["twin2"] == -140.0
    assert payload["twin3"] == 210.0
    assert payload["twin4"] == 280.0
    assert payload["averageStrainTwin"] == 105.0

    legacy = sensor_reader.FeedbackCommand(
        max_load_kg=4.2,
        stress_values=[1.1, -2.2, 3.3, 4.4],
        strain_value=12.5,
    ).payload()
    assert legacy["safeToPass"] == 1
    assert legacy["twin1"] == 1.1
    assert legacy["averageStrainTwin"] == 12.5


def check_damage_correction_batch_snapshot() -> None:
    damage_model = _import("damage_model")
    model = damage_model.DamageModel(n_members=1)
    model.record_pass_simple({0: 142e6}, n_cycles=1)
    batch = model.simple_increment_snapshot()
    model.record_pass_simple({0: 142e6}, n_cycles=1)
    before = model.get_damage(0)
    model.record_pass_accurate(
        {0: [(71e6, 0.0, 1.0)]},
        simple_increments_to_replace=batch,
        crack_state_to_replace=model.crack_state_snapshot(),
    )
    remaining = model.simple_increment_snapshot()
    assert remaining[0] > 0.0
    assert remaining[0] < before


def check_sensor_residuals() -> None:
    validation = _import("sensor_validation")
    ok = validation.compute_sensor_residuals(
        {1: 1000.0}, {1: 69e6}, [1], 69e9)[0]
    assert ok.status == "ok", ok
    drift = validation.compute_sensor_residuals(
        {1: 1500.0}, {1: 69e6}, [1], 69e9)[0]
    assert drift.status == "drift", drift
    missing = validation.compute_sensor_residuals(
        {}, {1: 69e6}, [1], 69e9)[0]
    assert missing.status == "missing", missing
    outlier = validation.compute_sensor_residuals(
        {1: 2500.0}, {1: 69e6}, [1], 69e9)[0]
    assert outlier.status == "outlier", outlier


def check_manager_fake_analyser_result_contract() -> None:
    import numpy as np
    bridge_manager = _import("bridge_manager")
    damage_model = _import("damage_model")
    environmental_model = _import("environmental_model")
    sensor_reader = _import("sensor_reader")
    safety_checker = _import("safety_checker")
    opensees_models = _import("opensees_models")

    class FakeAnalyser:
        is_3d_frame = True

        def run_crossing(self, **_kwargs):
            return opensees_models.CrossingResult(
                peak_stresses={0: 142e6},
                stress_histories={0: np.array([0.0, 142e6, 0.0])},
                natural_frequencies=[12.0],
                dynamic_amplification_factor=1.2,
                time_vector=np.array([0.0, 0.1, 0.2]),
                is_dynamic=True,
                steps_completed=3,
                converged=True,
            )

    damage = damage_model.DamageModel(n_members=1)
    reader = sensor_reader.SensorReader(sensor_reader.ConnectionConfig(mode="sim"))
    mgr = bridge_manager.DynamicAnalysisManager(
        damage, environmental_model.EnvironmentalModel(),
        reader, safety_checker.SafetyChecker(), passes_per_trigger=1)
    mgr.set_analyser(FakeAnalyser())
    vp = sensor_reader.VehiclePass(
        weight_kg=1.0, speed_ms=0.5, axle_position_frac=0.5,
        strain_readings={})
    damage.record_pass_simple({0: 142e6}, n_cycles=1)
    mgr.enqueue_pass(vp, {0: 142e6 * 2.25e-6}, 2.25e-6)
    item = mgr._work_queue.get_nowait()
    mgr._work_queue.put_nowait(item)
    mgr._stop.set()
    mgr._stop.clear()
    # Run one worker iteration by starting/stopping the thread path.
    mgr.start()
    import time
    deadline = time.monotonic() + 2.0
    result = None
    while time.monotonic() < deadline:
        result = mgr.poll_results()
        if result:
            break
        time.sleep(0.01)
    mgr.stop()
    assert result is not None
    assert result["apply_accurate_damage"] is True
    assert result["daf_calibration"] == (0.5, 1.2)
    assert result["analysis_mode"] == "OpenSees dynamic"


def check_environment_capacity_scaling() -> None:
    environmental_model = _import("environmental_model")
    model = environmental_model.EnvironmentalModel()
    base = model.get_degraded_properties()
    model.set_exposure("outdoor", humidity_rh=0.80)
    model.advance_time(hours=5000, n_temp_cycles=200, delta_T_C=25.0)
    degraded = model.get_degraded_properties()
    assert degraded.yield_pa < base.yield_pa
    assert degraded.fatigue_limit_pa < base.fatigue_limit_pa
    safe_load = 10.0
    assert safe_load * degraded.yield_pa / base.yield_pa < safe_load


CHECKS: List[tuple[str, CheckFn]] = [
    ("fem_solver: 2D analytical truss", check_fem_2d),
    ("fem_solver: 3D beam deflection", check_fem_3d),
    ("OpenSees: 3D beam bending moment", check_opensees_beam_3d),
    ("OpenSees: geometric nonlinearity comparison", check_opensees_geometric_nonlinearity),
    ("OpenSees: analyser static/dynamic pipeline", check_opensees_analyser_pipeline),
    ("damage_model: Miner + Paris crack growth", check_damage_model),
    ("rainflow_counter: sine wave cycle counting", check_rainflow_counter),
    ("environmental_model: degradation and floors", check_environmental_model),
    ("verify_mass: mass and frequency sanity", check_verify_mass),
    ("sensor_reader: traffic spectrum sampling", check_sensor_traffic_spectrum),
    ("sensor_reader: demo weight multiplier", check_sensor_weight_multiplier),
    ("sensor_reader: WebSocket JSON parser", check_sensor_websocket_parser),
    ("sensor_reader: feedback payload", check_feedback_payload),
    ("damage_model: batch correction snapshot", check_damage_correction_batch_snapshot),
    ("sensor_validation: residual classification", check_sensor_residuals),
    ("bridge_manager: fake analyser result contract", check_manager_fake_analyser_result_contract),
    ("environmental_model: capacity scaling inputs", check_environment_capacity_scaling),
]


def run_all() -> List[TestResult]:
    results: List[TestResult] = []
    for name, check in CHECKS:
        try:
            check()
        except SkipTest as exc:
            results.append(TestResult(name, "SKIP", str(exc)))
        except Exception as exc:
            tb = traceback.format_exc(limit=4).strip()
            reason = f"{exc}\n{tb}" if tb else str(exc)
            results.append(TestResult(name, "FAIL", reason))
        else:
            results.append(TestResult(name, "PASS"))
    return results


def main() -> int:
    results = run_all()
    for result in results:
        if result.reason:
            print(f"{result.status:<4} {result.name} -- {result.reason.splitlines()[0]}")
        else:
            print(f"{result.status:<4} {result.name}")

    passed = sum(r.status == "PASS" for r in results)
    failed = sum(r.status == "FAIL" for r in results)
    skipped = sum(r.status == "SKIP" for r in results)

    print()
    print(f"Total passed : {passed}")
    print(f"Total failed : {failed}")
    print(f"Total skipped: {skipped}")

    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print()
        print("Failures:")
        for result in failures:
            print(f"- {result.name}")
            print(result.reason)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


