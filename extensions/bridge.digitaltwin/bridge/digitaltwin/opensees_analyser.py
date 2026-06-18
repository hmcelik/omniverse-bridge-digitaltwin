# High-level wrapper around the OpenSees bridge solvers.
#
# OpenSeesAnalyser chooses the 3D frame path when available and falls back to the
# older 2D truss/static path when OpenSees cannot run. The extension calls this
# from the background analysis worker, not from the UI thread.

from __future__ import annotations

import os as _os, sys as _sys
if __name__ == "__main__":
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .opensees_models import CrossingResult, AnalyserConfig, _OPS_AVAILABLE, _GRAVITY, ops
    from .opensees_static import static_at_position, static_fallback
    from .opensees_dynamic import (
        run_dynamic, run_dynamic_3d_frame, run_3d_frame_beam_self_test,
        FrameMember3D, _bottom_groups_3d, _build_3d_frame_domain,
        _combined_frame_stress_from_response, _distribute_line_load_3d,
    )
    from .bridge_config import DENSITY as _DENSITY_DEFAULT, YIELD_STRENGTH as _YIELD_DEFAULT
except ImportError:
    from opensees_models import CrossingResult, AnalyserConfig, _OPS_AVAILABLE, _GRAVITY, ops  # type: ignore[no-redef]
    from opensees_static import static_at_position, static_fallback  # type: ignore[no-redef]
    from opensees_dynamic import (  # type: ignore[no-redef]
        run_dynamic, run_dynamic_3d_frame, run_3d_frame_beam_self_test,
        FrameMember3D, _bottom_groups_3d, _build_3d_frame_domain,
        _combined_frame_stress_from_response, _distribute_line_load_3d,
    )
    from bridge_config import DENSITY as _DENSITY_DEFAULT, YIELD_STRENGTH as _YIELD_DEFAULT  # type: ignore[no-redef]

# Re-export so existing callers (e.g. extension.py) keep working without change.
__all__ = ["OpenSeesAnalyser", "CrossingResult", "AnalyserConfig"]


class OpenSeesAnalyser:

    _ops_lock = threading.Lock()   # OpenSees uses global state

    def __init__(
        self,
        nodes_2d:             List[Tuple[float, float]],
        members:              List[Tuple[int, int, float, float]],
        fixed_dofs:           List[int],
        density:              float = _DENSITY_DEFAULT,
        yield_strength_pa:    float = _YIELD_DEFAULT,
        config:               Optional[AnalyserConfig] = None,
        member_mass_override: Optional[float] = None,
        nodes_3d:             Optional[List[Tuple[float, float, float]]] = None,
        members_3d:           Optional[List[FrameMember3D]] = None,
        fixed_dofs_3d:        Optional[List[int]] = None,
    ) -> None:
        self.nodes                = nodes_2d
        self.members              = members
        self.fixed_dofs           = set(fixed_dofs)
        self.nodes_3d             = nodes_3d
        self.members_3d           = members_3d
        self.fixed_dofs_3d        = set(fixed_dofs_3d or [])
        self.is_3d_frame          = bool(nodes_3d and members_3d and fixed_dofs_3d)
        self.density              = density
        self.yield_strength       = yield_strength_pa
        self.cfg                  = config or AnalyserConfig()
        self.member_mass_override = member_mass_override
        self._nat_freqs: List[float] = []
        # TrussFEM cache for static reference -- built once, reused every call
        self._fem_cache = None
        self._init_fem_cache()


    def _init_fem_cache(self) -> None:
        try:
            try:
                from .fem_solver import TrussFEM as _FEM
            except ImportError:
                from fem_solver import TrussFEM as _FEM  # standalone __main__
            self._fem_cache = _FEM(
                nodes=self.nodes,
                members=self.members,
                fixed_dofs=sorted(self.fixed_dofs),
                yield_strength=self.yield_strength,
            )
        except Exception as exc:
            print(f"[opensees_analyser] TrussFEM cache unavailable ({exc}); "
                  "static reference will use inline DSM fallback.")
            self._fem_cache = None


    def get_natural_frequencies(self) -> List[float]:
        return list(self._nat_freqs)

    def check_resonance(self, speed_ms: float, bridge_length_m: float = 0.5) -> bool:
        if not self._nat_freqs:
            return False
        f_cross = speed_ms / bridge_length_m
        for f_n in self._nat_freqs[:2]:
            if abs(f_cross - f_n) / max(f_n, 1e-6) < 0.15:
                return True
        return False

    def run_static_position(
        self,
        weight_kg:       float,
        x_frac:          float,
        lateral_frac:    float = 0.5,
        bridge_length_m: float = 0.5,
        E_override:      Optional[float] = None,
        yield_override:  Optional[float] = None,
    ) -> CrossingResult:
        load_n = weight_kg * _GRAVITY
        x_frac = max(0.0, min(1.0, x_frac))
        lateral_frac = max(0.0, min(1.0, lateral_frac))

        if not _OPS_AVAILABLE or ops is None:
            stresses = static_at_position(
                self.nodes, self.members,
                load_n * (1.0 - lateral_frac),
                x_frac * bridge_length_m,
                self._fem_cache,
            )
            return CrossingResult(
                peak_stresses=stresses,
                stress_histories={m: np.array([s]) for m, s in stresses.items()},
                natural_frequencies=list(self._nat_freqs),
                dynamic_amplification_factor=1.0,
                time_vector=np.array([0.0]),
                is_dynamic=False,
                steps_completed=1,
                converged=True,
            )

        if self.member_mass_override is not None and self.members:
            avg_area = sum(m[2] for m in self.members) / len(self.members)
            density_for_mass = self.member_mass_override / max(avg_area, 1e-12)
        else:
            density_for_mass = self.density

        with self._ops_lock:
            try:
                if self.is_3d_frame:
                    nodes = self.nodes_3d or []
                    members = self.members_3d or []
                    left_nodes, right_nodes = _bottom_groups_3d(nodes)
                    all_bottom = left_nodes + right_nodes
                    if not all_bottom:
                        raise RuntimeError("No bottom nodes available for 3D static solve.")
                    x_min = min(nodes[i][0] for i in all_bottom)
                    x_max = max(nodes[i][0] for i in all_bottom)
                    pos_x = x_min + x_frac * (x_max - x_min)

                    _build_3d_frame_domain(
                        nodes, members, self.fixed_dofs_3d,
                        density_for_mass, E_override,
                    )
                    ops.timeSeries("Linear", 1)
                    ops.pattern("Plain", 1, 1)
                    loads: Dict[int, float] = {}
                    for node, val in _distribute_line_load_3d(
                        nodes, left_nodes, pos_x, load_n * (1.0 - lateral_frac)
                    ).items():
                        loads[node] = loads.get(node, 0.0) + val
                    for node, val in _distribute_line_load_3d(
                        nodes, right_nodes, pos_x, load_n * lateral_frac
                    ).items():
                        loads[node] = loads.get(node, 0.0) + val
                    for node, fz in loads.items():
                        ops.load(node + 1, 0.0, 0.0, float(fz), 0.0, 0.0, 0.0)

                    ops.system("BandGeneral")
                    ops.numberer("RCM")
                    ops.constraints("Transformation")
                    ops.integrator("LoadControl", 1.0)
                    ops.algorithm("Linear")
                    ops.analysis("Static")
                    if ops.analyze(1) != 0:
                        raise RuntimeError("OpenSees 3D static analysis failed.")

                    stresses: Dict[int, float] = {}
                    for local_idx, member in enumerate(members, start=1):
                        area, ixx, iyy = member[2], member[4], member[5]
                        resp = ops.eleResponse(local_idx, "localForce")
                        stresses[member[-1]] = _combined_frame_stress_from_response(
                            resp, area, ixx, iyy)
                else:
                    stresses = static_at_position(
                        self.nodes, self.members,
                        load_n * (1.0 - lateral_frac),
                        x_frac * bridge_length_m,
                        self._fem_cache,
                    )

                return CrossingResult(
                    peak_stresses=stresses,
                    stress_histories={m: np.array([s]) for m, s in stresses.items()},
                    natural_frequencies=list(self._nat_freqs),
                    dynamic_amplification_factor=1.0,
                    time_vector=np.array([0.0]),
                    is_dynamic=False,
                    steps_completed=1,
                    converged=True,
                )
            finally:
                try:
                    ops.wipe()
                except Exception:
                    pass

    def run_crossing(
        self,
        weight_kg:       float,
        speed_ms:        float,
        lateral_frac:    float = 0.5,
        bridge_length_m: float = 0.5,
        E_override:      Optional[float] = None,
        yield_override:  Optional[float] = None,
    ) -> CrossingResult:
        if not _OPS_AVAILABLE:
            return static_fallback(
                self.nodes, self.members, self.fixed_dofs,
                self.yield_strength, self.cfg, self._fem_cache,
                weight_kg, speed_ms, lateral_frac, bridge_length_m, yield_override)

        # When member_mass_override is set, back-calculate an effective density
        # so that _member_mass_kg(area, length, eff_density) = override * length.
        # This keeps run_dynamic's signature unchanged while decoupling mass
        # from the stiffness area stored in each member tuple.
        if self.member_mass_override is not None and self.members:
            avg_area = sum(m[2] for m in self.members) / len(self.members)
            density_for_mass = self.member_mass_override / max(avg_area, 1e-12)
        else:
            density_for_mass = self.density

        with self._ops_lock:
            try:
                if self.is_3d_frame:
                    result, nat_freqs = run_dynamic_3d_frame(
                        self.nodes_3d or [], self.members_3d or [],
                        self.fixed_dofs_3d, density_for_mass, self.cfg,
                        self.yield_strength, weight_kg, speed_ms,
                        lateral_frac, bridge_length_m, E_override, yield_override,
                    )
                else:
                    result, nat_freqs = run_dynamic(
                        self.nodes, self.members, self.fixed_dofs,
                        density_for_mass, self.cfg, self.yield_strength, self._fem_cache,
                        weight_kg, speed_ms, lateral_frac,
                        bridge_length_m, E_override, yield_override,
                    )
                self._nat_freqs = nat_freqs
                return result
            except RuntimeError as exc:
                if self.is_3d_frame:
                    print(f"[opensees_analyser] 3D frame solve failed; "
                          f"falling back to 2D/static path ({exc})")
                    try:
                        result, nat_freqs = run_dynamic(
                            self.nodes, self.members, self.fixed_dofs,
                            density_for_mass, self.cfg, self.yield_strength,
                            self._fem_cache, weight_kg, speed_ms, lateral_frac,
                            bridge_length_m, E_override, yield_override,
                        )
                        self._nat_freqs = nat_freqs
                        return result
                    except RuntimeError:
                        pass
                return static_fallback(
                    self.nodes, self.members, self.fixed_dofs,
                    self.yield_strength, self.cfg, self._fem_cache,
                    weight_kg, speed_ms, lateral_frac, bridge_length_m, yield_override)


# Self-test
if __name__ == "__main__":
    # Self-test has two checks:
    # 1. Slow-speed response should match the static result closely.
    # 2. Dynamic analysis should return frequencies, DAF, and converged steps.
    run_3d_frame_beam_self_test(verbose=True)

    # Values sourced from bridge_config so self-test uses the live configuration.
    try:
        from bridge_config import (
            NUM_PANELS as PANELS, REAL_PANEL as PANEL_LEN, REAL_TRUSS_HEIGHT as HEIGHT,
            MEMBER_AREA as AREA, E_MODULUS as E_MOD, DENSITY, YIELD_STRENGTH as YIELD,
            REAL_BRIDGE_LENGTH as BRIDGE_L, MEMBER_MASS_PER_UNIT_LENGTH as MULL,
        )
    except ImportError:
        PANELS, PANEL_LEN, HEIGHT = 5, 0.10, 0.15
        AREA, E_MOD, DENSITY, YIELD = 0.0015 * 0.0015, 69e9, 2700.0, 270e6
        BRIDGE_L = 0.5
        MULL = AREA * DENSITY

    bottom = [(i * PANEL_LEN, 0.0) for i in range(PANELS + 1)]
    top    = [((i + 0.5) * PANEL_LEN, HEIGHT) for i in range(PANELS)]
    nodes  = bottom + top

    members: list = []
    for i in range(PANELS):                      # bottom chord
        members.append((i, i + 1, AREA, E_MOD))
    for i in range(PANELS - 1):                  # top chord
        members.append((PANELS + 1 + i, PANELS + 2 + i, AREA, E_MOD))
    for i in range(PANELS):                      # diagonals
        members.append((i,          PANELS + 1 + i, AREA, E_MOD))
        members.append((PANELS+1+i, i + 1,          AREA, E_MOD))

    fixed_dofs = [0, 1, 2 * PANELS + 1]   # pin at node 0, roller at last bottom node

    LOAD_N = 500.0   # N on this truss plane (half of a 1 kN vehicle, lateral_frac=0)

    # n_modes=0 skips eigenvalue analysis and the time-step refinement so the
    # step count stays manageable for a test runner.  dt=0.01 s with a 25-second
    # crossing (0.02 m/s over 0.5 m) gives 2500 steps -- fast enough for a test.
    print("Test 1: Static convergence (0.02 m/s, dt=0.01 s, n_modes=0) ...")
    slow_cfg = AnalyserConfig(dt_target=0.01, n_modes=0)
    analyser  = OpenSeesAnalyser(
        nodes_2d=nodes, members=members, fixed_dofs=fixed_dofs,
        density=DENSITY, yield_strength_pa=YIELD, config=slow_cfg,
        member_mass_override=MULL,
    )

    try:
        from opensees_static import static_envelope as _static_envelope
    except ImportError:
        _static_envelope = None

    static_env = (
        _static_envelope(nodes, members, LOAD_N, BRIDGE_L, analyser._fem_cache)
        if _static_envelope else {}
    )
    if not static_env:
        print("  SKIP: TrussFEM unavailable (cannot compute ground truth)")
        import sys; sys.exit(0)

    # lateral_frac=0.0 => load_n = LOAD_N * (1-0) = LOAD_N
    cr_slow = analyser.run_crossing(
        weight_kg=LOAD_N / _GRAVITY,
        speed_ms=0.02,
        lateral_frac=0.0,
        bridge_length_m=BRIDGE_L,
    )

    if not cr_slow.is_dynamic:
        print("  SKIP: OpenSeesPy unavailable -- static fallback used")
    else:
        assert cr_slow.converged, "Slow crossing must converge"

        # Only compare members with meaningful static stress (> 1% of LOAD_N/area)
        threshold = LOAD_N / AREA * 0.01
        significant = [(m, s) for m, s in static_env.items() if s > threshold]
        assert significant, "At least one significant member must exist"

        failed = False
        for m_idx, static_s in significant:
            dyn_s = cr_slow.peak_stresses.get(m_idx, 0.0)
            err   = abs(dyn_s - static_s) / max(static_s, 1e-9)
            if err >= 0.05:
                print(f"  FAIL member {m_idx}: "
                      f"static={static_s/1e6:.4f} MPa  "
                      f"dynamic={dyn_s/1e6:.4f} MPa  "
                      f"error={err*100:.1f}%  (limit 5%)")
                failed = True
        if failed:
            import sys; sys.exit(1)

        max_err = max(
            abs(cr_slow.peak_stresses.get(m, 0.0) - s) / max(s, 1e-9)
            for m, s in significant
        )
        print(f"  OK -- {len(significant)} members checked, "
              f"max error {max_err*100:.2f}% (limit 5%)")

    # Use n_modes=2 to limit the highest tracked frequency and keep step counts
    # below _MAX_STEPS even for stiff prototype-scale trusses.
    print("Test 2: Dynamic pipeline (0.5 m/s, n_modes=2) ...")
    dyn_cfg   = AnalyserConfig(dt_target=0.005, n_modes=2)
    analyser2 = OpenSeesAnalyser(
        nodes_2d=nodes, members=members, fixed_dofs=fixed_dofs,
        density=DENSITY, yield_strength_pa=YIELD, config=dyn_cfg,
        member_mass_override=MULL,
    )

    cr_dyn = analyser2.run_crossing(
        weight_kg=LOAD_N / _GRAVITY,
        speed_ms=0.5,
        lateral_frac=0.0,
        bridge_length_m=BRIDGE_L,
    )

    if not cr_dyn.is_dynamic:
        print("  SKIP: OpenSeesPy unavailable")
    else:
        assert cr_dyn.converged, (
            f"Pipeline test must converge "
            f"(completed {cr_dyn.steps_completed} steps)")
        assert cr_dyn.dynamic_amplification_factor >= 1.0, (
            f"DAF={cr_dyn.dynamic_amplification_factor:.4f} must be >= 1.0")
        assert cr_dyn.steps_completed > 0, "steps_completed must be > 0"

        if cr_dyn.natural_frequencies:
            f1 = cr_dyn.natural_frequencies[0]
            assert f1 > 0, "f1 must be positive"
            freqs_str = ", ".join(f"{f:.1f} Hz"
                                  for f in cr_dyn.natural_frequencies)
            print(f"  Natural frequencies: [{freqs_str}]")

        print(f"  DAF: {cr_dyn.dynamic_amplification_factor:.4f}  (>= 1.0 required)")
        print(f"  Steps: {cr_dyn.steps_completed}   converged: {cr_dyn.converged}")
        print("  OK")

    print("SELF-TEST PASSED")



