# OpenSees dynamic solvers for vehicle crossings.
#
# This module builds OpenSees models, applies moving loads, runs Newmark time
# integration, and converts element responses into stress histories. The caller
# uses these histories for DAF calibration and rainflow fatigue correction.
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .opensees_models import (
        CrossingResult, AnalyserConfig,
        _GRAVITY, _NEWMARK_GAMMA, _NEWMARK_BETA,
        _MAX_STEPS, _MIN_NODE_MASS_KG, _MAX_FREQ_FOR_DT_HZ,
        ops,
    )
    from .opensees_static import _member_mass_kg, static_envelope, static_inline
    from .bridge_config import (
        MEMBER_W, MEMBER_H, MEMBER_IXX, MEMBER_IYY, MEMBER_J, E_MODULUS, G_MODULUS,
        DENSITY, YIELD_STRENGTH,
        NONLINEAR_SELF_TEST_LOAD_N, NONLINEAR_SELF_TEST_SPAN_M,
        NONLINEAR_SELF_TEST_SPEED_MS, NONLINEAR_SELF_TEST_DT,
        NONLINEAR_SELF_TEST_MAX_DIFF,
    )
except ImportError:
    from opensees_models import (  # type: ignore[no-redef]
        CrossingResult, AnalyserConfig,
        _GRAVITY, _NEWMARK_GAMMA, _NEWMARK_BETA,
        _MAX_STEPS, _MIN_NODE_MASS_KG, _MAX_FREQ_FOR_DT_HZ,
        ops,
    )
    from opensees_static import _member_mass_kg, static_envelope, static_inline  # type: ignore[no-redef]
    from bridge_config import (  # type: ignore[no-redef]
        MEMBER_W, MEMBER_H, MEMBER_IXX, MEMBER_IYY, MEMBER_J, E_MODULUS, G_MODULUS,
        DENSITY, YIELD_STRENGTH,
        NONLINEAR_SELF_TEST_LOAD_N, NONLINEAR_SELF_TEST_SPAN_M,
        NONLINEAR_SELF_TEST_SPEED_MS, NONLINEAR_SELF_TEST_DT,
        NONLINEAR_SELF_TEST_MAX_DIFF,
    )


FrameMember3D = Tuple[int, int, float, float, float, float, float, float, int]


def run_dynamic(
    nodes:           List[Tuple[float, float]],
    members:         List[Tuple[int, int, float, float]],
    fixed_dofs:      Set[int],
    density:         float,
    cfg:             AnalyserConfig,
    yield_strength:  float,
    fem_cache:       Optional[object],
    weight_kg:       float,
    speed_ms:        float,
    lateral_frac:    float,
    bridge_length_m: float,
    E_override:      Optional[float],
    yield_override:  Optional[float],
) -> Tuple[CrossingResult, List[float]]:
    load_n = weight_kg * _GRAVITY * (1.0 - lateral_frac)

    # Trivial case: vehicle entirely on the other truss
    if load_n < 1e-9:
        n_m = len(members)
        result = CrossingResult(
            peak_stresses={m: 0.0 for m in range(n_m)},
            stress_histories={m: np.zeros(2) for m in range(n_m)},
            natural_frequencies=[],
            dynamic_amplification_factor=1.0,
            time_vector=np.array([0.0, bridge_length_m / max(speed_ms, 1e-3)]),
            is_dynamic=True, steps_completed=1, converged=True,
        )
        return result, []

    cross_duration = bridge_length_m / max(speed_ms, 1e-3)
    dt = min(cfg.dt_target, cross_duration / 20.0)
    n_steps = max(10, int(math.ceil(cross_duration / dt)))
    dt = cross_duration / n_steps

    n_nodes   = len(nodes)
    n_dof     = 2 * n_nodes
    free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]

    nat_freqs: List[float] = []
    steps_completed = 0
    converged       = True

    try:
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 2)

        for i, (x, y) in enumerate(nodes, start=1):
            ops.node(i, x, y)

        # Per-node flag dict avoids calling ops.fix() twice on the same node.
        fix_map: Dict[int, List[int]] = {}
        for d in fixed_dofs:
            n_id   = d // 2 + 1
            dof_id = d % 2
            if n_id not in fix_map:
                fix_map[n_id] = [0, 0]
            fix_map[n_id][dof_id] = 1
        for n_id, flags in fix_map.items():
            ops.fix(n_id, *flags)

        for m_idx, (ni, nj, area, E_orig) in enumerate(members, start=1):
            E_use = E_override if E_override is not None else E_orig
            ops.uniaxialMaterial('Elastic', m_idx, float(E_use))

        # Pre-compute totals before calling ops.mass() -- ops.mass() ADDS to
        # existing mass rather than setting it, so internal nodes shared by
        # multiple members would otherwise receive 3-4x the correct mass,
        # inflating f1 by ~sqrt(3).
        nodal_mass: Dict[int, float] = {}
        for ni, nj, area, E_orig in members:
            half = _member_mass_kg(nodes[ni], nodes[nj], area, density) / 2.0
            nodal_mass[ni] = nodal_mass.get(ni, 0.0) + half
            nodal_mass[nj] = nodal_mass.get(nj, 0.0) + half

        for m_idx, (ni, nj, area, E_orig) in enumerate(members, start=1):
            ops.element('Truss', m_idx, ni + 1, nj + 1, float(area), m_idx)

        # Set nodal mass exactly once per node with _MIN_NODE_MASS_KG floor.
        for node_idx, mass_val in nodal_mass.items():
            m = max(mass_val, _MIN_NODE_MASS_KG)
            ops.mass(node_idx + 1, m, m)

        n_modes_safe = min(cfg.n_modes, len(free_dofs))
        if n_modes_safe > 0:
            try:
                lambdas = ops.eigen(n_modes_safe)
                nat_freqs = [
                    math.sqrt(max(lam, 0.0)) / (2 * math.pi)
                    for lam in lambdas
                ]
            except Exception:
                nat_freqs = []

        # Refine dt only to resolve modes that vehicle loads can excite
        # (crossing frequency ~1-5 Hz -> cap at _MAX_FREQ_FOR_DT_HZ = 50 Hz).
        # Modes above that cap are not excited by traffic and do not need
        # resolving; attempting to do so would hit _MAX_STEPS at ~1682 Hz.
        if nat_freqs:
            relevant = [f for f in nat_freqs if f <= _MAX_FREQ_FOR_DT_HZ]
            if relevant:
                f_for_dt     = relevant[-1]
                dt_accuracy  = 1.0 / (10.0 * f_for_dt)
                if dt > dt_accuracy:
                    n_steps_refined = max(
                        10, int(math.ceil(cross_duration / dt_accuracy)))
                    n_steps_refined = min(n_steps_refined, _MAX_STEPS)
                    dt_new = cross_duration / n_steps_refined
                    print(f"[opensees_analyser] dt refined "
                          f"{dt*1e3:.3f} ms -> {dt_new*1e3:.3f} ms "
                          f"to resolve f={f_for_dt:.1f} Hz "
                          f"({n_steps_refined} steps)")
                    dt      = dt_new
                    n_steps = n_steps_refined

        if len(nat_freqs) >= 2:
            w1, w2 = (2 * math.pi * f for f in nat_freqs[:2])
            alpha  = 2 * cfg.damping_ratio * w1 * w2 / (w1 + w2)
            beta   = 2 * cfg.damping_ratio / (w1 + w2)
        elif len(nat_freqs) == 1:
            w1 = 2 * math.pi * nat_freqs[0]
            alpha, beta = cfg.damping_ratio * w1, 0.0
        else:
            alpha, beta = 0.0, 0.0
        ops.rayleigh(alpha, 0.0, beta, 0.0)

        bottom_nodes = [i for i, (x, y) in enumerate(nodes) if abs(y) < 1e-9]
        if not bottom_nodes:
            bottom_nodes = list(range(n_nodes))

        span_xs = [nodes[i][0] for i in bottom_nodes]
        x_min, x_max = min(span_xs), max(span_xs)
        span = x_max - x_min

        sorted_bottom = sorted(bottom_nodes, key=lambda i: nodes[i][0])
        sorted_xs     = [nodes[i][0] for i in sorted_bottom]
        load_matrix   = np.zeros((n_steps + 1, len(bottom_nodes)))
        time_vector   = np.linspace(0.0, cross_duration, n_steps + 1)
        b_idx_map     = {node_idx: b for b, node_idx in enumerate(bottom_nodes)}

        for t_idx, t in enumerate(time_vector):
            pos_x = x_min + (t * speed_ms / bridge_length_m) * span
            pos_x = max(x_min, min(x_max, pos_x))
            right = next((k for k, x in enumerate(sorted_xs) if x >= pos_x),
                         len(sorted_xs) - 1)
            left  = max(0, right - 1)
            n0    = sorted_bottom[left]
            x0    = sorted_xs[left]
            if left == right or abs(sorted_xs[right] - x0) < 1e-12:
                load_matrix[t_idx, b_idx_map[n0]] = -load_n
            else:
                n1   = sorted_bottom[right]
                x1   = sorted_xs[right]
                frac = (pos_x - x0) / (x1 - x0)
                load_matrix[t_idx, b_idx_map[n0]] = -load_n * (1.0 - frac)
                load_matrix[t_idx, b_idx_map[n1]] = -load_n * frac

        for b_idx, node_idx in enumerate(bottom_nodes):
            ts_tag  = 100 + b_idx
            pat_tag = 200 + b_idx
            ops.timeSeries('Path', ts_tag, '-dt', float(dt),
                           '-values', *list(load_matrix[:, b_idx]))
            ops.pattern('Plain', pat_tag, ts_tag)
            ops.load(node_idx + 1, 0.0, 1.0)

        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Plain')
        ops.integrator('Newmark', _NEWMARK_GAMMA, _NEWMARK_BETA)
        ops.algorithm('Linear')
        ops.analysis('Transient')

        n_members  = len(members)
        force_hist = np.zeros((n_steps + 1, n_members))

        for step in range(1, n_steps + 1):
            ok = ops.analyze(1, float(dt))
            if ok != 0:
                converged = False
                print(f"[opensees_analyser] WARNING: analysis failed at "
                      f"step {step}/{n_steps} -- result is partial.")
                break
            steps_completed = step
            for m_idx in range(n_members):
                try:
                    ni_idx, nj_idx, area, E_orig = members[m_idx]
                    E_use = E_override if E_override is not None else E_orig
                    xi, yi = nodes[ni_idx]
                    xj, yj = nodes[nj_idx]
                    dx, dy = xj - xi, yj - yi
                    L      = math.hypot(dx, dy)
                    cx, cy = dx / L, dy / L
                    ui = ops.nodeDisp(ni_idx + 1, 1)
                    vi = ops.nodeDisp(ni_idx + 1, 2)
                    uj = ops.nodeDisp(nj_idx + 1, 1)
                    vj = ops.nodeDisp(nj_idx + 1, 2)
                    elong = cx * (uj - ui) + cy * (vj - vi)
                    force_hist[step, m_idx] = (area * E_use / L) * elong
                except Exception:
                    pass

        ops.wipe()

    except Exception as exc:
        try:
            ops.wipe()
        except Exception:
            pass
        raise RuntimeError(f"Dynamic solve failed: {exc!r}") from exc

    valid_steps = steps_completed + 1   # indices 0..steps_completed
    n_members   = len(members)

    stress_histories: Dict[int, np.ndarray] = {}
    peak_stresses:    Dict[int, float]       = {}
    for m_idx in range(n_members):
        area   = members[m_idx][2]
        s_hist = force_hist[:valid_steps, m_idx] / area   # Pa signed
        stress_histories[m_idx] = s_hist
        peak_stresses[m_idx]    = float(np.max(np.abs(s_hist)))

    # Build static stress envelope (max stress per member at any load
    # position along the span).  This is the correct baseline: the dynamic
    # peak for member m should never be compared against a single arbitrary
    # static position.
    static_env = static_envelope(nodes, members, load_n, bridge_length_m, fem_cache)
    if not static_env:
        static_env = static_inline(nodes, members, fixed_dofs, load_n)

    Y = yield_override if yield_override is not None else yield_strength
    per_member_daf: Dict[int, float] = {}
    for m in peak_stresses:
        s_ref = max(static_env.get(m, 0.0), 1e-6)
        per_member_daf[m] = peak_stresses[m] / s_ref

    # Warn on suspiciously low DAF -- indicates a numerical problem
    for m, d in per_member_daf.items():
        if d < 0.95 and peak_stresses[m] > max(Y, 1.0) * 0.01:
            print(f"[opensees_analyser] WARNING: member {m} "
                  f"DAF={d:.3f} < 0.95 "
                  f"(peak={peak_stresses[m]/1e6:.3f} MPa) -- "
                  "possible numerical issue.")

    daf = max(per_member_daf.values()) if per_member_daf else 1.0
    daf = max(daf, 1.0)   # physical floor

    result = CrossingResult(
        peak_stresses=peak_stresses,
        stress_histories=stress_histories,
        natural_frequencies=list(nat_freqs),
        dynamic_amplification_factor=float(daf),
        time_vector=time_vector[:valid_steps],
        is_dynamic=True,
        steps_completed=steps_completed,
        converged=converged,
    )
    return result, nat_freqs


def _member_axis_3d(
    nodes: List[Tuple[float, float, float]], ni: int, nj: int
) -> Tuple[float, float, float, float]:
    xi, yi, zi = nodes[ni]
    xj, yj, zj = nodes[nj]
    dx, dy, dz = xj - xi, yj - yi, zj - zi
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-12:
        raise ValueError(f"Zero-length 3D member between nodes {ni} and {nj}")
    return dx / length, dy / length, dz / length, length


def _geom_transf_vector(
    nodes: List[Tuple[float, float, float]], ni: int, nj: int
) -> Tuple[float, float, float]:
    cx, cy, cz, _ = _member_axis_3d(nodes, ni, nj)
    dot_z = abs(cz)
    if dot_z < 0.90:
        return 0.0, 0.0, 1.0
    return 0.0, 1.0, 0.0


def _create_3d_geom_transf(
    tag: int,
    vx: float,
    vy: float,
    vz: float,
    geometric_nonlinearity: bool,
) -> str:
    if not geometric_nonlinearity:
        ops.geomTransf("Linear", tag, vx, vy, vz)
        return "Linear"

    for transf in ("Corotational", "Corotational02"):
        try:
            ops.geomTransf(transf, tag, vx, vy, vz)
            return transf
        except Exception:
            continue

    print("[opensees_analyser] WARNING: Corotational transform unavailable "
          "for 3D frame member; using Linear.")
    ops.geomTransf("Linear", tag, vx, vy, vz)
    return "Linear"


def _combined_frame_stress_from_response(
    response: list,
    area: float,
    ixx: float,
    iyy: float,
) -> float:
    if not response or len(response) < 12:
        return 0.0
    n_ax = max(abs(float(response[0])), abs(float(response[6])))
    my = max(abs(float(response[4])), abs(float(response[10])))
    mz = max(abs(float(response[5])), abs(float(response[11])))
    axial = n_ax / max(area, 1e-12)
    bend_y = my * (MEMBER_H / 2.0) / max(iyy, 1e-18)
    bend_z = mz * (MEMBER_W / 2.0) / max(ixx, 1e-18)
    return axial + bend_y + bend_z


def _bottom_groups_3d(
    nodes: List[Tuple[float, float, float]]
) -> Tuple[List[int], List[int]]:
    bottom = [i for i, (_x, _y, z) in enumerate(nodes) if abs(z) < 1e-9]
    if not bottom:
        return [], []
    ys = sorted({round(nodes[i][1], 12) for i in bottom})
    if len(ys) < 2:
        return sorted(bottom, key=lambda i: nodes[i][0]), []
    left_y, right_y = ys[0], ys[-1]
    left = [i for i in bottom if abs(round(nodes[i][1], 12) - left_y) < 1e-12]
    right = [i for i in bottom if abs(round(nodes[i][1], 12) - right_y) < 1e-12]
    return sorted(left, key=lambda i: nodes[i][0]), sorted(right, key=lambda i: nodes[i][0])


def _distribute_line_load_3d(
    nodes: List[Tuple[float, float, float]],
    sorted_nodes: List[int],
    pos_x: float,
    load_n: float,
) -> Dict[int, float]:
    if not sorted_nodes or abs(load_n) < 1e-12:
        return {}
    xs = [nodes[i][0] for i in sorted_nodes]
    x_clamped = max(xs[0], min(xs[-1], pos_x))
    right = next((k for k, x in enumerate(xs) if x >= x_clamped), len(xs) - 1)
    left = max(0, right - 1)
    n0 = sorted_nodes[left]
    x0 = xs[left]
    if left == right or abs(xs[right] - x0) < 1e-12:
        return {n0: -load_n}
    n1 = sorted_nodes[right]
    x1 = xs[right]
    frac = (x_clamped - x0) / (x1 - x0)
    return {n0: -load_n * (1.0 - frac), n1: -load_n * frac}


def _build_3d_frame_domain(
    nodes: List[Tuple[float, float, float]],
    members: List[FrameMember3D],
    fixed_dofs: Set[int],
    density: float,
    e_override: Optional[float],
    geometric_nonlinearity: bool = False,
) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    for i, (x, y, z) in enumerate(nodes, start=1):
        ops.node(i, float(x), float(y), float(z))

    fix_map: Dict[int, List[int]] = {}
    for d in fixed_dofs:
        n_id = d // 6 + 1
        dof_id = d % 6
        if n_id not in fix_map:
            fix_map[n_id] = [0, 0, 0, 0, 0, 0]
        fix_map[n_id][dof_id] = 1
    for n_id, flags in fix_map.items():
        ops.fix(n_id, *flags)

    nodal_mass: Dict[int, float] = {}
    for ni, nj, area, _E, _ixx, _iyy, _j, _G, _gidx in members:
        half = _member_mass_kg(nodes[ni], nodes[nj], area, density) / 2.0
        nodal_mass[ni] = nodal_mass.get(ni, 0.0) + half
        nodal_mass[nj] = nodal_mass.get(nj, 0.0) + half

    for node_idx, mass_val in nodal_mass.items():
        m = max(mass_val, _MIN_NODE_MASS_KG)
        # Tiny rotational mass avoids singular mass matrices during eigen solves
        # while keeping translational dynamics dominant.
        ops.mass(node_idx + 1, m, m, m, m * 1e-6, m * 1e-6, m * 1e-6)

    for m_idx, (ni, nj, area, E_orig, ixx, iyy, j, G, _gidx) in enumerate(members, start=1):
        vx, vy, vz = _geom_transf_vector(nodes, ni, nj)
        _create_3d_geom_transf(m_idx, vx, vy, vz, geometric_nonlinearity)
        E_use = e_override if e_override is not None else E_orig
        try:
            ops.element(
                "elasticBeamColumn", m_idx, ni + 1, nj + 1,
                float(area), float(E_use), float(G), float(j),
                float(iyy), float(ixx), m_idx,
            )
        except Exception:
            ops.element(
                "ElasticBeamColumn", m_idx, ni + 1, nj + 1,
                float(area), float(E_use), float(G), float(j),
                float(iyy), float(ixx), m_idx,
            )


def _static_3d_frame_envelope(
    nodes: List[Tuple[float, float, float]],
    members: List[FrameMember3D],
    fixed_dofs: Set[int],
    density: float,
    load_n: float,
    lateral_frac: float,
    e_override: Optional[float],
    n_positions: int = 21,
) -> Dict[int, float]:
    left_nodes, right_nodes = _bottom_groups_3d(nodes)
    if not left_nodes and not right_nodes:
        return {}
    all_bottom = left_nodes + right_nodes
    x_min = min(nodes[i][0] for i in all_bottom)
    x_max = max(nodes[i][0] for i in all_bottom)
    envelope = {m[-1]: 0.0 for m in members}

    for x_frac in np.linspace(0.0, 1.0, n_positions):
        pos_x = x_min + x_frac * (x_max - x_min)
        try:
            _build_3d_frame_domain(nodes, members, fixed_dofs, density, e_override)
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
                continue
            for local_idx, member in enumerate(members, start=1):
                area, ixx, iyy = member[2], member[4], member[5]
                resp = ops.eleResponse(local_idx, "localForce")
                s = _combined_frame_stress_from_response(resp, area, ixx, iyy)
                envelope[member[-1]] = max(envelope.get(member[-1], 0.0), s)
        finally:
            try:
                ops.wipe()
            except Exception:
                pass
    return envelope


def run_dynamic_3d_frame(
    nodes:           List[Tuple[float, float, float]],
    members:         List[FrameMember3D],
    fixed_dofs:      Set[int],
    density:         float,
    cfg:             AnalyserConfig,
    yield_strength:  float,
    weight_kg:       float,
    speed_ms:        float,
    lateral_frac:    float,
    bridge_length_m: float,
    E_override:      Optional[float],
    yield_override:  Optional[float],
) -> Tuple[CrossingResult, List[float]]:
    load_n = weight_kg * _GRAVITY
    if load_n < 1e-9:
        g_indices = [m[-1] for m in members]
        result = CrossingResult(
            peak_stresses={g: 0.0 for g in g_indices},
            stress_histories={g: np.zeros(2) for g in g_indices},
            natural_frequencies=[],
            dynamic_amplification_factor=1.0,
            time_vector=np.array([0.0, bridge_length_m / max(speed_ms, 1e-3)]),
            is_dynamic=True,
            steps_completed=1,
            converged=True,
        )
        return result, []

    cross_duration = bridge_length_m / max(speed_ms, 1e-3)
    dt = min(cfg.dt_target, cross_duration / 20.0)
    n_steps = max(10, int(math.ceil(cross_duration / dt)))
    n_steps = min(n_steps, _MAX_STEPS)
    dt = cross_duration / n_steps

    n_dof = 6 * len(nodes)
    free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]
    nat_freqs: List[float] = []
    steps_completed = 0
    converged = True
    g_indices = [m[-1] for m in members]
    stress_hist = np.zeros((n_steps + 1, len(members)))

    try:
        _build_3d_frame_domain(
            nodes, members, fixed_dofs, density, E_override,
            geometric_nonlinearity=cfg.geometric_nonlinearity,
        )

        n_modes_safe = min(cfg.n_modes, len(free_dofs))
        if n_modes_safe > 0:
            try:
                lambdas = ops.eigen(n_modes_safe)
                nat_freqs = [
                    math.sqrt(max(lam, 0.0)) / (2.0 * math.pi)
                    for lam in lambdas
                ]
            except Exception:
                nat_freqs = []

        if nat_freqs:
            relevant = [f for f in nat_freqs if f <= _MAX_FREQ_FOR_DT_HZ]
            if relevant:
                f_for_dt = relevant[-1]
                dt_accuracy = 1.0 / (10.0 * f_for_dt)
                if dt > dt_accuracy:
                    n_steps = min(
                        max(10, int(math.ceil(cross_duration / dt_accuracy))),
                        _MAX_STEPS,
                    )
                    dt = cross_duration / n_steps
                    stress_hist = np.zeros((n_steps + 1, len(members)))

        if len(nat_freqs) >= 2:
            w1, w2 = (2 * math.pi * f for f in nat_freqs[:2])
            alpha = 2 * cfg.damping_ratio * w1 * w2 / (w1 + w2)
            beta = 2 * cfg.damping_ratio / (w1 + w2)
        elif len(nat_freqs) == 1:
            alpha, beta = cfg.damping_ratio * 2 * math.pi * nat_freqs[0], 0.0
        else:
            alpha, beta = 0.0, 0.0
        ops.rayleigh(alpha, 0.0, beta, 0.0)

        left_nodes, right_nodes = _bottom_groups_3d(nodes)
        all_bottom = left_nodes + right_nodes
        if not all_bottom:
            raise RuntimeError("3D frame has no bottom/deck nodes for moving load")
        x_min = min(nodes[i][0] for i in all_bottom)
        x_max = max(nodes[i][0] for i in all_bottom)
        span = x_max - x_min
        time_vector = np.linspace(0.0, cross_duration, n_steps + 1)
        load_matrix = np.zeros((n_steps + 1, len(nodes)))

        for t_idx, t in enumerate(time_vector):
            pos_x = x_min + (t * speed_ms / bridge_length_m) * span
            pos_x = max(x_min, min(x_max, pos_x))
            for node, fz in _distribute_line_load_3d(
                nodes, left_nodes, pos_x, load_n * (1.0 - lateral_frac)
            ).items():
                load_matrix[t_idx, node] += fz
            for node, fz in _distribute_line_load_3d(
                nodes, right_nodes, pos_x, load_n * lateral_frac
            ).items():
                load_matrix[t_idx, node] += fz

        loaded_nodes = [i for i in range(len(nodes)) if np.any(load_matrix[:, i])]
        for k, node_idx in enumerate(loaded_nodes):
            ts_tag = 100 + k
            pat_tag = 200 + k
            ops.timeSeries(
                "Path", ts_tag, "-dt", float(dt),
                "-values", *list(load_matrix[:, node_idx])
            )
            ops.pattern("Plain", pat_tag, ts_tag)
            ops.load(node_idx + 1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Transformation")
        ops.integrator("Newmark", _NEWMARK_GAMMA, _NEWMARK_BETA)
        if cfg.geometric_nonlinearity:
            ops.test("NormUnbalance", cfg.newton_tolerance, cfg.newton_max_iter)
            ops.algorithm("Newton")
        else:
            ops.algorithm("Linear")
        ops.analysis("Transient")

        for step in range(1, n_steps + 1):
            ok = ops.analyze(1, float(dt))
            if ok != 0:
                converged = False
                solver_name = "nonlinear Newton" if cfg.geometric_nonlinearity else "linear"
                print(f"[opensees_analyser] WARNING: 3D {solver_name} analysis "
                      f"failed at step {step}/{n_steps}; result is partial.")
                break
            steps_completed = step
            for local_idx, member in enumerate(members, start=1):
                area, ixx, iyy = member[2], member[4], member[5]
                resp = ops.eleResponse(local_idx, "localForce")
                stress_hist[step, local_idx - 1] = (
                    _combined_frame_stress_from_response(resp, area, ixx, iyy)
                )

        ops.wipe()

    except Exception as exc:
        try:
            ops.wipe()
        except Exception:
            pass
        raise RuntimeError(f"3D dynamic solve failed: {exc!r}") from exc

    valid_steps = steps_completed + 1
    stress_histories: Dict[int, np.ndarray] = {}
    peak_stresses: Dict[int, float] = {}
    for local_idx, g_idx in enumerate(g_indices):
        h = stress_hist[:valid_steps, local_idx]
        stress_histories[g_idx] = h
        peak_stresses[g_idx] = float(np.max(np.abs(h)))

    static_env = _static_3d_frame_envelope(
        nodes, members, fixed_dofs, density, load_n, lateral_frac, E_override
    )
    Y = yield_override if yield_override is not None else yield_strength
    per_member_daf: Dict[int, float] = {}
    for g_idx, peak in peak_stresses.items():
        s_ref = max(static_env.get(g_idx, 0.0), 1e-6)
        per_member_daf[g_idx] = peak / s_ref
        if per_member_daf[g_idx] < 0.95 and peak > max(Y, 1.0) * 0.01:
            print(f"[opensees_analyser] WARNING: 3D member {g_idx} "
                  f"DAF={per_member_daf[g_idx]:.3f} < 0.95 "
                  f"(peak={peak/1e6:.3f} MPa).")

    daf = max(per_member_daf.values()) if per_member_daf else 1.0
    daf = max(daf, 1.0)

    return CrossingResult(
        peak_stresses=peak_stresses,
        stress_histories=stress_histories,
        natural_frequencies=list(nat_freqs),
        dynamic_amplification_factor=float(daf),
        time_vector=time_vector[:valid_steps],
        is_dynamic=True,
        steps_completed=steps_completed,
        converged=converged,
    ), nat_freqs


def run_3d_frame_beam_self_test(verbose: bool = True) -> bool:
    if ops is None:
        if verbose:
            print("SKIP: OpenSeesPy unavailable")
        return False

    L = 1.0
    P = 100.0
    area = MEMBER_W * MEMBER_H
    E = E_MODULUS
    G = G_MODULUS
    ixx = MEMBER_IXX
    iyy = MEMBER_IYY
    j = MEMBER_J
    nodes = [(0.0, 0.0, 0.0), (L / 2.0, 0.0, 0.0), (L, 0.0, 0.0)]
    members: List[FrameMember3D] = [
        (0, 1, area, E, ixx, iyy, j, G, 0),
        (1, 2, area, E, ixx, iyy, j, G, 1),
    ]
    # Pin left translations plus one torsional restraint; roller right in y/z.
    fixed = {0, 1, 2, 3, 6 * 2 + 1, 6 * 2 + 2}

    try:
        _build_3d_frame_domain(nodes, members, fixed, DENSITY, None)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 0.0, 0.0, -P, 0.0, 0.0, 0.0)
        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Transformation")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        ok = ops.analyze(1)
        if ok != 0:
            raise AssertionError(f"static beam solve failed with code {ok}")

        moments = []
        for ele in (1, 2):
            resp = ops.eleResponse(ele, "localForce")
            if resp and len(resp) >= 12:
                moments.extend([
                    abs(float(resp[4])), abs(float(resp[10])),
                    abs(float(resp[5])), abs(float(resp[11])),
                ])
        measured = max(moments) if moments else 0.0
        expected = P * L / 4.0
        err = abs(measured - expected) / expected
        if verbose:
            print("--- 3D frame beam self-test ---")
            print(f"  measured M_max = {measured:.6f} N*m")
            print(f"  expected M_max = {expected:.6f} N*m")
            print(f"  error = {err*100:.2f}%")
        if err > 0.05:
            raise AssertionError(
                f"3D beam moment error {err*100:.2f}% exceeds 5%")
        if verbose:
            print("SELF-TEST PASSED")
        return True
    finally:
        try:
            ops.wipe()
        except Exception:
            pass


def run_geometric_nonlinearity_self_test(verbose: bool = True) -> bool:
    if ops is None:
        if verbose:
            print("SKIP: OpenSeesPy unavailable")
        return False

    L = NONLINEAR_SELF_TEST_SPAN_M
    P = NONLINEAR_SELF_TEST_LOAD_N
    area = MEMBER_W * MEMBER_H
    E = E_MODULUS
    G = G_MODULUS
    ixx = MEMBER_IXX
    iyy = MEMBER_IYY
    j = MEMBER_J
    nodes = [(0.0, 0.0, 0.0), (L / 2.0, 0.0, 0.0), (L, 0.0, 0.0)]
    members: List[FrameMember3D] = [
        (0, 1, area, E, ixx, iyy, j, G, 0),
        (1, 2, area, E, ixx, iyy, j, G, 1),
    ]
    fixed = {0, 1, 2, 3, 6 * 2 + 1, 6 * 2 + 2}
    common = dict(
        nodes=nodes,
        members=members,
        fixed_dofs=fixed,
        density=DENSITY,
        yield_strength=YIELD_STRENGTH,
        weight_kg=P / _GRAVITY,
        speed_ms=NONLINEAR_SELF_TEST_SPEED_MS,
        lateral_frac=0.0,
        bridge_length_m=L,
        E_override=None,
        yield_override=None,
    )
    linear_cfg = AnalyserConfig(
        dt_target=NONLINEAR_SELF_TEST_DT,
        n_modes=0,
        geometric_nonlinearity=False,
    )
    nonlinear_cfg = AnalyserConfig(
        dt_target=NONLINEAR_SELF_TEST_DT,
        n_modes=0,
        geometric_nonlinearity=True,
    )

    linear_result, _ = run_dynamic_3d_frame(cfg=linear_cfg, **common)
    nonlinear_result, _ = run_dynamic_3d_frame(cfg=nonlinear_cfg, **common)

    if not nonlinear_result.converged:
        raise AssertionError("nonlinear low-load comparison did not converge")

    max_diff = 0.0
    for g_idx, linear_peak in linear_result.peak_stresses.items():
        nonlinear_peak = nonlinear_result.peak_stresses.get(g_idx, 0.0)
        denom = max(abs(linear_peak), 1e-6)
        max_diff = max(max_diff, abs(nonlinear_peak - linear_peak) / denom)

    if verbose:
        print("--- 3D geometric nonlinearity comparison ---")
        print(f"  max peak-stress difference = {max_diff*100:.2f}%")
        print(f"  tolerance = {NONLINEAR_SELF_TEST_MAX_DIFF*100:.2f}%")

    if max_diff > NONLINEAR_SELF_TEST_MAX_DIFF:
        raise AssertionError(
            f"linear/nonlinear low-load difference {max_diff*100:.2f}% "
            f"exceeds {NONLINEAR_SELF_TEST_MAX_DIFF*100:.2f}%")
    if verbose:
        print("SELF-TEST PASSED")
    return True



