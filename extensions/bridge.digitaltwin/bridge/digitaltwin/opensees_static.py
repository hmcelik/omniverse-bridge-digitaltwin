# Static analysis helpers for the bridge truss.
#
# All public functions are pure -- they accept explicit node/member/dof data
# rather than instance state, making them testable in isolation.
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .opensees_models import CrossingResult, AnalyserConfig, _GRAVITY
except ImportError:
    from opensees_models import CrossingResult, AnalyserConfig, _GRAVITY  # type: ignore[no-redef]


def _member_mass_kg(
    node_a: tuple,
    node_b: tuple,
    area_m2: float,
    density_kg_m3: float,
) -> float:
    if len(node_a) >= 3 and len(node_b) >= 3:
        length = math.dist(node_a[:3], node_b[:3])
    else:
        length = math.hypot(node_b[0] - node_a[0], node_b[1] - node_a[1])
    return area_m2 * length * density_kg_m3


def static_at_position(
    nodes:   List[Tuple[float, float]],
    members: List[Tuple[int, int, float, float]],
    load_n:  float,
    pos_x:   float,
    fem:     Optional[object],
) -> Dict[int, float]:
    if fem is None:
        return {}

    n_nodes = len(nodes)
    F = np.zeros(2 * n_nodes)

    bottom_nodes = [i for i, (x, y) in enumerate(nodes) if abs(y) < 1e-9]
    if not bottom_nodes:
        bottom_nodes = list(range(n_nodes))

    sorted_bn = sorted(bottom_nodes, key=lambda i: nodes[i][0])
    sorted_xs = [nodes[i][0] for i in sorted_bn]
    x_clamped = max(sorted_xs[0], min(sorted_xs[-1], pos_x))

    right = next((k for k, x in enumerate(sorted_xs) if x >= x_clamped),
                 len(sorted_xs) - 1)
    left  = max(0, right - 1)
    n0    = sorted_bn[left]
    x0    = sorted_xs[left]

    if left == right or abs(sorted_xs[right] - x0) < 1e-12:
        F[2 * n0 + 1] -= load_n
    else:
        n1   = sorted_bn[right]
        x1   = sorted_xs[right]
        frac = (x_clamped - x0) / (x1 - x0)
        F[2 * n0 + 1] -= load_n * (1.0 - frac)
        F[2 * n1 + 1] -= load_n * frac

    try:
        result = fem.solve(F)
        return {m: abs(s) for m, s in result.axial_stresses.items()}
    except Exception:
        return {}


def static_envelope(
    nodes:           List[Tuple[float, float]],
    members:         List[Tuple[int, int, float, float]],
    load_n:          float,
    bridge_length_m: float,
    fem:             Optional[object],
    n_positions:     int = 21,
) -> Dict[int, float]:
    if fem is None:
        return {}

    bottom_nodes = [i for i, (x, y) in enumerate(nodes) if abs(y) < 1e-9]
    if not bottom_nodes:
        bottom_nodes = list(range(len(nodes)))
    span_xs = [nodes[i][0] for i in bottom_nodes]
    x_min, x_max = min(span_xs), max(span_xs)

    envelope: Dict[int, float] = {}
    for x_frac in np.linspace(0.0, 1.0, n_positions):
        pos_x = x_min + x_frac * (x_max - x_min)
        for m_idx, s in static_at_position(nodes, members, load_n, pos_x, fem).items():
            envelope[m_idx] = max(envelope.get(m_idx, 0.0), s)
    return envelope


def static_inline(
    nodes:      List[Tuple[float, float]],
    members:    List[Tuple[int, int, float, float]],
    fixed_dofs: Set[int],
    load_n:     float,
) -> Dict[int, float]:
    import numpy.linalg as la

    n_nodes = len(nodes)
    n_dof   = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))

    for ni, nj, area, E in members:
        xi, yi = nodes[ni]; xj, yj = nodes[nj]
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        if L < 1e-12:
            continue
        cx, cy = dx / L, dy / L
        c2, s2, cs = cx*cx, cy*cy, cx*cy
        k = (area * E / L) * np.array([
            [ c2,  cs, -c2, -cs],
            [ cs,  s2, -cs, -s2],
            [-c2, -cs,  c2,  cs],
            [-cs, -s2,  cs,  s2],
        ])
        dofs = [2*ni, 2*ni+1, 2*nj, 2*nj+1]
        for a, da in enumerate(dofs):
            for b, db in enumerate(dofs):
                K[da, db] += k[a, b]

    free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]
    K_f = K[np.ix_(free_dofs, free_dofs)]

    bottom_nodes = [i for i, (x, y) in enumerate(nodes) if abs(y) < 1e-9]
    if not bottom_nodes:
        return {}
    span_xs = [nodes[i][0] for i in bottom_nodes]
    x_mid   = (max(span_xs) + min(span_xs)) / 2.0
    adj = sorted(bottom_nodes, key=lambda i: abs(nodes[i][0] - x_mid))
    F = np.zeros(n_dof)
    for bn in adj[:2]:
        F[2*bn + 1] -= load_n / 2.0

    try:
        u_f = la.solve(K_f, F[free_dofs])
    except Exception:
        return {}

    u = np.zeros(n_dof)
    for k_idx, d in enumerate(free_dofs):
        u[d] = u_f[k_idx]

    out: Dict[int, float] = {}
    for m_idx, (ni, nj, area, E) in enumerate(members):
        xi, yi = nodes[ni]; xj, yj = nodes[nj]
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        if L < 1e-12:
            out[m_idx] = 0.0
            continue
        cx, cy = dx / L, dy / L
        elong = cx*(u[2*nj]-u[2*ni]) + cy*(u[2*nj+1]-u[2*ni+1])
        out[m_idx] = abs((area * E / L) * elong / area)
    return out


def static_fallback(
    nodes:           List[Tuple[float, float]],
    members:         List[Tuple[int, int, float, float]],
    fixed_dofs:      Set[int],
    yield_strength:  float,
    cfg:             AnalyserConfig,
    fem_cache:       Optional[object],
    weight_kg:       float,
    speed_ms:        float,
    lateral_frac:    float,
    bridge_length_m: float,
    yield_override:  Optional[float],
) -> CrossingResult:
    try:
        from .sensor_reader import _daf as _sensor_daf
    except ImportError:
        def _sensor_daf(v): return 1.0 + 0.5 * min(v / 1.2, 1.0) ** 2

    daf    = max(_sensor_daf(speed_ms), 1.0)
    load_n = weight_kg * _GRAVITY * (1.0 - lateral_frac) * daf

    n_steps   = 10
    n_members = len(members)

    static_results = static_envelope(nodes, members, load_n, bridge_length_m, fem_cache)
    if not static_results:
        static_results = static_inline(nodes, members, fixed_dofs, load_n)

    stress_histories: Dict[int, np.ndarray] = {}
    for m_idx in range(n_members):
        s = static_results.get(m_idx, 0.0)
        stress_histories[m_idx] = np.full(n_steps + 1, s)

    peak_stresses = {m: float(np.max(np.abs(h)))
                     for m, h in stress_histories.items()}
    time_vector   = np.linspace(
        0.0, bridge_length_m / max(speed_ms, 1e-3), n_steps + 1)

    return CrossingResult(
        peak_stresses=peak_stresses,
        stress_histories=stress_histories,
        natural_frequencies=[],
        dynamic_amplification_factor=float(daf),
        time_vector=time_vector,
        is_dynamic=False,
        steps_completed=n_steps,
        converged=True,
    )


