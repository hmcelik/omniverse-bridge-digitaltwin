# Standalone finite element solvers used by the bridge twin.
#
# TrussFEM solves the older 2D axial truss model. FrameFEM3D solves the live
# 3D frame model with axial, bending, and torsional stiffness. Both solvers use
# SI units and do not import Omniverse.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np

try:
    from .bridge_config import (
        E_MODULUS as _E_DEFAULT,
        G_MODULUS as _G_DEFAULT,
        YIELD_STRENGTH as _YIELD_DEFAULT,
        MEMBER_W as _MEMBER_W_DEFAULT,
        MEMBER_H as _MEMBER_H_DEFAULT,
    )
except ImportError:
    from bridge_config import (  # type: ignore[no-redef]
        E_MODULUS as _E_DEFAULT,
        G_MODULUS as _G_DEFAULT,
        YIELD_STRENGTH as _YIELD_DEFAULT,
        MEMBER_W as _MEMBER_W_DEFAULT,
        MEMBER_H as _MEMBER_H_DEFAULT,
    )


@dataclass
class FEMResult:
    displacements: np.ndarray          # shape (2*n_nodes,), metres
    axial_forces: Dict[int, float]     # member_index -> N (+tension, -compression)
    axial_stresses: Dict[int, float]   # member_index -> Pa
    stress_ratios: Dict[int, float]    # member_index -> stress / yield_strength (0..inf)


@dataclass
class FEMResult3D:
    displacements: np.ndarray
    axial_forces: Dict[int, float]
    bending_moments_y: Dict[int, float]
    bending_moments_z: Dict[int, float]
    combined_stresses: Dict[int, float]
    axial_stresses: Dict[int, float]
    stress_ratios: Dict[int, float]


class TrussFEM:

    def __init__(
        self,
        nodes: List[Tuple[float, float]],
        members: List[Tuple[int, int, float, float]],
        fixed_dofs: List[int],
        yield_strength: float = _YIELD_DEFAULT,
    ) -> None:
        self.nodes = nodes
        self.members = members
        self.fixed_dofs = fixed_dofs
        self.yield_strength = yield_strength
        self._n = len(nodes)
        self._K_free: Optional[np.ndarray] = None   # reduced stiffness matrix
        self._free_dofs: Optional[List[int]] = None
        self._K_full: Optional[np.ndarray] = None
        self._assemble()

    # Private helpers
    def _element_stiffness(self, i: int, j: int, A: float, E: float) -> np.ndarray:
        xi, yi = self.nodes[i]
        xj, yj = self.nodes[j]
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        if L < 1e-14:
            raise ValueError(f"Zero-length member between nodes {i} and {j}")
        cx, cy = dx / L, dy / L
        c2, s2, cs = cx * cx, cy * cy, cx * cy
        k = (A * E / L) * np.array([
            [ c2,  cs, -c2, -cs],
            [ cs,  s2, -cs, -s2],
            [-c2, -cs,  c2,  cs],
            [-cs, -s2,  cs,  s2],
        ])
        return k

    def _assemble(self) -> None:
        n_dof = 2 * self._n
        K = np.zeros((n_dof, n_dof))
        for idx, (i, j, A, E) in enumerate(self.members):
            ke = self._element_stiffness(i, j, A, E)
            dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            for a, da in enumerate(dofs):
                for b, db in enumerate(dofs):
                    K[da, db] += ke[a, b]
        self._K_full = K
        all_dofs = list(range(n_dof))
        self._free_dofs = [d for d in all_dofs if d not in self.fixed_dofs]
        idx = np.ix_(self._free_dofs, self._free_dofs)
        self._K_free = K[idx]

    # Public API
    def solve(self, F: np.ndarray) -> FEMResult:
        if F.shape[0] != 2 * self._n:
            raise ValueError(f"F length {F.shape[0]} != 2*{self._n}")

        F_free = F[self._free_dofs]
        u_free = np.linalg.solve(self._K_free, F_free)

        u = np.zeros(2 * self._n)
        for k, d in enumerate(self._free_dofs):
            u[d] = u_free[k]

        axial_forces: Dict[int, float] = {}
        axial_stresses: Dict[int, float] = {}
        stress_ratios: Dict[int, float] = {}

        for m_idx, (i, j, A, E) in enumerate(self.members):
            xi, yi = self.nodes[i]
            xj, yj = self.nodes[j]
            dx, dy = xj - xi, yj - yi
            L = math.hypot(dx, dy)
            cx, cy = dx / L, dy / L
            # elongation = (uj - ui) projected onto member axis
            elongation = cx * (u[2*j] - u[2*i]) + cy * (u[2*j+1] - u[2*i+1])
            force = (A * E / L) * elongation      # +tension, -compression
            stress = force / A
            axial_forces[m_idx] = float(force)
            axial_stresses[m_idx] = float(stress)
            stress_ratios[m_idx] = float(abs(stress) / self.yield_strength)

        return FEMResult(
            displacements=u,
            axial_forces=axial_forces,
            axial_stresses=axial_stresses,
            stress_ratios=stress_ratios,
        )

    def member_length(self, m_idx: int) -> float:
        i, j, _, _ = self.members[m_idx]
        xi, yi = self.nodes[i]
        xj, yj = self.nodes[j]
        return math.hypot(xj - xi, yj - yi)


class FrameFEM3D:

    def __init__(
        self,
        nodes: List[Tuple[float, float, float]],
        members: List[tuple],
        fixed_dofs: List[int],
        yield_strength: float = _YIELD_DEFAULT,
        member_w: float = _MEMBER_W_DEFAULT,
        member_h: float = _MEMBER_H_DEFAULT,
    ) -> None:
        self.nodes = nodes
        self.members = members
        self.fixed_dofs = fixed_dofs
        self.yield_strength = yield_strength
        self.member_w = member_w
        self.member_h = member_h
        self._n = len(nodes)
        self._K_full: Optional[np.ndarray] = None
        self._K_free: Optional[np.ndarray] = None
        self._free_dofs: Optional[List[int]] = None
        self._element_cache: Dict[int, tuple] = {}
        self._assemble()

    def _member_id(self, idx: int, member: tuple) -> int:
        return int(member[8]) if len(member) >= 9 else idx

    def _local_axes(self, i: int, j: int) -> tuple[np.ndarray, float]:
        pi = np.array(self.nodes[i], dtype=float)
        pj = np.array(self.nodes[j], dtype=float)
        x_axis = pj - pi
        L = float(np.linalg.norm(x_axis))
        if L < 1e-14:
            raise ValueError(f"Zero-length 3D frame member between nodes {i} and {j}")
        x_axis = x_axis / L
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(x_axis, ref))) > 0.90:
            ref = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(ref, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        R = np.vstack([x_axis, y_axis, z_axis])
        return R, L

    def _transform_matrix(self, R: np.ndarray) -> np.ndarray:
        T = np.zeros((12, 12))
        for block in range(4):
            T[3 * block:3 * block + 3, 3 * block:3 * block + 3] = R
        return T

    def _local_stiffness(
        self, A: float, E: float, Ixx: float, Iyy: float, J: float, G: float, L: float
    ) -> np.ndarray:
        k = np.zeros((12, 12))

        # Axial u-x
        EA_L = E * A / L
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L

        # Torsion rx
        GJ_L = G * J / L
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L

        def add_bending(v1: int, r1: int, v2: int, r2: int, EI: float) -> None:
            vals = np.array([
                [ 12 * EI / L**3,  6 * EI / L**2, -12 * EI / L**3,  6 * EI / L**2],
                [  6 * EI / L**2,  4 * EI / L,    -6 * EI / L**2,  2 * EI / L],
                [-12 * EI / L**3, -6 * EI / L**2,  12 * EI / L**3, -6 * EI / L**2],
                [  6 * EI / L**2,  2 * EI / L,    -6 * EI / L**2,  4 * EI / L],
            ])
            dofs = [v1, r1, v2, r2]
            for a, da in enumerate(dofs):
                for b, db in enumerate(dofs):
                    k[da, db] += vals[a, b]

        # Local y displacement bends about local z; local z displacement bends about local y.
        add_bending(1, 5, 7, 11, E * Ixx)
        add_bending(2, 4, 8, 10, E * Iyy)
        return k

    def _assemble(self) -> None:
        n_dof = 6 * self._n
        K = np.zeros((n_dof, n_dof))
        self._element_cache.clear()
        for idx, member in enumerate(self.members):
            i, j, A, E, Ixx, Iyy, J, G = member[:8]
            R, L = self._local_axes(i, j)
            T = self._transform_matrix(R)
            k_local = self._local_stiffness(A, E, Ixx, Iyy, J, G, L)
            k_global = T.T @ k_local @ T
            dofs = [
                6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5,
                6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5,
            ]
            for a, da in enumerate(dofs):
                for b, db in enumerate(dofs):
                    K[da, db] += k_global[a, b]
            self._element_cache[idx] = (T, k_local, dofs, L)

        self._K_full = K
        all_dofs = list(range(n_dof))
        fixed = set(self.fixed_dofs)
        self._free_dofs = [d for d in all_dofs if d not in fixed]
        self._K_free = K[np.ix_(self._free_dofs, self._free_dofs)]

    def solve(self, F: np.ndarray) -> FEMResult3D:
        if F.shape[0] != 6 * self._n:
            raise ValueError(f"F length {F.shape[0]} != 6*{self._n}")
        F_free = F[self._free_dofs]
        u_free = np.linalg.solve(self._K_free, F_free)
        u = np.zeros(6 * self._n)
        for k, d in enumerate(self._free_dofs):
            u[d] = u_free[k]

        axial_forces: Dict[int, float] = {}
        axial_stresses: Dict[int, float] = {}
        bending_y: Dict[int, float] = {}
        bending_z: Dict[int, float] = {}
        combined: Dict[int, float] = {}
        ratios: Dict[int, float] = {}

        for idx, member in enumerate(self.members):
            _i, _j, A, _E, Ixx, Iyy, _J, _G = member[:8]
            member_id = self._member_id(idx, member)
            T, k_local, dofs, _L = self._element_cache[idx]
            u_global = u[dofs]
            u_local = T @ u_global
            f_local = k_local @ u_local

            n_ax = 0.5 * (f_local[6] - f_local[0])
            my = max(abs(f_local[4]), abs(f_local[10]))
            mz = max(abs(f_local[5]), abs(f_local[11]))
            axial = n_ax / A
            sigma = abs(axial) + my * (self.member_h / 2.0) / max(Iyy, 1e-18)
            sigma += mz * (self.member_w / 2.0) / max(Ixx, 1e-18)

            axial_forces[member_id] = float(n_ax)
            axial_stresses[member_id] = float(axial)
            bending_y[member_id] = float(my)
            bending_z[member_id] = float(mz)
            combined[member_id] = float(sigma)
            ratios[member_id] = float(abs(sigma) / self.yield_strength)

        return FEMResult3D(
            displacements=u,
            axial_forces=axial_forces,
            bending_moments_y=bending_y,
            bending_moments_z=bending_z,
            combined_stresses=combined,
            axial_stresses=axial_stresses,
            stress_ratios=ratios,
        )


# Self-test: simply-supported symmetric 3-member truss
def run_self_test(verbose: bool = True) -> None:
    A_area = 1e-4   # 10 mm x 10 mm cross-section
    E_mod = _E_DEFAULT

    nodes = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)]
    members = [
        (0, 2, A_area, E_mod),   # A-C
        (1, 2, A_area, E_mod),   # B-C
        (0, 1, A_area, E_mod),   # A-B (tie rod)
    ]
    # pin at A: fix u_A (dof 0) and v_A (dof 1)
    # roller at B: fix v_B (dof 3)
    fixed_dofs = [0, 1, 3]

    F = np.zeros(6)
    F[5] = -1000.0   # 1 kN downward at node C (dof 2*2+1 = 5)

    solver = TrussFEM(nodes, members, fixed_dofs)
    result = solver.solve(F)

    F_diag_anal = -500.0 * math.sqrt(2.0)   # compression in A-C and B-C
    F_ab_anal   =  500.0                      # tension in A-B

    F_AC = result.axial_forces[0]
    F_BC = result.axial_forces[1]
    F_AB = result.axial_forces[2]

    tol = 1e-4
    errors = []
    if abs(F_AC - F_diag_anal) / abs(F_diag_anal) > tol:
        errors.append(f"F_AC: {F_AC:.4f} vs analytical {F_diag_anal:.4f}")
    if abs(F_BC - F_diag_anal) / abs(F_diag_anal) > tol:
        errors.append(f"F_BC: {F_BC:.4f} vs analytical {F_diag_anal:.4f}")
    if abs(F_AB - F_ab_anal) / abs(F_ab_anal) > tol:
        errors.append(f"F_AB: {F_AB:.4f} vs analytical {F_ab_anal:.4f}")

    if errors:
        raise AssertionError("SELF-TEST FAILED: " + "; ".join(errors))

    if verbose:
        print("SELF-TEST PASSED")
        print(f"  F_AC = {F_AC:.3f} N  (analytical {F_diag_anal:.3f} N, compression)")
        print(f"  F_BC = {F_BC:.3f} N  (analytical {F_diag_anal:.3f} N, compression)")
        print(f"  F_AB = {F_AB:.3f} N  (analytical {F_ab_anal:.3f} N, tension)")
        print(f"  Max displacement = {max(abs(result.displacements)) * 1e6:.4f} um")


def run_3d_frame_self_test(verbose: bool = True) -> None:
    L = 1.0
    P = 100.0
    A = 1e-4
    E = _E_DEFAULT
    G = _G_DEFAULT
    I = 8.333333333333333e-10
    J = 1.406e-9
    nodes = [(0.0, 0.0, 0.0), (L / 2.0, 0.0, 0.0), (L, 0.0, 0.0)]
    members = [
        (0, 1, A, E, I, I, J, G, 0),
        (1, 2, A, E, I, I, J, G, 1),
    ]
    # Pin left translations and torsion; roller right in lateral/vertical.
    fixed = [0, 1, 2, 3, 6 * 2 + 1, 6 * 2 + 2]
    solver = FrameFEM3D(nodes, members, fixed)
    F = np.zeros(18)
    F[6 * 1 + 2] = -P
    result = solver.solve(F)
    measured = abs(result.displacements[6 * 1 + 2])
    expected = P * L**3 / (48.0 * E * I)
    err = abs(measured - expected) / expected
    if err > 0.02:
        raise AssertionError(
            f"3D frame deflection {measured:.6e} vs {expected:.6e} "
            f"({err*100:.2f}% error)"
        )
    if verbose:
        print("3D FRAME SELF-TEST PASSED")
        print(f"  midspan deflection = {measured*1e3:.6f} mm")
        print(f"  analytical         = {expected*1e3:.6f} mm")
        print(f"  error              = {err*100:.2f}%")


if __name__ == "__main__":
    run_self_test()
    run_3d_frame_self_test()



