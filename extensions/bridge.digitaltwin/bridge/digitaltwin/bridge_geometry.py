# Bridge topology and USD geometry helpers.
#
# _TrussTopology stores the Warren truss nodes and member connectivity. The helper
# functions create the visible USD members and build the topology data needed by
# FEM solvers.
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

from .fem_solver import TrussFEM
from .bridge_config import (
    E_MODULUS, G_MODULUS, YIELD_STRENGTH, DENSITY,
    NUM_PANELS, REAL_PANEL, REAL_TRUSS_HEIGHT, REAL_TRUSS_WIDTH,
    MEMBER_AREA, MEMBER_IXX, MEMBER_IYY, MEMBER_J, SCENE_SCALE, MEMBER_THICK,
)


# Truss topology -- geometry builder + FEM model factory
class _TrussTopology:

    def __init__(self) -> None:
        hw = REAL_TRUSS_WIDTH / 2.0
        # Nodes keyed by label, value = (x, y, z) in real metres
        self.nodes: Dict[str, tuple] = {}
        for side, y in (("L", -hw), ("R", hw)):
            for i in range(NUM_PANELS + 1):
                self.nodes[f"B{side}{i}"] = (i * REAL_PANEL, y, 0.0)
            for i in range(NUM_PANELS):
                self.nodes[f"T{side}{i}"] = (
                    (i + 0.5) * REAL_PANEL, y, REAL_TRUSS_HEIGHT
                )

        # Members in index order (must match USD prim creation order)
        self.members: List[tuple] = []   # (labelA, labelB, type_str)
        for side in ("L", "R"):
            for i in range(NUM_PANELS):
                self.members.append((f"B{side}{i}", f"B{side}{i+1}", "chord_bottom"))
            for i in range(NUM_PANELS - 1):
                self.members.append((f"T{side}{i}", f"T{side}{i+1}", "chord_top"))
            for i in range(NUM_PANELS):
                self.members.append((f"B{side}{i}", f"T{side}{i}", "diagonal"))
                self.members.append((f"T{side}{i}", f"B{side}{i+1}", "diagonal"))
        for i in range(NUM_PANELS):
            self.members.append((f"TL{i}", f"TR{i}", "cross"))
        for i in range(NUM_PANELS + 1):
            self.members.append((f"BL{i}", f"BR{i}", "deck"))
        for i in range(NUM_PANELS):
            side_a, side_b = ("L", "R") if i % 2 == 0 else ("R", "L")
            self.members.append((f"B{side_a}{i}", f"B{side_b}{i+1}", "deck_diagonal"))
        for i in range(NUM_PANELS - 1):
            side_a, side_b = ("L", "R") if i % 2 == 0 else ("R", "L")
            self.members.append((f"T{side_a}{i}", f"T{side_b}{i+1}", "deck_diagonal"))

        self.member_lengths = [
            math.dist(self.nodes[a], self.nodes[b])
            for a, b, _ in self.members
        ]

    def build_fem_for_side(self, side: str) -> tuple[TrussFEM, List[int]]:
        # Collect nodes in this plane using (x, z) as 2D (x, y)
        plane_labels = (
            [f"B{side}{i}" for i in range(NUM_PANELS + 1)] +
            [f"T{side}{i}" for i in range(NUM_PANELS)]
        )
        node_2d: Dict[str, int] = {lbl: k for k, lbl in enumerate(plane_labels)}
        nodes_xy = [
            (self.nodes[lbl][0], self.nodes[lbl][2])   # (x, z) -> 2D (x, y)
            for lbl in plane_labels
        ]

        global_indices: List[int] = []
        members_2d = []
        for g_idx, (la, lb, _) in enumerate(self.members):
            if la in node_2d and lb in node_2d:
                members_2d.append((node_2d[la], node_2d[lb], MEMBER_AREA, E_MODULUS))
                global_indices.append(g_idx)

        # BLs/BRs nodes are 0..NUM_PANELS; TLs/TRs are NUM_PANELS+1..
        # Supports: pin at B{side}0, roller (y only) at B{side}{NUM_PANELS}
        bl0 = node_2d[f"B{side}0"]
        blN = node_2d[f"B{side}{NUM_PANELS}"]
        fixed_dofs = [2 * bl0, 2 * bl0 + 1, 2 * blN + 1]   # u0,v0  +  vN

        fem = TrussFEM(
            nodes=nodes_xy,
            members=members_2d,
            fixed_dofs=fixed_dofs,
            yield_strength=YIELD_STRENGTH,
        )
        return fem, global_indices

    def get_3d_frame_topology(self) -> tuple[
        List[Tuple[float, float, float]],
        List[Tuple[int, int, float, float, float, float, float, float, int]],
        List[int],
    ]:
        labels = list(self.nodes.keys())
        node_index = {label: idx for idx, label in enumerate(labels)}
        nodes_3d = [tuple(self.nodes[label]) for label in labels]

        members_3d: List[
            Tuple[int, int, float, float, float, float, float, float, int]
        ] = []
        for g_idx, (la, lb, _) in enumerate(self.members):
            members_3d.append((
                node_index[la], node_index[lb],
                MEMBER_AREA, E_MODULUS, MEMBER_IXX, MEMBER_IYY,
                MEMBER_J, G_MODULUS, g_idx,
            ))

        fixed_dofs: List[int] = []
        for label in ("BL0", "BR0"):
            n = node_index[label]
            fixed_dofs.extend([6 * n + 0, 6 * n + 1, 6 * n + 2])
        for label in (f"BL{NUM_PANELS}", f"BR{NUM_PANELS}"):
            n = node_index[label]
            fixed_dofs.extend([6 * n + 1, 6 * n + 2])

        return nodes_3d, members_3d, fixed_dofs

    def build_3d_load_vector(
        self, load_n: float, load_x_frac: float, load_y_frac: float
    ) -> np.ndarray:
        labels = list(self.nodes.keys())
        node_index = {label: idx for idx, label in enumerate(labels)}
        F = np.zeros(6 * len(labels))
        x = max(0.0, min(1.0, load_x_frac)) * (NUM_PANELS * REAL_PANEL)

        for side, share in (("L", 1.0 - load_y_frac), ("R", load_y_frac)):
            side_load = load_n * share
            panel_frac = x / REAL_PANEL
            i0 = max(0, min(NUM_PANELS - 1, int(math.floor(panel_frac))))
            frac = panel_frac - i0
            n0 = node_index[f"B{side}{i0}"]
            n1 = node_index[f"B{side}{i0 + 1}"]
            F[6 * n0 + 2] -= side_load * (1.0 - frac)
            F[6 * n1 + 2] -= side_load * frac
        return F

    def solve_full(
        self, load_n: float, load_x_frac: float, load_y_frac: float
    ) -> Dict[int, float]:
        results: Dict[int, float] = {}

        for side, load_share in (("L", 1.0 - load_y_frac), ("R", load_y_frac)):
            fem, g_indices = self.build_fem_for_side(side)
            n_nodes_2d = len(fem.nodes)

            # Distribute the side's share to two nearest bottom chord nodes
            load_x = load_x_frac * (NUM_PANELS * REAL_PANEL)
            panel_frac = load_x / REAL_PANEL
            i0 = max(0, min(NUM_PANELS - 1, int(math.floor(panel_frac))))
            fx = panel_frac - i0
            side_load = load_n * load_share

            F = np.zeros(2 * n_nodes_2d)
            # Bottom chord nodes start at index 0 in the 2D numbering
            n_bl0 = i0        # B{side}{i0} is at 2D index i0
            n_bl1 = i0 + 1
            # Apply downward (-y in 2D = -z in 3D) loads
            F[2 * n_bl0 + 1] -= side_load * (1.0 - fx)
            F[2 * n_bl1 + 1] -= side_load * fx

            r = fem.solve(F)
            for k, g_idx in enumerate(g_indices):
                results[g_idx] = r.axial_forces[k]

        # Members outside the side truss planes are zero-force in this 2D model.
        for g_idx, _member in enumerate(self.members):
            if g_idx not in results:
                results[g_idx] = 0.0

        return results

    def member_real_length(self, g_idx: int) -> float:
        la, lb, _ = self.members[g_idx]
        xa, ya, za = self.nodes[la]
        xb, yb, zb = self.nodes[lb]
        return math.dist((xa, ya, za), (xb, yb, zb))

    def member_span_fraction(self, g_idx: int) -> float:
        la, lb, _ = self.members[g_idx]
        xa = self.nodes[la][0]
        xb = self.nodes[lb][0]
        span = max(NUM_PANELS * REAL_PANEL, 1e-9)
        return max(0.0, min(1.0, ((xa + xb) * 0.5) / span))

    def diagonal_member_index(self, side: str, panel: int, upward: bool) -> int:
        side = side.upper()
        if side not in ("L", "R"):
            raise ValueError(f"Unknown truss side: {side!r}")
        if panel < 0 or panel >= NUM_PANELS:
            raise ValueError(f"Panel index out of range: {panel}")

        if upward:
            target = (f"B{side}{panel}", f"T{side}{panel}", "diagonal")
        else:
            target = (f"T{side}{panel}", f"B{side}{panel + 1}", "diagonal")

        try:
            return self.members.index(target)
        except ValueError as exc:
            name = "DiagUp" if upward else "DiagDn"
            raise ValueError(f"{name}_{side}_{panel} not found") from exc


# USD geometry helpers (same mesh-building logic as original extension)
def _box_mesh_pts_faces(sx, sy, sz):
    pts = [
        (-sx, -sy, -sz), ( sx, -sy, -sz), ( sx, sy, -sz), (-sx, sy, -sz),
        (-sx, -sy,  sz), ( sx, -sy,  sz), ( sx, sy,  sz), (-sx, sy,  sz),
    ]
    faces = [0,1,2,3, 4,7,6,5, 0,4,5,1, 1,5,6,2, 2,6,7,3, 3,7,4,0]
    return pts, faces


def _define_box_mesh(stage, path, size):
    sx, sy, sz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    pts, faces = _box_mesh_pts_faces(sx, sy, sz)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateExtentAttr([(-sx, -sy, -sz), (sx, sy, sz)])
    return mesh


def _make_member(stage, path, p1, p2, ctype, member_index=None):
    p1 = Gf.Vec3d(*p1)
    p2 = Gf.Vec3d(*p2)
    center = (p1 + p2) * 0.5
    delta = p2 - p1
    length = delta.GetLength()
    if length < 1e-6:
        return None
    mesh = UsdGeom.Mesh.Define(stage, path)
    half = length / 2.0
    t = MEMBER_THICK / 2.0
    pts = [
        (-half, -t, -t), ( half, -t, -t), ( half, t, -t), (-half, t, -t),
        (-half, -t,  t), ( half, -t,  t), ( half, t,  t), (-half, t,  t),
    ]
    faces = [0,1,2,3, 4,7,6,5, 0,4,5,1, 1,5,6,2, 2,6,7,3, 3,7,4,0]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateExtentAttr([Gf.Vec3f(-half, -t, -t), Gf.Vec3f(half, t, t)])

    d = delta / length
    x_axis = Gf.Vec3d(1, 0, 0)
    axis = Gf.Cross(x_axis, d)
    dot = max(-1.0, min(1.0, Gf.Dot(x_axis, d)))
    angle = math.degrees(math.acos(dot))
    xform = UsdGeom.Xformable(mesh)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(center)
    if axis.GetLength() > 1e-6:
        qd = Gf.Rotation(axis.GetNormalized(), angle).GetQuat()
        qf = Gf.Quatf(float(qd.GetReal()),
                      Gf.Vec3f(*[float(c) for c in qd.GetImaginary()]))
        xform.AddOrientOp().Set(qf)

    prim = mesh.GetPrim()
    prim.CreateAttribute("component:type",       Sdf.ValueTypeNames.String).Set(ctype)
    prim.CreateAttribute("material:youngsModulus",Sdf.ValueTypeNames.Float).Set(E_MODULUS)
    prim.CreateAttribute("material:yieldStrength",Sdf.ValueTypeNames.Float).Set(YIELD_STRENGTH)
    prim.CreateAttribute("material:density",      Sdf.ValueTypeNames.Float).Set(DENSITY)
    prim.CreateAttribute("analysis:stress",       Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("analysis:stressRatio",  Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("analysis:axialForce",   Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("analysis:utilisation",  Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("analysis:failureMode",  Sdf.ValueTypeNames.String).Set("none")
    prim.CreateAttribute("analysis:memberLength", Sdf.ValueTypeNames.Float).Set(float(length))
    prim.CreateAttribute("scene:scaleFactor",     Sdf.ValueTypeNames.Float).Set(SCENE_SCALE)
    prim.CreateAttribute("analysis:damage",       Sdf.ValueTypeNames.Float).Set(0.0)
    if member_index is not None:
        prim.CreateAttribute("analysis:memberIndex", Sdf.ValueTypeNames.Int).Set(member_index)
    return prim


# Small USD attribute setter utility
def _set_attr(prim, name: str, value):
    a = prim.GetAttribute(name)
    if a:
        a.Set(value)



