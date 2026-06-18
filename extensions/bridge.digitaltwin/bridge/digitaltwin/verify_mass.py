# Standalone mass / frequency verification for the Warren truss bridge.
#
# Runs without Omniverse -- requires only openseespy.
# All bridge parameters are read from bridge_config.py.
#
# Physical specification (values from bridge_config)
from __future__ import annotations

import math
import sys
import os

# Allow running directly from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- Bridge parameters from bridge_config ------------------------------------
try:
    from bridge_config import (
        E_MODULUS as E_MOD, DENSITY, YIELD_STRENGTH as YIELD,
        MEMBER_AREA as AREA_SPEC, MEMBER_MASS_PER_UNIT_LENGTH as MULL_SPEC,
        NUM_PANELS as PANELS, REAL_PANEL as PANEL_LEN,
        REAL_TRUSS_HEIGHT as HEIGHT, REAL_BRIDGE_LENGTH as BRIDGE_L,
        MEMBER_W, MEMBER_H,
    )
except ImportError:
    E_MOD, DENSITY, YIELD = 69e9, 2700.0, 270e6
    AREA_SPEC = 0.0015 * 0.0015; MULL_SPEC = AREA_SPEC * DENSITY
    PANELS, PANEL_LEN, HEIGHT, BRIDGE_L = 5, 0.10, 0.15, 0.5
    MEMBER_W = MEMBER_H = 0.0015

# Scenario B "old model" stiffness area -- hardcoded 4mm x 4mm to demonstrate
# member_mass_override: stiffness uses this smaller area while mass stays at
# the physical spec (MULL_SPEC).  Changing bridge_config does NOT affect this.
AREA_OLD_MODEL = 0.004 * 0.004   # 1.6e-5 m^2  (4 mm x 4 mm)

# Nodes: bottom bays plus offset top chord nodes (single 2-D truss plane)
bottom = [(i * PANEL_LEN, 0.0)                for i in range(PANELS + 1)]
top    = [((i + 0.5) * PANEL_LEN, HEIGHT)     for i in range(PANELS)]
NODES  = bottom + top

# Boundary conditions: pin at first bottom node, roller (v only) at last bottom
# node. DOF index = 2*node + component (0=x, 1=y).
FIXED_DOFS = [0, 1, 2 * PANELS + 1]


def _build_members(area: float) -> list:
    m = []
    for i in range(PANELS):              # bottom chord
        m.append((i, i + 1, area, E_MOD))
    for i in range(PANELS - 1):          # top chord
        m.append((PANELS + 1 + i, PANELS + 2 + i, area, E_MOD))
    for i in range(PANELS):              # diagonals
        m.append((i,              PANELS + 1 + i, area, E_MOD))
        m.append((PANELS + 1 + i, i + 1,          area, E_MOD))
    return m


_SEP = "=" * 65


# 1.  Structural mass audit
print(_SEP)
print("STEP 1 -- STRUCTURAL MASS AUDIT")
print(_SEP)

members_spec = _build_members(AREA_SPEC)

total_mass = 0.0
nodal_mass: dict[int, float] = {}
member_detail: list[tuple] = []

for ni, nj, area, _ in members_spec:
    xi, yi = NODES[ni];  xj, yj = NODES[nj]
    length  = math.hypot(xj - xi, yj - yi)
    m_mem   = MULL_SPEC * length
    total_mass += m_mem
    nodal_mass[ni] = nodal_mass.get(ni, 0.0) + m_mem / 2.0
    nodal_mass[nj] = nodal_mass.get(nj, 0.0) + m_mem / 2.0
    member_detail.append((ni, nj, length, m_mem))

total_len = sum(r[2] for r in member_detail)

print("  Side plane: {} bottom bays, {} top chord segments".format(
    PANELS, PANELS - 1))
print("  Section: {:.1f} mm x {:.1f} mm,  A = {:.2e} m^2".format(
    MEMBER_W * 1e3, MEMBER_H * 1e3, AREA_SPEC))
print("  m/L = A x density = {:.4f} kg/m".format(MULL_SPEC))
print("  Members: {}  |  total member length = {:.4f} m".format(
    len(members_spec), total_len))
print()
print("  Computed total mass  = {:.2f} g  ({:.5f} kg)".format(
    total_mass * 1e3, total_mass))
hand = total_len * MULL_SPEC
match_str = "OK" if abs(hand - total_mass) < 1e-9 else "MISMATCH"
print("  Hand-check           = {:.4f} x {:.4f} = {:.2f} g  ({})".format(
    total_len, MULL_SPEC, hand * 1e3, match_str))
print()
print("  Expected range for a physical prototype: 50 - 200 g per 2-D truss plane")
if 0.05 <= total_mass <= 0.20:
    note = "within expected range"
else:
    note = "OUTSIDE expected range -- actual members are heavier than prototype assumption"
print("  Computed mass {:.1f} g --> {}".format(total_mass * 1e3, note))

# ---- Nodal mass breakdown ---------------------------------------------------
print()
print("  Nodal mass distribution (structural contribution, no floor applied):")
print("  {:>4}  {:>7}  {:>6}  {:>8}  {}".format(
    "Node", "x (m)", "y (m)", "mass (g)", "boundary"))
for i in range(len(NODES)):
    x, y = NODES[i]
    m_g  = nodal_mass.get(i, 0.0) * 1e3
    bc   = ""
    if i == 0:
        bc = "[pin]"
    elif i == PANELS:
        bc = "[roller]"
    print("  {:>4}  {:>7.4f}  {:>6.3f}  {:>8.2f}  {}".format(i, x, y, m_g, bc))

print()
total_nodal = sum(nodal_mass.values()) * 1e3
print("  Sum of nodal masses: {:.2f} g  (should equal {:.2f} g)".format(
    total_nodal, total_mass * 1e3))


# 2.  Analytical f1 estimate
print()
print(_SEP)
print("STEP 2 -- ANALYTICAL f1  (Euler-Bernoulli Warren-truss equivalent)")
print(_SEP)
print("  Formula:  f1 = pi / (2*L^2) * sqrt(EI_eff / m_per_unit_length)")
print()

h = HEIGHT
# Effective bending stiffness: two chord members at +/- h/2 from neutral axis
EI_eff  = E_MOD * AREA_SPEC * (h / 2.0) ** 2 * 2   # N*m^2
m_prime = total_mass / BRIDGE_L                      # kg/m
f1_anal = (math.pi / (2.0 * BRIDGE_L ** 2)) * math.sqrt(EI_eff / m_prime)

print("  EI_eff = E x A x (h/2)^2 x 2")
print("         = {:.3e} x {:.2e} x {:.4f} x 2".format(E_MOD, AREA_SPEC, (h/2)**2))
print("         = {:.1f} N*m^2".format(EI_eff))
print("  m'     = {:.5f} kg / {:.3f} m = {:.4f} kg/m".format(
    total_mass, BRIDGE_L, m_prime))
print("  f1     = {:.1f} Hz".format(f1_anal))
print()
print("  Notes:")
print("   * Beam formula ignores shear flexibility of diagonals (overestimates f1).")
print("   * For L/h = {:.1f}, shear deformation is significant.".format(BRIDGE_L / HEIGHT))
print("   * Because EI_eff and m' both scale with A, f1 is independent of".format())
print("     cross-section area when stiffness and mass use the SAME area.")
print("   * The OpenSeesPy FEM result is lower due to shear deformation.")


# 3.  OpenSeesPy direct eigenvalue analysis
try:
    import openseespy.opensees as ops
    _ops_ok = True
except Exception as exc:
    print()
    print(f"SKIP: openseespy not available ({exc}) -- skipping eigenvalue analysis.")
    print("Install with:  pip install openseespy")
    sys.exit(0)

_MIN_MASS = 0.005   # 5 g nodal floor (matches opensees_models._MIN_NODE_MASS_KG)


def _run_eigen(area_stiff: float, mull: float, label: str) -> list:
    members = _build_members(area_stiff)

    # Nodal masses for this mull (mass per unit length)
    nm: dict[int, float] = {}
    for ni, nj, _, _ in members:
        xi, yi = NODES[ni];  xj, yj = NODES[nj]
        length  = math.hypot(xj - xi, yj - yi)
        half    = mull * length / 2.0
        nm[ni] = nm.get(ni, 0.0) + half
        nm[nj] = nm.get(nj, 0.0) + half

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 2)

    for idx, (x, y) in enumerate(NODES, start=1):
        ops.node(idx, float(x), float(y))

    fix_map: dict[int, list] = {}
    for d in FIXED_DOFS:
        nid = d // 2 + 1
        dof = d % 2
        if nid not in fix_map:
            fix_map[nid] = [0, 0]
        fix_map[nid][dof] = 1
    for nid, flags in fix_map.items():
        ops.fix(nid, *flags)

    for m_idx, (ni, nj, area, E) in enumerate(members, start=1):
        ops.uniaxialMaterial('Elastic', m_idx, float(E))
        ops.element('Truss', m_idx, ni + 1, nj + 1, float(area), m_idx)

    for nidx, mass_val in nm.items():
        m = max(mass_val, _MIN_MASS)
        ops.mass(nidx + 1, m, m)

    n_free = 2 * len(NODES) - len(FIXED_DOFS)
    n_modes = min(4, n_free)
    freqs = []
    try:
        lambdas = ops.eigen(n_modes)
        freqs = [math.sqrt(max(lam, 0.0)) / (2.0 * math.pi) for lam in lambdas]
    except Exception as exc:
        print("  WARNING: eigenvalue solve failed ({})".format(exc))
    finally:
        ops.wipe()

    total_m_assigned = sum(max(v, _MIN_MASS) for v in nm.values())
    print()
    print("  {}".format(label))
    print("    A_stiffness = {:.2e} m^2   m/L = {:.5f} kg/m".format(area_stiff, mull))
    print("    Total nodal mass assigned (incl. 5g floor): {:.2f} g".format(
        total_m_assigned * 1e3))
    for k, f in enumerate(freqs):
        print("    Mode {:d}: {:>8.2f} Hz".format(k + 1, f))
    return freqs


print()
print(_SEP)
print("STEP 3 -- OPENSEESPY EIGENVALUE ANALYSIS")
print(_SEP)

# Scenario A: bridge_config spec -- same area for stiffness and mass
freqs_A = _run_eigen(
    AREA_SPEC, MULL_SPEC,
    "Scenario A: bridge_config spec  (A={:.2e} stiffness AND mass)".format(AREA_SPEC))

# Scenario B: member_mass_override demo -- old 4mm stiffness, physical 10mm mass.
# Mimics the situation where the FEM stiffness model uses a reduced area but the
# real inertia must match the physical cross-section.
freqs_B = _run_eigen(
    AREA_OLD_MODEL, MULL_SPEC,
    "Scenario B: member_mass_override demo  "
    "(A={:.2e} stiffness, m/L={:.5f} kg/m physical mass)".format(
        AREA_OLD_MODEL, MULL_SPEC))


# 4.  Summary
print()
print(_SEP)
print("SUMMARY")
print(_SEP)
print("  Physical bridge total mass (single 2-D plane): {:.1f} g".format(
    total_mass * 1e3))
print("  Mass per unit length (spec):  {:.4f} kg/m".format(MULL_SPEC))
print("  Analytical f1 (beam bound):   {:.1f} Hz  (upper bound; shear ignored)".format(
    f1_anal))
print()

f1_A = freqs_A[0] if freqs_A else float("nan")
f1_B = freqs_B[0] if freqs_B else float("nan")
mass_ratio = AREA_SPEC / AREA_OLD_MODEL   # 6.25 x
freq_ratio = math.sqrt(mass_ratio)        # 2.5 x
print("  OpenSeesPy f1:")
print("    Scenario A  (bridge_config section, consistent):       {:>8.2f} Hz".format(f1_A))
print("    Scenario B  (4mm stiffness + spec mass override):      {:>8.2f} Hz".format(f1_B))
print()
print("  member_mass_override effect:")
print("    Scenario B mass is {:.2f}x Scenario A mass (same m/L, smaller K)".format(mass_ratio))
print("    -> f1 drops by sqrt(K_ratio) = sqrt({:.4f}) = {:.3f}".format(
    AREA_OLD_MODEL / AREA_SPEC, 1.0 / freq_ratio))
print("    Expected B = {:.1f} / {:.3f} = {:.1f} Hz  (actual: {:.1f} Hz)".format(
    f1_A, freq_ratio, f1_A / freq_ratio, f1_B))
print()
print("  To calibrate with additional non-structural mass (joints, fasteners, deck):")
target = 200.0
needed = MULL_SPEC * (f1_A / target) ** 2
print("    member_mass_override to target f1 < {:.0f} Hz:  {:.4f} kg/m".format(
    int(target), needed))
print("    (set this in bridge_config MEMBER_MASS_PER_UNIT_LENGTH or pass directly)")
print()
print("  RESULT:")
print("    bridge_config spec (Scenario A):  f1 = {:.0f} Hz".format(f1_A))
print("    With mass override (Scenario B):  f1 = {:.0f} Hz".format(f1_B))
if not math.isnan(f1_B) and f1_B < 600:
    print("    f1 = {:.0f} Hz is physically reasonable for a 500 mm Al truss.".format(f1_B))
else:
    print("    Both values are above 600 Hz -- typical for a very stiff aluminium")
    print("    prototype.  Additional non-structural mass would further reduce f1.")


