# bridge_config.py -- Single source of truth for all bridge parameters.
#
# Edit this file to change the bridge physical properties, member cross-section,
# dynamic analysis settings, fatigue constants, or environmental model rates.
# Every other module imports its values from here.

# 1.  Material -- Aluminium 6061-T6
E_MODULUS      = 69e9     # Young's modulus (Pa)
G_MODULUS      = 26e9     # shear modulus (Pa), aluminium 6061-T6 typical
YIELD_STRENGTH = 270e6   # 0.2% proof strength (Pa)
DENSITY        = 2700.0  # density (kg/m^3)
GRAVITY        = 9.81    # gravitational acceleration (m/s^2)

# 2.  Bridge geometry
NUM_PANELS         = 5
REAL_BRIDGE_LENGTH = 0.5    # span (m)
REAL_TRUSS_HEIGHT  = 0.15   # truss depth, centre-to-centre (m)
REAL_TRUSS_WIDTH   = 0.10   # width between the two side-planes (m)
REAL_PANEL         = REAL_BRIDGE_LENGTH / NUM_PANELS  # panel length (m) -- derived

# 3.  Member cross-section  (1.5 mm x 1.5 mm solid rectangular bar)
MEMBER_W    = 0.0015                                       # width  (m)
MEMBER_H    = 0.002                                       # height (m)
MEMBER_AREA = MEMBER_W * MEMBER_H                          # cross-section area (m^2)
MEMBER_IXX  = MEMBER_W * MEMBER_H ** 3 / 12.0              # bending inertia about local x-like section axis (m^4)
MEMBER_IYY  = MEMBER_H * MEMBER_W ** 3 / 12.0              # bending inertia about local y-like section axis (m^4)
MEMBER_I    = MEMBER_IXX                                   # backward-compatible square-bar inertia alias
# Saint-Venant torsion constant for a solid rectangular bar.  For a square this
# gives J ~= 0.1406*a^4, close to the tabulated 0.1406 value.
_J_A = max(MEMBER_W, MEMBER_H)
_J_B = min(MEMBER_W, MEMBER_H)
MEMBER_J    = _J_A * _J_B ** 3 * (
    1.0 / 3.0 - 0.21 * (_J_B / _J_A) * (1.0 - (_J_B ** 4) / (12.0 * _J_A ** 4))
)                                                           # torsion constant (m^4)
MEMBER_MASS_PER_UNIT_LENGTH = MEMBER_AREA * DENSITY        # kg/m  (for mass assembly)
BUCKLING_K  = 1.0   # effective-length factor (1.0 = pin-pin Euler)

# 4.  Dynamic analysis solver settings
DAMPING_RATIO      = 0.02     # fraction of critical Rayleigh damping
DT_TARGET          = 0.001    # initial time-step (s)
N_MODES            = 4        # number of eigenvalues to extract per crossing
MIN_NODE_MASS_KG   = 0.005    # 5 g nodal mass floor (bolts / gusset plates)
MAX_FREQ_FOR_DT_HZ = 50.0     # cap: modes above this are not excited by traffic
MAX_STEPS          = 50_000   # hard cap on Newmark steps per crossing
NEWMARK_GAMMA      = 0.5      # Newmark constant-average-acceleration parameter
NEWMARK_BETA       = 0.25     # Newmark constant-average-acceleration parameter
GEOMETRIC_NONLINEARITY_DEFAULT = False  # optional corotational 3D frame solve
NEWTON_TOLERANCE   = 1e-8     # OpenSees nonlinear residual tolerance
NEWTON_MAX_ITER    = 20       # maximum Newton iterations per transient step
NONLINEAR_SELF_TEST_LOAD_N = 10.0       # low-load comparison force (N)
NONLINEAR_SELF_TEST_SPAN_M = 1.0        # diagnostic simply supported span (m)
NONLINEAR_SELF_TEST_SPEED_MS = 0.5      # moving-load comparison speed (m/s)
NONLINEAR_SELF_TEST_DT = 0.01           # diagnostic transient time-step (s)
NONLINEAR_SELF_TEST_MAX_DIFF = 0.05     # max linear/nonlinear stress difference

# 5.  Prototype operating limits
V_MAX_PROTOTYPE     = 1.2    # maximum prototype crossing speed (m/s)
SIM_CYCLES_PER_PASS = 1000   # fatigue cycles credited per simulated crossing
SIM_WEIGHT_MIN_KG   = 0.20   # minimum synthetic model-car mass (kg)
SIM_WEIGHT_MAX_KG   = 5.00   # maximum synthetic model-car mass (kg)

# 6.  Fatigue -- Eurocode 9 detail category 71 (welded aluminium joints)
FATIGUE_DETAIL_CATEGORY_PA = 71e6   # reference stress range at 2e6 cycles (Pa)
FATIGUE_EXPONENT           = 3      # S-N slope exponent m
FATIGUE_LIMIT_PA           = 35e6   # endurance limit -- no damage below this (Pa)

# Paris-law crack growth.  Units are SI:
#   da/dN = PARIS_C * (DeltaK)^PARIS_M
#   DeltaK = CRACK_F * DeltaSigma * sqrt(pi*a)
# PARIS_C is converted for Pa*sqrt(m); values around 1e-29 to 1e-28 are common
# SI-order magnitudes for aluminium when source data is quoted in MPa*sqrt(m).
PARIS_C = 3.0e-29                 # m/cycle/(Pa*sqrt(m))^m
PARIS_M = 3.0                     # Paris exponent
FRACTURE_TOUGHNESS_KIC = 29e6     # Pa*sqrt(m), 6061-T6 order of magnitude
CRACK_A0 = 0.20e-3                # initial equivalent crack size (m)
CRACK_F = 1.12                    # edge/surface crack geometry factor

# 7.  Environmental degradation model  (empirical knock-down rates)
# Residual-strength floors (design safety limits, not physical minimums)
ENV_MIN_E_FRACTION       = 0.90   # E never drops below 90% of reference
ENV_MIN_YIELD_FRACTION   = 0.75   # yield never drops below 75%
ENV_MIN_FATIGUE_FRACTION = 0.60   # fatigue limit never drops below 60%

# Temperature cycling (calibrated: ~3% yield loss per 10 000 cycles at DeltaT = 20 degC)
ENV_TEMP_CYCLE_YIELD_RATE   = 3e-7   # fractional yield loss per (cycle * DeltaT^2)
ENV_TEMP_CYCLE_FATIGUE_RATE = 5e-7   # fractional fatigue-limit loss per (cycle * DeltaT^2)
ENV_TEMP_CYCLE_E_RATE       = 5e-8   # fractional E loss per (cycle * DeltaT^2)

# Humidity / SCC pitting (calibrated: ~8% yield loss after 5 000 h at 80% RH outdoors)
ENV_HUMIDITY_YIELD_RATE   = 1.6e-5   # fractional yield loss per (hour * RH_fraction)
ENV_HUMIDITY_FATIGUE_RATE = 3.0e-5   # fatigue limit dominated by pitting
ENV_HUMIDITY_E_RATE       = 3.0e-7   # negligible E reduction from humidity

# Exposure-type multipliers
ENV_OUTDOOR_MULTIPLIER = 2.5   # UV + rain + pollutants
ENV_INDOOR_MULTIPLIER  = 1.0

# 8.  Safety alert thresholds
# Max acceptable divergence between fast (static) and accurate (Newmark) stress
# before a sensor-anomaly alert fires.  35% accounts for moving-load effects,
# rainflow distribution, and DAF-formula approximation.
FAST_VS_ACCURATE_THRESHOLD = 0.35

# 9.  Physical strain-gauge setup
STRAIN_GAUGE_COUNT = 4
STRAIN_FEEDBACK_GAIN = 70.0

# 10.  Scene / viewport display constants  (Omniverse only)
SCENE_SCALE  = 20.0                              # real metres -> scene units
TRUSS_LENGTH = REAL_BRIDGE_LENGTH * SCENE_SCALE  # scene-unit span
TRUSS_HEIGHT = REAL_TRUSS_HEIGHT  * SCENE_SCALE  # scene-unit height
TRUSS_WIDTH  = REAL_TRUSS_WIDTH   * SCENE_SCALE  # scene-unit width
MEMBER_THICK = 0.0015 * SCENE_SCALE              # visual tube thickness
VEHICLE_LENGTH = 0.15 * SCENE_SCALE              # visual model-car length
VEHICLE_WIDTH  = 0.05 * SCENE_SCALE              # visual model-car width
VEHICLE_HEIGHT = 0.05 * SCENE_SCALE              # visual model-car height
UPRIGHT_DEG  = -90.0                             # X rotation to stand bridge upright
DEFORM_SCALE = 50.0                              # displacement magnification


