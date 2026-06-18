# Omniverse extension entry point for the bridge digital twin.
#
# This class builds the USD bridge, starts the sensor reader, runs the live FEM
# solver, records damage after each crossing, and shows operator status in the UI.
# Omniverse scene edits stay on the main thread. Expensive dynamic analysis runs
# in DynamicAnalysisManager on a background thread.

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import omni.ext
import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom

from .bridge_config import (
    E_MODULUS, YIELD_STRENGTH, DENSITY, GRAVITY,
    NUM_PANELS, REAL_BRIDGE_LENGTH,
    MEMBER_AREA, MEMBER_I, BUCKLING_K, MEMBER_MASS_PER_UNIT_LENGTH,
    V_MAX_PROTOTYPE, SIM_CYCLES_PER_PASS,
    FATIGUE_LIMIT_PA,
    SIM_WEIGHT_MIN_KG, SIM_WEIGHT_MAX_KG, STRAIN_FEEDBACK_GAIN,
    STRAIN_GAUGE_COUNT,
    TRUSS_LENGTH, TRUSS_HEIGHT, TRUSS_WIDTH, MEMBER_THICK,
    VEHICLE_LENGTH, VEHICLE_WIDTH, VEHICLE_HEIGHT,
    UPRIGHT_DEG, SCENE_SCALE,
)
from .bridge_geometry import _TrussTopology, _make_member, _define_box_mesh, _set_attr
from .bridge_manager import DynamicAnalysisManager
from .bridge_ui import BridgeUIMixin
from .damage_model import DamageModel
from .environmental_model import EnvironmentalModel
from .fem_solver import FEMResult, FEMResult3D, FrameFEM3D
from .opensees_analyser import OpenSeesAnalyser
from .safety_checker import Alert, AlertLevel, SafetyChecker, VehicleParams
from .sensor_reader import (
    ConnectionConfig, FeedbackCommand, SensorReader, SensorTick, VehiclePass,
)
from .sensor_validation import SensorResidual, compute_sensor_residuals

# Colour helpers
_DAMAGE_THRESHOLDS = [
    (0.3, Gf.Vec3f(0.0, 0.8, 0.0)),   # HEALTHY  -> green
    (0.7, Gf.Vec3f(1.0, 0.8, 0.0)),   # WORN     -> yellow
    (1.0, Gf.Vec3f(1.0, 0.4, 0.0)),   # WARNING  -> orange
]
_DAMAGE_CRITICAL = Gf.Vec3f(1.0, 0.0, 0.0)  # CRITICAL -> red


def _stress_color(ratio: float) -> Gf.Vec3f:
    r = min(1.0, ratio * 2.0)
    g = min(1.0, max(0.0, 2.0 - ratio * 2.0))
    return Gf.Vec3f(r, g, 0.0)


def _damage_color(d: float) -> Gf.Vec3f:
    for threshold, color in _DAMAGE_THRESHOLDS:
        if d < threshold:
            return color
    return _DAMAGE_CRITICAL


def _alert_style(level: AlertLevel) -> dict:
    colors = {
        AlertLevel.INFO:     0xFF88CCFF,
        AlertLevel.WARNING:  0xFF00BBFF,
        AlertLevel.CRITICAL: 0xFF0000FF,
    }
    return {"color": colors.get(level, 0xFFFFFFFF)}


# Main extension class
class MyExtension(BridgeUIMixin, omni.ext.IExt):

    def on_startup(self, _ext_id):
        print("[bridge.digitaltwin] startup")

        ext_dir = Path(__file__).parent
        damage_json = ext_dir / "damage_state.json"
        alert_log   = ext_dir / "alerts.csv"

        self._topo    = _TrussTopology()
        self._damage  = DamageModel(n_members=len(self._topo.members),
                                    json_path=damage_json)
        self._safety  = SafetyChecker(yield_strength_pa=YIELD_STRENGTH,
                                      safe_load_kg=2.0,
                                      log_path=alert_log)
        self._gauged_members: List[int] = []   # most stressed; set by capacity solve
        self._conn_config = ConnectionConfig(mode="websocket")
        self._sensor = SensorReader(config=self._conn_config)
        self._sensor.start()

        # Environmental model -- persisted via damage_state.json env_state key
        self._env = EnvironmentalModel()
        self._load_env_state(damage_json)

        # Background dynamic analysis manager
        self._dyn_mgr = DynamicAnalysisManager(
            damage_model=self._damage,
            env_model=self._env,
            sensor_reader=self._sensor,
            safety_checker=self._safety,
            passes_per_trigger=3,   # fire every 3 passes; first-session pre-arm fires on pass 1
        )
        self._dyn_mgr.start()

        self._member_prims: Dict[int, list] = {}
        self._box_translate = None
        self._box_size = 1.0
        self._coloring_mode = "stress"   # "stress" | "damage"
        self._last_alerts: List[Alert] = []
        self._last_fem: Optional[Dict[int, float]] = None
        self._frame_fem3d: Optional[FrameFEM3D] = None
        self._use_3d_fast_solver = True
        self._updating = False           # guard for manual slider path
        self._safe_load_kg = 500.0       # overwritten by _compute_structural_capacity
        self._gauge_member_names: Dict[int, str] = {}
        self._last_sensor_tick_key = None
        self._last_sensor_tick_time: Optional[float] = None
        self._last_live_stoplight_state: Optional[bool] = None
        self._last_live_stoplight_sent_at = 0.0
        self._visual_load_x_frac: Optional[float] = None
        self._visual_load_active = False
        self._visual_load_tick_key = None
        self._last_completed_visual_tick_key = None
        self._visual_load_remove_at: Optional[float] = None
        self._load_anim_from = 0.0
        self._load_anim_to = 1.0
        self._load_anim_start = 0.0
        self._load_anim_duration = 1.0
        # Dynamic analysis state (updated from result queue)
        self._last_dyn_result: Optional[dict] = None
        self._env_yield_knockdown: float = 0.0
        self._fast_vs_accurate_error: Optional[float] = None
        self._last_sensor_residuals: List[SensorResidual] = []
        self._opensees_available = False

        self._build_ui()

        # Defer bridge build by two frames -- the USD stage is not guaranteed
        # to exist yet when on_startup runs synchronously.
        async def _deferred_build():
            app = omni.kit.app.get_app()
            await app.next_update_async()
            await app.next_update_async()
            self._build_bridge()

        self._task       = asyncio.ensure_future(self._sensor_loop())
        self._build_task = asyncio.ensure_future(_deferred_build())

    def on_shutdown(self):
        for attr in ("_task", "_build_task"):
            t = getattr(self, attr, None)
            if t:
                t.cancel()
            setattr(self, attr, None)   # break reference so coroutine can be GC'd
        if hasattr(self, "_dyn_mgr"):
            self._dyn_mgr.stop()
        if hasattr(self, "_sensor"):
            self._sensor.stop()
        if hasattr(self, "_damage"):
            self._damage.save()
        if hasattr(self, "_env"):
            ext_dir = Path(__file__).parent
            self._save_env_state(ext_dir / "damage_state.json")
        # Destroy the window so stale labels don't accumulate across hot-reloads.
        if hasattr(self, "_window") and self._window:
            self._window.destroy()
            self._window = None
        print("[bridge.digitaltwin] shutdown")

    # Environmental state persistence (piggybacks on damage_state.json)
    def _load_env_state(self, json_path: Path) -> None:
        if not json_path.exists():
            return
        try:
            import json as _json
            data = _json.loads(json_path.read_text())
            env_data = data.get("env_state")
            if env_data:
                self._env = EnvironmentalModel.from_dict(env_data)
        except Exception as exc:
            print(f"[bridge] Could not load env state: {exc}")

    def _save_env_state(self, json_path: Path) -> None:
        if not json_path.exists():
            return
        try:
            import json as _json
            data = _json.loads(json_path.read_text())
            data["env_state"] = self._env.to_dict()
            json_path.write_text(_json.dumps(data, indent=2))
        except Exception as exc:
            print(f"[bridge] Could not save env state: {exc}")

    # Bridge geometry build (preserves original USD structure exactly)
    def _build_bridge(self):
        import traceback
        try:
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            for p in ("/World/Bridge", "/World/Sensors", "/World/LoadBox", "/World/Floor"):
                if stage.GetPrimAtPath(p).IsValid():
                    stage.RemovePrim(p)

            UsdGeom.Xform.Define(stage, "/World")
            UsdGeom.Xform.Define(stage, "/World/Bridge")
            self._member_prims = {}

            panel = TRUSS_LENGTH / NUM_PANELS
            x0 = -TRUSS_LENGTH / 2.0
            hw = TRUSS_WIDTH / 2.0

            def bottom(side_y, i):
                return (x0 + i * panel, side_y, 0.0)
            def top(side_y, i):
                return (x0 + (i + 0.5) * panel, side_y, TRUSS_HEIGHT)

            midx = 0
            for side, y in (("L", -hw), ("R", hw)):
                for i in range(NUM_PANELS):
                    self._reg(_make_member(stage, f"/World/Bridge/BotChord_{side}_{i}",
                                           bottom(y, i), bottom(y, i+1),
                                           "chord_bottom", midx), midx)
                    midx += 1
                for i in range(NUM_PANELS - 1):
                    self._reg(_make_member(stage, f"/World/Bridge/TopChord_{side}_{i}",
                                           top(y, i), top(y, i+1),
                                           "chord_top", midx), midx)
                    midx += 1
                for i in range(NUM_PANELS):
                    self._reg(_make_member(stage, f"/World/Bridge/DiagUp_{side}_{i}",
                                           bottom(y, i), top(y, i),
                                           "diagonal", midx), midx)
                    midx += 1
                    self._reg(_make_member(stage, f"/World/Bridge/DiagDn_{side}_{i}",
                                           top(y, i), bottom(y, i+1),
                                           "diagonal", midx), midx)
                    midx += 1
            for i in range(NUM_PANELS):
                self._reg(_make_member(stage, f"/World/Bridge/Cross_{i}",
                                        top(-hw, i), top(hw, i),
                                        "cross", midx), midx)
                midx += 1
            for i in range(NUM_PANELS + 1):
                self._reg(_make_member(stage, f"/World/Bridge/Deck_{i}",
                                        bottom(-hw, i), bottom(hw, i),
                                        "deck", midx), midx)
                midx += 1
            for i in range(NUM_PANELS):
                y_a, y_b = (-hw, hw) if i % 2 == 0 else (hw, -hw)
                self._reg(_make_member(stage, f"/World/Bridge/DeckDiag_Bottom_{i}",
                                        bottom(y_a, i), bottom(y_b, i+1),
                                        "deck_diagonal", midx), midx)
                midx += 1
            for i in range(NUM_PANELS - 1):
                y_a, y_b = (-hw, hw) if i % 2 == 0 else (hw, -hw)
                self._reg(_make_member(stage, f"/World/Bridge/DeckDiag_Top_{i}",
                                        top(y_a, i), top(y_b, i+1),
                                        "deck_diagonal", midx), midx)
                midx += 1

            # Sensor nodes
            UsdGeom.Xform.Define(stage, "/World/Sensors")
            for i in range(NUM_PANELS + 1):
                bx = x0 + i * panel
                sx = UsdGeom.Xform.Define(stage, f"/World/Sensors/Node_{i}")
                sx.AddTranslateOp().Set(Gf.Vec3d(bx, 0.0, 0.0))
                sp = sx.GetPrim()
                sp.CreateAttribute("sensor:type",     Sdf.ValueTypeNames.String).Set("strain")
                sp.CreateAttribute("sensor:hardwareId",Sdf.ValueTypeNames.String).Set(f"Node_{i}")
                sp.CreateAttribute("sensor:value",    Sdf.ValueTypeNames.Float).Set(0.0)

            # Floor
            floor = _define_box_mesh(stage, "/World/Floor",
                                      (TRUSS_LENGTH, TRUSS_WIDTH, MEMBER_THICK * 0.4))
            UsdGeom.Xformable(floor).AddTranslateOp().Set(
                Gf.Vec3d(0, 0, MEMBER_THICK * 0.2))
            fc = UsdGeom.Gprim(floor.GetPrim()).CreateDisplayColorAttr()
            fc.Set([Gf.Vec3f(0.55, 0.55, 0.6)])
            UsdGeom.Primvar(fc).SetInterpolation(UsdGeom.Tokens.constant)

            # Load box is created only while a vehicle is in transit.
            self._box_size = VEHICLE_HEIGHT
            self._box_translate = None
            self._last_sensor_tick_key = None
            self._last_sensor_tick_time = None
            self._last_live_stoplight_state = None
            self._last_live_stoplight_sent_at = 0.0
            self._visual_load_x_frac = None
            self._visual_load_active = False
            self._visual_load_tick_key = None
            self._last_completed_visual_tick_key = None
            self._visual_load_remove_at = None
            self._load_anim_from = 0.0
            self._load_anim_to = 1.0
            self._load_anim_start = time.monotonic()
            self._load_anim_duration = 1.0

            for grp in ("/World/Bridge", "/World/Sensors", "/World/Floor"):
                gx = UsdGeom.Xformable(stage.GetPrimAtPath(grp))
                gx.ClearXformOpOrder()
                gx.AddRotateXYZOp().Set(Gf.Vec3f(UPRIGHT_DEG, 0.0, 0.0))

            n = len(self._topo.members)
            print(f"[bridge] Warren truss built: {n} members.")
            self._status.text = f"3D truss built ({n} members)."
            self._compute_structural_capacity()
            self._setup_fast_frame_solver()
            self._setup_dynamic_analyser()

        except Exception as exc:
            import traceback as tb
            print("[bridge] BUILD FAILED:", exc)
            tb.print_exc()

    def _reg(self, prim, m_idx):
        if prim is not None:
            self._member_prims.setdefault(m_idx, []).append(prim.GetPath())

    def _compute_structural_capacity(self):
        forces_unit = self._topo.solve_full(1.0, 0.5, 0.5)   # 1 N, midspan
        max_util_per_n = 0.0
        util_scores: List[tuple] = []   # (util, m_idx) for structural members only
        for m_idx, force in forces_unit.items():
            _, _, mtype = self._topo.members[m_idx]
            stress = abs(force) / MEMBER_AREA
            yield_util = stress / YIELD_STRENGTH
            if force < 0:
                length = self._topo.member_real_length(m_idx)
                eff = BUCKLING_K * length
                p_cr = (math.pi ** 2 * E_MODULUS * MEMBER_I / eff ** 2
                        if eff > 1e-9 else float("inf"))
                buckle_util = abs(force) / p_cr if p_cr > 0 else 0.0
            else:
                buckle_util = 0.0
            util = max(yield_util, buckle_util)
            max_util_per_n = max(max_util_per_n, util)
            if mtype not in ("cross", "deck", "deck_diagonal"):
                util_scores.append((util, m_idx))

        if max_util_per_n > 0:
            safe_n = 0.70 / max_util_per_n
            self._safe_load_kg = safe_n / GRAVITY
        else:
            self._safe_load_kg = 1000.0
        print(f"[bridge] Static capacity: {self._safe_load_kg:.1f} kg "
              f"(70% utilisation limit)")
        self._publish_max_load(self._safe_load_kg)

        self._gauged_members = self._preferred_gauge_members(util_scores)
        gauge_map = {
            ch: m for ch, m in enumerate(self._gauged_members[:STRAIN_GAUGE_COUNT])
        }
        gauge_span_map = {
            ch: self._topo.member_span_fraction(m_idx)
            for ch, m_idx in gauge_map.items()
        }
        self._conn_config.gauge_channel_map = gauge_map
        self._conn_config.gauge_span_map = gauge_span_map
        self._sensor.config.gauge_channel_map = dict(gauge_map)
        self._sensor.set_gauge_span_map(gauge_span_map)
        self._mark_gauge_prims()

        self._update_gauge_labels()

    def _preferred_gauge_members(self, util_scores: List[tuple]) -> List[int]:
        panel = min(2, NUM_PANELS - 1)
        placements = [
            ("DiagUp_L_2", "L", True),
            ("DiagDn_L_2", "L", False),
            ("DiagUp_R_2", "R", True),
            ("DiagDn_R_2", "R", False),
        ]
        members: List[int] = []
        names: Dict[int, str] = {}
        try:
            for label, side, upward in placements[:STRAIN_GAUGE_COUNT]:
                m_idx = self._topo.diagonal_member_index(side, panel, upward)
                members.append(m_idx)
                names[m_idx] = label
        except Exception as exc:
            print(f"[bridge] Preferred gauge placement unavailable ({exc}); "
                  "using most-stressed members.")
            util_scores.sort(reverse=True)
            members = [m for _, m in util_scores[:STRAIN_GAUGE_COUNT]]
            names = {}
        self._gauge_member_names = names
        return members

    def _active_gauged_members(self) -> List[int]:
        return self._gauged_members[:STRAIN_GAUGE_COUNT]

    def _update_gauge_labels(self) -> None:
        labels = getattr(self, "_conn_gauge_labels", [])
        for ch, lbl in enumerate(labels):
            if ch < len(self._gauged_members):
                m_idx = self._gauged_members[ch]
                _, _, mt = self._topo.members[m_idx]
                location = self._gauge_member_names.get(m_idx, mt)
                lbl.text = f"CH {ch} -> {location} / M{m_idx}"
            else:
                lbl.text = f"CH {ch} -> member: build bridge first"

    def _setup_fast_frame_solver(self) -> None:
        try:
            nodes_3d, members_3d, fixed_dofs_3d = self._topo.get_3d_frame_topology()
            self._frame_fem3d = FrameFEM3D(
                nodes=nodes_3d,
                members=members_3d,
                fixed_dofs=fixed_dofs_3d,
                yield_strength=YIELD_STRENGTH,
            )
            self._use_3d_fast_solver = True
            print("[bridge] Fast FEM: 3D frame solver ready.")
        except Exception as exc:
            self._frame_fem3d = None
            self._use_3d_fast_solver = False
            print(f"[bridge] Fast FEM: 3D frame unavailable; using 2D truss ({exc})")
        if hasattr(self, "_bending_cb"):
            self._bending_cb.model.set_value(bool(self._use_3d_fast_solver))
        if hasattr(self, "_refresh_model_labels"):
            self._refresh_model_labels()

    def _publish_max_load(self, max_load_kg: float) -> None:
        try:
            command = FeedbackCommand(max_load_kg=float(max_load_kg))
            self._sensor.set_control_feedback(command)
            if hasattr(self, "_update_feedback_readout"):
                self._update_feedback_readout(command.payload())
        except Exception as exc:
            print(f"[bridge] Could not queue maxLoad update: {exc}")

    # Gauge marker prims
    def _mark_gauge_prims(self):
        stage = omni.usd.get_context().get_stage()
        if not stage or not stage.GetPrimAtPath("/World/Bridge").IsValid():
            return

        gauge_root = "/World/Sensors/Gauges"
        if stage.GetPrimAtPath(gauge_root).IsValid():
            stage.RemovePrim(gauge_root)
        UsdGeom.Xform.Define(stage, gauge_root)

        x0_scene = -TRUSS_LENGTH / 2.0
        marker_sz = MEMBER_THICK * 2.0

        for ch, m_idx in enumerate(self._active_gauged_members()):
            la, lb, _ = self._topo.members[m_idx]
            xa, ya, za = self._topo.nodes[la]
            xb, yb, zb = self._topo.nodes[lb]
            # Convert real-metre midpoint to scene-unit local space (same frame as
            # the bridge members, which are children of the -90° X-rotated group).
            mx = (xa + xb) / 2.0 * SCENE_SCALE + x0_scene
            my = (ya + yb) / 2.0 * SCENE_SCALE
            mz = (za + zb) / 2.0 * SCENE_SCALE

            marker = _define_box_mesh(
                stage, f"{gauge_root}/Gauge_{ch}",
                (marker_sz, marker_sz, marker_sz),
            )
            xf = UsdGeom.Xformable(marker)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(mx, my, mz))

            gc = UsdGeom.Gprim(marker.GetPrim()).CreateDisplayColorAttr()
            gc.Set([Gf.Vec3f(0.0, 1.0, 1.0)])   # cyan
            UsdGeom.Primvar(gc).SetInterpolation(UsdGeom.Tokens.constant)
            marker.GetPrim().CreateAttribute(
                "sensor:gaugeChannel", Sdf.ValueTypeNames.Int).Set(ch)
            marker.GetPrim().CreateAttribute(
                "sensor:memberIndex", Sdf.ValueTypeNames.Int).Set(m_idx)

    # Dynamic analyser setup (called after bridge topology is known)
    def _setup_dynamic_analyser(self):
        try:
            fem, g_indices = self._topo.build_fem_for_side("L")
            nodes_3d, members_3d, fixed_dofs_3d = self._topo.get_3d_frame_topology()
            analyser = OpenSeesAnalyser(
                nodes_2d=fem.nodes,
                members=fem.members,
                fixed_dofs=fem.fixed_dofs,
                density=DENSITY,
                yield_strength_pa=YIELD_STRENGTH,
                member_mass_override=MEMBER_MASS_PER_UNIT_LENGTH,
                nodes_3d=nodes_3d,
                members_3d=members_3d,
                fixed_dofs_3d=fixed_dofs_3d,
            )
            self._dyn_mgr.set_analyser(analyser)
            try:
                from .opensees_models import _OPS_AVAILABLE
                self._opensees_available = bool(_OPS_AVAILABLE)
            except Exception:
                self._opensees_available = False
            model = "3D frame" if analyser.is_3d_frame else "2D truss"
            print(f"[bridge] OpenSeesAnalyser attached ({model}).")
            if hasattr(self, "_refresh_model_labels"):
                self._refresh_model_labels()
        except Exception as exc:
            print(f"[bridge] Could not create OpenSeesAnalyser: {exc}")
            if hasattr(self, "_refresh_model_labels"):
                self._refresh_model_labels()

    # 10 Hz loop: tick -> animate box + FEM;  pass -> record damage once
    async def _sensor_loop(self):
        app = omni.kit.app.get_app()
        last_pass_id = None
        while True:
            try:
                await app.next_update_async()
                now = time.monotonic()

                # Poll dynamic analysis results from background thread
                self._poll_dynamic_results()

                # Sensor data arrives around 10 Hz. Use each new tick for FEM/UI,
                # but interpolate the visible car every Kit frame.
                tick = self._sensor.current_tick
                if tick is not None:
                    if tick.in_transit:
                        tick_key = self._load_tick_key(tick)
                        if tick_key != self._last_completed_visual_tick_key and self._track_load_animation_target(tick, now):
                            self._process_tick(tick, update_visual=False)
                    else:
                        if self._visual_load_active and self._visual_load_remove_at is None:
                            self._visual_load_remove_at = now + 0.05

                if self._visual_load_active:
                    self._animate_load_box(now)

                # When a crossing completes, record damage exactly once.
                vp = self._sensor.latest_pass
                vp_id = id(vp) if vp else None
                if vp is not None and vp_id != last_pass_id:
                    last_pass_id = vp_id
                    self._record_damage(vp)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[bridge] sensor loop error: {exc}")

    def _load_tick_key(self, tick: SensorTick) -> tuple:
        return (
            tick.crossing_id,
            tick.timestamp_ms,
            round(tick.position_frac, 4),
            tick.in_transit,
            round(tick.weight_kg, 4),
        )

    def _visual_exit_fraction(self) -> float:
        return 1.0 + 0.5 * VEHICLE_LENGTH / max(TRUSS_LENGTH, 1e-9)

    def _track_load_animation_target(self, tick: SensorTick, now: float) -> bool:
        tick_key = self._load_tick_key(tick)
        if tick_key == self._last_sensor_tick_key:
            return False

        tick_pos = max(0.0, min(1.0, tick.position_frac))
        current = self._current_load_visual_position(now)
        if (
            current is None
            or not self._visual_load_active
            or tick_pos < current - 0.2
        ):
            current = tick_pos
        else:
            current = max(current, tick_pos)

        speed_frac_s = 0.0
        if tick.speed_ms is not None:
            speed_frac_s = max(0.0, tick.speed_ms / max(REAL_BRIDGE_LENGTH, 1e-9))
        if speed_frac_s <= 1e-9 and self._last_sensor_tick_time is not None:
            dt = max(now - self._last_sensor_tick_time, 1e-6)
            speed_frac_s = max(0.0, (tick_pos - self._load_anim_from) / dt)
        if speed_frac_s <= 1e-9:
            speed_frac_s = 1.0 / 0.5

        self._load_anim_from = current
        self._load_anim_to = self._visual_exit_fraction()
        self._load_anim_start = now
        self._load_anim_duration = max(
            1.0 / 120.0,
            (self._load_anim_to - current) / max(speed_frac_s, 1e-9))
        self._last_sensor_tick_time = now
        self._last_sensor_tick_key = tick_key
        self._visual_load_active = True
        self._visual_load_tick_key = tick_key
        self._visual_load_remove_at = None
        return True

    def _current_load_visual_position(self, now: float) -> Optional[float]:
        if not self._visual_load_active:
            return self._visual_load_x_frac
        t = (now - self._load_anim_start) / max(self._load_anim_duration, 1e-6)
        t = max(0.0, min(1.0, t))
        return self._load_anim_from + (self._load_anim_to - self._load_anim_from) * t

    def _animate_load_box(self, now: float) -> None:
        pos = self._current_load_visual_position(now)
        if pos is None:
            return
        self._set_load_box_position(pos, clamp_to_span=False)
        if pos >= self._visual_exit_fraction() - 1e-6 and self._visual_load_remove_at is None:
            self._visual_load_remove_at = now + 0.05
        if self._visual_load_remove_at is not None and now >= self._visual_load_remove_at:
            self._last_completed_visual_tick_key = self._visual_load_tick_key
            self._remove_load_box()
            self._reset_load_animation()

    def _reset_load_animation(self) -> None:
        self._last_sensor_tick_key = None
        self._last_sensor_tick_time = None
        self._visual_load_x_frac = None
        self._visual_load_active = False
        self._visual_load_tick_key = None
        self._visual_load_remove_at = None
        self._load_anim_from = 0.0
        self._load_anim_to = self._visual_exit_fraction()
        self._load_anim_start = time.monotonic()
        self._load_anim_duration = 1.0

    def _create_load_box(self, stage) -> None:
        if self._box_translate is not None:
            return
        if stage.GetPrimAtPath("/World/LoadBox").IsValid():
            stage.RemovePrim("/World/LoadBox")

        box = _define_box_mesh(
            stage, "/World/LoadBox",
            (VEHICLE_LENGTH, VEHICLE_WIDTH, VEHICLE_HEIGHT),
        )
        bx_form = UsdGeom.Xformable(box)
        bx_form.ClearXformOpOrder()
        self._box_translate = bx_form.AddTranslateOp()
        bc = UsdGeom.Gprim(box.GetPrim()).CreateDisplayColorAttr()
        bc.Set([Gf.Vec3f(0.2, 0.4, 0.9)])
        UsdGeom.Primvar(bc).SetInterpolation(UsdGeom.Tokens.constant)

    def _remove_load_box(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage and stage.GetPrimAtPath("/World/LoadBox").IsValid():
            stage.RemovePrim("/World/LoadBox")
        self._box_translate = None
        self._visual_load_active = False
        self._visual_load_tick_key = None
        self._visual_load_remove_at = None

    def _set_load_box_position(
        self,
        load_x_frac: float,
        load_y_frac: float = 0.5,
        clamp_to_span: bool = True,
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage or not stage.GetPrimAtPath("/World/Bridge").IsValid():
            return

        self._create_load_box(stage)
        if clamp_to_span:
            load_x_frac = max(0.0, min(1.0, load_x_frac))
        local_x = (-TRUSS_LENGTH / 2.0) + load_x_frac * TRUSS_LENGTH
        local_y = (-TRUSS_WIDTH / 2.0) + load_y_frac * TRUSS_WIDTH
        local_z = MEMBER_THICK * 0.4 + self._box_size / 2.0
        if self._box_translate:
            self._box_translate.Set(Gf.Vec3d(local_x, local_z, -local_y))
            self._visual_load_x_frac = load_x_frac

    def _poll_dynamic_results(self):
        result = self._dyn_mgr.poll_results()
        if result is None:
            # Update status badge
            status = self._dyn_mgr.status
            n_since = self._dyn_mgr.passes_since_last_run
            next_at = self._dyn_mgr._per_trigger - n_since
            if status == "running":
                self._lbl_dyn_status.text = "Dynamic FEM: running..."
                self._lbl_dyn_status.style = {"color": 0xFF00DDFF}
            else:
                self._lbl_dyn_status.text = (
                    f"Dynamic FEM: idle  (next in {max(0, next_at)} passes)")
                self._lbl_dyn_status.style = {"color": 0xFF666666}
            return

        # Got a result -- update all badges
        self._last_dyn_result = result
        if result.get("apply_accurate_damage"):
            self._damage.record_pass_accurate(
                result.get("member_cycles", {}),
                simple_increments_to_replace=result.get(
                    "simple_increments_to_replace", {}),
                crack_state_to_replace=result.get("crack_state_before_batch"),
            )
            self._damage.save()
            if self._coloring_mode == "damage":
                self._apply_damage_colors()
        calibration = result.get("daf_calibration")
        if calibration:
            speed_ms, measured_daf = calibration
            self._sensor.update_daf_calibration(speed_ms, measured_daf)
        freqs_for_checker = result.get("natural_frequencies", [])
        if freqs_for_checker:
            self._safety.natural_frequency_hz = freqs_for_checker[0]
        self._env.advance_time(hours=1.0, n_temp_cycles=0)

        if hasattr(self, "_refresh_model_labels"):
            self._refresh_model_labels()
        self._env_yield_knockdown = result.get("env_yield_knockdown", 0.0)

        converged  = result.get("converged", True)
        freqs      = result.get("natural_frequencies", [])
        daf        = result.get("daf", 1.0)
        n_analysed = result.get("n_passes_analysed", 0)
        is_dyn     = result.get("is_dynamic", False)
        steps_done = result.get("steps_completed", 0)

        # Only propagate fast_vs_accurate_error when the result is trustworthy.
        # A non-converged result has error=None from the worker, so this guard
        # also stops a spurious sensor-anomaly alert from firing.
        if converged:
            self._fast_vs_accurate_error = result.get("fast_vs_accurate_error")
        else:
            self._fast_vs_accurate_error = None

        f1_str = f"{freqs[0]:.2f} Hz" if freqs else "--"
        self._lbl_nat_freq.text = (
            f"f1: {f1_str}  |  DAF: {daf:.3f}"
            + ("  [dynamic]" if is_dyn else "  (static est.)"))
        self._lbl_nat_freq.style = {
            "color": 0xFF88CCFF if is_dyn else 0xFF888888}

        kd_pct = self._env_yield_knockdown * 100.0
        self._lbl_env_degradation.text = (
            f"Material ageing: yield -{kd_pct:.1f}%"
            if kd_pct > 0.1 else "Material ageing: negligible")

        if converged:
            self._lbl_dyn_status.text = (
                f"Dynamic FEM: updated {n_analysed} passes ago  "
                f"(acc. passes: {self._damage.accurate_pass_count})")
            self._lbl_dyn_status.style = {"color": 0xFF88FF88}
        else:
            # Non-converged -- show a convergence warning in amber so the
            # operator knows the result was not used for damage accumulation.
            self._lbl_dyn_status.text = (
                f"Dynamic FEM: CONVERGENCE WARNING  "
                f"({steps_done} steps, result discarded)")
            self._lbl_dyn_status.style = {"color": 0xFF00AAFF}
        if hasattr(self, "_lbl_analysis_source"):
            mode = result.get("analysis_mode", "fast-only")
            self._lbl_analysis_source.text = f"Source: {mode}"

        # Update Connection tab frequency label
        if hasattr(self, "_conn_freq_lbl"):
            if freqs:
                self._conn_freq_lbl.text = (
                    f"f1: {f1_str}  |  resonance at "
                    f"v = {(freqs[0] * 0.5):.3f} m/s"
                )
            else:
                self._conn_freq_lbl.text = "f1: --"

        # Update Connection tab env label with live knock-down
        if hasattr(self, "_conn_env_lbl"):
            _exp  = self._env.state.exposure
            _rh   = self._env.state.humidity_rh_avg
            _kpct = self._env_yield_knockdown * 100.0
            self._conn_env_lbl.text = (
                f"Env: {_exp} | RH: {_rh*100:.0f}% | yield -{_kpct:.1f}%"
            )

    def _solve_fast_path(
        self, load_n: float, load_x_frac: float, load_y_frac: float
    ) -> tuple[Dict[int, float], Optional[FEMResult3D]]:
        if self._use_3d_fast_solver and self._frame_fem3d is not None:
            try:
                F = self._topo.build_3d_load_vector(load_n, load_x_frac, load_y_frac)
                result3d = self._frame_fem3d.solve(F)
                pseudo_forces = {
                    m: s * MEMBER_AREA
                    for m, s in result3d.combined_stresses.items()
                }
                return pseudo_forces, result3d
            except Exception as exc:
                self._use_3d_fast_solver = False
                if hasattr(self, "_bending_cb"):
                    self._bending_cb.model.set_value(False)
                if hasattr(self, "_refresh_model_labels"):
                    self._refresh_model_labels()
                print(f"[bridge] Fast FEM: 3D solve failed; using 2D truss ({exc})")

        return self._topo.solve_full(load_n, load_x_frac, load_y_frac), None

    def _forces_with_physical_strain(
        self,
        base_forces: Dict[int, float],
        strain_readings: Dict[int, float],
    ) -> tuple[Dict[int, float], bool]:
        usable: Dict[int, float] = {}
        for m_idx, strain_ue in strain_readings.items():
            try:
                member_index = int(m_idx)
                value = float(strain_ue)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and abs(value) > 1e-6:
                usable[member_index] = value
        if not usable:
            return base_forces, False

        e_pa = self._env.get_degraded_properties().E_pa
        adjusted = dict(base_forces)
        for m_idx, strain_ue in usable.items():
            if 0 <= m_idx < len(self._topo.members):
                adjusted[m_idx] = strain_ue / 1_000_000.0 * e_pa * MEMBER_AREA
        return adjusted, True

    def _process_tick(self, tick: SensorTick, update_visual: bool = True):
        stage = omni.usd.get_context().get_stage()
        if not stage or not stage.GetPrimAtPath("/World/Bridge").IsValid():
            return

        load_x_frac = tick.position_frac
        load_y_frac = 0.5
        if update_visual:
            self._set_load_box_position(load_x_frac, load_y_frac)

        load_n = tick.weight_kg * GRAVITY
        sim_forces, sim_result3d = self._solve_fast_path(
            load_n, load_x_frac, load_y_frac)
        self._update_sensor_residuals(
            tick.strain_readings, sim_forces, sim_result3d)
        forces, using_physical_strain = self._forces_with_physical_strain(
            sim_forces, tick.strain_readings)
        self._last_fem = forces
        self._apply_colors(forces)

        # Live readouts (damage labels stay from last completed pass)
        multiplier = float(tick.metadata.get("weight_multiplier", 1.0))
        raw_weight = tick.metadata.get("raw_weight_kg")
        if multiplier > 1.0 and raw_weight is not None:
            weight_text = (
                f"{tick.weight_kg:.3f} kg ({float(raw_weight):.3f} kg x "
                f"{multiplier:g})")
        else:
            weight_text = f"{tick.weight_kg:.3f} kg"
        self._lbl_weight.text = (
            f"{weight_text}   pos {tick.position_frac:.2f}")
        if forces:
            worst_idx = max(forces, key=lambda m: abs(forces[m]))
            if sim_result3d is not None and not using_physical_strain:
                worst_ratio = sim_result3d.stress_ratios.get(worst_idx, 0.0)
            else:
                worst_ratio = abs(forces[worst_idx]) / MEMBER_AREA / YIELD_STRENGTH
            _, _, mt = self._topo.members[worst_idx]
            self._lbl_worst_stress.text = (
                f"M{worst_idx} ({mt})   {worst_ratio*100:.1f}% yield")
        self._publish_live_stoplight(
            tick, forces, result3d=None if using_physical_strain else sim_result3d,
            sim_forces=sim_forces, sim_result3d=sim_result3d)
        if hasattr(self, "_sync_simulation_readouts"):
            self._sync_simulation_readouts()

    def _resonance_detected(self, speed_ms: Optional[float]) -> bool:
        if speed_ms is None or not self._safety.natural_frequency_hz:
            return False
        return self._safety.natural_frequency_hz > 0 and (
            abs(speed_ms / REAL_BRIDGE_LENGTH - self._safety.natural_frequency_hz)
            / self._safety.natural_frequency_hz < 0.15
        )

    def _publish_live_stoplight(
        self,
        tick: SensorTick,
        forces: Dict[int, float],
        result3d: Optional[FEMResult3D],
        sim_forces: Dict[int, float],
        sim_result3d: Optional[FEMResult3D],
    ) -> None:
        env_props = self._env.get_degraded_properties()
        fem_result = self._make_fem_result(forces, result3d)
        vp = VehiclePass(
            weight_kg=tick.weight_kg,
            speed_ms=tick.speed_ms or 0.0,
            axle_position_frac=tick.position_frac,
            strain_readings=tick.strain_readings,
            pressure_raw=tick.pressure_raw,
            timestamp_ms=tick.timestamp_ms,
            crossing_id=tick.crossing_id,
            metadata=dict(tick.metadata),
        )
        current_capacity_kg, safe_speed = self._capacity_and_speed_limits(
            vp, fem_result, env_props.yield_pa)
        alerts = self._safety.check(
            fem_result,
            self._damage,
            VehicleParams(
                weight_kg=vp.weight_kg,
                speed_ms=vp.speed_ms,
                axle_position_frac=vp.axle_position_frac,
            ),
            safe_load_kg=current_capacity_kg,
            safe_speed=safe_speed,
            resonance_detected=self._resonance_detected(tick.speed_ms),
            env_yield_knockdown=self._env_yield_knockdown,
            fast_vs_accurate_error=self._fast_vs_accurate_error,
            log_alerts=False,
        )
        safe_to_pass = self._safe_to_pass_from_alerts(alerts)
        now = time.monotonic()
        state_changed = safe_to_pass != self._last_live_stoplight_state
        if not state_changed and now - self._last_live_stoplight_sent_at < 0.5:
            return
        self._last_live_stoplight_state = safe_to_pass
        self._last_live_stoplight_sent_at = now
        strain_values, strain_value = self._feedback_sensor_values(
            sim_forces, sim_result3d)
        self._publish_feedback(
            current_capacity_kg, safe_speed, alerts,
            strain_values, strain_value)

    def _update_sensor_residuals(
        self,
        measured_microstrain: Dict[int, float],
        forces: Dict[int, float],
        result3d: Optional[FEMResult3D],
    ) -> None:
        if result3d is not None:
            predicted_stress = dict(result3d.combined_stresses)
        else:
            predicted_stress = {m: f / MEMBER_AREA for m, f in forces.items()}
        e_pa = self._env.get_degraded_properties().E_pa
        self._last_sensor_residuals = compute_sensor_residuals(
            measured_microstrain,
            predicted_stress,
            self._active_gauged_members(),
            e_pa,
        )
        if hasattr(self, "_lbl_sensor_residuals"):
            if not self._last_sensor_residuals:
                self._lbl_sensor_residuals.text = "Gauge residuals: --"
                self._lbl_sensor_residuals.style = {"color": 0xFF888888}
                return
            parts = []
            worst_status = "ok"
            for r in self._last_sensor_residuals:
                parts.append(
                    f"M{r.member_index}: {r.residual_microstrain:+.0f}ue "
                    f"({r.status})")
                if r.status in ("outlier", "missing"):
                    worst_status = r.status
                elif worst_status == "ok" and r.status in ("drift", "stale"):
                    worst_status = r.status
            self._lbl_sensor_residuals.text = "Gauge residuals: " + " | ".join(parts)
            self._lbl_sensor_residuals.style = {
                "color": (
                    0xFF5555FF if worst_status in ("outlier", "missing")
                    else 0xFF00AAFF if worst_status in ("drift", "stale")
                    else 0xFF88FF88
                )
            }
        if hasattr(self, "_sync_simulation_readouts"):
            self._sync_simulation_readouts()

    def _record_damage(self, vp: VehiclePass):
        stage = omni.usd.get_context().get_stage()
        if not stage or not stage.GetPrimAtPath("/World/Bridge").IsValid():
            return

        load_n = vp.weight_kg * GRAVITY
        sim_forces, sim_result3d = self._solve_fast_path(
            load_n, vp.axle_position_frac, 0.5)
        self._update_sensor_residuals(vp.strain_readings, sim_forces, sim_result3d)
        forces, using_physical_strain = self._forces_with_physical_strain(
            sim_forces, vp.strain_readings)
        result3d = None if using_physical_strain else sim_result3d
        if result3d is not None:
            member_stresses = dict(result3d.combined_stresses)
        else:
            member_stresses = {m: abs(f) / MEMBER_AREA for m, f in forces.items()}

        env_props = self._env.get_degraded_properties()
        n_cycles = 1 if self._sensor.is_live else SIM_CYCLES_PER_PASS
        fatigue_factor = max(
            0.1, env_props.fatigue_limit_pa / max(FATIGUE_LIMIT_PA, 1.0))
        fatigue_adjusted_stresses = {
            m: s / fatigue_factor for m, s in member_stresses.items()
        }
        self._damage.record_pass_simple(
            fatigue_adjusted_stresses, n_cycles=n_cycles)
        self._damage.save()
        if self._coloring_mode == "damage":
            self._apply_damage_colors()

        fem_result = self._make_fem_result(forces, result3d)
        current_capacity_kg, safe_speed = self._capacity_and_speed_limits(
            vp, fem_result, env_props.yield_pa)
        vehicle_params = VehicleParams(
            weight_kg=vp.weight_kg,
            speed_ms=vp.speed_ms,
            axle_position_frac=vp.axle_position_frac,
        )
        # Check resonance using cached natural frequency from dynamic analyser
        resonance = False
        if self._safety.natural_frequency_hz:
            resonance = self._safety.natural_frequency_hz > 0 and (
                abs(vp.speed_ms / REAL_BRIDGE_LENGTH
                    - self._safety.natural_frequency_hz)
                / self._safety.natural_frequency_hz < 0.15
            )

        alerts = self._safety.check(
            fem_result, self._damage, vehicle_params,
            safe_load_kg=current_capacity_kg,
            safe_speed=safe_speed,
            resonance_detected=resonance,
            env_yield_knockdown=self._env_yield_knockdown,
            fast_vs_accurate_error=self._fast_vs_accurate_error,
        )
        self._last_alerts = alerts
        strain_values, strain_value = self._feedback_sensor_values(
            sim_forces, sim_result3d)
        self._publish_feedback(
            current_capacity_kg, safe_speed, alerts,
            strain_values, strain_value)

        # Enqueue to background dynamic analyser
        has_critical = any(a.level == AlertLevel.CRITICAL for a in alerts)
        self._dyn_mgr.enqueue_pass(
            vp, forces, MEMBER_AREA, forced=has_critical)

        # Refresh damage-specific UI
        self._update_damage_readout()
        self._lbl_passes.text = f"Passes: {self._damage.pass_count}"
        if hasattr(self, "_update_crack_label"):
            self._update_crack_label()
        daf = 1.0 + 0.5 * (vp.speed_ms / V_MAX_PROTOTYPE) ** 2
        self._lbl_speed.text = (
            f"{vp.speed_ms:.2f} m/s   DAF {daf:.3f}")
        multiplier = float(vp.metadata.get("weight_multiplier", 1.0))
        raw_weight = vp.metadata.get("raw_weight_kg")
        if multiplier > 1.0 and raw_weight is not None:
            self._lbl_weight.text = (
                f"{vp.weight_kg:.3f} kg ({float(raw_weight):.3f} kg x "
                f"{multiplier:g} x DAF)")
        else:
            self._lbl_weight.text = f"{vp.weight_kg:.3f} kg"
        self._lbl_capacity.text = (
            f"Capacity: {current_capacity_kg:.2f} kg  "
            f"|  Limit: {safe_speed:.2f} m/s")
        self._update_feedback_readout()

        shown = alerts[:3]
        for k, lbl in enumerate(self._alert_labels):
            if k < len(shown):
                a = shown[k]
                text = str(a)
                if hasattr(self, "_wrap_alert_text"):
                    text = self._wrap_alert_text(text)
                lbl.text = text
                lbl.style = _alert_style(a.level)
            else:
                lbl.text = ""
        if hasattr(self, "_sync_simulation_readouts"):
            self._sync_simulation_readouts()

    def _update_damage_readout(self) -> None:
        if not hasattr(self, "_lbl_worst_damage"):
            return
        damage = self._damage.all_damage()
        if not damage:
            self._lbl_worst_damage.text = "Damage: --"
            return
        top = sorted(damage.items(), key=lambda item: item[1], reverse=True)[:4]
        parts = []
        for m_idx, value in top:
            _, _, mt = self._topo.members[m_idx]
            parts.append(f"M{m_idx} ({mt}) D={value:.3f}")
        self._lbl_worst_damage.text = "Damage: " + " | ".join(parts)

    def _capacity_and_speed_limits(
        self,
        vp: VehiclePass,
        fem_result: FEMResult,
        env_yield_pa: float,
    ) -> tuple[float, float]:
        max_ratio = max(
            (abs(ratio) for ratio in fem_result.stress_ratios.values()),
            default=0.0,
        )
        if max_ratio > 1e-9 and vp.weight_kg > 0.0:
            live_capacity_kg = vp.weight_kg * 0.70 / max_ratio
            # The live FEM pass can expose a lower governing limit, but it must
            # not raise the bridge above the static undamaged rating.
            base_capacity_kg = min(self._safe_load_kg, live_capacity_kg)
        else:
            base_capacity_kg = self._safe_load_kg

        residual_factor = self._damage.residual_capacity_factor()
        env_capacity_factor = max(0.1, env_yield_pa / YIELD_STRENGTH)
        current_capacity_kg = (
            base_capacity_kg * residual_factor * env_capacity_factor)

        weight_ratio = min(1.0, vp.weight_kg / max(current_capacity_kg, 1e-9))
        safe_speed = max(
            0.3,
            V_MAX_PROTOTYPE * residual_factor * (1.0 - 0.7 * weight_ratio),
        )
        return current_capacity_kg, safe_speed

    def _publish_feedback(
        self,
        max_load_kg: float,
        safe_speed_ms: float,
        alerts: List[Alert],
        strain_values: Optional[List[float]] = None,
        strain_value: Optional[float] = None,
    ) -> None:
        max_level = max((a.level for a in alerts), default=AlertLevel.INFO)
        first = alerts[0].message if alerts else "Bridge within current limits"
        if max_level == AlertLevel.CRITICAL:
            advisory = "stop"
        elif max_level == AlertLevel.WARNING:
            advisory = "reduce_speed"
        else:
            advisory = "ok"
        safe_to_pass = self._safe_to_pass_from_alerts(alerts)
        try:
            command = FeedbackCommand(
                max_load_kg=max_load_kg,
                safe_speed_ms=safe_speed_ms,
                advisory=advisory,
                alert_level=max_level.name,
                reason=first,
                strain_values=strain_values or [],
                strain_value=strain_value,
                safe_to_pass=safe_to_pass,
            )
            self._sensor.set_control_feedback(command)
            self._update_feedback_readout(command.payload())
        except Exception as exc:
            print(f"[bridge] Could not queue feedback update: {exc}")

    def _safe_to_pass_from_alerts(self, alerts: List[Alert]) -> bool:
        return not any(a.level >= AlertLevel.WARNING for a in alerts)

    def _format_feedback_payload(self, payload: dict) -> str:
        if not payload:
            return (
                "maxLoad: --\n"
                "safeToPass: --\n"
                "twin1..4: --, --, --, --\n"
                "averageStrainTwin: --"
            )

        def fmt(name: str, suffix: str = "") -> str:
            value = payload.get(name)
            try:
                return f"{float(value):.2f}{suffix}"
            except (TypeError, ValueError):
                return "--"

        strains = ", ".join(
            fmt(f"twin{i}", " ue") for i in range(1, STRAIN_GAUGE_COUNT + 1)
        )
        safe_raw = payload.get("safeToPass")
        if safe_raw is None:
            safe_text = "--"
        else:
            try:
                safe = bool(int(safe_raw))
            except (TypeError, ValueError):
                safe = bool(safe_raw)
            safe_text = f"{'SAFE' if safe else 'STOP'} ({1 if safe else 0})"
        return (
            f"maxLoad: {fmt('maxLoad', ' kg')}\n"
            f"safeToPass: {safe_text}\n"
            f"twin1..4: {strains}\n"
            f"averageStrainTwin: {fmt('averageStrainTwin', ' ue')}"
        )

    def _update_feedback_readout(self, payload: Optional[dict] = None) -> None:
        if not hasattr(self, "_sensor"):
            return
        fs = self._sensor.feedback_status
        if hasattr(self, "_lbl_feedback_status"):
            text = f"Feedback: {fs.state}"
            if fs.last_error:
                text += f" ({fs.last_error[:48]})"
            self._lbl_feedback_status.text = text
            self._lbl_feedback_status.style = {
                "color": 0xFF88FF88 if fs.state == "sent"
                else 0xFF00AAFF if fs.state in ("pending", "failed")
                else 0xFF888888
            }

        payload = payload if payload is not None else fs.last_payload
        text = self._format_feedback_payload(payload)
        if hasattr(self, "_lbl_feedback_payload"):
            self._lbl_feedback_payload.text = text
        if hasattr(self, "_sim_lbl_feedback_payload"):
            self._sim_lbl_feedback_payload.text = text

    def _feedback_sensor_values(
        self,
        forces: Dict[int, float],
        result3d: Optional[FEMResult3D],
    ) -> tuple[List[float], float]:
        if result3d is not None:
            predicted_stress = dict(result3d.combined_stresses)
        else:
            predicted_stress = {m: f / MEMBER_AREA for m, f in forces.items()}

        active_members = self._active_gauged_members()
        e_pa = self._env.get_degraded_properties().E_pa
        strain_microstrain = [
            predicted_stress.get(m_idx, 0.0)
            / max(e_pa, 1e-9)
            * 1_000_000.0
            * STRAIN_FEEDBACK_GAIN
            for m_idx in active_members[:STRAIN_GAUGE_COUNT]
        ]
        while len(strain_microstrain) < STRAIN_GAUGE_COUNT:
            strain_microstrain.append(0.0)
        strain_microstrain = strain_microstrain[:STRAIN_GAUGE_COUNT]
        average_strain = sum(strain_microstrain) / len(strain_microstrain)
        return strain_microstrain, average_strain

    def _make_fem_result(
        self, forces: Dict[int, float], result3d: Optional[FEMResult3D] = None
    ) -> FEMResult:
        axial_forces = dict(forces)
        if result3d is not None:
            axial_stresses = {
                m: result3d.combined_stresses.get(m, 0.0)
                for m in forces
            }
        else:
            axial_stresses = {m: f / MEMBER_AREA for m, f in forces.items()}
        stress_ratios = {m: abs(s) / YIELD_STRENGTH for m, s in axial_stresses.items()}
        displacements = np.zeros(1)  # placeholder -- full u vector not needed here
        return FEMResult(
            displacements=displacements,
            axial_forces=axial_forces,
            axial_stresses=axial_stresses,
            stress_ratios=stress_ratios,
        )

    # USD color + attribute writer
    def _apply_colors(self, forces: Dict[int, float]):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        if self._coloring_mode == "damage":
            self._apply_damage_colors()
            return
        for m_idx, force in forces.items():
            stress = abs(force) / MEMBER_AREA
            ratio  = stress / YIELD_STRENGTH
            d      = self._damage.get_damage(m_idx)

            color = _stress_color(ratio)

            # Euler buckling utilisation for compression members
            if force < 0:
                length = self._topo.member_real_length(m_idx)
                eff = BUCKLING_K * length
                p_cr = (math.pi ** 2 * E_MODULUS * MEMBER_I / eff ** 2
                        if eff > 1e-9 else float("inf"))
                buckle_util = abs(force) / p_cr
            else:
                buckle_util = 0.0
            util = max(ratio, buckle_util)
            mode = "buckling" if buckle_util > ratio else "yield"

            for path in self._member_prims.get(m_idx, []):
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                _set_attr(prim, "analysis:stress",      float(stress))
                _set_attr(prim, "analysis:stressRatio", float(ratio))
                _set_attr(prim, "analysis:axialForce",  float(force))
                _set_attr(prim, "analysis:utilisation", float(util))
                _set_attr(prim, "analysis:damage",      float(d))
                _set_attr(prim, "analysis:failureMode",
                          mode if util >= 1.0 else "ok")
                gp = UsdGeom.Gprim(prim)
                ca = gp.CreateDisplayColorAttr()
                ca.Set([color])
                UsdGeom.Primvar(ca).SetInterpolation(UsdGeom.Tokens.constant)

    def _apply_damage_colors(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        n_members = len(getattr(self._topo, "members", []))
        for m_idx in range(n_members):
            d = self._damage.get_damage(m_idx)
            color = _damage_color(d)
            for path in self._member_prims.get(m_idx, []):
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                _set_attr(prim, "analysis:damage", float(d))
                gp = UsdGeom.Gprim(prim)
                ca = gp.CreateDisplayColorAttr()
                ca.Set([color])
                UsdGeom.Primvar(ca).SetInterpolation(UsdGeom.Tokens.constant)



