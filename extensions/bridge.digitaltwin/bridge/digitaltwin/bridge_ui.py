# Omniverse UI construction and button callbacks.
#
# BridgeUIMixin owns labels, tabs, sliders, and connection controls. It calls
# methods implemented by extension.py for bridge building, FEM solving, sensor
# reconnects, and damage resets.
from __future__ import annotations

import math
import random
import textwrap
from typing import List

import omni.usd
import omni.ui as ui
from pxr import Gf, UsdGeom

from .bridge_config import (
    E_MODULUS, YIELD_STRENGTH, GRAVITY, MEMBER_AREA, MEMBER_I, BUCKLING_K,
    REAL_BRIDGE_LENGTH, REAL_TRUSS_WIDTH, NUM_PANELS,
    TRUSS_LENGTH, TRUSS_WIDTH, MEMBER_THICK, SCENE_SCALE,
    V_MAX_PROTOTYPE, STRAIN_GAUGE_COUNT,
    SIM_WEIGHT_MIN_KG, SIM_WEIGHT_MAX_KG,
)
from .bridge_geometry import _define_box_mesh
from .sensor_reader import (
    ConnectionConfig, TrafficMode, VehiclePass, _daf_calibrated,
)


class BridgeUIMixin:

    # UI
    def _build_ui(self):
        gauge_field_hint = f"S0:<ue>...S{STRAIN_GAUGE_COUNT - 1}:<ue>"
        self._window = ui.Window("Bridge Digital Twin", width=420, height=660)
        with self._window.frame:
            with ui.VStack(spacing=6):
                # --- Always-visible build button ---
                ui.Button("Load New Bridge", height=34,
                          clicked_fn=self._build_bridge,
                          style={"background_color": 0xFF2255AA})
                ui.Separator(height=2)

                with ui.VStack(spacing=3):
                    with ui.HStack(height=22):
                        ui.Label("Display:", width=70)
                        self._mode_combo = ui.ComboBox(
                            0, "Live stress", "Cumulative damage")
                        self._mode_combo.model.add_item_changed_fn(
                            lambda m, i: self._on_mode_change(m, i))
                    with ui.HStack(height=22):
                        ui.Label("Weight:", width=70)
                        self._weight_mode_idx = self._weight_multiplier_to_index(
                            getattr(self._conn_config, "weight_multiplier", 1.0))
                        self._weight_mode_combo = ui.ComboBox(
                            self._weight_mode_idx,
                            "1x normal", "10x demo", "25x demo",
                            "50x demo", "100x demo")
                        self._weight_mode_combo.model.add_item_changed_fn(
                            self._on_weight_mode_changed)
                    with ui.HStack(height=28, spacing=6):
                        ui.Button("Reset damage", height=26,
                                  clicked_fn=self._on_reset_damage)
                ui.Separator(height=2)

                # --- Tab bar ---
                with ui.HStack(height=30):
                    self._tab_live   = ui.Button("Live Sensor", height=28,
                                                  clicked_fn=lambda: self._set_tab("live"))
                    self._tab_manual = ui.Button("Manual Test", height=28,
                                                  clicked_fn=lambda: self._set_tab("manual"))
                    self._tab_sim    = ui.Button("Simulation", height=28,
                                                  clicked_fn=lambda: self._set_tab("sim"))
                    self._tab_conn   = ui.Button("Connection", height=28,
                                                  clicked_fn=lambda: self._set_tab("conn"))

                self._live_frame   = ui.Frame()
                self._manual_frame = ui.Frame(visible=False)
                self._sim_frame    = ui.Frame(visible=False)
                self._conn_frame   = ui.Frame(visible=False)

                with self._live_frame:
                    with ui.VStack(spacing=3):
                        ui.Label("VEHICLE", style={"color": 0xFF888888})
                        with ui.HStack(height=18):
                            self._lbl_speed  = ui.Label("Speed: --")
                        with ui.HStack(height=18):
                            self._lbl_weight = ui.Label("Weight: --   pos: --")

                        ui.Separator(height=4)

                        ui.Label("BRIDGE", style={"color": 0xFF888888})
                        self._lbl_capacity = ui.Label(
                            "Capacity: --  |  Speed limit: --",
                            style={"color": 0xFF88CCFF})
                        self._lbl_worst_stress = ui.Label("Stress: --")
                        self._lbl_sensor_residuals = ui.Label(
                            "Gauge residuals: --",
                            style={"color": 0xFF888888})
                        with ui.HStack(height=18):
                            self._lbl_worst_damage = ui.Label("Damage: --")
                            self._lbl_passes = ui.Label(
                                "Passes: 0", style={"color": 0xFF888888})
                        self._lbl_worst_crack = ui.Label(
                            "Crack: --", style={"color": 0xFF88FF88})

                        ui.Separator(height=4)

                        ui.Label("ALERTS", style={"color": 0xFF888888})
                        self._alert_wrap_chars = 48
                        self._alert_labels = [
                            ui.Label("", height=42, width=0)
                            for _ in range(3)
                        ]

                        ui.Separator(height=4)

                        with ui.HStack(height=28, spacing=6):
                            ui.Button("Connect WebSocket", height=26,
                                      clicked_fn=self._on_connect_websocket,
                                      style={"background_color": 0xFF2255AA})

                        ui.Separator(height=4)

                        ui.Label("DYNAMIC FEM", style={"color": 0xFF888888})
                        self._lbl_analysis_source = ui.Label(
                            "Source: fast-only",
                            style={"color": 0xFF888888})
                        self._lbl_dyn_status = ui.Label(
                            "Dynamic FEM: idle",
                            style={"color": 0xFF666666})
                        self._lbl_fast_model = ui.Label(
                            "Model: 2D truss (axial only)",
                            style={"color": 0xFF888888})
                        self._lbl_dyn_model = ui.Label(
                            "Accurate: --",
                            style={"color": 0xFF888888})
                        self._lbl_nat_freq = ui.Label(
                            "f1: --  |  DAF: --",
                            style={"color": 0xFF88CCFF})
                        self._lbl_env_degradation = ui.Label(
                            "Material ageing: --",
                            style={"color": 0xFF888888})
                        self._lbl_feedback_status = ui.Label(
                            "Feedback: idle",
                            style={"color": 0xFF888888})
                        with ui.Frame(
                            height=104,
                            style={
                                "background_color": 0xFF1A1A1A,
                                "border_color": 0xFF555555,
                                "border_width": 1,
                            },
                        ):
                            with ui.VStack(spacing=2):
                                ui.Label("SENDING", style={"color": 0xFF888888})
                                self._lbl_feedback_payload = ui.Label(
                                    "maxLoad: --\n"
                                    "safeToPass: --\n"
                                    "twin1..4: --, --, --, --\n"
                                    "averageStrainTwin: --",
                                    style={"color": 0xFFCCCCCC},
                                )
                        with ui.HStack(height=26, spacing=6):
                            ui.Button("Run full analysis now", height=24,
                                      clicked_fn=self._on_run_analysis_now,
                                      style={"background_color": 0xFF553311})

                with self._manual_frame:
                    with ui.VStack(spacing=5):
                        ui.Label("LOAD BOX")
                        ui.Label("Weight (kg)")
                        self._weight_model = ui.SimpleFloatModel(20.0)
                        ui.FloatSlider(self._weight_model, min=0.0, max=200.0)
                        ui.Label("Position along span (X)  0=left  1=right")
                        self._posx_model = ui.SimpleFloatModel(0.5)
                        ui.FloatSlider(self._posx_model, min=0.0, max=1.0)
                        ui.Label("Position across width (Y)  0=left truss  1=right")
                        self._posy_model = ui.SimpleFloatModel(0.5)
                        ui.FloatSlider(self._posy_model, min=0.0, max=1.0)
                        ui.Button("Update Stress / Recolor",
                                  clicked_fn=self._update_manual_accurate, height=30)
                        with ui.HStack(height=22):
                            self._live_cb = ui.CheckBox(width=22)
                            self._live_cb.model.set_value(True)
                            ui.Label("  Live update")
                        with ui.HStack(height=22):
                            self._bending_cb = ui.CheckBox(width=22)
                            self._bending_cb.model.set_value(
                                bool(getattr(self, "_use_3d_fast_solver", False)))
                            self._bending_cb.model.add_value_changed_fn(
                                self._on_bending_toggle)
                            ui.Label("  Include bending stress")
                        ui.Separator(height=3)
                        self._status  = ui.Label("Build the bridge first.")
                        self._detail  = ui.Label("")
                        self._buckling_lbl = ui.Label("")
                        ui.Label("Educational model -- demo cross-sections.",
                                 style={"color": 0xFF888888})

                with self._sim_frame:
                    with ui.VStack(spacing=5):
                        ui.Label("Simulation traffic", style={"color": 0xFFCCCCCC})
                        ui.Separator(height=3)
                        with ui.HStack(height=22):
                            ui.Label("Pattern:", width=80)
                            self._conn_traffic_idx = self._traffic_mode_to_index(
                                getattr(self._conn_config, "traffic_mode",
                                        TrafficMode.UNIFORM.value))
                            self._conn_traffic_combo = ui.ComboBox(
                                self._conn_traffic_idx,
                                "Uniform random", "Realistic spectrum",
                                "Rush hour")
                            self._conn_traffic_combo.model.add_item_changed_fn(
                                self._on_traffic_mode_changed)
                        with ui.HStack(height=28, spacing=6):
                            ui.Button("Run simulation", height=26,
                                      clicked_fn=self._on_apply_simulation,
                                      style={"background_color": 0xFF224422})
                            ui.Button("Pause / resume", height=26,
                                      clicked_fn=self._on_toggle_sim)
                        self._sim_status_lbl = ui.Label(
                            "Simulation: not active",
                            style={"color": 0xFF888888})
                        ui.Separator(height=4)
                        ui.Label("VEHICLE", style={"color": 0xFF888888})
                        self._sim_lbl_speed = ui.Label("Speed: --")
                        self._sim_lbl_weight = ui.Label("Weight: --   pos: --")
                        ui.Separator(height=3)
                        ui.Label("BRIDGE", style={"color": 0xFF888888})
                        self._sim_lbl_capacity = ui.Label(
                            "Capacity: --  |  Speed limit: --",
                            style={"color": 0xFF88CCFF})
                        self._sim_lbl_worst_stress = ui.Label("Stress: --")
                        self._sim_lbl_sensor_residuals = ui.Label(
                            "Gauge residuals: --",
                            style={"color": 0xFF888888})
                        with ui.HStack(height=18):
                            self._sim_lbl_worst_damage = ui.Label("Damage: --")
                            self._sim_lbl_passes = ui.Label(
                                "Passes: 0", style={"color": 0xFF888888})
                        self._sim_lbl_worst_crack = ui.Label(
                            "Crack: --", style={"color": 0xFF88FF88})
                        ui.Separator(height=3)
                        ui.Label("OUTGOING DATA", style={"color": 0xFF888888})
                        with ui.Frame(
                            height=86,
                            style={
                                "background_color": 0xFF1A1A1A,
                                "border_color": 0xFF555555,
                                "border_width": 1,
                            },
                        ):
                            self._sim_lbl_feedback_payload = ui.Label(
                                "maxLoad: --\n"
                                "safeToPass: --\n"
                                "twin1..4: --, --, --, --\n"
                                "averageStrainTwin: --",
                                style={"color": 0xFFCCCCCC},
                            )
                        ui.Separator(height=3)
                        ui.Label("ALERTS", style={"color": 0xFF888888})
                        self._sim_alert_labels = [
                            ui.Label("", height=42, width=0)
                            for _ in range(3)
                        ]
                        ui.Separator(height=4)
                        ui.Label("Generated traffic uses model-scale vehicle weights.",
                                 style={"color": 0xFF888888})
                        ui.Label("Uniform: random weight, speed, and gap.",
                                 style={"color": 0xFF777777})
                        ui.Label("Realistic: light/heavy mix with traffic intensity.",
                                 style={"color": 0xFF777777})
                        ui.Label("Rush hour: denser flow, more heavy vehicles, slower centre speed.",
                                 style={"color": 0xFF777777})

                with self._conn_frame:
                    with ui.VStack(spacing=5):
                        ui.Label("Physical sensor connection", style={"color": 0xFFCCCCCC})
                        ui.Separator(height=3)
                        with ui.HStack(height=22):
                            ui.Label("Mode:", width=80)
                            self._conn_mode_idx = 0
                            self._conn_mode_combo = ui.ComboBox(
                                0, "WebSocket / ngrok", "Serial / COM",
                                "WiFi / UDP")
                            self._conn_mode_combo.model.add_item_changed_fn(
                                self._on_conn_mode_changed)
                        ui.Separator(height=3)
                        ui.Label("WebSocket settings", style={"color": 0xFF888888})
                        with ui.HStack(height=22):
                            ui.Label("URL:", width=50)
                            self._conn_ws_url_field = ui.StringField()
                            self._conn_ws_url_field.model.set_value(
                                self._conn_config.ws_url)
                        with ui.HStack(height=22):
                            ui.Label("Auth:", width=50)
                            self._conn_ws_auth_field = ui.StringField()
                            self._conn_ws_auth_field.model.set_value(
                                self._conn_config.ws_basic_auth)
                        ui.Separator(height=3)
                        ui.Label("Serial port  (e.g. COM3 or /dev/ttyUSB0)",
                                 style={"color": 0xFF888888})
                        self._conn_port_field = ui.StringField(height=22)
                        self._conn_port_field.model.set_value("COM3")
                        ui.Separator(height=3)
                        ui.Label("WiFi / UDP settings", style={"color": 0xFF888888})
                        with ui.HStack(height=22):
                            ui.Label("Host:", width=50)
                            self._conn_host_field = ui.StringField()
                            self._conn_host_field.model.set_value("0.0.0.0")
                        with ui.HStack(height=22):
                            ui.Label("Port:", width=50)
                            self._conn_udp_port = ui.IntField(height=22)
                            self._conn_udp_port.model.set_value(5555)
                        ui.Separator(height=3)
                        ui.Button("Apply & Reconnect", height=28,
                                  clicked_fn=self._on_reconnect,
                                  style={"background_color": 0xFF2255AA})
                        self._conn_status = ui.Label(
                            "Mode: WebSocket / ngrok", style={"color": 0xFF00BBFF})

                        ui.Separator(height=3)
                        ui.Label("Strain gauge channels -- attach sensors to these members:",
                                 style={"color": 0xFF888888})
                        self._conn_gauge_labels = [
                            ui.Label(f"CH {ch} -> member: build bridge first")
                            for ch in range(STRAIN_GAUGE_COUNT)
                        ]
                        ui.Label(
                            f"ESP32 packet format:  W:<kg>,P:<0|1>,{gauge_field_hint}",
                            style={"color": 0xFF666666},
                        )
                        ui.Separator(height=3)
                        ui.Label("Dynamic analysis", style={"color": 0xFF888888})
                        self._conn_freq_lbl = ui.Label(
                            "f1: -- Hz  (build bridge to compute)",
                            style={"color": 0xFF88CCFF})
                        ui.Separator(height=3)
                        ui.Label("Environmental exposure",
                                 style={"color": 0xFF888888})
                        with ui.HStack(height=22):
                            ui.Label("Exposure:", width=80)
                            _exp_idx = 1 if self._env.state.exposure == "outdoor" else 0
                            self._conn_exposure_idx = _exp_idx
                            self._conn_exposure_combo = ui.ComboBox(
                                _exp_idx, "Indoor", "Outdoor")
                            self._conn_exposure_combo.model.add_item_changed_fn(
                                self._on_env_exposure_changed)
                        with ui.HStack(height=22):
                            ui.Label("Humidity %:", width=80)
                            _rh_pct = self._env.state.humidity_rh_avg * 100.0
                            self._conn_humidity_slider = ui.FloatSlider(
                                min=0.0, max=100.0, step=1.0)
                            self._conn_humidity_slider.model.set_value(_rh_pct)
                            self._conn_humidity_val_lbl = ui.Label(
                                f"{_rh_pct:.0f}%", width=40,
                                style={"color": 0xFFAAAAAA})
                            self._conn_humidity_slider.model.add_value_changed_fn(
                                self._on_humidity_changed)
                        _exp0 = self._env.state.exposure
                        _rh0  = self._env.state.humidity_rh_avg
                        self._conn_env_lbl = ui.Label(
                            f"Env: {_exp0} | RH: {_rh0*100:.0f}% | yield -0.0%",
                            style={"color": 0xFF888888})

        self._weight_model.add_value_changed_fn(lambda m: self._maybe_live())
        self._posx_model.add_value_changed_fn(lambda m: self._maybe_live())
        self._posy_model.add_value_changed_fn(lambda m: self._maybe_live())
        self._update_crack_label()
        self._sync_simulation_readouts()
        self._refresh_model_labels()

    def _set_tab(self, tab: str):
        self._live_frame.visible   = (tab == "live")
        self._manual_frame.visible = (tab == "manual")
        self._sim_frame.visible    = (tab == "sim")
        self._conn_frame.visible   = (tab == "conn")

    def _wrap_alert_text(self, text: str) -> str:
        width = getattr(self, "_alert_wrap_chars", 48)
        return "\n".join(textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        ))

    def _copy_label_state(self, src, dst) -> None:
        if not src or not dst:
            return
        dst.text = src.text
        try:
            dst.style = src.style
        except Exception:
            pass

    def _sync_simulation_readouts(self) -> None:
        pairs = [
            ("_lbl_speed", "_sim_lbl_speed"),
            ("_lbl_weight", "_sim_lbl_weight"),
            ("_lbl_capacity", "_sim_lbl_capacity"),
            ("_lbl_worst_stress", "_sim_lbl_worst_stress"),
            ("_lbl_sensor_residuals", "_sim_lbl_sensor_residuals"),
            ("_lbl_worst_damage", "_sim_lbl_worst_damage"),
            ("_lbl_passes", "_sim_lbl_passes"),
            ("_lbl_worst_crack", "_sim_lbl_worst_crack"),
            ("_lbl_feedback_payload", "_sim_lbl_feedback_payload"),
        ]
        for src_name, dst_name in pairs:
            if hasattr(self, src_name) and hasattr(self, dst_name):
                self._copy_label_state(getattr(self, src_name), getattr(self, dst_name))
        if hasattr(self, "_alert_labels") and hasattr(self, "_sim_alert_labels"):
            for src, dst in zip(self._alert_labels, self._sim_alert_labels):
                self._copy_label_state(src, dst)

    def _on_mode_change(self, model, item):
        idx = model.get_item_value_model(item).as_int
        self._coloring_mode = "stress" if idx == 0 else "damage"
        if self._coloring_mode == "damage" and hasattr(self, "_apply_damage_colors"):
            self._apply_damage_colors()
        elif self._last_fem:
            self._apply_colors(self._last_fem)

    # Manual test tab (slider-driven, identical logic to original)
    def _maybe_live(self):
        if not self._live_cb.model.get_value_as_bool():
            return
        if self._updating:
            return
        self._updating = True
        try:
            self._update_manual()
        except Exception as exc:
            print("[bridge] live update error:", exc)
        finally:
            self._updating = False

    def _update_manual(self):
        stage = omni.usd.get_context().get_stage()
        bridge = stage.GetPrimAtPath("/World/Bridge") if stage else None
        if not bridge or not bridge.IsValid():
            self._status.text = "Build the bridge first."
            return

        weight_kg = self._weight_model.get_value_as_float()
        pos_x = self._posx_model.get_value_as_float()
        pos_y = self._posy_model.get_value_as_float()
        load_n = weight_kg * GRAVITY

        local_x = (-TRUSS_LENGTH / 2.0) + pos_x * TRUSS_LENGTH
        local_y = (-TRUSS_WIDTH / 2.0)  + pos_y * TRUSS_WIDTH
        local_z = MEMBER_THICK * 0.4 + self._box_size / 2.0
        if hasattr(self, "_set_load_box_position"):
            self._set_load_box_position(pos_x, pos_y)
        elif self._box_translate:
            self._box_translate.Set(Gf.Vec3d(local_x, local_z, -local_y))

        if hasattr(self, "_solve_fast_path"):
            forces, result3d = self._solve_fast_path(load_n, pos_x, pos_y)
        else:
            forces = self._topo.solve_full(load_n, pos_x, pos_y)
            result3d = None
        self._last_fem = forces
        self._apply_colors(forces)

        max_util = 0.0
        worst_member = ""
        worst_force = 0.0
        n_buckled = n_yielded = 0

        for m_idx, force in forces.items():
            if result3d is not None:
                force = result3d.axial_forces.get(m_idx, force)
                ratio = result3d.stress_ratios.get(m_idx, 0.0)
            else:
                stress = abs(force) / MEMBER_AREA
                ratio = stress / YIELD_STRENGTH
            length = self._topo.member_real_length(m_idx)
            if force < 0 and length > 1e-9:
                eff  = BUCKLING_K * length
                p_cr = (math.pi ** 2 * E_MODULUS * MEMBER_I / eff ** 2)
                bu   = abs(force) / p_cr
            else:
                bu = 0.0
            util = max(ratio, bu)
            mode = "buckling" if bu > ratio else "yield"
            if util >= 1.0:
                n_buckled += (mode == "buckling")
                n_yielded += (mode == "yield")
            if util > max_util:
                max_util = util
                _, _, mt = self._topo.members[m_idx]
                worst_member = f"M{m_idx} ({mt})"
                worst_force  = force

        state = "SAFE" if max_util < 0.6 else ("WARNING" if max_util < 1.0 else "FAILURE")
        sign  = "tension" if worst_force >= 0 else "compression"
        self._status.text = (f"Load {load_n:.2f} N  |  "
                              f"Max util {max_util:.2f}  |  {state}")
        self._detail.text = (f"Worst: {worst_member}  "
                              f"{abs(worst_force):.2f} N {sign}")
        self._buckling_lbl.text = (f"Failed: {n_buckled} buckled, {n_yielded} yielded")

    def _update_manual_accurate(self):
        stage = omni.usd.get_context().get_stage()
        bridge = stage.GetPrimAtPath("/World/Bridge") if stage else None
        if not bridge or not bridge.IsValid():
            self._status.text = "Build the bridge first."
            return

        analyser = getattr(getattr(self, "_dyn_mgr", None), "_analyser", None)
        if analyser is None:
            self._status.text = "Accurate FEM unavailable; using live solver."
            self._update_manual()
            return

        weight_kg = self._weight_model.get_value_as_float()
        pos_x = self._posx_model.get_value_as_float()
        pos_y = self._posy_model.get_value_as_float()
        if hasattr(self, "_set_load_box_position"):
            self._set_load_box_position(pos_x, pos_y)

        self._status.text = "Running accurate OpenSees FEM..."
        try:
            env_props = self._env.get_degraded_properties()
            cr = analyser.run_static_position(
                weight_kg=weight_kg,
                x_frac=pos_x,
                lateral_frac=pos_y,
                bridge_length_m=REAL_BRIDGE_LENGTH,
                E_override=env_props.E_pa,
                yield_override=env_props.yield_pa,
            )
        except Exception as exc:
            self._status.text = f"Accurate FEM failed; using live solver ({exc})"
            self._update_manual()
            return

        yield_pa = self._env.get_degraded_properties().yield_pa
        forces = {
            m_idx: stress * MEMBER_AREA
            for m_idx, stress in cr.peak_stresses.items()
        }
        self._last_fem = forces
        self._apply_colors(forces)

        if not cr.peak_stresses:
            self._status.text = "Accurate FEM returned no stresses."
            return

        worst_idx, worst_stress = max(
            cr.peak_stresses.items(), key=lambda item: abs(item[1]))
        worst_ratio = abs(worst_stress) / max(yield_pa, 1e-9)
        _, _, mt = self._topo.members[worst_idx]
        load_n = weight_kg * GRAVITY
        state = "SAFE" if worst_ratio < 0.6 else (
            "WARNING" if worst_ratio < 1.0 else "FAILURE")
        source = "OpenSees 3D static" if analyser.is_3d_frame else "OpenSees 2D static"
        conv = "" if cr.converged else " (partial)"
        self._status.text = (
            f"Load {load_n:.2f} N  |  Max util {worst_ratio:.2f}  |  {state}")
        self._detail.text = (
            f"Worst: M{worst_idx} ({mt})  {abs(worst_stress)/1e6:.2f} MPa  "
            f"{source}{conv}")
        self._buckling_lbl.text = (
            f"Accurate fixed-position solve  |  steps: {cr.steps_completed}")

    # Environmental exposure callbacks (Connection tab)
    def _combo_item_index(self, model, item, default: int = 0) -> int:
        try:
            return model.get_item_value_model(item).as_int
        except Exception:
            return default

    def _on_conn_mode_changed(self, model, item) -> None:
        self._conn_mode_idx = self._combo_item_index(
            model, item, getattr(self, "_conn_mode_idx", 0))

    def _traffic_mode_to_index(self, mode: str) -> int:
        values = [
            TrafficMode.UNIFORM.value,
            TrafficMode.REALISTIC.value,
            TrafficMode.RUSH_HOUR.value,
        ]
        try:
            return values.index(str(mode))
        except ValueError:
            return 0

    def _traffic_index_to_mode(self, index: int) -> str:
        values = [
            TrafficMode.UNIFORM.value,
            TrafficMode.REALISTIC.value,
            TrafficMode.RUSH_HOUR.value,
        ]
        return values[max(0, min(index, len(values) - 1))]

    def _weight_multiplier_to_index(self, multiplier: float) -> int:
        values = [1.0, 10.0, 25.0, 50.0, 100.0]
        try:
            nearest = min(values, key=lambda value: abs(value - float(multiplier)))
            return values.index(nearest)
        except (TypeError, ValueError):
            return 0

    def _weight_index_to_multiplier(self, index: int) -> float:
        values = [1.0, 10.0, 25.0, 50.0, 100.0]
        return values[max(0, min(index, len(values) - 1))]

    def _on_traffic_mode_changed(self, model, item) -> None:
        self._conn_traffic_idx = self._combo_item_index(
            model, item, getattr(self, "_conn_traffic_idx", 0))
        mode = self._traffic_index_to_mode(self._conn_traffic_idx)
        if hasattr(self, "_sensor"):
            self._sensor.set_traffic_mode(mode)
        if hasattr(self, "_conn_config"):
            self._conn_config.traffic_mode = mode

    def _on_weight_mode_changed(self, model, item) -> None:
        self._weight_mode_idx = self._combo_item_index(
            model, item, getattr(self, "_weight_mode_idx", 0))
        multiplier = self._weight_index_to_multiplier(self._weight_mode_idx)
        if hasattr(self, "_sensor"):
            self._sensor.set_weight_multiplier(multiplier)
        if hasattr(self, "_conn_config"):
            self._conn_config.weight_multiplier = multiplier
        if hasattr(self, "_sim_status_lbl"):
            self._sim_status_lbl.text = (
                "Simulation: weight mode "
                f"{multiplier:g}x" + (
                    " active" if getattr(self._sensor, "mode", "") == "sim" else " set"))

    def _on_bending_toggle(self, model) -> None:
        requested = model.get_value_as_bool()
        available = getattr(self, "_frame_fem3d", None) is not None
        self._use_3d_fast_solver = bool(requested and available)
        if requested and not available:
            model.set_value(False)
        self._refresh_model_labels()
        if hasattr(self, "_live_cb") and self._live_cb.model.get_value_as_bool():
            self._maybe_live()
        elif getattr(self, "_last_fem", None):
            self._apply_colors(self._last_fem)

    def _refresh_model_labels(self) -> None:
        enabled = bool(getattr(self, "_use_3d_fast_solver", False))
        available = getattr(self, "_frame_fem3d", None) is not None
        if hasattr(self, "_lbl_fast_model"):
            if enabled and available:
                self._lbl_fast_model.text = "Model: 3D frame (bending + torsion)"
                self._lbl_fast_model.style = {"color": 0xFF88CCFF}
            else:
                self._lbl_fast_model.text = "Model: 2D truss (axial only)"
                self._lbl_fast_model.style = {"color": 0xFF888888}
        analyser = getattr(getattr(self, "_dyn_mgr", None), "_analyser", None)
        if hasattr(self, "_lbl_dyn_model"):
            if analyser is None:
                self._lbl_dyn_model.text = "Accurate: unavailable"
                self._lbl_dyn_model.style = {"color": 0xFF888888}
            elif not getattr(self, "_opensees_available", True):
                self._lbl_dyn_model.text = "Accurate: static fallback (OpenSeesPy unavailable)"
                self._lbl_dyn_model.style = {"color": 0xFF00AAFF}
            elif getattr(analyser, "is_3d_frame", False):
                self._lbl_dyn_model.text = "Accurate: 3D frame (bending + torsion)"
                self._lbl_dyn_model.style = {"color": 0xFF88CCFF}
            else:
                self._lbl_dyn_model.text = "Accurate: 2D truss (axial only)"
                self._lbl_dyn_model.style = {"color": 0xFF888888}

    def _update_crack_label(self) -> None:
        if not hasattr(self, "_lbl_worst_crack"):
            return
        try:
            ratios = self._damage.all_crack_ratios()
        except Exception:
            self._lbl_worst_crack.text = "Crack: --"
            self._lbl_worst_crack.style = {"color": 0xFF888888}
            return
        if not ratios:
            self._lbl_worst_crack.text = "Crack: --"
            self._lbl_worst_crack.style = {"color": 0xFF888888}
            return
        member_idx, ratio = max(ratios.items(), key=lambda item: item[1])
        ratio = max(0.0, float(ratio))
        self._lbl_worst_crack.text = (
            f"Crack: {ratio*100:.1f}% of critical  (M{member_idx})")
        color = 0xFF88FF88 if ratio < 0.5 else (
            0xFF00AAFF if ratio < 0.9 else 0xFF5555FF)
        self._lbl_worst_crack.style = {"color": color}
        if hasattr(self, "_sync_simulation_readouts"):
            self._sync_simulation_readouts()

    def _on_env_exposure_changed(self, model, item) -> None:
        idx = self._combo_item_index(
            model, item, getattr(self, "_conn_exposure_idx", 0))
        self._conn_exposure_idx = idx
        exposure = "outdoor" if idx == 1 else "indoor"
        rh_pct = self._conn_humidity_slider.model.get_value_as_float()
        rh = rh_pct / 100.0
        self._env.set_exposure(exposure, humidity_rh=rh)
        kd_pct = self._env_yield_knockdown * 100.0
        self._conn_env_lbl.text = (
            f"Env: {exposure} | RH: {rh_pct:.0f}% | yield -{kd_pct:.1f}%")

    def _on_humidity_changed(self, model) -> None:
        rh_pct = model.get_value_as_float()
        rh = rh_pct / 100.0
        self._conn_humidity_val_lbl.text = f"{rh_pct:.0f}%"
        idx = getattr(self, "_conn_exposure_idx", 0)
        exposure = "outdoor" if idx == 1 else "indoor"
        self._env.set_exposure(exposure, humidity_rh=rh)
        kd_pct = self._env_yield_knockdown * 100.0
        self._conn_env_lbl.text = (
            f"Env: {exposure} | RH: {rh_pct:.0f}% | yield -{kd_pct:.1f}%")

    def _current_gauge_maps(self) -> tuple[dict[int, int], dict[int, float]]:
        gauge_map = {
            ch: m
            for ch, m in enumerate(self._gauged_members[:STRAIN_GAUGE_COUNT])
        }
        span_map: dict[int, float] = {}
        topo = getattr(self, "_topo", None)
        if topo is not None:
            for ch, m_idx in gauge_map.items():
                try:
                    span_map[ch] = topo.member_span_fraction(m_idx)
                except Exception:
                    pass
        return gauge_map, span_map

    # Connection tab callback
    def _on_connect_websocket(self):
        self._conn_mode_idx = 0
        self._on_reconnect()

    def _on_reconnect(self):
        mode_idx = getattr(self, "_conn_mode_idx", 0)
        mode = ("websocket", "serial", "wifi")[mode_idx]
        traffic_mode = self._traffic_index_to_mode(
            getattr(self, "_conn_traffic_idx", 0))
        weight_multiplier = self._weight_index_to_multiplier(
            getattr(self, "_weight_mode_idx", 0))

        gauge_map, gauge_span_map = self._current_gauge_maps()

        config = ConnectionConfig(
            mode=mode,
            serial_port=self._conn_port_field.model.get_value_as_string(),
            udp_host=self._conn_host_field.model.get_value_as_string(),
            udp_port=self._conn_udp_port.model.get_value_as_int(),
            ws_url=self._conn_ws_url_field.model.get_value_as_string(),
            ws_basic_auth=self._conn_ws_auth_field.model.get_value_as_string(),
            traffic_mode=traffic_mode,
            weight_multiplier=weight_multiplier,
            gauge_channel_map=gauge_map,
            gauge_span_map=gauge_span_map,
        )
        self._conn_config = config
        self._sensor.set_traffic_mode(traffic_mode)
        self._sensor.reconfigure(config)

        mode_label = {"websocket": "WebSocket / ngrok",
                      "serial": "serial / COM", "wifi": "WiFi / UDP"}[mode]
        self._conn_status.text = f"Mode: {mode_label}"
        self._conn_status.style = {"color": 0xFF00BBFF}
        if hasattr(self, "_sim_status_lbl"):
            self._sim_status_lbl.text = "Simulation: not active"
            self._sim_status_lbl.style = {"color": 0xFF888888}

    def _on_apply_simulation(self):
        gauge_map, gauge_span_map = self._current_gauge_maps()
        traffic_mode = self._traffic_index_to_mode(
            getattr(self, "_conn_traffic_idx", 0))
        weight_multiplier = self._weight_index_to_multiplier(
            getattr(self, "_weight_mode_idx", 0))
        config = ConnectionConfig(
            mode="sim",
            traffic_mode=traffic_mode,
            weight_multiplier=weight_multiplier,
            gauge_channel_map=gauge_map,
            gauge_span_map=gauge_span_map,
        )
        self._conn_config = config
        self._sensor.reconfigure(config)
        if hasattr(self, "_conn_status"):
            self._conn_status.text = "Mode: physical sensor disconnected"
            self._conn_status.style = {"color": 0xFF888888}
        if hasattr(self, "_sim_status_lbl"):
            labels = {
                TrafficMode.UNIFORM.value: "Uniform random",
                TrafficMode.REALISTIC.value: "Realistic spectrum",
                TrafficMode.RUSH_HOUR.value: "Rush hour",
            }
            self._sim_status_lbl.text = (
                f"Simulation: running ({labels.get(traffic_mode, traffic_mode)}, "
                f"{weight_multiplier:g}x)")
            self._sim_status_lbl.style = {"color": 0xFF88FF88}

    # Button callbacks
    def _on_toggle_sim(self):
        if self._sensor.mode != "sim":
            self._on_apply_simulation()
            return

        if self._sensor.sim_paused:
            self._sensor.resume_sim()
            if hasattr(self, "_sim_status_lbl"):
                self._sim_status_lbl.text = "Simulation: running"
                self._sim_status_lbl.style = {"color": 0xFF88FF88}
        else:
            self._sensor.pause_sim()
            if hasattr(self, "_sim_status_lbl"):
                self._sim_status_lbl.text = "Simulation: paused"
                self._sim_status_lbl.style = {"color": 0xFF00AAFF}

    def _on_reset_damage(self):
        self._damage.reset()
        if self._coloring_mode == "damage" and hasattr(self, "_apply_damage_colors"):
            self._apply_damage_colors()
        elif self._last_fem:
            self._apply_colors(self._last_fem)
        for lbl in self._alert_labels:
            lbl.text = ""
        if hasattr(self, "_sim_alert_labels"):
            for lbl in self._sim_alert_labels:
                lbl.text = ""
        self._lbl_worst_damage.text = "--"
        self._lbl_passes.text = "Passes: 0"
        self._lbl_dyn_status.text = "Dynamic FEM: idle"
        self._update_crack_label()
        self._refresh_model_labels()
        self._last_dyn_result = None
        self._env_yield_knockdown = 0.0
        self._fast_vs_accurate_error = None
        self._sync_simulation_readouts()

    def _on_simulate_pass(self):
        rng = random.Random()
        raw_weight = rng.uniform(SIM_WEIGHT_MIN_KG, SIM_WEIGHT_MAX_KG)
        multiplier = self._weight_index_to_multiplier(
            getattr(self, "_weight_mode_idx", 0))
        speed  = rng.uniform(0.3, 1.2)
        axle   = rng.uniform(0.2, 0.8)
        vp = VehiclePass(
            weight_kg=raw_weight * multiplier * _daf_calibrated(speed),
            speed_ms=speed,
            axle_position_frac=axle,
            strain_readings={},
            metadata={
                "raw_weight_kg": raw_weight,
                "weight_multiplier": multiplier,
            },
        )
        self._record_damage(vp)

    def _on_run_analysis_now(self):
        self._dyn_mgr.request_immediate()
        # Enqueue a synthetic representative pass so the worker has something to analyse
        rng = random.Random()
        raw = rng.uniform(SIM_WEIGHT_MIN_KG, SIM_WEIGHT_MAX_KG)
        multiplier = self._weight_index_to_multiplier(
            getattr(self, "_weight_mode_idx", 0))
        spd = rng.uniform(0.4, 1.0)
        vp  = VehiclePass(
            weight_kg=raw * multiplier * _daf_calibrated(spd),
            speed_ms=spd,
            axle_position_frac=0.5,
            strain_readings={},
            metadata={
                "raw_weight_kg": raw,
                "weight_multiplier": multiplier,
            },
        )
        if self._last_fem:
            load_n = vp.weight_kg * GRAVITY
            forces = self._topo.solve_full(load_n, 0.5, 0.5)
        else:
            forces = {}
        self._dyn_mgr.enqueue_pass(vp, forces, MEMBER_AREA, forced=True)
        self._lbl_dyn_status.text = "Dynamic FEM: queued..."
        self._lbl_dyn_status.style = {"color": 0xFF00DDFF}



