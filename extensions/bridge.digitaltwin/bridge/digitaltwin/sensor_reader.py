# Read vehicle data for the bridge digital twin.
#
# SensorReader runs one background transport at a time:
# - sim: generate synthetic crossings for testing.
# - websocket: read ESP32/ngrok JSON messages.
# - serial: read ESP32 text frames from a COM port.

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import socket
import ssl
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from urllib import request
from urllib.parse import urlparse


# Protocol / physics constants
try:
    from .bridge_config import (
        REAL_BRIDGE_LENGTH, V_MAX_PROTOTYPE,
        SIM_WEIGHT_MIN_KG, SIM_WEIGHT_MAX_KG,
        STRAIN_GAUGE_COUNT,
    )
except ImportError:
    from bridge_config import (  # type: ignore[no-redef]
        REAL_BRIDGE_LENGTH, V_MAX_PROTOTYPE,
        SIM_WEIGHT_MIN_KG, SIM_WEIGHT_MAX_KG,
        STRAIN_GAUGE_COUNT,
    )

_SIM_BRIDGE_LEN_M = REAL_BRIDGE_LENGTH  # metres
_SIM_TICK_DT      = 0.1                 # seconds per animation step
_V_MAX_PROTOTYPE  = V_MAX_PROTOTYPE     # m/s


def _daf(speed_ms: float) -> float:
    ratio = min(speed_ms / _V_MAX_PROTOTYPE, 1.0)
    return 1.0 + 0.5 * ratio ** 2


# Calibrated DAF lookup table populated by the background dynamic analyser.
# Keys are speed bins (m/s rounded to 1 d.p.); values are measured DAF.
# Falls back to _daf() formula for speeds not yet in the table.
_calibrated_daf: dict = {}


def _daf_calibrated(speed_ms: float) -> float:
    if not _calibrated_daf:
        return _daf(speed_ms)
    # Find nearest calibrated speed bin (within 0.15 m/s)
    key = round(speed_ms, 1)
    if key in _calibrated_daf:
        return _calibrated_daf[key]
    nearest = min(_calibrated_daf.keys(), key=lambda k: abs(k - speed_ms))
    if abs(nearest - speed_ms) < 0.15:
        return _calibrated_daf[nearest]
    return _daf(speed_ms)


# Public data classes
class TrafficMode(str, Enum):
    UNIFORM = "uniform"
    REALISTIC = "realistic"
    RUSH_HOUR = "rush_hour"


@dataclass(frozen=True)
class _TrafficSample:
    raw_weight_kg: float
    speed_ms: float
    wait_s: float
    vehicle_class: str
    equivalent_full_scale_kg: float


@dataclass
class ConnectionConfig:
    mode: str = "websocket"         # "websocket" | "sim" | "serial" | "wifi"
    serial_port: str = "COM3"
    baud: int = 115_200
    udp_host: str = "0.0.0.0"      # listen on all interfaces
    udp_port: int = 5555
    ws_url: str = "https://outward-confused-calm.ngrok-free.dev"
    ws_basic_auth: str = field(
        default_factory=lambda: os.environ.get(
            "ESP32_BASIC_AUTH", "melih:digitaltwin"))
    traffic_mode: str = TrafficMode.UNIFORM.value
    traffic_intensity_vpm: float = 12.0
    weight_multiplier: float = 1.0
    contact_length: float = 0.05   # metres -- pressure-sensor contact patch size
    pressure_contact_threshold: float = 0.0
    weight_contact_threshold: float = 0.0
    pressure_to_weight_gap_m: float = 0.14
    pressure_to_bridge_m: float = 0.30
    min_vehicle_speed_ms: float = 0.05
    # Maps strain-gauge channel index -> global member index in the truss.
    # Channel 0 = S0 in ESP32 packet, channel 1 = S1, etc.
    # Set automatically by extension after structural capacity solve.
    gauge_channel_map: Dict[int, int] = field(
        default_factory=lambda: {ch: ch for ch in range(STRAIN_GAUGE_COUNT)}
    )
    # Channel -> member midpoint fraction along the bridge span, for simulated
    # strain influence lines.
    gauge_span_map: Dict[int, float] = field(default_factory=dict)


@dataclass
class FeedbackCommand:
    max_load_kg: float
    safe_speed_ms: Optional[float] = None
    advisory: str = "ok"
    alert_level: str = "INFO"
    reason: str = ""
    timestamp_utc: str = ""
    strain_values: List[float] = field(default_factory=list)
    strain_value: Optional[float] = None
    safe_to_pass: bool = True
    # Legacy name accepted so older callers/tests do not silently lose values.
    stress_values: List[float] = field(default_factory=list)

    def payload(self) -> dict:
        strains = list((self.strain_values or self.stress_values)[:4])
        while len(strains) < 4:
            strains.append(0.0)
        average_strain = (
            float(self.strain_value)
            if self.strain_value is not None
            else sum(strains) / len(strains)
        )
        return {
            "maxLoad": self.max_load_kg,
            "safeToPass": 1 if self.safe_to_pass else 0,
            "twin1": strains[0],
            "twin2": strains[1],
            "twin3": strains[2],
            "twin4": strains[3],
            "averageStrainTwin": average_strain,
        }


@dataclass
class FeedbackStatus:
    state: str = "idle"       # idle | pending | sent | failed
    attempts: int = 0
    last_error: str = ""
    last_payload: dict = field(default_factory=dict)
    last_sent_utc: str = ""


@dataclass
class SensorTick:
    weight_kg: float
    position_frac: float              # 0.0 = left abutment, 1.0 = right
    in_transit: bool
    strain_readings: Dict[int, float] # member_index -> microstrain
    speed_ms: Optional[float] = None
    pressure_raw: Optional[float] = None
    timestamp_ms: Optional[float] = None
    crossing_id: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class VehiclePass:
    weight_kg: float                   # peak weight x DAF
    speed_ms: float
    axle_position_frac: float          # 0..1 at peak-weight moment
    strain_readings: Dict[int, float]  # member_index -> peak microstrain
    pressure_raw: Optional[float] = None
    timestamp_ms: Optional[float] = None
    crossing_id: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


# Internal hardware-state tracker (shared between serial and WiFi parsers)
@dataclass
class _HWState:
    contact_start: Optional[float] = None
    bridge_start: Optional[float] = None
    bridge_end: Optional[float] = None
    peak_weight: float = 0.0
    peak_raw_weight: float = 0.0
    peak_pressure: float = 0.0
    peak_strains: Dict[int, float] = field(default_factory=dict)
    last_speed_ms: Optional[float] = None
    last_timestamp_ms: Optional[float] = None
    crossing_id: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    prev_contact: int = 0
    prev_weight_contact: int = 0


# Main reader class
class _MinimalWebSocket:

    def __init__(self, base_url: str, path: str = "/ws",
                 basic_auth: str = "", timeout: float = 10.0) -> None:
        parsed = urlparse(base_url)
        secure = parsed.scheme in ("https", "wss")
        self.host = parsed.hostname or ""
        self.port = parsed.port or (443 if secure else 80)
        self.path = path
        self.secure = secure
        self.basic_auth = basic_auth
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        raw_sock = socket.create_connection((self.host, self.port),
                                            timeout=self.timeout)
        if self.secure:
            ctx = ssl.create_default_context()
            self.sock = ctx.wrap_socket(raw_sock, server_hostname=self.host)
        else:
            self.sock = raw_sock
        self.sock.settimeout(self.timeout)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        auth_header = ""
        if self.basic_auth:
            token = base64.b64encode(
                self.basic_auth.encode("utf-8")).decode("ascii")
            auth_header = f"Authorization: Basic {token}\r\n"

        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Origin: null\r\n"
            "User-Agent: bridge-digitaltwin/1.0\r\n"
            "Ngrok-Skip-Browser-Warning: true\r\n"
            f"{auth_header}"
            "\r\n"
        )
        self.sock.sendall(req.encode("ascii"))
        response = self._recv_until(b"\r\n\r\n").decode(
            "iso-8859-1", errors="replace")
        if " 101 " not in response.split("\r\n", 1)[0]:
            raise RuntimeError(f"WebSocket upgrade failed:\n{response}")

        expected = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        ).digest()).decode("ascii")
        headers = {}
        for line in response.split("\r\n")[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        if headers.get("sec-websocket-accept") != expected:
            raise RuntimeError("WebSocket upgrade accept key mismatch.")

    def send_text(self, message: str) -> None:
        payload = message.encode("utf-8")
        header = bytearray([0x81])
        if len(payload) < 126:
            header.append(0x80 | len(payload))
        elif len(payload) <= 0xFFFF:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", len(payload)))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", len(payload)))

        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._send(bytes(header) + mask + masked)

    def recv_text(self) -> Optional[str]:
        while True:
            first = self._recv_exact(2)
            opcode = first[0] & 0x0F
            masked = bool(first[1] & 0x80)
            length = first[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length) if length else b""
            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            if opcode == 0x1:
                return payload.decode("utf-8", errors="replace")
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._send(bytes([0x8A, len(payload)]) + payload)

    def close(self) -> None:
        try:
            if self.sock:
                self._send(b"\x88\x00")
                self.sock.close()
        finally:
            self.sock = None

    def _send(self, data: bytes) -> None:
        if not self.sock:
            raise RuntimeError("WebSocket is not connected.")
        self.sock.sendall(data)

    def _recv_until(self, marker: bytes) -> bytes:
        data = bytearray()
        while marker not in data:
            data.extend(self._recv_exact(1))
        return bytes(data)

    def _recv_exact(self, size: int) -> bytes:
        if not self.sock:
            raise RuntimeError("WebSocket is not connected.")
        data = bytearray()
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Socket closed while receiving data.")
            data.extend(chunk)
        return bytes(data)


class SensorReader:

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        # Legacy kwargs kept for backward-compat with existing on_startup call:
        port: Optional[str] = None,
        monitored_members: Optional[List[int]] = None,
    ) -> None:
        if config is None:
            config = ConnectionConfig()
            # honour legacy 'port' kwarg
            if port is not None:
                config.mode = "serial"
                config.serial_port = port

        self.config = config

        # If caller supplied legacy monitored_members, build a default gauge map.
        if monitored_members is not None and not config.gauge_channel_map:
            config.gauge_channel_map = {
                i: m
                for i, m in enumerate(monitored_members[:STRAIN_GAUGE_COUNT])
            }

        self._lock          = threading.Lock()
        self._latest:        Optional[VehiclePass] = None
        self._current_tick:  Optional[SensorTick]  = None
        self._is_live        = False
        self._stop_event     = threading.Event()
        self._sim_pause_event = threading.Event()
        self._sim_pause_event.set()   # start unpaused
        self._thread:        Optional[threading.Thread] = None
        self._pending_feedback: Optional[FeedbackCommand] = None
        self._last_sent_feedback_key: Optional[str] = None
        self._feedback_status = FeedbackStatus()
        self._feedback_next_try_monotonic = 0.0


    @property
    def latest_pass(self) -> Optional[VehiclePass]:
        with self._lock:
            return self._latest

    @property
    def current_tick(self) -> Optional[SensorTick]:
        with self._lock:
            return self._current_tick

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def mode(self) -> str:
        return self.config.mode

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._sim_pause_event.set()   # only used when mode == "sim"
        self._is_live = False

        mode = self.config.mode
        if mode == "websocket":
            target = self._websocket_loop
        elif mode == "serial":
            try:
                import serial
                self._serial = serial.Serial(
                    self.config.serial_port, self.config.baud, timeout=1.0)
                self._is_live = True
                target = self._serial_loop
                print(f"[SensorReader] Serial connected: {self.config.serial_port}")
            except Exception as exc:
                print(f"[SensorReader] Serial failed ({exc}); sensor idle.")
                target = self._idle_loop
        elif mode == "wifi":
            self._is_live = True   # assumed live; _wifi_loop clears if bind fails
            target = self._wifi_loop
        elif mode == "sim":
            target = self._sim_loop
        else:
            print(f"[SensorReader] Unknown mode '{mode}'; sensor idle.")
            target = self._idle_loop

        self._thread = threading.Thread(target=target, daemon=True,
                                        name=f"SensorReader-{mode}")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._sim_pause_event.set()   # wake sim thread if blocked on pause
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None

    def pause_sim(self) -> None:
        self._sim_pause_event.clear()

    def resume_sim(self) -> None:
        self._sim_pause_event.set()

    @property
    def sim_paused(self) -> bool:
        return not self._sim_pause_event.is_set()

    def reconfigure(self, config: ConnectionConfig) -> None:
        self.stop()
        self.config = config
        self._is_live = False
        self.start()

    def set_max_load(self, max_load_kg: float) -> None:
        self.set_control_feedback(FeedbackCommand(max_load_kg=float(max_load_kg)))

    def set_control_feedback(self, command: FeedbackCommand) -> None:
        with self._lock:
            self._pending_feedback = command
            self._feedback_status.state = "pending"
            self._feedback_status.attempts = 0
            self._feedback_status.last_error = ""
            self._feedback_next_try_monotonic = 0.0

    @property
    def feedback_status(self) -> FeedbackStatus:
        with self._lock:
            return FeedbackStatus(
                state=self._feedback_status.state,
                attempts=self._feedback_status.attempts,
                last_error=self._feedback_status.last_error,
                last_payload=dict(self._feedback_status.last_payload),
                last_sent_utc=self._feedback_status.last_sent_utc,
            )

    def set_traffic_mode(self, mode: TrafficMode | str) -> None:
        self.config.traffic_mode = self._coerce_traffic_mode(mode).value

    def set_traffic_intensity(self, vpm: float) -> None:
        self.config.traffic_intensity_vpm = max(0.1, float(vpm))

    def set_weight_multiplier(self, multiplier: float) -> None:
        allowed = (1.0, 10.0, 25.0, 50.0, 100.0)
        self.config.weight_multiplier = min(
            allowed, key=lambda value: abs(value - float(multiplier)))

    def set_gauge_span_map(self, span_map: Dict[int, float]) -> None:
        self.config.gauge_span_map = {
            int(ch): max(0.0, min(1.0, float(frac)))
            for ch, frac in span_map.items()
        }

    def update_daf_calibration(self, speed_ms: float, measured_daf: float) -> None:
        key = round(speed_ms, 1)
        # Exponential moving average: 80% old, 20% new -- avoids single-outlier jumps
        if key in _calibrated_daf:
            _calibrated_daf[key] = 0.8 * _calibrated_daf[key] + 0.2 * measured_daf
        else:
            _calibrated_daf[key] = measured_daf
        print(f"[SensorReader] DAF calibrated: {speed_ms:.2f} m/s -> "
              f"DAF={_calibrated_daf[key]:.4f} "
              f"(formula={_daf(speed_ms):.4f})")


    def _parse_fields(self, raw: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for tok in raw.split(","):
            if ":" in tok:
                k, v = tok.split(":", 1)
                out[k.strip()] = v.strip()
        return out

    def _field_float(self, fields: Dict[str, str], *names: str) -> Optional[float]:
        lower = {k.lower(): v for k, v in fields.items()}
        for name in names:
            value = lower.get(name.lower())
            if value is None or value == "":
                continue
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _process_frame(self, fields: Dict[str, str], state: _HWState) -> None:
        try:
            raw_weight = float(fields.get("W", 0))
            contact = int(fields.get("P", 0))
        except ValueError:
            return
        weight_multiplier = max(1.0, float(self.config.weight_multiplier))
        weight = raw_weight * weight_multiplier

        speed_ms = self._field_float(fields, "speed", "V", "speed_ms")
        pressure = self._field_float(fields, "pressure", "pressure_raw", "raw_pressure")
        timestamp_ms = self._field_float(fields, "t", "timestamp", "timestamp_ms")
        crossing_id = fields.get("id") or fields.get("crossing_id")
        if speed_ms is not None and speed_ms > self.config.min_vehicle_speed_ms:
            state.last_speed_ms = speed_ms
        if timestamp_ms is not None:
            state.last_timestamp_ms = timestamp_ms
        if crossing_id:
            state.crossing_id = str(crossing_id)
        if pressure is not None:
            state.peak_pressure = max(state.peak_pressure, pressure)
        weight_contact = (
            1 if raw_weight > self.config.weight_contact_threshold else 0)
        state.metadata = {
            "raw_fields": dict(fields),
            "raw_weight_kg": raw_weight,
            "weight_multiplier": weight_multiplier,
            "pressure_raw": pressure,
            "speed_ms": speed_ms,
            "inferred_speed_ms": state.last_speed_ms,
            "pressure_to_weight_gap_m": self.config.pressure_to_weight_gap_m,
            "pressure_to_bridge_m": self.config.pressure_to_bridge_m,
            "bridge_start": state.bridge_start,
            "bridge_end": state.bridge_end,
        }

        # Read configured gauge channels
        strains: Dict[int, float] = {}
        for ch, m_idx in self.config.gauge_channel_map.items():
            try:
                strains[m_idx] = float(fields.get(f"S{ch}", 0.0))
            except ValueError:
                strains[m_idx] = 0.0
        now = time.monotonic()

        def schedule_bridge_start() -> None:
            if state.contact_start is None or state.last_speed_ms is None:
                return
            speed = max(state.last_speed_ms, self.config.min_vehicle_speed_ms)
            state.bridge_start = (
                state.contact_start + self.config.pressure_to_bridge_m / speed
            )
            state.bridge_end = state.bridge_start + _SIM_BRIDGE_LEN_M / speed
            state.metadata["bridge_start"] = state.bridge_start
            state.metadata["bridge_end"] = state.bridge_end

        if contact == 1 and state.prev_contact == 0 and state.contact_start is None:
            # Rising pressure edge: the vehicle reached the first sensor. Keep
            # later pulses from the same vehicle grouped in this active window.
            state.contact_start = now
            state.bridge_start = None
            state.bridge_end = None
            state.peak_weight   = weight
            state.peak_raw_weight = raw_weight
            state.peak_pressure = max(pressure or 0.0, 0.0)
            state.peak_strains  = {m: abs(v) for m, v in strains.items()}
            if state.last_speed_ms is not None:
                schedule_bridge_start()
            with self._lock:
                self._current_tick = None

        if (weight_contact == 1 and state.prev_weight_contact == 0
                and state.contact_start is None
                and state.last_speed_ms is not None):
            # Fallback: if the tiny pressure pulse was missed but the weight
            # sensor fires, reconstruct the pressure time from the known gap.
            speed = max(state.last_speed_ms, self.config.min_vehicle_speed_ms)
            state.contact_start = now - self.config.pressure_to_weight_gap_m / speed
            state.bridge_start = None
            state.bridge_end = None
            state.peak_weight = weight
            state.peak_raw_weight = raw_weight
            state.peak_pressure = max(pressure or 0.0, 0.0)
            state.peak_strains = {m: abs(v) for m, v in strains.items()}
            schedule_bridge_start()

        if (weight_contact == 1 and state.prev_weight_contact == 0
                and state.contact_start is not None):
            # The weight sensor is 14 cm after the pressure sensor, so its
            # rising edge can infer speed when the ESP is not already sending it.
            if speed_ms is None and state.last_speed_ms is None:
                sensor_dt = max(now - state.contact_start, 1e-6)
                inferred_speed = self.config.pressure_to_weight_gap_m / sensor_dt
                state.last_speed_ms = max(
                    inferred_speed, self.config.min_vehicle_speed_ms)
            if state.bridge_start is None:
                schedule_bridge_start()
            state.metadata["inferred_speed_ms"] = state.last_speed_ms

        if state.contact_start is not None:
            state.peak_weight = max(state.peak_weight, weight)
            state.peak_raw_weight = max(state.peak_raw_weight, raw_weight)
            for m, v in strains.items():
                state.peak_strains[m] = max(state.peak_strains.get(m, 0.0), abs(v))
            if state.bridge_start is None and state.last_speed_ms is not None:
                schedule_bridge_start()

        live_pos: Optional[float] = None
        if state.bridge_start is not None and state.last_speed_ms is not None:
            elapsed_on_bridge = now - state.bridge_start
            if elapsed_on_bridge >= 0.0:
                live_pos = min(
                    1.0,
                    elapsed_on_bridge * state.last_speed_ms
                    / max(_SIM_BRIDGE_LEN_M, 1e-9),
                )
                with self._lock:
                    metadata = dict(state.metadata)
                    metadata["raw_weight_kg"] = state.peak_raw_weight
                    self._current_tick = SensorTick(
                        weight_kg=max(weight, state.peak_weight),
                        position_frac=live_pos,
                        in_transit=True, strain_readings=dict(strains),
                        speed_ms=state.last_speed_ms,
                        pressure_raw=pressure,
                        timestamp_ms=timestamp_ms or state.last_timestamp_ms,
                        crossing_id=state.crossing_id,
                        metadata=metadata)
            else:
                with self._lock:
                    approach_weight = max(weight, state.peak_weight)
                    if approach_weight > 0.0:
                        metadata = dict(state.metadata)
                        metadata["raw_weight_kg"] = state.peak_raw_weight
                        metadata["phase"] = "approach"
                        self._current_tick = SensorTick(
                            weight_kg=approach_weight,
                            position_frac=0.0,
                            in_transit=True,
                            strain_readings=dict(strains),
                            speed_ms=state.last_speed_ms,
                            pressure_raw=pressure,
                            timestamp_ms=timestamp_ms or state.last_timestamp_ms,
                            crossing_id=state.crossing_id,
                            metadata=metadata)
                    else:
                        self._current_tick = None

        self._print_live_sensor_data(weight, contact, live_pos, strains)

        bridge_finished = (
            live_pos is not None
            and live_pos >= 1.0
            and state.contact_start is not None
        )
        contact_finished_without_speed = (
            contact == 0
            and state.prev_contact == 1
            and state.contact_start is not None
            and state.last_speed_ms is None
        )
        if bridge_finished or contact_finished_without_speed:
            duration = max(now - state.contact_start, 1e-6)
            speed    = speed_ms or state.last_speed_ms or (
                self.config.contact_length / duration)
            daf      = _daf(speed)
            vp = VehiclePass(
                weight_kg=state.peak_weight * daf,
                speed_ms=speed,
                axle_position_frac=0.5,
                strain_readings=dict(state.peak_strains),
                pressure_raw=state.peak_pressure or pressure,
                timestamp_ms=timestamp_ms or state.last_timestamp_ms,
                crossing_id=state.crossing_id,
                metadata={
                    **dict(state.metadata),
                    "raw_weight_kg": state.peak_raw_weight,
                },
            )
            with self._lock:
                self._current_tick = SensorTick(
                    weight_kg=0.0, position_frac=1.0,
                    in_transit=False, strain_readings={},
                    speed_ms=speed,
                    pressure_raw=pressure,
                    timestamp_ms=timestamp_ms or state.last_timestamp_ms,
                    crossing_id=state.crossing_id,
                    metadata=dict(state.metadata))
                self._latest = vp
            state.contact_start = None
            state.bridge_start = None
            state.bridge_end = None
            state.peak_raw_weight = 0.0
            state.peak_pressure = 0.0
            state.last_speed_ms = None
            state.crossing_id = None
            state.metadata = {}

        state.prev_contact = contact
        state.prev_weight_contact = weight_contact


    def _print_live_sensor_data(
        self,
        weight_kg: float,
        contact: int,
        position_frac: Optional[float],
        strains: Dict[int, float],
    ) -> None:
        pos = "--" if position_frac is None else f"{position_frac:.2f}"
        strain_text = ", ".join(
            f"M{m_idx}:{value:.1f}ue" for m_idx, value in sorted(strains.items())
        ) or "none"
        print(
            f"[SensorReader] live mode={self.config.mode} "
            f"W={weight_kg:.3f}kg P={contact} pos={pos} strains={strain_text}"
        )

    def _idle_loop(self) -> None:
        with self._lock:
            self._current_tick = None
        while not self._stop_event.wait(timeout=1.0):
            self._send_pending_feedback()

    def _websocket_loop(self) -> None:
        state = _HWState()
        while not self._stop_event.is_set():
            ws: Optional[_MinimalWebSocket] = None
            try:
                ws = _MinimalWebSocket(
                    self.config.ws_url,
                    basic_auth=self.config.ws_basic_auth,
                    timeout=10.0,
                )
                ws.connect()
                ws.send_text("getReadings")
                self._is_live = True
                print(f"[SensorReader] WebSocket connected: {self.config.ws_url}")

                while not self._stop_event.is_set():
                    self._send_pending_max_load()
                    if ws.sock:
                        ws.sock.settimeout(1.0)
                    try:
                        text = ws.recv_text()
                    except socket.timeout:
                        continue
                    if text is None:
                        break
                    fields = self._parse_websocket_json(text)
                    if fields:
                        self._process_frame(fields, state)
            except Exception as exc:
                self._is_live = False
                print(f"[SensorReader] WebSocket error: {exc}")
                self._stop_event.wait(timeout=2.0)
            finally:
                self._is_live = False
                if ws:
                    ws.close()
        print("[SensorReader] WebSocket closed.")

    def _parse_websocket_json(self, raw: str) -> Dict[str, str]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}

        def pick(*names: str, default: object = 0) -> object:
            lower = {str(k).lower(): k for k in data}
            for name in names:
                key = lower.get(name.lower())
                if key is not None:
                    return data[key]
            return default

        pressure = pick("pressure", "pressure_raw", "raw_pressure", default="")
        contact = pick("P", "crossing", "contact", "in_transit", default=None)
        if contact is None:
            try:
                contact = (
                    1 if float(pressure) > self.config.pressure_contact_threshold
                    else 0
                )
            except (TypeError, ValueError):
                contact = 0
        if isinstance(contact, bool):
            contact = 1 if contact else 0
        fields = {
            "W": str(pick("W", "weight", "weight_kg", "load", "mass")),
            "P": str(contact),
            "pressure": str(pressure),
            "speed": str(pick("speed", "speed_ms", "V", default="")),
            "t": str(pick("t", "timestamp", "timestamp_ms", default="")),
            "id": str(pick("id", "crossing_id", default="")),
        }
        lower_keys = {str(k).lower() for k in data}
        zero_based_strains = any(
            key in lower_keys for key in ("s0", "strain0")
        )
        for ch in self.config.gauge_channel_map:
            strain_name = f"strain{ch if zero_based_strains else ch + 1}"
            fields[f"S{ch}"] = str(pick(f"S{ch}", strain_name, default=0))
        return fields

    def _send_pending_max_load(self) -> None:
        self._send_pending_feedback()

    def _send_pending_feedback(self) -> None:
        with self._lock:
            command = self._pending_feedback
            if command is None:
                return
            payload_dict = command.payload()
            payload_key = json.dumps(payload_dict, sort_keys=True)
            if payload_key == self._last_sent_feedback_key:
                return
            if time.monotonic() < self._feedback_next_try_monotonic:
                return
            self._feedback_status.attempts += 1
            self._feedback_status.last_payload = dict(payload_dict)

        url = self.config.ws_url.rstrip("/") + "/set-data"
        payload = json.dumps(payload_dict).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Ngrok-Skip-Browser-Warning": "true",
                "User-Agent": "bridge-digitaltwin/1.0",
            },
        )
        if self.config.ws_basic_auth:
            token = base64.b64encode(
                self.config.ws_basic_auth.encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {token}")

        try:
            with request.urlopen(req, timeout=5.0) as response:
                response.read()
            with self._lock:
                self._last_sent_feedback_key = payload_key
                if self._pending_feedback is command:
                    self._pending_feedback = None
                self._feedback_status.state = "sent"
                self._feedback_status.last_error = ""
                self._feedback_status.last_sent_utc = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(f"[SensorReader] Sent feedback: {payload_dict}")
        except Exception as exc:
            with self._lock:
                delay = min(30.0, 2.0 ** min(self._feedback_status.attempts, 5))
                self._feedback_next_try_monotonic = time.monotonic() + delay
                self._feedback_status.state = "failed"
                self._feedback_status.last_error = str(exc)
            print(f"[SensorReader] Could not send feedback: {exc}")

    def _serial_loop(self) -> None:
        state = _HWState()
        while not self._stop_event.is_set():
            try:
                raw = self._serial.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                time.sleep(0.01)
                continue
            if raw:
                self._process_frame(self._parse_fields(raw), state)

    def _wifi_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.config.udp_host, self.config.udp_port))
            sock.settimeout(1.0)
            print(f"[SensorReader] WiFi UDP listening on "
                  f"{self.config.udp_host}:{self.config.udp_port}")
        except OSError as exc:
            print(f"[SensorReader] UDP bind failed: {exc}")
            self._is_live = False
            sock.close()
            return

        state = _HWState()
        try:
            while not self._stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(256)
                    raw = data.decode("ascii", errors="ignore").strip()
                    self._process_frame(self._parse_fields(raw), state)
                except socket.timeout:
                    continue
                except Exception as exc:
                    print(f"[SensorReader] WiFi recv error: {exc}")
        finally:
            sock.close()
            print("[SensorReader] WiFi socket closed.")


    def _coerce_traffic_mode(self, mode: TrafficMode | str) -> TrafficMode:
        if isinstance(mode, TrafficMode):
            return mode
        try:
            return TrafficMode(str(mode).strip().lower())
        except ValueError:
            print(f"[SensorReader] Unknown traffic mode '{mode}'; using uniform.")
            return TrafficMode.UNIFORM

    def _normal_clamped(
        self,
        rng: random.Random,
        mean: float,
        std: float,
        low: float,
        high: float,
    ) -> float:
        return max(low, min(high, rng.gauss(mean, std)))

    def _scale_full_size_weight(self, full_scale_kg: float) -> float:
        full_min = 150.0
        full_max = 600.0
        frac = (max(full_min, min(full_max, full_scale_kg)) - full_min) / (
            full_max - full_min)
        return SIM_WEIGHT_MIN_KG + frac * (SIM_WEIGHT_MAX_KG - SIM_WEIGHT_MIN_KG)

    def _sample_traffic(self, rng: random.Random) -> _TrafficSample:
        mode = self._coerce_traffic_mode(self.config.traffic_mode)

        if mode is TrafficMode.UNIFORM:
            weight = rng.uniform(SIM_WEIGHT_MIN_KG, SIM_WEIGHT_MAX_KG)
            return _TrafficSample(
                raw_weight_kg=weight,
                speed_ms=rng.uniform(0.3, _V_MAX_PROTOTYPE),
                wait_s=rng.uniform(2.0, 5.0),
                vehicle_class="uniform",
                equivalent_full_scale_kg=0.0,
            )

        if mode is TrafficMode.RUSH_HOUR:
            heavy_fraction = 0.50
            speed_center = 0.5
            intensity_vpm = max(0.1, self.config.traffic_intensity_vpm * 2.0)
        else:
            heavy_fraction = 0.30
            speed_center = 0.7
            intensity_vpm = max(0.1, self.config.traffic_intensity_vpm)

        if rng.random() < heavy_fraction:
            full_scale_weight = self._normal_clamped(
                rng, 450.0, 80.0, 250.0, 600.0)
            vehicle_class = "heavy"
        else:
            full_scale_weight = self._normal_clamped(
                rng, 300.0, 50.0, 150.0, 450.0)
            vehicle_class = "light"

        weight = self._scale_full_size_weight(full_scale_weight)
        speed = max(0.1, min(_V_MAX_PROTOTYPE,
                             rng.lognormvariate(math.log(speed_center), 0.28)))
        wait = rng.expovariate(intensity_vpm / 60.0)
        return _TrafficSample(
            raw_weight_kg=weight,
            speed_ms=speed,
            wait_s=wait,
            vehicle_class=vehicle_class,
            equivalent_full_scale_kg=full_scale_weight,
        )

    def _sim_loop(self) -> None:
        rng = random.Random()

        while not self._stop_event.is_set():
            self._send_pending_feedback()

            # Block here while paused; stop() sets the event to unblock us.
            if not self._sim_pause_event.is_set():
                with self._lock:
                    self._current_tick = None
                self._sim_pause_event.wait()
                continue   # re-check _stop_event

            sample = self._sample_traffic(rng)
            wait_until = time.monotonic() + sample.wait_s
            while not self._stop_event.is_set():
                remaining = wait_until - time.monotonic()
                if remaining <= 0.0:
                    break
                self._send_pending_feedback()
                self._stop_event.wait(timeout=min(0.5, remaining))
            if self._stop_event.is_set():
                break

            raw_weight = sample.raw_weight_kg
            weight_multiplier = max(1.0, float(self.config.weight_multiplier))
            effective_raw_weight = raw_weight * weight_multiplier
            speed      = sample.speed_ms
            daf        = _daf_calibrated(speed)   # uses calibrated table if available
            weight     = effective_raw_weight * daf

            n_steps     = max(3, int((_SIM_BRIDGE_LEN_M / speed) / _SIM_TICK_DT))
            peak_strains: Dict[int, float] = {}

            for step in range(n_steps + 1):
                if self._stop_event.is_set():
                    break
                pos = step / n_steps

                # Influence-line strain per gauged member
                strain_readings: Dict[int, float] = {}
                for ch, m_idx in self.config.gauge_channel_map.items():
                    # Use the actual gauged member midpoint when the bridge
                    # topology has provided it; fall back to channel spacing.
                    n_mem = max(1, len(self.config.gauge_channel_map))
                    span_frac = self.config.gauge_span_map.get(
                        ch, ch / max(n_mem - 1, 1))
                    dist      = abs(span_frac - pos)
                    envelope  = math.cos(math.pi * min(dist, 0.5) * 2.0) ** 2
                    base      = effective_raw_weight * 50.0
                    noise     = rng.gauss(0.0, base * 0.03)
                    val       = max(0.0, base * envelope + noise)
                    strain_readings[m_idx] = val
                    peak_strains[m_idx]    = max(peak_strains.get(m_idx, 0.0), val)

                still_on = True
                with self._lock:
                    self._current_tick = SensorTick(
                        weight_kg=weight, position_frac=pos,
                        in_transit=still_on, strain_readings=strain_readings,
                        speed_ms=speed,
                        metadata={
                            "raw_weight_kg": raw_weight,
                            "weight_multiplier": weight_multiplier,
                        })
                self._print_live_sensor_data(
                    weight, 1, pos, strain_readings)

                if step < n_steps:
                    self._stop_event.wait(timeout=_SIM_TICK_DT)

            with self._lock:
                self._latest = VehiclePass(
                    weight_kg=weight, speed_ms=speed,
                    axle_position_frac=0.5, strain_readings=peak_strains,
                    metadata={
                        "raw_weight_kg": raw_weight,
                        "weight_multiplier": weight_multiplier,
                        "vehicle_class": sample.vehicle_class,
                        "equivalent_full_scale_kg": sample.equivalent_full_scale_kg,
                    })


def run_traffic_spectrum_self_test() -> None:
    reader = SensorReader(ConnectionConfig(mode="sim"))
    rng = random.Random(12345)

    uniform = [reader._sample_traffic(rng) for _ in range(200)]
    assert all(SIM_WEIGHT_MIN_KG <= s.raw_weight_kg <= SIM_WEIGHT_MAX_KG
               for s in uniform)
    assert all(0.3 <= s.speed_ms <= _V_MAX_PROTOTYPE for s in uniform)
    assert all(2.0 <= s.wait_s <= 5.0 for s in uniform)

    reader.set_traffic_mode(TrafficMode.REALISTIC)
    reader.set_traffic_intensity(12.0)
    realistic = [reader._sample_traffic(rng) for _ in range(2000)]
    heavy_frac = sum(s.vehicle_class == "heavy" for s in realistic) / len(realistic)
    mean_full_scale = sum(s.equivalent_full_scale_kg for s in realistic) / len(realistic)
    mean_model_weight = sum(s.raw_weight_kg for s in realistic) / len(realistic)
    assert 0.25 <= heavy_frac <= 0.35
    assert 330.0 <= mean_full_scale <= 370.0
    assert 2.0 <= mean_model_weight <= 2.6
    assert all(150.0 <= s.equivalent_full_scale_kg <= 600.0 for s in realistic)
    assert all(SIM_WEIGHT_MIN_KG <= s.raw_weight_kg <= SIM_WEIGHT_MAX_KG
               for s in realistic)
    assert all(0.1 <= s.speed_ms <= _V_MAX_PROTOTYPE for s in realistic)

    reader.set_traffic_mode("rush_hour")
    reader.set_traffic_intensity(12.0)
    rush = [reader._sample_traffic(rng) for _ in range(2000)]
    rush_heavy_frac = sum(s.vehicle_class == "heavy" for s in rush) / len(rush)
    rush_mean_wait = sum(s.wait_s for s in rush) / len(rush)
    realistic_mean_wait = sum(s.wait_s for s in realistic) / len(realistic)
    assert 0.45 <= rush_heavy_frac <= 0.55
    assert rush_mean_wait < realistic_mean_wait * 0.65


def run_self_test() -> None:
    run_traffic_spectrum_self_test()


if __name__ == "__main__":
    run_self_test()
    print("sensor_reader self-test passed")



