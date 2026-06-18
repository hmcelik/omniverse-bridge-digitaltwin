# Standalone ESP32 WebSocket probe for the bridge simulation.
#
# This file is intentionally isolated from the extension runtime. Use it to verify
# that the public ESP32/ngrok endpoint sends JSON readings over /ws and accepts
# bridge feedback updates over /set-data before wiring the transport into
# SensorReader.

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import socket
import ssl
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import request
from urllib.parse import urlparse


DEFAULT_URL = "https://outward-confused-calm.ngrok-free.dev"


@dataclass
class Reading:
    raw: Dict[str, Any]

    @property
    def weight_kg(self) -> Optional[float]:
        return _first_number(self.raw, "weight", "weight_kg", "W", "load", "mass")

    @property
    def contact(self) -> Optional[int]:
        value = _first_number(self.raw, "contact", "P", "pressure", "in_transit")
        return None if value is None else int(value)

    @property
    def speed_ms(self) -> Optional[float]:
        return _first_number(self.raw, "speed", "speed_ms", "velocity")


def _first_number(data: Dict[str, Any], *names: str) -> Optional[float]:
    lower_to_key = {str(key).lower(): key for key in data}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is None:
            continue
        try:
            return float(data[key])
        except (TypeError, ValueError):
            return None
    return None


class MinimalWebSocket:

    def __init__(
        self,
        base_url: str,
        path: str = "/ws",
        timeout: float = 10.0,
        basic_auth: Optional[str] = None,
    ):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https", "ws", "wss"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        secure = parsed.scheme in ("https", "wss")
        self.host = parsed.hostname or ""
        self.port = parsed.port or (443 if secure else 80)
        self.path = path
        self.secure = secure
        self.timeout = timeout
        self.basic_auth = basic_auth
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        raw_sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.secure:
            context = ssl.create_default_context()
            self.sock = context.wrap_socket(raw_sock, server_hostname=self.host)
        else:
            self.sock = raw_sock
        self.sock.settimeout(self.timeout)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        auth_header = ""
        if self.basic_auth:
            token = base64.b64encode(self.basic_auth.encode("utf-8")).decode("ascii")
            auth_header = f"Authorization: Basic {token}\r\n"

        request_text = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Origin: null\r\n"
            "User-Agent: bridge-digitaltwin-probe/1.0\r\n"
            "Ngrok-Skip-Browser-Warning: true\r\n"
            f"{auth_header}"
            "\r\n"
        )
        self.sock.sendall(request_text.encode("ascii"))

        response = self._recv_until(b"\r\n\r\n")
        header_text = response.decode("iso-8859-1", errors="replace")
        if " 101 " not in header_text.split("\r\n", 1)[0]:
            raise RuntimeError(f"WebSocket upgrade failed:\n{header_text}")

        accept_expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        headers = {}
        for line in header_text.split("\r\n")[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        if headers.get("sec-websocket-accept") != accept_expected:
            raise RuntimeError(
                "WebSocket upgrade response had an unexpected accept key:\n"
                f"{header_text}"
            )

    def send_text(self, message: str) -> None:
        payload = message.encode("utf-8")
        header = bytearray([0x81])
        mask_bit = 0x80
        if len(payload) < 126:
            header.append(mask_bit | len(payload))
        elif len(payload) <= 0xFFFF:
            header.append(mask_bit | 126)
            header.extend(struct.pack("!H", len(payload)))
        else:
            header.append(mask_bit | 127)
            header.extend(struct.pack("!Q", len(payload)))

        mask = os.urandom(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
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
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))

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
            chunk = self._recv_exact(1)
            data.extend(chunk)
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


def post_bridge_data(base_url: str, max_load: float,
                     basic_auth: Optional[str]) -> int:
    url = base_url.rstrip("/") + "/set-data"
    payload = json.dumps({
        "maxLoad": max_load,
        "safeToPass": 1,
        "twin1": 0.0,
        "twin2": 0.0,
        "twin3": 0.0,
        "twin4": 0.0,
        "averageStrainTwin": 0.0,
    }).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Ngrok-Skip-Browser-Warning": "true",
            "User-Agent": "bridge-digitaltwin-probe/1.0",
        },
    )
    if basic_auth:
        token = base64.b64encode(basic_auth.encode("utf-8")).decode("ascii")
        req.add_header("Authorization", f"Basic {token}")
    with request.urlopen(req, timeout=10.0) as response:
        response.read()
        return int(response.status)


def run_probe(
    base_url: str,
    duration: float,
    max_load: Optional[float],
    basic_auth: Optional[str],
    random_max_load: bool,
    send_interval: float,
) -> int:
    ws = MinimalWebSocket(base_url, basic_auth=basic_auth)
    last_reading: Optional[Reading] = None
    message_count = 0
    next_send_at = time.monotonic()

    print(f"Connecting to {base_url.rstrip('/')}/ws")
    ws.connect()
    print("WebSocket connected. Sending getReadings...")
    ws.send_text("getReadings")

    if max_load is not None:
        status = post_bridge_data(base_url, max_load, basic_auth)
        print(f"POST /set-data maxLoad={max_load:g} -> HTTP {status}")

    deadline = time.monotonic() + duration
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if random_max_load and now >= next_send_at:
                value = random.randint(20, 120)
                status = post_bridge_data(base_url, value, basic_auth)
                print(f"POST /set-data random maxLoad={value} -> HTTP {status}")
                next_send_at = now + send_interval

            remaining = max(0.1, deadline - time.monotonic())
            if ws.sock:
                timeout = min(1.0 if random_max_load else 5.0, remaining)
                ws.sock.settimeout(timeout)
            try:
                text = ws.recv_text()
            except socket.timeout:
                if random_max_load:
                    continue
                raise
            if text is None:
                print("WebSocket closed by server.")
                break
            message_count += 1
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                print(f"[{message_count}] non-JSON: {text}")
                continue

            reading = Reading(payload)
            last_reading = reading
            print(f"[{message_count}] JSON keys={list(payload.keys())} data={payload}")
    except socket.timeout:
        print("No more messages before timeout.")
    finally:
        ws.close()

    if last_reading is None:
        print("Result: no JSON readings received.")
        return 2

    print(
        "Result: received JSON readings. "
        f"Last weight={last_reading.weight_kg}, "
        f"contact={last_reading.contact}, speed={last_reading.speed_ms}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="ESP32/ngrok base URL")
    parser.add_argument("--duration", type=float, default=15.0, help="seconds to listen")
    parser.add_argument("--max-load", type=float, default=None, help="optional maxLoad to POST back")
    parser.add_argument(
        "--random-max-load",
        action="store_true",
        help="POST a random maxLoad between 20 and 120 while listening",
    )
    parser.add_argument(
        "--send-interval",
        type=float,
        default=3.0,
        help="seconds between random maxLoad POSTs",
    )
    parser.add_argument(
        "--basic-auth",
        default=os.environ.get("ESP32_BASIC_AUTH"),
        help="optional Basic Auth as username:password, or set ESP32_BASIC_AUTH",
    )
    args = parser.parse_args()
    try:
        return run_probe(
            args.url,
            args.duration,
            args.max_load,
            args.basic_auth,
            args.random_max_load,
            args.send_interval,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "401 Unauthorized" in message and not args.basic_auth:
            print(
                "Result: ngrok requires Basic Auth. Re-run with "
                "--basic-auth username:password or set ESP32_BASIC_AUTH."
            )
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())


