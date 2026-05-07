"""
Low-level wrapper around ~/pd3r3.py (PD3R3 Modbus driver).
Adds env-based config, thread safety, and auto-reconnect.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

sys.path.insert(0, os.path.expanduser("~"))
from pd3r3 import PD3R3, ModbusError

IO_PORT = os.getenv("IO_PORT",  "/dev/ttyACM0")
IO_ADDR = int(os.getenv("IO_ADDR", "1"))
IO_BAUD = int(os.getenv("IO_BAUD", "9600"))


class IOModule:
    """Thread-safe wrapper around PD3R3 with auto-reconnect."""

    def __init__(self):
        self._lock = threading.Lock()
        self._dev: Optional[PD3R3] = None
        self._ok    = False
        self._error = ""

    # ── connection ────────────────────────────────────────────────────
    def connect(self) -> bool:
        with self._lock:
            self._close_locked()
            try:
                self._dev = PD3R3(IO_PORT, IO_ADDR, IO_BAUD)
                # 修 USB CDC 接收延遲：原本每次 Modbus read 600-1200ms
                # set_low_latency_mode 對 cdc_acm 驅動可能無效 (CDC 是 packet-based)
                # 改用 inter_byte_timeout: byte 之間沒資料就立刻 return，避免硬等
                # 整個 timeout (0.3s)。短 frame Modbus 受益最大。
                try:
                    self._dev._ser.set_low_latency_mode(True)
                except Exception as e:
                    print(f"[io] set_low_latency_mode failed (non-fatal): {e}", flush=True)
                try:
                    # byte 間隔 > 10ms 視為 frame 結束
                    self._dev._ser.inter_byte_timeout = 0.01
                    # 整體 timeout 從 0.3s 降到 0.1s — 9600bps 8-byte response 7ms 應該夠
                    self._dev._ser.timeout = 0.1
                except Exception as e:
                    print(f"[io] timeout adjust failed (non-fatal): {e}", flush=True)
                self._dev.read_outputs()   # smoke-test
                self._ok    = True
                self._error = ""
                print(f"[io] connected on {IO_PORT}", flush=True)
                return True
            except Exception as e:
                self._ok    = False
                self._error = str(e)
                print(f"[io] connect failed: {e}", flush=True)
                return False

    def _close_locked(self) -> None:
        if self._dev:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None
            self._ok  = False

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def error(self) -> str:
        return self._error

    # ── retrying ops ─────────────────────────────────────────────────
    def _call(self, fn, retries: int = 1):
        for attempt in range(retries + 1):
            with self._lock:
                if not self._ok:
                    raise IOError("IO module not connected")
                try:
                    return fn()
                except (ModbusError, IOError, OSError) as e:
                    self._ok    = False
                    self._error = str(e)
                    print(f"[io] comm error (attempt {attempt}): {e}", flush=True)
            if attempt < retries:
                time.sleep(0.1)
                self.connect()
        raise IOError(f"IO module failed: {self._error}")

    # ── public API ────────────────────────────────────────────────────
    def read_inputs(self) -> list[bool]:
        return self._call(lambda: self._dev.read_inputs())

    def read_outputs(self) -> list[bool]:
        return self._call(lambda: self._dev.read_outputs())

    def set_relay(self, ch: int, on: bool) -> None:
        self._call(lambda: self._dev.set_relay(ch, on))

    def set_relays(self, states: list[bool]) -> None:
        self._call(lambda: self._dev.set_relays(states))

    def read_all_counters(self) -> list[int]:
        return self._call(lambda: self._dev.read_all_counters())


_module = IOModule()


def get_module() -> IOModule:
    return _module
