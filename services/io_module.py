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

# pd3r3 driver 隨 git 部署 (services/pd3r3.py)，不再依賴 ~/pd3r3.py
# 跨機部署 / docker container 都能 import
# 但 staging 環境可能沒裝 pyserial / 沒接實體 IO 模組 — graceful degrade，
# 讓 service 仍可啟動，IO 自動 disabled (connect 永遠回 False)
try:
    from services.pd3r3 import PD3R3, ModbusError
    _IO_DRIVER_AVAILABLE = True
except ImportError as _e:
    print(f"[io] pd3r3 driver unavailable, IO disabled: {_e}", flush=True)
    _IO_DRIVER_AVAILABLE = False
    PD3R3 = None  # type: ignore

    class ModbusError(IOError):
        pass

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
        if not _IO_DRIVER_AVAILABLE:
            with self._lock:
                self._ok = False
                self._error = "pd3r3 driver / pyserial not installed"
                return False
        with self._lock:
            self._close_locked()
            try:
                self._dev = PD3R3(IO_PORT, IO_ADDR, IO_BAUD)
                # 註：USB CDC ACM 的 read 延遲由 cdc_acm 驅動的 USB endpoint
                # polling interval 控制，pyserial 的 timeout / inter_byte_timeout
                # / set_low_latency_mode 對 ttyACM 都沒幫助 (前者 silent 後兩者
                # 觸發 RS485 ioctl 失敗)。改在 IOService.status() 加 cache 解
                # 多 caller 打爆 Modbus 問題 — 見 io_service.py。
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

    def read_holding(self, slave_addr: int, start: int, count: int) -> list[int]:
        """FC03 讀同一條 RS-485 上其他從機(如 E-1507 電子鎖)的 holding register。
        鎖是獨立從機,不受 PD3R3 連線狀態(_ok)影響 → 不走 _call 的 _ok gate,
        只要 serial 開著就能讀(PD3R3 沒接時 _ok=False 但 serial 仍可用)。
        仍用 self._lock 與 PD3R3 read/write 序列化,避免 bus 並發 SEGV。"""
        with self._lock:
            if self._dev is None:
                if not _IO_DRIVER_AVAILABLE:
                    raise IOError("pd3r3 driver / pyserial not installed")
                self._dev = PD3R3(IO_PORT, IO_ADDR, IO_BAUD)
            try:
                return self._dev.read_holding_at(slave_addr, start, count)
            except OSError:
                self._close_locked()   # serial 層壞了 → 下次重開
                raise

    def write_holding(self, slave_addr: int, reg: int, value: int) -> None:
        """FC06 寫單寄存器到 RS-485 上其他從機(電子鎖,如加卡觸發 0x2005)。
        同 read_holding 繞過 _ok gate,只要 serial 開著就能寫;寬容寫不驗回應。"""
        with self._lock:
            if self._dev is None:
                if not _IO_DRIVER_AVAILABLE:
                    raise IOError("pd3r3 driver / pyserial not installed")
                self._dev = PD3R3(IO_PORT, IO_ADDR, IO_BAUD)
            try:
                self._dev.write_holding_at(slave_addr, reg, value)
            except OSError:
                self._close_locked()
                raise

    def write_holding_multi(self, slave_addr: int, start: int, values: list[int]) -> None:
        """FC10 寫多寄存器到電子鎖(如刪卡 0x0047-0x0049)。同 read_holding 繞過 _ok gate。"""
        with self._lock:
            if self._dev is None:
                if not _IO_DRIVER_AVAILABLE:
                    raise IOError("pd3r3 driver / pyserial not installed")
                self._dev = PD3R3(IO_PORT, IO_ADDR, IO_BAUD)
            try:
                self._dev.write_multi_holding_at(slave_addr, start, values)
            except OSError:
                self._close_locked()
                raise

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
