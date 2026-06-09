"""
Modbus TCP IO driver — 支援 ICP DAS tPET / tET 系列 (跟現有 USB PD3R3 並存)。

設計目標:
- 多台 module 並列 (per-station 獨立連線)
- thread-safe (同一 station 一個 lock)
- auto-reconnect (連線斷自動重試)
- graceful degrade (pymodbus 沒裝仍可 import,所有 op 回 error)

tPET-P2POR2 預設:
- IP 192.168.255.1, port 502, unit_id 1
- DI: 2 channels at address 0 (FC02 read discrete inputs)
- DO: 2 channels at address 0 (FC01 read coils / FC05 write single coil)
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

try:
    from pymodbus.client import ModbusTcpClient
    _MODBUS_AVAILABLE = True
except ImportError as _e:
    print(f"[io_tcp] pymodbus unavailable: {_e}", flush=True)
    _MODBUS_AVAILABLE = False
    ModbusTcpClient = None  # type: ignore


@dataclass
class ModbusTcpConfig:
    """單一 Modbus TCP IO module 配置"""
    id: str                    # 站號 (e.g. "junction_1" / "site_a")
    name: str                  # 顯示名稱
    ip: str                    # 192.168.255.1
    port: int = 502
    unit_id: int = 1           # Modbus slave ID
    di_count: int = 2          # DI 通道數
    do_count: int = 2          # DO 通道數
    di_addr: int = 0           # DI 起始 address (FC02)
    do_addr: int = 0           # DO 起始 address (FC01/FC05)
    timeout: float = 1.5
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ModbusTcpIO:
    """單一 tPET 模組封裝 (thread-safe + auto-reconnect)"""

    def __init__(self, cfg: ModbusTcpConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._client: Optional["ModbusTcpClient"] = None
        self._ok = False
        self._error = ""
        self._last_di: List[bool] = [False] * cfg.di_count
        self._last_do: List[bool] = [False] * cfg.do_count
        self._last_read_at: float = 0.0

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def error(self) -> str:
        return self._error

    def connect(self) -> bool:
        if not _MODBUS_AVAILABLE:
            self._error = "pymodbus not installed"
            return False
        with self._lock:
            self._close_locked()
            try:
                self._client = ModbusTcpClient(self.cfg.ip, port=self.cfg.port,
                                                timeout=self.cfg.timeout)
                if not self._client.connect():
                    self._ok = False
                    self._error = f"connect to {self.cfg.ip}:{self.cfg.port} failed"
                    return False
                # smoke test — 讀 DO coils
                rr = self._client.read_coils(address=self.cfg.do_addr,
                                              count=self.cfg.do_count,
                                              device_id=self.cfg.unit_id)
                if rr.isError():
                    self._ok = False
                    self._error = f"smoke read_coils err: {rr}"
                    return False
                self._ok = True
                self._error = ""
                print(f"[io_tcp:{self.cfg.id}] connected {self.cfg.ip}:{self.cfg.port}", flush=True)
                return True
            except Exception as e:
                self._ok = False
                self._error = str(e)
                return False

    def _close_locked(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._ok = False

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _ensure_connected(self) -> bool:
        if self._ok and self._client is not None:
            return True
        return self.connect()

    def read_inputs(self) -> List[bool]:
        """讀 DI (FC02)"""
        if not self._ensure_connected():
            return list(self._last_di)
        with self._lock:
            try:
                rr = self._client.read_discrete_inputs(address=self.cfg.di_addr,
                                                       count=self.cfg.di_count,
                                                       device_id=self.cfg.unit_id)
                if rr.isError():
                    self._ok = False
                    self._error = f"read_discrete_inputs err: {rr}"
                    return list(self._last_di)
                self._last_di = list(rr.bits[:self.cfg.di_count])
                self._last_read_at = time.time()
                return list(self._last_di)
            except Exception as e:
                self._ok = False
                self._error = str(e)
                return list(self._last_di)

    def read_outputs(self) -> List[bool]:
        """讀 DO coils (FC01)"""
        if not self._ensure_connected():
            return list(self._last_do)
        with self._lock:
            try:
                rr = self._client.read_coils(address=self.cfg.do_addr,
                                              count=self.cfg.do_count,
                                              device_id=self.cfg.unit_id)
                if rr.isError():
                    self._ok = False
                    self._error = f"read_coils err: {rr}"
                    return list(self._last_do)
                self._last_do = list(rr.bits[:self.cfg.do_count])
                return list(self._last_do)
            except Exception as e:
                self._ok = False
                self._error = str(e)
                return list(self._last_do)

    def write_output(self, channel: int, on: bool) -> bool:
        """寫單一 DO (FC05)"""
        if channel < 0 or channel >= self.cfg.do_count:
            self._error = f"channel {channel} out of range (0..{self.cfg.do_count - 1})"
            return False
        if not self._ensure_connected():
            return False
        with self._lock:
            try:
                wr = self._client.write_coil(address=self.cfg.do_addr + channel,
                                              value=bool(on),
                                              device_id=self.cfg.unit_id)
                if wr.isError():
                    self._ok = False
                    self._error = f"write_coil err: {wr}"
                    return False
                self._last_do[channel] = bool(on)
                return True
            except Exception as e:
                self._ok = False
                self._error = str(e)
                return False


# ─── Registry — 多 station 管理 ───
_REGISTRY: Dict[str, ModbusTcpIO] = {}
_REGISTRY_LOCK = threading.Lock()
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "io_tcp_modules.json")


def _load_configs() -> List[ModbusTcpConfig]:
    if not os.path.exists(_CONFIG_PATH):
        return []
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for d in data.get("modules", []):
            try:
                out.append(ModbusTcpConfig(**d))
            except TypeError as e:
                print(f"[io_tcp] skip bad config: {d} ({e})", flush=True)
        return out
    except Exception as e:
        print(f"[io_tcp] load configs failed: {e}", flush=True)
        return []


def _save_configs(cfgs: List[ModbusTcpConfig]) -> None:
    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"modules": [c.to_dict() for c in cfgs]}, f, ensure_ascii=False, indent=2)


def list_configs() -> List[ModbusTcpConfig]:
    return _load_configs()


def get_module(module_id: str) -> Optional[ModbusTcpIO]:
    with _REGISTRY_LOCK:
        return _REGISTRY.get(module_id)


def upsert_config(cfg: ModbusTcpConfig) -> None:
    """新增或更新一個 module config + 重新建立 client"""
    cfgs = _load_configs()
    cfgs = [c for c in cfgs if c.id != cfg.id]
    cfgs.append(cfg)
    _save_configs(cfgs)
    with _REGISTRY_LOCK:
        old = _REGISTRY.pop(cfg.id, None)
        if old:
            try:
                old.close()
            except Exception:
                pass
        if cfg.enabled:
            mod = ModbusTcpIO(cfg)
            mod.connect()  # 連不上沒關係,後續 endpoint call 再 reconnect
            _REGISTRY[cfg.id] = mod


def delete_config(module_id: str) -> bool:
    cfgs = _load_configs()
    new_cfgs = [c for c in cfgs if c.id != module_id]
    if len(new_cfgs) == len(cfgs):
        return False
    _save_configs(new_cfgs)
    with _REGISTRY_LOCK:
        old = _REGISTRY.pop(module_id, None)
        if old:
            try:
                old.close()
            except Exception:
                pass
    return True


def init_from_config() -> None:
    """app startup 時呼叫,把 config 內所有 module 連起來"""
    for cfg in _load_configs():
        if not cfg.enabled:
            continue
        with _REGISTRY_LOCK:
            mod = ModbusTcpIO(cfg)
            mod.connect()
            _REGISTRY[cfg.id] = mod
