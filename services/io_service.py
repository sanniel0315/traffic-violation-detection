"""
IO service: maps system state -> DO outputs, monitors DI edges.

DO mapping:
  DO0 (red)    - 通訊故障:  ON=fault,  OFF=normal
  DO1 (green)  - 運作狀況:  solid ON=OK, OFF=fault
  DO2 (white)  - 操作模式:  ON=auto 或 下載中, OFF=手動+閒置

DI mapping:
  DI0 - not used
  DI1 - Reset button
  DI2 - 遠端下載 (triggers config sync)

Lamp states:
  正常/閒置(自動) : red=OFF  green=ON  white=ON
  正常/閒置(手動) : red=OFF  green=ON  white=OFF
  遠端下載中      : red=OFF  green=ON  white=ON  (不論自動/手動)
  通訊故障        : red=ON   green=OFF white=OFF
"""
from __future__ import annotations

import datetime
import threading
import time
from collections import deque
from typing import Callable, Optional

from services.io_module import IOModule, get_module

def _log(level: str, msg: str) -> None:
    try:
        from api.routes.logs import add_log
        add_log(level, msg, "io")
    except Exception:
        pass

DO_RED   = 0
DO_GREEN = 1
DO_WHITE = 2

# DI channel index in pd3r3.read_inputs() / read_all_counters() return list.
# DI0 釋出不用；DI1 = Reset；DI2 = 遠端下載 (見 IO_MODULE_PLANNING.md §1)
DI_RESET    = 1
DI_DOWNLOAD = 2


class IOService:
    def __init__(self, module: Optional[IOModule] = None):
        self._mod = module or get_module()
        self._lock = threading.Lock()
        # system state
        self._comm_ok     = True
        self._auto_mode   = True   # True=自動, False=手動
        self._downloading = False
        # threads
        self._di_thread: Optional[threading.Thread] = None
        self._stop_di     = threading.Event()
        self._di_counters = [0, 0, 0]
        # DI event callbacks + WS history
        self._di_callbacks: list[Callable[[int], None]] = []
        self.di_events: deque = deque(maxlen=50)

    # ── lifecycle ─────────────────────────────────────────────────────
    def start(self) -> None:
        if not self._mod.ok:
            self._mod.connect()
        self._apply_do()
        self._start_di_monitor()
        self._wire_download_button()
        from services import network_health
        network_health.start()
        print("[io_svc] started", flush=True)
        _log("info", "IO 模組啟動，連線成功")

    def stop(self) -> None:
        self._stop_di.set()
        from services import network_health
        network_health.stop()
        try:
            self._mod.set_relays([False, False, False])
        except Exception:
            pass
        self._mod.close()
        print("[io_svc] stopped", flush=True)
        _log("info", "IO 模組關閉")

    # ── state setters ─────────────────────────────────────────────────
    def set_comm_fault(self, fault: bool) -> None:
        with self._lock:
            changed = self._comm_ok == fault   # comm_ok was True, now fault=True → changed
            self._comm_ok = not fault
        if changed:
            _log("error" if fault else "info",
                 f"通訊故障: {'發生' if fault else '恢復正常'}")
        self._apply_do()

    def set_auto_mode(self, auto: bool) -> None:
        with self._lock:
            changed = self._auto_mode != auto
            self._auto_mode = auto
        if changed:
            _log("info", f"操作模式切換: {'自動' if auto else '手動'}")
        self._apply_do()

    def set_downloading(self, active: bool) -> None:
        with self._lock:
            self._downloading = active
        self._apply_do()

    # ── DO ────────────────────────────────────────────────────────────
    def _apply_do(self) -> None:
        try:
            fault = not self._comm_ok
            white = (self._auto_mode or self._downloading) and not fault
            self._mod.set_relay(DO_RED,   fault)
            self._mod.set_relay(DO_GREEN, not fault)
            self._mod.set_relay(DO_WHITE, white)
        except Exception as e:
            print(f"[io_svc] apply DO error: {e}", flush=True)
            _log("error", f"IO 燈號寫入失敗: {e}")

    # ── download button wiring ─────────────────────────────────────────
    def _wire_download_button(self) -> None:
        def _on_di(ch: int) -> None:
            if ch != DI_DOWNLOAD:
                return
            from services import config_sync
            if config_sync.is_running():
                _log("warning", "遠端下載已在執行中，忽略重複觸發")
                return
            _log("info", "DI2 按下，開始遠端 config 同步")
            self.set_downloading(True)

            def _done(success: bool, msg: str) -> None:
                self.set_downloading(False)
                if success:
                    _log("info", f"遠端 config 同步成功: {msg}")
                else:
                    _log("error", f"遠端 config 同步失敗: {msg}")
                    threading.Thread(target=self._pulse_red, daemon=True).start()

            config_sync.on_complete(_done)
            config_sync.trigger()

        self.on_di_event(_on_di)

    def _pulse_red(self, duration: float = 3.0) -> None:
        try:
            self._mod.set_relay(DO_RED, True)
            time.sleep(duration)
            if self._comm_ok:
                self._mod.set_relay(DO_RED, False)
        except Exception:
            pass

    # ── DI monitor ────────────────────────────────────────────────────
    def on_di_event(self, cb: Callable[[int], None]) -> None:
        self._di_callbacks.append(cb)

    def _start_di_monitor(self) -> None:
        try:
            self._di_counters = self._mod.read_all_counters()
        except Exception:
            self._di_counters = [0, 0, 0]
        self._di_thread = threading.Thread(
            target=self._di_loop, daemon=True, name="io-di-monitor"
        )
        self._di_thread.start()

    def _di_loop(self) -> None:
        interval = 1.0 / 20
        while not self._stop_di.is_set():
            try:
                cur = self._mod.read_all_counters()
                for ch in (DI_RESET, DI_DOWNLOAD):
                    if cur[ch] != self._di_counters[ch]:
                        self._fire_di(ch)
                        self._di_counters[ch] = cur[ch]
            except Exception as e:
                print(f"[io_svc] DI poll error: {e}", flush=True)
                _log("error", f"IO 通訊中斷，嘗試重連: {e}")
                time.sleep(1.0)
                if not self._mod.ok:
                    if self._mod.connect():
                        _log("info", "IO 模組重連成功")
            self._stop_di.wait(interval)

    def _fire_di(self, ch: int) -> None:
        labels = {DI_RESET: "Reset", DI_DOWNLOAD: "遠端下載"}
        label = labels.get(ch, f"DI{ch}")
        evt = {
            "channel": ch,
            "label":   label,
            "time":    datetime.datetime.now().isoformat(),
        }
        self.di_events.append(evt)
        _log("info", f"按鍵觸發: {label} (DI{ch})")
        for cb in self._di_callbacks:
            try:
                cb(ch)
            except Exception:
                pass

    # ── status snapshot ───────────────────────────────────────────────
    def status(self) -> dict:
        try:
            di = self._mod.read_inputs()
            do = self._mod.read_outputs()
            err = ""
        except Exception as e:
            di  = [None, None, None]
            do  = [None, None, None]
            err = str(e)

        from services import config_sync
        return {
            "connected": self._mod.ok,
            "error":     err or self._mod.error,
            "di": {
                "DI0": {"label": "未使用",   "state": di[0]},
                "DI1": {"label": "Reset",    "state": di[1]},
                "DI2": {"label": "遠端下載", "state": di[2]},
            },
            "do": {
                "DO0": {"color": "red",   "label": "通訊故障", "state": do[0]},
                "DO1": {"color": "green", "label": "運作狀況", "state": do[1]},
                "DO2": {"color": "white", "label": "操作模式", "state": do[2]},
            },
            "state": {
                "comm_ok":     self._comm_ok,
                "auto_mode":   self._auto_mode,
                "downloading": self._downloading,
            },
            "config_sync": config_sync.status(),
        }


_service: Optional[IOService] = None


def get_service() -> IOService:
    global _service
    if _service is None:
        _service = IOService()
    return _service


def start() -> None:
    get_service().start()


def stop() -> None:
    if _service:
        _service.stop()
