"""
IO service: maps system state -> DO outputs, monitors DI edges.

DO mapping:
  DO0 (red)    - 通訊故障:  ON=NTP/link/IP fault,  OFF=normal
                            (config sync 失敗只記 log 不動紅燈)
  DO1 (green)  - 運作狀況:  solid ON=OK, OFF=fault
  DO2 (white)  - 操作模式:  恆亮=手動模式;  閃爍 ~2.5Hz=遠端下載中;
                            滅=自動(遠端)+閒置 / 通訊故障

DI mapping:
  DI0 - 遠端下載 (triggers config sync)
  DI1 - Reset button
  DI2 - 保留 (尚未指派功能)

Lamp states:
  正常/閒置(自動/遠端) : red=OFF  green=ON  white=OFF
  正常/閒置(手動)      : red=OFF  green=ON  white=ON  (恆亮)
  遠端下載中           : red=OFF  green=ON  white=BLINK 2.5Hz
  通訊故障             : red=ON   green=OFF white=OFF
"""
from __future__ import annotations

import datetime
import os
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
# DI0 = 遠端下載；DI1 = Reset；DI2 = 保留（見 IO_MODULE_PLANNING.md §1）
DI_DOWNLOAD = 0
DI_RESET    = 1


_DAEMON_URL = os.getenv("IO_DAEMON_URL", "").rstrip("/")


class IOService:
    def __init__(self, module: Optional[IOModule] = None):
        # client mode: 透過 HTTP 跟 traffic-io.service daemon 通訊（process 隔離）
        self._daemon_url = _DAEMON_URL
        if self._daemon_url:
            import requests as _req
            self._session = _req.Session()
        # 不管 native / client mode 都要保留以下 state 給 status() / WS 用
        self._mod = module or get_module()
        self._lock = threading.Lock()
        # system state
        self._comm_ok       = True
        self._auto_mode     = True   # True=自動, False=手動
        self._downloading   = False
        self._reset_pending = False  # DI1 防連按重啟
        # 下載期間白燈閃爍控制
        self._dl_blink_stop   = threading.Event()
        self._dl_blink_thread: Optional[threading.Thread] = None
        # status() 不打硬體 — IOService 自己擁有所有 DO state，DI 由 _di_loop
        # 20Hz 抓 counter delta 推算「最近按下」。USB CDC ACM read 60-1200ms 延遲
        # 太大不適合放在 web 同步路徑。實機 instantaneous level 由 di_loop 自己
        # 維護 self._di_levels（每 50ms 更新一次）。
        self._do_state    = [False, False, False]   # 上次 _apply_do 寫的 DO 物理狀態
        self._di_levels   = [False, False, False]   # _di_loop 最近一次 read_inputs 結果
        # threads
        self._di_thread: Optional[threading.Thread] = None
        self._stop_di     = threading.Event()
        self._di_counters = [0, 0, 0]
        # DI event callbacks + WS history
        self._di_callbacks: list[Callable[[int], None]] = []
        self.di_events: deque = deque(maxlen=50)
        # Monotonic sequence number for WS tracking — deque maxlen 會讓 len() 飽和，
        # 超過 50 個事件後 len() 不再變大，需要獨立 seq 才能正確判斷新事件。
        self._di_event_seq = 0

    # ── lifecycle ─────────────────────────────────────────────────────
    def start(self) -> None:
        if self._daemon_url:
            # client mode: 不開 PD3R3 / serial port，改起 long-poll thread 從 daemon
            # /events 接 DI rising edge，本地觸發 reset / download callback。
            self._wire_download_button()
            self._wire_reset_button()
            threading.Thread(target=self._client_di_poller, daemon=True, name="io-di-poller").start()
            print(f"[io_svc] started (client mode, daemon={self._daemon_url})", flush=True)
            _log("info", f"IO 模組啟動 (client mode, daemon={self._daemon_url})")
            from services import network_health
            network_health.start(interval=5)
            return
        if not self._mod.ok:
            self._mod.connect()
        if self._mod.ok:
            # IO 硬體連線成功 → 啟動 DI 監聽
            self._apply_do()
            self._start_di_monitor()
            # daemon host mode 不 wire reset/download callback (client 端去接 events
            # 觸發 reset / config_sync — 避免雙重觸發)
            if os.getenv("IO_DAEMON_HOST", "0") != "1":
                self._wire_download_button()
                self._wire_reset_button()
            print("[io_svc] started (IO active)", flush=True)
            _log("info", "IO 模組啟動，連線成功")
        else:
            # 沒接 IO 或 driver 缺失 (staging) — 跳過 IO 監聽，避免 _di_loop 無限重試 spam log
            print(f"[io_svc] started (IO unavailable: {self._mod.error})", flush=True)
            _log("warning", f"IO 模組未啟動: {self._mod.error}")
        from services import network_health
        # 5 秒輪詢: 最壞 5 秒內偵測網路斷線 / NTP 失同步 / 無 IP → DO0 紅燈
        network_health.start(interval=5)

    def _client_di_poller(self) -> None:
        """Long-poll daemon /events，本地 fire callback (reset / download trigger)。"""
        while not self._stop_di.is_set():
            try:
                r = self._session.get(
                    f"{self._daemon_url}/events",
                    params={"since": self._di_event_seq},
                    timeout=10,
                )
                d = r.json()
                for evt in d.get("events", []) or []:
                    seq = int(evt.get("seq", 0))
                    ch = int(evt.get("channel", -1))
                    if ch < 0:
                        continue
                    if seq > self._di_event_seq:
                        self._di_event_seq = seq
                        self.di_events.append(evt)
                    for cb in self._di_callbacks:
                        try:
                            cb(ch)
                        except Exception:
                            pass
            except Exception:
                time.sleep(1.0)

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

    # ── client-mode HTTP helpers ──────────────────────────────────────
    def _daemon_post(self, path: str, json_body: dict) -> None:
        try:
            self._session.post(f"{self._daemon_url}{path}", json=json_body, timeout=2)
        except Exception as e:
            print(f"[io_svc] daemon {path} unreachable: {e}", flush=True)

    # ── state setters ─────────────────────────────────────────────────
    def set_comm_fault(self, fault: bool) -> None:
        if self._daemon_url:
            self._daemon_post("/set_comm_fault", {"fault": fault})
            return
        with self._lock:
            changed = self._comm_ok == fault   # comm_ok was True, now fault=True → changed
            self._comm_ok = not fault
        if changed:
            _log("error" if fault else "info",
                 f"通訊故障: {'發生' if fault else '恢復正常'}")
        self._apply_do()

    def set_auto_mode(self, auto: bool) -> None:
        if self._daemon_url:
            self._daemon_post("/set_auto_mode", {"auto": auto})
            return
        with self._lock:
            changed = self._auto_mode != auto
            self._auto_mode = auto
        if changed:
            _log("info", f"操作模式切換: {'自動' if auto else '手動'}")
        self._apply_do()

    def set_downloading(self, active: bool) -> None:
        if self._daemon_url:
            self._daemon_post("/set_downloading", {"active": active})
            return
        with self._lock:
            was = self._downloading
            self._downloading = active
        if active and not was:
            # 進入下載 → 啟動白燈閃爍 thread
            self._dl_blink_stop.clear()
            self._dl_blink_thread = threading.Thread(
                target=self._blink_white_loop, daemon=True, name="io-blink-dl"
            )
            self._dl_blink_thread.start()
        elif not active and was:
            # 結束下載 → 停止閃爍，由 _apply_do 寫回正確白燈狀態
            self._dl_blink_stop.set()
        self._apply_do()

    def _blink_white_loop(self) -> None:
        """下載期間白燈 ~2.5Hz 閃爍；通訊故障時讓 _apply_do 接管 (不寫白燈避開覆蓋)。"""
        on = False
        while not self._dl_blink_stop.is_set():
            on = not on
            if self._comm_ok:
                try:
                    self._mod.set_relay(DO_WHITE, on)
                except Exception:
                    pass
            # 0.2s on / 0.2s off → 2.5 Hz
            if self._dl_blink_stop.wait(0.2):
                break

    # ── DO ────────────────────────────────────────────────────────────
    def _apply_do(self) -> None:
        if not self._mod.ok:
            # 沒接 IO / driver 缺失：不嘗試寫 relay，避免每次 set_comm_fault 都 spam log
            return
        try:
            fault = not self._comm_ok
            self._mod.set_relay(DO_RED,   fault)
            self._do_state[DO_RED] = fault
            self._mod.set_relay(DO_GREEN, not fault)
            self._do_state[DO_GREEN] = not fault
            # 下載中 → 由 _blink_white_loop 控制白燈，這裡不要動白燈（除非故障）
            if fault:
                # 故障壓過閃爍：強制白燈滅，覆蓋 blink loop 上次寫的狀態
                self._mod.set_relay(DO_WHITE, False)
                self._do_state[DO_WHITE] = False
            elif not self._downloading:
                # 非下載 + 非故障 → 白燈 = 手動模式
                white = (not self._auto_mode)
                self._mod.set_relay(DO_WHITE, white)
                self._do_state[DO_WHITE] = white
            # 下載中 + 非故障 → 由 blink loop 自己刷新白燈 (state 不可信，UI 用 downloading flag 顯示閃爍)
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
            _log("info", f"DI{DI_DOWNLOAD} 按下，開始遠端 config 同步")
            self.set_downloading(True)

            def _done(success: bool, msg: str) -> None:
                self.set_downloading(False)
                if success:
                    _log("info", f"遠端 config 同步成功: {msg}")
                else:
                    # sync 失敗只記 log（不動紅燈，避免跟 DO0 通訊故障語意混淆）
                    _log("error", f"遠端 config 同步失敗: {msg}")

            config_sync.on_complete(_done)
            config_sync.trigger()

        self.on_di_event(_on_di)

    # ── reset button wiring ───────────────────────────────────────────
    def _wire_reset_button(self) -> None:
        """DI1 按下 → 2 秒後 systemctl restart traffic-api.service。

        2 秒延遲讓 log 寫進 journal、給操作者看到提示；
        Popen 用 start_new_session 避免被父程序帶走，systemd 會 SIGTERM 本身。
        """
        def _on_di(ch: int) -> None:
            if ch != DI_RESET:
                return
            with self._lock:
                if self._reset_pending:
                    _log("warning", "DI1 重啟流程進行中，忽略重複按鍵")
                    return
                self._reset_pending = True
            _log("warning", "DI1 按下，2 秒後重啟 traffic-api.service")
            threading.Thread(
                target=self._do_reset, daemon=True, name="io-reset"
            ).start()
        self.on_di_event(_on_di)

    def _do_reset(self) -> None:
        import subprocess
        # 先停 _di_loop 避免 reset 流程內 set_relays 跟 _di_loop read_inputs
        # 同時打 RS-485 serial port 造成 native SEGV race。
        self._stop_di.set()
        time.sleep(0.5)  # 等 _di_loop 結束當前 sample + 退出 loop
        time.sleep(1.5)  # 剩 1.5s 湊滿原本的 2s 提示間隔
        # 重啟前主動把 3 顆燈全滅，視覺上表達「系統沒在跑」；
        # 新 process 起來後 start() → _apply_do() 會自動恢復正常燈號。
        try:
            self._mod.set_relays([False, False, False])
            print("[io_svc] reset → all DO off (重啟中提示)", flush=True)
            time.sleep(0.2)  # 給 RS-485 frame 完整發送的時間
        except Exception as e:
            print(f"[io_svc] reset clear lamps failed: {e}", flush=True)
        try:
            subprocess.Popen(
                ["sudo", "-n", "systemctl", "restart", "traffic-api.service"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("[io_svc] reset triggered → systemd restart", flush=True)
        except Exception as e:
            print(f"[io_svc] reset failed: {e}", flush=True)
            _log("error", f"DI1 Reset 失敗: {e}")
            with self._lock:
                self._reset_pending = False

    # ── DI monitor ────────────────────────────────────────────────────
    def on_di_event(self, cb: Callable[[int], None]) -> None:
        self._di_callbacks.append(cb)

    def _start_di_monitor(self) -> None:
        # IO_DI_DISABLED=1 → 停掉 in-process _di_loop polling (DI 改由獨立 daemon
        # 透過 long-poll 觸發)。避開 pyserial / Tegra UART 在 traffic-api 內偶發
        # native SEGV 拉倒整個 process 的問題。
        if os.getenv("IO_DI_DISABLED", "0") == "1":
            print("[io_svc] _di_loop disabled (IO_DI_DISABLED=1, use external daemon)", flush=True)
            return
        # read_all_counters (FC4 input register at 30001) 對手上這顆 PD3R3 100% fail
        # (wrong slave addr)，改用 read_inputs (FC2 discrete input) 取 DI level
        # 做 edge detect。read_inputs 已測 100% 穩定。
        try:
            self._di_levels = list(self._mod.read_inputs())
        except Exception:
            self._di_levels = [False, False, False]
        self._di_thread = threading.Thread(
            target=self._di_loop, daemon=True, name="io-di-monitor"
        )
        self._di_thread.start()

    def _di_loop(self) -> None:
        # 50ms → 200ms：降低 RS-485 / pyserial / Tegra UART 半雙工 race window，
        # native SEGV 發生率明顯下降；DI rising edge 仍能 catch 一般按鍵 (≥200ms)
        interval = 0.2
        print(f"[io_svc] _di_loop entered (interval={interval}s)", flush=True)
        _sample_count = 0
        while not self._stop_di.is_set():
            try:
                cur = list(self._mod.read_inputs())
                _sample_count += 1
                # rising edge: prev False → current True
                for ch in (DI_RESET, DI_DOWNLOAD):
                    if cur[ch] and not self._di_levels[ch]:
                        print(f"[io_svc] DI rising edge: ch={ch} levels {self._di_levels} -> {cur}", flush=True)
                        self._fire_di(ch)
                self._di_levels = cur
                # 每 200 sample (~10s) 印一次心跳
                if _sample_count % 200 == 0:
                    print(f"[io_svc] _di_loop alive, samples={_sample_count} last_di={cur}", flush=True)
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
        self._di_event_seq += 1
        evt = {
            "seq":     self._di_event_seq,
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

    # ── public DO control (manual override, bypasses state machine) ──
    def set_relay(self, ch: int, on: bool) -> None:
        """供 API 路由直接驅動單顆 relay；不更新內部狀態旗標。"""
        if self._daemon_url:
            self._daemon_post(f"/do/{ch}", {"on": on})
            return
        self._mod.set_relay(ch, on)
        if 0 <= ch < 3:
            self._do_state[ch] = on

    @property
    def di_event_seq(self) -> int:
        """Monotonic counter；WS 端可用來判斷有沒有新事件。"""
        return self._di_event_seq

    # ── status snapshot ───────────────────────────────────────────────
    def status(self) -> dict:
        """回傳 IO 即時狀態。不打硬體 — 純從 IOService 內部 state cache，永遠 ~5ms。
        DO state 由 _apply_do / set_relay 同步維護；DI level 由 _di_loop 維護
        (按鍵是 momentary 大部分時間 False，UI 看「按下」改用 WS event 驅動
        的 recentDIPress 1.5s 高亮，比 polling instantaneous level 更可靠)。
        """
        if self._daemon_url:
            try:
                return self._session.get(f"{self._daemon_url}/status", timeout=2).json()
            except Exception as e:
                # daemon 不可達：回 placeholder 讓 UI 不要 crash
                return {
                    "connected": False,
                    "error": f"daemon unreachable: {e}",
                    "di": {
                        "DI0": {"label": "遠端下載", "state": False},
                        "DI1": {"label": "Reset",    "state": False},
                        "DI2": {"label": "保留",     "state": False},
                    },
                    "do": {
                        "DO0": {"color": "red",   "label": "通訊故障", "state": False},
                        "DO1": {"color": "green", "label": "運作狀況", "state": False},
                        "DO2": {"color": "white", "label": "操作模式", "state": False},
                    },
                    "state": {"comm_ok": False, "auto_mode": False, "downloading": False},
                    "config_sync": {"url_configured": False, "url": "", "running": False},
                }
        from services import config_sync
        di = self._di_levels
        do = self._do_state
        return {
            "connected": self._mod.ok,
            "error":     self._mod.error,
            "di": {
                "DI0": {"label": "遠端下載", "state": di[0]},
                "DI1": {"label": "Reset",    "state": di[1]},
                "DI2": {"label": "保留",     "state": di[2]},
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
