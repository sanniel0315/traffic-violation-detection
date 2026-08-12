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

# ── E-1507 電子鎖 (同一條 RS-485 上的另一個 Modbus 從機) ──────────────
# 設定 LOCK_MODBUS_ADDR(1-247,需與 PD3R3 的 IO_ADDR 不同) 才會啟用;未設=停用,零影響。
LOCK_ADDR = int(os.getenv("LOCK_MODBUS_ADDR", "0"))   # 0 = 停用
LOCK_SERIAL_PORT = os.getenv("LOCK_SERIAL_PORT", "").strip()  # 設了=鎖走獨立 USB-485(如 /dev/ttyUSB0,可完整讀 0xC000 卡號);空=走 THS2 io_module(讀不到卡號)
LOCK_ENABLED = 1 <= LOCK_ADDR <= 247
# 狀態寄存器: 0x0020 手柄 / 0x0021 門磁 / 0x0022 鑰匙 / 0x0023 鎖具動作
# ⚠ THS2 板載 RS-485 轉換器換向太慢,會吃掉「多暫存器長回應」的前段(實測不管讀
# 幾個,只剩最後 3 byte)。改用 count=1 逐個讀:7-byte 短回應能完整收到、CRC 自驗通過。
_LOCK_REG_HANDLE = 0x0020   # 手柄狀態
_LOCK_REG_DOOR   = 0x0021   # 門磁狀態
_LOCK_REG_KEY    = 0x0022   # 鑰匙狀態
_LOCK_REG_ACTION = 0x0023   # 鎖具動作(刷卡/密碼/指紋/鑰匙/手柄開)
# 動作 idle 實測回 0xfa(250)(此版固件哨兵=無動作),刷卡/開門時才變 1-5
_LOCK_ACTION_NAMES = {0: "無", 1: "刷卡", 2: "密碼", 3: "指紋", 4: "鑰匙轉動", 0xfa: "無"}  # 協議 0x0023:0-4
# 透過 485 輔助錄入用戶資訊 (0x2005): 寫 0x0033 → 鎖進入「加卡模式」,隨後在鎖上刷卡即學習錄入
_LOCK_REG_AUX_ENROLL = 0x2005
_LOCK_AUX_ADD_CARD = 0x0033
_LOCK_AUX_DEL_CARD = 0x0055   # 寫 0x2005=0x0055 → 鎖進入刪卡模式(隨後鎖上刷要刪除的卡)
_LOCK_REG_CLEAR = 0x2006      # 刪除用戶訊息寄存器
_LOCK_CLEAR_CARDS = 0x0011    # 寫 0x2006=0x0011 → 清空所有卡片(FC06,THS2/USB-485 都生效)
_LOCK_REG_CARD_HI = 0x0044    # 加卡寄存器2 (最近加的卡號高 16bit)
_LOCK_REG_CARD_LO = 0x0045    # 加卡寄存器3 (最近加的卡號低 16bit)
_LOCK_REG_DEL_CARD = 0x0047   # 刪卡寄存器1 (FC10 連寫 0x0047-0x0049: 工號+卡號,直接刪不用刷卡)
_LOCK_REG_UNLOCK = 0x2004     # 485 開鎖:寫 0x0033 遠端開鎖
_LOCK_UNLOCK_VAL = 0x0033
_DOOR_OPEN_ALARM_SEC = float(os.getenv("LOCK_DOOR_ALARM_SEC", "30"))  # 箱門開超過 N 秒觸發告警
_LOCK_OFFLINE_ALARM_SEC = float(os.getenv("LOCK_OFFLINE_ALARM_SEC", "60"))  # 斷電/失聯超過 N 秒觸發警報(斷電期間鑰匙機械開門記不到,至少留斷電時段紀錄供查核)


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
        # 電子鎖 poll 用獨立 thread(不依賴 PD3R3 連線,PD3R3 沒接也能讀鎖)
        self._lock_thread: Optional[threading.Thread] = None
        self._stop_lock   = threading.Event()
        self._di_counters = [0, 0, 0]
        # DI event callbacks + WS history
        self._di_callbacks: list[Callable[[int], None]] = []
        self.di_events: deque = deque(maxlen=50)
        # Monotonic sequence number for WS tracking — deque maxlen 會讓 len() 飽和，
        # 超過 50 個事件後 len() 不再變大，需要獨立 seq 才能正確判斷新事件。
        self._di_event_seq = 0
        # E-1507 電子鎖狀態 cache (由 _lock_loop 每 ~1s 更新;未啟用則恆 None)
        self._lock_enabled = LOCK_ENABLED
        self._lock_connected = False
        self._lock_status: dict = {}
        self._lock_err = ""
        # 電子鎖刷卡/開鎖事件 (仿 DI events;動作暫存器高頻邊沿偵測 → deque,client 拉去寫 DB)
        self.lock_events: deque = deque(maxlen=100)
        self._lock_event_seq = 0
        self._lock_prev_action: Optional[int] = None
        self._lock_prev_states: dict = {}        # 門磁/手柄/鑰匙上次狀態(邊沿偵測)
        self._door_open_since: Optional[float] = None
        self._door_alarmed = False
        self._offline_since: Optional[float] = None
        self._offline_alarmed = False
        self._lock_dev = None                       # USB-485 獨立 PD3R3(LOCK_SERIAL_PORT);None=走 THS2
        self._lock_serial_lock = threading.Lock()   # 序列化 USB-485 鎖存取

    # ── lifecycle ─────────────────────────────────────────────────────
    def start(self) -> None:
        if self._daemon_url:
            # client mode: 不開 PD3R3 / serial port，改起 long-poll thread 從 daemon
            # /events 接 DI rising edge，本地觸發 reset / download callback。
            self._wire_download_button()
            self._wire_reset_button()
            threading.Thread(target=self._client_di_poller, daemon=True, name="io-di-poller").start()
            threading.Thread(target=self._client_lock_poller, daemon=True, name="io-lock-poller").start()
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
            # 沒接 IO 或 driver 缺失 (staging) — 跳過 DI 監聽，避免 _di_loop 無限重試 spam log
            print(f"[io_svc] started (IO unavailable: {self._mod.error})", flush=True)
            _log("warning", f"IO 模組未啟動: {self._mod.error}")
        # 電子鎖 poll 獨立於 PD3R3 DI 監聽:即使 PD3R3 沒接(上面 else)也要能讀鎖
        if self._lock_enabled:
            self._start_lock_monitor()
        from services import network_health
        # 5 秒輪詢: 最壞 5 秒內偵測網路斷線 / NTP 失同步 / 無 IP → DO0 紅燈
        network_health.start(interval=5)

    def _client_di_poller(self) -> None:
        """Long-poll daemon /events，本地 fire callback (reset / download trigger)。
        啟動時先讀 daemon latest_seq，跳過所有歷史事件 — 否則 traffic-api 重啟後
        重播 daemon 內歷史 DI1 events 會無限觸發 reset 循環。

        sync 必須成功才能進 events long-poll：systemd 同時拉起 traffic-api 與
        traffic-io 時 daemon 可能尚未 ready，若 sync 失敗用 di_seq=0 進 long-poll
        會把 daemon 之後 publish 的歷史事件全部當新事件 replay。
        """
        # 同步 daemon 當前 latest_seq，本 client 從這之後才接事件；
        # daemon 還沒起就 retry，直到拿到 di_seq 才繼續。
        while not self._stop_di.is_set():
            try:
                r = self._session.get(f"{self._daemon_url}/health", timeout=3)
                self._di_event_seq = int(r.json().get("di_seq") or 0)
                print(f"[io_svc client] synced di_seq={self._di_event_seq} from daemon (skip historic events)", flush=True)
                # daemon 已 ready：清除可能殘留的 stale downloading。上個 traffic-api
                # 在 config_sync 進行中重啟/crash 會讓結束信號丟失、daemon downloading
                # 卡 true → 白燈永久閃。client 重啟代表任何 in-flight 下載都已中斷，
                # 開機清掉是安全且正確的 (download 只由 client 端觸發，daemon 不自發)。
                try:
                    self._session.post(f"{self._daemon_url}/set_downloading", json={"active": False}, timeout=5)
                    print("[io_svc client] cleared stale downloading flag on startup", flush=True)
                except Exception as _e:
                    print(f"[io_svc client] startup downloading reset failed (daemon blink watchdog 仍會兜底): {_e}", flush=True)
                break
            except Exception as e:
                print(f"[io_svc client] init di_seq sync failed, retry in 2s: {e}", flush=True)
                if self._stop_di.wait(2.0):
                    return
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

    def _client_lock_poller(self) -> None:
        """client mode: long-poll daemon /lock_events,寫 DB + 進本地 deque 供 WS 推送。"""
        # 同步 daemon lock_seq,跳過歷史事件 (避免 traffic-api 重啟 replay 寫一堆舊 DB)
        while not self._stop_di.is_set():
            try:
                r = self._session.get(f"{self._daemon_url}/health", timeout=3)
                self._lock_event_seq = int(r.json().get("lock_seq") or 0)
                print(f"[io_svc client] synced lock_seq={self._lock_event_seq}", flush=True)
                break
            except Exception:
                if self._stop_di.wait(2.0):
                    return
        while not self._stop_di.is_set():
            try:
                r = self._session.get(
                    f"{self._daemon_url}/lock_events",
                    params={"since": self._lock_event_seq},
                    timeout=10,
                )
                data = r.json()
                # daemon 重啟偵測:latest_seq 倒退 → daemon seq 歸0,client 舊 seq 過高會
                # 永遠抓不到新事件(警報沒紀錄)。重置為0,下輪從重啟後的事件重新接收。
                latest = int(data.get("latest_seq", 0))
                if latest < self._lock_event_seq:
                    print(f"[io_svc client] daemon lock_seq 倒退 ({self._lock_event_seq}->{latest}),"
                          f"daemon 重啟,resync", flush=True)
                    self._lock_event_seq = 0
                    continue
                for evt in data.get("events", []) or []:
                    seq = int(evt.get("seq", 0))
                    if seq > self._lock_event_seq:
                        self._lock_event_seq = seq
                        self.lock_events.append(evt)
                        self._persist_lock_event(evt)
            except Exception:
                time.sleep(1.0)

    def _persist_lock_event(self, evt: dict) -> None:
        """把鎖事件寫進 DB (LockEvent)。只有 traffic-api(client) 連 DB,daemon 不寫。"""
        try:
            from api.models import SessionLocal, LockEvent, LockCard
            db = SessionLocal()
            try:
                card = evt.get("card_no")
                holder = None
                if card:
                    lc = db.query(LockCard).filter(LockCard.card_no == card, LockCard.active == True).first()
                    holder = lc.holder_name if (lc and lc.holder_name) else None
                db.add(LockEvent(
                    lock_addr=evt.get("addr"),
                    event_type=evt.get("event_type", "swipe"),
                    action_code=evt.get("action"),
                    action_label=evt.get("label"),
                    door_closed=evt.get("door_closed"),
                    handle_in_place=evt.get("handle_in_place"),
                    key_in_place=evt.get("key_in_place"),
                    card_no=card,
                    holder_name=holder,
                ))
                db.commit()
            finally:
                db.close()
        except Exception as e:
            print(f"[io_svc client] lock event persist failed: {e}", flush=True)

    def stop(self) -> None:
        self._stop_di.set()
        self._stop_lock.set()
        from services import network_health
        network_health.stop()
        # client mode: 不要動 self._mod (沒 connect)，也不能關 daemon 上的 relay
        # — daemon 是獨立 systemd 服務，traffic-api 停止不該滅燈。
        if not self._daemon_url:
            try:
                self._mod.set_relays([False, False, False])
            except Exception:
                pass
            self._mod.close()
        print("[io_svc] stopped", flush=True)
        _log("info", "IO 模組關閉")

    # ── client-mode HTTP helpers ──────────────────────────────────────
    def _daemon_post(self, path: str, json_body: dict, timeout: float = 2.0) -> None:
        try:
            self._session.post(f"{self._daemon_url}{path}", json=json_body, timeout=timeout)
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
            # 停閃信號攸關白燈是否卡閃，daemon 寫 RS-485 relay 偶爾排隊 >2s，
            # 故這條路徑放寬到 5s，避免「停閃」被 2s timeout 砍掉而每次都得等 30s 兜底。
            self._daemon_post("/set_downloading", {"active": active}, timeout=5.0)
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
        """下載期間白燈 ~2.5Hz 閃爍；通訊故障時讓 _apply_do 接管 (不寫白燈避開覆蓋)。

        安全兜底：閃爍超過 IO_DOWNLOAD_BLINK_MAX_S 秒仍未收到結束信號就自動復位。
        避免結束信號丟失 (traffic-api 在 config_sync 進行中重啟/crash、daemon POST
        /set_downloading false 不可達) 造成白燈永久閃爍。正常下載僅數秒，遠端 config
        同步硬上限 CONFIG_SYNC_TIMEOUT(預設15s)，故 30s 兜底不會誤殺正常下載。"""
        on = False
        start = time.monotonic()
        try:
            max_s = float(os.getenv("IO_DOWNLOAD_BLINK_MAX_S", "30"))
        except (TypeError, ValueError):
            max_s = 30.0
        while not self._dl_blink_stop.is_set():
            if time.monotonic() - start > max_s:
                _log("warning", f"遠端下載白燈閃爍逾 {max_s:.0f}s 未結束，自動復位 downloading (兜底)")
                self.set_downloading(False)
                break
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
        # 2 秒提示間隔 (log 落地 + 操作者看到提示)，之後滅燈表達「系統沒在跑」。
        # 新 process 起來後 start() → _apply_do() 會自動恢復正常燈號。
        if self._daemon_url:
            # client mode: 本地 _mod 沒 connect，改用 HTTP 通知 daemon 滅燈。
            # daemon 端 _di_loop 持續跑 (IOModule._lock 序列化 read/write，不會
            # race)，重啟期間 DI 監測不中斷 — daemon 仍能 catch 任何 rising edge，
            # 重啟完成後 client poller 起來繼續處理。
            time.sleep(2.0)
            for ch in (DO_RED, DO_GREEN, DO_WHITE):
                self._daemon_post(f"/do/{ch}", {"on": False})
            print("[io_svc] reset → all DO off via daemon (重啟中提示)", flush=True)
        else:
            # native mode (legacy / daemon-less)：本機跑 _di_loop，
            # 先停 loop 避免 set_relays 跟 read_inputs 同時打 RS-485 造成 SEGV。
            self._stop_di.set()
            time.sleep(0.5)  # 等 _di_loop 結束當前 sample + 退出 loop
            time.sleep(1.5)  # 剩 1.5s 湊滿原本的 2s 提示間隔
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

    @staticmethod
    def _is_real_action(a) -> bool:
        """動作碼 1-4 才算真的開鎖動作 (協議 0x0023:0無/1刷卡/2密碼/3指紋/4鑰匙轉動; 0xfa/250=idle 哨兵)。"""
        return a is not None and 1 <= a <= 4

    # ── 鎖讀寫:USB-485(LOCK_SERIAL_PORT)優先,否則 THS2 io_module ──────────
    def _get_lock_dev(self):
        if not LOCK_SERIAL_PORT:
            return None
        if self._lock_dev is None:
            from services.pd3r3 import PD3R3
            self._lock_dev = PD3R3(port=LOCK_SERIAL_PORT, address=LOCK_ADDR)
        return self._lock_dev

    def _close_lock_dev(self):
        if self._lock_dev is not None:
            try:
                self._lock_dev.close()
            except Exception:
                pass
            self._lock_dev = None

    def _lock_read(self, reg: int, count: int = 1):
        dev = self._get_lock_dev()
        if dev is not None:
            with self._lock_serial_lock:
                try:
                    return dev.read_holding_at(LOCK_ADDR, reg, count)
                except Exception:
                    self._close_lock_dev()
                    raise
        return self._mod.read_holding(LOCK_ADDR, reg, count)

    def _lock_write(self, reg: int, value: int):
        dev = self._get_lock_dev()
        if dev is not None:
            with self._lock_serial_lock:
                try:
                    dev.write_holding_at(LOCK_ADDR, reg, value)
                    return
                except Exception:
                    self._close_lock_dev()
                    raise
        self._mod.write_holding(LOCK_ADDR, reg, value)

    def _lock_write_multi(self, reg: int, values: list):
        dev = self._get_lock_dev()
        if dev is not None:
            with self._lock_serial_lock:
                try:
                    dev.write_multi_holding_at(LOCK_ADDR, reg, values)
                    return
                except Exception:
                    self._close_lock_dev()
                    raise
        self._mod.write_holding_multi(LOCK_ADDR, reg, values)

    def _read_lock_swipe_record(self):
        """USB-485 連讀 0xC000 一條開鎖記錄(19byte完整),出隊一條。
        回傳 {rem,way,emp,card,t} 或 None(空記錄/讀失敗)。"""
        try:
            regs = self._lock_read(0xC000, 7)
        except Exception:
            return None
        if not regs or len(regs) < 7:
            return None
        way = regs[0] & 0xFF
        card = f"{regs[2]:04X}{regs[3]:04X}"
        if card == "00000000" or way == 0:
            return None   # 空記錄
        return {
            "rem": regs[0] >> 8, "way": way, "emp": regs[1], "card": card,
            "t": f"{2000 + (regs[4] >> 8)}-{regs[4] & 0xFF:02d}-{regs[5] >> 8:02d} "
                 f"{regs[5] & 0xFF:02d}:{regs[6] >> 8:02d}:{regs[6] & 0xFF:02d}",
        }

    def _read_lock_fail_record(self):
        """USB-485 連讀 0xD000 一條開鎖失敗記錄(連讀6),出隊一條。失效卡/未授權刷卡記在此。
        回傳 {way,card,t} 或 None(空/非卡片失敗/讀失敗)。"""
        try:
            regs = self._lock_read(0xD000, 6)
        except Exception:
            return None
        if not regs or len(regs) < 6:
            return None
        way = regs[0] & 0xFF
        card = f"{regs[1]:04X}{regs[2]:04X}"
        if card == "00000000":
            return None   # 空記錄/非卡片失敗(指紋/密碼失敗無卡號)
        return {
            "way": way, "card": card,
            "t": f"{2000 + (regs[3] >> 8)}-{regs[3] & 0xFF:02d}-{regs[4] >> 8:02d} "
                 f"{regs[4] & 0xFF:02d}:{regs[5] >> 8:02d}:{regs[5] & 0xFF:02d}",
        }

    def _drain_lock_records(self, limit: int = 200):
        """啟動清空 0xC000(成功)+0xD000(失敗)積壓(歷史不 fire,避免 flood)。"""
        n = 0
        for _ in range(limit):
            if self._read_lock_swipe_record() is None:
                break
            n += 1
        nf = 0
        for _ in range(limit):
            if self._read_lock_fail_record() is None:
                break
            nf += 1
        if n or nf:
            print(f"[io_svc] 鎖清空啟動前積壓:成功 {n} 失敗 {nf}", flush=True)

    def _lookup_card_holder(self, card_no: str):
        """查卡片庫該卡號持有人(daemon 直讀 sqlite,找不到回 None)。"""
        try:
            from api.models import SessionLocal, LockCard
            db = SessionLocal()
            try:
                r = db.query(LockCard).filter(LockCard.card_no == card_no, LockCard.active == True).first()
                return r.holder_name if (r and r.holder_name) else None
            finally:
                db.close()
        except Exception:
            return None

    @property
    def lock_event_seq(self) -> int:
        """電子鎖事件單調遞增序號 (WS / client poller 判斷新事件用)。"""
        return self._lock_event_seq

    def _poll_lock_states(self, action: Optional[int] = None) -> None:
        """讀門磁/手柄/鑰匙(慢變)+帶入已知 action,重建 _lock_status。失敗標斷線不拋出。
        逐個 count=1 讀(THS2 換向限制,長回應會被吃前段);每個 7-byte 回應 CRC 自驗。"""
        try:
            handle = self._lock_read(_LOCK_REG_HANDLE)[0]
            door   = self._lock_read(_LOCK_REG_DOOR)[0]
            key    = self._lock_read(_LOCK_REG_KEY)[0]
            act = action if action is not None else (self._lock_prev_action or 0)
            self._lock_status = {
                "handle": {"raw": handle, "in_place": handle == 0,
                           "label": "已上鎖" if handle == 0 else "已解鎖"},
                # 門磁是 NO(常開)接點:門關→磁鐵靠近→接點閉合→raw=1;門開→raw=0。
                # 與協議文件寫的 0=門關 相反,現場實測以此為準 (2026-08-12)。
                "door":   {"raw": door, "closed": door == 1,
                           "label": "門磁閉合(門關)" if door == 1 else "門磁斷開(門開)"},
                "key":    {"raw": key, "in_place": key == 0,
                           "label": "鑰匙在位" if key == 0 else "鑰匙不在位"},
                "action": {"raw": act, "label": _LOCK_ACTION_NAMES.get(act, f"未知({act})")},
            }
            self._lock_connected = True
            self._lock_err = ""
        except Exception as e:
            self._lock_connected = False
            self._lock_err = str(e)

    def _fire_lock_event(self, event_type: str, label: str, action_code=None, card_no=None) -> None:
        """建鎖事件進 deque (client 端拉去寫 DB + 推 WS)。
        event_type: swipe(刷卡)/door(門磁)/lock(上鎖/解鎖)/alarm(警報)/unlock(遠端開鎖)/info。
        card_no: USB-485 刷卡讀到的卡號(client 端比對卡片庫填持有人)。"""
        self._lock_event_seq += 1
        st = self._lock_status or {}
        evt = {
            "seq":     self._lock_event_seq,
            "addr":    LOCK_ADDR,
            "event_type": event_type,
            "action":  action_code,
            "label":   label,
            "card_no": card_no,
            "door_closed":     st.get("door", {}).get("closed"),
            "handle_in_place": st.get("handle", {}).get("in_place"),
            "key_in_place":    st.get("key", {}).get("in_place"),
            "time":    datetime.datetime.now().isoformat(),
        }
        self.lock_events.append(evt)
        _log("warning" if event_type == "alarm" else "info", f"電子鎖[{event_type}]: {label}")

    def _detect_state_events(self) -> None:
        """偵測門磁(門開/關)+鎖定狀態(0x0020,轉鑰匙帶動上鎖/解鎖)變化 → 事件;
        門開超過閾值 → 箱門未關告警。鑰匙 0x0022 恒0 不處理。"""
        st = self._lock_status or {}
        door = st.get("door", {}).get("closed")
        handle = st.get("handle", {}).get("in_place")  # 0x0020: True=上鎖(raw0) False=解鎖(raw1)
        prev = self._lock_prev_states
        if door is not None and prev.get("door") is not None and door != prev["door"]:
            self._fire_lock_event("door", "門關" if door else "門開")
        if handle is not None and prev.get("handle") is not None and handle != prev["handle"]:
            self._fire_lock_event("lock", "上鎖" if handle else "解鎖")
        self._lock_prev_states = {"door": door, "handle": handle}
        # 箱門未關告警 (門開超過閾值,只觸發一次,門關後重置)
        if door is False:
            if self._door_open_since is None:
                self._door_open_since = time.time()
            elif not self._door_alarmed and (time.time() - self._door_open_since) >= _DOOR_OPEN_ALARM_SEC:
                self._fire_lock_event("alarm", f"箱門未關 (>{int(_DOOR_OPEN_ALARM_SEC)}秒)")
                self._door_alarmed = True
        elif door is True:
            self._door_open_since = None
            self._door_alarmed = False

    def _detect_offline(self) -> None:
        """斷電/失聯告警:鎖持續讀不到(connected=false)超過閾值 → 告警一次;恢復連線記一筆。
        斷電期間鑰匙機械開門鎖無電記不到,此告警至少留下斷電時段供事後查核。"""
        if not self._lock_connected:
            if self._offline_since is None:
                self._offline_since = time.time()
            elif not self._offline_alarmed and (time.time() - self._offline_since) >= _LOCK_OFFLINE_ALARM_SEC:
                self._fire_lock_event("alarm", f"電子鎖斷電/失聯 (>{int(_LOCK_OFFLINE_ALARM_SEC)}秒)")
                self._offline_alarmed = True
        else:
            # 失聯持續 ≥5 秒後恢復 → 記恢復連線(不必等滿告警閾值,過濾<5秒偶發抖動;
            # 也讓 daemon 重啟後只要重新失聯過再恢復就記得到)
            if self._offline_since is not None and (time.time() - self._offline_since) >= 5:
                self._fire_lock_event("info", "電子鎖恢復連線")
            self._offline_since = None
            self._offline_alarmed = False

    def _start_lock_monitor(self) -> None:
        """啟動電子鎖獨立 poll thread。與 _di_loop 共用 IOModule._lock 序列化 RS-485,
        但不依賴 PD3R3 連線:PD3R3 沒接(_di_loop 不啟動)時鎖仍能讀。"""
        if self._lock_thread and self._lock_thread.is_alive():
            return
        self._lock_thread = threading.Thread(
            target=self._lock_loop, daemon=True, name="io-lock-monitor"
        )
        self._lock_thread.start()

    def _lock_loop(self) -> None:
        # USB-485 模式:讀 0xC000 完整記錄檢測刷卡(含卡號);THS2 模式:讀 0x0023 action 邊沿(無卡號)。
        # 門磁/手柄/鑰匙慢變 → 每 ~1s(tick 0,7,14...) 刷新一次。
        print(f"[io_svc] _lock_loop entered (port={LOCK_SERIAL_PORT or 'THS2'})", flush=True)
        if LOCK_SERIAL_PORT:
            try:
                self._drain_lock_records()   # 清空啟動前積壓,避免歷史刷卡 flood
            except Exception:
                pass
        tick = 0
        while not self._stop_lock.is_set():
            action = None
            try:
                if LOCK_SERIAL_PORT:
                    # USB-485:讀 0xC000 一條完整記錄,有刷卡 → 帶卡號 fire
                    rec = self._read_lock_swipe_record()
                    self._lock_connected = True
                    self._lock_err = ""
                    if rec:
                        action = rec["way"]
                        base = _LOCK_ACTION_NAMES.get(rec["way"], f"方式{rec['way']}")
                        holder = self._lookup_card_holder(rec["card"])
                        disp = f"{base}:{holder}" if holder else f"{base}:{rec['card']}"
                        self._fire_lock_event("swipe", disp, rec["way"], card_no=rec["card"])
                    # 失效卡/未授權刷卡 → 0xD000 失敗記錄 → 告警
                    frec = self._read_lock_fail_record()
                    if frec:
                        who = self._lookup_card_holder(frec["card"])
                        tag = f":{who}" if who else f":{frec['card']}"
                        self._fire_lock_event("alarm", f"未授權刷卡{tag}", frec["way"], card_no=frec["card"])
                else:
                    # THS2:0x0023 action 上升沿(讀不到卡號)
                    action = self._lock_read(_LOCK_REG_ACTION)[0]
                    self._lock_connected = True
                    self._lock_err = ""
                    if self._is_real_action(action) and not self._is_real_action(self._lock_prev_action):
                        self._fire_lock_event("swipe", _LOCK_ACTION_NAMES.get(action, f"未知({action})"), action)
                    self._lock_prev_action = action
            except Exception as e:
                self._lock_connected = False
                self._lock_err = str(e)
            if tick % 7 == 0:
                self._poll_lock_states(action)
                if self._lock_connected:
                    self._detect_state_events()   # 斷電時 _lock_status 為舊值,跳過避免誤報
                self._detect_offline()
            tick += 1
            self._stop_lock.wait(0.15)

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

    def add_lock_card(self) -> dict:
        """加卡並偵測新卡號:寫 0x2005=0x0033 進加卡模式 → 監測 ~12秒 0x0044/45 變化 → 拿新卡號。
        client mode 同步呼叫 daemon(長 timeout) 拿卡號寫卡片庫;native(daemon) 觸發+監測。"""
        if self._daemon_url:
            try:
                r = self._session.post(f"{self._daemon_url}/lock/add_card", timeout=20)
                d = r.json()
            except Exception as e:
                return {"ok": False, "msg": f"加卡失敗(daemon 無回應): {e}"}
            if d.get("ok") and d.get("card_no"):
                self._persist_lock_card(d.get("card_no"))
            return d
        if not self._lock_enabled:
            return {"ok": False, "msg": "電子鎖未啟用 (LOCK_MODBUS_ADDR 未設)"}
        try:
            before = self._read_lock_cardno()
            self._lock_write(_LOCK_REG_AUX_ENROLL, _LOCK_AUX_ADD_CARD)
            _log("info", "電子鎖進入加卡模式 (監測新卡 ~12s)")
            deadline = time.time() + 12.0
            while time.time() < deadline:
                time.sleep(0.6)
                cur = self._read_lock_cardno()
                if cur and cur != before and cur != "00000000":
                    _log("info", f"加卡成功,新卡號 {cur}")
                    return {"ok": True, "card_no": cur, "msg": f"加卡成功！卡號 {cur}"}
            # 超時未偵測到卡號變化:可能剛刷的是既有卡(卡號==before)。仍讀當前卡號回傳,
            # 讓卡片庫 upsert(既有卡會重新啟用),避免「刷了卡庫卻沒新增」。
            final = self._read_lock_cardno()
            if final and final != "00000000":
                _log("info", f"加卡完成(卡號未變,既有卡/重複),卡號 {final}")
                return {"ok": True, "card_no": final,
                        "msg": f"加卡完成,卡號 {final}（既有卡則重新啟用）"}
            return {"ok": True, "card_no": None, "msg": "加卡結束:未讀到卡號(沒刷/超時)"}
        except Exception as e:
            return {"ok": False, "msg": f"加卡失敗: {e}"}

    def _read_lock_cardno(self):
        """讀加卡寄存器 0x0044/0x0045 組成 hex 卡號字串(如 E14D0395)。失敗回 None。"""
        try:
            hi = self._lock_read(_LOCK_REG_CARD_HI)[0]
            lo = self._lock_read(_LOCK_REG_CARD_LO)[0]
            return f"{hi:04X}{lo:04X}"
        except Exception:
            return None

    def _persist_lock_card(self, card_no: str) -> None:
        """把新卡號寫進卡片庫 (LockCard);已存在(active)則不重複。只有 traffic-api(client) 連 DB。"""
        try:
            from api.models import SessionLocal, LockCard
            db = SessionLocal()
            try:
                ex = db.query(LockCard).filter(LockCard.card_no == card_no).first()
                if ex:
                    if not ex.active:          # 既有卡曾被刪(active=0) → 重新啟用
                        ex.active = True
                        db.commit()
                else:
                    db.add(LockCard(lock_addr=(LOCK_ADDR or None), card_no=card_no))
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            print(f"[io_svc client] lock card persist failed: {e}", flush=True)

    def remove_lock_card(self) -> dict:
        """讓電子鎖進入刪卡模式 (寫 0x2005=0x0055),使用者隨後在鎖上刷要刪除的卡即移除。"""
        if self._daemon_url:
            self._daemon_post("/lock/remove_card", {})
            return {"ok": True, "msg": "已送出刪卡指令,請在約 10 秒內到鎖上刷要刪除的卡"}
        if not self._lock_enabled:
            return {"ok": False, "msg": "電子鎖未啟用 (LOCK_MODBUS_ADDR 未設)"}
        try:
            self._lock_write(_LOCK_REG_AUX_ENROLL, _LOCK_AUX_DEL_CARD)
            _log("info", "電子鎖進入刪卡模式 (等待鎖上刷卡)")
            return {"ok": True, "msg": "鎖已進入刪卡模式,請在約 10 秒內到鎖上刷要刪除的卡"}
        except Exception as e:
            return {"ok": False, "msg": f"刪卡指令失敗: {e}"}

    def clear_all_lock_cards(self) -> dict:
        """清空鎖內所有卡片 (寫 0x2006=0x0011,FC06 單寄存器,THS2/USB-485 都寫得進)。client 轉發 daemon。"""
        if self._daemon_url:
            try:
                r = self._session.post(f"{self._daemon_url}/lock/clear_cards", timeout=8)
                return r.json()
            except Exception as e:
                return {"ok": False, "msg": f"清空失敗(daemon 無回應): {e}"}
        if not self._lock_enabled:
            return {"ok": False, "msg": "電子鎖未啟用 (LOCK_MODBUS_ADDR 未設)"}
        try:
            self._lock_write(_LOCK_REG_CLEAR, _LOCK_CLEAR_CARDS)
            _log("info", "電子鎖清空所有卡片")
            return {"ok": True, "msg": "已清空鎖內所有卡片"}
        except Exception as e:
            return {"ok": False, "msg": f"清空失敗: {e}"}

    def delete_lock_card_by_no(self, card_no: str) -> dict:
        """已知卡號直接刪卡 (FC10 寫刪卡寄存器 0x0047-0x0049: 工號0+卡號),不需現場刷卡。
        card_no = 8 位 hex 字串 (如 E14D0395)。client mode 轉發 daemon。"""
        if self._daemon_url:
            try:
                r = self._session.post(f"{self._daemon_url}/lock/delete_card",
                                       json={"card_no": card_no}, timeout=8)
                return r.json()
            except Exception as e:
                return {"ok": False, "msg": f"刪卡失敗(daemon 無回應): {e}"}
        if not self._lock_enabled:
            return {"ok": False, "msg": "電子鎖未啟用 (LOCK_MODBUS_ADDR 未設)"}
        try:
            cn = str(card_no or "").strip().upper()
            if len(cn) != 8:
                return {"ok": False, "msg": f"卡號格式錯誤(需8位hex): {card_no}"}
            hi = int(cn[:4], 16); lo = int(cn[4:], 16)
            if not LOCK_SERIAL_PORT:
                # THS2:指定刪 FC16 寫不進。改用學習刪——寫 0x2005=0x0055 進刪卡模式,
                # 使用者現場刷該卡才從鎖內白名單刪除(FC06 單寄存器 THS2 寫得進)。
                try:
                    self._lock_write(_LOCK_REG_AUX_ENROLL, _LOCK_AUX_DEL_CARD)
                except Exception as e:
                    return {"ok": False, "msg": f"進刪卡模式失敗: {e}"}
                _log("info", f"電子鎖學習刪卡模式(THS2),待現場刷 {cn}")
                return {"ok": True, "lock_removed": "pending",
                        "msg": f"已進刪卡模式:請在約 10 秒內到鎖上刷「{cn}」這張卡,刷了才會從鎖內白名單刪除(卡片庫記錄已移除)"}
            # USB-485:刪卡寄存器 0x0047=工號(0)/0x0048=卡號高/0x0049=卡號低,FC16 生效
            self._lock_write_multi(_LOCK_REG_DEL_CARD, [0, hi, lo])
            _log("info", f"電子鎖刪卡 {cn}(USB-485)")
            return {"ok": True, "lock_removed": True, "msg": f"已刪除卡號 {cn}(鎖內白名單已移除)"}
        except Exception as e:
            return {"ok": False, "msg": f"刪卡失敗: {e}"}

    def unlock_lock(self) -> dict:
        """遠端開鎖 (寫 0x2004=0x0033)。client mode 轉發 daemon;native 直接寫。"""
        if self._daemon_url:
            self._daemon_post("/lock/unlock", {})
            return {"ok": True, "msg": "已送出遠端開鎖指令"}
        if not self._lock_enabled:
            return {"ok": False, "msg": "電子鎖未啟用 (LOCK_MODBUS_ADDR 未設)"}
        try:
            self._lock_write(_LOCK_REG_UNLOCK, _LOCK_UNLOCK_VAL)
            self._fire_lock_event("unlock", "遠端開鎖")
            _log("info", "電子鎖遠端開鎖")
            return {"ok": True, "msg": "已遠端開鎖"}
        except Exception as e:
            return {"ok": False, "msg": f"遠端開鎖失敗: {e}"}

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
            "lock": {
                "enabled":   self._lock_enabled,
                "addr":      LOCK_ADDR if self._lock_enabled else None,
                "connected": self._lock_connected,
                "error":     self._lock_err,
                "status":    self._lock_status,
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
