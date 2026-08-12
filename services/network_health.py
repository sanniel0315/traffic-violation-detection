"""
網路健康監控 — 驅動 IOService 的通訊故障旗標（DO0 紅燈）。

故障條件（任一成立 → 紅燈亮，**持續亮到全部恢復才熄**）：
  1. 監看的網卡不存在
  2. 未接線（operstate != up）
  3. 沒有 IP

監看哪一張：環境變數 `NET_IFACE`（逗號可指定多張，全部正常才算正常）。
現場**一定要設**——這台有兩張網卡（現場網路 / 維運上網），通訊故障燈是給
客戶看的，不能跟著維運那條跑。未設時退回「當下預設路由那張」並記住它，
只是為了沒設定也能動。

註：NTP 不納入紅燈判定 — NTP server 不可達 / 跨網路同步延遲常導致
誤報，且時間同步問題本質不是「網路連線異常」。NTP 異常另由 system
log 觀察，不亮紅燈。

🛑 兩個會讓「斷網亮紅燈」驗收當場掛掉的坑（2026-08-12 實測後修正）：
  a. 監看對象跟著 default route 跑 → 拔「現場網路」時預設路由還在維運那張，
     紅燈根本不會亮。
  b. 拔線的瞬間 default route 會一起消失 → 舊碼退回猜 `eth0`（機器上不存在），
     而兩個檢查都寫成「讀不到就當正常」→ fault 變 False，紅燈亮 8 秒自己熄。
     實測 log：09:56:38 通訊故障、09:56:46 通訊恢復正常（線根本沒插回去）。
  → 現在：監看對象釘死不漂移；網卡讀不到一律當故障，不當正常。
"""
from __future__ import annotations

import os
import subprocess
import threading

_stop   = threading.Event()
_thread: threading.Thread | None = None
_last_iface = ""      # 記住上次的預設路由網卡:斷線時 default route 會消失,不能改猜別的


def _log(level: str, msg: str) -> None:
    try:
        from api.routes.logs import add_log
        add_log(level, msg, "io")
    except Exception:
        pass


def _default_route_iface() -> str:
    """目前預設路由走哪張網卡;沒有預設路由回空字串。"""
    try:
        out = subprocess.check_output(
            ["ip", "route", "show", "default"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode()
        parts = out.split()
        if "dev" in parts:
            return parts[parts.index("dev") + 1]
    except Exception:
        pass
    return ""


def _monitored_ifaces() -> list:
    """要監看的網卡清單。NET_IFACE 優先(現場建議明確指定);
    否則用當下預設路由那張,並記住 —— 斷線時它會消失,這時要沿用記住的那張
    繼續判定為故障,不能退回猜一個不存在的名字(那會讓紅燈自己熄)。"""
    global _last_iface
    env = os.getenv("NET_IFACE", "").strip()
    if env:
        return [i.strip() for i in env.split(",") if i.strip()]
    dev = _default_route_iface()
    if dev:
        _last_iface = dev
    return [_last_iface] if _last_iface else []


def _check_link(iface: str) -> bool:
    """網卡是否已接線。讀不到 = 網卡不存在 = 故障(不是正常)。"""
    try:
        state = open(f"/sys/class/net/{iface}/operstate").read().strip()
        return state == "up"
    except Exception:
        return False


def _check_ip(iface: str) -> bool:
    """網卡是否有 IPv4。讀不到 = 故障。"""
    try:
        out = subprocess.check_output(
            ["ip", "-4", "addr", "show", iface],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode()
        return "inet " in out
    except Exception:
        return False


def _evaluate() -> tuple:
    """回傳 (是否故障, 原因清單)。"""
    ifaces = _monitored_ifaces()
    if not ifaces:
        # 連要看哪張都判斷不出來(沒設 NET_IFACE 且從未有過預設路由) → 當故障,
        # 不能因為「不知道」就報正常。
        return True, ["找不到可監看的網卡 (未設 NET_IFACE 且無預設路由)"]
    reasons = []
    for iface in ifaces:
        if not _check_link(iface):
            reasons.append(f"{iface} 未接線")
        elif not _check_ip(iface):
            reasons.append(f"{iface} 無 IP")
    return bool(reasons), reasons


def _monitor_loop(interval: int) -> None:
    from services.io_service import get_service

    prev_fault: bool | None = None

    while not _stop.is_set():
        fault, reasons = _evaluate()

        if fault != prev_fault:
            try:
                get_service().set_comm_fault(fault)
            except Exception:
                pass

            if fault:
                _log("error", f"通訊故障: {', '.join(reasons)}")
            else:
                _log("info", "通訊恢復正常")

            prev_fault = fault

        _stop.wait(interval)


def start(interval: int = 30) -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(
        target=_monitor_loop, args=(interval,),
        daemon=True, name="net-health",
    )
    _thread.start()
    print(f"[net_health] started (監看: {_monitored_ifaces() or '未定'})", flush=True)


def stop() -> None:
    _stop.set()
