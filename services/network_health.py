"""
網路健康監控 — 驅動 IOService 的通訊故障旗標。

故障條件（任一成立 → DO0 紅燈亮）：
  1. NTP 未同步
  2. 主網卡未接線（operstate != up）
  3. 主網卡無 IP
"""
from __future__ import annotations

import os
import subprocess
import threading

_stop   = threading.Event()
_thread: threading.Thread | None = None


def _log(level: str, msg: str) -> None:
    try:
        from api.routes.logs import add_log
        add_log(level, msg, "io")
    except Exception:
        pass


def _check_ntp() -> bool:
    try:
        out = subprocess.check_output(
            ["timedatectl", "status"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode()
        return (
            "System clock synchronized: yes" in out
            or "NTP synchronized: yes" in out
        )
    except Exception:
        return True  # 無法判斷時不報錯


def _get_primary_iface() -> str:
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
    return os.getenv("NET_IFACE", "eth0")


def _check_link(iface: str) -> bool:
    try:
        state = open(f"/sys/class/net/{iface}/operstate").read().strip()
        return state == "up"
    except Exception:
        # 看不到 iface (例如 docker container 內的 sysfs 路徑差異) 時不誤報故障
        return True


def _check_ip(iface: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ip", "-4", "addr", "show", iface],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode()
        return "inet " in out
    except Exception:
        return True


def _monitor_loop(interval: int) -> None:
    from services.io_service import get_service

    prev_fault: bool | None = None

    while not _stop.is_set():
        iface   = _get_primary_iface()
        ntp_ok  = _check_ntp()
        link_ok = _check_link(iface)
        ip_ok   = _check_ip(iface)

        fault = not (ntp_ok and link_ok and ip_ok)

        if fault != prev_fault:
            try:
                get_service().set_comm_fault(fault)
            except Exception:
                pass

            if fault:
                reasons = []
                if not ntp_ok:  reasons.append("NTP 未同步")
                if not link_ok: reasons.append(f"{iface} 未接線")
                if not ip_ok:   reasons.append(f"{iface} 無 IP")
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
    print("[net_health] started", flush=True)


def stop() -> None:
    _stop.set()
