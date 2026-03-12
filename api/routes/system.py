"""
Jetson 硬體效能監測 API
讀取 CPU/GPU/記憶體/溫度/磁碟 等即時資訊
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from datetime import datetime
import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo
import threading
import time
import socket
import struct

from api.routes.logs import add_log

router = APIRouter(prefix="/api/system", tags=["系統監測"])
NTP_SETTINGS_PATH = "/workspace/config/system/ntp_settings.json"
TZ_TAIPEI = ZoneInfo("Asia/Taipei")
_ntp_worker_lock = threading.Lock()
_ntp_worker_started = False
_ntp_last_sync: Dict[str, Any] = {"status": "idle", "timestamp": None, "message": ""}
_ntp_last_sync_ts = 0.0
NTP_EPOCH_DELTA = 2208988800  # seconds between 1900 and 1970
NTP_SYNC_OK_OFFSET_SEC = 2.0


class NtpSettings(BaseModel):
    enabled: bool = True
    servers: List[str] = Field(default_factory=lambda: ["time.google.com"])
    sync_interval_minutes: int = 15


def _default_ntp_settings() -> Dict[str, Any]:
    return {
        "enabled": True,
        "servers": ["time.google.com"],
        "sync_interval_minutes": 15,
        "updated_at": None,
    }


def _normalize_ntp_settings(raw: Any) -> Dict[str, Any]:
    d = _default_ntp_settings()
    if isinstance(raw, dict):
        d["enabled"] = bool(raw.get("enabled", d["enabled"]))
        raw_servers = raw.get("servers", d["servers"])
        if isinstance(raw_servers, list):
            servers = [str(s).strip() for s in raw_servers if str(s).strip()]
            if servers:
                d["servers"] = servers[:5]
        try:
            interval = int(raw.get("sync_interval_minutes", d["sync_interval_minutes"]))
            d["sync_interval_minutes"] = max(1, min(1440, interval))
        except Exception:
            pass
        if raw.get("updated_at"):
            d["updated_at"] = str(raw["updated_at"])
    return d


def _load_ntp_settings() -> Dict[str, Any]:
    try:
        p = Path(NTP_SETTINGS_PATH)
        if p.exists():
            return _normalize_ntp_settings(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass
    return _default_ntp_settings()


def _save_ntp_settings(data: Dict[str, Any]) -> None:
    p = Path(NTP_SETTINGS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _query_ntp_server(server: str, timeout_sec: float = 1.5) -> Dict[str, Any]:
    server = str(server or "").strip()
    if not server:
        return {"server": server, "ok": False, "error": "empty_server"}
    packet = b"\x1b" + 47 * b"\0"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout_sec)
    try:
        t_send = time.time()
        s.sendto(packet, (server, 123))
        data, _addr = s.recvfrom(512)
        t_recv = time.time()
        if len(data) < 48:
            return {"server": server, "ok": False, "error": "short_response"}
        sec, frac = struct.unpack("!II", data[40:48])
        ntp_ts = sec + frac / 2**32
        server_unix = ntp_ts - NTP_EPOCH_DELTA
        offset_sec = float(server_unix - t_recv)
        return {
            "server": server,
            "ok": True,
            "offset_sec": round(offset_sec, 3),
            "rtt_ms": round((t_recv - t_send) * 1000.0, 1),
            "stratum": int(data[1]),
        }
    except Exception as e:
        return {"server": server, "ok": False, "error": str(e)}
    finally:
        s.close()


def _probe_ntp_servers(servers: List[str]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for srv in (servers or [])[:5]:
        r = _query_ntp_server(srv)
        results.append(r)
        if not r.get("ok"):
            continue
        if best is None:
            best = r
            continue
        if abs(float(r.get("offset_sec", 9999))) < abs(float(best.get("offset_sec", 9999))):
            best = r
    return {"ok": bool(best), "best": best, "results": results}


def _get_ntp_runtime_status(servers: List[str] | None = None) -> Dict[str, Any]:
    runtime = {"service": "unknown", "synced": None, "source": "", "note": ""}
    try:
        if shutil.which("chronyc"):
            tracking = subprocess.run(
                ["chronyc", "tracking"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            txt = (tracking.stdout or "") + "\n" + (tracking.stderr or "")
            runtime["service"] = "chrony"
            m = re.search(r"Reference ID\s*:\s*(.+)", txt)
            if m:
                runtime["source"] = m.group(1).strip()
            runtime["synced"] = "Not synchronised" not in txt
        elif shutil.which("timedatectl"):
            out = subprocess.run(
                ["timedatectl", "show", "-p", "NTPSynchronized", "-p", "NTP", "--value"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            vals = [v.strip().lower() for v in (out.stdout or "").splitlines() if v.strip()]
            runtime["service"] = "systemd-timesyncd"
            if vals:
                runtime["synced"] = vals[0] == "yes"
            if runtime["synced"] is None:
                runtime["note"] = (out.stderr or "").strip() or "無法判定 NTPSynchronized（可能非 systemd 主機環境）"
        else:
            runtime["note"] = "未偵測到 chronyc/timedatectl"
    except Exception as e:
        runtime["note"] = str(e)
    probe = _probe_ntp_servers(servers or _load_ntp_settings().get("servers", []))
    runtime["probe"] = probe
    if probe.get("ok") and probe.get("best"):
        best = probe["best"]
        runtime["source"] = best.get("server", runtime.get("source", ""))
        runtime["offset_sec"] = best.get("offset_sec")
        runtime["rtt_ms"] = best.get("rtt_ms")
        runtime["synced"] = abs(float(best.get("offset_sec", 9999))) <= NTP_SYNC_OK_OFFSET_SEC
        if runtime.get("note"):
            runtime["note"] = f"{runtime['note']} | probe=ok"
    else:
        runtime["note"] = (runtime.get("note") or "NTP 探測失敗") + " | probe=failed"
    return runtime


def _apply_ntp_servers(servers: List[str]) -> tuple[bool, str]:
    """Best-effort apply NTP servers in runtime environment."""
    try:
        if shutil.which("timedatectl"):
            cfg = "[Time]\nNTP=" + " ".join(servers) + "\nFallbackNTP=pool.ntp.org\n"
            cfg_path = Path("/etc/systemd/timesyncd.conf")
            cfg_path.write_text(cfg, encoding="utf-8")
            subprocess.run(["timedatectl", "set-ntp", "true"], check=False, timeout=3)
            if shutil.which("systemctl"):
                subprocess.run(["systemctl", "restart", "systemd-timesyncd"], check=False, timeout=4)
            return True, "已套用 systemd-timesyncd 設定"
        return False, "環境未提供 timedatectl，僅儲存設定"
    except Exception as e:
        return False, f"套用 NTP 設定失敗: {e}"


def _run_ntp_sync_once(reason: str = "manual") -> Dict[str, Any]:
    global _ntp_last_sync_ts
    settings = _load_ntp_settings()
    if not settings.get("enabled", True):
        result = {
            "status": "skipped",
            "timestamp": datetime.now(TZ_TAIPEI).isoformat(),
            "message": "NTP 同步已停用",
            "reason": reason,
            "runtime": _get_ntp_runtime_status(settings.get("servers", [])),
        }
        _ntp_last_sync.update(result)
        _ntp_last_sync_ts = time.time()
        add_log("info", f"NTP 同步略過（停用） reason={reason}", "system")
        return result

    servers = settings.get("servers", ["time.google.com"])
    ok_apply, apply_msg = _apply_ntp_servers(servers)
    runtime = _get_ntp_runtime_status(servers)
    synced_raw = runtime.get("synced")
    if synced_raw is True:
        status = "success"
    elif synced_raw is False:
        status = "error"
    else:
        status = "warning"
    message = f"{apply_msg} | service={runtime.get('service')} synced={runtime.get('synced')}"
    if runtime.get("note"):
        message += f" | note={runtime.get('note')}"
    result = {
        "status": status,
        "timestamp": datetime.now(TZ_TAIPEI).isoformat(),
        "message": message,
        "reason": reason,
        "runtime": runtime,
    }
    _ntp_last_sync.update(result)
    _ntp_last_sync_ts = time.time()
    if status == "success":
        add_log("success", f"NTP 同步成功 ({reason}) {message}", "system")
    elif status == "error":
        add_log("error", f"NTP 同步失敗 ({reason}) {message}", "system")
    else:
        add_log("warning", f"NTP 同步狀態未知 ({reason}) {message}", "system")
    if not ok_apply and status == "success":
        add_log("warning", f"NTP 設定未完全套用: {apply_msg}", "system")
    return result


def _ensure_ntp_worker():
    global _ntp_worker_started
    with _ntp_worker_lock:
        if _ntp_worker_started:
            return

        def _worker():
            while True:
                try:
                    settings = _load_ntp_settings()
                    interval_sec = int(settings.get("sync_interval_minutes", 15)) * 60
                    interval_sec = max(60, min(86400, interval_sec))
                    now_ts = time.time()
                    if settings.get("enabled", True) and (now_ts - _ntp_last_sync_ts >= interval_sec):
                        _run_ntp_sync_once("scheduled")
                except Exception as e:
                    add_log("error", f"NTP 排程執行失敗: {e}", "system")
                time.sleep(10)

        t = threading.Thread(target=_worker, name="ntp-sync-worker", daemon=True)
        t.start()
        _ntp_worker_started = True
        add_log("info", "NTP 排程服務已啟動", "system")


@router.get("/ntp/settings")
async def get_ntp_settings():
    _ensure_ntp_worker()
    settings = _load_ntp_settings()
    return {
        **settings,
        "runtime": _get_ntp_runtime_status(settings.get("servers", [])),
        "last_sync": _ntp_last_sync,
    }


@router.put("/ntp/settings")
async def update_ntp_settings(data: NtpSettings):
    _ensure_ntp_worker()
    settings = _normalize_ntp_settings(data.dict())
    settings["updated_at"] = datetime.now(TZ_TAIPEI).isoformat()
    _save_ntp_settings(settings)
    add_log(
        "info",
        f"NTP 設定更新: enabled={settings['enabled']} interval={settings['sync_interval_minutes']}m servers={','.join(settings['servers'])}",
        "system",
    )
    sync_result = _run_ntp_sync_once("settings_update")
    return {
        "status": "success",
        "message": "NTP 設定已儲存",
        **settings,
        "runtime": _get_ntp_runtime_status(settings.get("servers", [])),
        "last_sync": sync_result,
    }


@router.post("/ntp/sync-now")
async def ntp_sync_now():
    _ensure_ntp_worker()
    result = _run_ntp_sync_once("manual")
    return {"status": "success", "result": result}


def _read_file(path, default=""):
    """安全讀取檔案"""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return default


def _get_cpu_info():
    """取得 CPU 使用率 (每核心 + 總計)"""
    try:
        with open("/proc/stat", "r") as f:
            lines = f.readlines()

        cores = []
        total_idle = 0
        total_busy = 0

        for line in lines:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            name = parts[0]
            values = list(map(int, parts[1:]))

            idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
            total = sum(values)
            busy = total - idle

            if name == "cpu":
                total_idle = idle
                total_busy = busy
            else:
                cores.append({
                    "core": name,
                    "total": total,
                    "idle": idle,
                })

        # 計算使用率需要兩次取樣，這裡用瞬時值近似
        # 更精確的做法是前端每秒呼叫並計算差值
        return {
            "cores": len(cores),
            "raw": {"total": total_busy + total_idle, "idle": total_idle},
        }
    except Exception:
        return {"cores": 0, "raw": {"total": 0, "idle": 0}}


def _get_cpu_usage():
    """透過 /proc/stat 兩次取樣計算 CPU 使用率"""
    import time

    def read_stat():
        with open("/proc/stat", "r") as f:
            line = f.readline()  # 第一行是總計
        parts = line.split()
        values = list(map(int, parts[1:]))
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return total, idle

    try:
        t1, i1 = read_stat()
        time.sleep(0.1)  # 100ms 取樣間隔
        t2, i2 = read_stat()

        dt = t2 - t1
        di = i2 - i1
        if dt == 0:
            return 0.0
        return round((1.0 - di / dt) * 100, 1)
    except Exception:
        return 0.0


def _get_cpu_freq():
    """取得 CPU 頻率"""
    try:
        freqs = []
        i = 0
        while True:
            path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq"
            if not os.path.exists(path):
                break
            freq_khz = int(_read_file(path, "0"))
            freqs.append(freq_khz / 1000)  # MHz
            i += 1
        if freqs:
            return {"current_mhz": round(sum(freqs) / len(freqs), 0), "cores": len(freqs)}
    except Exception:
        pass
    return {"current_mhz": 0, "cores": 0}


def _get_gpu_info():
    """取得 Jetson GPU 使用率"""
    # Jetson Xavier NX GPU load
    load_paths = [
        "/sys/devices/platform/bus@0/17000000.gpu/load",  # Orin NX
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/gpu.0/load",
        "/sys/devices/17000000.ga10b/load",  # Orin
        "/sys/devices/17000000.gv11b/load",  # Xavier NX
    ]

    gpu_load = None
    for path in load_paths:
        val = _read_file(path)
        if val:
            try:
                gpu_load = round(int(val) / 10.0, 1)  # 值為 0-1000
                break
            except ValueError:
                continue

    # GPU 頻率
    freq_paths = [
        "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq",  # Orin NX
        "/sys/devices/gpu.0/devfreq/gpu.0/cur_freq",
        "/sys/devices/platform/gpu.0/devfreq/gpu.0/cur_freq",
        "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
        "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq",
    ]

    gpu_freq = 0
    for path in freq_paths:
        val = _read_file(path)
        if val:
            try:
                gpu_freq = round(int(val) / 1_000_000)  # Hz -> MHz
                break
            except ValueError:
                continue

    return {
        "usage_percent": gpu_load if gpu_load is not None else -1,
        "freq_mhz": gpu_freq,
    }


def _get_memory_info():
    """取得記憶體使用情況"""
    try:
        with open("/proc/meminfo", "r") as f:
            info = {}
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(":")
                val = int(parts[1])  # kB
                info[key] = val

        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        used = total - available
        swap_total = info.get("SwapTotal", 0)
        swap_free = info.get("SwapFree", 0)
        swap_used = swap_total - swap_free

        return {
            "total_mb": round(total / 1024),
            "used_mb": round(used / 1024),
            "available_mb": round(available / 1024),
            "usage_percent": round(used / total * 100, 1) if total > 0 else 0,
            "swap_total_mb": round(swap_total / 1024),
            "swap_used_mb": round(swap_used / 1024),
        }
    except Exception:
        return {
            "total_mb": 0, "used_mb": 0, "available_mb": 0,
            "usage_percent": 0, "swap_total_mb": 0, "swap_used_mb": 0,
        }


def _get_temperatures():
    """取得各溫度感測器"""
    temps = []
    thermal_base = "/sys/devices/virtual/thermal"

    try:
        for zone in sorted(os.listdir(thermal_base)):
            if not zone.startswith("thermal_zone"):
                continue
            zone_path = os.path.join(thermal_base, zone)
            temp_str = _read_file(os.path.join(zone_path, "temp"))
            type_str = _read_file(os.path.join(zone_path, "type"), zone)

            if temp_str:
                try:
                    temp_c = int(temp_str) / 1000.0
                    if -20 < temp_c < 120:  # 合理範圍
                        temps.append({
                            "zone": type_str,
                            "temp_c": round(temp_c, 1),
                        })
                except ValueError:
                    continue
    except Exception:
        pass

    return temps


def _get_disk_info():
    """取得磁碟使用情況"""
    try:
        stat = os.statvfs("/")
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bfree * stat.f_frsize
        used = total - free
        return {
            "total_gb": round(total / (1024 ** 3), 1),
            "used_gb": round(used / (1024 ** 3), 1),
            "free_gb": round(free / (1024 ** 3), 1),
            "usage_percent": round(used / total * 100, 1) if total > 0 else 0,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0}


def _get_uptime():
    """取得系統運行時間"""
    try:
        uptime_str = _read_file("/proc/uptime")
        seconds = float(uptime_str.split()[0])
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        return {"seconds": int(seconds), "display": f"{days}天 {hours}時 {minutes}分"}
    except Exception:
        return {"seconds": 0, "display": "未知"}


def _get_jetson_power():
    """取得 Jetson 功耗 (INA3221 電流/電壓感測器)"""
    power_info = {}
    ina_base = "/sys/bus/i2c/drivers/ina3221"

    try:
        # 嘗試讀取 tegrastats 風格的功耗
        # 方法 1: /sys/bus/i2c/drivers/ina3221
        if os.path.exists(ina_base):
            for entry in os.listdir(ina_base):
                rail_path = os.path.join(ina_base, entry)
                if not os.path.isdir(rail_path):
                    continue
                for item in os.listdir(rail_path):
                    if "power" in item and item.endswith("_input"):
                        val = _read_file(os.path.join(rail_path, item))
                        if val:
                            power_info[item] = int(val)  # mW

        # 方法 2: hwmon
        hwmon_base = "/sys/class/hwmon"
        if os.path.exists(hwmon_base):
            for hw in os.listdir(hwmon_base):
                hw_path = os.path.join(hwmon_base, hw)
                name = _read_file(os.path.join(hw_path, "name"))
                if "ina" in name.lower():
                    for item in os.listdir(hw_path):
                        if item.startswith("power") and item.endswith("_input"):
                            val = _read_file(os.path.join(hw_path, item))
                            label = _read_file(os.path.join(hw_path, item.replace("_input", "_label")), item)
                            if val:
                                power_info[label] = round(int(val) / 1000, 0)  # uW -> mW
    except Exception:
        pass

    return power_info


def _get_jetson_model():
    """取得 Jetson 型號"""
    # 方法 1: device-tree
    for path in ["/proc/device-tree/model", "/sys/firmware/devicetree/base/model"]:
        model = _read_file(path)
        if model:
            return model.replace("\x00", "").strip()

    # 方法 2: nv_tegra_release
    release = _read_file("/etc/nv_tegra_release")
    if release:
        return "NVIDIA Jetson (" + release.split(",")[0].strip() + ")"

    return "Jetson (unknown)"


@router.get("/status")
async def get_system_status():
    """取得完整系統狀態"""
    cpu_usage = _get_cpu_usage()
    cpu_freq = _get_cpu_freq()
    gpu = _get_gpu_info()
    memory = _get_memory_info()
    temps = _get_temperatures()
    disk = _get_disk_info()
    uptime = _get_uptime()
    power = _get_jetson_power()
    model = _get_jetson_model()

    # 最高溫度
    max_temp = max((t["temp_c"] for t in temps), default=0)

    # 溫度警告等級
    if max_temp >= 80:
        temp_level = "critical"
    elif max_temp >= 65:
        temp_level = "warning"
    else:
        temp_level = "normal"

    return {
        "model": model,
        "timestamp": datetime.now(TZ_TAIPEI).isoformat(),
        "uptime": uptime,
        "cpu": {
            "usage_percent": cpu_usage,
            "cores": cpu_freq["cores"],
            "freq_mhz": cpu_freq["current_mhz"],
        },
        "gpu": gpu,
        "memory": memory,
        "temperatures": temps,
        "max_temp": max_temp,
        "temp_level": temp_level,
        "disk": disk,
        "power": power,
    }
