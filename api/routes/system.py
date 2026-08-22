"""
Jetson 硬體效能監測 API
讀取 CPU/GPU/記憶體/溫度/磁碟 等即時資訊
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from datetime import datetime
import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import threading
import time
import socket
import struct

from api.routes.logs import add_log
from api.routes.auth import get_admin_user, get_current_user

router = APIRouter(prefix="/api/system", tags=["系統監測"])
NTP_SETTINGS_PATH = "/workspace/config/system/ntp_settings.json"
# systemd drop-in 才是最後生效的那份（優先權高於 /etc/systemd/timesyncd.conf）。
# 🛑 檔名要 zz- 開頭：drop-in 依檔名排序讀取，後讀的才蓋得掉先讀的。
#    Jetson 出廠自帶 nv-fallback-ntp.conf（0.pool.ntp.org 等三台外網），
#    叫 field-ntp.conf 會排在 nv- 前面 → 我們清空的 FallbackNTP 又被它加回來。
NTP_DROPIN_PATH = Path(os.getenv("NTP_DROPIN_PATH",
                                 "/etc/systemd/timesyncd.conf.d/zz-field-ntp.conf"))
NX_SETTINGS_PATH = "/workspace/config/system/nx_settings.json"
MONITOR_LAYOUT_PATH = "/workspace/config/system/monitor_layout.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_SYSTEM_CONFIG_DIR = PROJECT_ROOT / "config" / "system"
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


class NxSettings(BaseModel):
    proxy_base_url: str = ""
    server_base_url: str = ""
    username: str = ""
    password: str = ""
    devices_path: str = "/rest/v2/devices"
    media_path_template: str = "/media/{device_id}.{format}"
    timeout_sec: float = 12.0
    verify_ssl: bool = False


def _settings_candidates(path_str: str) -> List[Path]:
    filename = Path(path_str).name
    candidates: List[Path] = []
    env_dir = str(os.getenv("SYSTEM_CONFIG_DIR", "") or "").strip()
    if env_dir:
        candidates.append(Path(env_dir) / filename)
    candidates.append(Path(path_str))
    candidates.append(FALLBACK_SYSTEM_CONFIG_DIR / filename)
    seen = set()
    uniq: List[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


def _load_settings_json(path_str: str) -> Any:
    for path in _settings_candidates(path_str):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    raise FileNotFoundError(path_str)


def _save_settings_json(path_str: str, data: Dict[str, Any]) -> None:
    last_error: Exception | None = None
    for path in _settings_candidates(path_str):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        except Exception as e:
            last_error = e
            continue
    if last_error:
        raise last_error
    raise RuntimeError(f"unable to save settings: {path_str}")


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
        return _normalize_ntp_settings(_load_settings_json(NTP_SETTINGS_PATH))
    except Exception:
        pass
    return _default_ntp_settings()


def _save_ntp_settings(data: Dict[str, Any]) -> None:
    _save_settings_json(NTP_SETTINGS_PATH, data)


def _default_nx_settings() -> Dict[str, Any]:
    verify_ssl = str(os.getenv("NX_VERIFY_SSL", "false")).strip().lower() in {"1", "true", "yes", "on"}
    try:
        timeout_sec = max(3.0, min(120.0, float(os.getenv("NX_HTTP_TIMEOUT", "12"))))
    except Exception:
        timeout_sec = 12.0
    enabled = str(os.getenv("NX_ENABLED", "true")).strip().lower() not in {"0", "false", "no", "off"}
    return {
        # 各站是否有 NX 伺服器。關掉之後 NX 端點一律快速回應,不再對連不到的
        # 主機做連線嘗試 —— 實測連不到時 /api/nx/devices 要卡 60 秒才回 502
        # (多種認證策略各吃一次 12 秒逾時),而前端 NVR 頁一開就會呼叫它。
        "enabled": enabled,
        "proxy_base_url": str(os.getenv("NX_PROXY_BASE_URL", "") or "").strip().rstrip("/"),
        "server_base_url": str(os.getenv("NX_SERVER_BASE_URL", "") or "").strip().rstrip("/"),
        "username": str(os.getenv("NX_USERNAME", "") or "").strip(),
        "password": str(os.getenv("NX_PASSWORD", "") or ""),
        "devices_path": str(os.getenv("NX_DEVICES_PATH", "/rest/v2/devices") or "/rest/v2/devices").strip(),
        "media_path_template": str(
            os.getenv("NX_MEDIA_PATH_TEMPLATE", "/media/{device_id}.{format}") or "/media/{device_id}.{format}"
        ).strip(),
        "timeout_sec": timeout_sec,
        "verify_ssl": verify_ssl,
        "updated_at": None,
    }


def _normalize_nx_path(value: Any, default: str, allow_blank: bool = False) -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        return "" if allow_blank else default
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return text if text.startswith("/") else f"/{text}"


def _normalize_nx_settings(raw: Any) -> Dict[str, Any]:
    data = _default_nx_settings()
    if not isinstance(raw, dict):
        return data
    # 🛑 這個函式是「用預設值重建」,只有列在這裡的欄位會被保留 ——
    # 新增設定欄位一定要同時改 _default_nx_settings 與這裡,否則存了會消失。
    data["enabled"] = bool(raw.get("enabled", data["enabled"]))
    data["proxy_base_url"] = str(raw.get("proxy_base_url", data["proxy_base_url"]) or "").strip().rstrip("/")
    data["server_base_url"] = str(raw.get("server_base_url", data["server_base_url"]) or "").strip().rstrip("/")
    data["username"] = str(raw.get("username", data["username"]) or "").strip()
    data["password"] = str(raw.get("password", data["password"]) or "")
    data["devices_path"] = _normalize_nx_path(raw.get("devices_path", data["devices_path"]), data["devices_path"])
    media_path = str(raw.get("media_path_template", data["media_path_template"]) or "").strip()
    data["media_path_template"] = media_path or data["media_path_template"]
    try:
        timeout_sec = float(raw.get("timeout_sec", data["timeout_sec"]))
        data["timeout_sec"] = max(3.0, min(120.0, timeout_sec))
    except Exception:
        pass
    data["verify_ssl"] = bool(raw.get("verify_ssl", data["verify_ssl"]))
    if raw.get("updated_at"):
        data["updated_at"] = str(raw["updated_at"])
    return data


@router.get("/device-id", summary="終端控制器識別碼（設備編號）")
def get_device_identity():
    """規範 (C) 識別碼設定 —— 每台終端控制器的個別通訊識別碼。

    以軟體控制:`.env` 的 DEVICE_ID 優先;未設定時由板載網卡 MAC 自動產生
    唯一預設值(末 2 bytes = 16 位元,與 16 位元 DIP 開關等價)。
    """
    from api.utils.device_id import device_id_info
    return device_id_info()


def load_nx_settings() -> Dict[str, Any]:
    try:
        return _normalize_nx_settings(_load_settings_json(NX_SETTINGS_PATH))
    except Exception:
        pass
    return _default_nx_settings()


def _save_nx_settings(data: Dict[str, Any]) -> None:
    _save_settings_json(NX_SETTINGS_PATH, data)


def _query_ntp_server(server: str, timeout_sec: float = 1.5) -> Dict[str, Any]:
    server = str(server or "").strip()
    if not server:
        return {"server": server, "ok": False, "error": "empty_server"}
    # 先做 DNS 解析（不算進 rtt），rtt 只記 UDP 真實來回
    try:
        addr_info = socket.getaddrinfo(server, 123, socket.AF_UNSPEC, socket.SOCK_DGRAM)
        if not addr_info:
            return {"server": server, "ok": False, "error": "dns_no_result"}
        family, _, _, _, sockaddr = addr_info[0]
    except Exception as e:
        return {"server": server, "ok": False, "error": f"dns_failed: {e}"}

    packet = b"\x1b" + 47 * b"\0"
    s = socket.socket(family, socket.SOCK_DGRAM)
    s.settimeout(timeout_sec)
    try:
        t_send = time.time()
        s.sendto(packet, sockaddr)
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
    # 先塞預設值，避免 probe 暫時失敗時欄位缺失
    runtime = {
        "service": "unknown",
        "synced": None,
        "source": "",
        "note": "",
        "offset_sec": None,
        "rtt_ms": None,
    }
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
                ["timedatectl", "show", "-p", "NTPSynchronized", "-p", "NTP", "-p", "ServerName", "-p", "ServerAddress", "--value"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            vals = [v.strip() for v in (out.stdout or "").splitlines()]
            runtime["service"] = "systemd-timesyncd"
            # 順序：NTPSynchronized / NTP / ServerName / ServerAddress
            if len(vals) >= 1 and vals[0]:
                runtime["synced"] = vals[0].lower() == "yes"
            # timedatectl 的 ServerName 或 ServerAddress 當成 source
            if len(vals) >= 3 and vals[2]:
                runtime["source"] = vals[2]
            elif len(vals) >= 4 and vals[3]:
                runtime["source"] = vals[3]
            if runtime["synced"] is None:
                runtime["note"] = (out.stderr or "").strip() or "無法判定 NTPSynchronized（可能非 systemd 主機環境）"
        else:
            runtime["note"] = "未偵測到 chronyc/timedatectl"
    except Exception as e:
        runtime["note"] = str(e)
    # 探測所有設定的 servers 取 offset/rtt
    probe = _probe_ntp_servers(servers or _load_ntp_settings().get("servers", []))
    runtime["probe"] = probe
    if probe.get("ok") and probe.get("best"):
        best = probe["best"]
        # probe 的 server 只在 runtime.source 為空時才覆蓋
        if not runtime.get("source"):
            runtime["source"] = best.get("server", "")
        runtime["offset_sec"] = best.get("offset_sec")
        runtime["rtt_ms"] = best.get("rtt_ms")
        if runtime.get("synced") is None:
            runtime["synced"] = abs(float(best.get("offset_sec", 9999))) <= NTP_SYNC_OK_OFFSET_SEC
        if runtime.get("note"):
            runtime["note"] = f"{runtime['note']} | probe=ok"
    else:
        # probe 失敗時保留 timedatectl 的判斷不覆蓋 synced
        if not runtime.get("note"):
            runtime["note"] = "NTP 探測失敗"
        else:
            runtime["note"] = f"{runtime['note']} | probe=failed"
    return runtime


def _write_root_file(path: Path, content: str) -> None:
    """寫入需要 root 的設定檔：先直接寫，沒權限再退回 sudo tee。

    父目錄不一定存在（drop-in 目錄在某些環境是空的），要先建，
    否則 write_text 會丟 FileNotFoundError 而不是 PermissionError，
    走不到 sudo 那條路。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        subprocess.run(["sudo", "-n", "mkdir", "-p", str(path.parent)],
                       capture_output=True, timeout=5)
    try:
        path.write_text(content, encoding="utf-8")
        return
    except PermissionError:
        pass
    p = subprocess.run(["sudo", "-n", "tee", str(path)],
                       input=content.encode("utf-8"),
                       capture_output=True, timeout=5)
    if p.returncode != 0:
        raise PermissionError((p.stderr or b"").decode(errors="ignore")[:120] or "sudo tee 失敗")


def _sudo_if_needed(cmd: List[str]) -> List[str]:
    return cmd if os.geteuid() == 0 else ["sudo", "-n"] + cmd


def _is_systemd_host() -> bool:
    """PID 1 是不是 systemd。容器內為 False —— 這是 sd_booted() 的標準判法。"""
    return os.path.isdir("/run/systemd/system")


def _apply_ntp_servers(servers: List[str]) -> tuple[bool, str]:
    """Best-effort apply NTP servers in runtime environment.

    🛑 一定要寫 drop-in（timesyncd.conf.d/），不可以寫主檔 timesyncd.conf。
       systemd 的 drop-in 優先權高於主檔，現場機有 field-ntp.conf 指到
       現場 NTP（10.41.0.111）。寫主檔會被它整個蓋掉 —— 網頁按了儲存、
       畫面顯示新伺服器，實際生效的還是舊的，是最難察覺的那種假成功。
       2026-08-13 實測：主檔每 10 分鐘被排程重寫成 time.google.com，
       但 timedatectl 一直吃 drop-in 的 10.41.0.111。
    """
    try:
        if not shutil.which("timedatectl"):
            return False, "環境未提供 timedatectl，僅儲存設定"
        # 容器內有 timedatectl 執行檔但 PID 1 不是 systemd，套用一定失敗
        # （"System has not been booted with systemd" / "Failed to connect to bus"）。
        # 這是預期情況不是錯誤：容器的時間本來就跟著 host 走，改設定沒有意義。
        if not _is_systemd_host():
            return False, "非 systemd 環境（容器內），僅儲存設定；時間由 host 負責"
        # FallbackNTP 留空是刻意的：現場封閉網段沒有外網，留著外網 fallback
        # 只會讓 timesyncd 一直去試連不到的位址。空值會「重設」前面 drop-in
        # 累加進來的清單（Jetson 出廠那份會塞 pool.ntp.org），不是「不設定」。
        cfg = ("# 由網頁「硬體效能監測 → 系統時間校時」產生，手改會被覆蓋\n"
               "[Time]\nNTP=" + " ".join(servers) + "\n"
               "FallbackNTP=\n")
        _write_root_file(NTP_DROPIN_PATH, cfg)
        subprocess.run(_sudo_if_needed(["timedatectl", "set-ntp", "true"]),
                       check=False, timeout=3, capture_output=True)
        if shutil.which("systemctl"):
            subprocess.run(_sudo_if_needed(["systemctl", "restart", "systemd-timesyncd"]),
                           check=False, timeout=6, capture_output=True)
        return True, f"已套用 systemd-timesyncd 設定 ({NTP_DROPIN_PATH.name})"
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
        # 容器內套用不了是常態(時間跟著 host 走),每 10 分鐘噴一次 warning 是噪音
        level = "warning" if _is_systemd_host() else "info"
        add_log(level, f"NTP 設定未完全套用: {apply_msg}", "system")
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


def _timesyncd_last_sync() -> str | None:
    """讀 systemd-timesyncd 最後一次成功同步的時間(NTPMessage 的時戳)。讀不到回 None。"""
    try:
        out = subprocess.run(
            ["timedatectl", "show-timesync", "-p", "NTPMessage", "--value"],
            capture_output=True, text=True, timeout=2)
        m = re.search(r"DestinationTimestamp=([^,}]+)", out.stdout or "")
        if m and m.group(1).strip():
            return m.group(1).strip()
    except Exception:
        pass
    return None


@router.get("/ntp/settings")
async def get_ntp_settings():
    _ensure_ntp_worker()
    settings = _load_ntp_settings()
    runtime = _get_ntp_runtime_status(settings.get("servers", []))
    last_sync = dict(_ntp_last_sync)
    # app 自己沒套用過(idle),但 timesyncd 其實已同步 → 顯示真實狀態,別誤導成「沒同步」。
    if last_sync.get("status") in (None, "", "idle") and runtime.get("synced"):
        real_ts = _timesyncd_last_sync()
        last_sync = {"status": "synced", "timestamp": real_ts,
                     "message": "systemd-timesyncd 自動同步中（clock synchronized）"}
    return {
        **settings,
        "runtime": runtime,
        "last_sync": last_sync,
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


@router.get("/nx/settings")
async def get_nx_settings():
    settings = load_nx_settings()
    mode = "proxy" if settings.get("proxy_base_url") else ("direct" if settings.get("server_base_url") else "unconfigured")
    return {
        **settings,
        "mode": mode,
        "configured": bool(settings.get("proxy_base_url") or settings.get("server_base_url")),
    }


@router.put("/nx/settings")
async def update_nx_settings(data: NxSettings):
    settings = _normalize_nx_settings(data.dict())
    settings["updated_at"] = datetime.now(TZ_TAIPEI).isoformat()
    _save_nx_settings(settings)
    mode = "proxy" if settings.get("proxy_base_url") else ("direct" if settings.get("server_base_url") else "unconfigured")
    add_log(
        "info",
        (
            "NX 設定更新: "
            f"mode={mode} "
            f"proxy={settings['proxy_base_url'] or '-'} "
            f"server={settings['server_base_url'] or '-'} "
            f"verify_ssl={settings['verify_ssl']} timeout={settings['timeout_sec']}"
        ),
        "system",
    )
    return {
        "status": "success",
        "message": "NX 設定已儲存",
        **settings,
        "mode": mode,
        "configured": bool(settings.get("proxy_base_url") or settings.get("server_base_url")),
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


def _disk_stat(path: str) -> dict:
    """單一 mount point 容量。失敗回 None。"""
    try:
        stat = os.statvfs(path)
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bfree * stat.f_frsize
        used = total - free
        return {
            "path": path,
            "total_gb": round(total / (1024 ** 3), 1),
            "used_gb": round(used / (1024 ** 3), 1),
            "free_gb": round(free / (1024 ** 3), 1),
            "usage_percent": round(used / total * 100, 1) if total > 0 else 0,
        }
    except Exception:
        return None


def _get_disk_info():
    """取得磁碟使用情況。主回根目錄 (eMMC)，volumes 揭露所有 >10GB 的實體 mount (含 NVMe)。"""
    root = _disk_stat("/") or {"path": "/", "total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0}
    volumes = []
    seen_devs = set()
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                dev, mnt, fstype = parts[0], parts[1], parts[2]
                # 只看實體磁碟 (排除 tmpfs/proc/cgroup/overlay 等)
                if fstype not in ("ext4", "ext3", "xfs", "btrfs", "ntfs", "exfat", "vfat", "f2fs"):
                    continue
                if not dev.startswith("/dev/"):
                    continue
                if dev in seen_devs:
                    continue
                seen_devs.add(dev)
                info = _disk_stat(mnt)
                if info and info["total_gb"] >= 10:
                    info["device"] = dev
                    info["fstype"] = fstype
                    volumes.append(info)
    except Exception:
        pass
    # 回相容欄位 (root 為主) + volumes (含 NVMe 等所有實體磁碟)
    return {
        **{k: v for k, v in root.items() if k != "path"},
        "path": root.get("path", "/"),
        "volumes": volumes,
    }


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


# cache git version 在 module load 時讀一次（避免每次 request 都 fork git）
def _read_git_version() -> Dict[str, str]:
    base = "1.0.0"
    info: Dict[str, str] = {"base": base, "commit": "", "build_date": "", "version": base}
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        if h:
            info["commit"] = h
            info["version"] = f"{base}-{h}"
    except Exception:
        pass
    try:
        d = subprocess.check_output(
            ["git", "log", "-1", "--format=%cd", "--date=short"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        if d:
            info["build_date"] = d
    except Exception:
        pass
    return info

_GIT_VERSION_CACHE = _read_git_version()


@router.get("/version")
def get_version():
    """前後台共用版號 — 兩者用同一個 git commit hash 對齊。"""
    v = _GIT_VERSION_CACHE
    return {
        "frontend": v["version"],
        "backend":  v["version"],
        "base":     v["base"],
        "commit":   v["commit"],
        "build_date": v["build_date"],
    }


@router.post("/restart-api")
def restart_traffic_api():
    """重啟 traffic-api.service。
    UI 在改了 detection_config 等需要 detection thread 重讀的設定後呼叫，
    讓 user 不必 ssh 進 Jetson 自己 systemctl。
    用 start_new_session + DEVNULL 避免被父程序帶走；本 process 會被 systemd
    SIGTERM 後拉新 process 起來。
    """
    add_log("warning", "Web UI 觸發 traffic-api 重啟", "system")
    try:
        subprocess.Popen(
            ["sudo", "-n", "systemctl", "restart", "traffic-api.service"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"ok": True, "message": "重啟指令已發出，服務約 30-60 秒後恢復"}
    except Exception as e:
        add_log("error", f"重啟 traffic-api 失敗: {e}", "system")
        return {"ok": False, "error": str(e)}


# ---- 即時監控畫面配置（攝影機方塊排列順序 + 鎖定）----
# 全系統共用一份:所有使用者看到同一個排列。
# 讀取開放給所有登入者，修改僅限管理員(get_admin_user)。


class MonitorLayoutUpdate(BaseModel):
    order: Optional[List[int]] = None   # 攝影機 id 由左至右、由上而下的排列
    locked: Optional[bool] = None       # True=鎖定，前端停用拖曳


def _default_monitor_layout() -> Dict[str, Any]:
    return {"order": [], "locked": True, "updated_at": None, "updated_by": ""}


def _read_monitor_layout() -> Dict[str, Any]:
    data = _default_monitor_layout()
    try:
        raw = _load_settings_json(MONITOR_LAYOUT_PATH)
    except (FileNotFoundError, ValueError, OSError):
        return data
    if not isinstance(raw, dict):
        return data
    order = raw.get("order")
    if isinstance(order, list):
        seen = set()
        clean: List[int] = []
        for v in order:
            try:
                cid = int(v)
            except (TypeError, ValueError):
                continue
            if cid in seen:      # 去重，避免同一台出現兩次
                continue
            seen.add(cid)
            clean.append(cid)
        data["order"] = clean
    data["locked"] = bool(raw.get("locked", True))
    data["updated_at"] = raw.get("updated_at")
    data["updated_by"] = str(raw.get("updated_by") or "")
    return data


@router.get("/monitor-layout")
def get_monitor_layout(_user=Depends(get_current_user)):
    """取得即時監控的方塊排列。所有登入者都能讀（否則畫面排不出來）。"""
    return _read_monitor_layout()


@router.put("/monitor-layout")
def update_monitor_layout(
    payload: MonitorLayoutUpdate,
    admin=Depends(get_admin_user),
):
    """更新排列或鎖定狀態 —— 僅限管理員。

    order 只存 id 順序，不存攝影機其他資料;新增的攝影機不在 order 裡，
    前端會把它接在已排序的之後，不會消失。
    """
    current = _read_monitor_layout()
    if payload.order is not None:
        seen = set()
        clean: List[int] = []
        for v in payload.order:
            try:
                cid = int(v)
            except (TypeError, ValueError):
                continue
            if cid in seen:
                continue
            seen.add(cid)
            clean.append(cid)
        current["order"] = clean
    if payload.locked is not None:
        current["locked"] = bool(payload.locked)
    current["updated_at"] = datetime.now(TZ_TAIPEI).isoformat()
    current["updated_by"] = str(getattr(admin, "username", "") or "")
    _save_settings_json(MONITOR_LAYOUT_PATH, current)
    add_log(
        "info",
        f"監控畫面配置已更新 (鎖定={current['locked']}, {len(current['order'])} 台) by {current['updated_by']}",
        "system",
    )
    return current
