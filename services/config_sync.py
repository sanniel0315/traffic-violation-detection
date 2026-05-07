"""
遠端設定同步服務 (Option B)

從中央伺服器拉取 config JSON 並套用到本地，不需重啟。
URL 優先順序：網頁設定 (io_settings.json) > .env CONFIG_SYNC_URL
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).parent.parent
_IO_SETTINGS_PATH = PROJECT_ROOT / "config" / "system" / "io_settings.json"

CONFIG_SYNC_TOKEN   = os.getenv("CONFIG_SYNC_TOKEN", "")
CONFIG_SYNC_TIMEOUT = int(os.getenv("CONFIG_SYNC_TIMEOUT", "15"))

# 讀寫 io_settings.json（URL + token 可由網頁覆寫）
def _load_io_settings() -> dict:
    try:
        if _IO_SETTINGS_PATH.exists():
            return json.loads(_IO_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_io_settings(data: dict) -> None:
    _IO_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _IO_SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def get_sync_url() -> str:
    return _load_io_settings().get("sync_url") or os.getenv("CONFIG_SYNC_URL", "")

def get_sync_token() -> str:
    return _load_io_settings().get("sync_token") or CONFIG_SYNC_TOKEN

def save_sync_settings(url: str, token: str = "") -> None:
    cfg = _load_io_settings()
    cfg["sync_url"]   = url.strip()
    cfg["sync_token"] = token.strip()
    _save_io_settings(cfg)

# runtime shorthand (resolved at call time, not import time)
def CONFIG_SYNC_URL() -> str:   # type: ignore[override]
    return get_sync_url()

# 可寫入的 config 檔（key = JSON 欄位名稱，value = 本地路徑）
_CONFIG_FILES = {
    "mqtt_settings":  PROJECT_ROOT / "config" / "system" / "mqtt_settings.json",
    "ntp_settings":   PROJECT_ROOT / "config" / "system" / "ntp_settings.json",
    "nx_settings":    PROJECT_ROOT / "config" / "system" / "nx_settings.json",
    "feature_state":  PROJECT_ROOT / "config" / "system" / "feature_state.json",
}

_lock = threading.Lock()
_running = False

# callback: on_complete(success: bool, message: str)
_on_complete_callbacks: list[Callable[[bool, str], None]] = []


def on_complete(cb: Callable[[bool, str], None]) -> None:
    _on_complete_callbacks.append(cb)


def is_running() -> bool:
    return _running


def trigger() -> bool:
    """非阻塞觸發一次同步，回傳 False 表示已在執行中。"""
    global _running
    with _lock:
        if _running:
            return False
        _running = True
    threading.Thread(target=_run, daemon=True, name="config-sync").start()
    return True


def _run() -> None:
    global _running
    try:
        success, msg = _do_sync()
    except Exception as e:
        success, msg = False, str(e)
    finally:
        _running = False

    print(f"[config_sync] {'OK' if success else 'FAIL'}: {msg}", flush=True)
    for cb in _on_complete_callbacks:
        try:
            cb(success, msg)
        except Exception:
            pass


def _do_sync() -> tuple[bool, str]:
    url   = get_sync_url()
    token = get_sync_token()
    if not url:
        return False, "CONFIG_SYNC_URL 未設定"

    import urllib.request
    import urllib.error

    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=CONFIG_SYNC_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"連線失敗: {e.reason}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"JSON 解析失敗: {e}"

    if not isinstance(data, dict):
        return False, "回傳格式不是 JSON object"

    written = []
    for key, path in _CONFIG_FILES.items():
        if key not in data:
            continue
        section = data[key]
        if not isinstance(section, dict):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # 合併現有設定（不整個覆蓋，保留未提供的欄位）
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(section)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(key)

    if not written:
        return True, "無需更新（伺服器未回傳任何已知 config 欄位）"

    return True, f"已更新: {', '.join(written)}"


def status() -> dict:
    url = get_sync_url()
    return {
        "url_configured": bool(url),
        "url": url or "(未設定)",
        "running": _running,
    }
