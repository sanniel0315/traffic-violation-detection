"""攝影機設定 → Frigate / go2rtc 自動同步。

攝影機在系統裡新增/修改/刪除後，Frigate 的 `go2rtc.streams` 與 `cameras` 必須
跟著變。不同步會出現最難察覺的「半死」狀態：

    traffic-api 偵測 → 讀 DB      → IP 是新的，看起來一切正常
    Frigate 錄影    → 讀 config.yml → IP 還是舊的，全部連不上
    → 畫面有、偵測有，但開單管理點進去沒有影片。

2026-08-12 實測踩過：DB 已經改成 10.149.26.x，config.yml 還停在 10.42.40.x，
而且刪掉的攝影機在 go2rtc 留了殘影一直重試死 IP。

只管理 `cam_<id>` 這種本模組自己產的鍵，其它手動加的 stream 一律不動。

🛑 帳密只能出現在 go2rtc 的 URL，不可以寫進 Frigate 的 camera input
   —— 會被二次編碼（@ → %2540）變成 401。密碼含 @ 也一定要編碼，
   否則會被當成帳密與主機的分隔符。
"""
from __future__ import annotations

import copy
import os
import re
import threading
from typing import Optional
from urllib.parse import quote

import yaml

CONFIG_PATH = os.getenv("FRIGATE_CONFIG_PATH", "/workspace/config/frigate/config.yml")
RESTART_CMD = os.getenv("FRIGATE_RESTART_CMD", "sudo -n systemctl restart traffic-frigate")
# 連續編輯多台時去抖：frigate 重啟會中斷「所有」攝影機錄影，不能每改一台就重啟一次
RESTART_DEBOUNCE_SEC = float(os.getenv("FRIGATE_RESTART_DEBOUNCE_SEC", "8"))

_CAM_KEY = re.compile(r"^cam_(\d+)$")
_lock = threading.Lock()
_restart_timer: Optional[threading.Timer] = None


def _log(level: str, msg: str) -> None:
    try:
        from api.routes.logs import add_log
        add_log(level, msg, "nvr")
    except Exception:
        pass
    print(f"[frigate_sync] {msg}", flush=True)


def stream_url(cam) -> str:
    """由攝影機設定組出 go2rtc 用的 RTSP URL（帳密編碼、路徑補斜線）。"""
    path = (cam.stream_path or "/axis-media/media.amp").strip()
    if not path.startswith("/"):
        path = "/" + path          # 少了前面的斜線 Axis 會回 404
    try:
        port = int(cam.port or 554)
    except (TypeError, ValueError):
        port = 554
    user = quote(cam.username or "", safe="")
    pwd = quote(cam.password or "", safe="")     # 密碼含 @ 必須編碼
    auth = f"{user}:{pwd}@" if (user or pwd) else ""
    return f"rtsp://{auth}{cam.ip}:{port}{path}"


def _camera_template(cfg: dict, key: str) -> dict:
    """新攝影機的 Frigate 區塊：沿用既有那台當範本，沒有就用最小預設。"""
    for name, block in (cfg.get("cameras") or {}).items():
        if name != key and isinstance(block, dict):
            block = copy.deepcopy(block)
            block["zones"] = {}                       # ROI 要現場重畫，沿用會對到錯位置
            block.setdefault("detect", {})["enabled"] = False
            block["ffmpeg"] = {"inputs": [
                {"path": f"rtsp://127.0.0.1:8554/{key}", "roles": ["record"]}]}
            return block
    return {
        # 走本機 go2rtc restream，帳密不進 camera input（會二次編碼變 401）
        "ffmpeg": {"inputs": [{"path": f"rtsp://127.0.0.1:8554/{key}", "roles": ["record"]}]},
        "detect": {"enabled": False},                 # 偵測由我方服務做，不給 Frigate
        "record": {"enabled": True},                  # retain 走全域設定
        "zones": {},
    }


def sync(db=None, restart: bool = True) -> dict:
    """依 DB 的攝影機重建 go2rtc.streams 與 Frigate cameras。回傳異動摘要。

    restart=False 只寫檔不重啟（測試/批次匯入時用）。
    """
    own_db = db is None
    if own_db:
        from api.models import SessionLocal
        db = SessionLocal()
    try:
        from api.models import Camera
        cams = db.query(Camera).order_by(Camera.id).all()

        with _lock:
            if not os.path.exists(CONFIG_PATH):
                _log("error", f"NVR 設定檔不存在，跳過同步: {CONFIG_PATH}")
                return {"ok": False, "msg": "config not found"}
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

            streams = (cfg.setdefault("go2rtc", {}).setdefault("streams", {}) or {})
            cameras = cfg.setdefault("cameras", {})
            before = dict(streams)

            wanted = {}
            for c in cams:
                if not (c.ip or "").strip():
                    continue          # 檔案來源(影片)沒有 RTSP，不產 stream
                wanted[f"cam_{c.id}"] = [stream_url(c)]

            added, updated, removed = [], [], []
            for key, url in wanted.items():
                if key not in streams:
                    added.append(key)
                elif streams[key] != url:
                    updated.append(key)
                streams[key] = url
                if key not in cameras:
                    cameras[key] = _camera_template(cfg, key)

            # 只清本模組管的 cam_<id>，手動加的其它 stream 不動
            for key in list(streams):
                if _CAM_KEY.match(key) and key not in wanted:
                    streams.pop(key)
                    cameras.pop(key, None)
                    removed.append(key)

            changed = bool(added or updated or removed)
            if changed:
                from api.routes.frigate import _safe_write_config
                _safe_write_config(CONFIG_PATH, cfg)
                _log("info", "NVR 設定已同步 "
                             f"(新增 {added or '-'} / 更新 {updated or '-'} / 移除 {removed or '-'})")
            summary = {"ok": True, "changed": changed, "added": added,
                       "updated": updated, "removed": removed,
                       "streams": sorted(wanted), "before": sorted(before)}

        if changed and restart:
            _schedule_restart()
        return summary
    except Exception as e:
        _log("error", f"NVR 設定同步失敗: {e}")
        return {"ok": False, "msg": str(e)}
    finally:
        if own_db:
            try:
                db.close()
            except Exception:
                pass


def _schedule_restart() -> None:
    """去抖重啟：連續改好幾台只會在最後一次之後重啟一次。"""
    global _restart_timer
    with _lock:
        if _restart_timer is not None:
            _restart_timer.cancel()
        _restart_timer = threading.Timer(RESTART_DEBOUNCE_SEC, _do_restart)
        _restart_timer.daemon = True
        _restart_timer.start()
    _log("info", f"NVR 將於 {RESTART_DEBOUNCE_SEC:.0f} 秒後套用新設定（重啟）")


def _do_restart() -> None:
    import subprocess
    try:
        r = subprocess.run(RESTART_CMD.split(), capture_output=True, timeout=120)
        if r.returncode == 0:
            _log("info", "NVR 已重啟，新攝影機設定生效")
        else:
            _log("error", f"NVR 重啟失敗 (rc={r.returncode}): "
                          f"{(r.stderr or b'').decode(errors='ignore')[:200]}")
    except Exception as e:
        _log("error", f"NVR 重啟失敗: {e}")


def sync_async(db=None) -> None:
    """給 API 路由用的射後不理版本：同步失敗絕不能讓攝影機新增/修改本身失敗。
    另開 thread 是因為要重讀設定檔 + 寫檔，不該卡住 HTTP 回應。"""
    threading.Thread(target=sync, kwargs={"db": None}, daemon=True,
                     name="frigate-sync").start()
