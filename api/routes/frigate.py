#!/usr/bin/env python3
"""NVR 錄影管理 API"""
from fastapi import APIRouter, HTTPException, Response, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import yaml
import os
import requests
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from api.routes.logs import add_log

router = APIRouter(prefix="/api/frigate", tags=["NVR"])

# NVR 設定檔路徑
FRIGATE_CONFIG_PATH = "/workspace/config/frigate/config.yml"

# NX 串流路徑前綴（Frigate 無法直接讀取）
_NX_STREAM_PREFIX = "/api/nx/stream/"


def _sanitize_frigate_config(config: dict) -> dict:
    """修正 Frigate 0.17 相容性問題，避免 safe mode。"""
    cameras = config.get("cameras")
    if not isinstance(cameras, dict):
        return config

    nx_cams_to_remove = []
    for name, cam in cameras.items():
        if not isinstance(cam, dict):
            continue

        # NX 串流攝影機無法由 Frigate 錄影，移除
        inputs = cam.get("ffmpeg", {}).get("inputs", [])
        is_nx = any(_NX_STREAM_PREFIX in str(inp.get("path", "")) for inp in inputs) if inputs else False
        if is_nx:
            nx_cams_to_remove.append(name)
            continue

        # 關閉 detect（避免 onnxruntime crash）
        if "detect" in cam:
            cam["detect"]["enabled"] = False

        # 移除 camera 層級的 retain（0.17 不支援）
        rec = cam.get("record", {})
        if isinstance(rec, dict) and "retain" in rec:
            del rec["retain"]

        # 移除 snapshots 層級的 retain
        snap = cam.get("snapshots", {})
        if isinstance(snap, dict) and "retain" in snap:
            del snap["retain"]

        # record 預設啟用
        if "record" not in cam:
            cam["record"] = {"enabled": True}
        elif not cam["record"].get("enabled"):
            cam["record"]["enabled"] = True

        # roles 確保包含 record
        for inp in cam.get("ffmpeg", {}).get("inputs", []):
            roles = inp.get("roles", [])
            if "record" not in roles:
                roles.append("record")
            # 移除 detect role（不需要 Frigate 偵測）
            if "detect" in roles:
                roles.remove("detect")
            inp["roles"] = roles

    # 移除 NX 串流攝影機
    for name in nx_cams_to_remove:
        del cameras[name]

    # 全域 detect 關閉
    config["detect"] = {"enabled": False}

    # 確保 detectors 有 cpu1（避免 labelmap 錯誤）
    if not config.get("detectors"):
        config["detectors"] = {"cpu1": {"type": "cpu"}}

    # 修正全域 record 格式
    rec = config.get("record", {})
    if isinstance(rec, dict) and "retain" in rec:
        del rec["retain"]
    if "continuous" not in rec:
        rec["continuous"] = {"days": 3}
    if "motion" not in rec:
        rec["motion"] = {"days": 7}
    rec["enabled"] = True
    config["record"] = rec

    return config
NVR_UI_SETTINGS_PATH = "/workspace/config/frigate/ui_settings.json"
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "frigate")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))
FRIGATE_URL = f"http://{FRIGATE_HOST}:{FRIGATE_PORT}"
_nvr_last_status: Optional[str] = None


def _frigate_base_urls() -> List[str]:
    """Build candidate Frigate base URLs to avoid host-name specific false offline."""
    candidates: List[str] = []
    host = (FRIGATE_HOST or "").strip()

    if host.startswith("http://") or host.startswith("https://"):
        candidates.append(host.rstrip("/"))
    elif host:
        candidates.append(f"http://{host}:{FRIGATE_PORT}")

    for fallback_host in ("localhost", "127.0.0.1", "host.docker.internal"):
        candidates.append(f"http://{fallback_host}:{FRIGATE_PORT}")

    uniq: List[str] = []
    seen = set()
    for base in candidates:
        if base not in seen:
            seen.add(base)
            uniq.append(base)
    return uniq


def _get_frigate(path: str, timeout: int = 5, params: Optional[Dict[str, Any]] = None):
    """Try Frigate GET across candidate base URLs; return (response, url, error)."""
    last_error = None
    p = path if path.startswith("/") else f"/{path}"
    for base in _frigate_base_urls():
        url = f"{base}{p}"
        try:
            response = requests.get(url, timeout=timeout, params=params)
            return response, url, None
        except Exception as e:
            last_error = e
    return None, None, last_error


def _parse_time_to_epoch(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if text.isdigit():
            return float(int(text))
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _list_frigate_camera_names() -> List[str]:
    response, _url, _err = _get_frigate("/api/stats", timeout=8)
    if response is not None and response.status_code == 200:
        data = response.json() or {}
        cams = data.get("cameras", {})
        if isinstance(cams, dict):
            return [str(k) for k in cams.keys() if str(k).strip()]
    return []


def _default_record_schedule() -> List[List[bool]]:
    return [[True for _ in range(24)] for _ in range(7)]


def _normalize_record_schedule(schedule: Any) -> List[List[bool]]:
    """Normalize schedule to 7x24 bool matrix."""
    normalized = _default_record_schedule()
    if not isinstance(schedule, list):
        return normalized
    for d in range(min(7, len(schedule))):
        row = schedule[d]
        if not isinstance(row, list):
            continue
        for h in range(min(24, len(row))):
            normalized[d][h] = bool(row[h])
    return normalized


def _load_ui_settings() -> Dict[str, Any]:
    try:
        p = Path(NVR_UI_SETTINGS_PATH)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_ui_settings(data: Dict[str, Any]) -> None:
    p = Path(NVR_UI_SETTINGS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _camera_detect_size(cam_cfg: Dict[str, Any]) -> tuple[int, int]:
    detect_cfg = cam_cfg.get("detect", {}) if isinstance(cam_cfg, dict) else {}
    w = int(detect_cfg.get("width", 1920) or 1920)
    h = int(detect_cfg.get("height", 1080) or 1080)
    return max(1, w), max(1, h)


def _parse_motion_mask_points(mask_value: Any, width: int, height: int) -> List[List[int]]:
    polygons = _parse_motion_mask_polygons(mask_value, width, height)
    return polygons[0] if polygons else []


def _parse_motion_mask_polygons(mask_value: Any, width: int, height: int) -> List[List[List[int]]]:
    if not mask_value:
        return []
    raw_items = mask_value if isinstance(mask_value, list) else [mask_value]
    polygons: List[List[List[int]]] = []
    for raw in raw_items:
        text = str(raw or "")
        nums: List[float] = []
        for seg in text.split(","):
            s = seg.strip()
            if not s:
                continue
            try:
                nums.append(float(s))
            except Exception:
                pass
        if len(nums) < 6:
            continue
        pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        is_normalized = any((0 <= x <= 1 and 0 <= y <= 1) for x, y in pairs) and max(
            max(x for x, _ in pairs), max(y for _, y in pairs)
        ) <= 1.05
        points: List[List[int]] = []
        for x, y in pairs:
            if is_normalized:
                px = int(round(x * width))
                py = int(round(y * height))
            else:
                px = int(round(x))
                py = int(round(y))
            points.append([max(0, px), max(0, py)])
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def _format_motion_mask(points: Any, width: int, height: int) -> Any:
    polygons: List[List[List[float]]] = []
    if isinstance(points, list) and points:
        first = points[0]
        if isinstance(first, list) and len(first) >= 2 and isinstance(first[0], (int, float)):
            polygons = [points]
        elif isinstance(first, list):
            polygons = points
    w = max(1, int(width))
    h = max(1, int(height))
    encoded: List[str] = []
    for poly in polygons:
        coords: List[str] = []
        for p in poly:
            if not isinstance(p, list) or len(p) < 2:
                continue
            x = max(0.0, min(float(p[0]), float(w)))
            y = max(0.0, min(float(p[1]), float(h)))
            nx = x / float(w)
            ny = y / float(h)
            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")
        if len(coords) >= 6:
            encoded.append(",".join(coords))
    if not encoded:
        return ""
    return encoded if len(encoded) > 1 else encoded[0]


def _sanitize_polygon(points: List[List[float]], width: int, height: int) -> List[List[int]]:
    out: List[List[int]] = []
    w = max(1, int(width))
    h = max(1, int(height))
    for p in points or []:
        if not isinstance(p, list) or len(p) < 2:
            continue
        x = int(round(max(0.0, min(float(p[0]), float(w)))))
        y = int(round(max(0.0, min(float(p[1]), float(h)))))
        out.append([x, y])
    return out


def _outer_masks_from_include(include_points: List[List[int]], width: int, height: int) -> List[List[List[int]]]:
    # Frigate motion 原生只支援 mask；以 include 外接框近似生成外側遮罩。
    if not include_points or len(include_points) < 3:
        return []
    w = max(1, int(width))
    h = max(1, int(height))
    xs = [int(p[0]) for p in include_points]
    ys = [int(p[1]) for p in include_points]
    min_x = max(0, min(xs))
    max_x = min(w, max(xs))
    min_y = max(0, min(ys))
    max_y = min(h, max(ys))
    polys: List[List[List[int]]] = []
    if min_y > 0:
        polys.append([[0, 0], [w, 0], [w, min_y], [0, min_y]])
    if max_y < h:
        polys.append([[0, max_y], [w, max_y], [w, h], [0, h]])
    if min_x > 0 and max_y > min_y:
        polys.append([[0, min_y], [min_x, min_y], [min_x, max_y], [0, max_y]])
    if max_x < w and max_y > min_y:
        polys.append([[max_x, min_y], [w, min_y], [w, max_y], [max_x, max_y]])
    return polys


class NvrSettings(BaseModel):
    """NVR 設定"""
    enabled: bool = True
    host: str = "frigate"
    port: int = 5000
    # MQTT
    mqtt_enabled: bool = False
    mqtt_host: Optional[str] = None
    mqtt_port: int = 1883
    mqtt_user: Optional[str] = None
    mqtt_password: Optional[str] = None
    # 錄影模式
    record_mode: str = "motion"  # all, motion, off
    retain_days: int = 7
    event_retain_days: int = 14
    # 每週錄影排程（7 x 24, bool）
    record_schedule: List[List[bool]] = Field(default_factory=_default_record_schedule)
    # 動態偵測
    motion_enabled: bool = True
    motion_threshold: int = 25
    motion_contour_area: int = 20
    detect_fps: int = 5
    detect_resolution: str = "1920x1080"


class FrigateCameraConfig(BaseModel):
    """攝影機設定"""
    name: str
    enabled: bool = True
    rtsp_url: str
    width: int = 1920
    height: int = 1080
    fps: int = 10
    detect_objects: List[str] = ["car", "motorcycle", "person"]
    record_enabled: bool = True
    record_retain_days: int = 7
    snapshot_enabled: bool = True
    zones: List[Dict[str, Any]] = []


class FrigateCameraAdd(BaseModel):
    """新增攝影機"""
    name: str
    rtsp_url: str
    record: bool = True
    detect: bool = True
    snapshots: bool = True


class NvrCameraSwitch(BaseModel):
    """單台攝影機錄影/偵測切換"""
    record_enabled: Optional[bool] = None
    detect_enabled: Optional[bool] = None
    motion_enabled: Optional[bool] = None
    snapshots_enabled: Optional[bool] = None


class NvrMotionRoiUpdate(BaseModel):
    width: int = 1920
    height: int = 1080
    mode: str = "exclude"
    points: List[List[float]] = Field(default_factory=list)
    include_points: List[List[float]] = Field(default_factory=list)
    exclude_points: List[List[float]] = Field(default_factory=list)


# ============ 狀態 API ============

@router.get("/status")
async def get_nvr_status():
    """取得 NVR 連線狀態"""
    global _nvr_last_status
    try:
        response, _url, _err = _get_frigate("/api/stats", timeout=5)
        if response is not None and response.status_code == 200:
            stats = response.json()
            if _nvr_last_status != "online":
                if _nvr_last_status is None:
                    add_log("info", "NVR 連線狀態：online", "nvr")
                else:
                    add_log("success", "NVR 已恢復連線（online）", "nvr")
                _nvr_last_status = "online"
            return {
                "status": "online",
                "version": stats.get("service", {}).get("version", "unknown"),
                "uptime": stats.get("service", {}).get("uptime", 0),
                "cameras": list(stats.get("cameras", {}).keys()),
                "detectors": stats.get("detectors", {})
            }
    except Exception:
        pass
    if _nvr_last_status != "offline":
        if _nvr_last_status is not None:
            add_log("warning", "NVR 連線中斷（offline）", "nvr")
        else:
            add_log("warning", "NVR 目前離線（offline）", "nvr")
        _nvr_last_status = "offline"
    return {"status": "offline", "message": "無法連線到 NVR"}


@router.get("/version")
async def get_nvr_version():
    """取得 NVR 版本"""
    try:
        response, _url, _err = _get_frigate("/api/version", timeout=5)
        if response is not None and response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"version": "unknown", "status": "offline"}


# ============ 設定 API ============

@router.get("/config")
async def get_nvr_config():
    """取得 NVR 設定"""
    try:
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
                return {"status": "success", "config": config}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "設定檔不存在"}


@router.post("/config")
async def save_nvr_config(config: Dict[str, Any]):
    """儲存 NVR 設定"""
    try:
        os.makedirs(os.path.dirname(FRIGATE_CONFIG_PATH), exist_ok=True)
        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)
        return {"status": "success", "message": "設定已儲存，請重啟 NVR 生效"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings")
async def get_nvr_settings():
    """取得 NVR 連線與錄影設定"""
    try:
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}

            mqtt = config.get("mqtt", {})

            # 從第一台攝影機讀取錄影和偵測設定
            cameras = config.get("cameras", {})
            first_cam = next(iter(cameras.values()), {}) if cameras else {}

            # 判斷錄影模式
            record_cfg = first_cam.get("record", {})
            record_enabled = record_cfg.get("enabled", False)
            record_mode_str = record_cfg.get("retain", {}).get("mode", "motion")
            if not record_enabled:
                record_mode = "off"
            elif record_mode_str == "all":
                record_mode = "all"
            else:
                record_mode = "motion"

            # 偵測設定
            detect_cfg = first_cam.get("detect", {})
            detect_w = detect_cfg.get("width", 1920)
            detect_h = detect_cfg.get("height", 1080)
            detect_resolution = f"{detect_w}x{detect_h}"

            # Motion 設定
            motion_cfg = first_cam.get("motion", {})
            ui_settings = _load_ui_settings()
            record_schedule = _normalize_record_schedule(ui_settings.get("record_schedule"))

            return {
                "enabled": True,
                "host": FRIGATE_HOST,
                "port": FRIGATE_PORT,
                "mqtt_enabled": mqtt.get("enabled", False),
                "mqtt_host": mqtt.get("host"),
                "mqtt_port": mqtt.get("port", 1883),
                "mqtt_user": mqtt.get("user"),
                "mqtt_password": "",
                "record_mode": record_mode,
                "retain_days": record_cfg.get("retain", {}).get("days", 7),
                "event_retain_days": record_cfg.get("events", {}).get("retain", {}).get("default", 14),
                "record_schedule": record_schedule,
                "motion_enabled": motion_cfg.get("enabled", True),
                "motion_threshold": motion_cfg.get("threshold", 25),
                "motion_contour_area": motion_cfg.get("contour_area", 20),
                "detect_fps": detect_cfg.get("fps", 5),
                "detect_resolution": detect_resolution,
            }
    except Exception:
        pass
    return NvrSettings().dict()


@router.put("/settings")
async def update_nvr_settings(settings: NvrSettings):
    """更新 NVR 設定（錄影模式、動態偵測、MQTT）"""
    try:
        config = {}
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}

        # MQTT 設定
        if settings.mqtt_enabled:
            config["mqtt"] = {
                "enabled": True,
                "host": settings.mqtt_host,
                "port": settings.mqtt_port,
                "user": settings.mqtt_user,
                "password": settings.mqtt_password,
            }
        else:
            config["mqtt"] = {"enabled": False}

        # 保存 UI 錄影排程（不寫入 Frigate config，避免 schema 驗證失敗）
        ui_settings = _load_ui_settings()
        ui_settings["record_schedule"] = _normalize_record_schedule(settings.record_schedule)
        _save_ui_settings(ui_settings)
        # 清理舊版殘留欄位，避免 Frigate 無法啟動
        config.pop("record_schedule", None)

        # 解析偵測解析度
        try:
            w, h = settings.detect_resolution.split("x")
            detect_width = int(w)
            detect_height = int(h)
        except Exception:
            detect_width, detect_height = 1920, 1080

        # 錄影模式設定
        record_enabled = settings.record_mode != "off"
        record_retain_mode = "all" if settings.record_mode == "all" else "motion"

        # 更新每台攝影機的錄影和偵測設定
        for cam_name, cam_config in config.get("cameras", {}).items():
            # 錄影
            cam_config["record"] = {
                "enabled": record_enabled,
                "retain": {
                    "days": settings.retain_days,
                    "mode": record_retain_mode,
                },
            }
            # 兼容舊配置：移除 Frigate 不接受的 record.events
            if isinstance(cam_config.get("record"), dict):
                cam_config["record"].pop("events", None)

            # 偵測
            detect = cam_config.get("detect", {})
            detect["fps"] = settings.detect_fps
            detect["width"] = detect_width
            detect["height"] = detect_height
            cam_config["detect"] = detect

            # Motion 動態偵測：保留既有 mask 等欄位，避免覆蓋 ROI
            motion = cam_config.get("motion", {}) or {}
            motion["enabled"] = bool(settings.motion_enabled)
            motion["threshold"] = int(settings.motion_threshold)
            motion["contour_area"] = int(settings.motion_contour_area)
            cam_config["motion"] = motion

            # 確保 ffmpeg roles 正確
            inputs = cam_config.get("ffmpeg", {}).get("inputs", [])
            if inputs:
                roles = inputs[0].get("roles", [])
                if record_enabled and "record" not in roles:
                    roles.append("record")
                elif not record_enabled and "record" in roles:
                    roles.remove("record")
                if "detect" not in roles:
                    roles.append("detect")
                inputs[0]["roles"] = roles

        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)

        return {"status": "success", "message": "設定已更新，請重啟 NVR 生效"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 攝影機 API ============

@router.get("/cameras")
async def get_nvr_cameras():
    """取得 NVR 攝影機列表"""
    cameras = []

    # 即時數據
    live_stats = {}
    try:
        response = requests.get(f"{FRIGATE_URL}/api/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            for name, info in stats.get("cameras", {}).items():
                live_stats[name] = {
                    "fps": info.get("camera_fps", 0),
                    "detection_fps": info.get("detection_fps", 0),
                    "process_fps": info.get("process_fps", 0),
                    "pid": info.get("pid", 0),
                }
    except Exception:
        pass

    # config.yml 為主要來源
    try:
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}
                for name, cam_config in config.get("cameras", {}).items():
                    cam = {
                        "name": name,
                        "enabled": cam_config.get("enabled", True),
                        "rtsp_url": cam_config.get("ffmpeg", {})
                        .get("inputs", [{}])[0]
                        .get("path", ""),
                        "config": cam_config,
                        "fps": 0,
                        "detection_fps": 0,
                        "process_fps": 0,
                        "pid": 0,
                    }
                    if name in live_stats:
                        cam.update(live_stats[name])
                    cameras.append(cam)
    except Exception:
        pass

    return {"cameras": cameras}


@router.put("/cameras/{camera_name}")
async def update_nvr_camera(camera_name: str, camera: FrigateCameraConfig):
    """更新攝影機"""
    try:
        if not os.path.exists(FRIGATE_CONFIG_PATH):
            raise HTTPException(status_code=404, detail="設定檔不存在")

        with open(FRIGATE_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f) or {}

        if camera_name not in config.get("cameras", {}):
            raise HTTPException(status_code=404, detail="攝影機不存在")

        cam_config = config["cameras"][camera_name]
        cam_config["enabled"] = camera.enabled
        cam_config["ffmpeg"]["inputs"][0]["path"] = camera.rtsp_url
        cam_config["detect"] = {
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
        }
        cam_config["objects"]["track"] = camera.detect_objects
        cam_config["record"]["enabled"] = camera.record_enabled
        cam_config["record"]["retain"]["days"] = camera.record_retain_days
        cam_config["snapshots"]["enabled"] = camera.snapshot_enabled

        if camera.zones:
            cam_config["zones"] = {}
            for zone in camera.zones:
                cam_config["zones"][zone["name"]] = {
                    "coordinates": zone["coordinates"],
                    "objects": zone.get("objects", ["car", "motorcycle"]),
                }

        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)

        return {"status": "success", "message": f"攝影機 {camera_name} 已更新"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cameras/{camera_name}")
async def delete_nvr_camera(camera_name: str):
    """刪除攝影機"""
    try:
        if not os.path.exists(FRIGATE_CONFIG_PATH):
            raise HTTPException(status_code=404, detail="設定檔不存在")

        with open(FRIGATE_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f) or {}

        if camera_name in config.get("cameras", {}):
            del config["cameras"][camera_name]
            with open(FRIGATE_CONFIG_PATH, 'w') as f:
                yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)
            return {"status": "success", "message": f"攝影機 {camera_name} 已刪除"}

        raise HTTPException(status_code=404, detail="攝影機不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 事件 API ============

@router.get("/events")
async def get_nvr_events(
    camera: Optional[str] = None,
    label: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    all_pages: bool = Query(False),
    limit: int = Query(2000, ge=1, le=50000),
    batch_size: int = Query(300, ge=50, le=1000),
):
    """取得偵測事件"""
    try:
        after = _parse_time_to_epoch(start_time)
        before = _parse_time_to_epoch(end_time)
        if after and before and after > before:
            raise HTTPException(status_code=400, detail="start_time 不可晚於 end_time")

        common_params: Dict[str, Any] = {}
        if camera:
            common_params["camera"] = camera
        if label:
            common_params["label"] = label
        if after is not None:
            common_params["after"] = int(after)
        if before is not None:
            common_params["before"] = int(before)

        if not all_pages:
            params = {**common_params, "limit": min(int(limit), 1000)}
            response, _url, err = _get_frigate("/api/events", timeout=10, params=params)
            if response is not None and response.status_code == 200:
                events = response.json() or []
                return {"events": events, "total": len(events), "truncated": False}
            raise RuntimeError(str(err or "Frigate events query failed"))

        # 分批往前翻頁直到滿足限制或資料取完，避免一次查太大拖垮 API
        target = min(int(limit), 1000)
        page_size = max(50, min(int(batch_size), 1000))
        events: List[Dict[str, Any]] = []
        seen = set()
        cursor_before = int(before) if before is not None else None

        while len(events) < target:
            remaining = target - len(events)
            take = min(page_size, remaining)
            params = {**common_params, "limit": take}
            if cursor_before is not None:
                params["before"] = int(cursor_before)
            response, _url, err = _get_frigate("/api/events", timeout=15, params=params)
            if response is None or response.status_code != 200:
                raise RuntimeError(str(err or f"Frigate events query failed ({getattr(response, 'status_code', 'no-response')})"))
            batch = response.json() or []
            if not batch:
                break
            for evt in batch:
                eid = str(evt.get("id") or "")
                if eid and eid in seen:
                    continue
                if eid:
                    seen.add(eid)
                events.append(evt)
                if len(events) >= target:
                    break
            min_start = None
            for evt in batch:
                try:
                    ts = float(evt.get("start_time") or 0)
                    if ts <= 0:
                        continue
                    if min_start is None or ts < min_start:
                        min_start = ts
                except Exception:
                    continue
            if min_start is None:
                break
            cursor_before = int(min_start) - 1
            if after is not None and cursor_before <= int(after):
                break
            if len(batch) < take:
                break
        events.sort(key=lambda e: float(e.get("start_time") or 0), reverse=True)
        truncated = len(events) >= target
        return {"events": events[:target], "total": len(events[:target]), "truncated": truncated}
    except Exception as e:
        return {"events": [], "error": str(e)}
    return {"events": []}


@router.get("/events/{event_id}/snapshot")
async def get_event_snapshot(event_id: str):
    """取得事件快照圖片"""
    try:
        response = requests.get(
            f"{FRIGATE_URL}/api/events/{event_id}/snapshot.jpg", timeout=10
        )
        if response.status_code == 200:
            media_type = response.headers.get("content-type", "image/jpeg")
            return Response(content=response.content, media_type=media_type)
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="快照不存在")


@router.get("/events/{event_id}/clip")
async def get_event_clip(event_id: str):
    """取得事件影片 URL（走本服務代理，避免跨域/主機名不可達）"""
    return {"url": f"/api/frigate/events/{event_id}/clip.mp4"}


@router.get("/events/{event_id}/clip.mp4")
async def get_event_clip_media(event_id: str, request: Request):
    """代理事件影片內容"""
    try:
        upstream = None
        last_error = None
        upstream_headers = {}
        range_header = request.headers.get("range")
        if range_header:
            upstream_headers["Range"] = range_header
        for base in _frigate_base_urls():
            try:
                url = f"{base}/api/events/{event_id}/clip.mp4"
                r = requests.get(url, timeout=30, stream=True, headers=upstream_headers)
                if r.status_code in (200, 206):
                    upstream = r
                    break
                r.close()
            except Exception as e:
                last_error = e
        if upstream is None:
            if last_error:
                raise HTTPException(status_code=502, detail=str(last_error))
            raise HTTPException(status_code=404, detail="影片不存在")

        media_type = upstream.headers.get("content-type", "video/mp4")
        headers = {
            "Cache-Control": "no-store",
            "Accept-Ranges": "bytes",
        }
        for k in ("Content-Length", "Content-Range", "ETag", "Last-Modified"):
            v = upstream.headers.get(k)
            if v:
                headers[k] = v

        def _iter():
            try:
                for chunk in upstream.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        yield chunk
            finally:
                upstream.close()

        return StreamingResponse(
            _iter(),
            media_type=media_type,
            headers=headers,
            status_code=upstream.status_code,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recordings")
async def get_nvr_recordings(
    camera: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
):
    """查詢 NVR 錄影片段（全時錄影回放用）"""
    try:
        def _to_epoch_sec(v: Any) -> int:
            if v is None or v == "":
                return 0
            try:
                n = float(v)
                if n > 1e12:
                    return int(n / 1000)
                if n > 0:
                    return int(n)
            except Exception:
                pass
            try:
                dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                return int(dt.timestamp())
            except Exception:
                return 0

        after = _parse_time_to_epoch(start_time)
        before = _parse_time_to_epoch(end_time)
        if after and before and after > before:
            raise HTTPException(status_code=400, detail="start_time 不可晚於 end_time")

        params: Dict[str, Any] = {"limit": min(int(limit), 500)}
        if camera:
            params["camera"] = camera
        if after is not None:
            params["after"] = int(after)
        if before is not None:
            params["before"] = int(before)

        response, _url, err = _get_frigate("/api/recordings", timeout=15, params=params)
        if (response is None or response.status_code != 200) and camera:
            # 部分版本支援 /api/{camera}/recordings
            response, _url, err = _get_frigate(f"/api/{camera}/recordings", timeout=15, params=params)
        rows: List[Dict[str, Any]] = []
        if response is not None and response.status_code == 200:
            data = response.json() or []
            rows = data if isinstance(data, list) else []
        items = []
        if rows:
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                src = (
                    row.get("playback_url")
                    or row.get("url")
                    or row.get("path")
                    or row.get("file")
                    or ""
                )
                start = row.get("start_time") or row.get("start") or row.get("segment_start")
                end = row.get("end_time") or row.get("end") or row.get("segment_end")
                cam = row.get("camera") or camera or ""
                # 新版 Frigate /api/recordings 可能不回傳 path/url，
                # 改以 camera + start/end 動態組 clip 播放來源。
                if not src and cam:
                    s_sec = _to_epoch_sec(start)
                    e_sec = _to_epoch_sec(end)
                    if s_sec > 0:
                        if e_sec <= s_sec:
                            e_sec = s_sec + max(1, int(float(row.get("duration") or 10)))
                        src = f"/api/{cam}/start/{s_sec}/end/{e_sec}/clip.mp4"
                play_url = f"/api/frigate/recordings/play?src={quote(str(src), safe='')}" if src else ""
                items.append({
                    "id": row.get("id") or f"rec-{cam}-{idx}",
                    "camera": cam,
                    "start_time": start,
                    "end_time": end,
                    "duration": row.get("duration"),
                    "play_url": play_url,
                    "raw": row,
                })
            return {"items": items, "total": len(items), "fallback": False}

        # Fallback: 直接用 Frigate clip URL 依時間區間回放（相容舊版無 /api/recordings）
        import time
        now_sec = int(time.time())
        after_sec = int(after) if after is not None else max(0, now_sec - 3600)
        before_sec = int(before) if before is not None else now_sec
        if before_sec <= after_sec:
            before_sec = after_sec + 60
        cameras = [camera] if camera else _list_frigate_camera_names()
        for idx, cam in enumerate(cameras):
            src = f"/api/{cam}/start/{after_sec}/end/{before_sec}/clip.mp4"
            items.append({
                "id": f"fallback-{cam}-{idx}",
                "camera": cam,
                "start_time": after_sec,
                "end_time": before_sec,
                "duration": before_sec - after_sec,
                "play_url": f"/api/frigate/recordings/play?src={quote(src, safe='')}",
                "raw": {"mode": "fallback_range_clip"},
            })
        return {"items": items, "total": len(items), "fallback": True}
    except HTTPException:
        raise
    except Exception as e:
        return {"items": [], "error": str(e)}


@router.get("/recordings/play")
async def play_nvr_recording(src: str, request: Request):
    """代理錄影播放內容（支援 Range）"""
    try:
        target = unquote(str(src or "").strip())
        if not target:
            raise HTTPException(status_code=400, detail="缺少 src")

        upstream_headers = {}
        range_header = request.headers.get("range")
        if range_header:
            upstream_headers["Range"] = range_header

        candidates: List[str] = []
        if target.startswith("http://") or target.startswith("https://"):
            p = urlparse(target)
            allowed = {urlparse(base).hostname for base in _frigate_base_urls()}
            if p.hostname not in allowed:
                raise HTTPException(status_code=400, detail="不允許的來源主機")
            candidates.append(target)
        else:
            path = target if target.startswith("/") else f"/{target}"
            for base in _frigate_base_urls():
                candidates.append(f"{base}{path}")

        upstream = None
        last_error = None
        for url in candidates:
            try:
                r = requests.get(url, timeout=40, stream=True, headers=upstream_headers)
                if r.status_code in (200, 206):
                    upstream = r
                    break
                r.close()
            except Exception as e:
                last_error = e
        if upstream is None:
            if last_error:
                raise HTTPException(status_code=502, detail=str(last_error))
            raise HTTPException(status_code=404, detail="找不到錄影檔")

        headers = {
            "Cache-Control": "no-store",
            "Accept-Ranges": "bytes",
        }
        for k in ("Content-Length", "Content-Range", "ETag", "Last-Modified"):
            v = upstream.headers.get(k)
            if v:
                headers[k] = v

        def _iter():
            try:
                for chunk in upstream.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        yield chunk
            finally:
                upstream.close()

        return StreamingResponse(
            _iter(),
            media_type=upstream.headers.get("content-type", "video/mp4"),
            headers=headers,
            status_code=upstream.status_code,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 同步 API ============

@router.post("/sync-cameras")
async def sync_cameras_to_nvr():
    """雙向同步：系統攝影機 <-> NVR config"""
    from api.models import SessionLocal, Camera
    from datetime import datetime

    db = SessionLocal()
    try:
        add_log("info", "開始同步攝影機（DB ↔ NVR）", "nvr")
        config = {}
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}

        if "detectors" not in config:
            config["detectors"] = {"tensorrt": {"type": "tensorrt", "device": 0}}
        if "ffmpeg" not in config:
            config["ffmpeg"] = {"hwaccel_args": "preset-jetson-h264"}
        if "cameras" not in config:
            config["cameras"] = {}

        db_cameras = db.query(Camera).all()
        frigate_cams = dict(config.get("cameras", {}))
        synced_to_nvr = []
        imported_to_db = []

        # 讀取全域設定（如果有的話）
        global_record = config.get("record", {})
        global_motion = config.get("motion", {})

        # DB -> NVR
        for cam in db_cameras:
            if not cam.source:
                continue
            cam_name = f"cam_{cam.id}"
            zones_config = {}
            if cam.zones:
                for zone in cam.zones:
                    points = zone.get("points", [])
                    if points:
                        coords = ",".join([f"{p[0]},{p[1]}" for p in points])
                        zones_config[zone.get("name", "zone")] = {
                            "coordinates": coords,
                            "objects": ["car", "motorcycle"],
                        }

            # 保留現有攝影機的個別設定
            existing = config["cameras"].get(cam_name, {})
            existing_record = existing.get("record", {})
            existing_motion = existing.get("motion", {})
            existing_detect = existing.get("detect", {})

            config["cameras"][cam_name] = {
                "enabled": cam.enabled if cam.enabled is not None else True,
                "ffmpeg": {
                    "inputs": [
                        {"path": cam.source, "roles": ["detect", "record"]}
                    ]
                },
                "detect": {
                    "enabled": (
                        cam.detection_enabled
                        if cam.detection_enabled is not None
                        else True
                    ),
                    "width": existing_detect.get("width", 1920),
                    "height": existing_detect.get("height", 1080),
                    "fps": existing_detect.get("fps", 10),
                },
                "objects": {
                    "track": ["car", "motorcycle", "bicycle", "person"],
                    "filters": {
                        "car": {"min_area": 5000, "min_score": 0.5},
                        "motorcycle": {"min_area": 1000, "min_score": 0.5},
                    },
                },
                "zones": zones_config,
                "record": existing_record or {
                    "enabled": True,
                    "retain": {"days": 7, "mode": "motion"},
                },
                "motion": existing_motion or {},
                "snapshots": {
                    "enabled": True,
                    "timestamp": True,
                    "bounding_box": True,
                },
            }
            synced_to_nvr.append(cam_name)

        # NVR -> DB
        db_sources = set(cam.source for cam in db_cameras if cam.source)
        db_cam_names = set(f"cam_{cam.id}" for cam in db_cameras)
        for fname, fcfg in frigate_cams.items():
            if fname in db_cam_names:
                continue
            inputs = fcfg.get("ffmpeg", {}).get("inputs", [])
            rtsp_url = inputs[0].get("path", "") if inputs else ""
            if not rtsp_url or rtsp_url in db_sources:
                continue
            ip, username, password, port, stream_path = "", "", "", "554", ""
            try:
                from urllib.parse import urlparse

                parsed = urlparse(rtsp_url)
                ip = parsed.hostname or ""
                username = parsed.username or ""
                password = parsed.password or ""
                port = str(parsed.port) if parsed.port else "554"
                stream_path = (
                    parsed.path.lstrip("/") if parsed.path else ""
                )
            except Exception:
                pass
            new_cam = Camera(
                name=fname,
                source=rtsp_url,
                ip=ip,
                username=username,
                password=password,
                port=port,
                stream_path=stream_path,
                location=f"NVR - {fname}",
                status="offline",
                enabled=fcfg.get("enabled", True),
                detection_enabled=fcfg.get("detect", {}).get("enabled", True),
                detection_config={
                    "red_light": True,
                    "speeding": True,
                    "illegal_parking": True,
                },
                zones=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(new_cam)
            imported_to_db.append(fname)
        db.commit()

        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)

        msgs = []
        if synced_to_nvr:
            msgs.append(f"DB→NVR: {len(synced_to_nvr)} 台")
        if imported_to_db:
            msgs.append(
                f"NVR→DB: 匯入 {len(imported_to_db)} 台 ({', '.join(imported_to_db)})"
            )
        if not msgs:
            msgs.append("無需同步")
        add_log("success", f"攝影機同步完成: {' | '.join(msgs)}", "nvr")
        return {
            "status": "success",
            "message": " | ".join(msgs),
            "synced_to_nvr": synced_to_nvr,
            "imported_to_db": imported_to_db,
        }
    except Exception as e:
        db.rollback()
        add_log("error", f"攝影機同步失敗: {e}", "nvr")
        return {"status": "error", "message": f"同步失敗: {str(e)}"}
    finally:
        db.close()


@router.post("/restart")
async def restart_nvr():
    """重啟 NVR"""
    add_log("info", "收到 NVR 重啟請求", "nvr")
    # Use NVR's own restart API. Avoid docker CLI dependency inside API container.
    try:
        response = requests.post(f"{FRIGATE_URL}/api/restart", timeout=8)
        if response.status_code in (200, 202):
            payload = response.json() if response.content else {}
            add_log("warning", "NVR 重啟中（等待恢復）", "nvr")
            return {
                "status": "success",
                "message": payload.get("message", "NVR 正在重啟（約 1 分鐘）"),
            }
        # Frigate restart may transiently return 500 while auth/app is reloading.
        if response.status_code >= 500:
            add_log("warning", f"NVR 重啟中（暫時 HTTP {response.status_code}）", "nvr")
            return {
                "status": "success",
                "message": "NVR 正在重啟中，請稍候自動恢復",
            }
        text = response.text[:180] if response.text else ""
        add_log("error", f"NVR 重啟失敗：HTTP {response.status_code}", "nvr")
        return {
            "status": "error",
            "message": f"NVR 暫時無法重啟（HTTP {response.status_code}）{text}",
        }
    except Exception as e:
        add_log("error", f"NVR 重啟失敗：{e}", "nvr")
        return {"status": "error", "message": f"NVR 連線失敗，請稍後再試: {e}"}


# ============ 新增/刪除攝影機 ============

@router.post("/camera")
async def add_nvr_camera(camera: FrigateCameraAdd):
    """新增攝影機到 NVR"""
    try:
        config_path = Path(FRIGATE_CONFIG_PATH)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="NVR 設定檔不存在")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        if 'cameras' not in config:
            config['cameras'] = {}

        if camera.name in config['cameras']:
            return {
                "status": "error",
                "message": f"攝影機 {camera.name} 已存在",
            }

        cam_config = {
            'enabled': True,
            'ffmpeg': {
                'inputs': [{'path': camera.rtsp_url, 'roles': ['detect']}]
            },
            'detect': {
                'enabled': camera.detect,
                'width': 1920,
                'height': 1080,
                'fps': 10,
            },
            'objects': {
                'track': ['car', 'motorcycle', 'bicycle', 'person']
            },
        }

        if camera.record:
            cam_config['ffmpeg']['inputs'][0]['roles'].append('record')
            cam_config['record'] = {
                'enabled': True,
                'retain': {'days': 7, 'mode': 'motion'},
            }

        if camera.snapshots:
            cam_config['snapshots'] = {
                'enabled': True,
                'bounding_box': True,
                'timestamp': True,
                'retain': {'default': 14},
            }

        config['cameras'][camera.name] = cam_config

        with open(config_path, 'w') as f:
            yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)

        return {
            "status": "success",
            "message": f"已新增攝影機 {camera.name}，請重啟 NVR 生效",
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": f"新增失敗: {str(e)}"}


@router.delete("/camera/{name}")
async def delete_nvr_camera_by_name(name: str):
    """從 NVR 刪除攝影機"""
    try:
        config_path = Path(FRIGATE_CONFIG_PATH)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="NVR 設定檔不存在")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        cameras = config.get('cameras', {})
        if name not in cameras:
            return {"status": "error", "message": f"找不到攝影機 {name}"}

        del cameras[name]
        config['cameras'] = cameras

        with open(config_path, 'w') as f:
            yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)

        return {
            "status": "success",
            "message": f"已刪除攝影機 {name}，請重啟 NVR 生效",
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": f"刪除失敗: {str(e)}"}


@router.put("/camera/{name}/switch")
async def switch_nvr_camera_features(name: str, data: NvrCameraSwitch):
    """切換單台攝影機錄影/偵測開關"""
    try:
        config_path = Path(FRIGATE_CONFIG_PATH)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="NVR 設定檔不存在")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        cameras = config.get("cameras", {})
        if name not in cameras:
            raise HTTPException(status_code=404, detail=f"找不到攝影機 {name}")

        cam = cameras[name]
        if "ffmpeg" not in cam:
            cam["ffmpeg"] = {"inputs": [{"path": "", "roles": ["detect"]}]}
        if "inputs" not in cam["ffmpeg"] or not cam["ffmpeg"]["inputs"]:
            cam["ffmpeg"]["inputs"] = [{"path": "", "roles": ["detect"]}]
        roles = cam["ffmpeg"]["inputs"][0].get("roles", []) or []

        if data.record_enabled is not None:
            record_cfg = cam.get("record", {}) or {}
            record_cfg["enabled"] = bool(data.record_enabled)
            retain_cfg = record_cfg.get("retain", {}) or {}
            retain_cfg["days"] = int(retain_cfg.get("days", 7))
            retain_cfg["mode"] = retain_cfg.get("mode", "motion")
            record_cfg["retain"] = retain_cfg
            cam["record"] = record_cfg
            if data.record_enabled and "record" not in roles:
                roles.append("record")
            if not data.record_enabled and "record" in roles:
                roles.remove("record")

        if data.detect_enabled is not None:
            detect_on = bool(data.detect_enabled)
            detect_cfg = cam.get("detect", {}) or {}
            detect_cfg["enabled"] = detect_on
            detect_cfg["fps"] = int(detect_cfg.get("fps", 10) or 10)
            detect_cfg["width"] = int(detect_cfg.get("width", 1920) or 1920)
            detect_cfg["height"] = int(detect_cfg.get("height", 1080) or 1080)
            cam["detect"] = detect_cfg

            # 相容舊版 UI：detect 開關同時帶動 motion.enabled
            motion_cfg = cam.get("motion", {}) or {}
            motion_cfg["enabled"] = detect_on
            motion_cfg["threshold"] = int(motion_cfg.get("threshold", 25))
            motion_cfg["contour_area"] = int(motion_cfg.get("contour_area", 20))
            cam["motion"] = motion_cfg

        if data.motion_enabled is not None:
            motion_on = bool(data.motion_enabled)
            motion_cfg = cam.get("motion", {}) or {}
            motion_cfg["enabled"] = motion_on
            motion_cfg["threshold"] = int(motion_cfg.get("threshold", 25))
            motion_cfg["contour_area"] = int(motion_cfg.get("contour_area", 20))
            cam["motion"] = motion_cfg

        if data.snapshots_enabled is not None:
            snapshots_cfg = cam.get("snapshots", {}) or {}
            snapshots_cfg["enabled"] = bool(data.snapshots_enabled)
            retain_cfg = snapshots_cfg.get("retain", {}) or {}
            retain_cfg["default"] = int(retain_cfg.get("default", 14))
            snapshots_cfg["retain"] = retain_cfg
            cam["snapshots"] = snapshots_cfg

        cam["ffmpeg"]["inputs"][0]["roles"] = roles
        cameras[name] = cam
        config["cameras"] = cameras

        with open(config_path, 'w') as f:
            yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)

        if data.record_enabled is not None:
            add_log("info", f"{name} 錄影已{'開啟' if data.record_enabled else '關閉'}", "nvr")
        if data.detect_enabled is not None:
            add_log("info", f"{name} 偵測已{'開啟' if data.detect_enabled else '關閉'}", "nvr")
        if data.motion_enabled is not None:
            add_log("info", f"{name} Motion 已{'開啟' if data.motion_enabled else '關閉'}", "nvr")
        if data.snapshots_enabled is not None:
            add_log("info", f"{name} Snapshots 已{'開啟' if data.snapshots_enabled else '關閉'}", "nvr")

        return {
            "status": "success",
            "message": f"{name} 設定已更新，請重啟 NVR 生效",
            "record_enabled": bool(cam.get("record", {}).get("enabled", False)),
            "detect_enabled": bool(cam.get("detect", {}).get("enabled", True)),
            "motion_enabled": bool(cam.get("motion", {}).get("enabled", True)),
            "snapshots_enabled": bool(cam.get("snapshots", {}).get("enabled", False)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/camera/{name}/motion-roi")
async def get_nvr_camera_motion_roi(name: str):
    """取得單台攝影機 motion ROI（對應 Frigate motion.mask）"""
    try:
        config_path = Path(FRIGATE_CONFIG_PATH)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="NVR 設定檔不存在")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        cameras = config.get("cameras", {})
        cam = cameras.get(name)
        if not cam:
            raise HTTPException(status_code=404, detail=f"找不到攝影機 {name}")

        detect_w, detect_h = _camera_detect_size(cam)
        ui_settings = _load_ui_settings()
        ui_motion = ui_settings.get("nvr_motion_roi", {}).get(name, {})
        points = ui_motion.get("points", []) if isinstance(ui_motion, dict) else []
        include_points = ui_motion.get("include_points", []) if isinstance(ui_motion, dict) else []
        exclude_points = ui_motion.get("exclude_points", []) if isinstance(ui_motion, dict) else []
        mode = str(ui_motion.get("mode", "exclude")) if isinstance(ui_motion, dict) else "exclude"
        width = int(ui_motion.get("width", detect_w)) if isinstance(ui_motion, dict) else detect_w
        height = int(ui_motion.get("height", detect_h)) if isinstance(ui_motion, dict) else detect_h

        if not points and not include_points and not exclude_points:
            motion_cfg = cam.get("motion", {}) if isinstance(cam, dict) else {}
            polys = _parse_motion_mask_polygons(motion_cfg.get("mask"), detect_w, detect_h)
            points = polys[0] if polys else []
            exclude_points = points
            include_points = []
            mode = "exclude"
            width = detect_w
            height = detect_h
        if not exclude_points and points:
            exclude_points = points

        return {
            "status": "success",
            "camera": name,
            "width": width,
            "height": height,
            "points": points or [],
            "include_points": include_points or [],
            "exclude_points": exclude_points or [],
            "mode": mode,
            "has_mask": bool(exclude_points or points),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/camera/{name}/motion-roi")
async def update_nvr_camera_motion_roi(name: str, data: NvrMotionRoiUpdate):
    """更新單台攝影機 motion ROI（寫入 Frigate motion.mask）"""
    try:
        config_path = Path(FRIGATE_CONFIG_PATH)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="NVR 設定檔不存在")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        cameras = config.get("cameras", {})
        cam = cameras.get(name)
        if not cam:
            raise HTTPException(status_code=404, detail=f"找不到攝影機 {name}")

        width = max(1, int(data.width or 1))
        height = max(1, int(data.height or 1))
        mode = str(data.mode or "exclude").strip().lower()
        if mode not in ("exclude", "include", "both"):
            mode = "exclude"
        include_points = _sanitize_polygon(data.include_points or [], width, height)
        exclude_points = _sanitize_polygon(data.exclude_points or [], width, height)
        # 舊版相容：未提供 exclude_points 時，沿用 points
        if not exclude_points and data.points:
            exclude_points = _sanitize_polygon(data.points or [], width, height)
        if include_points and len(include_points) < 3:
            raise HTTPException(status_code=400, detail="偵測區至少需要 3 個點")
        if exclude_points and len(exclude_points) < 3:
            raise HTTPException(status_code=400, detail="排除區至少需要 3 個點")

        mask_polygons: List[List[List[int]]] = []
        if mode in ("include", "both") and include_points:
            mask_polygons.extend(_outer_masks_from_include(include_points, width, height))
        if exclude_points:
            mask_polygons.append(exclude_points)

        if "motion" not in cam or not isinstance(cam.get("motion"), dict):
            cam["motion"] = {}
        if mask_polygons:
            cam["motion"]["mask"] = _format_motion_mask(mask_polygons, width, height)
        else:
            cam["motion"].pop("mask", None)

        cameras[name] = cam
        config["cameras"] = cameras
        with open(config_path, "w") as f:
            yaml.dump(_sanitize_frigate_config(config), f, default_flow_style=False, allow_unicode=True)

        ui_settings = _load_ui_settings()
        nvr_motion_roi = ui_settings.get("nvr_motion_roi", {})
        if not isinstance(nvr_motion_roi, dict):
            nvr_motion_roi = {}
        nvr_motion_roi[name] = {
            "width": width,
            "height": height,
            "points": exclude_points,  # 舊欄位相容
            "include_points": include_points,
            "exclude_points": exclude_points,
            "mode": mode,
            "updated_at": datetime.utcnow().isoformat(),
        }
        ui_settings["nvr_motion_roi"] = nvr_motion_roi
        _save_ui_settings(ui_settings)

        add_log("info", f"{name} Motion ROI 已更新（偵測區 {len(include_points)} 點 / 排除區 {len(exclude_points)} 點）", "nvr")
        return {
            "status": "success",
            "message": f"{name} Motion ROI 已儲存，請重啟 NVR 套用",
            "points": exclude_points,
            "include_points": include_points,
            "exclude_points": exclude_points,
            "mode": mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/camera/{name}/latest.jpg")
async def get_nvr_camera_latest_jpg(name: str):
    """代理 Frigate 單台攝影機最新影像，供 ROI 編輯器使用"""
    try:
        response, _url, _err = _get_frigate(f"/api/{name}/latest.jpg", timeout=8)
        if response is None or response.status_code != 200:
            raise HTTPException(status_code=404, detail="無法取得攝影機最新影像")
        media_type = response.headers.get("content-type", "image/jpeg")
        return Response(content=response.content, media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
