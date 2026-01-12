#!/usr/bin/env python3
"""Frigate NVR 設定 API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import yaml
import os
import requests
from pathlib import Path

router = APIRouter(prefix="/api/frigate", tags=["frigate"])

# Frigate 設定檔路徑
FRIGATE_CONFIG_PATH = "/workspace/config/frigate/config.yml"
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "frigate")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))
FRIGATE_URL = f"http://{FRIGATE_HOST}:{FRIGATE_PORT}"


class FrigateSettings(BaseModel):
    """Frigate 基本設定"""
    enabled: bool = True
    host: str = "frigate"
    port: int = 5000
    mqtt_enabled: bool = False
    mqtt_host: Optional[str] = None
    mqtt_port: int = 1883
    mqtt_user: Optional[str] = None
    mqtt_password: Optional[str] = None


class FrigateCameraConfig(BaseModel):
    """Frigate 攝影機設定"""
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


class FrigateZoneConfig(BaseModel):
    """Frigate 區域設定"""
    name: str
    camera: str
    coordinates: str  # "x1,y1,x2,y2,x3,y3,..."
    objects: List[str] = ["car", "motorcycle"]
    violation_type: Optional[str] = None


# ============ 狀態 API ============

@router.get("/status")
async def get_frigate_status():
    """取得 Frigate 連線狀態"""
    try:
        response = requests.get(f"{FRIGATE_URL}/api/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            return {
                "status": "online",
                "version": stats.get("service", {}).get("version", "unknown"),
                "uptime": stats.get("service", {}).get("uptime", 0),
                "cameras": list(stats.get("cameras", {}).keys()),
                "detectors": stats.get("detectors", {})
            }
    except Exception as e:
        pass
    
    return {
        "status": "offline",
        "message": "無法連線到 Frigate NVR"
    }


@router.get("/version")
async def get_frigate_version():
    """取得 Frigate 版本"""
    try:
        response = requests.get(f"{FRIGATE_URL}/api/version", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"version": "unknown", "status": "offline"}


# ============ 設定 API ============

@router.get("/config")
async def get_frigate_config():
    """取得 Frigate 設定"""
    try:
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
                return {"status": "success", "config": config}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    return {"status": "error", "message": "設定檔不存在"}


@router.post("/config")
async def save_frigate_config(config: Dict[str, Any]):
    """儲存 Frigate 設定"""
    try:
        os.makedirs(os.path.dirname(FRIGATE_CONFIG_PATH), exist_ok=True)
        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return {"status": "success", "message": "設定已儲存，請重啟 Frigate 生效"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings")
async def get_frigate_settings():
    """取得 Frigate 連線設定"""
    try:
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}
                mqtt = config.get("mqtt", {})
                return {
                    "enabled": True,
                    "host": FRIGATE_HOST,
                    "port": FRIGATE_PORT,
                    "mqtt_enabled": mqtt.get("enabled", False),
                    "mqtt_host": mqtt.get("host"),
                    "mqtt_port": mqtt.get("port", 1883),
                    "mqtt_user": mqtt.get("user")
                }
    except:
        pass
    
    return FrigateSettings().dict()


@router.put("/settings")
async def update_frigate_settings(settings: FrigateSettings):
    """更新 Frigate 設定"""
    try:
        config = {}
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        # 更新 MQTT 設定
        if settings.mqtt_enabled:
            config["mqtt"] = {
                "enabled": True,
                "host": settings.mqtt_host,
                "port": settings.mqtt_port,
                "user": settings.mqtt_user,
                "password": settings.mqtt_password
            }
        else:
            config["mqtt"] = {"enabled": False}
        
        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return {"status": "success", "message": "設定已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 攝影機 API ============

@router.get("/cameras")
async def get_frigate_cameras():
    """取得 Frigate 攝影機列表"""
    cameras = []
    
    try:
        # 從 Frigate API 取得
        response = requests.get(f"{FRIGATE_URL}/api/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            for name, info in stats.get("cameras", {}).items():
                cameras.append({
                    "name": name,
                    "fps": info.get("camera_fps", 0),
                    "detection_fps": info.get("detection_fps", 0),
                    "process_fps": info.get("process_fps", 0),
                    "pid": info.get("pid", 0)
                })
    except:
        pass
    
    # 也從設定檔取得
    try:
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}
                for name, cam_config in config.get("cameras", {}).items():
                    existing = next((c for c in cameras if c["name"] == name), None)
                    if not existing:
                        cameras.append({
                            "name": name,
                            "enabled": cam_config.get("enabled", True),
                            "rtsp_url": cam_config.get("ffmpeg", {}).get("inputs", [{}])[0].get("path", ""),
                            "config": cam_config
                        })
                    else:
                        existing["config"] = cam_config
                        existing["enabled"] = cam_config.get("enabled", True)
    except:
        pass
    
    return {"cameras": cameras}


@router.post("/cameras")
async def add_frigate_camera(camera: FrigateCameraConfig):
    """新增 Frigate 攝影機"""
    try:
        config = {}
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        if "cameras" not in config:
            config["cameras"] = {}
        
        # 建立攝影機設定
        cam_config = {
            "enabled": camera.enabled,
            "ffmpeg": {
                "inputs": [{
                    "path": camera.rtsp_url,
                    "roles": ["detect", "record"] if camera.record_enabled else ["detect"]
                }]
            },
            "detect": {
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps
            },
            "objects": {
                "track": camera.detect_objects
            },
            "record": {
                "enabled": camera.record_enabled,
                "retain": {
                    "days": camera.record_retain_days,
                    "mode": "motion"
                }
            },
            "snapshots": {
                "enabled": camera.snapshot_enabled,
                "timestamp": True,
                "bounding_box": True
            }
        }
        
        # 加入區域
        if camera.zones:
            cam_config["zones"] = {}
            for zone in camera.zones:
                cam_config["zones"][zone["name"]] = {
                    "coordinates": zone["coordinates"],
                    "objects": zone.get("objects", ["car", "motorcycle"])
                }
        
        config["cameras"][camera.name] = cam_config
        
        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return {"status": "success", "message": f"攝影機 {camera.name} 已新增"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/cameras/{camera_name}")
async def update_frigate_camera(camera_name: str, camera: FrigateCameraConfig):
    """更新 Frigate 攝影機"""
    try:
        if not os.path.exists(FRIGATE_CONFIG_PATH):
            raise HTTPException(status_code=404, detail="設定檔不存在")
        
        with open(FRIGATE_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        if camera_name not in config.get("cameras", {}):
            raise HTTPException(status_code=404, detail="攝影機不存在")
        
        # 更新設定
        cam_config = config["cameras"][camera_name]
        cam_config["enabled"] = camera.enabled
        cam_config["ffmpeg"]["inputs"][0]["path"] = camera.rtsp_url
        cam_config["detect"] = {
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps
        }
        cam_config["objects"]["track"] = camera.detect_objects
        cam_config["record"]["enabled"] = camera.record_enabled
        cam_config["record"]["retain"]["days"] = camera.record_retain_days
        cam_config["snapshots"]["enabled"] = camera.snapshot_enabled
        
        # 更新區域
        if camera.zones:
            cam_config["zones"] = {}
            for zone in camera.zones:
                cam_config["zones"][zone["name"]] = {
                    "coordinates": zone["coordinates"],
                    "objects": zone.get("objects", ["car", "motorcycle"])
                }
        
        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return {"status": "success", "message": f"攝影機 {camera_name} 已更新"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cameras/{camera_name}")
async def delete_frigate_camera(camera_name: str):
    """刪除 Frigate 攝影機"""
    try:
        if not os.path.exists(FRIGATE_CONFIG_PATH):
            raise HTTPException(status_code=404, detail="設定檔不存在")
        
        with open(FRIGATE_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        if camera_name in config.get("cameras", {}):
            del config["cameras"][camera_name]
            
            with open(FRIGATE_CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            return {"status": "success", "message": f"攝影機 {camera_name} 已刪除"}
        
        raise HTTPException(status_code=404, detail="攝影機不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 事件 API ============

@router.get("/events")
async def get_frigate_events(
    camera: Optional[str] = None,
    label: Optional[str] = None,
    limit: int = 50
):
    """取得 Frigate 事件"""
    try:
        params = {"limit": limit}
        if camera:
            params["camera"] = camera
        if label:
            params["label"] = label
        
        response = requests.get(f"{FRIGATE_URL}/api/events", params=params, timeout=10)
        if response.status_code == 200:
            return {"events": response.json()}
    except Exception as e:
        return {"events": [], "error": str(e)}
    
    return {"events": []}


@router.get("/events/{event_id}/snapshot")
async def get_event_snapshot(event_id: str):
    """取得事件快照 URL"""
    return {"url": f"{FRIGATE_URL}/api/events/{event_id}/snapshot.jpg"}


@router.get("/events/{event_id}/clip")
async def get_event_clip(event_id: str):
    """取得事件影片 URL"""
    return {"url": f"{FRIGATE_URL}/api/events/{event_id}/clip.mp4"}


# ============ 同步 API ============

@router.post("/sync-cameras")
async def sync_cameras_to_frigate():
    """將系統攝影機同步到 Frigate"""
    from api.database import SessionLocal
    from api.models import Camera
    
    db = SessionLocal()
    try:
        cameras = db.query(Camera).filter(Camera.status == "online").all()
        
        config = {}
        if os.path.exists(FRIGATE_CONFIG_PATH):
            with open(FRIGATE_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        # 確保基本設定存在
        if "detectors" not in config:
            config["detectors"] = {
                "tensorrt": {
                    "type": "tensorrt",
                    "device": 0
                }
            }
        
        if "ffmpeg" not in config:
            config["ffmpeg"] = {
                "hwaccel_args": "preset-jetson-h264"
            }
        
        if "cameras" not in config:
            config["cameras"] = {}
        
        synced = []
        for cam in cameras:
            cam_name = f"cam_{cam.id}"
            
            # 建立攝影機設定
            zones_config = {}
            if cam.zones:
                for zone in cam.zones:
                    # 轉換座標格式
                    points = zone.get("points", [])
                    if points:
                        coords = ",".join([f"{p[0]},{p[1]}" for p in points])
                        zones_config[zone["name"]] = {
                            "coordinates": coords,
                            "objects": ["car", "motorcycle"]
                        }
            
            config["cameras"][cam_name] = {
                "enabled": True,
                "ffmpeg": {
                    "inputs": [{
                        "path": cam.source,
                        "roles": ["detect", "record"]
                    }]
                },
                "detect": {
                    "width": 1920,
                    "height": 1080,
                    "fps": 10
                },
                "objects": {
                    "track": ["car", "motorcycle", "person"],
                    "filters": {
                        "car": {"min_area": 5000, "min_score": 0.5},
                        "motorcycle": {"min_area": 1000, "min_score": 0.5}
                    }
                },
                "zones": zones_config,
                "record": {
                    "enabled": True,
                    "retain": {"days": 7, "mode": "motion"}
                },
                "snapshots": {
                    "enabled": True,
                    "timestamp": True,
                    "bounding_box": True
                }
            }
            synced.append(cam_name)
        
        with open(FRIGATE_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return {
            "status": "success",
            "message": f"已同步 {len(synced)} 台攝影機到 Frigate",
            "cameras": synced
        }
    
    finally:
        db.close()


@router.post("/restart")
async def restart_frigate():
    """重啟 Frigate (需要 Docker 權限)"""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "restart", "frigate"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return {"status": "success", "message": "Frigate 正在重啟"}
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}
