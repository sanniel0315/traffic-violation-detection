#!/usr/bin/env python3
"""壅塞偵測 API"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from typing import Dict
from pydantic import BaseModel, Field
import cv2
import time
import threading
from datetime import datetime

from api.models import get_db, Camera, SessionLocal
from api.utils.roi_scope import SCOPE_CONGESTION, select_zones
from api.utils.feature_state import get_feature_state, set_feature_state

router = APIRouter(prefix="/api/congestion", tags=["壅塞偵測"])

# 壅塞偵測服務狀態
congestion_services: Dict[int, dict] = {}
congestion_results: Dict[int, dict] = {}
congestion_params: Dict[int, dict] = {}

DEFAULT_CONGESTION_PARAMS = {
    "medium_threshold": 0.2,
    "high_threshold": 0.4,
    "critical_threshold": 0.6,
    "smoothing_window": 10,
    "analyze_interval_sec": 1.0,
}


def _normalize_params(params: dict):
    merged = {**DEFAULT_CONGESTION_PARAMS, **(params or {})}
    ordered = sorted(
        [merged["medium_threshold"], merged["high_threshold"], merged["critical_threshold"]]
    )
    merged["medium_threshold"], merged["high_threshold"], merged["critical_threshold"] = ordered
    return merged


def _load_params_from_camera(camera: Camera):
    cfg = camera.detection_config or {}
    if isinstance(cfg, dict):
        return _normalize_params(cfg.get("congestion", {}))
    return DEFAULT_CONGESTION_PARAMS.copy()


def _save_params_to_camera(camera: Camera, params: dict):
    cfg = dict(camera.detection_config) if isinstance(camera.detection_config, dict) else {}
    cfg["congestion"] = _normalize_params(params)
    camera.detection_config = cfg
    flag_modified(camera, "detection_config")


class CongestionParamsUpdate(BaseModel):
    medium_threshold: float = Field(ge=0.0, le=1.0, default=0.2)
    high_threshold: float = Field(ge=0.0, le=1.0, default=0.4)
    critical_threshold: float = Field(ge=0.0, le=1.0, default=0.6)
    smoothing_window: int = Field(ge=1, le=60, default=10)
    analyze_interval_sec: float = Field(ge=0.1, le=5.0, default=1.0)

# 共享偵測器實例與推論鎖（避免 YOLO 並發推論造成不穩定）
_detector = None
_detector_lock = threading.Lock()

def get_detector():
    global _detector
    if _detector is None:
        from detection.congestion_detector import CongestionDetector
        _detector = CongestionDetector()
    return _detector


def get_effective_params(camera_id: int):
    return _normalize_params(congestion_params.get(camera_id, {}))


def analyze_with_lock(frame, zones, camera_id: int):
    """序列化壅塞推論，降低原生庫並發崩潰風險"""
    detector = get_detector()
    params = get_effective_params(camera_id)
    with _detector_lock:
        return detector.analyze(frame, zones, camera_key=str(camera_id), params=params)


@router.post("/{camera_id}/start")
async def start_congestion(camera_id: int, db: Session = Depends(get_db)):
    """啟動壅塞偵測"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    
    if camera_id in congestion_services and congestion_services[camera_id].get('running'):
        return {"status": "already_running", "message": "壅塞偵測已在執行中"}
    _start_congestion_service(camera)
    set_feature_state("congestion", camera_id, True)
    
    return {"status": "started", "message": f"壅塞偵測已啟動: {camera.name}"}


@router.post("/{camera_id}/stop")
async def stop_congestion(camera_id: int):
    """停止壅塞偵測"""
    if camera_id in congestion_services:
        congestion_services[camera_id]['running'] = False
    set_feature_state("congestion", camera_id, False)
    return {"status": "stopped", "message": "壅塞偵測已停止"}


@router.get("/{camera_id}/params")
async def get_congestion_params(camera_id: int, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    params = _load_params_from_camera(camera)
    congestion_params[camera_id] = params
    return {"camera_id": camera_id, "params": params}


@router.put("/{camera_id}/params")
async def update_congestion_params(camera_id: int, data: CongestionParamsUpdate, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    params = _normalize_params(data.dict())
    _save_params_to_camera(camera, params)
    db.commit()
    congestion_params[camera_id] = params
    return {"status": "success", "camera_id": camera_id, "params": params}


@router.get("/{camera_id}/status")
async def congestion_status(camera_id: int):
    """取得壅塞偵測狀態"""
    service = congestion_services.get(camera_id, {'running': False})
    result = congestion_results.get(camera_id, {})
    return {**service, 'result': result, 'params': get_effective_params(camera_id)}


@router.get("/status/all")
async def all_congestion_status():
    """取得所有壅塞偵測狀態"""
    return {
        cid: {
            **congestion_services.get(cid, {}),
            'result': congestion_results.get(cid, {}),
            'params': get_effective_params(cid),
        }
        for cid in set(list(congestion_services.keys()) + list(congestion_results.keys()))
    }


@router.get("/{camera_id}/snapshot")
async def congestion_snapshot(camera_id: int, db: Session = Depends(get_db)):
    """取得單次壅塞分析截圖"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    
    cap = cv2.VideoCapture(camera.source)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=503, detail="無法取得影像")
    
    congestion_params[camera_id] = _load_params_from_camera(camera)
    zones = select_zones(camera.zones or [], scope=SCOPE_CONGESTION, allowed_types=("detection", "flow_detection"))
    result = analyze_with_lock(frame, zones, camera_id)
    
    # 繪製結果
    annotated = draw_congestion(frame, result)
    
    _, buffer = cv2.imencode('.jpg', annotated)
    return StreamingResponse(iter([buffer.tobytes()]), media_type="image/jpeg")


@router.get("/{camera_id}/stream")
async def congestion_stream(camera_id: int, db: Session = Depends(get_db)):
    """壅塞偵測視覺化串流"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    
    congestion_params[camera_id] = _load_params_from_camera(camera)
    zones = select_zones(camera.zones or [], scope=SCOPE_CONGESTION, allowed_types=("detection", "flow_detection"))
    return StreamingResponse(
        generate_congestion_stream(camera_id, camera.source, zones),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def generate_congestion_stream(camera_id: int, source: str, zones: list):
    """產生壅塞偵測串流"""
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        result = analyze_with_lock(frame, zones, camera_id)
        annotated = draw_congestion(frame, result)
        annotated = cv2.resize(annotated, (640, 360))
        
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.2)  # 5 FPS
    
    cap.release()


def draw_congestion(frame, result):
    """繪製壅塞視覺化"""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    colors = {'car': (255,0,0), 'motorcycle': (0,255,255), 'bus': (255,165,0), 'truck': (128,0,128)}
    level_colors = {'low': (0,255,0), 'medium': (0,255,255), 'high': (0,165,255), 'critical': (0,0,255)}
    
    # 繪製車輛
    for v in result.get('vehicles', []):
        bbox = v['bbox']
        color = colors.get(v['type'], (255,255,255))
        cv2.rectangle(annotated, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
    
    # 狀態列
    cv2.rectangle(annotated, (0, 0), (w, 35), (0,0,0), -1)
    level_color = level_colors.get(result.get('level', 'low'), (255,255,255))
    text = f"壅塞: {result.get('level_name', '-')} | 車輛: {result.get('vehicle_count', 0)} | 佔用率: {result.get('occupancy', 0)*100:.1f}%"
    cv2.putText(annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 2)
    
    # 底部進度條
    bar_w = int((w - 20) * result.get('occupancy', 0))
    cv2.rectangle(annotated, (10, h-15), (w-10, h-5), (50,50,50), -1)
    cv2.rectangle(annotated, (10, h-15), (10 + bar_w, h-5), level_color, -1)
    
    return annotated


def run_congestion_detection(camera_id: int, source: str, zones: list):
    """背景壅塞偵測任務"""
    cap = cv2.VideoCapture(source)
    
    print(f"🚦 壅塞偵測啟動: camera_id={camera_id}")
    try:
        if not cap.isOpened():
            congestion_services[camera_id]["running"] = False
            congestion_services[camera_id]["error"] = "無法開啟攝影機串流"
            return

        while congestion_services.get(camera_id, {}).get('running', False):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.2)
                continue

            result = analyze_with_lock(frame, zones, camera_id)
            congestion_results[camera_id] = result
            congestion_services[camera_id]['last_update'] = datetime.now().isoformat()
            time.sleep(get_effective_params(camera_id).get("analyze_interval_sec", 1.0))
    except Exception as e:
        congestion_services[camera_id]["error"] = str(e)
    finally:
        congestion_services[camera_id]["running"] = False
        cap.release()
        print(f"⏹️ 壅塞偵測停止: camera_id={camera_id}")


def _start_congestion_service(camera: Camera) -> bool:
    camera_id = int(camera.id)
    if camera_id in congestion_services and congestion_services[camera_id].get("running"):
        return False
    if camera_id not in congestion_params:
        congestion_params[camera_id] = _load_params_from_camera(camera)
    congestion_services[camera_id] = {
        "running": True,
        "started_at": datetime.now().isoformat(),
        "camera_name": camera.name,
    }
    zones = select_zones(camera.zones or [], scope=SCOPE_CONGESTION, allowed_types=("detection", "flow_detection"))
    worker = threading.Thread(
        target=run_congestion_detection,
        args=(camera_id, camera.source, zones),
        daemon=True,
        name=f"congestion-{camera_id}",
    )
    congestion_services[camera_id]["thread_name"] = worker.name
    worker.start()
    return True


def resume_congestion_services() -> dict:
    wants = get_feature_state("congestion")
    if not wants:
        return {"total": 0, "resumed": 0}
    db = SessionLocal()
    resumed = 0
    total = 0
    try:
        for camera_id, enabled in wants.items():
            if not enabled:
                continue
            total += 1
            cam = db.query(Camera).filter(Camera.id == camera_id).first()
            if not cam:
                continue
            if _start_congestion_service(cam):
                resumed += 1
    finally:
        db.close()
    return {"total": total, "resumed": resumed}
