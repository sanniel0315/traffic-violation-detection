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
from datetime import datetime, timezone

from api.models import CongestionSample, get_db, Camera, SessionLocal
from api.utils.roi_scope import SCOPE_CONGESTION, SCOPE_TRAFFIC, select_zones
from api.utils.feature_state import get_feature_state, set_feature_state
from api.utils.camera_stream import resolve_analysis_source

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


def _congestion_zones(camera: Camera):
    return select_zones(
        camera.zones or [],
        scope=SCOPE_CONGESTION,
        fallback_scopes=(SCOPE_TRAFFIC,),
        allowed_types=("detection", "flow_detection"),
    )


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


def _to_utc_naive(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _store_congestion_samples(camera_id: int, camera_name: str, result: dict, sample_interval_sec: float):
    db = SessionLocal()
    try:
        created_at = datetime.utcnow()
        base_items = [{
            "camera_id": camera_id,
            "camera_name": camera_name,
            "zone_name": "",
            "lane_no": None,
            "movement": "",
            "direction": "",
            "is_overall": True,
            "vehicle_count": int(result.get("vehicle_count") or 0),
            "stopped_vehicle_count": int(result.get("stopped_vehicle_count") or 0),
            "occupancy": float(result.get("occupancy") or 0.0),
            "raw_occupancy": float(result.get("raw_occupancy") or 0.0),
            "queue_score": float(result.get("queue_score") or 0.0),
            "estimated_queue_length_m": float(result.get("estimated_queue_length_m") or 0.0),
            "queue_duration_sec": float(result.get("queue_duration_sec") or 0.0),
            "sample_interval_sec": float(sample_interval_sec or 0.0),
            "queue_active": bool(result.get("queue_active")),
            "created_at": created_at,
        }]
        for zr in result.get("zone_results") or []:
            base_items.append({
                "camera_id": camera_id,
                "camera_name": camera_name,
                "zone_name": str(zr.get("name") or ""),
                "lane_no": zr.get("lane_no"),
                "movement": str(zr.get("movement") or ""),
                "direction": str(zr.get("direction") or ""),
                "is_overall": False,
                "vehicle_count": int(zr.get("vehicle_count") or 0),
                "stopped_vehicle_count": int(zr.get("stopped_vehicle_count") or 0),
                "occupancy": float(zr.get("occupancy") or 0.0),
                "raw_occupancy": float(zr.get("raw_occupancy") or 0.0),
                "queue_score": float(zr.get("queue_score") or 0.0),
                "estimated_queue_length_m": float(zr.get("estimated_queue_length_m") or 0.0),
                "queue_duration_sec": float(zr.get("queue_duration_sec") or 0.0),
                "sample_interval_sec": float(sample_interval_sec or 0.0),
                "queue_active": bool(zr.get("queue_active")),
                "created_at": created_at,
            })
        db.bulk_insert_mappings(CongestionSample, base_items)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


@router.post("/{camera_id}/start")
async def start_congestion(camera_id: int, db: Session = Depends(get_db)):
    """啟動壅塞偵測"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")
    if camera_id in congestion_services and congestion_services[camera_id].get('running'):
        return {"status": "already_running", "message": "壅塞偵測已在執行中"}
    started = _start_congestion_service(camera)
    if not started:
        raise HTTPException(status_code=409, detail="壅塞偵測啟動失敗")
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
            **{k: v for k, v in congestion_services.get(cid, {}).items() if not k.startswith("_")},
            'result': congestion_results.get(cid, {}),
            'params': get_effective_params(cid),
        }
        for cid in set(list(congestion_services.keys()) + list(congestion_results.keys()))
    }


@router.get("/samples")
def get_congestion_samples(
    camera_id: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    is_overall: bool | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(CongestionSample)
    if camera_id is not None:
        query = query.filter(CongestionSample.camera_id == camera_id)
    start_time = _to_utc_naive(start_time)
    end_time = _to_utc_naive(end_time)
    if start_time is not None:
        query = query.filter(CongestionSample.created_at >= start_time)
    if end_time is not None:
        query = query.filter(CongestionSample.created_at <= end_time)
    if is_overall is not None:
        query = query.filter(CongestionSample.is_overall == bool(is_overall))
    items = query.order_by(CongestionSample.created_at.desc()).limit(200000).all()
    return {
        "items": [
            {
                "id": row.id,
                "camera_id": row.camera_id,
                "camera_name": row.camera_name,
                "zone_name": row.zone_name,
                "lane_no": row.lane_no,
                "movement": row.movement,
                "direction": row.direction,
                "is_overall": bool(row.is_overall),
                "vehicle_count": row.vehicle_count,
                "stopped_vehicle_count": row.stopped_vehicle_count,
                "occupancy": row.occupancy,
                "raw_occupancy": row.raw_occupancy,
                "queue_score": row.queue_score,
                "estimated_queue_length_m": row.estimated_queue_length_m,
                "queue_duration_sec": row.queue_duration_sec,
                "sample_interval_sec": row.sample_interval_sec,
                "queue_active": bool(row.queue_active),
                "created_at": (
                    row.created_at.replace(tzinfo=timezone.utc).isoformat()
                    if row.created_at
                    else None
                ),
            }
            for row in items
        ]
    }


@router.get("/{camera_id}/snapshot")
def congestion_snapshot(camera_id: int, db: Session = Depends(get_db)):
    """取得單次壅塞分析截圖"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")
    
    cap = cv2.VideoCapture(resolve_analysis_source(camera))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=503, detail="無法取得影像")
    
    congestion_params[camera_id] = _load_params_from_camera(camera)
    zones = _congestion_zones(camera)
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
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")
    
    congestion_params[camera_id] = _load_params_from_camera(camera)
    zones = _congestion_zones(camera)
    return StreamingResponse(
        generate_congestion_stream(camera_id, resolve_analysis_source(camera), zones),
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
    text = (
        f"壅塞: {result.get('level_name', '-')} | "
        f"車輛: {result.get('vehicle_count', 0)} | "
        f"停滯: {result.get('stopped_vehicle_count', 0)} | "
        f"排隊: {result.get('estimated_queue_length_m', 0):.1f}m | "
        f"分數: {result.get('occupancy', 0)*100:.1f}%"
    )
    cv2.putText(annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 2)
    
    # 底部進度條
    bar_w = int((w - 20) * result.get('occupancy', 0))
    cv2.rectangle(annotated, (10, h-15), (w-10, h-5), (50,50,50), -1)
    cv2.rectangle(annotated, (10, h-15), (10 + bar_w, h-5), level_color, -1)
    
    return annotated


def run_congestion_detection(camera_id: int, camera_name: str, source: str, zones: list):
    """背景壅塞偵測任務"""
    import os as _os
    # RTSP 強制走 TCP 避免封包掉包
    if str(source).lower().startswith("rtsp://"):
        _os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                               "rtsp_transport;tcp|stimeout;5000000|buffer_size;65536")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    print(f"🚦 壅塞偵測啟動: camera_id={camera_id}")
    fail_count = 0
    last_ok = time.time()
    try:
        while congestion_services.get(camera_id, {}).get('running', False):
            if not cap.isOpened() or (time.time() - last_ok) > 10.0:
                try:
                    cap.release()
                except Exception:
                    pass
                print(f"🔄 congestion cam_{camera_id} reconnect (fail={fail_count})", flush=True)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                fail_count = 0
                last_ok = time.time()
                time.sleep(1.0)
                continue

            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"⚠️ congestion cam_{camera_id} cap.read exception: {e}", flush=True)
                ret, frame = False, None

            if not ret:
                fail_count += 1
                if fail_count >= 50:
                    # 連續 50 次讀不到 → 強制 reconnect
                    try: cap.release()
                    except: pass
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    fail_count = 0
                    last_ok = time.time()
                time.sleep(0.2)
                continue

            fail_count = 0
            last_ok = time.time()

            try:
                result = analyze_with_lock(frame, zones, camera_id)
                congestion_results[camera_id] = result
                congestion_services[camera_id]['last_update'] = datetime.now().isoformat()
                sample_interval_sec = float(get_effective_params(camera_id).get("analyze_interval_sec", 1.0))
                _store_congestion_samples(camera_id, camera_name, result, sample_interval_sec)
                time.sleep(sample_interval_sec)
            except Exception as e:
                print(f"⚠️ congestion cam_{camera_id} analyze error: {e}", flush=True)
                time.sleep(1.0)
    except Exception as e:
        congestion_services[camera_id]["error"] = str(e)
    finally:
        congestion_services[camera_id]["running"] = False
        try: cap.release()
        except: pass
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
    zones = _congestion_zones(camera)
    worker = threading.Thread(
        target=run_congestion_detection,
        args=(camera_id, camera.name, resolve_analysis_source(camera), zones),
        daemon=True,
        name=f"congestion-{camera_id}",
    )
    congestion_services[camera_id]["thread_name"] = worker.name
    congestion_services[camera_id]["_thread"] = worker
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
    # 啟動 watchdog
    _ensure_congestion_watchdog()
    return {"total": total, "resumed": resumed}


_congestion_watchdog_started = False


def _ensure_congestion_watchdog():
    """監控 congestion thread：last_update 超過 60s 沒推進就重啟該 cam thread"""
    global _congestion_watchdog_started
    if _congestion_watchdog_started:
        return
    _congestion_watchdog_started = True

    def _watchdog_loop():
        while True:
            try:
                time.sleep(30.0)
                now = datetime.now()
                for camera_id, info in list(congestion_services.items()):
                    if not info.get("running"):
                        continue
                    last = info.get("last_update")
                    if not last:
                        continue
                    try:
                        dt = datetime.fromisoformat(last)
                        age = (now - dt).total_seconds()
                    except Exception:
                        continue
                    if age > 60.0:
                        print(f"🐕 [congestion watchdog] cam_{camera_id} stale {age:.0f}s，重啟", flush=True)
                        try:
                            info["running"] = False
                            time.sleep(2.0)
                            db = SessionLocal()
                            try:
                                cam = db.query(Camera).filter(Camera.id == camera_id).first()
                                if cam:
                                    _start_congestion_service(cam)
                            finally:
                                db.close()
                        except Exception as e:
                            print(f"⚠️ [congestion watchdog] 重啟 cam_{camera_id} 失敗: {e}", flush=True)
            except Exception as e:
                print(f"⚠️ [congestion watchdog] loop error: {e}", flush=True)

    threading.Thread(target=_watchdog_loop, daemon=True, name="congestion-watchdog").start()
    print("🐕 [congestion watchdog] 監控啟動", flush=True)
