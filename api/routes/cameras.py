#!/usr/bin/env python3
"""攝影機 API"""
import base64
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
import asyncio
import cv2
import numpy as np
import re
from urllib.parse import urlparse
import requests

from api.routes.logs import add_log
from api.models import get_db, Camera
from api.utils.camera_stream import resolve_capture_source
from api.utils.feature_state import set_feature_state

router = APIRouter(prefix="/api/cameras", tags=["攝影機"])
TEST_SOURCE_TIMEOUT_SEC = 8.0
CAMERA_SOURCE_DIR = Path("./output/camera_sources")
CAMERA_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANALYZE_MODELS_LOCK = threading.Lock()
_ANALYZE_MODELS: dict[str, object] = {}
_FRAME_TRACK_SESSIONS_LOCK = threading.Lock()
_FRAME_TRACK_SESSIONS: dict[str, dict] = {}
_FRAME_TRACK_SESSION_TTL_SEC = 20.0


class CameraCreate(BaseModel):
    name: str
    source: Optional[str] = ""
    ip: Optional[str] = ""
    username: Optional[str] = ""
    password: Optional[str] = ""
    port: Optional[str] = "554"
    stream_path: Optional[str] = ""
    location: Optional[str] = ""
    detection_config: Optional[dict] = None
    enabled: bool = True


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    ip: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    port: Optional[str] = None
    stream_path: Optional[str] = None
    location: Optional[str] = None
    detection_config: Optional[dict] = None
    enabled: Optional[bool] = None
    detection_enabled: Optional[bool] = None
    zones: Optional[list] = None
    zones: Optional[list] = None
    status: Optional[str] = None


class TestUrlRequest(BaseModel):
    url: str


class AnalyzeSourceRequest(BaseModel):
    url: str
    sample_count: int = 6
    rois: Optional[list[dict]] = None
    current_time: Optional[float] = None


class AnalyzeFrameRequest(BaseModel):
    image_base64: str
    rois: Optional[list[dict]] = None
    current_time: Optional[float] = None
    session_id: Optional[str] = None
    reset_tracker: bool = False


class ImportSourceFileRequest(BaseModel):
    path: str


def _normalize_bbox_dict(bbox) -> dict:
    if not isinstance(bbox, dict):
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "width": 0, "height": 0}
    x1 = int(float(bbox.get("x1", 0) or 0))
    y1 = int(float(bbox.get("y1", 0) or 0))
    x2 = int(float(bbox.get("x2", 0) or 0))
    y2 = int(float(bbox.get("y2", 0) or 0))
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": max(0, x2 - x1),
        "height": max(0, y2 - y1),
    }


def _apply_roi_mask(frame: np.ndarray, polygons: list[list[list[int]]]) -> np.ndarray:
    if frame is None or getattr(frame, "size", 0) == 0 or not polygons:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    valid_polygons = [np.array(points, dtype=np.int32) for points in polygons if len(points) >= 3]
    if not valid_polygons:
        return frame
    cv2.fillPoly(mask, valid_polygons, 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def _get_analyze_models():
    with _ANALYZE_MODELS_LOCK:
        detector = _ANALYZE_MODELS.get("detector")
        fallback_detector = _ANALYZE_MODELS.get("fallback_detector")
        plate_detector = _ANALYZE_MODELS.get("plate_detector")
        plate_error = _ANALYZE_MODELS.get("plate_error")

        if detector is None or fallback_detector is None:
            from detection.vehicle_detector import VehicleDetector

            detector = VehicleDetector(conf_threshold=0.12)
            fallback_detector = VehicleDetector(conf_threshold=0.05)
            _ANALYZE_MODELS["detector"] = detector
            _ANALYZE_MODELS["fallback_detector"] = fallback_detector

        if "plate_detector" not in _ANALYZE_MODELS:
            try:
                from api.routes.lpr import get_plate_detector

                plate_detector = get_plate_detector()
                plate_error = "OCR 僅在車牌辨識流程啟用"
            except Exception as plate_exc:
                plate_detector = None
                plate_error = str(plate_exc)
            _ANALYZE_MODELS["plate_detector"] = plate_detector
            _ANALYZE_MODELS["plate_error"] = plate_error

        return detector, fallback_detector, _ANALYZE_MODELS.get("plate_detector"), str(_ANALYZE_MODELS.get("plate_error") or "OCR 僅在車牌辨識流程啟用")


def _prune_frame_track_sessions(now_monotonic: Optional[float] = None):
    now_monotonic = now_monotonic or time.monotonic()
    expired = [
        session_id
        for session_id, state in _FRAME_TRACK_SESSIONS.items()
        if (now_monotonic - float(state.get("updated_at", now_monotonic))) > _FRAME_TRACK_SESSION_TTL_SEC
    ]
    for session_id in expired:
        _FRAME_TRACK_SESSIONS.pop(session_id, None)


def _get_frame_track_session(session_id: str, *, reset: bool = False) -> dict:
    key = re.sub(r"[^A-Za-z0-9._:-]+", "", str(session_id or "").strip())[:128]
    if not key:
        return {}
    now_monotonic = time.monotonic()
    with _FRAME_TRACK_SESSIONS_LOCK:
        _prune_frame_track_sessions(now_monotonic)
        state = _FRAME_TRACK_SESSIONS.get(key)
        if reset or state is None:
            from detection.violation_detector import VehicleTracker

            state = {
                "tracker": VehicleTracker(max_age=6, iou_threshold=0.15),
                "meta": {},
                "updated_at": now_monotonic,
            }
            _FRAME_TRACK_SESSIONS[key] = state
        else:
            state["updated_at"] = now_monotonic
        state["session_id"] = key
        return state


def _apply_frame_tracking(session_id: Optional[str], detections: list[dict], *, reset: bool = False) -> list[dict]:
    state = _get_frame_track_session(str(session_id or ""), reset=reset)
    if not state:
        return [
            {
                "track_id": None,
                "class_name": str(det.get("class_name") or "vehicle"),
                "confidence": float(det.get("confidence") or 0.0),
                "bbox": _normalize_bbox_dict(det.get("bbox") or {}),
                "stale": False,
                "missed_frames": 0,
            }
            for det in detections or []
        ]

    tracker = state["tracker"]
    meta = state["meta"]
    tracker_input = []
    for det in detections or []:
        tracker_input.append({
            "class_name": str(det.get("class_name") or "vehicle"),
            "confidence": float(det.get("confidence") or 0.0),
            "bbox": _normalize_bbox_dict(det.get("bbox") or {}),
        })

    tracked_now = tracker.update(tracker_input) or []
    matched_ids: set[int] = set()
    for det in tracked_now:
        track_id = int(det.get("track_id"))
        matched_ids.add(track_id)
        meta[track_id] = {
            "class_name": str(det.get("class_name") or "vehicle"),
            "confidence": float(det.get("confidence") or 0.0),
            "bbox": _normalize_bbox_dict(det.get("bbox") or {}),
            "updated_at": time.monotonic(),
        }

    alive_ids = set(int(track_id) for track_id in tracker.tracks.keys())
    for track_id in list(meta.keys()):
        if int(track_id) not in alive_ids:
            meta.pop(track_id, None)

    output: list[dict] = []
    for det in tracked_now:
        track_int = int(det.get("track_id"))
        output.append({
            "track_id": track_int,
            "class_name": str(det.get("class_name") or "vehicle"),
            "confidence": float(det.get("confidence") or 0.0),
            "bbox": _normalize_bbox_dict(det.get("bbox") or {}),
            "stale": False,
            "missed_frames": 0,
        })

    output.sort(key=lambda item: int(item.get("track_id") or 0))
    state["updated_at"] = time.monotonic()
    return output


def _server_source_browse_roots() -> list[Path]:
    roots: list[Path] = []
    home_dir = Path.home()
    env_dirs = str(os.getenv("CAMERA_SOURCE_BROWSE_DIRS") or "").strip()
    if env_dirs:
        for raw in env_dirs.split(os.pathsep):
            text = str(raw or "").strip()
            if text:
                roots.append(Path(text))
    roots.extend([
        _PROJECT_ROOT / "host_downloads",
        _PROJECT_ROOT / "downloads",
        _PROJECT_ROOT / "Downloads",
        home_dir / "Downloads",
        home_dir / "downloads",
    ])
    uniq: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


def _server_source_allowed_roots() -> list[Path]:
    roots = [CAMERA_SOURCE_DIR]
    roots.extend(_server_source_browse_roots())
    return roots


def _iter_server_mp4_files(limit: int = 200) -> list[dict]:
    items: list[dict] = []
    for root in _server_source_browse_roots():
        try:
            resolved_root = root.resolve()
        except Exception:
            continue
        if not resolved_root.exists() or not resolved_root.is_dir():
            continue
        try:
            paths = []
            for ext in ALLOWED_VIDEO_EXTS:
                paths.extend(resolved_root.rglob(f"*{ext}"))
            for path in sorted(paths):
                if not path.is_file():
                    continue
                try:
                    rel = path.relative_to(resolved_root)
                except Exception:
                    rel = Path(path.name)
                items.append({
                    "path": str(path),
                    "label": f"{resolved_root.name}/{rel.as_posix()}",
                    "size": int(path.stat().st_size or 0),
                    "root": str(resolved_root),
                })
                if len(items) >= limit:
                    return items
        except Exception:
            continue
    return items


def _resolve_allowed_source_file(path_text: str) -> Path:
    raw = str(path_text or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="檔案路徑不得為空")
    candidate = Path(raw)
    try:
        resolved = candidate.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="檔案路徑無效")
    if resolved.suffix.lower() not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"只支援 {', '.join(e[1:].upper() for e in ALLOWED_VIDEO_EXTS)} 檔案")
    for root in _server_source_allowed_roots():
        try:
            resolved.relative_to(root.resolve())
            if resolved.exists() and resolved.is_file():
                return resolved
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="找不到可用的主機 MP4 檔案")


@router.post("/upload-source")
def upload_camera_source(file: UploadFile = File(...)):
    filename = str(file.filename or "").strip()
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"只支援 {', '.join(e[1:].upper() for e in ALLOWED_VIDEO_EXTS)} 檔案")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._-") or "camera_source"
    stored_name = f"{safe_name}_{uuid.uuid4().hex[:8]}{ext}"
    target = CAMERA_SOURCE_DIR / stored_name
    try:
        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        try:
            file.file.close()
        except Exception:
            pass
    return {
        "status": "success",
        "filename": stored_name,
        "source": f"/files/camera_sources/{stored_name}",
    }


@router.get("/source-files")
async def list_source_files():
    files = _iter_server_mp4_files()
    return {
        "files": files,
        "roots": [str(p) for p in _server_source_browse_roots() if p.exists()],
    }


@router.post("/import-source-file")
def import_source_file(data: ImportSourceFileRequest):
    source_path = _resolve_allowed_source_file(data.path)
    if source_path.parent.resolve() == CAMERA_SOURCE_DIR.resolve():
        return {
            "status": "success",
            "filename": source_path.name,
            "source": f"/files/camera_sources/{source_path.name}",
            "original_path": str(source_path),
        }
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem).strip("._-") or "camera_source"
    stored_name = f"{safe_name}_{uuid.uuid4().hex[:8]}{source_path.suffix.lower()}"
    target = CAMERA_SOURCE_DIR / stored_name
    shutil.copy2(source_path, target)
    return {
        "status": "success",
        "filename": stored_name,
        "source": f"/files/camera_sources/{stored_name}",
        "original_path": str(source_path),
    }


def _normalize_camera_name(name: Optional[str]) -> str:
    return str(name or "").strip()


def _normalize_camera_source(source: Optional[str]) -> str:
    raw = str(source or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        scheme = str(parsed.scheme or "").lower()
        hostname = str(parsed.hostname or "").lower()
        port = parsed.port
        if scheme in {"rtsp", "rtsps"}:
            port = port or 554
        elif scheme in {"http", "https"}:
            port = port or (443 if scheme == "https" else 80)
        authless_path = parsed.path or ""
        query = f"?{parsed.query}" if parsed.query else ""
        if hostname:
            if port:
                return f"{scheme}://{hostname}:{port}{authless_path}{query}"
            return f"{scheme}://{hostname}{authless_path}{query}"
    except Exception:
        pass
    return re.sub(r"//[^@]*@", "//", raw).lower()


def _find_duplicate_camera(
    db: Session,
    *,
    name: Optional[str],
    source: Optional[str],
    exclude_camera_id: Optional[int] = None,
):
    normalized_name = _normalize_camera_name(name)
    normalized_source = _normalize_camera_source(source)
    for camera in db.query(Camera).all():
        if exclude_camera_id is not None and int(camera.id) == int(exclude_camera_id):
            continue
        if normalized_name and _normalize_camera_name(camera.name).lower() == normalized_name.lower():
            return camera, "name"
        if normalized_source and _normalize_camera_source(camera.source) == normalized_source:
            return camera, "source"
    return None, None


@router.get("")
async def get_cameras(db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    return {"total": len(cameras), "items": [_to_dict(c) for c in cameras]}


@router.get("/statistics")
async def get_camera_statistics(db: Session = Depends(get_db)):
    total = db.query(Camera).count()
    online = db.query(Camera).filter(Camera.status == "online").count()
    return {"total": total, "online": online, "offline": total - online}


@router.get("/{camera_id}")
async def get_camera(camera_id: int, db: Session = Depends(get_db)):
    c = db.query(Camera).filter(Camera.id == camera_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    return _to_dict(c)


@router.post("")
async def create_camera(data: CameraCreate, db: Session = Depends(get_db)):
    name = _normalize_camera_name(data.name)
    source = str(data.source or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="攝影機名稱不得為空")
    if not source:
        raise HTTPException(status_code=400, detail="攝影機來源不得為空")
    dup, dup_field = _find_duplicate_camera(db, name=name, source=source)
    if dup is not None:
        detail = f"攝影機名稱重複: {dup.name}" if dup_field == "name" else f"攝影機來源重複: {dup.source}"
        raise HTTPException(status_code=409, detail=detail)
    c = Camera(
        name=name,
        source=source,
        ip=data.ip,
        username=data.username,
        password=data.password,
        port=data.port or "554",
        stream_path=data.stream_path,
        location=data.location,
        detection_config=data.detection_config or {
            "red_light": True,
            "speeding": True,
            "illegal_parking": True,
            "wrong_way": False,
            "no_helmet": False,
            "speed_limit": 50
        },
        enabled=data.enabled
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return _to_dict(c)


@router.put("/{camera_id}")
async def update_camera(camera_id: int, data: CameraUpdate, db: Session = Depends(get_db)):
    c = db.query(Camera).filter(Camera.id == camera_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    previous_enabled = bool(c.enabled)

    payload = data.dict(exclude_unset=True)
    next_name = _normalize_camera_name(payload.get("name", c.name))
    next_source = str(payload.get("source", c.source) or "").strip()
    dup, dup_field = _find_duplicate_camera(
        db,
        name=next_name,
        source=next_source,
        exclude_camera_id=camera_id,
    )
    if dup is not None:
        detail = f"攝影機名稱重複: {dup.name}" if dup_field == "name" else f"攝影機來源重複: {dup.source}"
        raise HTTPException(status_code=409, detail=detail)

    if "name" in payload:
        payload["name"] = next_name
    if "source" in payload:
        payload["source"] = next_source

    for key, value in payload.items():
        setattr(c, key, value)
    if "enabled" in payload and not bool(payload.get("enabled")):
        try:
            from api.routes.stream import detection_services
            if camera_id in detection_services:
                detection_services[camera_id]["running"] = False
        except Exception:
            pass
        try:
            from api.routes.congestion import congestion_services
            if camera_id in congestion_services:
                congestion_services[camera_id]["running"] = False
        except Exception:
            pass
        try:
            from api.routes.lpr_stream import _lpr_tasks
            if camera_id in _lpr_tasks:
                _lpr_tasks[camera_id].stop()
                del _lpr_tasks[camera_id]
        except Exception:
            pass
        c.detection_enabled = False
        c.status = "offline"
        set_feature_state("detection", camera_id, False)
        set_feature_state("congestion", camera_id, False)
        set_feature_state("lpr", camera_id, False)
    elif "enabled" in payload and bool(payload.get("enabled")) and not previous_enabled:
        c.status = c.status or "offline"
    c.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(c)
    return _to_dict(c)


@router.delete("/{camera_id}")
async def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    c = db.query(Camera).filter(Camera.id == camera_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    db.delete(c)
    db.commit()
    return {"message": "已刪除"}


@router.post("/{camera_id}/test")
async def test_camera(camera_id: int, db: Session = Depends(get_db)):
    """測試攝影機來源連線"""
    c = db.query(Camera).filter(Camera.id == camera_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    
    try:
        result = await _run_test_source(c.source, c.name)
    except asyncio.TimeoutError:
        add_log("error", f"連線逾時: {c.name}", "camera")
        result = {"status": "error", "message": "連線逾時（來源回應過慢）"}
    if result["status"] == "success":
        c.status = "online"
        c.last_seen = datetime.utcnow()
    else:
        c.status = "offline"
    db.commit()
    return result


@router.post("/test-url")
async def test_url(data: TestUrlRequest):
    """測試攝影機來源 URL 連線"""
    name = str(data.url or "").strip()[:40] or "unknown"
    try:
        return await _run_test_source(data.url, name)
    except asyncio.TimeoutError:
        add_log("error", f"連線逾時: {name}", "camera")
        return {"status": "error", "message": "連線逾時（來源回應過慢）"}


@router.post("/analyze-source")
def analyze_source(data: AnalyzeSourceRequest):
    """快速分析任意來源，供獨立測試頁使用。"""
    source = str(data.url or "").strip()
    if not source:
        raise HTTPException(status_code=400, detail="來源不得為空")

    sample_count = max(1, min(int(data.sample_count or 6), 12))
    roi_defs = list(data.rois or [])
    requested_time = None
    if data.current_time is not None:
        try:
            requested_time = max(0.0, float(data.current_time))
        except Exception:
            requested_time = None
    resolved = resolve_capture_source(source)
    cap = cv2.VideoCapture(resolved)
    if not cap.isOpened():
        raise HTTPException(status_code=503, detail="無法分析來源")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    try:
        detector, fallback_detector, plate_detector, plate_analysis_error = _get_analyze_models()
        plate_analysis_available = False

        positions: list[int] = []
        analysis_mode = "samples"
        if requested_time is not None and fps > 0 and total_frames > 0:
            analysis_mode = "current_frame"
            positions = [max(0, min(total_frames - 1, int(round(requested_time * fps))))]
        elif total_frames > 0:
            positions = sorted({max(0, min(total_frames - 1, int(total_frames * i / (sample_count + 1)))) for i in range(1, sample_count + 1)})
        elif requested_time is not None:
            analysis_mode = "current_frame"

        sample_results = []
        best_plate_result = {"plate_number": None, "confidence": 0.0, "valid": False, "raw": ""}
        unique_plates: list[str] = []
        max_vehicle_count = 0
        total_plate_boxes = 0
        frames_with_vehicles = 0
        roi_reports: list[dict] = []
        preview_image = None
        preview_meta = {}
        overlay_result = None
        detector_threshold_used = float(getattr(detector, "conf_threshold", 0.12) or 0.12)
        detector_fallback_used = False
        detector_runtime = str(getattr(detector, "runtime_device", "unknown") or "unknown")

        def _append_plate(plate: str):
            text = str(plate or "").strip().upper()
            if text and text not in unique_plates:
                unique_plates.append(text)

        def _normalize_roi_points(points, frame_w: int, frame_h: int):
            output = []
            for point in points or []:
                try:
                    px = float(point.get("x"))
                    py = float(point.get("y"))
                except Exception:
                    continue
                if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                    output.append([int(px * frame_w), int(py * frame_h)])
                else:
                    output.append([int(px), int(py)])
            return output

        def _bbox_anchor_xy(bbox):
            if isinstance(bbox, dict):
                return (
                    int((float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2.0),
                    int(float(bbox.get("y2", 0))),
                )
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return (
                    int((float(bbox[0]) + float(bbox[2])) / 2.0),
                    int(float(bbox[3])),
                )
            return (0, 0)

        def _point_in_roi(point_xy, polygon_points) -> bool:
            if len(polygon_points) < 3:
                return False
            contour = np.array(polygon_points, dtype=np.int32)
            return cv2.pointPolygonTest(contour, point_xy, False) >= 0

        def _filter_items_in_rois(items, polygons):
            if not polygons:
                return list(items)
            filtered = []
            for item in items or []:
                anchor = _bbox_anchor_xy(item.get("bbox"))
                if any(_point_in_roi(anchor, polygon) for polygon in polygons):
                    filtered.append(item)
            return filtered

        def _item_hits_any_roi(item, polygons) -> bool:
            if not polygons:
                return True
            anchor = _bbox_anchor_xy(item.get("bbox"))
            return any(_point_in_roi(anchor, polygon) for polygon in polygons)

        def _item_hits_any_roi(item, polygons) -> bool:
            if not polygons:
                return True
            anchor = _bbox_anchor_xy(item.get("bbox"))
            return any(_point_in_roi(anchor, polygon) for polygon in polygons)

        def _collect_roi_stats(frame, detections, plate_boxes):
            frame_stats = []
            for idx, roi in enumerate(roi_defs):
                points = _normalize_roi_points(roi.get("points"), frame.shape[1], frame.shape[0])
                if len(points) < 3:
                    continue
                while len(roi_reports) <= idx:
                    roi_reports.append({
                        "name": str((roi_defs[len(roi_reports)] or {}).get("name") or f"ROI {len(roi_reports) + 1}"),
                        "frames_hit": 0,
                        "vehicle_total": 0,
                        "plate_box_total": 0,
                        "max_vehicle_count": 0,
                    })
                roi_vehicle_count = sum(1 for det in detections if _point_in_roi(_bbox_anchor_xy(det.get("bbox")), points))
                roi_plate_count = sum(1 for det in plate_boxes if _point_in_roi(_bbox_anchor_xy(det.get("bbox")), points))
                if roi_vehicle_count or roi_plate_count:
                    roi_reports[idx]["frames_hit"] += 1
                roi_reports[idx]["vehicle_total"] += roi_vehicle_count
                roi_reports[idx]["plate_box_total"] += roi_plate_count
                roi_reports[idx]["max_vehicle_count"] = max(roi_reports[idx]["max_vehicle_count"], roi_vehicle_count)
                frame_stats.append({
                    "name": roi_reports[idx]["name"],
                    "vehicle_count": roi_vehicle_count,
                    "plate_boxes": roi_plate_count,
                })
            return frame_stats

        def _draw_bbox(frame, bbox, color, label: str = ""):
            if isinstance(bbox, dict):
                x1 = int(float(bbox.get("x1", 0)))
                y1 = int(float(bbox.get("y1", 0)))
                x2 = int(float(bbox.get("x2", 0)))
                y2 = int(float(bbox.get("y2", 0)))
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
            else:
                return
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        def _encode_preview(frame, detections, plate_boxes, frame_roi_stats, frame_index: int):
            annotated = frame.copy()
            for idx, roi in enumerate(roi_defs):
                points = _normalize_roi_points(roi.get("points"), annotated.shape[1], annotated.shape[0])
                if len(points) < 3:
                    continue
                contour = np.array(points, dtype=np.int32)
                cv2.polylines(annotated, [contour], True, (20, 184, 166), 2)
                px, py = points[0]
                cv2.putText(
                    annotated,
                    str(roi.get("name") or f"ROI {idx + 1}"),
                    (int(px), max(20, int(py) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (20, 184, 166),
                    2,
                    cv2.LINE_AA,
                )
            for det in detections:
                label = str(det.get("class_name") or "vehicle")
                conf = float(det.get("confidence") or 0.0)
                _draw_bbox(annotated, det.get("bbox"), (34, 197, 94), f"{label} {conf:.2f}")
            footer = f"frame {frame_index}"
            if fps > 0:
                footer += f" | {frame_index / fps:.2f}s"
            if frame_roi_stats:
                roi_text = " | ".join(
                    f"{item['name']}:car {item['vehicle_count']}, plate {item['plate_boxes']}"
                    for item in frame_roi_stats
                )
                footer = f"{footer} | {roi_text}"
            cv2.putText(
                annotated,
                footer[:180],
                (16, max(28, annotated.shape[0] - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ok:
                return None
            return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

        def _analyze_frame(frame, frame_index: int):
            nonlocal max_vehicle_count, total_plate_boxes, frames_with_vehicles, best_plate_result, preview_image, preview_meta, overlay_result, detector_threshold_used, detector_fallback_used
            roi_polygons = [
                _normalize_roi_points(roi.get("points"), frame.shape[1], frame.shape[0])
                for roi in roi_defs
            ]
            roi_polygons = [polygon for polygon in roi_polygons if len(polygon) >= 3]
            detections = detector.detect(frame) or []
            if not detections:
                fallback_detections = fallback_detector.detect(frame) or []
                if fallback_detections:
                    detections = fallback_detections
                    detector_fallback_used = True
                    detector_threshold_used = float(getattr(fallback_detector, "conf_threshold", 0.05) or 0.05)
            plate_boxes = plate_detector.detect(frame) or [] if plate_detector is not None else []
            detections = _filter_items_in_rois(detections, roi_polygons)
            plate_boxes = _filter_items_in_rois(plate_boxes, roi_polygons)
            vehicle_count = len(detections)
            plate_count = len(plate_boxes)
            best = {"plate_number": None, "confidence": 0.0, "valid": False, "raw": ""}
            frame_roi_stats = _collect_roi_stats(frame, detections, plate_boxes)
            max_vehicle_count = max(max_vehicle_count, vehicle_count)
            total_plate_boxes += plate_count
            if vehicle_count > 0:
                frames_with_vehicles += 1
            if float(best.get("confidence") or 0.0) > float(best_plate_result.get("confidence") or 0.0):
                best_plate_result = best
            _append_plate(best.get("plate_number"))
            sample_results.append({
                "frame_index": frame_index,
                "time_sec": round(frame_index / fps, 3) if fps > 0 else None,
                "vehicle_count": vehicle_count,
                "plate_boxes": plate_count,
                "plate_number": best.get("plate_number"),
                "confidence": float(best.get("confidence") or 0.0),
                "valid": bool(best.get("valid", False)),
                "roi_stats": frame_roi_stats,
            })
            if preview_image is None:
                preview_image = _encode_preview(frame, detections, plate_boxes, frame_roi_stats, frame_index)
                preview_meta = {
                    "frame_index": frame_index,
                    "time_sec": round(frame_index / fps, 3) if fps > 0 else None,
                }
            raw_detections_overlay = [
                {
                    **det,
                    "roi_hit": _item_hits_any_roi(det, roi_polygons),
                }
                for det in detections
            ]
            overlay_result = {
                "frame_index": frame_index,
                "time_sec": round(frame_index / fps, 3) if fps > 0 else None,
                "raw_detections": raw_detections_overlay,
                "detections": [
                    {
                        "class_name": str(det.get("class_name") or "vehicle"),
                        "confidence": float(det.get("confidence") or 0.0),
                        "bbox": det.get("bbox") or {},
                        "roi_hit": _item_hits_any_roi(det, roi_polygons),
                    }
                    for det in detections
                ],
                "plate_boxes": [
                    {
                        "bbox": det.get("bbox") or {},
                    }
                    for det in plate_boxes
                ],
                "roi_polygons": [
                    {
                        "name": str(roi.get("name") or f"ROI {idx + 1}"),
                        "points": _normalize_roi_points(roi.get("points"), frame.shape[1], frame.shape[0]),
                    }
                    for idx, roi in enumerate(roi_defs)
                    if len(_normalize_roi_points(roi.get("points"), frame.shape[1], frame.shape[0])) >= 3
                ],
                "roi_stats": frame_roi_stats,
            }

        try:
            if positions:
                for pos in positions:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    _analyze_frame(frame, pos)
            elif requested_time is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, requested_time * 1000.0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 1)
                    _analyze_frame(frame, max(0, current_pos - 1))
            else:
                for idx in range(sample_count):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    _analyze_frame(frame, idx)
        finally:
            cap.release()

        if not sample_results:
            raise HTTPException(status_code=503, detail="無法分析來源")

        return {
            "status": "success",
            "mode": analysis_mode,
            "source": source,
            "resolved_source": resolved,
            "requested_time": requested_time,
            "media": {
                "width": width,
                "height": height,
                "fps": round(fps, 2) if fps else 0,
                "total_frames": total_frames,
                "sample_count": len(sample_results),
            },
            "analysis": {
                "frames_with_vehicles": frames_with_vehicles,
                "max_vehicle_count": max_vehicle_count,
                "total_plate_boxes": total_plate_boxes,
                "best_plate": best_plate_result.get("plate_number"),
                "best_confidence": float(best_plate_result.get("confidence") or 0.0),
                "best_valid": bool(best_plate_result.get("valid", False)),
                "plates": unique_plates,
                "detector_threshold": detector_threshold_used,
                "detector_fallback_used": detector_fallback_used,
                "detector_runtime": detector_runtime,
                "plate_analysis_available": plate_analysis_available,
                "plate_analysis_error": plate_analysis_error,
            },
            "roi_report": roi_reports,
            "overlay": overlay_result,
            "preview_image": preview_image,
            "preview_meta": preview_meta,
            "samples": sample_results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        add_log("error", f"分析來源失敗: {resolved} - {exc}", "camera")
        raise HTTPException(status_code=503, detail=f"無法分析來源: {exc}")


@router.post("/analyze-frame")
def analyze_frame(data: AnalyzeFrameRequest):
    image_base64 = str(data.image_base64 or "").strip()
    if not image_base64:
        raise HTTPException(status_code=400, detail="影格資料不得為空")

    payload = image_base64
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"影格解碼失敗: {exc}")

    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or getattr(frame, "size", 0) == 0:
        raise HTTPException(status_code=400, detail="無法解析影格")

    roi_defs = list(data.rois or [])
    requested_time = None
    if data.current_time is not None:
        try:
            requested_time = max(0.0, float(data.current_time))
        except Exception:
            requested_time = None
    session_id = re.sub(r"[^A-Za-z0-9._:-]+", "", str(data.session_id or "").strip())[:128] or None
    reset_tracker = bool(data.reset_tracker)

    height, width = frame.shape[:2]
    fps = 0.0

    try:
        detector, fallback_detector, plate_detector, plate_analysis_error = _get_analyze_models()
        plate_analysis_available = False

        detector_threshold_used = float(getattr(detector, "conf_threshold", 0.12) or 0.12)
        detector_fallback_used = False
        detector_runtime = str(getattr(detector, "runtime_device", "unknown") or "unknown")
        roi_reports: list[dict] = []

        def _normalize_roi_points(points, frame_w: int, frame_h: int):
            output = []
            for point in points or []:
                try:
                    px = float(point.get("x"))
                    py = float(point.get("y"))
                except Exception:
                    continue
                if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                    output.append([int(px * frame_w), int(py * frame_h)])
                else:
                    output.append([int(px), int(py)])
            return output

        def _bbox_anchor_xy(bbox):
            if isinstance(bbox, dict):
                return (
                    int((float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2.0),
                    int(float(bbox.get("y2", 0))),
                )
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return (
                    int((float(bbox[0]) + float(bbox[2])) / 2.0),
                    int(float(bbox[3])),
                )
            return (0, 0)

        def _point_in_roi(point_xy, polygon_points) -> bool:
            if len(polygon_points) < 3:
                return False
            contour = np.array(polygon_points, dtype=np.int32)
            return cv2.pointPolygonTest(contour, point_xy, False) >= 0

        def _filter_items_in_rois(items, polygons):
            if not polygons:
                return list(items)
            filtered = []
            for item in items or []:
                anchor = _bbox_anchor_xy(item.get("bbox"))
                if any(_point_in_roi(anchor, polygon) for polygon in polygons):
                    filtered.append(item)
            return filtered

        def _item_hits_any_roi(item, polygons) -> bool:
            if not polygons:
                return True
            anchor = _bbox_anchor_xy(item.get("bbox"))
            return any(_point_in_roi(anchor, polygon) for polygon in polygons)

        def _collect_roi_stats(frame_img, detections, plate_boxes):
            frame_stats = []
            for idx, roi in enumerate(roi_defs):
                points = _normalize_roi_points(roi.get("points"), frame_img.shape[1], frame_img.shape[0])
                if len(points) < 3:
                    continue
                while len(roi_reports) <= idx:
                    roi_reports.append({
                        "name": str((roi_defs[len(roi_reports)] or {}).get("name") or f"ROI {len(roi_reports) + 1}"),
                        "frames_hit": 0,
                        "vehicle_total": 0,
                        "plate_box_total": 0,
                        "max_vehicle_count": 0,
                    })
                roi_vehicle_count = sum(1 for det in detections if _point_in_roi(_bbox_anchor_xy(det.get("bbox")), points))
                roi_plate_count = sum(1 for det in plate_boxes if _point_in_roi(_bbox_anchor_xy(det.get("bbox")), points))
                if roi_vehicle_count or roi_plate_count:
                    roi_reports[idx]["frames_hit"] += 1
                roi_reports[idx]["vehicle_total"] += roi_vehicle_count
                roi_reports[idx]["plate_box_total"] += roi_plate_count
                roi_reports[idx]["max_vehicle_count"] = max(roi_reports[idx]["max_vehicle_count"], roi_vehicle_count)
                frame_stats.append({
                    "name": roi_reports[idx]["name"],
                    "vehicle_count": roi_vehicle_count,
                    "plate_boxes": roi_plate_count,
                })
            return frame_stats

        def _draw_bbox(frame_img, bbox, color, label: str = ""):
            if isinstance(bbox, dict):
                x1 = int(float(bbox.get("x1", 0)))
                y1 = int(float(bbox.get("y1", 0)))
                x2 = int(float(bbox.get("x2", 0)))
                y2 = int(float(bbox.get("y2", 0)))
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
            else:
                return
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(frame_img, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        roi_polygons = [
            _normalize_roi_points(roi.get("points"), frame.shape[1], frame.shape[0])
            for roi in roi_defs
        ]
        roi_polygons = [polygon for polygon in roi_polygons if len(polygon) >= 3]
        raw_detections = detector.detect(frame) or []
        if not raw_detections:
            fallback_detections = fallback_detector.detect(frame) or []
            if fallback_detections:
                raw_detections = fallback_detections
                detector_fallback_used = True
                detector_threshold_used = float(getattr(fallback_detector, "conf_threshold", 0.05) or 0.05)
        raw_plate_boxes = plate_detector.detect(frame) or [] if plate_detector is not None else []
        detections = _filter_items_in_rois(raw_detections, roi_polygons)
        plate_boxes = _filter_items_in_rois(raw_plate_boxes, roi_polygons)
        tracked_detections = _apply_frame_tracking(session_id, detections, reset=reset_tracker)
        frame_roi_stats = _collect_roi_stats(frame, detections, plate_boxes)
        raw_detections_overlay = [
            {
                **det,
                "roi_hit": _item_hits_any_roi(det, roi_polygons),
            }
            for det in raw_detections
        ]
        tracked_detections_overlay = [
            {
                **det,
                "roi_hit": _item_hits_any_roi(det, roi_polygons),
            }
            for det in tracked_detections
        ]

        annotated = frame.copy()
        for idx, roi in enumerate(roi_defs):
            points = _normalize_roi_points(roi.get("points"), annotated.shape[1], annotated.shape[0])
            if len(points) < 3:
                continue
            contour = np.array(points, dtype=np.int32)
            cv2.polylines(annotated, [contour], True, (20, 184, 166), 2)
            px, py = points[0]
            cv2.putText(annotated, str(roi.get("name") or f"ROI {idx + 1}"), (int(px), max(20, int(py) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 184, 166), 2, cv2.LINE_AA)
        for det in tracked_detections:
            label = str(det.get("class_name") or "vehicle")
            track_id = det.get("track_id")
            stale = bool(det.get("stale"))
            color = (56, 189, 248) if stale else (34, 197, 94)
            label_text = f"{label} #{track_id}" if track_id is not None else label
            _draw_bbox(annotated, det.get("bbox"), color, label_text)
        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        preview_image = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None

        return {
            "status": "success",
            "mode": "frame_capture",
            "requested_time": requested_time,
            "media": {
                "width": width,
                "height": height,
                "fps": round(fps, 2) if fps else 0,
                "total_frames": 1,
                "sample_count": 1,
            },
            "analysis": {
                "frames_with_vehicles": 1 if raw_detections else 0,
                "raw_detection_count": len(raw_detections),
                "roi_detection_count": len(detections),
                "active_track_count": len(tracked_detections),
                "max_vehicle_count": len(detections),
                "total_plate_boxes": len(plate_boxes),
                "tracked_vehicle_count": len(tracked_detections),
                "best_plate": None,
                "best_confidence": 0.0,
                "best_valid": False,
                "plates": [],
                "detector_threshold": detector_threshold_used,
                "detector_fallback_used": detector_fallback_used,
                "detector_runtime": detector_runtime,
                "plate_analysis_available": plate_analysis_available,
                "plate_analysis_error": plate_analysis_error,
            },
            "roi_report": roi_reports,
            "overlay": {
                "frame_index": 0,
                "time_sec": requested_time,
                "session_id": session_id,
                "raw_detections": raw_detections_overlay,
                "detections": tracked_detections_overlay,
                "plate_boxes": [
                    {
                        "bbox": det.get("bbox") or {},
                    }
                    for det in plate_boxes
                ],
                "roi_polygons": [
                    {
                        "name": str(roi.get("name") or f"ROI {idx + 1}"),
                        "points": _normalize_roi_points(roi.get("points"), width, height),
                    }
                    for idx, roi in enumerate(roi_defs)
                    if len(_normalize_roi_points(roi.get("points"), width, height)) >= 3
                ],
                "roi_stats": frame_roi_stats,
            },
            "preview_image": preview_image,
            "preview_meta": {
                "frame_index": 0,
                "time_sec": requested_time,
            },
            "tracking": {
                "session_id": session_id,
                "tracked_count": len(tracked_detections),
                "reset": reset_tracker,
            },
            "samples": [{
                "frame_index": 0,
                "time_sec": requested_time,
                "vehicle_count": len(detections),
                "raw_detection_count": len(raw_detections),
                "tracked_vehicle_count": len(tracked_detections),
                "plate_boxes": len(plate_boxes),
                "plate_number": None,
                "confidence": 0.0,
                "valid": False,
                "roi_stats": frame_roi_stats,
            }],
        }
    except HTTPException:
        raise
    except Exception as exc:
        add_log("error", f"分析影格失敗: {exc}", "camera")
        raise HTTPException(status_code=503, detail=f"無法分析影格: {exc}")


async def _run_test_source(url: str, camera_name: str) -> dict:
    # Avoid UI blocking when the source is very slow to open.
    return await asyncio.wait_for(
        asyncio.to_thread(_test_source, url, camera_name),
        timeout=TEST_SOURCE_TIMEOUT_SEC,
    )


def _test_http_source(url: str, camera_name: str) -> dict:
    try:
        with requests.get(
            url,
            headers={"User-Agent": "traffic-camera-tester/1.0"},
            stream=True,
            timeout=5,
            verify=False,
        ) as response:
            status = int(response.status_code or 200)
            content_type = str(response.headers.get("Content-Type") or "").strip()
            chunk = next(response.iter_content(chunk_size=4096), b"")
        if status >= 400:
            add_log("error", f"HTTP 來源失敗: {camera_name} ({status})", "camera")
            return {"status": "error", "message": f"HTTP {status}"}
        if not chunk:
            add_log("error", f"HTTP 來源無資料: {camera_name}", "camera")
            return {"status": "error", "message": "HTTP 串流無資料"}
        detail = f"HTTP 串流連線成功 ({content_type or 'unknown'})"
        add_log("success", f"{detail}: {camera_name}", "camera")
        return {"status": "success", "message": detail}
    except requests.HTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", 500)
        add_log("error", f"HTTP 來源錯誤: {camera_name} ({code})", "camera")
        return {"status": "error", "message": f"HTTP {code}"}
    except requests.RequestException as e:
        add_log("error", f"HTTP 來源連線錯誤: {camera_name} - {e}", "camera")
        return {"status": "error", "message": f"HTTP 連線錯誤: {e}"}


def _test_source(url: str, camera_name: str = None) -> dict:
    """測試攝影機來源連線"""
    name = camera_name or url[:40] if url else "unknown"
    add_log("info", f"開始測試: {name}", "camera")
    if not url:
        add_log("error", f"URL 為空: {name}", "camera")
        return {"status": "error", "message": "URL 為空"}
    resolved = resolve_capture_source(url)
    if resolved.startswith("http://") or resolved.startswith("https://"):
        return _test_http_source(resolved, name)
    try:
        cap = cv2.VideoCapture(resolved)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            add_log("error", f"無法連線: {name}", "camera")
            return {"status": "error", "message": "無法連線到攝影機"}
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            add_log("success", f"連線成功: {name} ({w}x{h})", "camera")
            return {"status": "success", "message": f"連線成功 ({w}x{h})"}
        else:
            add_log("error", f"無法讀取影像: {name}", "camera")
            return {"status": "error", "message": "無法讀取影像"}
    except Exception as e:
        add_log("error", f"連線錯誤: {name} - {str(e)}", "camera")
        return {"status": "error", "message": f"連線錯誤: {str(e)}"}

        return {"status": "error", "message": f"連線錯誤: {str(e)}"}

def _to_dict(c: Camera) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "source": c.source,
        "ip": c.ip,
        "username": c.username,
        "password": c.password,
        "port": c.port,
        "stream_path": c.stream_path,
        "location": c.location,
        "detection_config": c.detection_config,
        "zones": c.zones,
        "status": c.status,
        "enabled": c.enabled,
        "detection_enabled": c.detection_enabled,
        "total_violations": c.total_violations,
        "today_violations": c.today_violations,
        "last_seen": c.last_seen.isoformat() if c.last_seen else None,
        "created_at": c.created_at.isoformat() if c.created_at else None
    }
