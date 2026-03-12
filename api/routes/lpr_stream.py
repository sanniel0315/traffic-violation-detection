#!/usr/bin/env python3
"""車牌辨識串流 API - YOLO + Tesseract"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import cv2
import time
import threading
import os
from typing import Dict, Any, List, Optional
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import re

sys.path.insert(0, '/workspace')

from model_paths import get_detect_model_engine, get_detect_model_pt
from api.models import SessionLocal, Camera
from api.utils.feature_state import get_feature_state, set_feature_state

router = APIRouter(prefix="/api/lpr/stream", tags=["lpr-stream"])

_lpr_tasks: Dict[int, 'LPRStreamTask'] = {}
_yolo_model = None
_yolo_lock = threading.Lock()
_recognizer = None

SNAPSHOT_DIR = '/workspace/storage/lpr_snapshots'
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
TZ_TAIPEI = ZoneInfo("Asia/Taipei")

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = get_detect_model_engine()
        if not os.path.exists(model_path):
            model_path = get_detect_model_pt()
        _yolo_model = YOLO(model_path, task='detect')
    return _yolo_model

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from recognition.plate_recognizer import PlateRecognizer
        _recognizer = PlateRecognizer()
    return _recognizer


def _fmt_taipei(dt: datetime) -> str:
    if not dt:
        return ""
    # Existing DB rows are naive UTC; normalize then convert.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")


class LPRStreamTask:
    """LPR 串流辨識"""
    
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    def __init__(self, camera_id: int, source: str, camera_name: str):
        self.camera_id = camera_id
        self.source = source
        self.camera_name = camera_name
        self.running = False
        self.thread = None
        self.results = []
        self.max_results = 100
        self.last_plates = {}
        self.cooldown = 5
        self.total_frames = 0
        self.total_detections = 0
        self.vehicles_detected = 0
        self.zones = []
        self.last_error = None
        self.last_frame_at = 0.0
        self.started_at = time.time()

    def _save_record_db(self, record: dict):
        try:
            from api.models import SessionLocal, LPRRecord
            db = SessionLocal()
            try:
                row = LPRRecord(
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    plate_number=record.get("plate_number"),
                    confidence=float(record.get("confidence", 0)),
                    valid=bool(record.get("valid", False)),
                    vehicle_type=record.get("vehicle_type"),
                    snapshot=record.get("snapshot"),
                    raw=record.get("raw"),
                    created_at=datetime.utcnow(),
                )
                db.add(row)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            self.last_error = f"db_write_failed: {e}"

    def _normalize_plate_candidate(self, plate: str) -> str:
        if not plate:
            return ""
        t = plate.strip().upper().replace(" ", "")
        t = re.sub(r"[^A-Z0-9-]", "", t)
        return t

    def _plate_variants(self, plate: str) -> List[str]:
        plate = self._normalize_plate_candidate(plate)
        if not plate:
            return []

        to_digit = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
        to_alpha = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
        variants = set()
        variants.add(plate)

        plain = plate.replace("-", "")
        if 5 <= len(plain) <= 8:
            for cut in (2, 3, 4):
                if 2 <= cut < len(plain) - 1:
                    variants.add(f"{plain[:cut]}-{plain[cut:]}")

        for cand in list(variants):
            if "-" not in cand:
                continue
            left, right = cand.split("-", 1)
            fixed = f"{''.join(to_alpha.get(ch, ch) for ch in left)}-{''.join(to_digit.get(ch, ch) for ch in right)}"
            variants.add(fixed)

        out = []
        for v in variants:
            n = self._normalize_plate_candidate(v)
            core_len = len(n.replace("-", ""))
            if 5 <= core_len <= 8:
                out.append(n)
        return out

    def _score_ocr_result(self, result: Dict[str, Any], plate: str) -> float:
        conf = float(result.get("confidence") or 0.0)
        valid = bool(result.get("valid", False))
        score = conf * 1.5
        if valid:
            score += 2.5
        if re.match(r"^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$", plate):
            score += 0.3
        if re.match(r"^[A-Z]{2,4}-[0-9]{2,4}$", plate):
            score += 0.5
        return score

    def _recognize_plate_best(self, frame, x1: int, y1: int, x2: int, y2: int, recognizer) -> Dict[str, Any]:
        vh, vw = max(1, y2 - y1), max(1, x2 - x1)
        y_ratios = (0.30, 0.40, 0.50, 0.60, 0.68)
        x_trim_ratios = (0.00, 0.04, 0.08)

        best: Dict[str, Any] = {"plate_number": None, "confidence": 0.0, "valid": False, "raw": "", "_score": -1.0}
        aggregate: Dict[str, Dict[str, Any]] = {}

        for yr in y_ratios:
            py1 = y1 + int(vh * yr)
            for xr in x_trim_ratios:
                trim = int(vw * xr)
                px1, px2 = x1 + trim, x2 - trim
                if px2 - px1 < 28 or y2 - py1 < 16:
                    continue
                roi = frame[py1:y2, px1:px2]
                if roi.size == 0:
                    continue
                rh, rw = roi.shape[:2]
                if rw < 280:
                    scale = 280 / max(1, rw)
                    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                res = recognizer.recognize(roi)
                candidates = set()
                candidates.update(self._plate_variants(res.get("plate_number") or ""))
                candidates.update(self._plate_variants(res.get("raw") or ""))
                for candidate_plate in candidates:
                    score = self._score_ocr_result(res, candidate_plate)
                    cur = aggregate.get(candidate_plate)
                    if not cur:
                        aggregate[candidate_plate] = {
                            "plate_number": candidate_plate,
                            "confidence": float(res.get("confidence") or 0.0),
                            "valid": bool(res.get("valid", False)),
                            "raw": res.get("raw", ""),
                            "_score": score,
                            "_hits": 1,
                        }
                    else:
                        cur["_hits"] += 1
                        cur["confidence"] = max(cur["confidence"], float(res.get("confidence") or 0.0))
                        cur["valid"] = cur["valid"] or bool(res.get("valid", False))
                        cur["_score"] = max(cur["_score"], score)

        for plate, item in aggregate.items():
            final_score = item["_score"] + (item["_hits"] - 1) * 0.25
            if final_score > best["_score"]:
                best = dict(item)
                best["_score"] = final_score

        best.pop("_score", None)
        best.pop("_hits", None)
        return best

    def load_zones(self):
        try:
            from api.models import SessionLocal, Camera
            db = SessionLocal()
            cam = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if cam and cam.zones:
                self.zones = cam.zones
                print(f"[LPR] 載入 {len(self.zones)} 個 ROI 區域")
            db.close()
        except Exception as e:
            print(f"[LPR] 載入 zones 失敗: {e}")

    def in_any_zone(self, cx, cy):
        if not self.zones:
            return True
        import numpy as np
        for z in self.zones:
            pts = z.get("points", [])
            if len(pts) < 3:
                continue
            polygon = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
                return True
        return False
        return False
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.load_zones()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[LPR] 開始: {self.camera_name}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        print(f"[LPR] 停止: {self.camera_name}")
            
    def _run(self):
        try:
            yolo = get_yolo()
            recognizer = get_recognizer()
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.last_error = f"無法開啟來源: {self.source}"
                print(f"[LPR] {self.last_error}")
                self.running = False
                return

            frame_skip = 10
            while self.running:
                # 避免 read 卡死太久：若長時間沒幀就重連
                if self.last_frame_at and (time.time() - self.last_frame_at > 12):
                    cap.release()
                    time.sleep(0.3)
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.last_frame_at = 0.0

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.2)
                    continue

                self.last_frame_at = time.time()
                self.total_frames += 1
                if self.total_frames % frame_skip != 0:
                    continue

                try:
                    with _yolo_lock:
                        results = yolo(frame, verbose=False, conf=0.5)

                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            if cls_id not in self.VEHICLE_CLASSES:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            if not self.in_any_zone(cx, cy):
                                continue
                            self.vehicles_detected += 1
                            vehicle_type = self.VEHICLE_CLASSES[cls_id]

                            # 多裁切候選 + OCR 投票
                            result = self._recognize_plate_best(frame, x1, y1, x2, y2, recognizer)
                            plate = result.get('plate_number')
                            conf = result.get('confidence', 0)
                            raw = result.get('raw', '')
                            if raw:
                                print(f"[LPR] OCR原始: {raw} → 格式化: {plate} 信心度: {conf:.2f}")

                            if plate and 4 <= len(plate) <= 10 and (result.get("valid") or conf > 0.30):
                                now = time.time()
                                if plate not in self.last_plates or (now - self.last_plates[plate]) > self.cooldown:
                                    self.last_plates[plate] = now
                                    self.total_detections += 1

                                    # 截圖
                                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                                    snapshot_name = f"{timestamp}_{plate.replace('-', '')}_{self.total_detections}.jpg"
                                    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_name)

                                    snapshot = frame.copy()
                                    cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    label = f"{plate} ({conf:.0%})"
                                    cv2.putText(snapshot, label, (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                                    cv2.imwrite(snapshot_path, snapshot)

                                    record = {
                                        'id': self.total_detections,
                                        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                        'plate_number': plate,
                                        'confidence': round(conf, 3),
                                        'valid': result.get('valid', False),
                                        'vehicle_type': vehicle_type,
                                        'camera_id': self.camera_id,
                                        'camera_name': self.camera_name,
                                        'snapshot': snapshot_name,
                                        'raw': raw,
                                    }
                                    self.results.insert(0, record)
                                    if len(self.results) > self.max_results:
                                        self.results.pop()
                                    self._save_record_db(record)

                                    print(f"[LPR] ✅ {plate} ({vehicle_type}) {conf:.0%}")

                except Exception as e:
                    self.last_error = str(e)
                    print(f"[LPR] 處理錯誤: {e}")

                time.sleep(0.05)

            cap.release()
        except Exception as e:
            self.last_error = str(e)
            self.running = False
            print(f"[LPR] 執行緒異常結束: {e}")


@router.post("/start/{camera_id}")
async def start_lpr_stream(camera_id: int):
    if camera_id in _lpr_tasks and _lpr_tasks[camera_id].running:
        return {"status": "running", "message": "已在運行中"}
    
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="攝影機不存在")
        _start_lpr_task(camera)
        set_feature_state("lpr", camera_id, True)
        return {"status": "started", "message": f"開始辨識: {camera.name}"}
    finally:
        db.close()


@router.post("/stop/{camera_id}")
async def stop_lpr_stream(camera_id: int):
    if camera_id in _lpr_tasks:
        _lpr_tasks[camera_id].stop()
        del _lpr_tasks[camera_id]
        set_feature_state("lpr", camera_id, False)
        return {"status": "stopped"}
    set_feature_state("lpr", camera_id, False)
    return {"status": "not_running"}


@router.get("/status/{camera_id}")
async def get_stream_status(camera_id: int):
    if camera_id in _lpr_tasks:
        task = _lpr_tasks[camera_id]
        thread_alive = bool(task.thread and task.thread.is_alive())
        if task.running and not thread_alive:
            task.running = False
        return {
            "running": task.running,
            "thread_alive": thread_alive,
            "camera_id": camera_id,
            "camera_name": task.camera_name,
            "total_frames": task.total_frames,
            "vehicles_detected": task.vehicles_detected,
            "total_detections": task.total_detections,
            "last_plate": list(task.last_plates.keys())[-1] if task.last_plates else None,
            "last_error": task.last_error,
        }
    return {"running": False, "camera_id": camera_id}


@router.get("/results/{camera_id}")
async def get_stream_results(camera_id: int, limit: int = 50):
    from api.models import SessionLocal, LPRRecord

    db = SessionLocal()
    try:
        rows = (
            db.query(LPRRecord)
            .filter(LPRRecord.camera_id == camera_id)
            .order_by(LPRRecord.created_at.desc(), LPRRecord.id.desc())
            .limit(limit)
            .all()
        )
        results = [
            {
                "id": r.id,
                "time": _fmt_taipei(r.created_at),
                "plate_number": r.plate_number,
                "confidence": float(r.confidence or 0),
                "valid": bool(r.valid),
                "vehicle_type": r.vehicle_type,
                "camera_id": r.camera_id,
                "camera_name": r.camera_name,
                "snapshot": r.snapshot,
                "raw": r.raw or "",
            }
            for r in rows
        ]
    finally:
        db.close()

    # 若 DB 暫時無資料且任務運作中，回退記憶體快取（避免剛啟動時空白）
    if not results and camera_id in _lpr_tasks:
        return {"results": _lpr_tasks[camera_id].results[:limit]}
    return {"results": results}


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@router.get("/history")
async def get_history(
    camera_id: Optional[int] = None,
    q: Optional[str] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
):
    from sqlalchemy import or_
    from api.models import SessionLocal, LPRRecord

    start_dt = _parse_iso_datetime(start)
    end_dt = _parse_iso_datetime(end)
    keyword = str(q or "").strip()

    db = SessionLocal()
    try:
        query = db.query(LPRRecord)
        if camera_id is not None:
            query = query.filter(LPRRecord.camera_id == int(camera_id))
        if min_confidence > 0:
            query = query.filter(LPRRecord.confidence >= float(min_confidence))
        if start_dt is not None:
            query = query.filter(LPRRecord.created_at >= start_dt)
        if end_dt is not None:
            query = query.filter(LPRRecord.created_at <= end_dt)
        if keyword:
            like_kw = f"%{keyword}%"
            query = query.filter(
                or_(
                    LPRRecord.plate_number.ilike(like_kw),
                    LPRRecord.vehicle_type.ilike(like_kw),
                    LPRRecord.camera_name.ilike(like_kw),
                )
            )
        rows = (
            query.order_by(LPRRecord.created_at.desc(), LPRRecord.id.desc())
            .limit(int(limit))
            .all()
        )
        results = [
            {
                "id": r.id,
                "time": _fmt_taipei(r.created_at),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "plate_number": r.plate_number,
                "confidence": float(r.confidence or 0),
                "valid": bool(r.valid),
                "vehicle_type": r.vehicle_type,
                "camera_id": r.camera_id,
                "camera_name": r.camera_name,
                "snapshot": r.snapshot,
                "raw": r.raw or "",
            }
            for r in rows
        ]
        return {"results": results}
    finally:
        db.close()


@router.get("/snapshot/{filename}")
async def get_snapshot(filename: str):
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="截圖不存在")


@router.get("/all")
async def get_all_streams():
    return {"streams": [{"camera_id": k, "running": v.running, "detections": v.total_detections} for k, v in _lpr_tasks.items()]}


def _start_lpr_task(camera: Camera) -> bool:
    camera_id = int(camera.id)
    if camera_id in _lpr_tasks and _lpr_tasks[camera_id].running:
        return False
    task = LPRStreamTask(camera_id, camera.source, camera.name)
    task.start()
    _lpr_tasks[camera_id] = task
    return True


def resume_lpr_streams() -> dict:
    wants = get_feature_state("lpr")
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
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                continue
            if _start_lpr_task(camera):
                resumed += 1
    finally:
        db.close()
    return {"total": total, "resumed": resumed}
