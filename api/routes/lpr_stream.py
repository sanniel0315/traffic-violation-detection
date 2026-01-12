#!/usr/bin/env python3
"""車牌辨識串流 API - YOLO + Tesseract"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import cv2
import time
import threading
import os
from typing import Dict
import sys

sys.path.insert(0, '/workspace')

router = APIRouter(prefix="/api/lpr/stream", tags=["lpr-stream"])

_lpr_tasks: Dict[int, 'LPRStreamTask'] = {}
_yolo_model = None
_recognizer = None

SNAPSHOT_DIR = '/workspace/storage/lpr_snapshots'
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO('/workspace/yolov8n.pt')
    return _yolo_model

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from recognition.plate_recognizer import PlateRecognizer
        _recognizer = PlateRecognizer()
    return _recognizer


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
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[LPR] 開始: {self.camera_name}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        print(f"[LPR] 停止: {self.camera_name}")
            
    def _run(self):
        yolo = get_yolo()
        recognizer = get_recognizer()
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"[LPR] 無法開啟: {self.source}")
            self.running = False
            return
            
        frame_skip = 10
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(self.source)
                continue
                
            self.total_frames += 1
            if self.total_frames % frame_skip != 0:
                continue
                
            try:
                results = yolo(frame, verbose=False, conf=0.5)
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in self.VEHICLE_CLASSES:
                            continue
                            
                        self.vehicles_detected += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicle_type = self.VEHICLE_CLASSES[cls_id]
                        
                        # 車牌區域（車輛下方 1/3）
                        vh = y2 - y1
                        plate_y1 = y1 + int(vh * 0.6)  # 下方 40%
                        vehicle_img = frame[plate_y1:y2, x1:x2]
                        if vehicle_img.size == 0:
                            continue
                        
                        # 放大
                        vh, vw = vehicle_img.shape[:2]
                        if vw < 300:
                            scale = 300 / vw
                            vehicle_img = cv2.resize(vehicle_img, None, fx=scale, fy=scale)
                        
                        # 辨識
                        result = recognizer.recognize(vehicle_img)
                        plate = result.get('plate_number')
                        conf = result.get('confidence', 0)
                        raw = result.get('raw', '')
                        if raw:
                            print(f"[LPR] OCR原始: {raw} → 格式化: {plate} 信心度: {conf:.2f}")
                        raw = result.get('raw', '')
                        if raw:
                            print(f"[LPR] OCR原始: {raw} → 格式化: {plate} 信心度: {conf:.2f}")
                        
                        if plate and 4 <= len(plate) <= 10 and (result.get("valid") or conf > 0.2):
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
                                    'snapshot': snapshot_name
                                }
                                self.results.insert(0, record)
                                if len(self.results) > self.max_results:
                                    self.results.pop()
                                    
                                print(f"[LPR] ✅ {plate} ({vehicle_type}) {conf:.0%}")
                        
            except Exception as e:
                print(f"[LPR] 錯誤: {e}")
                
            time.sleep(0.05)
            
        cap.release()


@router.post("/start/{camera_id}")
async def start_lpr_stream(camera_id: int):
    from api.models import SessionLocal, Camera
    
    if camera_id in _lpr_tasks and _lpr_tasks[camera_id].running:
        return {"status": "running", "message": "已在運行中"}
    
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="攝影機不存在")
        
        task = LPRStreamTask(camera_id, camera.source, camera.name)
        task.start()
        _lpr_tasks[camera_id] = task
        
        return {"status": "started", "message": f"開始辨識: {camera.name}"}
    finally:
        db.close()


@router.post("/stop/{camera_id}")
async def stop_lpr_stream(camera_id: int):
    if camera_id in _lpr_tasks:
        _lpr_tasks[camera_id].stop()
        del _lpr_tasks[camera_id]
        return {"status": "stopped"}
    return {"status": "not_running"}


@router.get("/status/{camera_id}")
async def get_stream_status(camera_id: int):
    if camera_id in _lpr_tasks:
        task = _lpr_tasks[camera_id]
        return {
            "running": task.running,
            "camera_id": camera_id,
            "camera_name": task.camera_name,
            "total_frames": task.total_frames,
            "vehicles_detected": task.vehicles_detected,
            "total_detections": task.total_detections,
            "last_plate": list(task.last_plates.keys())[-1] if task.last_plates else None
        }
    return {"running": False, "camera_id": camera_id}


@router.get("/results/{camera_id}")
async def get_stream_results(camera_id: int, limit: int = 50):
    if camera_id in _lpr_tasks:
        return {"results": _lpr_tasks[camera_id].results[:limit]}
    return {"results": []}


@router.get("/snapshot/{filename}")
async def get_snapshot(filename: str):
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="截圖不存在")


@router.get("/all")
async def get_all_streams():
    return {"streams": [{"camera_id": k, "running": v.running, "detections": v.total_detections} for k, v in _lpr_tasks.items()]}
