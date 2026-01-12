#!/usr/bin/env python3
"""串流與偵測服務 API"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import cv2
import threading
import time
from datetime import datetime
from typing import Dict

from api.models import get_db, Camera

router = APIRouter(prefix="/api/stream", tags=["串流"])

# 偵測服務狀態
detection_services: Dict[int, dict] = {}


def generate_frames(source: str):
    """產生 MJPEG 串流"""
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重播影片
            continue
        
        # 縮小尺寸加快傳輸
        frame = cv2.resize(frame, (640, 360))
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.1)  # ~10 FPS
    
    cap.release()


@router.get("/{camera_id}/live")
async def live_stream(camera_id: int, db: Session = Depends(get_db)):
    """即時影像串流 (MJPEG)"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    
    return StreamingResponse(
        generate_frames(camera.source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/{camera_id}/snapshot")
async def snapshot(camera_id: int, db: Session = Depends(get_db)):
    """取得單張截圖"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    
    cap = cv2.VideoCapture(camera.source)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=503, detail="無法取得影像")
    
    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg"
    )


@router.post("/{camera_id}/detection/start")
async def start_detection(camera_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """啟動偵測服務"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    
    if camera_id in detection_services and detection_services[camera_id].get('running'):
        return {"status": "already_running", "message": "偵測服務已在執行中"}
    
    # 啟動背景偵測
    detection_services[camera_id] = {
        'running': True,
        'started_at': datetime.now().isoformat(),
        'camera_name': camera.name
    }
    
    background_tasks.add_task(run_detection, camera_id, camera.source, camera.location, camera.detection_config)
    
    # 更新攝影機狀態
    camera.status = "online"
    camera.detection_enabled = True
    db.commit()
    
    return {"status": "started", "message": f"偵測服務已啟動: {camera.name}"}


@router.post("/{camera_id}/detection/stop")
async def stop_detection(camera_id: int, db: Session = Depends(get_db)):
    """停止偵測服務"""
    if camera_id in detection_services:
        detection_services[camera_id]['running'] = False
    
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera:
        camera.detection_enabled = False
        db.commit()
    
    return {"status": "stopped", "message": "偵測服務已停止"}


@router.get("/{camera_id}/detection/status")
async def detection_status(camera_id: int):
    """取得偵測服務狀態"""
    if camera_id in detection_services:
        return detection_services[camera_id]
    return {"running": False}


@router.get("/detection/all")
async def all_detection_status():
    """取得所有偵測服務狀態"""
    return detection_services


def run_detection(camera_id: int, source: str, location: str, detection_config: dict):
    """背景偵測任務"""
    import sys
    sys.path.insert(0, '/workspace')
    
    from detection.vehicle_detector import VehicleDetector
    import requests
    import os
    from pathlib import Path
    
    output_dir = Path("./output/violations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector = VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.5)
    cap = cv2.VideoCapture(source)
    
    frame_count = 0
    detection_count = 0
    
    # 取得啟用的偵測類型
    enabled_types = []
    if detection_config.get('red_light'): enabled_types.append(('RED_LIGHT', '闘紅燈', 2700))
    if detection_config.get('speeding'): enabled_types.append(('SPEEDING', '超速', 1800))
    if detection_config.get('illegal_parking'): enabled_types.append(('ILLEGAL_PARKING', '違規停車', 600))
    if detection_config.get('wrong_way'): enabled_types.append(('WRONG_WAY', '逆向行駛', 900))
    
    print(f"🚀 偵測服務啟動: camera_id={camera_id}, 啟用類型={[t[1] for t in enabled_types]}")
    
    while detection_services.get(camera_id, {}).get('running', False):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        if frame_count % 10 != 0:  # 每 10 幀處理一次
            continue
        
        detections = detector.detect(frame)
        vehicles = [d for d in detections if d['class_name'] in ['car', 'motorcycle', 'truck', 'bus']]
        
        if vehicles and enabled_types:
            detection_count += 1
            
            # 更新服務狀態
            detection_services[camera_id]['detections'] = detection_count
            detection_services[camera_id]['last_detection'] = datetime.now().isoformat()
            
            # 每 50 次偵測記錄一次違規 (模擬)
            if detection_count % 50 == 1 and enabled_types:
                import random
                v_type, v_name, v_fine = random.choice(enabled_types)
                plate = f"{random.choice('ABCDEFGH')}{random.choice('ABCDEFGH')}{random.choice('ABCDEFGH')}-{random.randint(1000,9999)}"
                
                # 繪製標註
                annotated = frame.copy()
                for det in vehicles:
                    bbox = det['bbox']
                    cv2.rectangle(annotated, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 2)
                
                # 儲存截圖
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = f"{v_type}_{timestamp_str}.jpg"
                image_path = output_dir / image_name
                cv2.imwrite(str(image_path), annotated)
                
                # 發送到 API
                data = {
                    "violation_type": v_type,
                    "violation_name": v_name,
                    "vehicle_type": vehicles[0]['class_name'],
                    "license_plate": plate,
                    "location": location,
                    "camera_id": camera_id,
                    "confidence": vehicles[0]['confidence'],
                    "fine_amount": v_fine,
                    "points": 1,
                    "image_path": f"/files/violations/{image_name}"
                }
                
                try:
                    requests.post("http://localhost:8000/api/violations", json=data, timeout=5)
                    print(f"🚨 違規記錄: {v_name} | {plate}")
                except:
                    pass
        
        time.sleep(0.03)
    
    cap.release()
    print(f"⏹️ 偵測服務停止: camera_id={camera_id}")
