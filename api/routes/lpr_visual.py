#!/usr/bin/env python3
"""車牌辨識視覺化串流 - 顯示車輛框和車牌號碼"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import time
import sys

sys.path.insert(0, '/workspace')

router = APIRouter(prefix="/api/lpr/visual", tags=["lpr-visual"])

_recognizer = None
_yolo_model = None

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from recognition.plate_recognizer import PlateRecognizer
        _recognizer = PlateRecognizer(use_gpu=True)
    return _recognizer

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO('/workspace/yolov8n.pt')
    return _yolo_model


def generate_frames(source: str):
    """產生視覺化串流"""
    recognizer = get_recognizer()
    yolo = get_yolo()
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        return
    
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    frame_count = 0
    last_results = []  # 快取上次結果
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(source)
            time.sleep(0.1)
            continue
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # 每 3 幀偵測一次（平衡效能）
        if frame_count % 3 == 0:
            last_results = []
            try:
                results = yolo(frame, verbose=False, conf=0.5)
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in VEHICLE_CLASSES:
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicle_type = VEHICLE_CLASSES[cls_id]
                        
                        # 車牌區域（下半部）
                        vh = y2 - y1
                        plate_y1 = y1 + int(vh * 0.3)
                        plate_region = frame[plate_y1:y2, x1:x2]
                        
                        plate_text = ""
                        plate_conf = 0
                        plate_valid = False
                        
                        if plate_region.size > 0:
                            # 放大車牌區域提高辨識率
                            ph, pw = plate_region.shape[:2]
                            if pw < 300:
                                scale = 300 / pw
                                plate_region = cv2.resize(plate_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                            result = recognizer.recognize(plate_region)
                            plate_text = result.get('plate_number', '') or ''
                            plate_conf = result.get('confidence', 0)
                            plate_valid = result.get('valid', False)
                            raw_text = result.get('raw', '')
                        
                        last_results.append({
                            'bbox': (x1, y1, x2, y2),
                            'plate_bbox': (x1, plate_y1, x2, y2),
                            'vehicle_type': vehicle_type,
                            'plate_text': plate_text,
                            'plate_conf': plate_conf,
                            'plate_valid': plate_valid,
                            'raw': raw_text if 'raw_text' in dir() else ''
                        })
            except Exception as e:
                print(f"[Visual] 錯誤: {e}")
        
        # 繪製結果
        for item in last_results:
            x1, y1, x2, y2 = item['bbox']
            px1, py1, px2, py2 = item['plate_bbox']
            
            # 車輛框（綠色）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 車牌區域框（黃色）
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
            
            # 車輛類型
            cv2.putText(frame, item['vehicle_type'], (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 車牌號碼
            if item['plate_text']:
                color = (0, 255, 0) if item['plate_valid'] else (0, 165, 255)
                label = f"{item['plate_text']} ({item['plate_conf']:.0%})"
                
                # 背景框
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (px1, py2), (px1 + tw + 10, py2 + th + 10), color, -1)
                cv2.putText(frame, label, (px1 + 5, py2 + th + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 狀態資訊
        info = f"Vehicles: {len(last_results)} | Frame: {frame_count}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 編碼輸出
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()


@router.get("/stream/{camera_id}")
async def visual_stream(camera_id: int):
    """視覺化串流端點"""
    from api.models import SessionLocal, Camera
    
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="攝影機不存在")
        
        return StreamingResponse(
            generate_frames(camera.source),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    finally:
        db.close()


@router.get("/snapshot/{camera_id}")
async def visual_snapshot(camera_id: int):
    """單張視覺化快照"""
    from api.models import SessionLocal, Camera
    from fastapi.responses import Response
    
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="攝影機不存在")
        
        recognizer = get_recognizer()
        yolo = get_yolo()
        cap = cv2.VideoCapture(camera.source)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=500, detail="無法擷取畫面")
        
        VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        detected = []
        
        results = yolo(frame, verbose=False, conf=0.5)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_type = VEHICLE_CLASSES[cls_id]
                
                vh = y2 - y1
                plate_y1 = y1 + int(vh * 0.3)
                plate_region = frame[plate_y1:y2, x1:x2]
                
                plate_text = ""
                plate_conf = 0
                plate_valid = False
                
                if plate_region.size > 0:
                    result = recognizer.recognize(plate_region)
                    plate_text = result.get('plate_number', '') or ''
                    plate_conf = result.get('confidence', 0)
                    plate_valid = result.get('valid', False)
                
                # 繪製
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, plate_y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, vehicle_type, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if plate_text:
                    color = (0, 255, 0) if plate_valid else (0, 165, 255)
                    label = f"{plate_text} ({plate_conf:.0%})"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + th + 10), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y2 + th + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                detected.append({
                    'vehicle_type': vehicle_type,
                    'plate': plate_text,
                    'confidence': plate_conf,
                    'valid': plate_valid
                })
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
    finally:
        db.close()
