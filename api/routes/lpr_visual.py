#!/usr/bin/env python3
"""車牌辨識視覺化串流 - 顯示車輛框和車牌號碼"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import time
import sys
import re

sys.path.insert(0, '/workspace')

from model_paths import get_detect_model_pt

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
        _yolo_model = YOLO(get_detect_model_pt())
    return _yolo_model


def _normalize_plate_candidate(plate: str) -> str:
    if not plate:
        return ""
    t = plate.strip().upper().replace(" ", "")
    t = re.sub(r"[^A-Z0-9-]", "", t)
    return t


def _plate_variants(plate: str):
    plate = _normalize_plate_candidate(plate)
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
        n = _normalize_plate_candidate(v)
        core_len = len(n.replace("-", ""))
        if 5 <= core_len <= 8:
            out.append(n)
    return out


def _score_ocr_result(result: dict, plate: str) -> float:
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


def _recognize_plate_ensemble(frame, x1: int, y1: int, x2: int, y2: int, recognizer) -> dict:
    vh, vw = max(1, y2 - y1), max(1, x2 - x1)
    y_ratios = (0.30, 0.40, 0.50, 0.60, 0.68)
    x_trim_ratios = (0.00, 0.04, 0.08)
    best = {"plate_number": "", "confidence": 0.0, "valid": False, "raw": "", "_score": -1.0}
    aggregate = {}

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
            result = recognizer.recognize(roi)
            candidates = set()
            candidates.update(_plate_variants(result.get("plate_number") or ""))
            candidates.update(_plate_variants(result.get("raw") or ""))
            for plate in candidates:
                score = _score_ocr_result(result, plate)
                cur = aggregate.get(plate)
                if not cur:
                    aggregate[plate] = {
                        "plate_number": plate,
                        "confidence": float(result.get("confidence") or 0.0),
                        "valid": bool(result.get("valid", False)),
                        "raw": result.get("raw", ""),
                        "_score": score,
                        "_hits": 1,
                    }
                else:
                    cur["_hits"] += 1
                    cur["confidence"] = max(cur["confidence"], float(result.get("confidence") or 0.0))
                    cur["valid"] = cur["valid"] or bool(result.get("valid", False))
                    cur["_score"] = max(cur["_score"], score)

    for _, item in aggregate.items():
        final_score = item["_score"] + (item["_hits"] - 1) * 0.25
        if final_score > best["_score"]:
            best = dict(item)
            best["_score"] = final_score

    best.pop("_score", None)
    best.pop("_hits", None)
    return best


def generate_frames(source: str):
    """產生視覺化串流"""
    recognizer = get_recognizer()
    yolo = get_yolo()
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        return
    
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    frame_count = 0
    last_results = []  # 快取上次結果
    last_ok = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if time.time() - last_ok > 2.0:
                cap.release()
                time.sleep(0.1)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                last_ok = time.time()
            time.sleep(0.1)
            continue
        last_ok = time.time()
        
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
                        
                        vh = y2 - y1
                        plate_y1 = y1 + int(vh * 0.5)
                        result = _recognize_plate_ensemble(frame, x1, y1, x2, y2, recognizer)
                        plate_text = result.get('plate_number', '') or ''
                        plate_conf = result.get('confidence', 0)
                        plate_valid = result.get('valid', False)
                        ocr_raw_text = result.get('raw', '') or ''
                        
                        last_results.append({
                            'bbox': (x1, y1, x2, y2),
                            'plate_bbox': (x1, plate_y1, x2, y2),
                            'vehicle_type': vehicle_type,
                            'plate_text': plate_text,
                            'ocr_text': ocr_raw_text,
                            'plate_conf': plate_conf,
                            'plate_valid': plate_valid,
                            'raw': ocr_raw_text
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
            ocr_text = (item.get('plate_text') or item.get('ocr_text') or '').strip()
            if ocr_text:
                color = (0, 255, 0) if item['plate_valid'] else (0, 165, 255)
                label = f"{ocr_text} ({item['plate_conf']:.0%})"
                
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
                plate_y1 = y1 + int(vh * 0.5)
                result = _recognize_plate_ensemble(frame, x1, y1, x2, y2, recognizer)
                plate_text = result.get('plate_number', '') or ''
                plate_conf = result.get('confidence', 0)
                plate_valid = result.get('valid', False)
                ocr_raw_text = result.get('raw', '') or ''
                
                # 繪製
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, plate_y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, vehicle_type, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                ocr_text = (plate_text or ocr_raw_text).strip()
                if ocr_text:
                    color = (0, 255, 0) if plate_valid else (0, 165, 255)
                    label = f"{ocr_text} ({plate_conf:.0%})"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + th + 10), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y2 + th + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                detected.append({
                    'vehicle_type': vehicle_type,
                    'plate': ocr_text,
                    'confidence': plate_conf,
                    'valid': plate_valid
                })
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
    finally:
        db.close()
