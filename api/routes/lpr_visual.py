#!/usr/bin/env python3
"""車牌辨識視覺化串流 - 顯示車輛框和車牌號碼"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import pytesseract
import time
import os
import sys
import re

sys.path.insert(0, '/workspace')

from model_paths import get_detect_model_engine, get_detect_model_pt
from api.utils.camera_stream import resolve_analysis_source

router = APIRouter(prefix="/api/lpr/visual", tags=["lpr-visual"])

_recognizer = None
_yolo_model = None
_OCR_FAST_MODE = os.getenv("LPR_OCR_MODE", "accurate").strip().lower() in {"fast", "quick", "speed"}
_stream_helpers = None


def get_stream_helpers():
    global _stream_helpers
    if _stream_helpers is None:
        from api.routes import lpr_stream as _lpr_stream

        _stream_helpers = {
            "expand_plate_bbox": _lpr_stream._expand_plate_bbox,
            "recognize_plate_on_crop": _lpr_stream._recognize_plate_on_crop,
            "tighten_plate_crop_with_bbox": _lpr_stream._tighten_plate_crop_with_bbox,
            "get_plate_detector": _lpr_stream.get_plate_detector,
        }
    return _stream_helpers


def _placeholder_jpeg(text: str = "LPR stream waiting", width: int = 640, height: int = 360) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, "LPR VISUAL UNAVAILABLE", (24, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
    cv2.putText(frame, str(text or ""), (24, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buffer.tobytes() if ok else b""

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
        model_path = get_detect_model_engine()
        if not os.path.exists(model_path):
            model_path = get_detect_model_pt()
        _yolo_model = YOLO(model_path, task='detect')
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


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _flatten_plate_roi(roi):
    if roi is None or getattr(roi, "size", 0) == 0:
        return roi
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi
        h, w = gray.shape[:2]
        img_area = float(max(1, h * w))
        best = None
        best_score = 0.0
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            area = float(cv2.contourArea(cnt))
            if area < img_area * 0.01:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(cnt)
                pts = cv2.boxPoints(rect).astype(np.float32)
            ordered = _order_quad_points(pts)
            w1 = np.linalg.norm(ordered[1] - ordered[0])
            w2 = np.linalg.norm(ordered[2] - ordered[3])
            h1 = np.linalg.norm(ordered[3] - ordered[0])
            h2 = np.linalg.norm(ordered[2] - ordered[1])
            ww = max(w1, w2)
            hh = max(h1, h2)
            if ww < 28 or hh < 10:
                continue
            aspect = ww / max(1.0, hh)
            if not (1.5 <= aspect <= 8.5):
                continue
            score = area * min(3.0, aspect)
            if score > best_score:
                best_score = score
                best = ordered
        if best is None:
            return roi
        w1 = np.linalg.norm(best[1] - best[0])
        w2 = np.linalg.norm(best[2] - best[3])
        h1 = np.linalg.norm(best[3] - best[0])
        h2 = np.linalg.norm(best[2] - best[1])
        out_w = int(max(40, max(w1, w2)))
        out_h = int(max(14, max(h1, h2)))
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(best, dst)
        flat = cv2.warpPerspective(roi, M, (out_w, out_h))
        if out_w < 280:
            scale = 280 / max(1, out_w)
            flat = cv2.resize(flat, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return flat
    except Exception:
        return roi


def _recognize_plate_fast(roi, recognizer) -> dict:
    best = {"plate_number": "", "confidence": 0.0, "valid": False, "raw": ""}
    if roi is None or getattr(roi, "size", 0) == 0:
        return best
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        h, w = gray.shape[:2]
        if w < 220:
            scale = 260 / max(1, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = getattr(
            recognizer,
            "config_single",
            "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        )
        clean_fn = getattr(recognizer, "_clean", None)
        valid_fn = getattr(recognizer, "_validate", None)

        for p_img in (gray, binary):
            data = pytesseract.image_to_data(p_img, config=config, output_type=pytesseract.Output.DICT)
            texts = []
            confs = []
            for i, text in enumerate(data.get("text", [])):
                t = str(text or "").strip()
                if len(t) < 2:
                    continue
                texts.append(t)
                raw_conf = str(data.get("conf", ["0"])[i])
                try:
                    c = float(raw_conf)
                except Exception:
                    c = 0.0
                if c < 0:
                    c = 0.0
                confs.append(c / 100.0)
            if not texts:
                continue

            raw = "".join(texts)
            if callable(clean_fn):
                plate, _ = clean_fn(raw)
            else:
                plate = re.sub(r"[^A-Z0-9-]", "", raw.upper())
            conf = max(confs) if confs else 0.0
            valid = bool(valid_fn(plate)) if callable(valid_fn) else bool(re.match(r"^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$", plate))
            if valid and ((not best["valid"]) or conf > best["confidence"]):
                best = {"plate_number": plate, "confidence": conf, "valid": True, "raw": raw}
            elif (not best["valid"]) and 5 <= len(str(plate or "")) <= 9 and conf > best["confidence"]:
                best = {"plate_number": plate, "confidence": conf, "valid": valid, "raw": raw}
    except Exception:
        return best
    return best


def _recognize_plate_ensemble(frame, x1: int, y1: int, x2: int, y2: int, recognizer) -> dict:
    best = {
        "plate_number": "",
        "confidence": 0.0,
        "valid": False,
        "raw": "",
        "plate_bbox": None,
        "_score": -1.0,
    }

    vehicle_crop = frame[y1:y2, x1:x2]
    if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
        best.pop("_score", None)
        return best

    try:
        helpers = get_stream_helpers()
    except Exception:
        best.pop("_score", None)
        return best

    detector = helpers["get_plate_detector"]()
    expand_plate_bbox = helpers["expand_plate_bbox"]
    tighten_plate_crop_with_bbox = helpers["tighten_plate_crop_with_bbox"]
    recognize_plate_on_crop = helpers["recognize_plate_on_crop"]
    vh, vw = vehicle_crop.shape[:2]
    detections = detector.detect(vehicle_crop, conf=0.20)
    for det in detections:
        bx1, by1, bx2, by2 = det.get("bbox", [0, 0, 0, 0])
        ex1, ey1, ex2, ey2 = expand_plate_bbox([bx1, by1, bx2, by2], vw, vh)
        plate_crop = detector.crop(vehicle_crop, [ex1, ey1, ex2, ey2])
        plate_crop, local_bbox = tighten_plate_crop_with_bbox(plate_crop)
        if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
            continue

        result = recognize_plate_on_crop(plate_crop, recognizer)
        candidates = set()
        candidates.update(_plate_variants(result.get("plate_number") or ""))
        candidates.update(_plate_variants(result.get("raw") or ""))
        if not candidates:
            continue

        if isinstance(local_bbox, (list, tuple)) and len(local_bbox) == 4:
            lx1, ly1, lx2, ly2 = [int(v) for v in local_bbox]
        else:
            lx1, ly1, lx2, ly2 = 0, 0, max(1, ex2 - ex1), max(1, ey2 - ey1)

        px1 = int(x1 + ex1 + lx1)
        py1 = int(y1 + ey1 + ly1)
        px2 = int(x1 + ex2 if lx2 <= lx1 else x1 + ex1 + lx2)
        py2 = int(y1 + ey2 if ly2 <= ly1 else y1 + ey1 + ly2)

        for plate in candidates:
            score = _score_ocr_result(result, plate)
            if score <= best["_score"]:
                continue
            best = {
                "plate_number": plate,
                "confidence": float(result.get("confidence") or 0.0),
                "valid": bool(result.get("valid", False)),
                "raw": result.get("raw", ""),
                "plate_bbox": [px1, py1, px2, py2],
                "_score": score,
            }

    best.pop("_score", None)
    return best


def generate_frames(source: str):
    """產生視覺化串流"""
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    recognizer = None
    yolo = None
    model_error = None
    next_model_retry_at = 0.0

    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    frame_count = 0
    last_results = []  # 快取上次結果
    last_ok = time.time()
    last_placeholder_ts = 0.0
    had_frame = False

    while True:
        ret, frame = cap.read()
        if not ret:
            if time.time() - last_ok > 2.0:
                cap.release()
                time.sleep(0.1)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                last_ok = time.time()
            now_ts = time.time()
            if (not had_frame or (now_ts - last_ok > 3.0)) and (now_ts - last_placeholder_ts) > 1.0:
                ph = _placeholder_jpeg("Waiting for camera frame...")
                if ph:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + ph + b'\r\n')
                last_placeholder_ts = now_ts
            time.sleep(0.1)
            continue
        last_ok = time.time()
        had_frame = True

        frame_count += 1
        now = time.time()

        # 每 3 幀偵測一次（平衡效能）
        if frame_count % 3 == 0:
            if (recognizer is None or yolo is None) and now >= next_model_retry_at:
                try:
                    recognizer = get_recognizer()
                    yolo = get_yolo()
                    model_error = None
                except Exception as e:
                    model_error = str(e)
                    next_model_retry_at = now + 5.0
                    print(f"[Visual] 模型載入失敗: {e}")

            if recognizer is not None and yolo is not None:
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

                            result = _recognize_plate_ensemble(frame, x1, y1, x2, y2, recognizer)
                            plate_text = result.get('plate_number', '') or ''
                            plate_conf = result.get('confidence', 0)
                            plate_valid = result.get('valid', False)
                            ocr_raw_text = result.get('raw', '') or ''
                            pb = result.get('plate_bbox')
                            if isinstance(pb, (list, tuple)) and len(pb) == 4:
                                px1, py1, px2, py2 = map(int, pb)
                            else:
                                vh = y2 - y1
                                px1, py1, px2, py2 = x1, y1 + int(vh * 0.5), x2, y2

                            last_results.append({
                                'bbox': (x1, y1, x2, y2),
                                'plate_bbox': (px1, py1, px2, py2),
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
        if model_error:
            info = f"Visual stream fallback | Frame: {frame_count}"
        else:
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
        if not bool(camera.enabled):
            raise HTTPException(status_code=409, detail="攝影機已關閉")
        
        return StreamingResponse(
            generate_frames(resolve_analysis_source(camera)),
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
        if not bool(camera.enabled):
            raise HTTPException(status_code=409, detail="攝影機已關閉")
        
        recognizer = get_recognizer()
        yolo = get_yolo()
        cap = cv2.VideoCapture(resolve_analysis_source(camera))
        
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
                
                result = _recognize_plate_ensemble(frame, x1, y1, x2, y2, recognizer)
                plate_text = result.get('plate_number', '') or ''
                plate_conf = result.get('confidence', 0)
                plate_valid = result.get('valid', False)
                ocr_raw_text = result.get('raw', '') or ''
                pb = result.get('plate_bbox')
                if isinstance(pb, (list, tuple)) and len(pb) == 4:
                    px1, py1, px2, py2 = map(int, pb)
                else:
                    vh = y2 - y1
                    px1, py1, px2, py2 = x1, y1 + int(vh * 0.5), x2, y2
                
                # 繪製
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                cv2.putText(frame, vehicle_type, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                ocr_text = (plate_text or ocr_raw_text).strip()
                if ocr_text:
                    color = (0, 255, 0) if plate_valid else (0, 165, 255)
                    label = f"{ocr_text} ({plate_conf:.0%})"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (px1, py2), (px1 + tw + 10, py2 + th + 10), color, -1)
                    cv2.putText(frame, label, (px1 + 5, py2 + th + 5),
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
