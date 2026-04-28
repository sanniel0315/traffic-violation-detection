#!/usr/bin/env python3
"""車牌辨識 API"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import base64
import os
import re
import sys
from api.utils.camera_stream import resolve_analysis_source

sys.path.insert(0, '/workspace')

router = APIRouter(prefix="/api/lpr", tags=["lpr"])

_recognizer = None
_plate_detector = None

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from recognition.plate_recognizer import PlateRecognizer
        _recognizer = PlateRecognizer(use_gpu=True)
    return _recognizer


def get_plate_detector():
    global _plate_detector
    if _plate_detector is None:
        from recognition.plate_detector import PlateDetector
        _plate_detector = PlateDetector()
    return _plate_detector


def _normalize_plate_candidate(raw: str) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    text = "".join(ch for ch in text if ch.isalnum() or ch == "-")
    if "-" in text:
        parts = [p for p in text.split("-") if p]
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}"
    plain = text.replace("-", "")
    if 5 <= len(plain) <= 8:
        for cut in (3, 2, 4):
            if 2 <= cut < len(plain) - 1 and len(plain[:cut]) <= 4 and len(plain[cut:]) <= 4:
                return f"{plain[:cut]}-{plain[cut:]}"
    return text


def _plate_variants(raw: str) -> set[str]:
    plate = _normalize_plate_candidate(raw)
    if not plate:
        return set()
    variants = {plate}
    plain = plate.replace("-", "")
    if 5 <= len(plain) <= 8:
        for cut in (2, 3, 4):
            if 2 <= cut < len(plain) - 1:
                variants.add(f"{plain[:cut]}-{plain[cut:]}")
    return {_normalize_plate_candidate(v) for v in variants if _normalize_plate_candidate(v)}


def _plate_layout_score(plate: str) -> float:
    plate = _normalize_plate_candidate(plate)
    if not plate:
        return -3.0
    plain = plate.replace("-", "")
    if len(plain) < 5 or len(plain) > 8:
        return -2.5
    digits = sum(ch.isdigit() for ch in plain)
    letters = sum(ch.isalpha() for ch in plain)
    score = 0.0
    if digits == 0:
        return -3.0
    if digits == 1:
        score -= 1.5
    elif digits >= 2:
        score += 0.8
    if letters == 0:
        score -= 1.2
    else:
        score += 0.4
    if re.match(r"^[A-Z]{2,4}-[0-9]{2,4}$", plate):
        score += 1.4
    elif re.match(r"^[0-9]{2,4}-[A-Z]{2,4}$", plate):
        score += 0.9
    elif re.match(r"^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$", plate):
        score += 0.3
    if re.search(r"([A-Z0-9])\1\1", plain):
        score -= 0.8
    return score


def _score_ocr_result(result: dict, plate: str) -> float:
    conf = float(result.get("confidence") or 0.0)
    valid = bool(result.get("valid", False))
    score = (conf * 1.5) + _plate_layout_score(plate)
    if valid:
        score += 2.5
    if re.match(r"^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$", plate):
        score += 0.3
    if re.match(r"^[A-Z]{2,4}-[0-9]{2,4}$", plate):
        score += 0.5
    return score


def _recognize_plate_on_crop(plate_crop, recognizer):
    from api.routes.lpr_stream import _recognize_plate_on_crop as stream_recognize_plate_on_crop
    return stream_recognize_plate_on_crop(plate_crop, recognizer)


def recognize_best_plate(frame, recognizer):
    empty = {"plate_number": None, "confidence": 0.0, "valid": False, "raw": ""}
    if frame is None or getattr(frame, "size", 0) == 0:
        return empty

    detector = get_plate_detector()
    candidates = detector.detect(frame, conf=0.12)
    best = dict(empty)
    best_score = -10.0

    for det in candidates:
        crop = detector.crop(frame, det.get("bbox", [0, 0, 0, 0]))
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        res = _recognize_plate_on_crop(crop, recognizer)
        candidates_text = set()
        candidates_text.update(_plate_variants(res.get("plate_number") or ""))
        candidates_text.update(_plate_variants(res.get("raw") or ""))
        for plate in candidates_text:
            score = _score_ocr_result(res, plate) + float(det.get("confidence") or 0.0)
            if score > best_score:
                best_score = score
                best = {
                    "plate_number": plate,
                    "confidence": float(res.get("confidence") or 0.0),
                    "valid": bool(res.get("valid", False)),
                    "raw": res.get("raw", ""),
                }

    if best.get("plate_number"):
        return best
    return recognizer.recognize(frame)


class LPRResult(BaseModel):
    plate_number: Optional[str]
    confidence: float
    valid: bool


@router.get("/status")
async def get_lpr_status():
    """取得 LPR 狀態"""
    try:
        recognizer = get_recognizer()
        return {
            "status": "online",
            "engine": "Tesseract",
            "gpu": False  # Tesseract CPU
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/recognize-upload")
async def recognize_plate_upload(file: UploadFile = File(...)):
    """上傳圖片辨識車牌"""
    try:
        recognizer = get_recognizer()
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="無法解碼圖片")
        
        result = recognize_best_plate(frame, recognizer)
        
        return {
            "plate_number": result.get('plate_number'),
            "confidence": result.get('confidence', 0),
            "valid": result.get('valid', False),
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize-base64")
async def recognize_plate_base64(image_base64: str):
    """Base64 圖片辨識車牌"""
    try:
        recognizer = get_recognizer()
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="無法解碼圖片")
        
        result = recognize_best_plate(frame, recognizer)
        
        return {
            "plate_number": result.get('plate_number'),
            "confidence": result.get('confidence', 0),
            "valid": result.get('valid', False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize-camera/{camera_id}")
def recognize_from_camera(camera_id: int):
    """從攝影機擷取畫面辨識"""
    from api.models import SessionLocal, Camera
    
    
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="攝影機不存在")
        
        cap = cv2.VideoCapture(resolve_analysis_source(camera))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=500, detail="無法擷取畫面")
        
        recognizer = get_recognizer()
        result = recognize_best_plate(frame, recognizer)
        
        return {
            "camera_id": camera_id,
            "camera_name": camera.name,
            "plate_number": result.get('plate_number'),
            "confidence": result.get('confidence', 0),
            "valid": result.get('valid', False)
        }
    finally:
        db.close()
