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
import sys

sys.path.insert(0, '/workspace')

router = APIRouter(prefix="/api/lpr", tags=["lpr"])

_recognizer = None

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from recognition.plate_recognizer import PlateRecognizer
        _recognizer = PlateRecognizer(use_gpu=True)
    return _recognizer


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
        
        result = recognizer.recognize(frame)
        
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
        
        result = recognizer.recognize(frame)
        
        return {
            "plate_number": result.get('plate_number'),
            "confidence": result.get('confidence', 0),
            "valid": result.get('valid', False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize-camera/{camera_id}")
async def recognize_from_camera(camera_id: int):
    """從攝影機擷取畫面辨識"""
    from api.models import SessionLocal, Camera
    
    
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="攝影機不存在")
        
        cap = cv2.VideoCapture(camera.source)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=500, detail="無法擷取畫面")
        
        recognizer = get_recognizer()
        result = recognizer.recognize(frame)
        
        return {
            "camera_id": camera_id,
            "camera_name": camera.name,
            "plate_number": result.get('plate_number'),
            "confidence": result.get('confidence', 0),
            "valid": result.get('valid', False)
        }
    finally:
        db.close()
