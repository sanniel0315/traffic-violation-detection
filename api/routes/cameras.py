#!/usr/bin/env python3
"""攝影機 API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
import cv2

from api.models import get_db, Camera

router = APIRouter(prefix="/api/cameras", tags=["攝影機"])


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
    status: Optional[str] = None


class TestUrlRequest(BaseModel):
    url: str


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
    c = Camera(
        name=data.name,
        source=data.source,
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
    
    for key, value in data.dict(exclude_unset=True).items():
        setattr(c, key, value)
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
    """測試攝影機連線"""
    c = db.query(Camera).filter(Camera.id == camera_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    
    result = _test_rtsp(c.source)
    if result["status"] == "success":
        c.status = "online"
        c.last_seen = datetime.utcnow()
    else:
        c.status = "offline"
    db.commit()
    return result


@router.post("/test-url")
async def test_url(data: TestUrlRequest):
    """測試 RTSP URL 連線"""
    return _test_rtsp(data.url)


def _test_rtsp(url: str) -> dict:
    """測試 RTSP 連線"""
    if not url:
        return {"status": "error", "message": "URL 為空"}
    
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return {"status": "error", "message": "無法連線到攝影機"}
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return {"status": "success", "message": f"連線成功 ({w}x{h})"}
        else:
            return {"status": "error", "message": "無法讀取影像"}
    except Exception as e:
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
