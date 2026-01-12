#!/usr/bin/env python3
"""違規事件 API"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from api.models import get_db, Violation

router = APIRouter(prefix="/api/violations", tags=["違規事件"])


class ViolationCreate(BaseModel):
    violation_type: str
    violation_name: str
    license_plate: Optional[str] = None
    vehicle_type: str
    location: str
    camera_id: int
    confidence: float
    track_id: Optional[int] = None
    bbox: Optional[dict] = None
    image_path: Optional[str] = None
    fine_amount: Optional[int] = None
    points: Optional[int] = None


class ViolationReview(BaseModel):
    status: str  # confirmed, rejected
    comment: Optional[str] = None


@router.get("")
async def get_violations(
    status: Optional[str] = None,
    violation_type: Optional[str] = None,
    license_plate: Optional[str] = None,
    camera_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """取得違規列表"""
    query = db.query(Violation)
    
    if status:
        query = query.filter(Violation.status == status)
    if violation_type:
        query = query.filter(Violation.violation_type == violation_type)
    if license_plate:
        query = query.filter(Violation.license_plate.ilike(f"%{license_plate}%"))
    if camera_id:
        query = query.filter(Violation.camera_id == camera_id)
    if start_date:
        query = query.filter(Violation.violation_time >= start_date)
    if end_date:
        query = query.filter(Violation.violation_time <= end_date)
    
    total = query.count()
    items = query.order_by(desc(Violation.created_at)).offset((page-1)*page_size).limit(page_size).all()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [_to_dict(v) for v in items]
    }


@router.get("/statistics")
async def get_statistics(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """取得統計資料"""
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    
    # 總數統計
    total = db.query(Violation).count()
    pending = db.query(Violation).filter(Violation.status == "pending").count()
    confirmed = db.query(Violation).filter(Violation.status == "confirmed").count()
    
    # 今日統計
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.query(Violation).filter(Violation.created_at >= today_start).count()
    
    # 按類型統計
    by_type = db.query(
        Violation.violation_type,
        Violation.violation_name,
        func.count(Violation.id)
    ).filter(
        Violation.created_at >= start
    ).group_by(Violation.violation_type, Violation.violation_name).all()
    
    # 按日期統計
    by_date = db.query(
        func.date(Violation.created_at).label('date'),
        func.count(Violation.id)
    ).filter(
        Violation.created_at >= start
    ).group_by(func.date(Violation.created_at)).all()
    
    return {
        "total": total,
        "pending": pending,
        "confirmed": confirmed,
        "today": today_count,
        "by_type": [{"type": t, "name": n, "count": c} for t, n, c in by_type],
        "by_date": [{"date": str(d), "count": c} for d, c in by_date]
    }


@router.get("/{violation_id}")
async def get_violation(violation_id: int, db: Session = Depends(get_db)):
    """取得單一違規記錄"""
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="違規記錄不存在")
    return _to_dict(v)


@router.post("")
async def create_violation(data: ViolationCreate, db: Session = Depends(get_db)):
    """建立違規記錄"""
    v = Violation(
        violation_type=data.violation_type,
        violation_name=data.violation_name,
        license_plate=data.license_plate,
        vehicle_type=data.vehicle_type,
        location=data.location,
        camera_id=data.camera_id,
        confidence=data.confidence,
        track_id=data.track_id,
        bbox=data.bbox,
        image_path=data.image_path,
        fine_amount=data.fine_amount,
        points=data.points,
        violation_time=datetime.utcnow()
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return _to_dict(v)


@router.put("/{violation_id}/review")
async def review_violation(
    violation_id: int,
    review: ViolationReview,
    db: Session = Depends(get_db)
):
    """審核違規記錄"""
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="違規記錄不存在")
    
    v.status = review.status
    v.review_comment = review.comment
    v.reviewed_at = datetime.utcnow()
    db.commit()
    
    return {"message": "審核完成", "status": review.status}


@router.delete("/{violation_id}")
async def delete_violation(violation_id: int, db: Session = Depends(get_db)):
    """刪除違規記錄"""
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="違規記錄不存在")
    
    db.delete(v)
    db.commit()
    return {"message": "已刪除"}


def _to_dict(v: Violation) -> dict:
    return {
        "id": v.id,
        "violation_type": v.violation_type,
        "violation_name": v.violation_name,
        "license_plate": v.license_plate,
        "vehicle_type": v.vehicle_type,
        "location": v.location,
        "camera_id": v.camera_id,
        "violation_time": v.violation_time.isoformat() if v.violation_time else None,
        "confidence": v.confidence,
        "image_path": v.image_path,
        "status": v.status,
        "fine_amount": v.fine_amount,
        "points": v.points,
        "created_at": v.created_at.isoformat() if v.created_at else None
    }
