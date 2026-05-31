#!/usr/bin/env python3
"""違規事件 API"""
import calendar
import os
import subprocess
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from api.models import get_db, Violation
from api.routes.logs import add_log

router = APIRouter(prefix="/api/violations", tags=["違規事件"])

# 違規 4 張時間軸快照 cache dir (mount 在 /files/violations/snapshots/)
_SNAPSHOT_CACHE_DIR = Path("./output/violations/snapshots").resolve()
_SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# 4 張時間軸偏移 (秒)，相對 violation_time
_SNAPSHOT_OFFSETS = [("before", -2.0), ("mid_a", -0.5), ("mid_b", 0.5), ("after", 2.0)]


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
    speed_kmh: Optional[float] = None
    speed_limit_kmh: Optional[float] = None
    overspeed_kmh: Optional[float] = None
    flow_roi_hit: Optional[bool] = None
    speed_roi_hit: Optional[bool] = None


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


@router.get("/repeat-offenders")
def repeat_offenders(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    threshold: int = 2,
    violation_type: Optional[str] = None,
    plate_kw: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """車輛累犯查詢: GROUP BY 車牌 + 套用時間/類型/車牌關鍵字篩選。
    必須定義在 /{violation_id} 之前 — FastAPI 按順序 match，否則 'repeat-offenders'
    會被當 violation_id 嘗試 parse 成 int → 422。"""
    base = db.query(Violation).filter(
        Violation.license_plate.isnot(None),
        Violation.license_plate != "",
    )
    if start_time:
        try:
            base = base.filter(Violation.created_at >= datetime.fromisoformat(start_time.replace("Z", "+00:00")))
        except Exception:
            pass
    if end_time:
        try:
            base = base.filter(Violation.created_at < datetime.fromisoformat(end_time.replace("Z", "+00:00")))
        except Exception:
            pass
    if violation_type:
        base = base.filter(Violation.violation_type == violation_type)
    if plate_kw:
        base = base.filter(Violation.license_plate.like(f"%{plate_kw}%"))

    th = max(1, int(threshold or 1))
    rows = (
        base.with_entities(
            Violation.license_plate,
            func.count(Violation.id).label("cnt"),
            func.min(Violation.created_at).label("first_at"),
            func.max(Violation.created_at).label("last_at"),
            func.sum(Violation.fine_amount).label("fine_total"),
        )
        .group_by(Violation.license_plate)
        .having(func.count(Violation.id) >= th)
        .order_by(desc("cnt"))
        .limit(500)
        .all()
    )
    plates = [r[0] for r in rows]
    types_map: dict = {}
    if plates:
        tq = base.with_entities(
            Violation.license_plate,
            Violation.violation_type,
            func.count(Violation.id),
        ).filter(Violation.license_plate.in_(plates)).group_by(
            Violation.license_plate, Violation.violation_type
        ).all()
        for p, t, c in tq:
            types_map.setdefault(p, {})[t or "UNKNOWN"] = int(c)
    items = [
        {
            "plate": p,
            "count": int(cnt),
            "first_at": fa.isoformat() if fa else None,
            "last_at": la.isoformat() if la else None,
            "fine_total": int(ft or 0),
            "types": types_map.get(p, {}),
        }
        for (p, cnt, fa, la, ft) in rows
    ]
    return {"items": items, "total": len(items)}


def _fetch_frigate_clip(camera_name: str, start_unix: int, end_unix: int) -> Optional[bytes]:
    """從 frigate 抓 mp4 clip bytes (失敗回 None)。"""
    import requests as _rq
    base_urls = ["http://127.0.0.1:5000", "http://frigate:5000"]
    for base in base_urls:
        try:
            r = _rq.get(
                f"{base}/api/{camera_name}/start/{start_unix}/end/{end_unix}/clip.mp4",
                timeout=15,
            )
            if r.status_code == 200 and len(r.content) > 1024:
                return r.content
        except Exception:
            continue
    return None


def _extract_frame_with_ffmpeg(clip_path: Path, seek_sec: float, out_path: Path) -> bool:
    """ffmpeg seek to `seek_sec` in clip → 抽 1 frame 存 out_path。"""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{seek_sec:.3f}", "-i", str(clip_path),
             "-frames:v", "1", "-q:v", "3", str(out_path)],
            capture_output=True, timeout=10, check=False,
        )
        return out_path.exists() and out_path.stat().st_size > 1024
    except Exception:
        return False


@router.get("/{violation_id}/snapshots")
def get_violation_snapshots(violation_id: int, db: Session = Depends(get_db)):
    """違規 4 張時間軸快照 (前/中/中/後): 從 frigate clip 用 ffmpeg 抽 ±2s/±0.5s 共 4 frame，
    存 ./output/violations/snapshots/{id}_{tag}.jpg，回 /files/... URL。
    Cache：4 張都已存在直接回。第一次 fetch 抓 frigate clip ~1-2s。"""
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="violation not found")
    if not v.created_at or not v.camera_id:
        return {"snapshots": {}, "violation_id": violation_id, "reason": "missing time/camera"}

    # check cache 是否齊全
    cached = {tag: (_SNAPSHOT_CACHE_DIR / f"{violation_id}_{tag}.jpg") for tag, _ in _SNAPSHOT_OFFSETS}
    need_fetch = [tag for tag, p in cached.items() if not (p.exists() and p.stat().st_size > 1024)]

    if need_fetch:
        # frigate camera name 慣例 = f"cam_{camera_id}"
        camera_name = f"cam_{int(v.camera_id)}"
        # violation_time 是 UTC datetime (DB stored as naive UTC via datetime.utcnow)
        ts_unix = int(calendar.timegm(v.created_at.utctimetuple()))
        # 抓 clip ±3s (含緩衝)
        clip_bytes = _fetch_frigate_clip(camera_name, ts_unix - 3, ts_unix + 3)
        if clip_bytes:
            clip_path = _SNAPSHOT_CACHE_DIR / f"{violation_id}_clip.mp4"
            clip_path.write_bytes(clip_bytes)
            for tag, offset in _SNAPSHOT_OFFSETS:
                if tag not in need_fetch:
                    continue
                # clip 從 ts-3 開始, 所以 t 在 clip 內的 seek = 3 + offset
                seek = 3.0 + offset
                _extract_frame_with_ffmpeg(clip_path, seek, cached[tag])
            try:
                clip_path.unlink(missing_ok=True)
            except Exception:
                pass

    result = {}
    for tag, _ in _SNAPSHOT_OFFSETS:
        p = cached[tag]
        if p.exists() and p.stat().st_size > 1024:
            result[tag] = f"/files/violations/snapshots/{p.name}"
    return {
        "snapshots": result,
        "violation_id": violation_id,
        "available": list(result.keys()),
        "violation_time": v.created_at.isoformat() if v.created_at else None,
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
        speed_kmh=data.speed_kmh,
        speed_limit_kmh=data.speed_limit_kmh,
        overspeed_kmh=data.overspeed_kmh,
        flow_roi_hit=bool(data.flow_roi_hit),
        speed_roi_hit=bool(data.speed_roi_hit),
        violation_time=datetime.utcnow()
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    add_log(
        "warning",
        f"新增違規紀錄: {v.violation_name} | 車牌 {v.license_plate or '未知'} | 攝影機 ID={v.camera_id}",
        "violation",
    )
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
        "speed_kmh": v.speed_kmh,
        "speed_limit_kmh": v.speed_limit_kmh,
        "overspeed_kmh": v.overspeed_kmh,
        "flow_roi_hit": bool(v.flow_roi_hit),
        "speed_roi_hit": bool(v.speed_roi_hit),
        "created_at": v.created_at.isoformat() if v.created_at else None
    }
