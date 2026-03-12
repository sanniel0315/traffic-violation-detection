#!/usr/bin/env python3
"""系統日誌 API"""
from fastapi import APIRouter, Query
from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy.orm import Session
import threading
from zoneinfo import ZoneInfo

from api.models import SessionLocal, SystemLog

router = APIRouter(prefix="/api/logs", tags=["系統日誌"])

_lock = threading.Lock()
TZ_TAIPEI = ZoneInfo("Asia/Taipei")
LOG_RETAIN_DAYS = 365


def _to_taipei(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # 舊資料庫時間欄位為 naive UTC，輸出時轉為台北時區
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_TAIPEI)


def _to_dict(row: SystemLog):
    ts = _to_taipei(row.created_at or datetime.now(timezone.utc))
    return {
        "id": row.id,
        "time": ts.strftime("%H:%M:%S"),
        "timestamp": ts.isoformat(),
        "level": row.level,
        "source": row.source,
        "message": row.message,
    }


def add_log(level: str, message: str, source: str = "system"):
    """新增日誌 (供其他模組呼叫)"""
    db: Session = SessionLocal()
    with _lock:
        try:
            row = SystemLog(level=level, source=source, message=message, created_at=datetime.now(timezone.utc))
            db.add(row)
            db.commit()
            db.refresh(row)
            # 保留策略：僅保留最近 1 年
            cutoff = datetime.now(timezone.utc) - timedelta(days=LOG_RETAIN_DAYS)
            db.query(SystemLog).filter(SystemLog.created_at < cutoff).delete()
            db.commit()
            log_entry = _to_dict(row)
        except Exception:
            db.rollback()
            now = _to_taipei(datetime.now(timezone.utc))
            log_entry = {
                "id": 0,
                "time": now.strftime("%H:%M:%S"),
                "timestamp": now.isoformat(),
                "level": level,
                "source": source,
                "message": message,
            }
        finally:
            db.close()
    icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}
    print(f"{icons.get(level, '📝')} [{source}] {message}")
    return log_entry


@router.get("")
async def get_logs(limit: int = 100, level: str = None, source: str = None):
    """取得日誌"""
    return await query_logs(limit=limit, level=level, source=source)


@router.get("/query")
async def query_logs(
    limit: Optional[int] = Query(None, ge=1, le=5000),
    page: int = Query(1, ge=1),
    page_size: int = Query(200, ge=1, le=1000),
    level: Optional[str] = None,
    source: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    """查詢日誌（含時間區間）"""
    db: Session = SessionLocal()
    try:
        query = db.query(SystemLog)
        if level:
            query = query.filter(SystemLog.level == level)
        if source:
            query = query.filter(SystemLog.source == source)
        if start_time is not None:
            query = query.filter(SystemLog.created_at >= start_time)
        if end_time is not None:
            query = query.filter(SystemLog.created_at <= end_time)
        total = query.count()
        ordered = query.order_by(SystemLog.id.desc())
        if limit is not None:
            rows = ordered.limit(limit).all()
            return {"logs": [_to_dict(r) for r in rows], "total": total, "page": 1, "page_size": limit}
        rows = ordered.offset((page - 1) * page_size).limit(page_size).all()
        return {"logs": [_to_dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}
    finally:
        db.close()


@router.delete("")
async def clear_logs():
    """清除日誌"""
    db: Session = SessionLocal()
    try:
        db.query(SystemLog).delete()
        db.commit()
    finally:
        db.close()
    add_log("info", "日誌已清除", "system")
    return {"status": "success", "message": "日誌已清除"}
