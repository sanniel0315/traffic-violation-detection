#!/usr/bin/env python3
"""交通流事件 API"""
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from api.models import TrafficEvent, get_db
from api.utils.report_aggregation import (
    build_vd_report_rows,
    normalize_bucket_size,
    refresh_traffic_aggregates,
)

router = APIRouter(prefix="/api/traffic", tags=["交通流"])

# 聚合 cache：避免每次查詢都重建（rebuild 35 萬筆要 16 秒）
_AGG_CACHE: dict[tuple, float] = {}
_AGG_CACHE_TTL_SEC = 60.0  # 60 秒內同樣的範圍/bucket/cam 跳過 rebuild


def _to_utc_naive(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _is_sqlite_locked(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _run_with_sqlite_retry(db: Session, fn: Callable[[], Any], retries: int = 4) -> Any:
    delay = 0.15
    last_error: OperationalError | None = None
    for attempt in range(retries):
        try:
            return fn()
        except OperationalError as exc:
            if not _is_sqlite_locked(exc):
                raise
            db.rollback()
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(delay)
            delay *= 2
    if last_error is not None:
        raise last_error


@router.get("/vd-report")
async def get_vd_report(
    camera_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    bucket_size: str = Query("5m"),
    db: Session = Depends(get_db),
):
    start_time = _to_utc_naive(start_time) or datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    end_time = _to_utc_naive(end_time) or datetime.utcnow()
    bucket_size = normalize_bucket_size(bucket_size)
    if end_time < start_time:
        start_time, end_time = end_time, start_time

    try:
        db.execute(text("PRAGMA busy_timeout = 5000"))
    except Exception:
        pass

    def rebuild_aggs():
        # VD 報表只需要 traffic_events 聚合；不跑 congestion + LPR 聚合節省時間
        n = refresh_traffic_aggregates(db, start_time, end_time, bucket_size, camera_id=camera_id)
        db.commit()
        return {f"traffic_{bucket_size}": n}

    def build_rows():
        return build_vd_report_rows(
            db,
            start_time=start_time,
            end_time=end_time,
            bucket_size=bucket_size,
            camera_id=camera_id,
        )

    # Cache 鍵：相同的時間範圍 + bucket + camera 在 TTL 內跳過 rebuild
    cache_key = (
        start_time.replace(second=0, microsecond=0),
        end_time.replace(second=0, microsecond=0),
        bucket_size,
        camera_id,
    )
    now_ts = time.time()
    cached_at = _AGG_CACHE.get(cache_key, 0.0)
    if (now_ts - cached_at) < _AGG_CACHE_TTL_SEC:
        # 跳過 rebuild，直接讀已有聚合
        refreshed = {"cached": True}
    else:
        refreshed = _run_with_sqlite_retry(db, rebuild_aggs)
        _AGG_CACHE[cache_key] = now_ts
        # 清掉太舊的 cache 項目
        if len(_AGG_CACHE) > 100:
            cutoff = now_ts - _AGG_CACHE_TTL_SEC * 2
            for k in list(_AGG_CACHE.keys()):
                if _AGG_CACHE[k] < cutoff:
                    del _AGG_CACHE[k]
    rows = _run_with_sqlite_retry(db, build_rows)
    return {
        "bucket_size": bucket_size,
        "camera_id": camera_id,
        "start_time": start_time.replace(tzinfo=timezone.utc).isoformat(),
        "end_time": end_time.replace(tzinfo=timezone.utc).isoformat(),
        "aggregation": refreshed,
        "items": rows,
    }


@router.get("/events")
async def get_traffic_events(
    camera_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    label: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(1000, ge=1, le=200000),
    include_total: bool = Query(True),
    db: Session = Depends(get_db),
):
    start_time = _to_utc_naive(start_time)
    end_time = _to_utc_naive(end_time)

    # SQLite 在事件持續寫入時容易短暫鎖住；先提高 busy timeout，再對讀取做短重試。
    try:
        db.execute(text("PRAGMA busy_timeout = 5000"))
    except Exception:
        pass

    def build_query():
        query = db.query(TrafficEvent)
        if camera_id is not None:
            query = query.filter(TrafficEvent.camera_id == camera_id)
        if start_time is not None:
            query = query.filter(TrafficEvent.created_at >= start_time)
        if end_time is not None:
            query = query.filter(TrafficEvent.created_at <= end_time)
        if label:
            query = query.filter(TrafficEvent.label == str(label).lower())
        return query

    def fetch_total() -> int:
        return int(build_query().count())

    def fetch_rows(limit: int):
        return (
            build_query()
            .order_by(desc(TrafficEvent.created_at))
            .offset((page - 1) * page_size)
            .limit(limit)
            .all()
        )

    total = _run_with_sqlite_retry(db, fetch_total) if include_total else None
    if include_total:
        items = _run_with_sqlite_retry(db, lambda: fetch_rows(page_size))
        has_more = (page * page_size) < int(total or 0)
    else:
        rows = _run_with_sqlite_retry(db, lambda: fetch_rows(page_size + 1))
        has_more = len(rows) > page_size
        items = rows[:page_size]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": has_more,
        "items": [
            {
                "id": x.id,
                "camera_id": x.camera_id,
                "label": x.label,
                "speed_kmh": x.speed_kmh,
                "occupancy": x.occupancy,
                "lane_no": x.lane_no,
                "direction": x.direction,
                "entered_zones": x.entered_zones or [],
                "bbox": x.bbox,
                "source": x.source,
                "created_at": (
                    x.created_at.replace(tzinfo=timezone.utc).isoformat()
                    if x.created_at
                    else None
                ),
            }
            for x in items
        ],
    }
