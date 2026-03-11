#!/usr/bin/env python3
"""交通流事件 API"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from api.models import TrafficEvent, get_db

router = APIRouter(prefix="/api/traffic", tags=["交通流"])


def _to_utc_naive(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


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
    query = db.query(TrafficEvent)
    if camera_id is not None:
        query = query.filter(TrafficEvent.camera_id == camera_id)
    if start_time is not None:
        query = query.filter(TrafficEvent.created_at >= start_time)
    if end_time is not None:
        query = query.filter(TrafficEvent.created_at <= end_time)
    if label:
        query = query.filter(TrafficEvent.label == str(label).lower())

    base = query.order_by(desc(TrafficEvent.created_at)).offset((page - 1) * page_size)
    total = query.count() if include_total else None
    if include_total:
        items = base.limit(page_size).all()
        has_more = (page * page_size) < int(total or 0)
    else:
        rows = base.limit(page_size + 1).all()
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
                "lane_no": x.lane_no,
                "direction": x.direction,
                "entered_zones": x.entered_zones or [],
                "bbox": x.bbox,
                "source": x.source,
                "created_at": x.created_at.isoformat() if x.created_at else None,
            }
            for x in items
        ],
    }
