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
                "snapshot_url": f"/api/traffic/events/{x.id}/snapshot.jpg",
            }
            for x in items
        ],
    }


# 從 Frigate 錄影擷取事件當下的截圖（cv2 first frame extraction，磁碟快取）
_EVENT_SNAPSHOT_CACHE_DIR = "/tmp/event_snapshots"


@router.get("/events/{event_id}/snapshot.jpg")
async def get_event_snapshot(event_id: int, db: Session = Depends(get_db)):
    import os
    import io
    from fastapi.responses import Response, FileResponse
    import requests as _req

    os.makedirs(_EVENT_SNAPSHOT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_EVENT_SNAPSHOT_CACHE_DIR, f"{event_id}.jpg")
    # 優先用 detection worker 即時存的截圖（不依賴 Frigate 錄影，符合「事件截圖無錄影」設計）
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return FileResponse(cache_path, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

    evt = db.query(TrafficEvent).filter(TrafficEvent.id == event_id).first()
    if not evt:
        return Response(status_code=404, content="event not found")
    if not evt.created_at or not evt.camera_id:
        return Response(status_code=404, content="missing camera/timestamp")

    # 該 event id 沒對應 snapshot（被 throttle 跳過）→ 找同 cam 時間最近的 snapshot
    try:
        evt_ts = evt.created_at.replace(tzinfo=timezone.utc).timestamp()
        # 撈該 cam 同時段（前後 30 秒）有 snapshot 的 event
        from sqlalchemy import and_
        nearby = db.query(TrafficEvent).filter(
            and_(
                TrafficEvent.camera_id == evt.camera_id,
                TrafficEvent.created_at >= datetime.utcfromtimestamp(evt_ts - 30),
                TrafficEvent.created_at <= datetime.utcfromtimestamp(evt_ts + 30),
            )
        ).order_by(TrafficEvent.id).all()
        # 找最接近 event_id 且有檔案的
        best = None; best_dist = 10**9
        for n in nearby:
            np = os.path.join(_EVENT_SNAPSHOT_CACHE_DIR, f"{n.id}.jpg")
            if os.path.exists(np) and os.path.getsize(np) > 0:
                dist = abs(n.id - event_id)
                if dist < best_dist:
                    best_dist = dist; best = np
        if best:
            return FileResponse(best, media_type="image/jpeg",
                                headers={"Cache-Control": "public, max-age=300"})
    except Exception:
        pass

    ts = int(evt.created_at.replace(tzinfo=timezone.utc).timestamp())
    cam_name = f"cam_{evt.camera_id}"
    # 從 Frigate 拉 clip — Frigate segment 邊界不規則，逐步擴大 window 重試
    r = None
    for half_window in (5, 15, 30):
        clip_url = f"http://localhost:5000/api/{cam_name}/start/{ts - half_window}/end/{ts + half_window}/clip.mp4"
        try:
            r = _req.get(clip_url, timeout=15)
            if r.status_code == 200 and len(r.content) >= 1024:
                break
        except Exception:
            r = None
    if r is None or r.status_code != 200 or len(r.content) < 1024:
        return Response(status_code=404, content="recording not available")

    # 寫到暫存檔讓 cv2 開
    tmp_clip = os.path.join(_EVENT_SNAPSHOT_CACHE_DIR, f"_tmp_{event_id}.mp4")
    try:
        with open(tmp_clip, "wb") as f:
            f.write(r.content)
        import cv2
        cap = cv2.VideoCapture(tmp_clip)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return Response(status_code=500, content="failed to decode frame")
        # 縮成 thumbnail 320x180
        h, w = frame.shape[:2]
        if w > 480:
            scale = 480.0 / w
            frame = cv2.resize(frame, (480, int(h * scale)))
        # 在 bbox 上畫框
        try:
            bbox = evt.bbox or []
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # bbox 是原始解析度，要按比例縮
                if w > 480:
                    rs = 480.0 / w
                    x1, y1, x2, y2 = int(x1 * rs), int(y1 * rs), int(x2 * rs), int(y2 * rs)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        except Exception:
            pass
        ok2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok2:
            return Response(status_code=500, content="encode failed")
        with open(cache_path, "wb") as f:
            f.write(buf.tobytes())
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
    finally:
        try:
            os.unlink(tmp_clip)
        except Exception:
            pass
