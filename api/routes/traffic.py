#!/usr/bin/env python3
"""交通流事件 API"""
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import os
from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from api.models import TrafficEvent, get_db
from api.utils.report_aggregation import (
    build_vd_report_rows,
    normalize_bucket_size,
)

router = APIRouter(prefix="/api/traffic", tags=["交通流"])

# 聚合結果 cache：快取「完整結果」(range/bucket/cam → (ts, result))。
# 命中時直接回先前已重建好的完整結果,不去讀持久化聚合表(該表的 5m/1m 可能稀疏 → 會缺口)。
_AGG_CACHE: dict[tuple, tuple] = {}
_AGG_CACHE_TTL_SEC = 60.0

# 趨勢查詢 cache。GROUP BY 的分桶鍵是 strftime 算出來的運算式,任何索引都用不上,
# 必然 SCAN + TEMP B-TREE;traffic_events 已累積 230 萬筆,全表掃描實測閒置 4.4 秒、
# 高負載下 17 秒。單一慢請求就會把瀏覽器僅剩的連線佔住,害 /api/health 排不到 socket
# 逾時 → 前端誤判 SERVICE OFFLINE。所以這裡快取結果 + 限制掃描範圍雙管齊下。
_TREND_CACHE: dict[tuple, tuple] = {}
_TREND_CACHE_TTL_SEC = 60.0
_TREND_CACHE_MAX = 64


def _trend_cache_ttl(bucket_sec: int) -> float:
    """快取存活時間隨桶大小放大。

    1 小時桶畫的是 10~20 天的趨勢圖,5 分鐘的資料新鮮度差異在圖上看不出來,
    但每次冷查要掃幾百萬列(實測 5.4 秒)。桶越大 → 圖越長期 → 可以快取越久。
    小桶(秒/分級)是即時觀察用的,維持 60 秒。
    """
    bs = int(bucket_sec or 0)
    if bs >= 3600:
        return 300.0
    if bs >= 300:
        return 120.0
    return _TREND_CACHE_TTL_SEC


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


@router.get("/events/trend")
def get_traffic_events_trend(
    camera_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    bucket_sec: int = Query(3600, ge=1, le=86400),
    mode: str = Query("count"),
    lane: Optional[str] = None,
    direction: Optional[str] = None,
    max_buckets: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    """事件趨勢（時間桶聚合）。

    直接在 DB 依 bucket_sec 分桶算 COUNT / AVG(speed_kmh)，取代前端「抓最新 N 筆原始
    事件再分桶」——高流量相機一小時可破 2000 筆，被 page_size 截斷後整段趨勢會塌成單一
    bucket。回傳最新 max_buckets 個「非空」桶（空桶由前端補 0）；ts 為 UTC epoch 毫秒。"""
    start_time = _to_utc_naive(start_time)
    end_time = _to_utc_naive(end_time)

    # 沒給起點時補一個隱含下界:end(或現在)往回推 max_buckets 個桶。
    # 本來就只回「最新 max_buckets 個非空桶」,而前端也只畫 max_buckets/2 個桶,
    # 所以這個下界不會少給任何前端畫得出來的資料,卻能讓 created_at 索引派上用場,
    # 把全表掃描縮成範圍掃描。
    if start_time is None:
        _span = int(bucket_sec) * int(max_buckets)
        _anchor = end_time or datetime.utcnow()
        # 對齊到桶格線(至少 60 秒),否則 utcnow() 每次都不同 → cache key 每次都變、永遠 miss
        _grid = max(int(bucket_sec), 60)
        _anchor_epoch = (int(_anchor.replace(tzinfo=timezone.utc).timestamp()) // _grid) * _grid
        _anchor = datetime.utcfromtimestamp(_anchor_epoch)
        start_time = _anchor - timedelta(seconds=_span)

    cache_key = (
        int(bucket_sec), int(max_buckets), str(mode),
        camera_id, lane, direction,
        start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else None,
    )
    now_ts = time.time()
    hit = _TREND_CACHE.get(cache_key)
    if hit and (now_ts - hit[0]) < _trend_cache_ttl(bucket_sec):
        return hit[1]

    try:
        db.execute(text("PRAGMA busy_timeout = 5000"))
    except Exception:
        pass

    conds = []
    params: dict[str, Any] = {"bs": int(bucket_sec)}
    if camera_id is not None:
        conds.append("camera_id = :cid"); params["cid"] = int(camera_id)
    if start_time is not None:
        conds.append("created_at >= :start"); params["start"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    if end_time is not None:
        conds.append("created_at <= :end"); params["end"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    if lane is not None and str(lane).strip() != "":
        try:
            params["lane"] = int(str(lane).strip()); conds.append("lane_no = :lane")
        except ValueError:
            pass
    if direction is not None and str(direction).strip() != "":
        conds.append("direction = :dir"); params["dir"] = str(direction).strip()
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    params["maxb"] = int(max_buckets)
    # 用 substr(1,19)+replace 把 created_at 正規化成 'YYYY-MM-DD HH:MM:SS'（去掉小數秒/時區，
    # 兼容 SQLite 空白分隔與 ISO 'T'+tz 兩種儲存格式），strftime 才能穩定算 UTC epoch。
    sql = text(
        "SELECT (CAST(strftime('%s', replace(substr(created_at,1,19),'T',' ')) AS INTEGER) / :bs) * :bs AS bucket, "
        "COUNT(*) AS cnt, AVG(speed_kmh) AS avg_speed "
        "FROM traffic_events" + where +
        " GROUP BY bucket ORDER BY bucket DESC LIMIT :maxb"
    )

    def run():
        return db.execute(sql, params).fetchall()

    rows = _run_with_sqlite_retry(db, run) or []
    buckets = [
        {
            "ts": int(r[0]) * 1000,
            "count": int(r[1] or 0),
            "avg_speed": (round(float(r[2]), 1) if r[2] is not None else None),
        }
        for r in rows if r[0] is not None
    ]
    buckets.sort(key=lambda b: b["ts"])
    result = {"bucket_sec": int(bucket_sec), "mode": mode, "buckets": buckets}
    if len(_TREND_CACHE) >= _TREND_CACHE_MAX:
        for _k in sorted(_TREND_CACHE, key=lambda k: _TREND_CACHE[k][0])[:_TREND_CACHE_MAX // 2]:
            _TREND_CACHE.pop(_k, None)
    _TREND_CACHE[cache_key] = (now_ts, result)
    return result


@router.get("/vd-report")
def get_vd_report(
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

    def build_rows():
        return build_vd_report_rows(
            db,
            start_time=start_time,
            end_time=end_time,
            bucket_size=bucket_size,
            camera_id=camera_id,
        )

    cache_key = (
        start_time.replace(second=0, microsecond=0),
        end_time.replace(second=0, microsecond=0),
        bucket_size,
        camera_id,
    )
    now_ts = time.time()
    hit = _AGG_CACHE.get(cache_key)
    if hit is not None and (now_ts - hit[0]) < _AGG_CACHE_TTL_SEC:
        # 回先前已重建好的「完整結果」,不重讀可能稀疏的聚合表 → 不會缺口
        return hit[1]
    # 未命中: 只讀聚合表（背景 job 每分鐘增量維護）→ 快取完整結果。
    # 不在請求當下重建 — 重建的 DELETE 會與即時事件寫入搶鎖(報表 0 筆/500 根因)。
    rows = _run_with_sqlite_retry(db, build_rows)
    result = {
        "bucket_size": bucket_size,
        "camera_id": camera_id,
        "start_time": start_time.replace(tzinfo=timezone.utc).isoformat(),
        "end_time": end_time.replace(tzinfo=timezone.utc).isoformat(),
        "aggregation": {},
        "items": rows,
    }
    _AGG_CACHE[cache_key] = (now_ts, result)
    if len(_AGG_CACHE) > 100:
        cutoff = now_ts - _AGG_CACHE_TTL_SEC * 2
        for k in list(_AGG_CACHE.keys()):
            if _AGG_CACHE[k][0] < cutoff:
                del _AGG_CACHE[k]
    return result


@router.get("/events")
def get_traffic_events(
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
# 事件快照 cache。預設 /tmp 在 eMMC 上,現場實測兩天就累積 9.5 萬檔 / 2.4GB,
# 把 54G 的系統碟推到 85%。用 EVENT_SNAPSHOT_DIR 指到 NVMe 即可。
_EVENT_SNAPSHOT_CACHE_DIR = os.getenv("EVENT_SNAPSHOT_DIR", "/tmp/event_snapshots")


@router.get("/events/{event_id}/snapshot.jpg")
def get_event_snapshot(event_id: int, db: Session = Depends(get_db)):
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
