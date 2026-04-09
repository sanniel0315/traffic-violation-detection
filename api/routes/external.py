"""對外報表 API 端點 — VD 車流報表 + 壅塞報表"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.models import ApiKey, CongestionReportAgg, get_db
from api.utils.api_key_auth import require_scope
from api.utils.report_aggregation import (
    build_vd_report_rows,
    normalize_bucket_size,
    refresh_report_aggregates_for_range,
)

router = APIRouter(prefix="/api/v1/external", tags=["External API"])

TZ_TAIPEI = timezone(timedelta(hours=8))

_BUCKET_INTERVALS = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "1h": timedelta(hours=1)}
_MAX_RANGE = {"1m": timedelta(hours=24), "5m": timedelta(days=7), "1h": timedelta(days=90)}
_MAX_RECORDS = 10000
_DEVICE_ID = "jetson-nx-001"


def _meta(fmt: str = "json") -> dict:
    return {
        "request_time": datetime.now(TZ_TAIPEI).isoformat(),
        "api_version": "1.0",
        "device_id": _DEVICE_ID,
        "format": fmt,
    }


def _validate_time_range(start_time: datetime, end_time: datetime, bucket_size: str) -> None:
    if end_time <= start_time:
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "error": {"code": "INVALID_PARAMETER", "message": "end_time 必須大於 start_time"},
        })
    max_range = _MAX_RANGE.get(bucket_size, timedelta(days=7))
    if (end_time - start_time) > max_range:
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "error": {"code": "RANGE_TOO_LARGE", "message": f"時間範圍超過上限 ({bucket_size} 最多 {max_range})"},
        })


# ── VD 車流報表 ──────────────────────────────────────────────

@router.get("/vd-report")
async def external_vd_report(
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
    detector_id: Optional[int] = Query(None),
    interval: str = Query("5m"),
    format: str = Query("json"),
    api_key: ApiKey = Depends(require_scope("vd_report")),
    db: Session = Depends(get_db),
):
    bucket = normalize_bucket_size(interval)
    _validate_time_range(start_time, end_time, bucket)

    refresh_report_aggregates_for_range(db, start_time, end_time, bucket_sizes=(bucket,), camera_id=detector_id)
    rows = build_vd_report_rows(db, start_time, end_time, bucket, camera_id=detector_id)

    bucket_delta = _BUCKET_INTERVALS.get(bucket, timedelta(minutes=5))
    records = []
    for row in rows:
        ts = row.get("timeKey")
        time_start = datetime.fromtimestamp(ts / 1000, tz=TZ_TAIPEI).isoformat() if ts else None
        time_end = (datetime.fromtimestamp(ts / 1000, tz=TZ_TAIPEI) + bucket_delta).isoformat() if ts else None

        lanes_raw = row.get("lanes") or {}
        lanes = []
        for lane_no, ld in lanes_raw.items():
            lanes.append({
                "lane_no": int(lane_no) if str(lane_no).isdigit() else lane_no,
                "flow": ld.get("flow", 0),
                "small_vehicle_flow": ld.get("smallFlow", 0),
                "large_vehicle_flow": ld.get("largeFlow", 0),
                "avg_speed_kmh": round(ld.get("avgSpeed") or 0, 1),
                "avg_occupancy_pct": round(ld.get("avgOccupancyPct") or 0, 1),
                "avg_queue_length_m": round(ld.get("avgQueueLengthM") or 0, 1) if ld.get("avgQueueLengthM") else None,
                "max_queue_length_m": round(ld.get("maxQueueLengthM") or 0, 1) if ld.get("maxQueueLengthM") else None,
            })

        records.append({
            "detector_id": row.get("deviceId", ""),
            "road_name": row.get("roadName", ""),
            "time_start": time_start,
            "time_end": time_end,
            "direction": row.get("direction", ""),
            "direction_label": row.get("directionText", ""),
            "total_flow": row.get("totalFlow", 0),
            "small_vehicle_flow": row.get("smallFlow", 0),
            "large_vehicle_flow": row.get("largeFlow", 0),
            "avg_speed_kmh": round(row.get("avgSpeed") or 0, 1),
            "avg_occupancy_pct": round(row.get("avgOccupancyPct") or 0, 1),
            "lane_count": row.get("laneCount", 0),
            "lanes": lanes,
        })

    if len(records) > _MAX_RECORDS:
        raise HTTPException(status_code=413, detail={
            "status": "error",
            "error": {"code": "TOO_MANY_RECORDS", "message": f"結果超過 {_MAX_RECORDS} 筆，請縮小時間範圍"},
        })

    if format == "csv":
        return _vd_csv_response(records, start_time, end_time, bucket)

    return {
        "status": "success",
        "data": {
            "interval": bucket,
            "period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "records": records,
        },
        "meta": _meta("json"),
    }


def _vd_csv_response(records: list, start_time, end_time, bucket):
    output = io.StringIO()
    writer = csv.writer(output)
    header = [
        "detector_id", "road_name", "time_start", "time_end", "direction",
        "lane_no", "flow", "small_vehicle_flow", "large_vehicle_flow",
        "avg_speed_kmh", "avg_occupancy_pct", "avg_queue_length_m", "max_queue_length_m",
    ]
    writer.writerow(header)
    for rec in records:
        lanes = rec.get("lanes") or []
        if lanes:
            for lane in lanes:
                writer.writerow([
                    rec["detector_id"], rec["road_name"], rec["time_start"], rec["time_end"],
                    rec["direction"], lane["lane_no"], lane["flow"],
                    lane["small_vehicle_flow"], lane["large_vehicle_flow"],
                    lane["avg_speed_kmh"], lane["avg_occupancy_pct"],
                    lane.get("avg_queue_length_m", ""), lane.get("max_queue_length_m", ""),
                ])
        else:
            writer.writerow([
                rec["detector_id"], rec["road_name"], rec["time_start"], rec["time_end"],
                rec["direction"], "", rec["total_flow"],
                rec["small_vehicle_flow"], rec["large_vehicle_flow"],
                rec["avg_speed_kmh"], rec["avg_occupancy_pct"], "", "",
            ])

    output.seek(0)
    filename = f"vd_report_{bucket}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── 壅塞報表 ──────────────────────────────────────────────

@router.get("/congestion-report")
async def external_congestion_report(
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
    detector_id: Optional[int] = Query(None),
    interval: str = Query("5m"),
    format: str = Query("json"),
    api_key: ApiKey = Depends(require_scope("congestion_report")),
    db: Session = Depends(get_db),
):
    bucket = normalize_bucket_size(interval)
    _validate_time_range(start_time, end_time, bucket)

    refresh_report_aggregates_for_range(db, start_time, end_time, bucket_sizes=(bucket,), camera_id=detector_id)

    query = db.query(CongestionReportAgg).filter(
        CongestionReportAgg.bucket_size == bucket,
        CongestionReportAgg.bucket_start >= start_time,
        CongestionReportAgg.bucket_start < end_time,
    )
    if detector_id:
        query = query.filter(CongestionReportAgg.camera_id == detector_id)

    rows = query.order_by(CongestionReportAgg.bucket_start).all()

    if len(rows) > _MAX_RECORDS:
        raise HTTPException(status_code=413, detail={
            "status": "error",
            "error": {"code": "TOO_MANY_RECORDS", "message": f"結果超過 {_MAX_RECORDS} 筆，請縮小時間範圍"},
        })

    bucket_delta = _BUCKET_INTERVALS.get(bucket, timedelta(minutes=5))
    records = []
    for r in rows:
        ts = r.bucket_start.replace(tzinfo=timezone.utc).astimezone(TZ_TAIPEI) if r.bucket_start else None
        records.append({
            "detector_id": str(r.camera_id or ""),
            "camera_name": r.camera_name or "",
            "time_start": ts.isoformat() if ts else None,
            "time_end": (ts + bucket_delta).isoformat() if ts else None,
            "zone_name": r.zone_name or "",
            "lane_no": r.lane_no,
            "direction": r.direction or "",
            "avg_occupancy_pct": round((r.avg_occupancy or 0) * 100, 1),
            "max_occupancy_pct": round((r.max_occupancy or 0) * 100, 1),
            "avg_vehicle_count": round(r.avg_vehicle_count or 0, 1),
            "avg_stopped_vehicle_count": round(r.avg_stopped_vehicle_count or 0, 1),
            "avg_queue_length_m": round(r.avg_queue_length_m or 0, 1),
            "max_queue_length_m": round(r.max_queue_length_m or 0, 1),
            "queue_active_duration_sec": round(r.queue_active_duration_sec or 0, 1),
            "sample_count": r.sample_count or 0,
        })

    if format == "csv":
        return _congestion_csv_response(records, start_time, end_time, bucket)

    return {
        "status": "success",
        "data": {
            "interval": bucket,
            "period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "records": records,
        },
        "meta": _meta("json"),
    }


def _congestion_csv_response(records: list, start_time, end_time, bucket):
    output = io.StringIO()
    writer = csv.writer(output)
    header = [
        "detector_id", "camera_name", "time_start", "time_end",
        "zone_name", "lane_no", "direction",
        "avg_occupancy_pct", "max_occupancy_pct",
        "avg_vehicle_count", "avg_stopped_vehicle_count",
        "avg_queue_length_m", "max_queue_length_m",
        "queue_active_duration_sec", "sample_count",
    ]
    writer.writerow(header)
    for r in records:
        writer.writerow([r.get(h, "") for h in header])

    output.seek(0)
    filename = f"congestion_report_{bucket}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
