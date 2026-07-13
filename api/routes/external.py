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

    # 只讀聚合表（背景 job 每分鐘增量維護），不在請求當下重建 — 重建的 DELETE
    # 會與即時事件寫入搶鎖，正是報表 0 筆/500 的根因。
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
                "queue_duration_sec": round(ld.get("queueDurationSec") or 0, 1) if ld.get("queueDurationSec") else None,
                "max_queue_duration_sec": round(ld.get("maxQueueDurationSec") or 0, 1) if ld.get("maxQueueDurationSec") else None,
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
            "direction_counts": row.get("directionCounts") or {},
            "avg_queue_length_m": round(row.get("avgQueueLengthM") or 0, 1) if row.get("avgQueueLengthM") else None,
            "max_queue_length_m": round(row.get("maxQueueLengthM") or 0, 1) if row.get("maxQueueLengthM") else None,
            "queue_duration_sec": round(row.get("queueDurationSec") or 0, 1) if row.get("queueDurationSec") else None,
            "max_queue_duration_sec": round(row.get("maxQueueDurationSec") or 0, 1) if row.get("maxQueueDurationSec") else None,
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
        "queue_duration_sec", "max_queue_duration_sec",
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
                    lane.get("queue_duration_sec", ""), lane.get("max_queue_duration_sec", ""),
                ])
        else:
            writer.writerow([
                rec["detector_id"], rec["road_name"], rec["time_start"], rec["time_end"],
                rec["direction"], "", rec["total_flow"],
                rec["small_vehicle_flow"], rec["large_vehicle_flow"],
                rec["avg_speed_kmh"], rec["avg_occupancy_pct"],
                rec.get("avg_queue_length_m", "") or "", rec.get("max_queue_length_m", "") or "",
                rec.get("queue_duration_sec", "") or "", rec.get("max_queue_duration_sec", "") or "",
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

    # 同 vd-report：只讀聚合表，重建交給背景 job
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


# ── 即時影像串流清單 ──────────────────────────────────────────────
# 給機關既設數位影像平台統一拿 URL + 規格，後續直接連 RTSP/HLS/MJPEG

def _detect_host_ip() -> str:
    """自動偵測本機對外 IP (UDP socket trick — 不真連線)，env STREAM_HOST 可覆寫"""
    import os, socket
    override = os.getenv("STREAM_HOST", "").strip()
    if override:
        return override
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _parse_go2rtc_sdp(sdp: str) -> dict:
    """從 SDP 抽 codec/profile/resolution/fps"""
    result = {"codec": None, "resolution": None, "fps": None, "profile": None}
    for line in sdp.split("\r\n"):
        if line.startswith("a=rtpmap:") and "H264" in line.upper():
            result["codec"] = "h264"
        elif line.startswith("a=rtpmap:") and ("HEVC" in line.upper() or "H265" in line.upper()):
            result["codec"] = "h265"
        elif line.startswith("a=framesize:"):
            try:
                size = line.split()[-1].replace("-", "x")
                result["resolution"] = size
            except Exception:
                pass
        elif line.startswith("a=framerate:"):
            try:
                result["fps"] = float(line.split(":")[1])
            except Exception:
                pass
        elif "profile-level-id=" in line:
            try:
                pid = line.split("profile-level-id=")[1].split(";")[0][:6]
                # H.264 profile_idc: 42=Baseline 4D=Main 64=High
                profile_idc = pid[:2].upper()
                result["profile"] = {
                    "42": "Baseline", "4D": "Main", "64": "High",
                }.get(profile_idc, profile_idc)
            except Exception:
                pass
    return result


@router.get("/streams")
def external_streams(
    api_key: ApiKey = Depends(require_scope("streams")),
    db: Session = Depends(get_db),
):
    """回傳本系統可對外提供的所有即時影像串流 URL + 規格。
    供機關既設數位影像平台統一查詢。
    """
    import requests as _req
    from api.models import Camera

    host = _detect_host_ip()
    rtsp_port = 8554
    http_port = 1984

    # 從 go2rtc HTTP API 拿目前活躍 streams 跟 producer SDP
    g2rtc = {}
    try:
        r = _req.get(f"http://127.0.0.1:{http_port}/api/streams", timeout=3)
        if r.ok:
            g2rtc = r.json() or {}
    except Exception:
        pass

    # cameras DB 對應 (id → Camera)
    cam_map = {c.id: c for c in db.query(Camera).all()}

    streams = []
    for stream_id, info in g2rtc.items():
        # 只匯出 cam_X 命名的 (跳過內部 / 暫存 stream)
        if not stream_id.startswith("cam_"):
            continue
        try:
            cam_id = int(stream_id.replace("cam_", ""))
        except ValueError:
            continue
        cam = cam_map.get(cam_id)

        producers = info.get("producers") or []
        online = bool(producers)
        spec = {"codec": None, "resolution": None, "fps": None, "profile": None}
        bytes_recv = 0
        if producers:
            p = producers[0]
            spec = _parse_go2rtc_sdp(p.get("sdp", ""))
            for rcv in (p.get("receivers") or []):
                bytes_recv += int(rcv.get("bytes") or 0)

        streams.append({
            "stream_id": stream_id,
            "camera_id": cam_id,
            "name": (cam.name if cam else None),
            "location": (cam.location if cam else None),
            "online": online,
            "codec": spec["codec"],
            "profile": spec["profile"],
            "resolution": spec["resolution"],
            "fps": spec["fps"],
            "bytes_received": bytes_recv,
            "urls": {
                "rtsp":  f"rtsp://{host}:{rtsp_port}/{stream_id}",
                "hls":   f"http://{host}:{http_port}/api/stream.m3u8?src={stream_id}",
                "mjpeg": f"http://{host}:{http_port}/api/stream.mjpeg?src={stream_id}",
                "webrtc_signal": f"http://{host}:{http_port}/api/ws?src={stream_id}",
            },
        })

    streams.sort(key=lambda s: s["camera_id"])

    return {
        "meta": _meta(),
        "status": "success",
        "data": {
            "device_id": _DEVICE_ID,
            "host": host,
            "ports": {"rtsp": rtsp_port, "http": http_port},
            "stream_count": len(streams),
            "streams": streams,
            "spec_requirement": {
                "codec": "H.264 (passthrough，無重編碼)",
                "min_resolution": "1280x720",
                "min_fps": 15,
                "note": "實際解析度/FPS 視相機本機設定，本系統 go2rtc 多路分發不重編碼",
            },
        },
    }
