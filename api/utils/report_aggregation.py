#!/usr/bin/env python3
"""報表聚合工具"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import time
from typing import Iterable, Optional

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from api.models import (
    AggregationJobState,
    Camera,
    CongestionReportAgg,
    CongestionSample,
    LPRRecord,
    LprReportAgg,
    TrafficEvent,
    TrafficReportAgg,
)


TZ_TAIPEI = timezone(timedelta(hours=8))
BUCKET_SECONDS = {"1m": 60, "5m": 300, "1h": 3600}
INCREMENTAL_JOB_NAME = "report_aggregation_incremental"


def normalize_bucket_size(bucket_size: str | None) -> str:
    value = str(bucket_size or "5m").strip().lower()
    return value if value in BUCKET_SECONDS else "5m"


def _is_sqlite_locked(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _set_busy_timeout(db: Session, timeout_ms: int = 5000) -> None:
    try:
        db.execute(text(f"PRAGMA busy_timeout = {int(timeout_ms)}"))
    except Exception:
        pass


def _run_with_retry(db: Session, fn, retries: int = 4):
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


def to_utc_naive(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def bucket_floor(value: datetime, bucket_size: str) -> datetime:
    bucket_seconds = BUCKET_SECONDS[normalize_bucket_size(bucket_size)]
    ts = int(value.replace(tzinfo=timezone.utc).timestamp())
    floored = ts - (ts % bucket_seconds)
    return datetime.utcfromtimestamp(floored)


def bucket_ceil(value: datetime, bucket_size: str) -> datetime:
    floored = bucket_floor(value, bucket_size)
    if floored == value:
        return floored
    return floored + timedelta(seconds=BUCKET_SECONDS[normalize_bucket_size(bucket_size)])


def bucket_to_epoch_ms(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


def format_bucket_time(value: datetime, bucket_size: str) -> str:
    dt_local = value.replace(tzinfo=timezone.utc).astimezone(TZ_TAIPEI)
    if bucket_size == "1h":
        return dt_local.strftime("%Y-%m-%d %H:00")
    return dt_local.strftime("%Y-%m-%d %H:%M")


def direction_label(value: str | None) -> str:
    return {
        "left": "左轉",
        "straight": "直行",
        "right": "右轉",
        "IN": "IN",
        "OUT": "OUT",
        "N2S": "北向南",
        "S2N": "南向北",
        "E2W": "東向西",
        "W2E": "西向東",
    }.get(str(value or "").strip(), "未知")


def normalize_direction(value: str | None) -> str:
    raw = str(value or "").strip()
    upper = raw.upper()
    if upper in {"IN", "OUT", "N2S", "S2N", "E2W", "W2E"}:
        return upper
    lower = raw.lower()
    if "left" in lower or "左" in lower:
        return "left"
    if "straight" in lower or "middle" in lower or "mid" in lower or "直" in lower:
        return "straight"
    if "right" in lower or "右" in lower:
        return "right"
    if "in" in lower or "進" in lower or "入口" in lower:
        return "IN"
    if "out" in lower or "出" in lower or "出口" in lower:
        return "OUT"
    return "unknown"


def classify_vehicle_group(label: str | None) -> str:
    text = str(label or "").lower()
    if any(token in text for token in (
        "heavy_truck", "trailer", "tractor", "聯結", "bus", "大車", "大貨車", "大客車",
    )):
        return "large"
    if any(token in text for token in (
        "light_truck", "小貨車",
        "car", "motorcycle", "bicycle", "小車",
    )):
        return "small"
    # 未細分的 truck 仍歸 large
    if "truck" in text:
        return "large"
    return "other"


def is_vd_zone(zone: dict | None) -> bool:
    scope = str((zone or {}).get("scope") or "").strip()
    zone_type = str((zone or {}).get("type") or "").strip()
    if scope and scope != "traffic_flow_settings":
        return False
    return zone_type in {"flow_detection", "detection", "lane_left", "lane_straight", "lane_right"}


def _camera_meta(db: Session):
    by_id: dict[int, dict] = {}
    by_name: dict[str, dict] = {}
    cameras = db.query(Camera).all()
    for cam in cameras:
        zones = list(cam.zones or []) if isinstance(cam.zones, list) else []
        lane_set = set()
        direction_counts: dict[str, int] = {}
        for zone in zones:
            if not is_vd_zone(zone):
                continue
            lane_no = zone.get("lane_no", zone.get("lane_id", zone.get("laneNo")))
            lane_no = int(lane_no) if str(lane_no).isdigit() else None
            if lane_no and lane_no > 0:
                lane_set.add(lane_no)
            direction = normalize_direction(zone.get("direction"))
            if direction != "unknown":
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
        main_direction = "unknown"
        if direction_counts:
            main_direction = sorted(direction_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]
        meta = {
            "camera_id": int(cam.id),
            "camera_name": str(cam.name or f"cam_{cam.id}"),
            "road_name": str(cam.location or "").strip() or "未知",
            "lane_count": len(lane_set),
            "direction": main_direction,
            "vd_eligible": any(is_vd_zone(zone) for zone in zones),
        }
        by_id[int(cam.id)] = meta
        by_name[str(meta["camera_name"])] = meta
        by_name[f"cam_{cam.id}"] = meta
    return by_id, by_name


def _delete_agg_range(db: Session, model, bucket_size: str, start: datetime, end: datetime, camera_id: int | None = None):
    query = db.query(model).filter(
        model.bucket_size == bucket_size,
        model.bucket_start >= start,
        model.bucket_start < end,
    )
    if camera_id is not None:
        query = query.filter(model.camera_id == int(camera_id))
    query.delete(synchronize_session=False)


def refresh_traffic_aggregates(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    bucket_size: str,
    camera_id: int | None = None,
) -> int:
    bucket_size = normalize_bucket_size(bucket_size)
    start = bucket_floor(start_time, bucket_size)
    end = bucket_ceil(end_time, bucket_size)
    if end <= start:
        return 0
    camera_by_id, _ = _camera_meta(db)
    query = db.query(TrafficEvent).filter(TrafficEvent.created_at >= start, TrafficEvent.created_at < end)
    if camera_id is not None:
        query = query.filter(TrafficEvent.camera_id == int(camera_id))
    events = query.all()
    buckets: dict[tuple[datetime, int, str, int], dict] = {}
    for evt in events:
        if evt.camera_id is None:
            continue
        bucket_start = bucket_floor(evt.created_at or start, bucket_size)
        direction = normalize_direction(evt.direction)
        lane_no = int(evt.lane_no) if evt.lane_no is not None else 0
        key = (bucket_start, int(evt.camera_id), direction, lane_no)
        meta = camera_by_id.get(int(evt.camera_id), {})
        if key not in buckets:
            buckets[key] = {
                "bucket_start": bucket_start,
                "bucket_size": bucket_size,
                "camera_id": int(evt.camera_id),
                "camera_name": str(meta.get("camera_name") or f"cam_{evt.camera_id}"),
                "road_name": str(meta.get("road_name") or "未知"),
                "direction": direction,
                "lane_no": lane_no or None,
                "total_flow": 0,
                "speed_sum": 0.0,
                "speed_count": 0,
                "max_speed": 0.0,
                "occupancy_sum": 0.0,
                "occupancy_count": 0,
                "small_vehicle_flow": 0,
                "large_vehicle_flow": 0,
                "other_vehicle_flow": 0,
                "event_count": 0,
            }
        row = buckets[key]
        row["total_flow"] += 1
        row["event_count"] += 1
        speed_val = float(evt.speed_kmh) if evt.speed_kmh is not None else None
        if speed_val is not None and speed_val > 0:
            row["speed_sum"] += speed_val
            row["speed_count"] += 1
            row["max_speed"] = max(float(row["max_speed"] or 0.0), speed_val)
        occupancy = float(evt.occupancy) if evt.occupancy is not None else None
        if occupancy is not None and occupancy >= 0:
            row["occupancy_sum"] += occupancy
            row["occupancy_count"] += 1
        group = classify_vehicle_group(evt.label)
        row[f"{group}_vehicle_flow"] += 1
    _delete_agg_range(db, TrafficReportAgg, bucket_size, start, end, camera_id=camera_id)
    insert_rows = []
    for row in buckets.values():
        insert_rows.append(
            {
                "bucket_start": row["bucket_start"],
                "bucket_size": row["bucket_size"],
                "camera_id": row["camera_id"],
                "camera_name": row["camera_name"],
                "road_name": row["road_name"],
                "direction": row["direction"],
                "lane_no": row["lane_no"],
                "total_flow": row["total_flow"],
                "avg_speed": (row["speed_sum"] / row["speed_count"]) if row["speed_count"] else None,
                "max_speed": row["max_speed"] if row["speed_count"] else None,
                "avg_occupancy": (row["occupancy_sum"] / row["occupancy_count"]) if row["occupancy_count"] else None,
                "small_vehicle_flow": row["small_vehicle_flow"],
                "large_vehicle_flow": row["large_vehicle_flow"],
                "other_vehicle_flow": row["other_vehicle_flow"],
                "event_count": row["event_count"],
                "updated_at": datetime.utcnow(),
            }
        )
    if insert_rows:
        db.bulk_insert_mappings(TrafficReportAgg, insert_rows)
    return len(insert_rows)


def refresh_congestion_aggregates(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    bucket_size: str,
    camera_id: int | None = None,
) -> int:
    bucket_size = normalize_bucket_size(bucket_size)
    start = bucket_floor(start_time, bucket_size)
    end = bucket_ceil(end_time, bucket_size)
    if end <= start:
        return 0
    query = db.query(CongestionSample).filter(CongestionSample.created_at >= start, CongestionSample.created_at < end)
    if camera_id is not None:
        query = query.filter(CongestionSample.camera_id == int(camera_id))
    samples = query.all()
    buckets: dict[tuple[datetime, int, bool, str, int], dict] = {}
    for sample in samples:
        if sample.camera_id is None:
            continue
        bucket_start = bucket_floor(sample.created_at or start, bucket_size)
        lane_no = int(sample.lane_no) if sample.lane_no is not None else 0
        zone_name = str(sample.zone_name or "")
        key = (bucket_start, int(sample.camera_id), bool(sample.is_overall), zone_name, lane_no)
        if key not in buckets:
            buckets[key] = {
                "bucket_start": bucket_start,
                "bucket_size": bucket_size,
                "camera_id": int(sample.camera_id),
                "camera_name": str(sample.camera_name or f"cam_{sample.camera_id}"),
                "zone_name": zone_name,
                "lane_no": lane_no or None,
                "direction": normalize_direction(sample.direction),
                "movement": str(sample.movement or ""),
                "is_overall": bool(sample.is_overall),
                "occupancy_sum": 0.0,
                "occupancy_count": 0,
                "max_occupancy": 0.0,
                "vehicle_count_sum": 0.0,
                "stopped_vehicle_count_sum": 0.0,
                "queue_length_sum": 0.0,
                "queue_length_count": 0,
                "max_queue_length_m": 0.0,
                "queue_active_duration_sec": 0.0,
                "max_queue_duration_sec": 0.0,
                "sample_count": 0,
            }
        row = buckets[key]
        occupancy = float(sample.occupancy) if sample.occupancy is not None else None
        if occupancy is not None and occupancy >= 0:
            row["occupancy_sum"] += occupancy
            row["occupancy_count"] += 1
            row["max_occupancy"] = max(float(row["max_occupancy"] or 0.0), occupancy)
        row["vehicle_count_sum"] += float(sample.vehicle_count or 0)
        row["stopped_vehicle_count_sum"] += float(sample.stopped_vehicle_count or 0)
        queue_length = float(sample.estimated_queue_length_m) if sample.estimated_queue_length_m is not None else None
        if queue_length is not None and queue_length >= 0:
            row["queue_length_sum"] += queue_length
            row["queue_length_count"] += 1
            row["max_queue_length_m"] = max(float(row["max_queue_length_m"] or 0.0), queue_length)
        if bool(sample.queue_active):
            sample_interval = float(sample.sample_interval_sec or 0.0)
            if sample_interval > 0:
                row["queue_active_duration_sec"] += sample_interval
        queue_duration = float(sample.queue_duration_sec) if sample.queue_duration_sec is not None else None
        if queue_duration is not None and queue_duration >= 0:
            row["max_queue_duration_sec"] = max(float(row["max_queue_duration_sec"] or 0.0), queue_duration)
        row["sample_count"] += 1
    _delete_agg_range(db, CongestionReportAgg, bucket_size, start, end, camera_id=camera_id)
    insert_rows = []
    for row in buckets.values():
        sample_count = row["sample_count"] or 0
        insert_rows.append(
            {
                "bucket_start": row["bucket_start"],
                "bucket_size": row["bucket_size"],
                "camera_id": row["camera_id"],
                "camera_name": row["camera_name"],
                "zone_name": row["zone_name"],
                "lane_no": row["lane_no"],
                "direction": row["direction"],
                "movement": row["movement"],
                "is_overall": row["is_overall"],
                "avg_occupancy": (row["occupancy_sum"] / row["occupancy_count"]) if row["occupancy_count"] else None,
                "max_occupancy": row["max_occupancy"] if row["occupancy_count"] else None,
                "avg_vehicle_count": (row["vehicle_count_sum"] / sample_count) if sample_count else None,
                "avg_stopped_vehicle_count": (row["stopped_vehicle_count_sum"] / sample_count) if sample_count else None,
                "avg_queue_length_m": (row["queue_length_sum"] / row["queue_length_count"]) if row["queue_length_count"] else None,
                "max_queue_length_m": row["max_queue_length_m"] if row["queue_length_count"] else None,
                "queue_active_duration_sec": row["queue_active_duration_sec"] if sample_count else None,
                "max_queue_duration_sec": row["max_queue_duration_sec"] if sample_count else None,
                "sample_count": sample_count,
                "updated_at": datetime.utcnow(),
            }
        )
    if insert_rows:
        db.bulk_insert_mappings(CongestionReportAgg, insert_rows)
    return len(insert_rows)


def refresh_lpr_aggregates(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    bucket_size: str = "1h",
    camera_id: int | None = None,
) -> int:
    bucket_size = normalize_bucket_size(bucket_size)
    start = bucket_floor(start_time, bucket_size)
    end = bucket_ceil(end_time, bucket_size)
    if end <= start:
        return 0
    query = db.query(LPRRecord).filter(LPRRecord.created_at >= start, LPRRecord.created_at < end)
    if camera_id is not None:
        query = query.filter(LPRRecord.camera_id == int(camera_id))
    records = query.all()
    buckets: dict[tuple[datetime, int], dict] = {}
    for record in records:
        if record.camera_id is None:
            continue
        bucket_start = bucket_floor(record.created_at or start, bucket_size)
        key = (bucket_start, int(record.camera_id))
        if key not in buckets:
            buckets[key] = {
                "bucket_start": bucket_start,
                "bucket_size": bucket_size,
                "camera_id": int(record.camera_id),
                "camera_name": str(record.camera_name or f"cam_{record.camera_id}"),
                "total_records": 0,
                "confirmed_records": 0,
                "unknown_records": 0,
                "confidence_sum": 0.0,
                "confidence_count": 0,
            }
        row = buckets[key]
        row["total_records"] += 1
        if bool(record.valid):
            row["confirmed_records"] += 1
        if str(record.plate_number or "").strip().upper() in {"", "UNKNOWN", "VEHICLE_ONLY"}:
            row["unknown_records"] += 1
        conf = float(record.confidence) if record.confidence is not None else None
        if conf is not None and conf >= 0:
            row["confidence_sum"] += conf
            row["confidence_count"] += 1
    _delete_agg_range(db, LprReportAgg, bucket_size, start, end, camera_id=camera_id)
    insert_rows = []
    for row in buckets.values():
        insert_rows.append(
            {
                "bucket_start": row["bucket_start"],
                "bucket_size": row["bucket_size"],
                "camera_id": row["camera_id"],
                "camera_name": row["camera_name"],
                "total_records": row["total_records"],
                "confirmed_records": row["confirmed_records"],
                "unknown_records": row["unknown_records"],
                "avg_confidence": (row["confidence_sum"] / row["confidence_count"]) if row["confidence_count"] else None,
                "updated_at": datetime.utcnow(),
            }
        )
    if insert_rows:
        db.bulk_insert_mappings(LprReportAgg, insert_rows)
    return len(insert_rows)


def refresh_report_aggregates_for_range(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    bucket_sizes: Iterable[str] = ("1m", "5m", "1h"),
    camera_id: int | None = None,
) -> dict[str, int]:
    _set_busy_timeout(db)
    start = to_utc_naive(start_time)
    end = to_utc_naive(end_time)
    if start is None or end is None or end < start:
        return {}
    result: dict[str, int] = {}
    for bucket_size in {normalize_bucket_size(item) for item in bucket_sizes}:
        result[f"traffic_{bucket_size}"] = refresh_traffic_aggregates(db, start, end, bucket_size, camera_id=camera_id)
        result[f"congestion_{bucket_size}"] = refresh_congestion_aggregates(db, start, end, bucket_size, camera_id=camera_id)
    result["lpr_1h"] = refresh_lpr_aggregates(db, start, end, "1h", camera_id=camera_id)
    db.commit()
    return result


def _first_raw_timestamp(db: Session) -> Optional[datetime]:
    candidates = []
    for model, field in (
        (TrafficEvent, TrafficEvent.created_at),
        (CongestionSample, CongestionSample.created_at),
        (LPRRecord, LPRRecord.created_at),
    ):
        value = db.query(model).order_by(field.asc()).with_entities(field).first()
        if value and value[0]:
            candidates.append(value[0])
    if not candidates:
        return None
    return min(candidates)


def run_incremental_report_aggregation(
    db: Session,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, object]:
    _set_busy_timeout(db)
    explicit_start = to_utc_naive(start_time)
    explicit_end = to_utc_naive(end_time) or datetime.utcnow()

    def _run():
        state = db.query(AggregationJobState).filter(AggregationJobState.job_name == INCREMENTAL_JOB_NAME).first()
        if state is None:
            state = AggregationJobState(job_name=INCREMENTAL_JOB_NAME)
            db.add(state)
            db.flush()
        if explicit_start is not None:
            start = explicit_start
        elif state.last_processed_at is not None:
            start = state.last_processed_at - timedelta(hours=1)
        else:
            start = _first_raw_timestamp(db)
        if start is None:
            state.last_processed_at = explicit_end
            db.commit()
            return {"refreshed": {}, "start_time": None, "end_time": explicit_end}
        refreshed = refresh_report_aggregates_for_range(db, start, explicit_end)
        state.last_processed_at = explicit_end
        state.updated_at = datetime.utcnow()
        db.commit()
        return {"refreshed": refreshed, "start_time": start, "end_time": explicit_end}

    return _run_with_retry(db, _run)


def build_vd_report_rows(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    bucket_size: str,
    camera_id: int | None = None,
) -> list[dict]:
    start = bucket_floor(to_utc_naive(start_time) or datetime.utcnow(), bucket_size)
    end_exclusive = bucket_ceil(to_utc_naive(end_time) or datetime.utcnow(), bucket_size)
    camera_by_id, camera_by_name = _camera_meta(db)

    traffic_query = db.query(TrafficReportAgg).filter(
        TrafficReportAgg.bucket_size == bucket_size,
        TrafficReportAgg.bucket_start >= start,
        TrafficReportAgg.bucket_start < end_exclusive,
    )
    congestion_query = db.query(CongestionReportAgg).filter(
        CongestionReportAgg.bucket_size == bucket_size,
        CongestionReportAgg.bucket_start >= start,
        CongestionReportAgg.bucket_start < end_exclusive,
    )
    if camera_id is not None:
        traffic_query = traffic_query.filter(TrafficReportAgg.camera_id == int(camera_id))
        congestion_query = congestion_query.filter(CongestionReportAgg.camera_id == int(camera_id))
    traffic_rows = traffic_query.all()
    congestion_rows = congestion_query.all()

    buckets: dict[tuple[str, datetime], dict] = {}

    def ensure_row(device_id: str, bucket_start: datetime) -> dict:
        key = (device_id, bucket_start)
        meta = camera_by_name.get(device_id, {})
        if key not in buckets:
            buckets[key] = {
                "deviceId": device_id,
                "roadName": str(meta.get("road_name") or "未知"),
                "timeKey": bucket_to_epoch_ms(bucket_start),
                "timeText": format_bucket_time(bucket_start, bucket_size),
                "direction": str(meta.get("direction") or "unknown"),
                "directionText": direction_label(str(meta.get("direction") or "unknown")),
                "directionCounts": {},
                "totalFlow": 0,
                "avgSpeed": None,
                "avgOccupancyPct": None,
                "smallFlow": 0,
                "largeFlow": 0,
                "avgQueueLengthM": None,
                "maxQueueLengthM": None,
                "queueDurationSec": None,
                "maxQueueDurationSec": None,
                "laneCount": int(meta.get("lane_count") or 0),
                "lanes": {},
                "_speed_weight_sum": 0.0,
                "_speed_weight_count": 0,
                "_occupancy_sum_pct": 0.0,
                "_occupancy_count": 0,
                "_queue_sum": 0.0,
                "_queue_count": 0,
            }
        return buckets[key]

    def ensure_lane(row: dict, lane_no: int) -> dict:
        lane_key = int(lane_no)
        if lane_key not in row["lanes"]:
            row["lanes"][lane_key] = {
                "flow": 0,
                "avgSpeed": None,
                "avgOccupancyPct": None,
                "smallFlow": 0,
                "largeFlow": 0,
                "avgQueueLengthM": None,
                "maxQueueLengthM": None,
                "queueDurationSec": None,
                "maxQueueDurationSec": None,
                "_speed_weight_sum": 0.0,
                "_speed_weight_count": 0,
                "_occupancy_sum_pct": 0.0,
                "_occupancy_count": 0,
                "_queue_sum": 0.0,
                "_queue_count": 0,
            }
        return row["lanes"][lane_key]

    for agg in traffic_rows:
        device_id = str(agg.camera_name or camera_by_id.get(int(agg.camera_id), {}).get("camera_name") or f"cam_{agg.camera_id}")
        row = ensure_row(device_id, agg.bucket_start)
        direction = normalize_direction(agg.direction)
        row["directionCounts"][direction] = row["directionCounts"].get(direction, 0) + int(agg.total_flow or 0)
        row["totalFlow"] += int(agg.total_flow or 0)
        row["smallFlow"] += int(agg.small_vehicle_flow or 0)
        row["largeFlow"] += int(agg.large_vehicle_flow or 0)
        event_count = int(agg.event_count or agg.total_flow or 0)
        if agg.avg_speed is not None and event_count > 0:
            row["_speed_weight_sum"] += float(agg.avg_speed) * event_count
            row["_speed_weight_count"] += event_count
        if agg.avg_occupancy is not None and event_count > 0:
            occ_pct = float(agg.avg_occupancy)
            if occ_pct <= 1:
                occ_pct *= 100.0
            row["_occupancy_sum_pct"] += occ_pct * event_count
            row["_occupancy_count"] += event_count
        lane_no = int(agg.lane_no) if agg.lane_no is not None else 0
        if lane_no > 0:
            lane = ensure_lane(row, lane_no)
            lane["flow"] += int(agg.total_flow or 0)
            lane["smallFlow"] += int(agg.small_vehicle_flow or 0)
            lane["largeFlow"] += int(agg.large_vehicle_flow or 0)
            if agg.avg_speed is not None and event_count > 0:
                lane["_speed_weight_sum"] += float(agg.avg_speed) * event_count
                lane["_speed_weight_count"] += event_count
            if agg.avg_occupancy is not None and event_count > 0:
                occ_pct = float(agg.avg_occupancy)
                if occ_pct <= 1:
                    occ_pct *= 100.0
                lane["_occupancy_sum_pct"] += occ_pct * event_count
                lane["_occupancy_count"] += event_count
            row["laneCount"] = max(int(row["laneCount"] or 0), lane_no)

    for agg in congestion_rows:
        device_id = str(agg.camera_name or camera_by_id.get(int(agg.camera_id), {}).get("camera_name") or f"cam_{agg.camera_id}")
        row = ensure_row(device_id, agg.bucket_start)
        queue_count_weight = int(agg.sample_count or 0)
        if agg.is_overall:
            if agg.avg_queue_length_m is not None and queue_count_weight > 0:
                row["_queue_sum"] += float(agg.avg_queue_length_m) * queue_count_weight
                row["_queue_count"] += queue_count_weight
            if agg.max_queue_length_m is not None:
                row["maxQueueLengthM"] = max(float(row["maxQueueLengthM"] or 0.0), float(agg.max_queue_length_m))
            if agg.queue_active_duration_sec is not None:
                row["queueDurationSec"] = float(row["queueDurationSec"] or 0.0) + float(agg.queue_active_duration_sec)
            if agg.max_queue_duration_sec is not None:
                row["maxQueueDurationSec"] = max(float(row["maxQueueDurationSec"] or 0.0), float(agg.max_queue_duration_sec))
        lane_no = int(agg.lane_no) if agg.lane_no is not None else 0
        if lane_no > 0:
            lane = ensure_lane(row, lane_no)
            if agg.avg_queue_length_m is not None and queue_count_weight > 0:
                lane["_queue_sum"] += float(agg.avg_queue_length_m) * queue_count_weight
                lane["_queue_count"] += queue_count_weight
            if agg.max_queue_length_m is not None:
                lane["maxQueueLengthM"] = max(float(lane["maxQueueLengthM"] or 0.0), float(agg.max_queue_length_m))
            if agg.queue_active_duration_sec is not None:
                lane["queueDurationSec"] = float(lane["queueDurationSec"] or 0.0) + float(agg.queue_active_duration_sec)
            if agg.max_queue_duration_sec is not None:
                lane["maxQueueDurationSec"] = max(float(lane["maxQueueDurationSec"] or 0.0), float(agg.max_queue_duration_sec))
            row["laneCount"] = max(int(row["laneCount"] or 0), lane_no)

    selected_device_id = None
    if camera_id is not None:
        selected_device_id = str(camera_by_id.get(int(camera_id), {}).get("camera_name") or f"cam_{camera_id}")
    device_ids = {device_id for device_id, _ in buckets.keys()}
    if selected_device_id:
        device_ids.add(selected_device_id)
    elif not device_ids:
        for meta in camera_by_id.values():
            if meta.get("vd_eligible"):
                device_ids.add(str(meta["camera_name"]))

    current = start
    while current < end_exclusive:
        for device_id in device_ids:
            ensure_row(device_id, current)
        current += timedelta(seconds=BUCKET_SECONDS[bucket_size])

    rows = []
    for row in buckets.values():
        direction = "unknown"
        if row["directionCounts"]:
            ordered = sorted(
                row["directionCounts"].items(),
                key=lambda item: (-int(item[1] or 0), item[0] == "unknown", str(item[0])),
            )
            direction = next((item[0] for item in ordered if item[0] != "unknown"), ordered[0][0])
        row["direction"] = direction
        row["directionText"] = direction_label(direction)
        if row["_speed_weight_count"] > 0:
            row["avgSpeed"] = row["_speed_weight_sum"] / row["_speed_weight_count"]
        if row["_occupancy_count"] > 0:
            row["avgOccupancyPct"] = row["_occupancy_sum_pct"] / row["_occupancy_count"]
        if row["_queue_count"] > 0:
            row["avgQueueLengthM"] = row["_queue_sum"] / row["_queue_count"]
        if row["maxQueueLengthM"] in (0, 0.0):
            row["maxQueueLengthM"] = None
        if row["queueDurationSec"] in (0, 0.0):
            row["queueDurationSec"] = None
        if row["maxQueueDurationSec"] in (0, 0.0):
            row["maxQueueDurationSec"] = None
        for lane_no, lane in list(row["lanes"].items()):
            if lane["_speed_weight_count"] > 0:
                lane["avgSpeed"] = lane["_speed_weight_sum"] / lane["_speed_weight_count"]
            if lane["_occupancy_count"] > 0:
                lane["avgOccupancyPct"] = lane["_occupancy_sum_pct"] / lane["_occupancy_count"]
            if lane["_queue_count"] > 0:
                lane["avgQueueLengthM"] = lane["_queue_sum"] / lane["_queue_count"]
            if lane["maxQueueLengthM"] in (0, 0.0):
                lane["maxQueueLengthM"] = None
            if lane["queueDurationSec"] in (0, 0.0):
                lane["queueDurationSec"] = None
            if lane["maxQueueDurationSec"] in (0, 0.0):
                lane["maxQueueDurationSec"] = None
            for key in list(lane.keys()):
                if key.startswith("_"):
                    lane.pop(key, None)
        row["laneCount"] = max(int(row["laneCount"] or 0), len(row["lanes"]))
        for key in list(row.keys()):
            if key.startswith("_"):
                row.pop(key, None)
        rows.append(row)
    rows.sort(key=lambda item: (-int(item["timeKey"]), str(item["deviceId"])))
    return rows
