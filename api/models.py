#!/usr/bin/env python3
"""資料庫模型"""
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    text,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import hashlib
import secrets

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/violations.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Violation(Base):
    __tablename__ = "violations"
    id = Column(Integer, primary_key=True, index=True)
    violation_type = Column(String(50), index=True)
    violation_name = Column(String(100))
    license_plate = Column(String(20), index=True)
    vehicle_type = Column(String(50))
    location = Column(String(200))
    camera_id = Column(Integer, index=True)
    track_id = Column(Integer)  # 追蹤 ID
    violation_time = Column(DateTime, default=datetime.utcnow, index=True)
    confidence = Column(Float)
    bbox = Column(JSON)
    image_path = Column(String(500))
    video_path = Column(String(500))
    status = Column(String(20), default="pending", index=True)
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    fine_amount = Column(Integer)
    points = Column(Integer)
    speed_kmh = Column(Float)
    speed_limit_kmh = Column(Float)
    overspeed_kmh = Column(Float)
    flow_roi_hit = Column(Boolean, default=False)
    speed_roi_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    source = Column(String(500))
    ip = Column(String(50))
    username = Column(String(50))
    password = Column(String(100))
    port = Column(String(10), default="554")
    stream_path = Column(String(200))
    location = Column(String(200))
    detection_config = Column(JSON, default={})
    zones = Column(JSON)
    status = Column(String(20), default="offline")
    enabled = Column(Boolean, default=True)
    detection_enabled = Column(Boolean, default=True)
    total_violations = Column(Integer, default=0)
    today_violations = Column(Integer, default=0)
    last_seen = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemLog(Base):
    __tablename__ = "system_logs"
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), index=True)
    source = Column(String(50), index=True)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class LPRRecord(Base):
    __tablename__ = "lpr_records"
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    camera_name = Column(String(100))
    plate_number = Column(String(20), index=True)
    confidence = Column(Float)
    valid = Column(Boolean, default=False)
    vehicle_type = Column(String(50))
    snapshot = Column(String(255))
    raw = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class LPRCameraStat(Base):
    __tablename__ = "lpr_camera_stats"
    camera_id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String(100))
    total_frames = Column(Integer, default=0)
    vehicles_detected = Column(Integer, default=0)
    plate_boxes_detected = Column(Integer, default=0)
    ocr_candidates_detected = Column(Integer, default=0)
    vote_candidates_detected = Column(Integer, default=0)
    confirmed_candidates = Column(Integer, default=0)
    committed_candidates = Column(Integer, default=0)
    last_history_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrafficEvent(Base):
    __tablename__ = "traffic_events"
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    label = Column(String(50), index=True)
    speed_kmh = Column(Float)
    occupancy = Column(Float)
    lane_no = Column(Integer, index=True)
    direction = Column(String(20), index=True)
    entered_zones = Column(JSON)
    bbox = Column(JSON)
    source = Column(String(32), default="roi_detection", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class CongestionSample(Base):
    __tablename__ = "congestion_samples"
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    camera_name = Column(String(100), index=True)
    zone_name = Column(String(100), index=True)
    lane_no = Column(Integer, index=True)
    movement = Column(String(20), index=True)
    direction = Column(String(20), index=True)
    is_overall = Column(Boolean, default=False, index=True)
    vehicle_count = Column(Integer, default=0)
    stopped_vehicle_count = Column(Integer, default=0)
    occupancy = Column(Float)
    raw_occupancy = Column(Float)
    queue_score = Column(Float)
    estimated_queue_length_m = Column(Float)
    queue_duration_sec = Column(Float)
    sample_interval_sec = Column(Float)
    queue_active = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class TrafficReportAgg(Base):
    __tablename__ = "traffic_report_aggs"
    __table_args__ = (
        UniqueConstraint("bucket_start", "bucket_size", "camera_id", "direction", "lane_no", name="uq_traffic_report_aggs_bucket"),
        Index("ix_traffic_report_aggs_camera_bucket", "camera_id", "bucket_start"),
    )

    id = Column(Integer, primary_key=True, index=True)
    bucket_start = Column(DateTime, index=True, nullable=False)
    bucket_size = Column(String(8), index=True, nullable=False)
    camera_id = Column(Integer, index=True, nullable=False)
    camera_name = Column(String(100), index=True)
    road_name = Column(String(200))
    direction = Column(String(20), index=True, default="unknown")
    lane_no = Column(Integer, index=True)
    total_flow = Column(Integer, default=0)
    avg_speed = Column(Float)
    max_speed = Column(Float)
    avg_occupancy = Column(Float)
    small_vehicle_flow = Column(Integer, default=0)
    large_vehicle_flow = Column(Integer, default=0)
    other_vehicle_flow = Column(Integer, default=0)
    event_count = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CongestionReportAgg(Base):
    __tablename__ = "congestion_report_aggs"
    __table_args__ = (
        UniqueConstraint("bucket_start", "bucket_size", "camera_id", "zone_name", "lane_no", name="uq_congestion_report_aggs_bucket"),
        Index("ix_congestion_report_aggs_camera_bucket", "camera_id", "bucket_start"),
    )

    id = Column(Integer, primary_key=True, index=True)
    bucket_start = Column(DateTime, index=True, nullable=False)
    bucket_size = Column(String(8), index=True, nullable=False)
    camera_id = Column(Integer, index=True, nullable=False)
    camera_name = Column(String(100), index=True)
    zone_name = Column(String(100), index=True, default="")
    lane_no = Column(Integer, index=True)
    direction = Column(String(20), index=True, default="unknown")
    movement = Column(String(20), index=True, default="")
    is_overall = Column(Boolean, default=False, index=True)
    avg_occupancy = Column(Float)
    max_occupancy = Column(Float)
    avg_vehicle_count = Column(Float)
    avg_stopped_vehicle_count = Column(Float)
    avg_queue_length_m = Column(Float)
    max_queue_length_m = Column(Float)
    queue_active_duration_sec = Column(Float)
    max_queue_duration_sec = Column(Float)
    sample_count = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LprReportAgg(Base):
    __tablename__ = "lpr_report_aggs"
    __table_args__ = (
        UniqueConstraint("bucket_start", "bucket_size", "camera_id", name="uq_lpr_report_aggs_bucket"),
        Index("ix_lpr_report_aggs_camera_bucket", "camera_id", "bucket_start"),
    )

    id = Column(Integer, primary_key=True, index=True)
    bucket_start = Column(DateTime, index=True, nullable=False)
    bucket_size = Column(String(8), index=True, nullable=False)
    camera_id = Column(Integer, index=True, nullable=False)
    camera_name = Column(String(100), index=True)
    total_records = Column(Integer, default=0)
    confirmed_records = Column(Integer, default=0)
    unknown_records = Column(Integer, default=0)
    avg_confidence = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AggregationJobState(Base):
    __tablename__ = "aggregation_job_state"

    job_name = Column(String(64), primary_key=True, index=True)
    last_processed_at = Column(DateTime, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(512), nullable=False)
    role = Column(String(20), default="viewer", index=True)
    enabled = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def hash_password(password: str, salt: str | None = None) -> str:
    s = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), s.encode("utf-8"), 120000
    ).hex()
    return f"{s}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, expect = str(password_hash or "").split("$", 1)
    except ValueError:
        return False
    got = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000
    ).hex()
    return secrets.compare_digest(got, expect)


def init_db():
    os.makedirs("./data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _migrate_violation_columns()
    _migrate_traffic_event_columns()
    _migrate_report_indexes()
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin_user = os.getenv("ADMIN_USERNAME", "admin")
            admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
            db.add(
                User(
                    username=admin_user,
                    password_hash=hash_password(admin_pass),
                    role="admin",
                    enabled=True,
                )
            )
            db.commit()
            print(f"🔐 預設管理者已建立: {admin_user}")
    finally:
        db.close()
    print("✅ 資料庫初始化完成")


def _migrate_violation_columns():
    """輕量欄位遷移：為既有 violations 表補上車速與 ROI 命中欄位。"""
    try:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(violations)")).fetchall()
            col_names = {str(c[1]) for c in cols}
            if "speed_kmh" not in col_names:
                conn.execute(text("ALTER TABLE violations ADD COLUMN speed_kmh FLOAT"))
            if "speed_limit_kmh" not in col_names:
                conn.execute(text("ALTER TABLE violations ADD COLUMN speed_limit_kmh FLOAT"))
            if "overspeed_kmh" not in col_names:
                conn.execute(text("ALTER TABLE violations ADD COLUMN overspeed_kmh FLOAT"))
            if "flow_roi_hit" not in col_names:
                conn.execute(text("ALTER TABLE violations ADD COLUMN flow_roi_hit BOOLEAN DEFAULT 0"))
            if "speed_roi_hit" not in col_names:
                conn.execute(text("ALTER TABLE violations ADD COLUMN speed_roi_hit BOOLEAN DEFAULT 0"))
    except Exception:
        # 非 sqlite 或 migration 失敗時忽略，不阻斷啟動
        pass


def _migrate_traffic_event_columns():
    """為既有 traffic_events 表補上 occupancy 欄位。"""
    try:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(traffic_events)")).fetchall()
            col_names = {str(c[1]) for c in cols}
            if "occupancy" not in col_names:
                conn.execute(text("ALTER TABLE traffic_events ADD COLUMN occupancy FLOAT"))
    except Exception:
        pass


def _migrate_report_indexes():
    """補齊報表查詢用索引。"""
    statements = [
        "CREATE INDEX IF NOT EXISTS ix_traffic_events_camera_created_at ON traffic_events(camera_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_traffic_events_camera_direction_lane_created_at ON traffic_events(camera_id, direction, lane_no, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_congestion_samples_camera_created_at ON congestion_samples(camera_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_congestion_samples_camera_lane_created_at ON congestion_samples(camera_id, lane_no, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_lpr_records_camera_created_at ON lpr_records(camera_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_lpr_records_plate_created_at ON lpr_records(plate_number, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_traffic_report_aggs_bucket_size_bucket_start ON traffic_report_aggs(bucket_size, bucket_start)",
        "CREATE INDEX IF NOT EXISTS ix_congestion_report_aggs_bucket_size_bucket_start ON congestion_report_aggs(bucket_size, bucket_start)",
        "CREATE INDEX IF NOT EXISTS ix_lpr_report_aggs_bucket_size_bucket_start ON lpr_report_aggs(bucket_size, bucket_start)",
    ]
    try:
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
    except Exception:
        pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
