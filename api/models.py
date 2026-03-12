#!/usr/bin/env python3
"""資料庫模型"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, text
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


class TrafficEvent(Base):
    __tablename__ = "traffic_events"
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    label = Column(String(50), index=True)
    speed_kmh = Column(Float)
    lane_no = Column(Integer, index=True)
    direction = Column(String(20), index=True)
    entered_zones = Column(JSON)
    bbox = Column(JSON)
    source = Column(String(32), default="roi_detection", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
