#!/usr/bin/env python3
"""資料庫模型"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

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


def init_db():
    os.makedirs("./data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print("✅ 資料庫初始化完成")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
