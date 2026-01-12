#!/usr/bin/env python3
"""系統設定"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """應用程式設定"""
    
    # 基本設定
    APP_NAME: str = "交通違規影像分析系統"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API 設定
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # 資料庫
    DATABASE_URL: str = "sqlite:///./data/violations.db"
    
    # AI 模型
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.5
    DEVICE: str = "cuda:0"
    
    # 檔案儲存
    UPLOAD_DIR: str = "./output"
    VIOLATION_IMAGES_DIR: str = "./output/violations"
    RECORDINGS_DIR: str = "./recordings"
    
    # 違規設定
    VIOLATION_COOLDOWN: int = 10  # 秒
    PARKING_THRESHOLD: int = 180  # 秒
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class CameraConfig:
    """攝影機設定"""
    
    def __init__(
        self,
        id: int,
        name: str,
        source: str,
        location: str = "",
        speed_limit: int = 50,
        zones: List[dict] = None,
        enabled: bool = True
    ):
        self.id = id
        self.name = name
        self.source = source
        self.location = location
        self.speed_limit = speed_limit
        self.zones = zones or []
        self.enabled = enabled
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'source': self.source,
            'location': self.location,
            'speed_limit': self.speed_limit,
            'zones': self.zones,
            'enabled': self.enabled
        }


# 預設攝影機設定
DEFAULT_CAMERAS = [
    CameraConfig(
        id=1,
        name="路口攝影機 A",
        source="rtsp://admin:admin@192.168.0.100:554/stream1",
        location="中正路與忠孝路口",
        speed_limit=50,
        zones=[
            {
                'name': '停止線',
                'type': 'stop_line',
                'points': [[100, 300], [540, 300], [540, 350], [100, 350]]
            }
        ]
    ),
    CameraConfig(
        id=2,
        name="測試影片",
        source="./source/test.mp4",
        location="測試路段",
        speed_limit=50
    )
]


# 全域設定實例
settings = Settings()


# 確保目錄存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VIOLATION_IMAGES_DIR, exist_ok=True)
os.makedirs(settings.RECORDINGS_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)
