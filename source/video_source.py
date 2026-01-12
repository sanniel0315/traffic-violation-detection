#!/usr/bin/env python3
"""影像來源模組 - 支援 RTSP 和影片檔案"""
import cv2
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from loguru import logger


class VideoSource(ABC):
    """影像來源抽象類別"""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[any]]:
        pass
    
    @abstractmethod
    def release(self):
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        pass


class RTSPSource(VideoSource):
    """RTSP 串流來源"""
    
    def __init__(self, url: str, reconnect_delay: int = 5):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self._connect()
    
    def _connect(self):
        """連接 RTSP 串流"""
        logger.info(f"連接 RTSP: {self.url}")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        
        if self.cap.isOpened():
            logger.success("RTSP 連接成功")
        else:
            logger.error("RTSP 連接失敗")
    
    def read(self) -> Tuple[bool, Optional[any]]:
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            if not self.cap.isOpened():
                time.sleep(self.reconnect_delay)
                return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("讀取失敗，嘗試重新連接...")
            self.cap.release()
            time.sleep(self.reconnect_delay)
            self._connect()
            return False, None
        
        return True, frame
    
    def release(self):
        if self.cap:
            self.cap.release()
            logger.info("RTSP 連接已關閉")
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


class VideoFileSource(VideoSource):
    """影片檔案來源"""
    
    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        
        if self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.success(f"影片載入: {path}")
            logger.info(f"  解析度: {self.width}x{self.height}")
            logger.info(f"  FPS: {self.fps}, 總幀數: {self.frame_count}")
        else:
            logger.error(f"無法開啟影片: {path}")
    
    def read(self) -> Tuple[bool, Optional[any]]:
        ret, frame = self.cap.read()
        
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        return ret, frame
    
    def release(self):
        if self.cap:
            self.cap.release()
            logger.info("影片來源已關閉")
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def get_progress(self) -> float:
        """取得播放進度 (0-100%)"""
        if self.frame_count > 0:
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return (current / self.frame_count) * 100
        return 0


class VideoSourceFactory:
    """影像來源工廠"""
    
    @staticmethod
    def create(source: str, **kwargs) -> VideoSource:
        """
        根據來源類型建立對應的 VideoSource
        
        Args:
            source: RTSP URL 或影片檔案路徑
        """
        if source.startswith('rtsp://') or source.startswith('http://'):
            return RTSPSource(source, **kwargs)
        else:
            return VideoFileSource(source, **kwargs)


# 測試
if __name__ == '__main__':
    print("=" * 50)
    print("影像來源模組測試")
    print("=" * 50)
    
    # 測試工廠
    print("\n支援的來源類型:")
    print("  - RTSP 串流: rtsp://user:pass@ip:port/stream")
    print("  - 影片檔案: /path/to/video.mp4")
    
    print("\n✅ 影像來源模組準備就緒！")
