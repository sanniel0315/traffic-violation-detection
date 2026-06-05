#!/usr/bin/env python3
"""雷達 track 抽象介面 + Mock driver

雷達到貨前先用 MockRadarDriver 從 YOLO bbox 偽造雷達 detection,
讓 fusion algorithm 跟 UI 可以先開發測試。

實體雷達到貨後，繼承 RadarDriver 寫對應 driver:
- Continental ARS548: CAN / Ethernet UDP
- Smartmicro UMRR-11: CAN
- TI IWR6843: UART/SPI
- 共同輸出: List[RadarDetection]
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math
import random
import time


@dataclass
class RadarDetection:
    """單一雷達 detection (一個 track 在某個 timestamp 的狀態)。
    座標系: BEV (Bird's Eye View) world coord,單位米,attached to cam ground plane。
    """
    track_id: int                   # 雷達 tracker 給的 ID,跨 frame 持續
    x: float                        # BEV X (m),通常 cam 正前方往遠端
    y: float                        # BEV Y (m),通常 cam 左右橫向
    vx: float                       # X 方向速度 (m/s)
    vy: float                       # Y 方向速度 (m/s)
    rcs: float = 0.0                # Radar Cross-Section (dBsm),物體反射強度
    timestamp: float = 0.0          # unix epoch sec
    confidence: float = 1.0         # 雷達 tracker 信心 0-1
    extra: Dict[str, Any] = field(default_factory=dict)  # 額外資訊 (range, azimuth 等)

    @property
    def speed_ms(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)

    @property
    def speed_kmh(self) -> float:
        return self.speed_ms * 3.6


class RadarDriver(ABC):
    """雷達 driver 抽象介面"""

    def __init__(self, camera_id: int):
        self.camera_id = int(camera_id)

    @abstractmethod
    def get_detections(self) -> List[RadarDetection]:
        """非阻塞,回傳當前最新一輪 detection (空 list = 沒新資料)"""
        ...

    def close(self) -> None:
        """釋放資源 (CAN bus close / socket close 等),子類可 override"""
        pass


class MockRadarDriver(RadarDriver):
    """模擬雷達 — 從外部餵 YOLO bbox + camera calibration,偽造 RadarDetection。
    加少量雜訊模擬真實雷達誤差。

    使用方式:
        driver = MockRadarDriver(camera_id=7)
        # detection loop 內把 visual tracks 丟進來:
        driver.feed_visual_tracks([{'track_id': 1, 'world_x': 25.3, 'world_y': -1.2,
                                    'vx': 16.7, 'vy': 0.1, 'class_name': 'car'}, ...])
        radar_dets = driver.get_detections()  # 取最新一輪
    """
    POSITION_NOISE_M = 0.3      # 雷達位置噪聲 σ (m)
    VELOCITY_NOISE_MS = 0.5     # 雷達速度噪聲 σ (m/s)
    DROP_RATE = 0.05            # 5% 機率 drop track (模擬雷達遺漏)

    def __init__(self, camera_id: int, seed: Optional[int] = None):
        super().__init__(camera_id)
        self._latest: List[RadarDetection] = []
        self._rng = random.Random(seed)

    def feed_visual_tracks(self, visual_tracks: List[Dict[str, Any]]) -> None:
        """把當前 frame 的 visual tracks (已含 world coord) 轉成 mock 雷達 detection。
        visual_tracks 每筆要有: track_id, world_x, world_y, vx, vy"""
        now = time.time()
        out: List[RadarDetection] = []
        for vt in visual_tracks or []:
            if self._rng.random() < self.DROP_RATE:
                continue
            wx = float(vt.get("world_x", 0.0))
            wy = float(vt.get("world_y", 0.0))
            vx = float(vt.get("vx", 0.0))
            vy = float(vt.get("vy", 0.0))
            tid = int(vt.get("track_id", 0))
            # 加雜訊
            wx += self._rng.gauss(0.0, self.POSITION_NOISE_M)
            wy += self._rng.gauss(0.0, self.POSITION_NOISE_M)
            vx += self._rng.gauss(0.0, self.VELOCITY_NOISE_MS)
            vy += self._rng.gauss(0.0, self.VELOCITY_NOISE_MS)
            # RCS 依車型粗略給:car ~10, truck ~25, motorcycle ~5
            cls = str(vt.get("class_name", "car")).lower()
            rcs_map = {"car": 10.0, "truck": 25.0, "heavy_truck": 30.0,
                       "light_truck": 20.0, "bus": 28.0, "motorcycle": 5.0}
            rcs = rcs_map.get(cls, 8.0) + self._rng.gauss(0.0, 1.5)
            out.append(RadarDetection(
                track_id=tid,
                x=wx, y=wy, vx=vx, vy=vy,
                rcs=rcs,
                timestamp=now,
                confidence=0.9 + self._rng.uniform(-0.05, 0.05),
                extra={"source": "mock", "visual_track_id": tid},
            ))
        self._latest = out

    def get_detections(self) -> List[RadarDetection]:
        return list(self._latest)
