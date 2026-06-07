#!/usr/bin/env python3
"""逆向行駛 detector (§45)

判定邏輯 — 用「zone 內 dominant flow direction」統計法,不需 user 額外設定:
1. 每個 flow_detection / speed_roi zone 內,持續累積 active vehicles 的速度向量
2. 計算該 zone 的「dominant flow vector」(過去 N 秒 active vehicles 平均 vx, vy)
3. 對每台 vehicle,算其方向跟 dominant 的 cosine similarity
4. cosine < -0.5 (反向 > 120°) 且連續 N frame 都反 → 視為逆向
5. dedup 60 秒同 (cam, track) 不重複

避免雜訊:
- dominant flow 至少需 ≥ MIN_DOMINANT_SAMPLES 台車支持
- vehicle 自己速度 < MIN_SPEED_FOR_CHECK 不算 (停車不算逆向)
- 軌跡 < MIN_TRACK_FRAMES 不算 (剛偵測到還沒穩)

跟 ParkingEvaluator / PedestrianYieldEvaluator 同模式: per camera state。
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


MIN_DOMINANT_SAMPLES = 5         # zone 內 dominant flow 至少需這麼多 vehicle 支持才可信
MIN_SPEED_FOR_CHECK = 5.0        # km/h,vehicle 太慢不檢查 (停止/慢行)
MIN_TRACK_FRAMES = 10            # vehicle track frame 數 < 此不檢查 (剛偵測到不穩)
CONSECUTIVE_WRONG_FRAMES = 5     # 連續這麼多 frame 都反向才觸發
COSINE_WRONG_THRESHOLD = -0.5    # cosine < 此值 = 反向 > 120°
DEDUP_WINDOW_SEC = 60.0
DOMINANT_WINDOW_FRAMES = 60      # dominant flow 用最近這麼多 sample 算


def _point_in_zone(cx: float, cy: float, zone: dict) -> bool:
    pts = zone.get("points") or []
    if len(pts) < 3:
        return False
    poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def _cosine_similarity(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """兩個 2D 向量的 cosine,範圍 [-1, 1]。1 = 同向,-1 = 反向,0 = 垂直。"""
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)


@dataclass
class WrongWayTrigger:
    vehicle_track_id: int
    vehicle_bbox: dict
    vehicle_class: str
    vehicle_speed_kmh: float
    zone_id: str
    zone_name: str
    cosine: float                  # 跟 dominant 的 cos (越接近 -1 越逆向)
    dominant_direction_deg: float  # dominant 方向角 (供 debug)


class WrongWayEvaluator:
    """逆向行駛 evaluator。per camera,跨 frame 累積 dominant flow + 連續反向計數。"""

    VIOLATION_TYPE = "WRONG_WAY"
    VIOLATION_NAME = "逆向行駛"

    def __init__(self, camera_id: int):
        self.camera_id = int(camera_id)
        # zone_id → deque of (vx, vy) recent samples (用來算 dominant)
        self._zone_samples: Dict[str, deque] = {}
        # (zone_id, track_id) → consecutive wrong frame count
        self._wrong_counts: Dict[Tuple[str, int], int] = {}
        # dedup: (cam, track_id) → last_fire_t
        self._fired: Dict[Tuple[int, int], float] = {}

    def _dominant_flow(self, zone_id: str) -> Optional[Tuple[float, float]]:
        """回 zone 的 dominant flow vector,sample 不夠就 None"""
        samples = self._zone_samples.get(zone_id)
        if not samples or len(samples) < MIN_DOMINANT_SAMPLES:
            return None
        sx = sum(s[0] for s in samples) / len(samples)
        sy = sum(s[1] for s in samples) / len(samples)
        # 若 dominant magnitude 太小 (車流靜止/亂流) 不可信
        if math.sqrt(sx * sx + sy * sy) < 1.0:
            return None
        return (sx, sy)

    def feed(
        self,
        vehicles: List[dict],
        tracks: Dict[Any, Dict[str, Any]],
        flow_zones: List[dict],
        now_ts: float,
    ) -> List[WrongWayTrigger]:
        """vehicles: detector 當前 frame 結果,需含 bbox + track_id + class_name + speed_kmh
        tracks: stream.py 的 tracks dict (有 world_xy / kalman / samples 等持續狀態)
        flow_zones: flow_detection 跟 speed_roi 類 zone 清單
        """
        triggered: List[WrongWayTrigger] = []
        if not flow_zones or not vehicles:
            return triggered

        # ---- 1. 累積每 zone dominant flow samples ----
        for v in vehicles:
            tid = v.get("track_id")
            if tid is None:
                continue
            tr = tracks.get(tid) or {}
            frames = int(tr.get("frames", 0) or 0)
            if frames < MIN_TRACK_FRAMES:
                continue
            spd = float(v.get("speed_kmh") or 0.0)
            if spd < MIN_SPEED_FOR_CHECK:
                continue
            # 算 vehicle 速度向量 (從 kalman 或 samples 差分)
            kf = tr.get("kalman")
            if kf is not None and getattr(kf, "vx", None) is not None:
                vx, vy = float(getattr(kf, "vx", 0.0)), float(getattr(kf, "vy", 0.0))
            else:
                # fallback: 用 bbox center 差分粗算
                samples = tr.get("samples") or []
                if len(samples) < 2:
                    continue
                t0, x0, y0 = samples[-2]
                t1, x1, y1 = samples[-1]
                dt = max(1e-3, t1 - t0)
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt

            b = v.get("bbox") or {}
            cx = (float(b.get("x1", 0)) + float(b.get("x2", 0))) / 2
            cy = (float(b.get("y1", 0)) + float(b.get("y2", 0))) / 2
            for z in flow_zones:
                zid = str(z.get("id") or id(z))
                if not _point_in_zone(cx, cy, z):
                    continue
                # 加入 dominant samples
                buf = self._zone_samples.setdefault(zid, deque(maxlen=DOMINANT_WINDOW_FRAMES))
                buf.append((vx, vy))

        # ---- 2. 對每 vehicle 在 zone 內檢查是否逆向 ----
        for v in vehicles:
            tid = v.get("track_id")
            if tid is None:
                continue
            tr = tracks.get(tid) or {}
            frames = int(tr.get("frames", 0) or 0)
            if frames < MIN_TRACK_FRAMES:
                continue
            spd = float(v.get("speed_kmh") or 0.0)
            if spd < MIN_SPEED_FOR_CHECK:
                continue
            kf = tr.get("kalman")
            if kf is not None and getattr(kf, "vx", None) is not None:
                vx, vy = float(getattr(kf, "vx", 0.0)), float(getattr(kf, "vy", 0.0))
            else:
                samples = tr.get("samples") or []
                if len(samples) < 2:
                    continue
                t0, x0, y0 = samples[-2]
                t1, x1, y1 = samples[-1]
                dt = max(1e-3, t1 - t0)
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt

            b = v.get("bbox") or {}
            cx = (float(b.get("x1", 0)) + float(b.get("x2", 0))) / 2
            cy = (float(b.get("y1", 0)) + float(b.get("y2", 0))) / 2

            for z in flow_zones:
                zid = str(z.get("id") or id(z))
                if not _point_in_zone(cx, cy, z):
                    continue
                dom = self._dominant_flow(zid)
                if dom is None:
                    continue
                cos = _cosine_similarity((vx, vy), dom)
                key = (zid, int(tid))
                if cos < COSINE_WRONG_THRESHOLD:
                    self._wrong_counts[key] = self._wrong_counts.get(key, 0) + 1
                else:
                    # 重置 (一次正向就清掉計數,避免長期累積誤觸發)
                    self._wrong_counts.pop(key, None)
                    continue

                if self._wrong_counts[key] < CONSECUTIVE_WRONG_FRAMES:
                    continue
                # dedup
                dedup_key = (self.camera_id, int(tid))
                last_fire = self._fired.get(dedup_key, 0.0)
                if (now_ts - last_fire) < DEDUP_WINDOW_SEC:
                    continue
                self._fired[dedup_key] = now_ts

                dom_deg = math.degrees(math.atan2(dom[1], dom[0]))
                triggered.append(WrongWayTrigger(
                    vehicle_track_id=int(tid),
                    vehicle_bbox=b,
                    vehicle_class=str(v.get("class_name") or ""),
                    vehicle_speed_kmh=spd,
                    zone_id=zid,
                    zone_name=str(z.get("name") or ""),
                    cosine=cos,
                    dominant_direction_deg=dom_deg,
                ))
                # 重置 count 避免下次 frame 又觸發
                self._wrong_counts.pop(key, None)

        return triggered
