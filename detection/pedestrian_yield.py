#!/usr/bin/env python3
"""轉彎未禮讓行人 detector (§44 §48)

判定邏輯:
1. crosswalk zone 內有 person tracks (行人在斑馬線上)
2. 同時 vehicle 進入 crosswalk zone 或 zone 周圍 buffer (車輛通過/接近)
3. vehicle 速度 > MIN_VIOLATING_SPEED_KMH (沒減速停讓)
4. vehicle 跟最近 person 距離 < MIN_DISTANCE_TO_PERSON_PX (太近,顯然沒讓)
5. dedup 60 秒同 (cam, vehicle_track, person_track) 不重複

跟 ParkingEvaluator 同模式: per camera 一個 instance,跨 frame 累積判定。
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# 預設違規門檻 (user 可在 zone 設定覆寫)
MIN_VIOLATING_SPEED_KMH = 8.0       # 車速 > 此值算「沒停讓」
MIN_DISTANCE_TO_PERSON_PX = 200     # bbox 中心點 pixel 距離 < 此值算「太近」
ZONE_BUFFER_PX = 30                 # crosswalk zone 周圍 buffer (車進這範圍即算靠近)
DEDUP_WINDOW_SEC = 60.0


def _point_in_zone(cx: float, cy: float, zone: dict) -> bool:
    pts = zone.get("points") or []
    if len(pts) < 3:
        return False
    poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def _bbox_center(b: dict) -> Tuple[float, float]:
    return (
        (float(b.get("x1", 0)) + float(b.get("x2", 0))) / 2,
        (float(b.get("y1", 0)) + float(b.get("y2", 0))) / 2,
    )


def _bbox_distance_to_zone(b: dict, zone_pts: list, buffer_px: float = 0.0) -> float:
    """bbox 中心點到 zone polygon 邊緣最短距離 (pixel)。
    在 zone 內 = 負值 (in_distance = -|d|), 在 zone 外 = 正值。
    buffer_px > 0 時,在 zone 內或距離 < buffer 都返回負值。"""
    cx, cy = _bbox_center(b)
    poly = np.array(zone_pts, dtype=np.float32).reshape(-1, 1, 2)
    # pointPolygonTest measureDist=True 回傳有號距離
    dist = float(cv2.pointPolygonTest(poly, (float(cx), float(cy)), True))
    # dist > 0 在內部, dist < 0 在外部
    return dist


@dataclass
class PedestrianYieldTrigger:
    """違規事件: 車輛 + 行人 + 觸發位置"""
    vehicle_track_id: int
    vehicle_bbox: dict
    vehicle_class: str
    vehicle_speed_kmh: float
    person_track_id: int
    person_bbox: dict
    zone_id: str
    zone_name: str
    distance_px: float


class PedestrianYieldEvaluator:
    """轉彎未禮讓行人 evaluator。
    跟 ParkingEvaluator 同模式,per camera 一個 instance,跨 frame 累積。"""

    VIOLATION_TYPE = "TURN_NOT_YIELD_PEDESTRIAN"
    VIOLATION_NAME = "轉彎未禮讓行人"

    def __init__(self, camera_id: int):
        self.camera_id = int(camera_id)
        # dedup: (cam, vehicle_track, person_track) → last_fire_t
        self._fired: Dict[Tuple[int, int, int], float] = {}

    def feed(
        self,
        vehicles: List[dict],
        persons: List[dict],
        crosswalk_zones: List[dict],
        now_ts: float,
    ) -> List[PedestrianYieldTrigger]:
        """vehicles + persons 是當前 frame YOLO detection,需含 bbox + track_id。
        crosswalk_zones 是過濾後的斑馬線 zone (type=crosswalk)。"""
        triggered: List[PedestrianYieldTrigger] = []
        if not crosswalk_zones or not vehicles or not persons:
            return triggered

        for z in crosswalk_zones:
            zid = str(z.get("id") or id(z))
            zone_pts = z.get("points") or []
            if len(zone_pts) < 3:
                continue

            # 找這個 zone 內的所有 person
            persons_in_zone = []
            for p in persons:
                tid = p.get("track_id")
                if tid is None:
                    continue
                b = p.get("bbox") or {}
                cx, cy = _bbox_center(b)
                # person 用 bbox 底中 (腳的位置) 判定
                foot_y = float(b.get("y2", cy))
                if _point_in_zone(cx, foot_y, z):
                    persons_in_zone.append((int(tid), b, (cx, foot_y)))

            if not persons_in_zone:
                continue  # 此 zone 內沒人,跳過

            # 找有進入 zone 或周圍 buffer 的 vehicle
            for v in vehicles:
                vtid = v.get("track_id")
                if vtid is None:
                    continue
                vbb = v.get("bbox") or {}
                v_dist_to_zone = _bbox_distance_to_zone(vbb, zone_pts)
                # 車輛中心點在 zone 內 (dist > 0) 或在 buffer 內 (dist > -buffer)
                if v_dist_to_zone < -ZONE_BUFFER_PX:
                    continue
                # 車速門檻
                v_speed = float(v.get("speed_kmh") or 0.0)
                if v_speed < MIN_VIOLATING_SPEED_KMH:
                    continue  # 已減速,視為禮讓

                # 找最近 person 距離
                vcx, vcy = _bbox_center(vbb)
                min_d = float("inf")
                closest_person: Optional[Tuple[int, dict, Tuple[float, float]]] = None
                for ptid, pbb, (pcx, pcy) in persons_in_zone:
                    d = math.sqrt((vcx - pcx) ** 2 + (vcy - pcy) ** 2)
                    if d < min_d:
                        min_d = d
                        closest_person = (ptid, pbb, (pcx, pcy))

                if closest_person is None or min_d >= MIN_DISTANCE_TO_PERSON_PX:
                    continue

                # dedup
                ptid, pbb, _ = closest_person
                key = (self.camera_id, int(vtid), int(ptid))
                last_fire = self._fired.get(key, 0.0)
                if (now_ts - last_fire) < DEDUP_WINDOW_SEC:
                    continue
                self._fired[key] = now_ts

                triggered.append(PedestrianYieldTrigger(
                    vehicle_track_id=int(vtid),
                    vehicle_bbox=vbb,
                    vehicle_class=str(v.get("class_name") or ""),
                    vehicle_speed_kmh=v_speed,
                    person_track_id=int(ptid),
                    person_bbox=pbb,
                    zone_id=zid,
                    zone_name=str(z.get("name") or ""),
                    distance_px=min_d,
                ))

        return triggered
