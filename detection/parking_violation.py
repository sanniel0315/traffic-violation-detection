#!/usr/bin/env python3
"""停車類違規通用 evaluator

依 zone 設定（sub_kind / stop_threshold_sec / violation_type）統一判定
「車輛在 zone 內靜止超過 N 秒」→ 觸發違規。共用 SPEEDING 的 plate-vehicle
association + composite + OSD pipeline。

支援違規 (zone.sub_kind):
    general          → ILLEGAL_PARKING       違規停車
    red_line         → RED_LINE_STOP         紅線臨停
    red_line_long    → RED_LINE_PARKING      紅線停車
    yellow_line      → YELLOW_LINE_PARKING   黃線停車
    sidewalk         → SIDEWALK              駕車行駛人行道（進入即觸發）
    sidewalk_parking → SIDEWALK_PARKING      在人行道停車
    bus_stop_stop    → BUS_STOP_STOP         公車招呼站臨停
    bus_stop_parking → BUS_STOP_PARKING      公車招呼站停車
    crosswalk_stop   → CROSSWALK_STOP        停在斑馬線
"""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# sub_kind -> (default_threshold_sec, violation_type, violation_name, fine_amount, points)
# 預設門檻秒數依「道路交通管理處罰條例」法條：
#   §56 ①「停車」= 停止逾 3 分鐘 (180s)；臨停 ≤ 3 分鐘
#   §57 ① 紅線禁停（即使臨停亦違規）/ 黃線允許 ≤ 3 分鐘臨停
#   §74 駕車行駛人行道（瞬時違規，0.5s 過濾抖動）
#   §56 公車招呼站 10m 內禁臨停（瞬時違規）
#   斑馬線禁停（瞬時）
# 預設只是建議，UI 允許 user 個別 zone 覆寫 stop_threshold_sec
DEFAULT_SUB_KIND_CONFIG: Dict[str, Tuple[float, str, str, int, int]] = {
    "general":          (180.0, "ILLEGAL_PARKING",     "違規停車",         600, 0),   # §56 ① 停車 >3min
    "red_line":         (1.0,   "RED_LINE_STOP",       "紅線臨停",         900, 0),   # §57 ① 紅線即使臨停亦違規
    "red_line_long":    (180.0, "RED_LINE_PARKING",    "紅線停車",         1200, 0),  # §57 ① + §56 ①
    "yellow_line":      (180.0, "YELLOW_LINE_PARKING", "黃線停車",         600, 0),   # §57 黃線 >3min 違規
    "sidewalk":         (0.5,   "SIDEWALK",            "駕車行駛人行道",   1200, 1),  # §74 瞬時
    "sidewalk_parking": (60.0,  "SIDEWALK_PARKING",    "在人行道停車",     1200, 1),  # §56 in §74
    "bus_stop_stop":    (1.0,   "BUS_STOP_STOP",       "公車招呼站臨停",   600, 0),   # §56 10m 禁臨停
    "bus_stop_parking": (180.0, "BUS_STOP_PARKING",    "公車招呼站停車",   900, 0),   # §56 ① + 招呼站
    "crosswalk_stop":   (1.0,   "CROSSWALK_STOP",      "停在斑馬線",       900, 1),   # §44 斑馬線禁停
}

# 視為「靜止」的位移速度上限（pixel / sec），低於此值即視為沒在動
STATIC_VELOCITY_PX_PER_SEC = 8.0
# 同 (cam, track, zone, type) 觸發後 cooldown 不再重發，避免 dwell 持續產生多筆
DEDUP_WINDOW_SEC = 600.0


def resolve_zone_config(zone: dict) -> Tuple[float, str, str, int, int]:
    """從 zone 屬性 + sub_kind 預設值決定 (threshold, type, name, fine, points)。
    zone 上的欄位優先：stop_threshold_sec / violation_type / violation_name /
    fine_amount / points。"""
    sub_kind = str(zone.get("sub_kind") or "general")
    base = DEFAULT_SUB_KIND_CONFIG.get(sub_kind, DEFAULT_SUB_KIND_CONFIG["general"])
    threshold = float(zone.get("stop_threshold_sec", base[0]))
    vio_type = str(zone.get("violation_type") or base[1])
    vio_name = str(zone.get("violation_name") or base[2])
    fine = int(zone.get("fine_amount") or base[3])
    pts = int(zone.get("points_penalty") or base[4])
    return threshold, vio_type, vio_name, fine, pts


def _point_in_zone(cx: float, cy: float, zone: dict) -> bool:
    """legacy: 中心點測試 (保留給其他模組相容,evaluator 已改用 _bbox_overlaps_zone)"""
    pts = zone.get("points") or []
    if len(pts) < 3:
        return False
    poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def _bbox_overlaps_zone(bbox: dict, zone: dict) -> bool:
    """車輛 bbox 跟 zone 多邊形「任何重疊」即視為「車輛進入」(車頭/車尾/車身
    任一部分壓到紅線就算)。
    判定: bbox 4 corner 任一在 polygon 內 OR polygon 任一頂點在 bbox 內 OR
          bbox 中心在 polygon 內。涵蓋:
      - 車輛剛進來 (前輪/車頭跨入 zone)
      - 完全在 zone 內 (中心點測試)
      - zone 比 bbox 小,zone 整個被 bbox 包覆 (polygon 點在 bbox 內)
    """
    pts = zone.get("points") or []
    if len(pts) < 3:
        return False
    if not bbox:
        return False
    x1 = float(bbox.get("x1", 0)); y1 = float(bbox.get("y1", 0))
    x2 = float(bbox.get("x2", 0)); y2 = float(bbox.get("y2", 0))
    if x2 <= x1 or y2 <= y1:
        return False
    poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    # 1. bbox 4 corner + bottom-center + center 任一在 polygon 內
    test_points = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2),  # 4 corners
        ((x1 + x2) * 0.5, y2),                    # bottom-center (車輪位置)
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5),       # bbox 中心
    ]
    for px, py in test_points:
        if cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0:
            return True
    # 2. polygon 任一頂點在 bbox 內 (zone 小於 bbox 的 case)
    for pt in pts:
        if len(pt) < 2:
            continue
        px, py = float(pt[0]), float(pt[1])
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


class ParkingEvaluator:
    """跨 frame 累積 per (track, zone) 駐留時間，達閾值即回 trigger spec。
    呼叫端負責拿 trigger 跑 plate association + POST violation。"""

    def __init__(self, camera_id: int):
        self.camera_id = int(camera_id)
        # track_id -> { zone_id -> {enter_t, last_move_t, last_center, last_seen_t} }
        self._dwell: Dict[Any, Dict[str, Dict[str, Any]]] = {}
        # (cam, track_id, zone_id, vio_type) -> last_fire_t
        self._fired: Dict[Tuple[int, Any, str, str], float] = {}

    def _gc(self, active_track_ids: set, stale_zone_keys: set) -> None:
        for tid in list(self._dwell.keys()):
            if tid not in active_track_ids:
                self._dwell.pop(tid, None)
                continue
            for zid in list(self._dwell[tid].keys()):
                if zid in stale_zone_keys:
                    self._dwell[tid].pop(zid, None)
            if not self._dwell[tid]:
                self._dwell.pop(tid, None)

    def feed(
        self,
        vehicles: List[dict],
        now_ts: float,
        parking_zones: List[dict],
    ) -> List[dict]:
        """vehicles: 來自 detector 的 dict list，需含 bbox + track_id。
        parking_zones: zones 過濾後屬於停車類的清單。
        回 [trigger_spec, ...]，每個 spec 描述一個違規候選。"""
        triggered: List[dict] = []
        if not parking_zones or not vehicles:
            self._gc(active_track_ids=set(), stale_zone_keys=set())
            return triggered

        active_tids = set()
        zone_ids = {str(z.get("id") or id(z)) for z in parking_zones}
        for v in vehicles:
            tid = v.get("track_id")
            if tid is None:
                continue
            active_tids.add(tid)
            b = v.get("bbox") or {}
            cx = float((b.get("x1", 0) + b.get("x2", 0)) / 2)
            cy = float((b.get("y1", 0) + b.get("y2", 0)) / 2)
            for z in parking_zones:
                zid = str(z.get("id") or id(z))
                # 「車身任一部分壓 zone」即算進入 (車頭/車尾/車輪 → 紅線停車邏輯)
                # 不是中心點判定 (中心點對紅線停車過嚴,user 反映「車頭碰到就算」)
                inside = _bbox_overlaps_zone(b, z)
                track_zones = self._dwell.setdefault(tid, {})
                rec = track_zones.get(zid)
                if not inside:
                    if rec is not None:
                        track_zones.pop(zid, None)
                    continue
                if rec is None:
                    track_zones[zid] = {
                        "enter_t": now_ts,
                        "last_move_t": now_ts,
                        "last_center": (cx, cy),
                        "last_seen_t": now_ts,
                    }
                    continue
                last_cx, last_cy = rec["last_center"]
                dt = max(1e-3, now_ts - rec["last_seen_t"])
                disp = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
                if (disp / dt) >= STATIC_VELOCITY_PX_PER_SEC:
                    rec["last_move_t"] = now_ts
                rec["last_center"] = (cx, cy)
                rec["last_seen_t"] = now_ts
                dwell = now_ts - rec["last_move_t"]

                threshold, vio_type, vio_name, fine, pts_penalty = resolve_zone_config(z)
                if dwell < threshold:
                    continue
                key = (self.camera_id, tid, zid, vio_type)
                last_fire = self._fired.get(key, 0.0)
                if (now_ts - last_fire) < DEDUP_WINDOW_SEC:
                    continue
                self._fired[key] = now_ts
                triggered.append({
                    "track_id": tid,
                    "vehicle_bbox": b,
                    "vehicle_class": v.get("class_name"),
                    "vehicle_conf": v.get("confidence"),
                    "zone_id": zid,
                    "zone_name": z.get("name") or "",
                    "sub_kind": str(z.get("sub_kind") or "general"),
                    "violation_type": vio_type,
                    "violation_name": vio_name,
                    "fine_amount": fine,
                    "points_penalty": pts_penalty,
                    "dwell_sec": dwell,
                })
        self._gc(active_tids, set())
        return triggered
