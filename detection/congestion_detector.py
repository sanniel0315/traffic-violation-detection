#!/usr/bin/env python3
"""壅塞偵測模組"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class CongestionDetector:
    """壅塞偵測器"""
    
    LEVEL_NAMES = {'low': '暢通', 'medium': '中等', 'high': '擁擠', 'critical': '嚴重壅塞'}
    LEVEL_RANK = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}

    def __init__(self, vehicle_detector=None):
        if vehicle_detector is None:
            from detection.vehicle_detector import VehicleDetector
            vehicle_detector = VehicleDetector()
        self.detector = vehicle_detector
        self.history_map = defaultdict(list)
        print("✅ 壅塞偵測器初始化完成")

    def analyze(
        self,
        frame: np.ndarray,
        zones: Optional[List[Dict]] = None,
        camera_key: str = "default",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """分析壅塞程度"""
        if frame is None or frame.size == 0:
            return self._empty_result()
        params = params or {}
        medium_t = float(params.get("medium_threshold", 0.2))
        high_t = float(params.get("high_threshold", 0.4))
        critical_t = float(params.get("critical_threshold", 0.6))
        window = max(1, int(params.get("smoothing_window", 10)))

        h, w = frame.shape[:2]
        roi_mask = None
        roi_area = w * h
        det_zones = []
        
        if zones:
            det_zones = [z for z in zones if z.get('type') in ('detection', 'flow_detection')]
            if det_zones:
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                for z in det_zones:
                    pts = self._zone_points(z, w, h)
                    if len(pts) >= 3:
                        cv2.fillPoly(roi_mask, [np.array(pts, np.int32)], 255)
                roi_area = cv2.countNonZero(roi_mask)
                if roi_area == 0:
                    roi_mask = None
                    roi_area = w * h
        
        detections = self.detector.detect(frame)
        vehicles = [d for d in detections if d['class_name'] in ['car', 'motorcycle', 'bus', 'truck']]
        
        if roi_mask is not None:
            vehicles = self._filter_in_roi(vehicles, roi_mask)
        
        vehicle_area = sum(v['bbox']['width'] * v['bbox']['height'] for v in vehicles)
        occupancy = min(vehicle_area / roi_area, 1.0) if roi_area > 0 else 0

        history = self.history_map[camera_key]
        history.append(occupancy)
        if len(history) > window:
            history.pop(0)
        smoothed = sum(history) / len(history)

        level = (
            'critical' if smoothed >= critical_t
            else 'high' if smoothed >= high_t
            else 'medium' if smoothed >= medium_t
            else 'low'
        )
        
        stats = {}
        for v in vehicles:
            t = v['class_name']
            stats[t] = stats.get(t, 0) + 1

        zone_results = []
        for idx, z in enumerate(det_zones):
            pts = self._zone_points(z, w, h)
            if len(pts) < 3:
                continue
            zmask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(zmask, [np.array(pts, np.int32)], 255)
            zarea = cv2.countNonZero(zmask)
            if zarea <= 0:
                continue
            zvehicles = self._filter_in_roi(vehicles, zmask)
            z_vehicle_area = sum(v['bbox']['width'] * v['bbox']['height'] for v in zvehicles)
            z_occ_raw = min(z_vehicle_area / zarea, 1.0)

            zkey = f"{camera_key}::zone::{z.get('name') or idx}"
            zhist = self.history_map[zkey]
            zhist.append(z_occ_raw)
            if len(zhist) > window:
                zhist.pop(0)
            z_occ = sum(zhist) / len(zhist)

            z_level = (
                'critical' if z_occ >= critical_t
                else 'high' if z_occ >= high_t
                else 'medium' if z_occ >= medium_t
                else 'low'
            )
            lane_no = self._parse_lane_no(z)
            movement = self._normalize_movement(z.get("lane"), z.get("type"))
            lane_tags = z.get("lane_tags") if isinstance(z.get("lane_tags"), list) else []
            if not movement and lane_tags:
                movement = self._normalize_movement(lane_tags[0], "")
            direction = str(z.get("direction") or "").strip()
            zone_results.append({
                "name": z.get("name") or f"區域{idx+1}",
                "type": z.get("type", "detection"),
                "lane_no": lane_no,
                "movement": movement,
                "direction": direction,
                "vehicle_count": len(zvehicles),
                "raw_occupancy": round(z_occ_raw, 3),
                "occupancy": round(z_occ, 3),
                "level": z_level,
                "level_name": self.LEVEL_NAMES[z_level],
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'vehicle_count': len(vehicles),
            'vehicle_stats': stats,
            'raw_occupancy': round(occupancy, 3),
            'occupancy': round(smoothed, 3),
            'level': level,
            'level_name': self.LEVEL_NAMES[level],
            'zone_results': zone_results,
            'vehicles': [{'type': v['class_name'], 'bbox': v['bbox']} for v in vehicles]
        }

    def _parse_coordinates(self, coords: str) -> List[tuple]:
        try:
            nums = [int(x) for x in coords.split(',')]
            return [(nums[i], nums[i+1]) for i in range(0, len(nums)-1, 2)]
        except:
            return []

    def _zone_points(self, zone: Dict[str, Any], frame_w: int, frame_h: int) -> List[tuple]:
        points = zone.get("points")
        if isinstance(points, list) and len(points) >= 3:
            src_w = zone.get("source_width") or frame_w
            src_h = zone.get("source_height") or frame_h
            coord_space = zone.get("coord_space", "")
            out = []
            for p in points:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x, y = float(p[0]), float(p[1])
                if coord_space == "natural" and src_w and src_h:
                    x = x * frame_w / src_w
                    y = y * frame_h / src_h
                out.append((int(round(x)), int(round(y))))
            return out

        coords = zone.get("coordinates", "")
        if coords:
            return self._parse_coordinates(coords)
        return []

    def _filter_in_roi(self, vehicles: List, roi_mask: np.ndarray) -> List:
        filtered = []
        for v in vehicles:
            bbox = v['bbox']
            cx, cy = (bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2
            if 0 <= cy < roi_mask.shape[0] and 0 <= cx < roi_mask.shape[1]:
                if roi_mask[cy, cx] > 0:
                    filtered.append(v)
        return filtered

    def _empty_result(self) -> Dict:
        return {'timestamp': datetime.now().isoformat(), 'vehicle_count': 0, 'vehicle_stats': {}, 
                'occupancy': 0, 'level': 'low', 'level_name': '暢通', 'zone_results': [], 'vehicles': []}

    def _normalize_movement(self, lane: Any, zone_type: str = "") -> str:
        raw = str(lane or "").strip().lower()
        if raw in ("left", "lane_left"):
            return "left"
        if raw in ("middle", "straight", "lane_straight"):
            return "middle"
        if raw in ("right", "lane_right"):
            return "right"
        zt = str(zone_type or "").strip().lower()
        if zt == "lane_left":
            return "left"
        if zt == "lane_straight":
            return "middle"
        if zt == "lane_right":
            return "right"
        return ""

    def _parse_lane_no(self, zone: Dict[str, Any]) -> Optional[int]:
        candidates = [zone.get("lane_no"), zone.get("lane_id"), zone.get("laneNo"), zone.get("lane"), zone.get("name")]
        for raw in candidates:
            if raw in (None, ""):
                continue
            m = str(raw).strip()
            if not m:
                continue
            digits = "".join(ch for ch in m if ch.isdigit())
            if not digits:
                continue
            n = int(digits)
            if n > 0:
                return n
        return None
