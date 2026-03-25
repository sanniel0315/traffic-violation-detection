#!/usr/bin/env python3
"""壅塞偵測模組"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
from detection.violation_detector import VehicleTracker


class CongestionDetector:
    """壅塞偵測器"""
    
    LEVEL_NAMES = {'low': '暢通', 'medium': '中等', 'high': '擁擠', 'critical': '嚴重壅塞'}
    LEVEL_RANK = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    DEFAULT_DETECT_CONF = 0.12
    DEFAULT_FALLBACK_CONF = 0.05
    VEHICLE_EQUIVALENT_LENGTH_M = {
        'motorcycle': 2.0,
        'car': 6.0,
        'bus': 12.0,
        'truck': 12.0,
    }
    DEFAULT_SAFETY_GAP_M = 1.5

    def __init__(self, vehicle_detector=None):
        if vehicle_detector is None:
            from detection.vehicle_detector import VehicleDetector
            vehicle_detector = VehicleDetector(conf_threshold=self.DEFAULT_DETECT_CONF)
        self.detector = vehicle_detector
        self.fallback_detector = None
        if getattr(self.detector, "conf_threshold", self.DEFAULT_DETECT_CONF) > self.DEFAULT_DETECT_CONF:
            from detection.vehicle_detector import VehicleDetector
            self.detector = VehicleDetector(conf_threshold=self.DEFAULT_DETECT_CONF)
        try:
            from detection.vehicle_detector import VehicleDetector
            self.fallback_detector = VehicleDetector(conf_threshold=self.DEFAULT_FALLBACK_CONF)
        except Exception:
            self.fallback_detector = None
        self.history_map = defaultdict(list)
        self.tracker_map: Dict[str, VehicleTracker] = {}
        self.track_meta_map: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        self.queue_state_map: Dict[str, Dict[str, Any]] = defaultdict(dict)
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
        stop_distance_px = max(4.0, float(params.get("stop_distance_px", 18.0)))
        stop_min_frames = max(2, int(params.get("stop_min_frames", 4)))
        queue_min_vehicles = max(2, int(params.get("queue_min_vehicles", 2)))
        track_hold_frames = max(1, int(params.get("track_hold_frames", 3)))
        safety_gap_m = max(0.0, float(params.get("queue_vehicle_gap_m", self.DEFAULT_SAFETY_GAP_M)))
        queue_activate_score = max(0.0, float(params.get("queue_activate_score", medium_t)))
        now = datetime.now()

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
        if not detections and self.fallback_detector is not None:
            detections = self.fallback_detector.detect(frame)
        vehicles = [d for d in detections if d['class_name'] in ['car', 'motorcycle', 'bus', 'truck']]
        
        if roi_mask is not None:
            vehicles = self._filter_in_roi(vehicles, roi_mask)

        tracker = self.tracker_map.get(camera_key)
        if tracker is None:
            tracker = VehicleTracker(max_age=max(window * 3, 12), iou_threshold=0.15)
            self.tracker_map[camera_key] = tracker
        tracked_vehicles = tracker.update([dict(v) for v in vehicles])
        if not tracked_vehicles:
            tracked_vehicles = self._recover_recent_tracks(camera_key, tracker, max_age_frames=track_hold_frames)
        stopped_track_ids = self._update_track_motion(
            camera_key,
            tracked_vehicles,
            stop_distance_px=stop_distance_px,
            stop_min_frames=stop_min_frames,
        )
        
        vehicle_area = sum(v['bbox']['width'] * v['bbox']['height'] for v in tracked_vehicles)
        occupancy = min(vehicle_area / roi_area, 1.0) if roi_area > 0 else 0

        history = self.history_map[camera_key]
        count_density = self._vehicle_density_score(tracked_vehicles, roi_area)
        stopped_count = sum(1 for v in tracked_vehicles if int(v.get("track_id", 0)) in stopped_track_ids)
        stopped_ratio = (stopped_count / len(tracked_vehicles)) if tracked_vehicles else 0.0
        queue_score = 0.0
        if len(tracked_vehicles) >= queue_min_vehicles:
            queue_score = min(1.0, count_density * (0.45 + (0.55 * stopped_ratio)))
        queue_vehicles = [
            v for v in tracked_vehicles
            if int(v.get("track_id", 0)) in stopped_track_ids
        ]
        queue_active = (
            len(queue_vehicles) >= queue_min_vehicles
            and (queue_score >= queue_activate_score or occupancy >= medium_t or stopped_ratio >= 0.5)
        )
        estimated_queue_length_m = (
            self._estimate_queue_length_m(queue_vehicles, safety_gap_m=safety_gap_m)
            if queue_active else 0.0
        )
        queue_duration_sec = self._update_queue_duration(f"{camera_key}::overall", active=queue_active, now=now)
        congestion_score = max(occupancy, queue_score)
        history.append(congestion_score)
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
        for v in tracked_vehicles:
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
            zvehicles = self._filter_in_roi(tracked_vehicles, zmask)
            z_vehicle_area = sum(v['bbox']['width'] * v['bbox']['height'] for v in zvehicles)
            z_occ_raw = min(z_vehicle_area / zarea, 1.0)
            z_density = self._vehicle_density_score(zvehicles, zarea)
            z_stopped = sum(1 for v in zvehicles if int(v.get("track_id", 0)) in stopped_track_ids)
            z_stopped_ratio = (z_stopped / len(zvehicles)) if zvehicles else 0.0
            z_queue_score = 0.0
            if len(zvehicles) >= queue_min_vehicles:
                z_queue_score = min(1.0, z_density * (0.45 + (0.55 * z_stopped_ratio)))
            z_queue_vehicles = [
                v for v in zvehicles
                if int(v.get("track_id", 0)) in stopped_track_ids
            ]
            z_queue_active = (
                len(z_queue_vehicles) >= queue_min_vehicles
                and (z_queue_score >= queue_activate_score or z_occ_raw >= medium_t or z_stopped_ratio >= 0.5)
            )
            z_queue_length_m = (
                self._estimate_queue_length_m(z_queue_vehicles, safety_gap_m=safety_gap_m)
                if z_queue_active else 0.0
            )
            z_queue_duration_sec = self._update_queue_duration(
                f"{camera_key}::zone::{z.get('name') or idx}",
                active=z_queue_active,
                now=now,
            )
            z_score_raw = max(z_occ_raw, z_queue_score)

            zkey = f"{camera_key}::zone::{z.get('name') or idx}"
            zhist = self.history_map[zkey]
            zhist.append(z_score_raw)
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
                "stopped_vehicle_count": z_stopped,
                "stopped_ratio": round(z_stopped_ratio, 3),
                "raw_occupancy": round(z_occ_raw, 3),
                "raw_score": round(z_score_raw, 3),
                "queue_score": round(z_queue_score, 3),
                "queue_active": z_queue_active,
                "estimated_queue_length_m": round(z_queue_length_m, 1),
                "queue_duration_sec": int(round(z_queue_duration_sec)),
                "occupancy": round(z_occ, 3),
                "level": z_level,
                "level_name": self.LEVEL_NAMES[z_level],
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'vehicle_count': len(tracked_vehicles),
            'stopped_vehicle_count': stopped_count,
            'stopped_ratio': round(stopped_ratio, 3),
            'vehicle_stats': stats,
            'raw_occupancy': round(occupancy, 3),
            'queue_score': round(queue_score, 3),
            'queue_active': queue_active,
            'estimated_queue_length_m': round(estimated_queue_length_m, 1),
            'queue_duration_sec': int(round(queue_duration_sec)),
            'raw_score': round(congestion_score, 3),
            'density_score': round(count_density, 3),
            'occupancy': round(smoothed, 3),
            'level': level,
            'level_name': self.LEVEL_NAMES[level],
            'zone_results': zone_results,
            'vehicles': [{'type': v['class_name'], 'bbox': v['bbox'], 'track_id': v.get('track_id')} for v in tracked_vehicles]
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
                'occupancy': 0, 'level': 'low', 'level_name': '暢通', 'zone_results': [], 'vehicles': [],
                'estimated_queue_length_m': 0.0, 'queue_duration_sec': 0, 'queue_active': False}

    def _vehicle_density_score(self, vehicles: List[Dict[str, Any]], roi_area: int) -> float:
        if roi_area <= 0 or not vehicles:
            return 0.0
        areas = [max(1, int(v['bbox']['width']) * int(v['bbox']['height'])) for v in vehicles if v.get('bbox')]
        if not areas:
            return 0.0
        avg_vehicle_area = max(2500.0, float(sum(areas)) / len(areas))
        estimated_capacity = max(1.0, roi_area / (avg_vehicle_area * 2.2))
        return min(1.0, len(vehicles) / estimated_capacity)

    def _update_track_motion(
        self,
        camera_key: str,
        vehicles: List[Dict[str, Any]],
        *,
        stop_distance_px: float,
        stop_min_frames: int,
    ) -> set[int]:
        meta = self.track_meta_map[camera_key]
        active_ids: set[int] = set()
        stopped_ids: set[int] = set()
        for v in vehicles:
            track_id = int(v.get("track_id") or 0)
            if track_id <= 0:
                continue
            active_ids.add(track_id)
            bbox = v.get("bbox") or {}
            center = (
                int((bbox.get("x1", 0) + bbox.get("x2", 0)) / 2),
                int((bbox.get("y1", 0) + bbox.get("y2", 0)) / 2),
            )
            state = meta.setdefault(track_id, {"history": []})
            history = state.setdefault("history", [])
            state["class_name"] = str(v.get("class_name") or state.get("class_name") or "car")
            state["bbox"] = bbox
            history.append(center)
            if len(history) > max(stop_min_frames * 2, 12):
                del history[:-max(stop_min_frames * 2, 12)]
            if len(history) >= stop_min_frames:
                recent = history[-stop_min_frames:]
                move_dist = self._path_displacement(recent)
                state["stopped"] = move_dist <= stop_distance_px
                if state["stopped"]:
                    stopped_ids.add(track_id)
            else:
                state["stopped"] = False
        for track_id in list(meta.keys()):
            if track_id not in active_ids and track_id not in getattr(self.tracker_map.get(camera_key), "tracks", {}):
                meta.pop(track_id, None)
        return stopped_ids

    def _path_displacement(self, centers: List[tuple[int, int]]) -> float:
        if len(centers) < 2:
            return 0.0
        xs = [p[0] for p in centers]
        ys = [p[1] for p in centers]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        return float((dx * dx + dy * dy) ** 0.5)

    def _recover_recent_tracks(
        self,
        camera_key: str,
        tracker: VehicleTracker,
        *,
        max_age_frames: int,
    ) -> List[Dict[str, Any]]:
        meta = self.track_meta_map.get(camera_key, {})
        recovered: List[Dict[str, Any]] = []
        for track_id, track in tracker.tracks.items():
            age = int(track.get("age", 999))
            if age > max_age_frames:
                continue
            state = meta.get(int(track_id), {})
            bbox = track.get("bbox") or state.get("bbox")
            if not isinstance(bbox, dict):
                continue
            recovered.append({
                "track_id": int(track_id),
                "class_name": str(state.get("class_name") or "car"),
                "confidence": float(state.get("confidence") or 0.0),
                "bbox": bbox,
                "recovered": True,
            })
        return recovered

    def _estimate_queue_length_m(self, vehicles: List[Dict[str, Any]], *, safety_gap_m: float) -> float:
        total = 0.0
        for idx, vehicle in enumerate(sorted(
            vehicles,
            key=lambda item: int((item.get('bbox') or {}).get('y2', 0)),
            reverse=True,
        )):
            class_name = str(vehicle.get("class_name") or "car").lower()
            total += self.VEHICLE_EQUIVALENT_LENGTH_M.get(class_name, self.VEHICLE_EQUIVALENT_LENGTH_M["car"])
            if idx < len(vehicles) - 1:
                total += safety_gap_m
        return total

    def _update_queue_duration(self, state_key: str, *, active: bool, now: datetime) -> float:
        state = self.queue_state_map[state_key]
        if active:
            active_since = state.get("active_since")
            if not isinstance(active_since, datetime):
                state["active_since"] = now
                return 0.0
            return max(0.0, (now - active_since).total_seconds())
        state.pop("active_since", None)
        return 0.0

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
