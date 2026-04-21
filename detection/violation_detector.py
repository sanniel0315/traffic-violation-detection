#!/usr/bin/env python3
"""違規行為偵測模組"""
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


class ViolationType(Enum):
    """違規類型"""
    RED_LIGHT = ("闖紅燈", 2700, 3)
    SPEEDING = ("超速", 1800, 1)
    WRONG_WAY = ("逆向行駛", 900, 1)
    NO_HELMET = ("未戴安全帽", 500, 0)
    ILLEGAL_PARKING = ("違規停車", 600, 0)
    ILLEGAL_TURN = ("違規轉彎", 600, 1)
    CROSSWALK = ("行人穿越道違規", 1200, 2)
    PHONE_USE = ("使用手機", 3000, 1)
    NO_SEATBELT = ("未繫安全帶", 1500, 0)
    
    @property
    def name_zh(self) -> str:
        return self.value[0]
    
    @property
    def fine(self) -> int:
        return self.value[1]
    
    @property
    def points(self) -> int:
        return self.value[2]


@dataclass
class DetectionZone:
    """偵測區域"""
    name: str
    points: List[Tuple[int, int]]
    zone_type: str  # stop_line, crosswalk, no_parking, sidewalk
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """檢查點是否在區域內"""
        return self._point_in_polygon(point, self.points)
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """射線法判斷點是否在多邊形內"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


@dataclass
class ViolationEvent:
    """違規事件"""
    violation_type: ViolationType
    timestamp: float
    track_id: int
    vehicle_type: str
    license_plate: Optional[str]
    confidence: float
    location: str
    bbox: Dict[str, int]
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'violation_type': self.violation_type.name,
            'violation_name': self.violation_type.name_zh,
            'fine_amount': self.violation_type.fine,
            'points': self.violation_type.points,
            'timestamp': self.timestamp,
            'track_id': self.track_id,
            'vehicle_type': self.vehicle_type,
            'license_plate': self.license_plate,
            'confidence': self.confidence,
            'location': self.location,
            'bbox': self.bbox,
            'image_path': self.image_path,
            'video_path': self.video_path,
            'details': self.details
        }


class VehicleTracker:
    """簡易車輛追蹤器 (IoU-based)"""
    
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}  # track_id -> {'bbox', 'age', 'history'}
        self.next_id = 1
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """更新追蹤，回傳帶有 track_id 的偵測結果"""
        if not detections:
            self._age_tracks()
            return []
        
        results = []
        matched_tracks = set()
        
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = self._calculate_iou(det['bbox'], track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # 更新現有追蹤
                matched_tracks.add(best_track_id)
                self.tracks[best_track_id]['bbox'] = det['bbox']
                self.tracks[best_track_id]['age'] = 0
                self.tracks[best_track_id]['history'].append(self._get_center(det['bbox']))
                det['track_id'] = best_track_id
            else:
                # 新增追蹤
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'age': 0,
                    'history': [self._get_center(det['bbox'])]
                }
                self.next_id += 1
            
            results.append(det)
        
        self._age_tracks(matched_tracks)
        return results
    
    def get_history(self, track_id: int) -> List[Tuple[int, int]]:
        """取得追蹤歷史軌跡"""
        if track_id in self.tracks:
            return self.tracks[track_id]['history']
        return []
    
    def _age_tracks(self, matched: set = None):
        matched = matched or set()
        to_remove = []
        for track_id in self.tracks:
            if track_id not in matched:
                self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_center(self, bbox: Dict) -> Tuple[int, int]:
        return (
            (bbox['x1'] + bbox['x2']) // 2,
            (bbox['y1'] + bbox['y2']) // 2
        )


class ViolationDetector:
    """違規偵測器"""
    
    def __init__(
        self,
        camera_id: int,
        location: str,
        zones: List[DetectionZone] = None,
        speed_limit: int = 50
    ):
        self.camera_id = camera_id
        self.location = location
        self.zones = zones or []
        self.speed_limit = speed_limit
        
        # 追蹤狀態
        self.tracker = VehicleTracker()
        self.violation_cooldown = {}  # (track_id, type) -> timestamp
        self.parking_candidates = {}  # track_id -> first_seen_time
        
        # 紅綠燈狀態
        self.traffic_light = "unknown"  # red, green, unknown
        
        # 設定
        self.cooldown_seconds = 10
        self.parking_threshold = 180  # 3分鐘
    
    def set_zones(self, zones: List[Dict]):
        """設定偵測區域"""
        self.zones = [
            DetectionZone(
                name=z.get('name', ''),
                points=[(p[0], p[1]) for p in z.get('points', [])],
                zone_type=z.get('type', '')
            )
            for z in zones
        ]
    
    def update_traffic_light(self, state: str):
        """更新紅綠燈狀態"""
        self.traffic_light = state
    
    def process(
        self,
        detections: List[Dict],
        frame: np.ndarray,
        timestamp: float = None
    ) -> List[ViolationEvent]:
        """
        處理偵測結果，檢查違規
        
        Args:
            detections: 車輛偵測結果
            frame: 原始影像
            timestamp: 時間戳
            
        Returns:
            違規事件列表
        """
        timestamp = timestamp or time.time()
        violations = []
        
        # 更新追蹤
        tracked = self.tracker.update(detections)
        
        for det in tracked:
            track_id = det['track_id']
            vehicle_type = det['class_name']
            bbox = det['bbox']
            center = ((bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2)
            
            # 檢查各種違規
            
            # 1. 闖紅燈
            if self.traffic_light == "red":
                violation = self._check_red_light(track_id, center, det, timestamp)
                if violation:
                    violations.append(violation)
            
            # 2. 違規停車
            violation = self._check_illegal_parking(track_id, center, det, timestamp)
            if violation:
                violations.append(violation)
            
            # 3. 行駛人行道
            violation = self._check_sidewalk(track_id, center, det, timestamp)
            if violation:
                violations.append(violation)
            
            # 4. 機車未戴安全帽 (需要額外模型，這裡簡化)
            if vehicle_type == 'motorcycle':
                pass  # TODO: 安全帽偵測
        
        return violations
    
    def _check_red_light(self, track_id: int, center: Tuple, det: Dict, timestamp: float) -> Optional[ViolationEvent]:
        """檢查闖紅燈"""
        for zone in self.zones:
            if zone.zone_type == 'stop_line' and zone.contains_point(center):
                if self._check_cooldown(track_id, ViolationType.RED_LIGHT, timestamp):
                    return ViolationEvent(
                        violation_type=ViolationType.RED_LIGHT,
                        timestamp=timestamp,
                        track_id=track_id,
                        vehicle_type=det['class_name'],
                        license_plate=det.get('plate_number'),
                        confidence=det['confidence'],
                        location=self.location,
                        bbox=det['bbox'],
                        details={'zone': zone.name}
                    )
        return None
    
    def _check_illegal_parking(self, track_id: int, center: Tuple, det: Dict, timestamp: float) -> Optional[ViolationEvent]:
        """檢查違規停車"""
        for zone in self.zones:
            if zone.zone_type == 'no_parking' and zone.contains_point(center):
                if track_id not in self.parking_candidates:
                    self.parking_candidates[track_id] = timestamp
                elif timestamp - self.parking_candidates[track_id] > self.parking_threshold:
                    if self._check_cooldown(track_id, ViolationType.ILLEGAL_PARKING, timestamp):
                        return ViolationEvent(
                            violation_type=ViolationType.ILLEGAL_PARKING,
                            timestamp=timestamp,
                            track_id=track_id,
                            vehicle_type=det['class_name'],
                            license_plate=det.get('plate_number'),
                            confidence=det['confidence'],
                            location=self.location,
                            bbox=det['bbox'],
                            details={
                                'zone': zone.name,
                                'duration': timestamp - self.parking_candidates[track_id]
                            }
                        )
        return None
    
    def _check_sidewalk(self, track_id: int, center: Tuple, det: Dict, timestamp: float) -> Optional[ViolationEvent]:
        """檢查行駛人行道"""
        if det['class_name'] not in ['car', 'motorcycle', 'truck', 'bus', 'heavy_truck', 'light_truck']:
            return None
            
        for zone in self.zones:
            if zone.zone_type == 'sidewalk' and zone.contains_point(center):
                if self._check_cooldown(track_id, ViolationType.CROSSWALK, timestamp):
                    return ViolationEvent(
                        violation_type=ViolationType.CROSSWALK,
                        timestamp=timestamp,
                        track_id=track_id,
                        vehicle_type=det['class_name'],
                        license_plate=det.get('plate_number'),
                        confidence=det['confidence'],
                        location=self.location,
                        bbox=det['bbox'],
                        details={'zone': zone.name}
                    )
        return None
    
    def _check_cooldown(self, track_id: int, v_type: ViolationType, timestamp: float) -> bool:
        """檢查冷卻時間，避免重複觸發"""
        key = (track_id, v_type)
        if key in self.violation_cooldown:
            if timestamp - self.violation_cooldown[key] < self.cooldown_seconds:
                return False
        self.violation_cooldown[key] = timestamp
        return True


if __name__ == '__main__':
    # 測試
    detector = ViolationDetector(
        camera_id=1,
        location="測試路口",
        speed_limit=50
    )
    
    # 設定測試區域
    detector.set_zones([
        {
            'name': '停止線',
            'type': 'stop_line',
            'points': [[100, 300], [500, 300], [500, 350], [100, 350]]
        }
    ])
    
    print("✅ 違規偵測器測試完成")
    print(f"支援違規類型: {[v.name_zh for v in ViolationType]}")
