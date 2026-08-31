#!/usr/bin/env python3
"""壅塞偵測模組"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from detection.violation_detector import VehicleTracker


class CongestionDetector:
    """壅塞偵測器"""
    
    LEVEL_NAMES = {'low': '暢通', 'medium': '中等', 'high': '擁擠', 'critical': '嚴重壅塞'}
    LEVEL_RANK = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    # 大型車 = 大貨車 + 大客車。只有這類車能「一台就把佔用率灌到嚴重壅塞」,
    # 車輛數封頂只針對它們,不誤傷「幾台小客車真的塞住」的情況。
    # 'truck' 也算:細分類沒跑或判不出來時類別會停在 truck,前端顯示就是「大貨車」。
    LARGE_VEHICLE_CLASSES = frozenset({'heavy_truck', 'bus', 'truck'})
    DEFAULT_DETECT_CONF = 0.12
    DEFAULT_FALLBACK_CONF = 0.05
    # 停等長度評估用：每種車輛佔用的等效路面長度（公尺）
    VEHICLE_EQUIVALENT_LENGTH_M = {
        'bicycle':     1.8,
        'motorcycle':  2.0,
        'car':         5.0,    # 小客車
        'non_truck':   5.0,    # YOLO 誤判為 truck 但實為小客車
        'light_truck': 6.0,    # 小貨車
        'truck':       8.0,    # 未細分 truck（保守取中值）
        'bus':         12.0,   # 大客車
        'heavy_truck': 12.0,   # 大貨車
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
        # 固定物候選點:{camera_key: [{"center":(x,y),"first_seen":dt,"last_seen":dt}]}
        # 以「位置」記憶,跨 track 重生累積存在時間(低信心誤判會閃爍、track id 一直換)。
        self.static_spot_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # 上一幀所有偵測中心(跨 track):靜止證據 = 本幀中心跟上一幀某中心幾乎重合。
        self.prev_center_map: Dict[str, List[tuple]] = {}
        # 流量:{camera_key: {"seen": {track_id: 最後出現時間}, "passed": [通過時間, ...]}}
        # 「通過」= 一個 track 出現過又消失。用來區分「車多但在動」與「車多且動不了」。
        self.flow_state_map: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"seen": {}, "passed": []}
        )
        print("✅ 壅塞偵測器初始化完成")

    def reset_camera_state(self, camera_key: str) -> None:
        """清空單一攝影機的所有累積狀態。檔案來源 loop 回開頭時呼叫，
        避免上一輪播放殘留的追蹤 ID / 分數歷史 / queue 持續秒數把新一輪的判定往上灌。"""
        key = str(camera_key)
        self.history_map.pop(key, None)
        self.tracker_map.pop(key, None)
        self.track_meta_map.pop(key, None)
        self.queue_state_map.pop(key, None)
        self.flow_state_map.pop(key, None)
        self.static_spot_map.pop(key, None)
        self.prev_center_map.pop(key, None)
        # zone-level 也清（分數歷史 key 格式是 f"{camera_key}::zone_{idx}" 與 f"{camera_key}::overall"）
        prefix = f"{key}::"
        for sub_key in list(self.history_map.keys()):
            if sub_key.startswith(prefix):
                self.history_map.pop(sub_key, None)
        for sub_key in list(self.queue_state_map.keys()):
            if sub_key.startswith(prefix):
                self.queue_state_map.pop(sub_key, None)
        for sub_key in list(self.flow_state_map.keys()):
            if sub_key.startswith(prefix):
                self.flow_state_map.pop(sub_key, None)

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
        # 【流量納入判級】車在順暢通過就不是壅塞,不管佔用率多高。
        # 0 = 不啟用(舊行為)。只會往下修等級,不會往上升 —— 見 _update_flow_vpm 的說明。
        free_flow_vpm = max(0.0, float(params.get("free_flow_vpm", 0.0)))
        flow_window_sec = max(10.0, float(params.get("flow_window_sec", 60.0)))
        # 【最少車輛數】一台大貨車停在近鏡頭就能吃掉 ROI 六成面積,但一台車不是壅塞。
        # 流量條件擋不住這個(停著的車流量趨近 0,看起來反而更像壅塞),要靠車輛數。
        min_veh_high = max(1, int(params.get("min_vehicles_high", 2)))
        min_veh_critical = max(1, int(params.get("min_vehicles_critical", 3)))
        window = max(1, int(params.get("smoothing_window", 10)))
        # stop_distance_px 18→45 / stop_min_frames 4→3: 原值只認「完全靜止」,抓不到緩行車隊;
        # 放寬後緩行排隊(實測國8 匝道 19m)抓得到,自由車流(>45px/3f)仍排除。
        stop_distance_px = max(4.0, float(params.get("stop_distance_px", 45.0)))
        stop_min_frames = max(2, int(params.get("stop_min_frames", 3)))
        # queue_min_vehicles 維持 2: 降為 1 會讓單一台停/緩行車誤判成排隊(實測疏流段 19/26 幀假陽性)。
        queue_min_vehicles = max(2, int(params.get("queue_min_vehicles", 2)))
        track_hold_frames = max(1, int(params.get("track_hold_frames", 3)))
        safety_gap_m = max(0.0, float(params.get("queue_vehicle_gap_m", self.DEFAULT_SAFETY_GAP_M)))
        queue_activate_score = max(0.0, float(params.get("queue_activate_score", medium_t)))
        # 🛑 排隊佔用率下限:真排隊車擠滿路面,佔用率必然夠高。佔用率極低卻宣告排隊
        #    (實測 1.5% 佔用 + 2 台誤判停滯 → 假排隊 11.5m)是物理不可能,一律擋掉。
        #    真的匝道排隊佔用率遠高於此,不受影響。
        queue_min_occupancy = max(0.0, float(params.get("queue_min_occupancy", 0.05)))
        # 佔用率虛高防呆:車在流動(停車比例 < 此值)時,佔用率不納入壅塞判級,只看排隊。
        # 匝道近鏡頭幾台流動車就吃掉 ROI 40%,但無排隊/停等 ≠ 壅塞(實例 ID3:43%佔用/0m排隊)。
        flowing_stopped_ratio = max(0.0, float(params.get("flowing_stopped_ratio", 0.3)))
        # 固定物抑制：同一個「位置」被幾乎不動(每幀位移<=static_object_px)的偵測連續佔據
        # 超過 static_object_sec → 視為被誤判成車的固定物（實例：cam_3 地上白色轉彎箭頭被
        # 低信心偵測判成 car，靜止數小時、佔用率長灌 1.6%）。紅燈停等車開走後計時歸零
        # 不會累積；門檻要遠小於 stop_distance_px（那是抓緩行排隊用的，45px 太鬆）。
        static_object_sec = max(30.0, float(params.get("static_object_sec", 300.0)))
        static_object_px = max(2.0, float(params.get("static_object_px", 12.0)))
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
        vehicles = [d for d in detections if d['class_name'] in ['car', 'motorcycle', 'bus', 'truck', 'heavy_truck', 'light_truck']]
        # 過濾異常大的 bbox（面積 > 畫面 40% 不可能是車）
        max_area = w * h * 0.4
        vehicles = [v for v in vehicles if v['bbox'].get('width', 0) * v['bbox'].get('height', 0) < max_area]
        # 過濾過小 bbox（避免路標 / 反光鏡誤判成 car）
        # 註：原 8000 太嚴,會把「匝道遠處因透視變小的排隊車」整排濾掉(實測 2296~5917px 都是真車),
        # 降到 2000；三角錐經實測 YOLO 信心度不足、不會被分類成車,不靠面積門檻擋。
        MIN_VEHICLE_AREA = 2000
        vehicles = [v for v in vehicles if v['bbox'].get('width', 0) * v['bbox'].get('height', 0) >= MIN_VEHICLE_AREA]

        if roi_mask is not None:
            vehicles = self._filter_in_roi(vehicles, roi_mask)

        tracker = self.tracker_map.get(camera_key)
        if tracker is None:
            tracker = VehicleTracker(max_age=max(window, 5), iou_threshold=0.15)  # 即時化：原 window*3=30s 太久，改 window=10s
            self.tracker_map[camera_key] = tracker
        tracked_vehicles = tracker.update([dict(v) for v in vehicles])
        if not tracked_vehicles:
            tracked_vehicles = self._recover_recent_tracks(camera_key, tracker, max_age_frames=track_hold_frames)
        stopped_track_ids, static_track_ids = self._update_track_motion(
            camera_key,
            tracked_vehicles,
            stop_distance_px=stop_distance_px,
            stop_min_frames=stop_min_frames,
            static_object_sec=static_object_sec,
            static_object_px=static_object_px,
            now=now,
        )
        if static_track_ids:
            tracked_vehicles = [
                v for v in tracked_vehicles
                if int(v.get("track_id", 0)) not in static_track_ids
            ]
        
        # 佔用率 = 車輛 bbox「聯集 ∩ ROI」/ ROI面積。原本用 bbox 面積「加總」會把重疊區與
        # 超出 ROI 的部分重複計入,近鏡頭大車 2 台就灌到 100%(實測 100%→45%)→ 假性嚴重壅塞。
        if tracked_vehicles:
            _vm = np.zeros((h, w), dtype=np.uint8)
            for _v in tracked_vehicles:
                _b = _v['bbox']
                cv2.rectangle(_vm, (int(_b.get('x1', 0)), int(_b.get('y1', 0))),
                              (int(_b.get('x2', 0)), int(_b.get('y2', 0))), 255, -1)
            _covered = cv2.countNonZero(cv2.bitwise_and(_vm, roi_mask)) if roi_mask is not None else cv2.countNonZero(_vm)
            occupancy = min(_covered / roi_area, 1.0) if roi_area > 0 else 0
        else:
            occupancy = 0.0

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
            and occupancy >= queue_min_occupancy
            and (queue_score >= queue_activate_score or occupancy >= medium_t or stopped_ratio >= 0.5)
        )
        estimated_queue_length_m = (
            self._estimate_queue_length_m(queue_vehicles, safety_gap_m=safety_gap_m)
            if queue_active else 0.0
        )
        queue_duration_sec = self._update_queue_duration(f"{camera_key}::overall", active=queue_active, now=now)
        # 【壅塞判級:加排隊條件(根治佔用率虛高誤報)】占用率只在「有停等」時才納入判級;
        # 車在流動(stopped_ratio < flowing_stopped_ratio)時只看排隊分數。車在流動 ≠ 壅塞。
        occ_for_level = occupancy if stopped_ratio >= flowing_stopped_ratio else 0.0
        congestion_score = max(occ_for_level, queue_score)
        history.append(congestion_score)
        if len(history) > window:
            history.pop(0)
        smoothed = sum(history) / len(history)

        # 流量(輛/分):判級封頂用,見下方【流量 + 車輛數封頂】
        flow_vpm = self._update_flow_vpm(
            camera_key, tracked_vehicles, window_sec=flow_window_sec, now=now
        )

        # 即時 fast-path：raw vehicles 連 2 frame 都 0 車 → 立刻 force level=low
        # 不等 smoothing + tracker max_age，車一清就馬上降級
        _zero_key = f"{camera_key}::raw_zero_count"
        if not hasattr(self, "_raw_zero_streak"):
            self._raw_zero_streak = {}
        if len(vehicles) == 0:
            self._raw_zero_streak[_zero_key] = self._raw_zero_streak.get(_zero_key, 0) + 1
        else:
            self._raw_zero_streak[_zero_key] = 0
        _force_low = self._raw_zero_streak.get(_zero_key, 0) >= 2

        # 全域 level：車輛數 >=2 走原邏輯；單車例外：停著且 occupancy 達 medium 以上仍升 level
        # （單一大車卡住前方 ≠ 「短暫路過」，仍應視為壅塞）
        if _force_low:
            level = 'low'
        elif len(tracked_vehicles) >= 2:
            level = (
                'critical' if smoothed >= critical_t
                else 'high' if smoothed >= high_t
                else 'medium' if smoothed >= medium_t
                else 'low'
            )
        elif len(tracked_vehicles) == 1 and stopped_count >= 1 and smoothed >= medium_t:
            level = (
                'critical' if smoothed >= critical_t
                else 'high' if smoothed >= high_t
                else 'medium'
            )
        else:
            level = 'low'

        large_vehicle_count = sum(
            1 for v in tracked_vehicles
            if v.get('class_name') in self.LARGE_VEHICLE_CLASSES
        )
        level, _cap_reason = self._cap_level(
            level,
            vehicle_count=len(tracked_vehicles),
            large_vehicle_present=large_vehicle_count > 0,
            flow_vpm=flow_vpm,
            min_vehicles_high=min_veh_high,
            min_vehicles_critical=min_veh_critical,
            free_flow_vpm=free_flow_vpm,
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
            # 同上: zone 佔用率改聯集∩zone / zone面積,避免 bbox 加總灌爆
            if zvehicles:
                _zvm = np.zeros((h, w), dtype=np.uint8)
                for _v in zvehicles:
                    _b = _v['bbox']
                    cv2.rectangle(_zvm, (int(_b.get('x1', 0)), int(_b.get('y1', 0))),
                                  (int(_b.get('x2', 0)), int(_b.get('y2', 0))), 255, -1)
                z_occ_raw = min(cv2.countNonZero(cv2.bitwise_and(_zvm, zmask)) / zarea, 1.0)
            else:
                z_occ_raw = 0.0
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
                and z_occ_raw >= queue_min_occupancy
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

            # zone level：車輛數 >=2 走原邏輯；單車例外：停著且 occupancy 達 medium 以上仍升 level
            zveh_n = len(zvehicles)
            if zveh_n >= 2:
                z_level = (
                    'critical' if z_occ >= critical_t
                    else 'high' if z_occ >= high_t
                    else 'medium' if z_occ >= medium_t
                    else 'low'
                )
            elif zveh_n == 1 and z_stopped >= 1 and z_occ >= medium_t:
                z_level = (
                    'critical' if z_occ >= critical_t
                    else 'high' if z_occ >= high_t
                    else 'medium'
                )
            else:
                z_level = 'low'
            # 🛑 車道層要跟整體層套同一組封頂,否則整體被壓成「擁擠」、
            #    車道還在喊「嚴重壅塞」,兩邊講的話不一樣。
            #    (2026-08-31 現場:cam3 車道1 全畫面只有 1 台車、佔用率 64%,
            #     整體判級已被封頂,車道層沒套 → 前端顯示「車道1 嚴重壅塞」。)
            z_large = sum(
                1 for v in zvehicles
                if v.get('class_name') in self.LARGE_VEHICLE_CLASSES
            )
            z_flow = self._update_flow_vpm(
                zkey, zvehicles, window_sec=flow_window_sec, now=now
            )
            z_level, z_cap_reason = self._cap_level(
                z_level,
                vehicle_count=zveh_n,
                large_vehicle_present=z_large > 0,
                flow_vpm=z_flow,
                min_vehicles_high=min_veh_high,
                min_vehicles_critical=min_veh_critical,
                free_flow_vpm=free_flow_vpm,
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
                "flow_vpm": round(z_flow, 1),
                "large_vehicle_count": z_large,
                "level_capped_by": z_cap_reason,
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
            'flow_vpm': round(flow_vpm, 1),
            'large_vehicle_count': large_vehicle_count,
            'level_capped_by': _cap_reason,
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

    def _cap_level(
        self,
        level: str,
        *,
        vehicle_count: int,
        large_vehicle_present: bool,
        flow_vpm: float,
        min_vehicles_high: int,
        min_vehicles_critical: int,
        free_flow_vpm: float,
    ) -> tuple[str, Optional[str]]:
        """把等級往下修到合理上限。回傳 (等級, 封頂原因或 None)。

        🛑 只往下修,不往上升。兩個理由分別擋掉兩種誤報:

        1. `vehicle_count` —— 一台大貨車停在近鏡頭就能吃掉 ROI 六成面積,
           但一台車不是壅塞。流量條件擋不住這個(停著的車流量趨近 0,
           看起來反而更像壅塞),只能靠車輛數。
           「單一大車卡住前方」仍看得到,只是封頂在「擁擠」不喊最高級。
           🛑 只在畫面上有大貨車/大客車時才套用 —— 會「一台就灌爆佔用率」的
              只有大型車;幾台小客車就把 ROI 塞到 60% 那是真的擠,不該被壓下來。
        2. `free_flow` —— 車正在順暢通過,佔用率再高也不是壅塞。
           流量會低估(見 _update_flow_vpm),所以只准降級不准升級。
        """
        ceiling: Optional[str] = None
        reason: Optional[str] = None
        if large_vehicle_present:
            if vehicle_count < min_vehicles_high:
                ceiling, reason = 'medium', 'vehicle_count'
            elif vehicle_count < min_vehicles_critical:
                ceiling, reason = 'high', 'vehicle_count'
        if free_flow_vpm > 0 and flow_vpm >= free_flow_vpm:
            if ceiling is None or self.LEVEL_RANK['medium'] < self.LEVEL_RANK[ceiling]:
                ceiling, reason = 'medium', 'free_flow'
        if ceiling is not None and self.LEVEL_RANK[level] > self.LEVEL_RANK[ceiling]:
            return ceiling, reason
        return level, None

    def _update_flow_vpm(
        self,
        camera_key: str,
        vehicles: List[Dict[str, Any]],
        *,
        window_sec: float = 60.0,
        now: Optional[datetime] = None,
    ) -> float:
        """回傳這台相機最近 window_sec 秒的通過流量(輛/分)。

        「通過」= 一個 track_id 出現過、之後從畫面消失。停著不走的車 track 一直在,
        不會被計入 —— 這正是要的:流量衡量的是「車走掉的速率」,不是「車有多少」。

        🛑 這個數字會低估。分析率只有 0.8 fps(2026-08-31 87 實測),快車可能兩幀
           之間就穿過 ROI,track 接不起來就當成沒出現過。所以它只能用來
           「往下修」判級(流量高 → 一定不是壅塞),絕對不能拿來往上升級 ——
           低估的流量會看起來像壅塞,那是錯的方向。
        """
        now = now or datetime.now()
        st = self.flow_state_map[camera_key]
        seen: Dict[int, datetime] = st["seen"]
        passed: List[datetime] = st["passed"]

        cur_ids = {int(v.get("track_id", 0)) for v in vehicles if v.get("track_id") is not None}
        # 消失的 track = 通過一台
        for tid in list(seen.keys()):
            if tid not in cur_ids:
                passed.append(seen.pop(tid))
        for tid in cur_ids:
            seen[tid] = now

        cutoff = now - timedelta(seconds=window_sec)
        st["passed"] = [t for t in passed if t >= cutoff]
        return len(st["passed"]) * (60.0 / window_sec) if window_sec > 0 else 0.0

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
        static_object_sec: float = 300.0,
        static_object_px: float = 12.0,
        now: Optional[datetime] = None,
    ) -> tuple[set[int], set[int]]:
        """回傳 (停等中的 track ids, 固定物 track ids)。

        固定物 = 同一個「位置」被幾乎不動的偵測連續佔據 static_object_sec 以上
        （地上標線/反光鏡等被誤判成車的靜態物）。存在時間記在 static_spot_map
        的「固定點」上而非 track 上：低信心誤判會閃爍、track id 一直換
        （實測 cam_3 箭頭 6 分鐘內 id 1→21），掛在 track 上永遠累積不到門檻。
        固定點超過 _SPOT_GAP_SEC 沒被靜態偵測命中就重置——紅燈停等車開走後
        （綠燈期間該點空著）計時歸零，不會跨紅燈週期累積到誤殺停止線車。
        """
        _SPOT_GAP_SEC = 30.0
        now = now or datetime.now()
        meta = self.track_meta_map[camera_key]
        spots = self.static_spot_map[camera_key]
        # 過期固定點直接丟(等同計時歸零)
        spots[:] = [s for s in spots if (now - s["last_seen"]).total_seconds() <= _SPOT_GAP_SEC]
        prev_frame_centers = self.prev_center_map.get(camera_key) or []
        curr_frame_centers: List[tuple] = []
        frame_boxes: List[tuple] = []  # (bbox, 是否停止中) — 給固定點遮擋保命用
        active_ids: set[int] = set()
        stopped_ids: set[int] = set()
        static_ids: set[int] = set()
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
            curr_frame_centers.append(center)
            # 靜止證據 = 本幀中心跟「上一幀任一偵測中心」幾乎重合(跨 track 比對)。
            # 不能用 track 生涯位移——真車開過標線時 tracker 會把標線 track 短暫
            # 接到車上(IoU 匹配),位移史被污染後永遠不再是靜態候選(實測 87 抑制會退開);
            # 也不能只看同 track 的上一幀——閃爍重生的新 track 第一幀沒有上一幀。
            _still = any(
                ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5 <= static_object_px
                for px, py in prev_frame_centers
            )
            spot = None
            for s in spots:
                sx, sy = s["center"]
                if ((center[0] - sx) ** 2 + (center[1] - sy) ** 2) ** 0.5 <= static_object_px:
                    spot = s
                    break
            if _still:
                if spot is None:
                    spot = {"center": center, "first_seen": now, "last_seen": now}
                    spots.append(spot)
                spot["last_seen"] = now
            if spot is not None and (now - spot["first_seen"]).total_seconds() >= static_object_sec:
                static_ids.add(track_id)
                state["stopped"] = False
                frame_boxes.append((bbox, True))
                continue
            if len(history) >= stop_min_frames:
                recent = history[-stop_min_frames:]
                move_dist = self._path_displacement(recent)
                state["stopped"] = move_dist <= stop_distance_px
                if state["stopped"]:
                    stopped_ids.add(track_id)
            else:
                state["stopped"] = False
            frame_boxes.append((bbox, bool(state["stopped"])))
        # 固定點遮擋保命:紅燈時真車會停在標線上蓋住它(標線偵測消失>30秒→固定點
        # 過期歸零→車開走後又要重新累積 5 分鐘)。本幀沒被命中的固定點,若中心被
        # 「停止中」車輛的 bbox 蓋住,視為被遮擋而非消失,last_seen 續命(年齡照累積)。
        # 只認停止中的車:綠燈流動車輛掃過停止線點不續命,停等熱點在綠燈期間照樣過期。
        for s in spots:
            if s["last_seen"] is now:
                continue
            sx, sy = s["center"]
            for _b, _stopped in frame_boxes:
                if _stopped and _b.get("x1", 0) <= sx <= _b.get("x2", 0) and _b.get("y1", 0) <= sy <= _b.get("y2", 0):
                    s["last_seen"] = now
                    break
        for track_id in list(meta.keys()):
            if track_id not in active_ids and track_id not in getattr(self.tracker_map.get(camera_key), "tracks", {}):
                meta.pop(track_id, None)
        self.prev_center_map[camera_key] = curr_frame_centers
        return stopped_ids, static_ids

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
