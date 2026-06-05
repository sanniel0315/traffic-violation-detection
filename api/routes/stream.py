#!/usr/bin/env python3
"""串流與偵測服務 API"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import cv2
import asyncio
import os
import threading
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except Exception:
    PILImage = None
    ImageDraw = None
    ImageFont = None
    _PIL_AVAILABLE = False

from api.models import get_db, Camera, SessionLocal, TrafficEvent
from api.routes.logs import add_log
from api.utils.roi_scope import SCOPE_TRAFFIC, SCOPE_SPEED, SCOPE_CONGESTION, select_zones
from api.utils.feature_state import get_feature_enabled, set_feature_state
from api.utils.camera_stream import resolve_analysis_source, resolve_capture_source, resolve_local_api_source
from api.utils.shutdown import shutdown_event

router = APIRouter(prefix="/api/stream", tags=["串流"])

# 偵測服務狀態
detection_services: Dict[int, dict] = {}
# detection 服務共享最新 frame 給 overlay（避免 NX 串流開第二條連線）
_shared_frames: Dict[int, dict] = {}  # {camera_id: {"frame": ndarray, "ts": float}}

# ---- 違規 4 frame ring buffer (方案 C 100% 命中) ----
# 違規觸發時直接從 ring 撈 t-2s frame + 觸發當下 + timer 抓未來 t+0.5/t+2s
# 取代 frigate clip + ffmpeg seek (容易 lost frame)
import threading as _vbuf_threading
from collections import deque as _vbuf_deque
_violation_frame_ring: Dict[int, _vbuf_deque] = {}
_VIOLATION_RING_MAXLEN = 30  # 30 frame @ ~10fps push = 3 秒 buffer (覆蓋 t-2s)
_VIOLATION_PUSH_INTERVAL_SEC = 0.1  # 限 10 fps push 避免 IO 過量
_violation_ring_lock = _vbuf_threading.Lock()
_violation_last_push: Dict[int, float] = {}

# 停車類違規 evaluator (per camera 一個 instance,跨 frame 保 dwell 狀態)
_PARKING_EVALUATORS: Dict[int, "object"] = {}

# 視覺 track snapshot (per camera, BEV world coord) — 給 sensor_fusion router 拉。
# 每 frame post-process 結尾覆寫,sensor_fusion API 即時讀。
# 結構: [{track_id, world_x, world_y, vx, vy, class_name, confidence, bbox, timestamp}]
_VEHICLE_TRACK_SNAPSHOTS: Dict[int, list] = {}


def _push_violation_ring(camera_id: int, frame):
    """worker tick 呼叫，每 ~0.1s push 一張原解析度 frame 進 ring buffer。
    保留原 1920x1080 (不縮)，避免 composite 兩次 downscale 模糊。
    RAM cost: 30 frame × 6MB ≈ 180MB / cam × 4 cam = 720MB (Jetson 64GB OK)。"""
    import time as _t
    now = _t.time()
    last = _violation_last_push.get(camera_id, 0.0)
    if now - last < _VIOLATION_PUSH_INTERVAL_SEC:
        return
    _violation_last_push[camera_id] = now
    try:
        with _violation_ring_lock:
            ring = _violation_frame_ring.get(camera_id)
            if ring is None:
                ring = _vbuf_deque(maxlen=_VIOLATION_RING_MAXLEN)
                _violation_frame_ring[camera_id] = ring
            ring.append((now, frame.copy()))
    except Exception:
        pass


def _associate_plate_for_vehicle(frame, vehicle_bbox: dict, camera_id: int):
    """對指定 vehicle bbox 跑 plate detect + OCR，回 (plate_number, plate_crop_bgr)。
    嚴格綁定觸發車輛（不走 _latest_lpr_plate ±20s 避免抓到前車）。
    OCR 走 tightened crop（準確）、save 走 expanded crop（composite 上下不被切）。
    返回 ("", None) 表示沒抓到 conf>=0.5 的 plate。"""
    if not vehicle_bbox:
        return "", None
    try:
        bx1 = max(0, int(vehicle_bbox.get('x1', 0)))
        by1 = max(0, int(vehicle_bbox.get('y1', 0)))
        bx2 = int(vehicle_bbox.get('x2', 0))
        by2 = int(vehicle_bbox.get('y2', 0))
        veh_crop = frame[by1:by2, bx1:bx2]
        if veh_crop is None or veh_crop.size == 0:
            return "", None
        from api.routes.lpr_stream import (
            get_plate_detector as _vpgd,
            get_recognizer as _vpr,
            _PLATE_DETECT_CONF as _vpconf,
            _expand_plate_bbox as _vpex,
            _tighten_plate_crop_with_bbox as _vptight,
            _rank_plate_candidates as _vprank,
            _propose_plate_bboxes as _vpfb,
            _recognize_plate_on_crop as _vpo,
        )
        det = _vpgd()
        rec = _vpr()
        vh, vw = veh_crop.shape[:2]
        detections = det.detect(veh_crop, conf=_vpconf)
        if not detections:
            detections = _vpfb(veh_crop)
        ranked = _vprank(detections, vw, vh)
        for r in ranked:
            rb = r.get('bbox') or []
            if len(rb) != 4:
                continue
            ex1, ey1, ex2, ey2 = _vpex([int(x) for x in rb], vw, vh)
            pc = det.crop(veh_crop, [ex1, ey1, ex2, ey2])
            if pc is None or getattr(pc, 'size', 0) == 0:
                continue
            pc_save = pc.copy()
            pc_tight, _ = _vptight(pc)
            pc_ocr = pc_tight if (pc_tight is not None and getattr(pc_tight, 'size', 0) > 0) else pc
            res = _vpo(pc_ocr, rec)
            pn = (res or {}).get('plate_number')
            conf = float((res or {}).get('confidence') or 0.0)
            if pn and conf >= 0.5:
                return str(pn).strip(), pc_save
        return "", None
    except Exception as e:
        print(f"⚠️ plate-vehicle association err cam{camera_id}: {e}", flush=True)
        return "", None


def _emit_violation_for_vehicle(
    *,
    camera_id: int,
    location: str,
    frame,
    vehicle_bbox: dict,
    vehicle_class: str,
    vehicle_conf,
    violation_type: str,
    violation_name: str,
    fine_amount: int,
    points: int = 0,
    output_dir,
    extra_fields: Optional[dict] = None,
    trigger_ts: Optional[float] = None,
):
    """通用違規 emitter：plate-vehicle association → 標 annotated.jpg → POST →
    ring buffer 4-frame save + plate.png 寫盤。回 (violation_id, plate_number)。
    所有違規種類共用此 path，等同 SPEEDING pipeline。"""
    import time as _t
    plate, plate_crop = _associate_plate_for_vehicle(frame, vehicle_bbox, camera_id)
    annotated = frame.copy()
    if vehicle_bbox:
        cv2.rectangle(
            annotated,
            (int(vehicle_bbox.get('x1', 0)), int(vehicle_bbox.get('y1', 0))),
            (int(vehicle_bbox.get('x2', 0)), int(vehicle_bbox.get('y2', 0))),
            (0, 255, 0), 2,
        )
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    image_name = f"{violation_type}_{ts_str}.jpg"
    image_path = output_dir / image_name
    cv2.imwrite(str(image_path), annotated)
    data = {
        "violation_type": violation_type,
        "violation_name": violation_name,
        "vehicle_type": vehicle_class,
        "license_plate": plate or None,
        "location": location,
        "camera_id": camera_id,
        "confidence": vehicle_conf,
        "fine_amount": fine_amount,
        "points": points,
        "image_path": f"/files/violations/{image_name}",
        "bbox": vehicle_bbox,
    }
    if extra_fields:
        data.update(extra_fields)
    vid = None
    try:
        resp = requests.post("http://localhost:8000/api/violations", json=data, timeout=5)
        vid = (resp.json() or {}).get("id") if resp.ok else None
    except Exception as e:
        print(f"⚠️ emit {violation_type} POST fail cam{camera_id}: {e}", flush=True)
        return None, plate
    if not vid:
        return None, plate
    now_t = trigger_ts if trigger_ts is not None else _t.time()
    try:
        _save_violation_4frames_async(
            camera_id, int(vid), now_t, frame,
            vehicle_bbox=vehicle_bbox, plate_text=plate,
        )
    except Exception:
        pass
    if plate_crop is not None and plate_crop.size > 0:
        try:
            from pathlib import Path as _PP
            vp_dir = _PP("./output/violations/snapshots").resolve()
            vp_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(vp_dir / f"{int(vid)}_violation_plate.png"),
                plate_crop,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
        except Exception:
            pass
    return int(vid), plate


def _save_violation_4frames_async(camera_id: int, violation_id: int, trigger_ts: float,
                                   current_frame, vehicle_bbox=None, plate_text: str = ""):
    """違規觸發後拿到 id 立刻 save 4 frame (before/mid_a/mid_b/after)。
    - mid_a: 觸發當下 frame + 綠色 vehicle bbox + plate 標記 (讓 user 看出哪輛是違規車)
    - before: 從 ring 撈 trigger_ts - 2.0s 最接近的 frame (不標 bbox，車輛位置已不同)
    - mid_b / after: schedule threading.Timer 在 0.6s / 2.1s 後從 ring 撈最新 frame (不標)"""
    import time as _t
    import cv2 as _cv2_local
    from pathlib import Path as _Path
    out_dir = _Path("./output/violations/snapshots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(tag: str, img):
        if img is None:
            return
        try:
            _cv2_local.imwrite(
                str(out_dir / f"{violation_id}_{tag}.jpg"),
                img,
                [_cv2_local.IMWRITE_JPEG_QUALITY, 92],
            )
        except Exception:
            pass

    # mid_a: 當下 frame 原圖 (不畫綠框 / 不標 label，user 要原本風格)
    # 車輛識別走 composite 左上 plate crop overlay，不在 frame 上 draw
    _write("mid_a", current_frame)

    # before: 從 ring 撈 t-2s 最接近
    with _violation_ring_lock:
        ring_snapshot = list(_violation_frame_ring.get(camera_id, []))
    if ring_snapshot:
        target = trigger_ts - 2.0
        best = min(ring_snapshot, key=lambda x: abs(x[0] - target))
        if abs(best[0] - target) <= 1.5:
            _write("before", best[1])

    def _save_future(tag: str, wait: float):
        try:
            _t.sleep(wait)
            with _violation_ring_lock:
                latest_list = list(_violation_frame_ring.get(camera_id, []))
            if latest_list:
                _write(tag, latest_list[-1][1])
        except Exception:
            pass

    _vbuf_threading.Thread(target=_save_future, args=("mid_b", 0.6), daemon=True).start()
    _vbuf_threading.Thread(target=_save_future, args=("after", 2.1), daemon=True).start()

# ---- P1 Homography helpers (per-zone calibration → real-meter speed) ----
_zone_h_cache: Dict[str, tuple] = {}  # zone_id -> (H_matrix, ts)

def _get_zone_homography(zone):
    """Return cv2 perspective matrix that maps pixel (x,y) to world (m,m).
    Cached per zone_id. Returns None if no valid calibration.
    Accept two forms:
      A) zone.calibration = {points_pixel, width_m, length_m}
      B) zone.points (4 pts) + zone.calibration_width_m + zone.calibration_length_m
    """
    if not zone or not isinstance(zone, dict):
        return None
    pixel_pts = None
    width_m = None
    length_m = None
    # Form B: points on zone + calibration_width/length on zone
    pts = zone.get("points")
    if pts and len(pts) == 4:
        cw = zone.get("calibration_width_m")
        cl = zone.get("calibration_length_m")
        if cw and cl:
            pixel_pts = pts
            width_m = cw
            length_m = cl
    # Form A: explicit calibration object
    if pixel_pts is None:
        calib = zone.get("calibration")
        if calib and isinstance(calib, dict):
            pp = calib.get("points_pixel")
            if pp and len(pp) == 4:
                pixel_pts = pp
                width_m = calib.get("width_m")
                length_m = calib.get("length_m")
    if not pixel_pts or len(pixel_pts) != 4 or not width_m or not length_m:
        return None
    try:
        width_m = float(width_m); length_m = float(length_m)
    except Exception:
        return None
    if width_m <= 0 or length_m <= 0:
        return None
    zone_id = str(zone.get("id") or "") + f"::{width_m}x{length_m}"
    cached = _zone_h_cache.get(zone_id)
    if cached is not None:
        return cached[0]
    try:
        import cv2 as _cv2
        import numpy as _np
        px = _np.array(pixel_pts, dtype=_np.float32)
        # Clockwise from TL: (0,L)(W,L)(W,0)(0,0)
        wp = _np.array([[0, length_m], [width_m, length_m],
                        [width_m, 0], [0, 0]], dtype=_np.float32)
        H, _mask = _cv2.findHomography(px, wp)
        if H is None:
            return None
        _zone_h_cache[zone_id] = (H, time.time())
        return H
    except Exception as _e:
        print(f"[homography] zone {zone.get('id')} compute failed: {_e}", flush=True)
        return None


def _pixel_to_world_m(H, px_x, px_y):
    """Project pixel to world (meters). Returns (wx, wy) or None."""
    if H is None:
        return None
    try:
        import cv2 as _cv2
        import numpy as _np
        pt = _np.array([[[float(px_x), float(px_y)]]], dtype=_np.float32)
        wp = _cv2.perspectiveTransform(pt, H)
        return float(wp[0][0][0]), float(wp[0][0][1])
    except Exception:
        return None


def _bbox_bottom_center(bbox, class_name=None):
    """車輛接地點估計：bbox 底邊中點。P10: per-class offset."""
    if not bbox:
        return None, None
    try:
        x1, y1, x2, y2 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0)), int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        cx = (x1 + x2) // 2
        h = max(1, y2 - y1)
        _OFFSET = {'heavy_truck': 0.06, 'truck': 0.05, 'light_truck': 0.04, 'bus': 0.07, 'car': 0.02, 'motorcycle': 0.01, 'bicycle': 0.01, 'person': 0.0}
        off = _OFFSET.get(str(class_name or ''), 0.0)
        cy = y2 - int(h * off)
        return cx, cy
    except Exception:
        return None, None
# ---- /P1 helpers ----

# ---- P4 Trip Wire helpers ----
def _signed_distance_to_line(px, py, x1, y1, x2, y2):
    """Signed distance from (px,py) to segment line (x1,y1)-(x2,y2).
    Sign indicates which side of the line. cross product."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _line_crossed(prev_pt, curr_pt, line_pts):
    """Return True if track moved across the infinite line defined by line_pts (2 points).
    Cross detected when sign of signed distance flips between prev and curr.
    """
    if not prev_pt or not curr_pt or not line_pts or len(line_pts) < 2:
        return False
    try:
        x1, y1 = float(line_pts[0][0]), float(line_pts[0][1])
        x2, y2 = float(line_pts[1][0]), float(line_pts[1][1])
    except Exception:
        return False
    s_prev = _signed_distance_to_line(prev_pt[0], prev_pt[1], x1, y1, x2, y2)
    s_curr = _signed_distance_to_line(curr_pt[0], curr_pt[1], x1, y1, x2, y2)
    # sign change crosses zero (the line)
    return (s_prev > 0 and s_curr <= 0) or (s_prev < 0 and s_curr >= 0)


def _find_trip_wire_pair(speed_zones):
    """Return list of (lane_no, in_zone, out_zone, distance_m) pairs."""
    if not speed_zones:
        return []
    line_in_by_lane = {}
    line_out_by_lane = {}
    for z in speed_zones:
        if not isinstance(z, dict):
            continue
        ztype = str(z.get("type") or "")
        lane_no = z.get("lane_no")
        if not lane_no:
            continue
        try:
            lane_key = int(lane_no)
        except Exception:
            continue
        if ztype == "speed_line_in":
            line_in_by_lane[lane_key] = z
        elif ztype == "speed_line_out":
            line_out_by_lane[lane_key] = z
    pairs = []
    for lane_key, in_z in line_in_by_lane.items():
        out_z = line_out_by_lane.get(lane_key)
        if not out_z:
            continue
        try:
            dist_m = float(in_z.get("line_distance_m") or out_z.get("line_distance_m") or 0)
        except Exception:
            dist_m = 0.0
        if dist_m <= 0:
            continue
        pairs.append((lane_key, in_z, out_z, dist_m))
    return pairs
# ---- /P4 helpers ----

# ---- P11 Kalman filter (constant-velocity model) ----
class _KalmanCV:
    """Per-track 2D Kalman filter (constant velocity).
    State: [x, y, vx, vy]. Measurement: [x, y].
    Velocity derived in same units as input (world_m/s or px/s).
    """
    __slots__ = ("x", "y", "vx", "vy", "last_t", "P", "Q", "R")
    def __init__(self):
        self.x = self.y = None
        self.vx = self.vy = 0.0
        self.last_t = None
        self.P = 1.0
        self.Q = 0.05
        self.R = 0.5

    def update(self, mx: float, my: float, t: float):
        if self.x is None:
            self.x, self.y, self.last_t = mx, my, t
            return 0.0, 0.0
        dt = max(1e-3, t - self.last_t)
        # Predict
        pred_x = self.x + self.vx * dt
        pred_y = self.y + self.vy * dt
        P_pred = self.P + self.Q * dt
        # Kalman gain (scalar, same for x/y assumed)
        K = P_pred / (P_pred + self.R)
        ix = mx - pred_x
        iy = my - pred_y
        # Update position
        new_x = pred_x + K * ix
        new_y = pred_y + K * iy
        # Update velocity (need separate gain for velocity component, half-magnitude)
        Kv = K * 0.5
        new_vx = self.vx + Kv * (ix / dt)
        new_vy = self.vy + Kv * (iy / dt)
        self.x, self.y = new_x, new_y
        self.vx, self.vy = new_vx, new_vy
        self.P = (1 - K) * P_pred
        self.last_t = t
        return new_vx, new_vy
# ---- /P11 ----






# 事件截圖節流：每 cam 上次存截圖的 ts，limit ~1 張/2秒
_per_cam_last_snap_ts: Dict[int, float] = {}
# Frigate snap 設定快取：{camera_id: (enabled_bool, cached_at_ts)}，30s 過期
_cam_snap_enabled_cache: Dict[int, tuple] = {}


def _is_snapshot_enabled_for_cam(camera_id: int) -> bool:
    """讀 Frigate config 看該 cam 是否啟用「事件截圖」(snapshots.enabled)。30s 快取。"""
    cached = _cam_snap_enabled_cache.get(camera_id)
    if cached and (time.time() - cached[1]) < 30.0:
        return cached[0]
    enabled = False
    try:
        import yaml
        with open("/home/ubuntu/traffic-violation-detection/config/frigate/config.yml") as f:
            cfg = yaml.safe_load(f) or {}
        cam_cfg = (cfg.get("cameras") or {}).get(f"cam_{camera_id}") or {}
        enabled = bool((cam_cfg.get("snapshots") or {}).get("enabled", False))
    except Exception:
        enabled = False
    _cam_snap_enabled_cache[camera_id] = (enabled, time.time())
    return enabled
# Per-camera detector instances（每 cam 獨立 → 並行推理、無 lock 競爭）
_per_cam_detectors: Dict[int, "object"] = {}
_per_cam_detectors_lock = threading.Lock()
# 舊的 fallback singleton（給非 cam 上下文呼叫，例如 cameras.py 的 analyze）
_shared_overlay_detector = None
_shared_overlay_detector_lock = threading.Lock()

def _get_per_cam_detector(camera_id: int):
    """每個 camera 一個獨立 VehicleDetector，避免 4 cams 共享同一 detector 被 lock 序列化。"""
    if camera_id in _per_cam_detectors:
        return _per_cam_detectors[camera_id]
    with _per_cam_detectors_lock:
        if camera_id in _per_cam_detectors:
            return _per_cam_detectors[camera_id]
        try:
            from detection.vehicle_detector import VehicleDetector
            _per_cam_detectors[camera_id] = VehicleDetector(conf_threshold=0.15)
            print(f"⚡ cam_{camera_id} 獨立 detector 初始化完成", flush=True)
        except Exception as e:
            add_log("warning", f"cam_{camera_id} 獨立 detector 初始化失敗: {e}", "stream")
            _per_cam_detectors[camera_id] = None
        return _per_cam_detectors[camera_id]


def _get_shared_overlay_detector():
    """非 cam 場景的 fallback singleton（cameras.py analyze 等）"""
    global _shared_overlay_detector
    if _shared_overlay_detector is not None:
        return _shared_overlay_detector
    with _shared_overlay_detector_lock:
        if _shared_overlay_detector is not None:
            return _shared_overlay_detector
        try:
            from detection.vehicle_detector import VehicleDetector
            _shared_overlay_detector = VehicleDetector(conf_threshold=0.15)
        except Exception as e:
            add_log("warning", f"overlay 共用偵測器初始化失敗: {e}", "stream")
        return _shared_overlay_detector
snapshot_cache: Dict[int, dict] = {}
snapshot_locks: Dict[int, asyncio.Lock] = {}
snapshot_warm_tasks: Dict[int, asyncio.Task] = {}
_unicode_font_cache: Dict[int, object] = {}
http_mjpeg_workers: Dict[str, dict] = {}
http_mjpeg_workers_lock = threading.Lock()
HTTP_MJPEG_IDLE_TIMEOUT_SEC = 6.0


async def _wait_lock_release(lock: asyncio.Lock, timeout: float = 3.0):
    start = time.time()
    while lock.locked():
        if (time.time() - start) >= timeout:
            raise asyncio.TimeoutError()
        await asyncio.sleep(0.03)


async def _warm_snapshot(camera_id: int, source: str):
    """Background warm-up snapshot to improve next request hit-rate (僅 raw 版本)。"""
    cache_key = (camera_id, False)
    lock = snapshot_locks.setdefault(cache_key, asyncio.Lock())
    if lock.locked():
        return
    async with lock:
        image = None
        for _ in range(2):
            try:
                image = await asyncio.wait_for(
                    asyncio.to_thread(_capture_snapshot_bytes, source, camera_id=camera_id),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                image = None
            if image:
                break
        if image:
            snapshot_cache[cache_key] = {"image": image, "ts": time.time()}


def _open_capture(source: str):
    """OpenCV capture with backend fallback to avoid hard dependency on FFMPEG backend.
    強制 RTSP 走 TCP 避免封包遺失導致殘影 / 條紋。
    """
    source = resolve_capture_source(source)
    source_lc = str(source or "").lower()
    is_rtsp = source_lc.startswith("rtsp://")
    if is_rtsp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;10000000|buffer_size;131072|allowed_media_types;video|analyzeduration;1000000|probesize;1000000"

    backends = []
    is_http = source_lc.startswith("http://") or source_lc.startswith("https://")
    is_http_mjpeg = is_http and ("mpjpeg" in source_lc or source_lc.endswith(".mjpg") or source_lc.endswith(".mjpeg"))
    # 只有 HTTP MJPEG 才用 CAP_OPENCV_MJPEG；其他 HTTP 格式會 crash
    if is_http_mjpeg:
        mjpeg_backend = getattr(cv2, "CAP_OPENCV_MJPEG", None)
        if mjpeg_backend is not None:
            backends.append(mjpeg_backend)
    ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
    gst_backend = getattr(cv2, "CAP_GSTREAMER", None)
    if ffmpeg_backend is not None:
        backends.append(ffmpeg_backend)
    if gst_backend is not None and gst_backend not in backends:
        backends.append(gst_backend)
    if None not in backends:
        backends.append(None)

    last_cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
        except Exception:
            cap = None
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            last_cap = cap
            try:
                cap.release()
            except Exception:
                pass

    return last_cap if last_cap is not None else cv2.VideoCapture(source)


def _is_http_mjpeg_source(source: str) -> bool:
    text = str(source or "").strip().lower()
    if not (text.startswith("http://") or text.startswith("https://")):
        return False
    try:
        parsed = urlsplit(text)
        path = parsed.path or ""
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    except Exception:
        path = text
        params = {}
    return (
        path.endswith(".mjpg")
        or path.endswith(".mjpeg")
        or path.endswith(".mpjpeg")
        or str(params.get("format", "")).lower() == "mpjpeg"
    )


def _iter_http_mjpeg_jpegs(source: str):
    response = requests.get(
        source,
        headers={"User-Agent": "traffic-stream/1.0"},
        stream=True,
        timeout=(5, 20),
        verify=False,
    )
    response.raise_for_status()
    buffer = b""
    try:
        for chunk in response.iter_content(chunk_size=16 * 1024):
            if not chunk:
                continue
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(buffer) > (2 * 1024 * 1024):
                        buffer = buffer[-64 * 1024 :]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    if start > 0:
                        buffer = buffer[start:]
                    break
                jpg = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                if jpg:
                    yield jpg
    finally:
        response.close()


def _iter_http_mjpeg_frames(source: str):
    for jpg in _iter_http_mjpeg_jpegs(source):
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None and getattr(frame, "size", 0) > 0:
            yield frame


def _http_mjpeg_worker_loop(source: str, state: dict):
    while state.get("running", True):
        if time.time() - float(state.get("last_consumer_ts") or 0.0) > HTTP_MJPEG_IDLE_TIMEOUT_SEC:
            break
        try:
            for jpg in _iter_http_mjpeg_jpegs(source):
                if not state.get("running", True):
                    break
                if time.time() - float(state.get("last_consumer_ts") or 0.0) > HTTP_MJPEG_IDLE_TIMEOUT_SEC:
                    state["running"] = False
                    break
                state["last_jpeg"] = jpg
                state["last_ts"] = time.time()
                state["error"] = ""
        except Exception as e:
            state["error"] = str(e)
            time.sleep(1.0)
    state["running"] = False
    with http_mjpeg_workers_lock:
        current = http_mjpeg_workers.get(source)
        if current is state:
            http_mjpeg_workers.pop(source, None)


def _ensure_http_mjpeg_worker(source: str) -> dict:
    with http_mjpeg_workers_lock:
        state = http_mjpeg_workers.get(source)
        thread = state.get("thread") if isinstance(state, dict) else None
        if state and thread is not None and thread.is_alive():
            return state
        state = {
            "running": True,
            "last_jpeg": None,
            "last_ts": 0.0,
            "last_consumer_ts": time.time(),
            "error": "",
            "thread": None,
        }
        worker = threading.Thread(
            target=_http_mjpeg_worker_loop,
            args=(source, state),
            daemon=True,
            name=f"http-mjpeg-{abs(hash(source)) % 100000}",
        )
        state["thread"] = worker
        http_mjpeg_workers[source] = state
        worker.start()
        return state


def _touch_http_mjpeg_worker(state: dict | None) -> None:
    if isinstance(state, dict):
        state["last_consumer_ts"] = time.time()


def _try_frigate_snapshot(source: str, camera_id: int = None):
    """嘗試透過 Frigate latest.jpg API 取得截圖（適用於 Frigate/go2rtc 管理的串流）。
    優先用 camera_id → cam_{id} 對應 frigate stream name；source URL 解析為 fallback。
    """
    candidates = []
    if camera_id is not None:
        candidates.append(f"cam_{camera_id}")
    src = str(source or "").lower()
    if src.startswith("rtsp://127.0.0.1:8554/"):
        n = source.split("/")[-1]
        if n and n not in candidates:
            candidates.append(n)
    for stream_name in candidates:
        try:
            resp = requests.get(
                f"http://127.0.0.1:5000/api/{stream_name}/latest.jpg",
                timeout=3.0,
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
        except Exception:
            pass
    return None


def _capture_snapshot_bytes(source: str, camera_id: int = None, overlay_zones: list = None):
    """嘗試以較短超時擷取單張影像，避免 RTSP 長時間阻塞。
    overlay_zones=None: 直接 encode raw frame（cam tile 縮圖預設）
    overlay_zones=list: 疊 ROI + detection bbox（snapshot?overlay=1 用）
    """
    # 優先嘗試從 detection 服務的共享 frame 取得（避免重開連線）
    if camera_id is not None:
        sf = _shared_frames.get(camera_id)
        if sf and (time.time() - sf.get("ts", 0)) < 30.0 and sf.get("frame") is not None:
            frame_out = sf["frame"]
            if overlay_zones is not None:
                # 疊 ROI 多邊形 + detection bbox（讓 cam tile 縮圖也能看到偵測結果）
                annotated = frame_out.copy()
                if overlay_zones:
                    try:
                        _draw_roi_labels(annotated, overlay_zones)
                    except Exception:
                        pass
                _ZH = {"car": "小客車", "motorcycle": "機車", "truck": "大貨車", "bus": "公車",
                       "heavy_truck": "重型貨車", "light_truck": "小貨車"}
                for det in sf.get("detections") or []:
                    cls = det.get("class_name", "")
                    if cls not in ("car", "motorcycle", "truck", "bus", "heavy_truck", "light_truck"):
                        continue
                    b = det.get("bbox") or {}
                    if not all(k in b for k in ("x1", "y1", "x2", "y2")):
                        continue
                    cv2.rectangle(annotated, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 200, 0), 2)
                    cv2.putText(annotated, _ZH.get(cls, cls), (int(b["x1"]), max(20, int(b["y1"]) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                frame_out = annotated
            ok, buffer = cv2.imencode(".jpg", frame_out)
            if ok:
                return buffer.tobytes()
    # 嘗試 Frigate snapshot API（比重新建立 RTSP 連線快很多）
    frigate_jpg = _try_frigate_snapshot(source, camera_id=camera_id)
    if frigate_jpg:
        return frigate_jpg
    # go2rtc HTTP frame.jpeg fallback：cv2 對 idle/半死攝影機處理不好，
    # go2rtc 已維持 stream session 可直接給 frame。stream 名稱用 cam_{id} 對應 go2rtc.yaml。
    if camera_id is not None:
        try:
            resp = requests.get(
                f"http://127.0.0.1:1984/api/frame.jpeg?src=cam_{camera_id}",
                timeout=4.0,
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
        except Exception:
            pass
    http_source = resolve_local_api_source(source)
    if _is_http_mjpeg_source(http_source):
        state = _ensure_http_mjpeg_worker(http_source)
        deadline = time.time() + 4.0
        while time.time() < deadline:
            _touch_http_mjpeg_worker(state)
            jpg = state.get("last_jpeg")
            if jpg:
                return jpg
            time.sleep(0.05)
        return None
    source = resolve_capture_source(source)
    cap = _open_capture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if not ret:
            return None
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return buffer.tobytes()
    finally:
        cap.release()


def _placeholder_jpeg(text: str = "NO SIGNAL", width: int = 640, height: int = 360) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, "STREAM UNAVAILABLE", (28, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 180, 255), 2)
    cv2.putText(frame, str(text or ""), (28, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        return buffer.tobytes()
    return b""


def generate_frames(source: str):
    """產生 MJPEG 串流"""
    http_source = resolve_local_api_source(source)
    if _is_http_mjpeg_source(http_source):
        state = _ensure_http_mjpeg_worker(http_source)
        last_sent_ts = 0.0
        while True:
            _touch_http_mjpeg_worker(state)
            jpg = state.get("last_jpeg")
            ts = float(state.get("last_ts") or 0.0)
            if jpg and ts > 0 and ts != last_sent_ts:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                last_sent_ts = ts
                time.sleep(0.08)
                continue
            if (time.time() - ts) > 3.0:
                ph = _placeholder_jpeg(state.get("error") or "Waiting for HTTP MJPEG frame...")
                if ph:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + ph + b'\r\n')
            time.sleep(0.2)
        return

    cap = _open_capture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    last_ok = time.time()
    last_placeholder_ts = 0.0
    had_frame = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # RTSP 串流失幀時重連，避免長時間卡住無輸出
            if time.time() - last_ok > 2.0:
                cap.release()
                time.sleep(0.1)
                cap = _open_capture(source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 若長時間無幀，持續輸出保底畫面，避免前端 img 一直 loading 無回應
            now_ts = time.time()
            if ((not had_frame) or (now_ts - last_ok > 3.0)) and (now_ts - last_placeholder_ts) > 1.0:
                ph = _placeholder_jpeg("Waiting for camera frame...")
                if ph:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + ph + b'\r\n')
                last_placeholder_ts = now_ts
            time.sleep(0.03)
            continue
        last_ok = time.time()
        had_frame = True
        
        # 縮小尺寸加快傳輸
        frame = cv2.resize(frame, (640, 360))
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.1)  # ~10 FPS
    
    cap.release()


def _zone_points_for_frame(zone: dict, frame_w: int, frame_h: int):
    pts = zone.get("points", [])
    if len(pts) < 3:
        return None
    src_w = int(zone.get("source_width") or 0)
    src_h = int(zone.get("source_height") or 0)
    if src_w > 0 and src_h > 0 and (src_w != frame_w or src_h != frame_h):
        scaled = []
        for p in pts:
            x = float(p[0]) * frame_w / src_w
            y = float(p[1]) * frame_h / src_h
            scaled.append([x, y])
        return np.array(scaled, dtype=np.float32).reshape(-1, 1, 2)
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def _point_in_zone(cx: int, cy: int, zone: dict, frame_w: int, frame_h: int) -> bool:
    poly = _zone_points_for_frame(zone, frame_w, frame_h)
    if poly is None:
        return False
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def _nearest_track_id(center: tuple, class_name: str, tracks: dict, max_dist: float = 90.0):
    cx, cy = center
    best_id = None
    best_d = float("inf")
    for tid, tr in tracks.items():
        if tr.get("class_name") != class_name:
            continue
        tx, ty = tr.get("center", (0, 0))
        d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if d < best_d and d <= max_dist:
            best_d = d
            best_id = tid
    return best_id


def _latest_lpr_plate(camera_id: int, max_age_sec: int = 20):
    """讀取近期 LPR 結果，作為違規事件車牌來源。"""
    db = None
    try:
        from api.models import SessionLocal, LPRRecord
        db = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(seconds=max(1, int(max_age_sec)))
        row = (
            db.query(LPRRecord)
            .filter(LPRRecord.camera_id == int(camera_id))
            .filter(LPRRecord.created_at >= cutoff)
            .filter(LPRRecord.plate_number.isnot(None))
            .filter(LPRRecord.plate_number != "")
            .order_by(LPRRecord.created_at.desc(), LPRRecord.id.desc())
            .first()
        )
        if not row:
            return None
        return {
            "plate": str(row.plate_number or "").strip(),
            "confidence": float(row.confidence or 0.0),
            "created_at": row.created_at.isoformat() if row.created_at else "",
        }
    except Exception:
        return None
    finally:
        if db is not None:
            db.close()


def _get_unicode_font(size: int = 16):
    cached = _unicode_font_cache.get(size)
    if cached is not None:
        return cached
    candidates = [
        "/workspace/web/fonts/NotoSansCJK-Regular.ttc",
        "/workspace/fonts/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    font = None
    if _PIL_AVAILABLE:
        for p in candidates:
            if Path(p).exists():
                try:
                    font = ImageFont.truetype(p, size=size)
                    break
                except Exception:
                    continue
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
    _unicode_font_cache[size] = font
    return font


def _draw_roi_labels(frame, zones: list):
    if not zones:
        return
    h, w = frame.shape[:2]
    color_map = {
        "detection": (64, 200, 80),
        "flow_detection": (64, 200, 80),
        "speed": (0, 140, 255),
        "speed_roi": (0, 140, 255),
        "speed_line_in": (255, 120, 0),
        "speed_line_out": (180, 80, 255),
        "lane_left": (220, 60, 120),
        "lane_straight": (30, 170, 235),
        "lane_right": (0, 190, 255),
    }
    pil_img = None
    pil_draw = None
    font = None
    for z in zones:
        poly = _zone_points_for_frame(z, w, h)
        if poly is None:
            continue
        ztype = str(z.get("type") or "")
        color = color_map.get(ztype, (140, 140, 140))
        cv2.polylines(frame, [poly.astype(np.int32)], True, color, 2, lineType=cv2.LINE_AA)
        name = str(z.get("name") or "").strip()
        if not name:
            continue
        x, y = poly.reshape(-1, 2)[0]
        tx, ty = int(x), max(18, int(y) - 8)
        if name.isascii() or not _PIL_AVAILABLE:
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(frame, (tx - 2, ty - th - 6), (tx + tw + 6, ty + 4), color, -1)
            cv2.putText(frame, name, (tx + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
            continue

        if pil_img is None:
            pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_draw = ImageDraw.Draw(pil_img)
            font = _get_unicode_font(16)

        if pil_draw is None or font is None:
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(frame, (tx - 2, ty - th - 6), (tx + tw + 6, ty + 4), color, -1)
            cv2.putText(frame, name, (tx + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
            continue

        try:
            l, t, r, b = pil_draw.textbbox((0, 0), name, font=font)
            tw, th = int(r - l), int(b - t)
        except Exception:
            tw, th = pil_draw.textsize(name, font=font)
        bg = (int(color[2]), int(color[1]), int(color[0]))
        top = max(0, ty - th - 10)
        pil_draw.rectangle([tx - 2, top, tx + tw + 8, ty + 4], fill=bg)
        pil_draw.text((tx + 2, top + 2), name, fill=(0, 0, 0), font=font)

    if pil_img is not None:
        frame[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def generate_frames_overlay(
    source: str,
    zones: list = None,
    detection_config: dict = None,
    *,
    render_overlay: bool = True,
    render_roi_labels: bool = True,
    camera_id: int = None,
    high_quality: bool = False,
):
    """產生即時 MJPEG 串流，可選擇是否繪製辨識疊加。
    high_quality=True: 輸出 1080p JPEG Q75；否則輸出 720p JPEG Q60。
    """
    http_source = resolve_local_api_source(source)
    use_http_mjpeg = _is_http_mjpeg_source(http_source)
    source = resolve_capture_source(source)
    # file source 判斷：mp4/mkv/mov/avi/webm 或非 http/rtsp 開頭
    _src_lc_overlay = str(source or "").lower()
    _is_file_overlay = (
        not _src_lc_overlay.startswith(("rtsp://", "http://", "https://"))
        or _src_lc_overlay.endswith((".mp4", ".mkv", ".mov", ".avi", ".webm"))
    )
    # 背景開啟 RTSP（避免阻塞 generator）
    cap = None
    _cap_holder = [None]
    _cap_event = threading.Event()
    if use_http_mjpeg or _is_file_overlay:
        # file source 完全靠 detection 的 _shared_frames，不浪費開 cap（同一 mkv 被多開）
        _cap_event.set()
    else:
        def _bg_open():
            _cap_holder[0] = _open_capture(source)
            _cap_event.set()
        threading.Thread(target=_bg_open, daemon=True).start()
    zones = zones or []
    detection_config = detection_config or {}
    det_zones = select_zones(zones, scope=SCOPE_TRAFFIC, allowed_types=("detection", "flow_detection"), fallback_scopes=(SCOPE_CONGESTION,))
    speed_zones = select_zones(zones, scope=SCOPE_SPEED, allowed_types=("speed", "speed_roi", "speed_line_in", "speed_line_out"))
    # 粗略像素速度轉換係數（可在 detection_config.speed_kmh_per_pxps 調整）
    speed_kmh_per_pxps = float(detection_config.get("speed_kmh_per_pxps", 0.12))
    speed_smooth_alpha = float(detection_config.get("speed_smooth_alpha", 0.35))
    tracks = {}
    next_track_id = 1
    track_ttl_sec = 1.2
    _det_last = []
    _det_count = 0

    last_ok = time.time()
    last_placeholder_ts = 0.0
    had_frame = False
    http_frames = None
    http_state = _ensure_http_mjpeg_worker(http_source) if use_http_mjpeg else None
    last_http_ts = 0.0
    last_shared_ts = 0.0
    _shared_warm = False
    # 輸出限速：避免瀏覽器同時解 4 cam × 14 FPS × 100KB = 5.6 MB/s 累積延遲導致頓
    OUTPUT_FPS_CAP = 8.0
    _min_yield_interval = 1.0 / OUTPUT_FPS_CAP
    _last_yield_ts = 0.0
    while True:
        # process shutdown 時讓 generator 乾淨退出，避免卡到 systemd SIGKILL
        if shutdown_event.is_set():
            return
        # 優先用 detection worker 已解碼好的 frame（省一次 RTSP 解碼、單一時間源）；
        # 若 detection 完全未啟動才 fallback 自己讀 RTSP。
        ret = False
        frame = None
        _sf_early = _shared_frames.get(camera_id) if camera_id else None
        if _sf_early and _sf_early.get("frame") is not None:
            ts = float(_sf_early.get("ts") or 0.0)
            age = time.time() - ts
            if ts > 0 and age < 2.0:
                _shared_warm = True
                if ts != last_shared_ts:
                    frame = _sf_early["frame"].copy() if hasattr(_sf_early["frame"], "copy") else _sf_early["frame"]
                    ret = frame is not None and getattr(frame, "size", 0) > 0
                    last_shared_ts = ts
                else:
                    # 同 ts → 沒新 frame，短暫等再 poll，避免重複送舊 frame 浪費 CPU
                    time.sleep(0.01)
                    continue
        if not ret and _shared_warm:
            # 偵測 worker 之前有 frame 但這次拿不到（短暫 stale），不要切去 RTSP，等下一輪
            time.sleep(0.02)
            continue
        if not ret:
            if use_http_mjpeg:
                _touch_http_mjpeg_worker(http_state)
                ts = float(http_state.get("last_ts") or 0.0) if http_state else 0.0
                jpg = http_state.get("last_jpeg") if http_state else None
                if jpg and ts > 0 and ts != last_http_ts:
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    ret = frame is not None and getattr(frame, "size", 0) > 0
                    last_http_ts = ts
            else:
                if cap is None and _cap_event.wait(timeout=0.05):
                    cap = _cap_holder[0]
                if cap is not None:
                    ret, frame = cap.read()
        # frigate/go2rtc fallback：cap.read / _shared_frames 都拿不到時，
        # 試 frigate latest.jpg（很快）或 go2rtc frame.jpeg（較慢），throttle 0.15s ~ 6fps
        if not ret and camera_id is not None:
            _now_fb = time.time()
            if (_now_fb - globals().setdefault('_live_fb_last', {}).get(camera_id, 0.0)) > 0.15:
                _live_fb_jpg = _try_frigate_snapshot(source, camera_id=camera_id)
                if not _live_fb_jpg:
                    try:
                        _resp = requests.get(
                            f"http://127.0.0.1:1984/api/frame.jpeg?src=cam_{camera_id}",
                            timeout=3.0,
                        )
                        if _resp.status_code == 200 and len(_resp.content) > 1000:
                            _live_fb_jpg = _resp.content
                    except Exception:
                        pass
                if _live_fb_jpg:
                    _live_fb_arr = np.frombuffer(_live_fb_jpg, dtype=np.uint8)
                    _decoded = cv2.imdecode(_live_fb_arr, cv2.IMREAD_COLOR)
                    if _decoded is not None and _decoded.size > 0:
                        frame = _decoded
                        ret = True
                globals()['_live_fb_last'][camera_id] = _now_fb
        if not ret:
            if cap is not None and time.time() - last_ok > 5.0:
                cap.release()
                cap = None
                _cap_event.clear()
                def _bg_reopen():
                    _cap_holder[0] = _open_capture(source)
                    _cap_event.set()
                threading.Thread(target=_bg_reopen, daemon=True).start()
            now_ts = time.time()
            if ((not had_frame) or (now_ts - last_ok > 3.0)) and (now_ts - last_placeholder_ts) > 1.0:
                ph = _placeholder_jpeg("Overlay waiting for camera frame...")
                if ph:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + ph + b'\r\n')
                last_placeholder_ts = now_ts
            time.sleep(0.03)
            continue
        last_ok = time.time()
        had_frame = True

        if not render_overlay:
            fh0, fw0 = frame.shape[:2]
            if fw0 > 1920:
                s0 = 1920.0 / fw0
                out = cv2.resize(frame, (1920, int(fh0 * s0)))
            else:
                out = frame
            _, buffer = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        # 影像永遠走自己讀到的 RTSP frame（25 FPS 順暢，無殘影）。
        # bbox 從 _shared_frames 拿最近偵測結果疊上去；detection 8 FPS → bbox 約延遲 120ms。
        # 偵測落後 > 1.5s 用橙色細框標示「stale」，> 3s 不畫。
        _label_items = []
        _sf = _shared_frames.get(camera_id) if camera_id else None
        det_age = None
        stale_overlay = False
        if _sf:
            det_age = time.time() - float(_sf.get("ts", 0) or 0)
            if det_age < 3.0:
                detections = _sf.get("detections", [])
                _det_last = detections
                if det_age >= 1.5:
                    stale_overlay = True
            else:
                detections = _det_last
                stale_overlay = True
        else:
            detections = _det_last
        if detections:
            try:
                fh, fw = frame.shape[:2]
                now_ts = time.time()
                valid_dets = []
                for det in detections:
                    if det.get("class_name") not in ["car", "motorcycle", "truck", "bus", "heavy_truck", "light_truck"]:
                        continue
                    b = det["bbox"]
                    cx = (b["x1"] + b["x2"]) // 2
                    cy = (b["y1"] + b["y2"]) // 2
                    in_det_roi = True
                    if det_zones:
                        in_det_roi = any(_point_in_zone(cx, cy, z, fw, fh) for z in det_zones)
                    if not in_det_roi:
                        continue
                    in_speed_roi = any(_point_in_zone(cx, cy, z, fw, fh) for z in speed_zones) if speed_zones else False
                    valid_dets.append((det, cx, cy, in_speed_roi))

                stale_ids = [tid for tid, tr in tracks.items() if (now_ts - tr.get("t", now_ts)) > track_ttl_sec]
                for tid in stale_ids:
                    tracks.pop(tid, None)

                for det, cx, cy, in_speed_roi in valid_dets:
                    track_id = _nearest_track_id((cx, cy), det["class_name"], tracks)
                    if track_id is None:
                        track_id = next_track_id
                        next_track_id += 1
                        tracks[track_id] = {"center": (cx, cy), "t": now_ts, "class_name": det["class_name"], "speed_kmh": None}
                    prev = tracks.get(track_id, {})
                    prev_center = prev.get("center", (cx, cy))
                    prev_t = prev.get("t", now_ts)
                    dt = max(1e-3, now_ts - prev_t)
                    px_dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                    pxps = px_dist / dt
                    raw_kmh = pxps * speed_kmh_per_pxps
                    prev_kmh = prev.get("speed_kmh")
                    if prev_kmh is None:
                        speed_kmh = raw_kmh
                    else:
                        speed_kmh = (speed_smooth_alpha * raw_kmh) + ((1.0 - speed_smooth_alpha) * prev_kmh)
                    tracks[track_id] = {"center": (cx, cy), "t": now_ts, "class_name": det["class_name"], "speed_kmh": speed_kmh}

                    b = det["bbox"]
                    if render_overlay:
                        # stale 時改橙色、單線，作為「偵測跟不上影像」的視覺提示
                        color = (0, 140, 255) if stale_overlay else (0, 200, 0)
                        thick = 1 if stale_overlay else 2
                        cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thick)
                        _ZH = {"car": "小客車", "motorcycle": "機車", "truck": "大貨車", "bus": "公車", "heavy_truck": "重型貨車", "light_truck": "小貨車", "person": "行人", "bicycle": "自行車"}
                        truck_cls = det.get("truck_cls")
                        zh = str(truck_cls["label"]) if truck_cls and truck_cls.get("label") else _ZH.get(det["class_name"], det["class_name"])
                        label = zh
                        if in_speed_roi:
                            # 優先用 detection thread 算好的 speed（Homography + 5-frame gate），fallback 才用 overlay 自己估的
                            _det_speed = det.get("speed_kmh")
                            _show_speed = _det_speed if isinstance(_det_speed, (int, float)) and _det_speed > 0 else None
                            if _show_speed is None and isinstance(speed_kmh, (int, float)) and 0 < speed_kmh < 150:
                                _show_speed = speed_kmh
                            if _show_speed is not None:
                                label += f" {int(_show_speed)}km/h"
                        _label_items.append((label, (b["x1"], max(2, b["y1"] - 24)), color, (0, 0, 0)))
            except Exception:
                pass
        # 畫標籤（中文用 PIL，ASCII 用 cv2）
        for text, (tx, ty), fg, bg in _label_items:
            if not text.isascii() and _PIL_AVAILABLE:
                try:
                    font = _get_unicode_font(18)
                    if font:
                        _tw = int(len(text) * 14)
                        _th = 24
                        _rx1 = max(0, tx - 2)
                        _ry1 = max(0, ty - 2)
                        _rx2 = min(frame.shape[1], tx + _tw + 4)
                        _ry2 = min(frame.shape[0], ty + _th + 4)
                        roi = frame[_ry1:_ry2, _rx1:_rx2].copy()
                        pil_roi = PILImage.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        d = ImageDraw.Draw(pil_roi)
                        d.rectangle([0, 0, pil_roi.width, pil_roi.height], fill=bg)
                        d.text((2, 2), text, fill=fg, font=font)
                        frame[_ry1:_ry2, _rx1:_rx2] = cv2.cvtColor(np.asarray(pil_roi), cv2.COLOR_RGB2BGR)
                except Exception:
                    cv2.putText(frame, text, (tx, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            else:
                cv2.rectangle(frame, (tx - 2, ty - 2), (tx + len(text) * 10, ty + 18), bg, -1)
                cv2.putText(frame, text, (tx, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1)
        if render_roi_labels:
            _draw_roi_labels(frame, zones)
        # 輸出解析度與品質（依 high_quality 切換）：
        #   low  (預設): 1280 寬 Q60 (監控網格用，頻寬輕)
        #   high        : 1920 寬 Q75 (放大/轉發用，清晰度優先)
        fh, fw = frame.shape[:2]
        _target_w = 1920 if high_quality else 1280
        _jpg_q = 75 if high_quality else 60
        if fw > _target_w:
            scale = _target_w / fw
            frame = cv2.resize(frame, (_target_w, int(fh * scale)))
        # 輸出限速 8 FPS：避免瀏覽器解碼負荷累積
        now_yt = time.time()
        wait = _min_yield_interval - (now_yt - _last_yield_ts)
        if wait > 0:
            time.sleep(wait)
        _last_yield_ts = time.time()
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, _jpg_q])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    if cap is not None:
        cap.release()


@router.get("/{camera_id}/live")
async def live_stream(camera_id: int, db: Session = Depends(get_db)):
    """即時影像串流 (MJPEG，背景仍持續分析但不畫辨識框)"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")
    return StreamingResponse(
        generate_frames_overlay(
            resolve_analysis_source(camera),
            camera.zones or [],
            camera.detection_config or {},
            render_overlay=False,
            render_roi_labels=False,
            camera_id=camera_id,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/{camera_id}/live-overlay")
async def live_stream_overlay(camera_id: int, q: str = "low", roi: str = "0", db: Session = Depends(get_db)):
    """即時影像串流 (MJPEG + AI 辨識框)

    q=low (預設): 720p, JPEG Q60, 監控網格用 (頻寬輕)
    q=high: 1080p, JPEG Q75, 放大/轉發用 (清晰度優先)
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")
    overlay_source = resolve_analysis_source(camera)
    hq = str(q or "").lower() == "high"
    # roi 預設疊加；前端要關 (live-overlay?roi=0) 時才 false
    show_roi = str(roi or "").strip() != "0"
    return StreamingResponse(
        generate_frames_overlay(
            overlay_source,
            camera.zones or [],
            camera.detection_config or {},
            render_roi_labels=show_roi,
            camera_id=camera_id,
            high_quality=hq,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/{camera_id}/snapshot")
async def snapshot(camera_id: int, overlay: int = 0, db: Session = Depends(get_db)):
    """取得單張截圖。overlay=1 疊加 ROI + detection bbox（cam tile 縮圖也能看到偵測）"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")

    overlay_zones = (camera.zones or []) if overlay else None
    # cache 區分原始/疊加版本（同 cam 兩種預先 generate）
    cache_key = (camera_id, bool(overlay))
    # 若最近已有快照，直接回傳，避免前端縮圖連續請求造成 RTSP 阻塞
    # TTL 0.3s：對 RTSP cam 仍能擋連續 thrashing，對 file source / 有 detection worker 持續產 frame
    # 的 cam，前端每秒 polling 都能拿到新 frame，影像不再卡 1-2 秒
    cached = snapshot_cache.get(cache_key)
    if cached and (time.time() - cached.get("ts", 0) <= 1.5):
        return StreamingResponse(iter([cached.get("image")]), media_type="image/jpeg")

    lock = snapshot_locks.setdefault(cache_key, asyncio.Lock())
    image = None

    # 同一攝影機已有擷取進行中時，先嘗試回傳快取；若沒有快取，短暫等待前一個請求完成
    if lock.locked():
        if cached and (time.time() - cached.get("ts", 0) <= 600):
            return StreamingResponse(iter([cached.get("image")]), media_type="image/jpeg")
        try:
            await _wait_lock_release(lock, timeout=3.0)
        except asyncio.TimeoutError:
            ph = _placeholder_jpeg("影像來源忙碌中")
            if ph:
                return StreamingResponse(iter([ph]), media_type="image/jpeg")
            raise HTTPException(status_code=503, detail="影像來源忙碌中")
        cached_after_wait = snapshot_cache.get(cache_key)
        if cached_after_wait and (time.time() - cached_after_wait.get("ts", 0) <= 600):
            return StreamingResponse(iter([cached_after_wait.get("image")]), media_type="image/jpeg")

    async with lock:
        # Keep snapshot latency very short for UI usage; slow sources should fallback fast
        # and let warm-up task fill cache for subsequent requests.
        attempts = [(8.0, 1)]
        for timeout_sec, count in attempts:
            if image:
                break
            for _ in range(count):
                if image:
                    break
                try:
                    image = await asyncio.wait_for(
                        asyncio.to_thread(_capture_snapshot_bytes, resolve_analysis_source(camera), camera_id=camera.id, overlay_zones=overlay_zones),
                        timeout=timeout_sec,
                    )
                except asyncio.TimeoutError:
                    image = None

    if image:
        snapshot_cache[cache_key] = {"image": image, "ts": time.time()}
    else:
        if cached and (time.time() - cached.get("ts", 0) <= 600):
            image = cached.get("image")
        else:
            # No immediate snapshot/cached image: trigger background warm-up for next request.
            task = snapshot_warm_tasks.get(camera_id)
            if task is None or task.done():
                snapshot_warm_tasks[camera_id] = asyncio.create_task(
                    _warm_snapshot(camera_id, resolve_analysis_source(camera))
                )
                task = snapshot_warm_tasks[camera_id]
            # Wait briefly for warm-up result, then fallback to latest cache if available.
            try:
                await asyncio.wait_for(task, timeout=0.8)
            except asyncio.TimeoutError:
                pass
            warmed = snapshot_cache.get(camera_id)
            if warmed and (time.time() - warmed.get("ts", 0) <= 600):
                image = warmed.get("image")

    if not image:
        ph = _placeholder_jpeg("無法取得影像")
        if ph:
            return StreamingResponse(iter([ph]), media_type="image/jpeg")
        raise HTTPException(status_code=503, detail="無法取得影像")

    return StreamingResponse(
        iter([image]),
        media_type="image/jpeg"
    )


@router.post("/{camera_id}/detection/start")
async def start_detection(camera_id: int, db: Session = Depends(get_db)):
    """啟動偵測服務"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")
    if camera_id in detection_services and detection_services[camera_id].get('running'):
        add_log("info", f"偵測服務已在執行中: {camera.name} (ID={camera_id})", "detection")
        return {"status": "already_running", "message": "偵測服務已在執行中"}
    
    started = _start_detection_service(camera)
    if not started:
        raise HTTPException(status_code=409, detail="偵測服務啟動失敗")
    
    # 更新攝影機狀態
    camera.status = "online"
    camera.detection_enabled = True
    db.commit()
    set_feature_state("detection", camera_id, True)
    add_log("success", f"偵測服務已啟動: {camera.name} (ID={camera_id})", "detection")
    
    return {"status": "started", "message": f"偵測服務已啟動: {camera.name}"}


@router.post("/{camera_id}/detection/stop")
async def stop_detection(camera_id: int, db: Session = Depends(get_db)):
    """停止偵測服務"""
    if camera_id in detection_services:
        detection_services[camera_id]['running'] = False
    
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    camera_name = f"camera_{camera_id}"
    if camera:
        camera_name = camera.name
        camera.detection_enabled = False
        db.commit()
    set_feature_state("detection", camera_id, False)
    add_log("info", f"偵測服務已停止: {camera_name} (ID={camera_id})", "detection")
    
    return {"status": "stopped", "message": "偵測服務已停止"}


@router.get("/{camera_id}/detection/status")
async def detection_status(camera_id: int):
    """取得偵測服務狀態"""
    if camera_id in detection_services:
        v = detection_services[camera_id]
        return {k: vv for k, vv in v.items() if not k.startswith("_")}
    return {"running": False}


@router.get("/detection/all")
async def all_detection_status():
    """取得所有偵測服務狀態"""
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in detection_services.items()}


@router.get("/debug/shared-frames")
async def debug_shared_frames():
    """Debug: 查看 shared_frames 狀態"""
    import time as _t
    result = {}
    for cam_id, sf in _shared_frames.items():
        result[cam_id] = {
            "age_sec": round(_t.time() - sf.get("ts", 0), 1),
            "has_frame": sf.get("frame") is not None,
            "detections_count": len(sf.get("detections", [])),
        }
    # also check thread liveness
    threads = {}
    for cid, info in detection_services.items():
        t = info.get("_thread")
        threads[cid] = {
            "running_flag": info.get("running"),
            "thread_alive": t.is_alive() if t else None,
        }
    return {"shared_frames": result, "threads": threads}


def _start_detection_service(camera: Camera) -> bool:
    if camera.id in detection_services and detection_services[camera.id].get("running"):
        return False
    detection_services[camera.id] = {
        "running": True,
        "started_at": datetime.now().isoformat(),
        "camera_name": camera.name,
    }
    worker = threading.Thread(
        target=run_detection,
        args=(camera.id, resolve_analysis_source(camera), camera.location, camera.detection_config, camera.zones or []),
        daemon=True,
        name=f"detection-{camera.id}",
    )
    detection_services[camera.id]["thread_name"] = worker.name
    detection_services[camera.id]["_thread"] = worker
    worker.start()
    return True


def _schedule_snapshot_warm(camera_id: int, source: str) -> bool:
    """Schedule non-blocking snapshot warm-up for UI thumbnails."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    task = snapshot_warm_tasks.get(camera_id)
    if task is not None and not task.done():
        return False
    snapshot_warm_tasks[camera_id] = loop.create_task(_warm_snapshot(camera_id, source))
    return True


def resume_detection_services() -> dict:
    """API 啟動時恢復上次偵測啟停狀態。"""
    db = SessionLocal()
    resumed = 0
    total = 0
    try:
        cams = db.query(Camera).all()
        for cam in cams:
            total += 1
            # Pre-warm snapshots at boot to reduce first-open placeholder rate.
            if cam.enabled and cam.source:
                _schedule_snapshot_warm(cam.id, cam.source)
            want = get_feature_enabled("detection", cam.id, default=bool(cam.detection_enabled))
            if not want:
                cam.detection_enabled = False
                continue
            if _start_detection_service(cam):
                resumed += 1
            cam.detection_enabled = True
        db.commit()
    finally:
        db.close()
    if resumed:
        add_log("info", f"重啟恢復偵測服務: {resumed}/{total}", "detection")
    return {"total": total, "resumed": resumed}


# Option D 啟用名單：哪些 cam 要把 annotated frame 推 go2rtc 給 WebRTC
_ANNOTATED_STREAM_CAM_IDS = set()  # confirmed annotated_streamer triggers SEGV race


def run_detection(camera_id: int, source: str, location: str, detection_config: dict, zones: list = []):
    """背景偵測任務"""
    import sys
    sys.path.insert(0, '/workspace')
    
    from detection.vehicle_detector import VehicleDetector
    import requests
    import os
    import numpy as np
    from pathlib import Path
    
    output_dir = Path("./output/violations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 錯開各 camera 的 RTSP/HTTP 連線，避免 FFMPEG 同時 open 互相阻塞
    time.sleep(camera_id * 3)
    # 每 cam 獨立 detector instance：避免 4 cams 共享同一 detector 被 lock 序列化
    detector = _get_per_cam_detector(camera_id)
    if detector is None:
        detector = VehicleDetector(conf_threshold=0.5)
    add_log("info", f"cam_{camera_id} 偵測器 device: {getattr(detector, 'runtime_device', 'unknown')}", "detection")
    _src_lc = str(source or "").lower()
    if _src_lc.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;10000000|buffer_size;131072|allowed_media_types;video|analyzeduration;1000000|probesize;1000000"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    speed_kmh_per_pxps = float(detection_config.get("speed_kmh_per_pxps", 0.12) or 0.12)
    speed_smooth_alpha = float(detection_config.get("speed_smooth_alpha", 0.35) or 0.35)
    # track_ttl 從 1.2 → 5 秒：塞車場景偶爾 1 frame detection miss 不會讓 track
    # 被刪除產生新 ID，避免 traffic_event cooldown 被頻繁 reset 導致重複計數
    track_ttl_sec = float(detection_config.get("speed_track_ttl_sec", 5.0) or 5.0)
    tracks = {}
    next_track_id = 1
    
    frame_count = 0
    detection_count = 0
    
    # 取得啟用的偵測類型
    enabled_types = []
    if detection_config.get('red_light'): enabled_types.append(('RED_LIGHT', '闘紅燈', 2700))
    if detection_config.get('speeding'): enabled_types.append(('SPEEDING', '超速', 1800))
    if detection_config.get('illegal_parking'): enabled_types.append(('ILLEGAL_PARKING', '違規停車', 600))
    if detection_config.get('wrong_way'): enabled_types.append(('WRONG_WAY', '逆向行駛', 900))
    
    det_zones = select_zones(zones, scope=SCOPE_TRAFFIC, allowed_types=("detection", "flow_detection"), fallback_scopes=(SCOPE_CONGESTION,))
    speed_zones = select_zones(zones, scope=SCOPE_SPEED, allowed_types=("speed", "speed_roi", "speed_line_in", "speed_line_out"))
    print(
        f"🚀 偵測服務啟動: camera_id={camera_id}, 啟用類型={[t[1] for t in enabled_types]}, "
        f"traffic_roi={len(det_zones)}, speed_roi={len(speed_zones)}"
    )

    def _vehicle_in_any_zone(vehicle: dict, zone_list: list) -> bool:
        if not zone_list:
            return False
        b = vehicle.get("bbox", {}) or {}
        cx = int((b.get("x1", 0) + b.get("x2", 0)) / 2)
        cy = int((b.get("y1", 0) + b.get("y2", 0)) / 2)
        for z in zone_list:
            pts = z.get("points", [])
            if len(pts) >= 3:
                poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    return True
        return False
    def _vehicle_hit_zones(vehicle: dict, zone_list: list) -> list:
        hits = []
        if not zone_list:
            return hits
        b = vehicle.get("bbox", {}) or {}
        cx = int((b.get("x1", 0) + b.get("x2", 0)) / 2)
        cy = int((b.get("y1", 0) + b.get("y2", 0)) / 2)
        for z in zone_list:
            pts = z.get("points", [])
            if len(pts) < 3:
                continue
            poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                hits.append(z)
        return hits
    def _zone_key(zone: dict) -> int:
        return id(zone)
    def _zone_occupancy(zone: dict, vehicle_list: list) -> float | None:
        pts = zone.get("points", [])
        if len(pts) < 3:
            return None
        poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        zone_area = float(cv2.contourArea(poly))
        if zone_area <= 0:
            return None
        vehicle_area = 0.0
        for veh in vehicle_list:
            b = veh.get("bbox", {}) or {}
            cx = int((b.get("x1", 0) + b.get("x2", 0)) / 2)
            cy = int((b.get("y1", 0) + b.get("y2", 0)) / 2)
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) < 0:
                continue
            width = max(0.0, float(b.get("width", 0) or (b.get("x2", 0) - b.get("x1", 0))))
            height = max(0.0, float(b.get("height", 0) or (b.get("y2", 0) - b.get("y1", 0))))
            vehicle_area += width * height
        return min(vehicle_area / zone_area, 1.0)
    def _parse_lane_no(zone: dict):
        raw = zone.get("lane_no")
        if raw is None:
            raw = zone.get("lane_id")
        if raw is None:
            raw = zone.get("name")
        if raw is None:
            return None
        try:
            import re
            m = re.search(r"(\d+)", str(raw))
            if not m:
                return None
            n = int(m.group(1))
            return n if n > 0 else None
        except Exception:
            return None
    def _normalize_event_direction(raw):
        text = str(raw or "").strip()
        if not text:
            return "straight"
        lower = text.lower()
        if lower in ("left", "straight", "right"):
            return lower
        if lower in ("unknown", "-", "--", "n/a"):
            return "straight"
        upper = text.upper()
        if upper in ("IN", "OUT", "N2S", "S2N", "E2W", "W2E"):
            return upper
        if "左" in text:
            return "left"
        if "右" in text:
            return "right"
        if "直" in text:
            return "straight"
        return "straight"
    
    # 拆成兩支 thread：reader 連續讀 frame，worker 跑 YOLO + DB。
    # 關鍵：worker 把「frame 跟 bbox 綁成一組」原子寫入 _shared_frames，
    # overlay 只讀這組綁定 → frame 跟 bbox 永遠同一時間點，bbox 必對齊車輛、無雙時間源。
    _latest = {"frame": None, "ts": 0.0, "stop": False}
    _read_fail_count = [0]

    # 判斷 source 是不是檔案（影響 EOF 處理）
    _src_lc_outer = str(source or "").lower()
    _is_file_source = (
        not _src_lc_outer.startswith(("rtsp://", "http://", "https://"))
        or _src_lc_outer.endswith((".mp4", ".mkv", ".mov", ".avi", ".webm"))
    )

    def _reader_loop():
        nonlocal cap
        while detection_services.get(camera_id, {}).get('running', False) and not _latest["stop"]:
            try:
                ret, frm = cap.read()
            except Exception:
                ret, frm = False, None
            if not ret:
                # 檔案來源 EOF：先試 seek 回開頭（最快），連續失敗就 release+reopen
                # （壞 metadata 的 mkv frame_count 是 garbage，seek 後 decoder state 可能異常）
                if _is_file_source:
                    _read_fail_count[0] += 1
                    if _read_fail_count[0] <= 3:
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        except Exception:
                            pass
                        continue
                    # 連續 fail >3 次 → seek 救不回來，乾淨重 open（decoder fresh state）
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    _read_fail_count[0] = 0
                    continue
                # RTSP/HTTP：連線真的斷了 → reconnect
                _read_fail_count[0] += 1
                if _read_fail_count[0] == 1 or _read_fail_count[0] % 100 == 0:
                    print(f"⚠️ [detection] cam{camera_id} cap.read() failed (count={_read_fail_count[0]}), reconnecting...", flush=True)
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(2)
                if _src_lc_outer.startswith("rtsp://"):
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;10000000|buffer_size;131072|allowed_media_types;video|analyzeduration;1000000|probesize;1000000"
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                continue
            _read_fail_count[0] = 0
            _latest["frame"] = frm
            _latest["ts"] = time.time()
            if camera_id in _ANNOTATED_STREAM_CAM_IDS:
                try:
                    from detection.annotated_streamer import push_frame as _push_f
                    _push_f(camera_id, frm)
                except Exception:
                    pass
            # file source 限速到實時播放（~30 fps）。RTSP/HTTP 自然由 server 推送速度限速；
            # file source 無限速會 burst decode（1080p 200+ fps）把 CPU 燒爆，導致網頁卡頓。
            if _is_file_source:
                time.sleep(0.033)

    threading.Thread(target=_reader_loop, daemon=True, name=f"reader-{camera_id}").start()

    # 以下是 _process_post_yolo 函式體（原本內聯在 worker 迴圈裡，現在抽出來）
    def _process_post_yolo(infer_frame, detections, cur_ts):
        nonlocal next_track_id, detection_count
        vehicles = [d for d in detections if d['class_name'] in ['car', 'motorcycle', 'truck', 'bus', 'heavy_truck', 'light_truck']]
        
        # ROI 區域過濾：只保留中心點在偵測區域內的車輛
        if vehicles and det_zones:
            if det_zones:
                filtered = []
                for v in vehicles:
                    b = v['bbox']
                    cx, cy = (b['x1'] + b['x2']) // 2, (b['y1'] + b['y2']) // 2
                    for z in det_zones:
                        pts = z.get('points', [])
                        if len(pts) >= 3:
                            poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                                filtered.append(v)
                                break
                vehicles = filtered

        # 車速 ROI：保留給超速判定，同時背景任務也會估算 speed_kmh 寫入 traffic_events
        speed_zone_vehicles = []
        if vehicles and speed_zones:
            if speed_zones:
                for v in vehicles:
                    b = v['bbox']
                    cx, cy = (b['x1'] + b['x2']) // 2, (b['y1'] + b['y2']) // 2
                    for z in speed_zones:
                        pts = z.get('points', [])
                        if len(pts) >= 3:
                            poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                                speed_zone_vehicles.append(v)
                                break

        # 背景速度估算（與 live-overlay 同邏輯）
        if vehicles:
                    now_ts = time.time()
                    stale_ids = [tid for tid, tr in tracks.items() if (now_ts - tr.get("t", now_ts)) > track_ttl_sec]
                    for tid in stale_ids:
                        tracks.pop(tid, None)
                    # P1: 找有 calibration 的 speed_zone (拿第一個)
                    _calib_zone = None
                    for _z in (speed_zones or []):
                        if _z and isinstance(_z, dict) and _z.get("calibration"):
                            _calib_zone = _z
                            break
                    _H = _get_zone_homography(_calib_zone) if _calib_zone else None
                    # P3: 速度平穩化 — N=5 滑動窗口 + median 多 sample + outlier reject
                    # 3 個參數從 hard-code 改成 detection_config 可調 (web UI 車速設定頁)
                    _SPEED_WINDOW = 5      # 保留最近 5 個 (t, x, y) sample
                    _SPEED_MIN_SAMPLES = int(detection_config.get("speed_min_samples", 5) or 5)
                    _SPEED_OUTLIER_FACTOR = float(detection_config.get("speed_outlier_factor", 2.0) or 2.0)
                    _SPEED_ABS_CAP = float(detection_config.get("speed_abs_cap", 130.0) or 130.0)
                    for v in vehicles:
                        b = v.get("bbox", {}) or {}
                        cx = int((b.get("x1", 0) + b.get("x2", 0)) / 2)
                        cy = int((b.get("y1", 0) + b.get("y2", 0)) / 2)
                        cls = str(v.get("class_name") or "")
                        track_id = _nearest_track_id((cx, cy), cls, tracks)
                        if track_id is None:
                            track_id = next_track_id
                            next_track_id += 1
                            tracks[track_id] = {"center": (cx, cy), "t": now_ts, "class_name": cls, "speed_kmh": None, "samples": [], "world_xy": None, "frames": 0, "kalman": None, "tw_samples": []}
                        prev = tracks.get(track_id, {})
                        prev_frames = int(prev.get("frames", 0) or 0)
                        prev_speed = prev.get("speed_kmh") or 0.0
                        # 接地點：bbox 底中 (P10 per-class offset)
                        _ground_x, _ground_y = _bbox_bottom_center(b, cls)
                        if _ground_x is None:
                            _ground_x, _ground_y = cx, cy
                        world_xy = _pixel_to_world_m(_H, _ground_x, _ground_y) if _H is not None else None
                        # sample = (timestamp, x, y) 在 world m (若校正) 或 pixel 空間
                        if world_xy is not None:
                            sample = (now_ts, float(world_xy[0]), float(world_xy[1]))
                            sample_unit = "world_m"
                        else:
                            sample = (now_ts, float(_ground_x), float(_ground_y))
                            sample_unit = "pixel"
                        # B: ID swap 偵測 — 1 frame 內 bbox 中心 pixel 跳 > 150px (~約 180 km/h
                        # 等級的位移) 視為 tracker 把同一 track_id 分配給新車輛，舊 samples
                        # 已不可信，清空重新累積。
                        prev_center_xy = prev.get("center")
                        if prev_center_xy is not None and prev_frames > 0:
                            _dx = cx - prev_center_xy[0]
                            _dy = cy - prev_center_xy[1]
                            if (_dx * _dx + _dy * _dy) ** 0.5 > 150.0:
                                # ID swap：reset 速度 state + traffic_event cooldown 紀錄
                                prev_frames = 0
                                prev_speed = 0.0
                                prev = dict(prev)
                                prev["samples"] = []
                                prev["kalman"] = None
                                prev["_event_log_ts"] = {}  # 不然新車繼承舊 cooldown 不寫入
                        samples = list(prev.get("samples") or [])
                        samples.append(sample)
                        if len(samples) > _SPEED_WINDOW:
                            samples.pop(0)
                        # P11: Kalman filter on world coords (if calibrated) gives smoother velocity
                        speed_kmh = None
                        if world_xy is not None:
                            kf = prev.get("kalman") or _KalmanCV()
                            vx, vy = kf.update(world_xy[0], world_xy[1], now_ts)
                            v_ms = (vx * vx + vy * vy) ** 0.5
                            kalman_kmh = v_ms * 3.6
                            tracks[track_id]["kalman"] = kf
                            if (prev_frames + 1) >= _SPEED_MIN_SAMPLES:
                                # A: Kalman 路徑也套 abs_cap + outlier reject
                                # (之前只 clamp 150，飆值會直接記成 150 寫進 DB)
                                if kalman_kmh > _SPEED_ABS_CAP:
                                    speed_kmh = prev_speed if prev_speed > 5 else None
                                elif prev_speed > 5 and kalman_kmh > prev_speed * _SPEED_OUTLIER_FACTOR + 30:
                                    speed_kmh = prev_speed
                                else:
                                    speed_kmh = max(0.0, min(150.0, kalman_kmh))
                        if speed_kmh is None and len(samples) >= _SPEED_MIN_SAMPLES:
                            t0, x0, y0 = samples[0]
                            t1, x1, y1 = samples[-1]
                            dt_total = max(1e-3, t1 - t0)
                            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                            if sample_unit == "world_m":
                                raw_kmh = (dist / dt_total) * 3.6
                            else:
                                raw_kmh = (dist / dt_total) * speed_kmh_per_pxps
                            # absolute cap：raw_kmh > 130 視為 ID tracker 跳動 / bbox 抖動偽值，
                            # 不論有沒有 prev_speed 都直接 drop (用 prev 或 None)
                            if raw_kmh > _SPEED_ABS_CAP:
                                speed_kmh = prev_speed if prev_speed > 5 else None
                            elif prev_speed > 5 and raw_kmh > prev_speed * _SPEED_OUTLIER_FACTOR + 30:
                                # outlier reject：突然飆高超過合理範圍 → 用 prev_speed
                                speed_kmh = prev_speed
                            else:
                                # median of pairwise instantaneous speeds across consecutive samples
                                inst_speeds = []
                                for i in range(1, len(samples)):
                                    pt_t, pt_x, pt_y = samples[i-1]
                                    nt_t, nt_x, nt_y = samples[i]
                                    _dt = max(1e-3, nt_t - pt_t)
                                    _dist = ((nt_x - pt_x) ** 2 + (nt_y - pt_y) ** 2) ** 0.5
                                    if sample_unit == "world_m":
                                        inst_speeds.append((_dist / _dt) * 3.6)
                                    else:
                                        inst_speeds.append((_dist / _dt) * speed_kmh_per_pxps)
                                inst_speeds.sort()
                                median_kmh = inst_speeds[len(inst_speeds) // 2]
                                # 取 raw (window-based) 跟 median (instantaneous median) 平均，更穩
                                speed_kmh = (raw_kmh + median_kmh) / 2.0
                            # 同樣 clamp 從 220 → 150
                            if speed_kmh is not None:
                                speed_kmh = max(0.0, min(150.0, float(speed_kmh)))
                        # 用 update 而非 replace — 保留 _event_log_ts (cooldown), kalman,
                        # tw_samples, tw_speed_kmh 等 state，否則每 frame 都被丟掉，
                        # cooldown / trip_wire median 都會失效
                        _existing = tracks.get(track_id) or {}
                        _existing.update({
                            "center": (cx, cy),
                            "t": now_ts,
                            "class_name": cls,
                            "speed_kmh": speed_kmh,
                            "samples": samples,
                            "world_xy": world_xy,
                            "frames": prev_frames + 1,
                        })
                        tracks[track_id] = _existing
                        v["track_id"] = track_id
                        # 優先用 trip_wire 持續值（如果這條 track 已測過 trip wire）
                        _persisted_tw = prev.get("tw_speed_kmh")
                        if isinstance(_persisted_tw, (int, float)) and _persisted_tw > 0:
                            v["speed_kmh"] = _persisted_tw
                            v["speed_method"] = "trip_wire"
                            v["speed_calibrated"] = True
                            tracks[track_id]["tw_speed_kmh"] = _persisted_tw  # 沿襲下去
                        else:
                            v["speed_kmh"] = speed_kmh if (speed_kmh is not None and (prev_frames + 1) >= _SPEED_MIN_SAMPLES) else None
                            v["speed_calibrated"] = world_xy is not None

                        # P4: Trip wire 跨線測速（用 bbox 底中 pixel 座標）
                        _tw_pairs = _find_trip_wire_pair(speed_zones)
                        if _tw_pairs:
                            curr_pt = (float(_ground_x), float(_ground_y))
                            prev_pt = prev.get("center_bottom_prev")
                            cross_state = dict(prev.get("trip_wire") or {})
                            for lane_key, in_z, out_z, dist_m in _tw_pairs:
                                if prev_pt is None:
                                    continue
                                in_pts = in_z.get("points") or []
                                out_pts = out_z.get("points") or []
                                if len(in_pts) >= 2 and _line_crossed(prev_pt, curr_pt, in_pts):
                                    if cross_state.get(f"in_{lane_key}") is None:
                                        cross_state[f"in_{lane_key}"] = now_ts
                                if len(out_pts) >= 2 and _line_crossed(prev_pt, curr_pt, out_pts):
                                    if cross_state.get(f"out_{lane_key}") is None:
                                        cross_state[f"out_{lane_key}"] = now_ts
                                t_in = cross_state.get(f"in_{lane_key}")
                                t_out = cross_state.get(f"out_{lane_key}")
                                if t_in and t_out:
                                    dt_cross = abs(t_out - t_in)
                                    if 0.05 <= dt_cross <= 30.0:
                                        tw_speed_kmh = (dist_m / dt_cross) * 3.6
                                        if 1.0 <= tw_speed_kmh <= 220.0:
                                            # P9: 累積到 tw_samples，取 median 為最終 speed
                                            tw_samples = list(tracks[track_id].get("tw_samples") or [])
                                            tw_samples.append(tw_speed_kmh)
                                            if len(tw_samples) > 5:
                                                tw_samples.pop(0)
                                            tw_samples_sorted = sorted(tw_samples)
                                            tw_median = tw_samples_sorted[len(tw_samples_sorted) // 2]
                                            tracks[track_id]["tw_samples"] = tw_samples
                                            tracks[track_id]["tw_speed_kmh"] = tw_median
                                            v["speed_kmh"] = tw_median
                                            v["speed_method"] = "trip_wire_median" if len(tw_samples) > 1 else "trip_wire"
                                            v["speed_calibrated"] = True
                                            v["tw_sample_count"] = len(tw_samples)
                                            print(f"[trip_wire] cam={camera_id} track={track_id} lane={lane_key} {dist_m}m in {dt_cross:.3f}s -> raw {tw_speed_kmh:.1f} km/h, median(n={len(tw_samples)})={tw_median:.1f}", flush=True)
                                    cross_state[f"in_{lane_key}"] = None
                                    cross_state[f"out_{lane_key}"] = None
                            tracks[track_id]["trip_wire"] = cross_state
                            tracks[track_id]["center_bottom_prev"] = curr_pt
                        else:
                            tracks[track_id]["center_bottom_prev"] = (float(_ground_x), float(_ground_y))
                
        if vehicles and det_zones:
            detection_count += 1

            # 更新服務狀態
            detection_services[camera_id]['detections'] = detection_count
            detection_services[camera_id]['last_detection'] = datetime.now().isoformat()
            db = None
            row_to_vehicle = []  # 平行 list 用於存截圖時對應 bbox
            try:
                db = SessionLocal()
                rows = []
                zone_occupancy_map = {
                    _zone_key(z): _zone_occupancy(z, vehicles)
                    for z in det_zones
                } if det_zones else {}
                # 防重複計數：同 track 在同一 zone 30 秒內只算 1 筆 traffic_event。
                # 之前每 frame 寫 → 塞車車輛 30 秒沒動 = 30s × 10fps = 300 筆同車。
                # 30s 對快速通過 ROI 仍是 1 車 1 筆；塞車中 30s 也只 1 筆 (ID swap
                # 偵測會清狀態，新 track 視為新車重新算)。
                _EVENT_LOG_COOLDOWN = 30.0
                for v in vehicles:
                    bbox = v.get("bbox", {}) or {}
                    hit_zones = _vehicle_hit_zones(v, det_zones)
                    if not hit_zones:
                        continue
                    pick_zone = hit_zones[0]
                    # cooldown check — 跳過同 track 同 zone 已記錄過且未冷卻完的
                    _track_id = v.get("track_id")
                    if _track_id is not None:
                        _zone_log_key = str(pick_zone.get("name") or pick_zone.get("id") or "")
                        _track_state = tracks.setdefault(_track_id, {})
                        _log_state = _track_state.setdefault("_event_log_ts", {})
                        _last = _log_state.get(_zone_log_key, 0.0)
                        if now_ts - _last < _EVENT_LOG_COOLDOWN:
                            continue
                        _log_state[_zone_log_key] = now_ts
                    occupancy_val = zone_occupancy_map.get(_zone_key(pick_zone)) if pick_zone else None
                    speed_raw = v.get("speed_kmh")
                    speed_method = str(v.get("speed_method") or "")
                    try:
                        speed_num = float(speed_raw)
                    except Exception:
                        speed_num = None
                    # Drop saturated 飽和值（>= 200 表 sanity gate clamp，是 ID tracker
                    # 跳動造成的 pixel jump 偽值）。Drop 信任度低的 raw pixel-based
                    # method (沒有 calibration / kalman / trip_wire 校正) — 那些容易
                    # 飆到 100+ km/h，跟 trip_wire 觀察到的真實塞車速度 (3-15) 矛盾。
                    if speed_num is None or speed_num <= 0:
                        speed_val = None
                    elif speed_num >= 150.0:  # 飽和不可信 (clamp ceiling = 150)
                        speed_val = None
                    elif speed_method in ("trip_wire", "trip_wire_median", "kalman"):
                        # 校正過的 method，相對可信
                        speed_val = speed_num
                    else:
                        # 純 pixel-based，限制 < 100 km/h 才可信 (高速車流會被 trip_wire 補上)
                        speed_val = speed_num if speed_num < 100.0 else None
                    row = TrafficEvent(
                        camera_id=int(camera_id),
                        label=str(v.get("class_name", "unknown")).lower(),
                        speed_kmh=speed_val,
                        occupancy=occupancy_val,
                        lane_no=_parse_lane_no(pick_zone),
                        direction=_normalize_event_direction(pick_zone.get("direction")),
                        entered_zones=[str(z.get("name") or "") for z in hit_zones if str(z.get("name") or "")],
                        bbox=[bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")],
                        source="roi_detection",
                    )
                    rows.append(row)
                    row_to_vehicle.append(v)
                if rows:
                    db.add_all(rows)
                    db.commit()
                    # MQTT publish: 對齊 toggle（MQTT bridge mode != off 才發）
                    try:
                        from services.mqtt_bridge import bridge as _mqtt
                        if _mqtt.connected() and _mqtt.settings.get("mode") != "off" and rows and row_to_vehicle:
                            base = _mqtt.settings.get("base_topic") or "traffic"
                            # 節流：每 cam 每 1 秒最多發一筆（避免 MQTT 洪水），取第一筆代表
                            last = getattr(_mqtt, "_last_event_ts", {}) or {}
                            if (cur_ts - last.get(camera_id, 0.0)) >= 1.0:
                                row = rows[0]; v = row_to_vehicle[0]
                                _mqtt.publish(f"{base}/event/cam_{camera_id}", {
                                    "event_id": int(row.id),
                                    "camera_id": camera_id,
                                    "label": row.label,
                                    "speed_kmh": row.speed_kmh,
                                    "lane_no": row.lane_no,
                                    "direction": row.direction,
                                    "zones": row.entered_zones,
                                    "bbox": row.bbox,
                                    "timestamp": cur_ts,
                                })
                                last[camera_id] = cur_ts
                                _mqtt._last_event_ts = last
                    except Exception:
                        pass
                    # 事件截圖：要對應 cam 在 Frigate config 開「事件截圖」(snapshots.enabled) 才存
                    # 沒開的 cam (例如只設定全時錄影) → 不存事件截圖
                    if _is_snapshot_enabled_for_cam(camera_id):
                        try:
                            import os as _os
                            SNAP_DIR = "/tmp/event_snapshots"
                            _os.makedirs(SNAP_DIR, exist_ok=True)
                            last_snap_ts = _per_cam_last_snap_ts.get(camera_id, 0.0)
                            if (cur_ts - last_snap_ts) >= 2.0 and rows and row_to_vehicle:
                                row = rows[0]; v = row_to_vehicle[0]
                                eid = int(row.id)
                                if eid > 0:
                                    path = f"{SNAP_DIR}/{eid}.jpg"
                                    snap = infer_frame.copy()
                                    fh, fw = snap.shape[:2]
                                    bbox = v.get("bbox", {}) or {}
                                    if bbox:
                                        cv2.rectangle(snap, (int(bbox.get("x1",0)), int(bbox.get("y1",0))),
                                                      (int(bbox.get("x2",0)), int(bbox.get("y2",0))), (0,200,0), 2)
                                    if fw > 480:
                                        sc = 480.0/fw
                                        snap = cv2.resize(snap, (480, int(fh*sc)))
                                    cv2.imwrite(path, snap, [cv2.IMWRITE_JPEG_QUALITY, 60])
                                    _per_cam_last_snap_ts[camera_id] = cur_ts
                        except Exception:
                            pass
                db.close()
            except Exception:
                try:
                    if db is not None:
                        db.close()
                except Exception:
                    pass

            # 每 50 次偵測記錄一次違規 (模擬)
            # P0: real SPEEDING only - no random fakes
            if detection_config.get('speeding') and speed_zones and speed_zone_vehicles:
                _speed_limit_kmh = float(detection_config.get("speed_limit", 50) or 50)
                _overspeed_threshold_kmh = 10.0
                _zone_cfg = speed_zones[0] or {}
                _speed_limit_kmh = float(_zone_cfg.get("speed_limit") or _speed_limit_kmh or 50)
                _overspeed_threshold_kmh = float(
                    _zone_cfg.get("overspeed_kmh")
                    or _zone_cfg.get("speed_margin")
                    or _overspeed_threshold_kmh
                )
                _effective_limit = _speed_limit_kmh + _overspeed_threshold_kmh
                global _violation_dedup
                try:
                    _violation_dedup
                except NameError:
                    _violation_dedup = {}
                _now_t = time.time()
                _DEDUP_WINDOW = 5.0
                for _v in speed_zone_vehicles:
                    _raw_speed = _v.get("speed_kmh")
                    if not isinstance(_raw_speed, (int, float)) or _raw_speed <= 0:
                        continue
                    if not (5.0 <= _raw_speed <= 200.0):
                        continue
                    _tid = _v.get("track_id")
                    _track_state = tracks.get(_tid) if _tid is not None else None
                    if _track_state is None:
                        continue
                    _track_frames = int(_track_state.get("frames", 0) or 0)
                    if _track_frames < 5:
                        continue
                    _has_tw = bool(_track_state.get("tw_speed_kmh"))
                    if not _has_tw and _track_frames < 8:
                        continue
                    if _raw_speed < _effective_limit:
                        continue
                    # 飽和不可信 (clamp ceiling = 150): TrafficEvent 寫入已過濾 (stream.py:1881
                    # 將 >=150 設 None)，Violation 線之前漏掉同樣過濾 → speed 估算飆高的 case
                    # 全被當「超速 150 km/h」誤開單。此處補上一致過濾。
                    if _raw_speed >= 150.0:
                        continue
                    _track_id = _v.get("track_id")
                    if _track_id is None:
                        _bb = _v.get("bbox", {}) or {}
                        _track_id = (int((_bb.get("x1", 0) + _bb.get("x2", 0)) / 2),
                                     int((_bb.get("y1", 0) + _bb.get("y2", 0)) / 2))
                    _dedup_key = (camera_id, _track_id, 'SPEEDING')
                    if (_now_t - _violation_dedup.get(_dedup_key, 0.0)) < _DEDUP_WINDOW:
                        continue
                    _violation_dedup[_dedup_key] = _now_t
                    _speed_kmh = float(_raw_speed)
                    _overspeed_kmh = max(0.0, _speed_kmh - _speed_limit_kmh)
                    _bbox = _v.get('bbox', {}) or {}
                    # plate-vehicle association: 對該違規 vehicle bbox 即時跑 plate
                    # detection + OCR，嚴格綁定觸發車輛 (不再用 _latest_lpr_plate ±20s
                    # 抓最近 LPR 結果可能是前車)。crop 暫存待 POST 拿 vid 後寫盤。
                    _plate = ""
                    _violation_plate_crop = None  # numpy array (BGR)，POST 後 save 用
                    if _bbox:
                        try:
                            _bx1 = max(0, int(_bbox.get('x1', 0)))
                            _by1 = max(0, int(_bbox.get('y1', 0)))
                            _bx2 = int(_bbox.get('x2', 0))
                            _by2 = int(_bbox.get('y2', 0))
                            _veh_crop = frame[_by1:_by2, _bx1:_bx2]
                            if _veh_crop is not None and _veh_crop.size > 0:
                                # 用跟 LPR 同樣 pipeline 抓 plate (真 YOLO plate detector
                                # + expand + tighten)，而非 sliding-window heuristic
                                from api.routes.lpr_stream import (
                                    get_plate_detector as _vpgd,
                                    get_recognizer as _vpr,
                                    _PLATE_DETECT_CONF as _vpconf,
                                    _expand_plate_bbox as _vpex,
                                    _tighten_plate_crop_with_bbox as _vptight,
                                    _rank_plate_candidates as _vprank,
                                    _propose_plate_bboxes as _vpfb,
                                    _recognize_plate_on_crop as _vpo,
                                )
                                _det = _vpgd()
                                _rec = _vpr()
                                _vh, _vw = _veh_crop.shape[:2]
                                _detections = _det.detect(_veh_crop, conf=_vpconf)
                                if not _detections:
                                    _detections = _vpfb(_veh_crop)  # fallback heuristic
                                _ranked = _vprank(_detections, _vw, _vh)
                                for _r in _ranked:
                                    _rb = _r.get('bbox') or []
                                    if len(_rb) != 4:
                                        continue
                                    _ex1, _ey1, _ex2, _ey2 = _vpex(
                                        [int(x) for x in _rb], _vw, _vh
                                    )
                                    _pc = _det.crop(_veh_crop, [_ex1, _ey1, _ex2, _ey2])
                                    if _pc is None or getattr(_pc, 'size', 0) == 0:
                                        continue
                                    # tighten 收緊邊界供 OCR (準確率高)
                                    # save 用「更寬鬆」pad (上下 55%, 左右 30%) 重 crop 一次,
                                    # 因為 detector plate bbox 常常只抓「字寬度」漏掉上下框框,
                                    # 25% pad 救不回 — user 報「車牌裁切一半」就是這個 (5df9f45 沒解徹底)
                                    _es1, _et1, _es2, _et2 = _vpex(
                                        [int(x) for x in _rb], _vw, _vh,
                                        pad_x_ratio=0.30, pad_y_ratio=0.55,
                                    )
                                    _pc_for_save = _det.crop(_veh_crop, [_es1, _et1, _es2, _et2])
                                    if _pc_for_save is None or getattr(_pc_for_save, 'size', 0) == 0:
                                        _pc_for_save = _pc.copy()  # fallback 用小 pad 版
                                    _pc_tight, _ = _vptight(_pc)
                                    if _pc_tight is not None and getattr(_pc_tight, 'size', 0) > 0:
                                        _pc_for_ocr = _pc_tight
                                    else:
                                        _pc_for_ocr = _pc
                                    _res = _vpo(_pc_for_ocr, _rec)
                                    _pn = (_res or {}).get('plate_number')
                                    _conf = float((_res or {}).get('confidence') or 0.0)
                                    if _pn and _conf >= 0.5:
                                        _plate = str(_pn).strip()
                                        _violation_plate_crop = _pc_for_save
                                        break
                        except Exception as _ex_lpv:
                            print(f"⚠️ plate-vehicle association err cam{camera_id}: {_ex_lpv}", flush=True)
                    # 嚴格 plate detection 沒抓到 → license_plate = None
                    # (寧可沒車牌也不要 fallback _latest_lpr_plate ±20s 抓前車誤標。
                    # user 之前抱怨 Honda 配 Hyundai / AZ-1122 對不上正是這個)
                    _annotated = frame.copy()
                    if _bbox:
                        cv2.rectangle(_annotated, (int(_bbox.get('x1', 0)), int(_bbox.get('y1', 0))),
                                      (int(_bbox.get('x2', 0)), int(_bbox.get('y2', 0))), (0, 255, 0), 2)
                    _ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    _image_name = f"SPEEDING_{_ts_str}.jpg"
                    _image_path = output_dir / _image_name
                    cv2.imwrite(str(_image_path), _annotated)
                    _data = {
                        "violation_type": 'SPEEDING',
                        "violation_name": '超速',
                        "vehicle_type": _v.get('class_name'),
                        "license_plate": _plate or None,
                        "location": location,
                        "camera_id": camera_id,
                        "confidence": _v.get('confidence'),
                        "fine_amount": 1800,
                        "points": 1,
                        "image_path": f"/files/violations/{_image_name}",
                        "bbox": _v.get("bbox"),
                        "speed_kmh": round(_speed_kmh, 1),
                        "speed_limit_kmh": round(_speed_limit_kmh, 1),
                        "overspeed_kmh": round(_overspeed_kmh, 1),
                        "flow_roi_hit": _vehicle_in_any_zone(_v, det_zones),
                        "speed_roi_hit": True,
                    }
                    try:
                        _resp = requests.post("http://localhost:8000/api/violations", json=_data, timeout=5)
                        _plate_text = _plate or "NODATA(LPR)"
                        print(f"⚠️ real SPEEDING: {_plate_text} {_data['speed_kmh']}km/h (limit {_data['speed_limit_kmh']})")
                        try:
                            _vid = (_resp.json() or {}).get("id") if _resp.ok else None
                            if _vid:
                                # 方案 C: ring buffer save 4 frame，mid_a 加綠框 + plate 標記
                                _save_violation_4frames_async(
                                    camera_id, int(_vid), _now_t, frame,
                                    vehicle_bbox=_bbox, plate_text=_plate,
                                )
                                # plate-vehicle association: save 嚴格綁定觸發車輛的 plate crop
                                if _violation_plate_crop is not None and _violation_plate_crop.size > 0:
                                    try:
                                        from pathlib import Path as _PP
                                        _vp_dir = _PP("./output/violations/snapshots").resolve()
                                        _vp_dir.mkdir(parents=True, exist_ok=True)
                                        cv2.imwrite(
                                            str(_vp_dir / f"{int(_vid)}_violation_plate.png"),
                                            _violation_plate_crop,
                                            [cv2.IMWRITE_PNG_COMPRESSION, 1],
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    except Exception:
                        pass

            # ---- 停車類違規通用 evaluator (共通邏輯解 10 種違規) ----
            # 條件：vehicle 在 parking zone (no_parking/sidewalk/crosswalk/bus_stop
            # /red_line/yellow_line) 內靜止超過 zone.stop_threshold_sec
            # 走跟 SPEEDING 同一條 _emit_violation_for_vehicle pipeline:
            # plate-vehicle association + composite + OSD + ring buffer save
            try:
                # parking zone 預期 scope = traffic_flow_settings,但舊版前端
                # 透過「停車違規設定」mode 寫入時 scope 誤設為 parking_violation_settings
                # → fallback 也吃這個 scope 容錯
                _parking_zones = select_zones(
                    zones, scope=SCOPE_TRAFFIC,
                    allowed_types=(
                        "no_parking", "sidewalk", "crosswalk",
                        "bus_stop", "red_line", "yellow_line",
                    ),
                    fallback_scopes=("parking_violation_settings",),
                )
                if _parking_zones and vehicles:
                    _peval = _PARKING_EVALUATORS.get(camera_id)
                    if _peval is None:
                        from detection.parking_violation import ParkingEvaluator
                        _peval = ParkingEvaluator(camera_id)
                        _PARKING_EVALUATORS[camera_id] = _peval
                    _pk_now = time.time()
                    _triggers = _peval.feed(vehicles, _pk_now, _parking_zones)
                    for _trig in _triggers:
                        try:
                            _vid_pk, _plate_pk = _emit_violation_for_vehicle(
                                camera_id=camera_id,
                                location=location,
                                frame=frame,
                                vehicle_bbox=_trig.get("vehicle_bbox") or {},
                                vehicle_class=_trig.get("vehicle_class") or "car",
                                vehicle_conf=_trig.get("vehicle_conf"),
                                violation_type=_trig["violation_type"],
                                violation_name=_trig["violation_name"],
                                fine_amount=int(_trig.get("fine_amount") or 600),
                                points=int(_trig.get("points_penalty") or 0),
                                output_dir=output_dir,
                                trigger_ts=_pk_now,
                                extra_fields={
                                    "flow_roi_hit": True,
                                    "speed_roi_hit": False,
                                },
                            )
                            print(
                                f"⚠️ {_trig['violation_type']}: zone={_trig['zone_name']!r} "
                                f"track={_trig['track_id']} dwell={_trig['dwell_sec']:.1f}s "
                                f"plate={_plate_pk or 'NODATA'} cam{camera_id}",
                                flush=True,
                            )
                        except Exception as _ex_pk:
                            print(f"⚠️ parking emit cam{camera_id}: {_ex_pk}", flush=True)
            except Exception as _ex_peval:
                print(f"⚠️ parking evaluator cam{camera_id}: {_ex_peval}", flush=True)

            # 視覺 track snapshot — 給 sensor_fusion router 拉。寫 module-level dict。
            # 優先用 world coord (走過 calibration 的攝影機),沒設 calibration 用
            # bbox bottom-center pixel / 100 fallback 當虛擬 world (demo 用,真雷達
            # 接入前 zone 沒 calibration 也能跑 fusion 算法看效果)。
            try:
                _snap = []
                for _vv in vehicles or []:
                    _tid = _vv.get("track_id")
                    if _tid is None:
                        continue
                    _tr = tracks.get(_tid) or {}
                    _wxy = _tr.get("world_xy")
                    if _wxy:
                        _wx, _wy = float(_wxy[0]), float(_wxy[1])
                        _calibrated = True
                    else:
                        # fallback: bbox bottom-center pixel / 100 ≈ 假設 1m=100px
                        _bb = _vv.get("bbox") or {}
                        _bx = (float(_bb.get("x1", 0)) + float(_bb.get("x2", 0))) / 200.0
                        _by = float(_bb.get("y2", 0)) / 100.0
                        _wx, _wy = _bx, _by
                        _calibrated = False
                    _kf = _tr.get("kalman")
                    if _kf is not None and getattr(_kf, "vx", None) is not None:
                        _vxs, _vys = float(getattr(_kf, "vx", 0.0)), float(getattr(_kf, "vy", 0.0))
                    else:
                        _spd = float(_vv.get("speed_kmh") or 0.0) / 3.6
                        _vxs, _vys = _spd, 0.0
                    _snap.append({
                        "track_id": int(_tid),
                        "world_x": _wx, "world_y": _wy,
                        "vx": _vxs, "vy": _vys,
                        "class_name": str(_vv.get("class_name") or ""),
                        "confidence": float(_vv.get("confidence") or 0.0),
                        "bbox": _vv.get("bbox"),
                        "timestamp": float(_tr.get("t") or 0.0),
                        "calibrated": _calibrated,
                    })
                _VEHICLE_TRACK_SNAPSHOTS[int(camera_id)] = _snap
            except Exception as _ex_snap:
                print(f"⚠️ track snapshot cam{camera_id}: {_ex_snap}", flush=True)

    # 後處理 thread：drain queue，跑 ROI/tracking/DB/violations。不阻塞 worker。
    import queue as _queue
    _post_q: _queue.Queue = _queue.Queue(maxsize=4)

    def _post_loop():
        while detection_services.get(camera_id, {}).get('running', False):
            try:
                item = _post_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            if item is None:
                break
            try:
                _process_post_yolo(*item)
            except Exception as _ex:
                print(f"⚠️ [post-process] cam{camera_id}: {_ex}", flush=True)

    threading.Thread(target=_post_loop, daemon=True, name=f"post-{camera_id}").start()

    # worker: 拿最新 frame 跑 YOLO，把 frame + detections 一起寫入 shared_frames，後處理丟給 queue
    # 限制每 cam 最多 10 FPS 偵測 → 4 cam × 10 = 40 inferences/s，降 GPU 負載（監控夠用）
    _last_proc_ts = 0.0
    _last_infer_wall = 0.0
    MAX_INFER_FPS = 10.0
    _min_infer_interval = 1.0 / MAX_INFER_FPS
    _frigate_fb_last = 0.0  # frigate fallback throttle (worker-local，不影響 reader_loop)
    while detection_services.get(camera_id, {}).get('running', False):
        cur_ts = _latest.get("ts", 0.0)
        frame = _latest.get("frame")
        # frigate latest.jpg fallback：(a) _latest["frame"] 從沒被填 (None) 或 (b) frame
        # 已 stale > 2s（reader_loop cap.read 失敗中）時主動 poll frigate，throttle 0.2s。
        # 檔案 source 與正常 RTSP 持續更新 _latest，stale 永遠 < 2s 所以不會觸發。
        _stale = cur_ts > 0 and (time.time() - cur_ts) > 2.0
        if frame is None or _stale:
            _now_fb = time.time()
            if _now_fb - _frigate_fb_last > 0.2:
                _frigate_fb_last = _now_fb
                try:
                    _fb_jpg = _try_frigate_snapshot(source, camera_id=camera_id)
                    if _fb_jpg:
                        _arr = np.frombuffer(_fb_jpg, dtype=np.uint8)
                        _dec = cv2.imdecode(_arr, cv2.IMREAD_COLOR)
                        if _dec is not None and _dec.size > 0:
                            _latest["frame"] = _dec
                            _latest["ts"] = _now_fb
                            continue
                except Exception:
                    pass
        if frame is None or cur_ts <= _last_proc_ts:
            time.sleep(0.01)
            continue
        # 節流：距離上次 inference 不到 100ms 就再等一下
        now_wall = time.time()
        since_last = now_wall - _last_infer_wall
        if since_last < _min_infer_interval:
            time.sleep(_min_infer_interval - since_last)
        _last_proc_ts = cur_ts
        _last_infer_wall = time.time()
        frame_count += 1
        infer_frame = frame.copy()
        # 加回 process-level lock — 雖然每 cam 獨立 detector，但底層 ultralytics /
        # cuDNN / TRT global state (default CUDA stream / cuBLAS handle / cuDNN
        # workspace) 並非 thread-safe；多 cam 並行 detect() 會觸發 native lib race → SEGV。
        # 之前 1e3330a 為拚 fps 移掉，但代價是 process SEGV 持續發生（faulthandler 抓到
        # ultralytics predictor.py preprocess 在 SEGV 時段）。
        # fps 影響：cam_1 13.3→7.5, cam_2 9.2→7.3 — 換穩定。
        with _shared_overlay_detector_lock:
            detections = detector.detect(infer_frame)
        # 原子綁定：frame + bbox + ts 一起寫入
        _shared_frames[camera_id] = {
            "frame": infer_frame,
            "detections": detections,
            "ts": cur_ts,
        }
        # push 進違規 ring buffer (限 10fps + 縮圖 1280x720，方案 C 100% 命中 4 frame)
        try:
            _push_violation_ring(camera_id, infer_frame)
        except Exception:
            pass
        try:
            _post_q.put_nowait((infer_frame, list(detections), cur_ts))
        except _queue.Full:
            pass
        # Option D v3: worker 只更新 dets，畫面由 reader 30 fps push
        if camera_id in _ANNOTATED_STREAM_CAM_IDS:
            try:
                from detection.annotated_streamer import update_detections as _upd_det
                _upd_det(camera_id, detections)
                if frame_count % 80 == 1:
                    print(f"[annotated_dets] cam={camera_id} frame={frame_count} dets={len(detections or [])}", flush=True)
            except Exception as _e:
                print(f"[annotated_dets] cam={camera_id} ERR: {_e}", flush=True)

    
    cap.release()
    add_log("info", f"偵測背景任務結束: camera_id={camera_id}", "detection")
    print(f"⏹️ 偵測服務停止: camera_id={camera_id}")
