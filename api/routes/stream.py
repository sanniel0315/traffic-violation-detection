#!/usr/bin/env python3
"""串流與偵測服務 API"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import cv2
import asyncio
import threading
import time
import requests
from datetime import datetime, timedelta
from typing import Dict
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
from api.utils.roi_scope import SCOPE_TRAFFIC, SCOPE_SPEED, select_zones
from api.utils.feature_state import get_feature_enabled, set_feature_state
from api.utils.camera_stream import resolve_analysis_source, resolve_capture_source, resolve_local_api_source

router = APIRouter(prefix="/api/stream", tags=["串流"])

# 偵測服務狀態
detection_services: Dict[int, dict] = {}
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
    """Background warm-up snapshot to improve next request hit-rate."""
    lock = snapshot_locks.setdefault(camera_id, asyncio.Lock())
    if lock.locked():
        return
    async with lock:
        image = None
        for _ in range(2):
            try:
                image = await asyncio.wait_for(
                    asyncio.to_thread(_capture_snapshot_bytes, source),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                image = None
            if image:
                break
        if image:
            snapshot_cache[camera_id] = {"image": image, "ts": time.time()}


def _open_capture(source: str):
    """OpenCV capture with backend fallback to avoid hard dependency on FFMPEG backend."""
    source = resolve_capture_source(source)
    backends = []
    source_lc = str(source or "").lower()
    if source_lc.startswith("http://") or source_lc.startswith("https://"):
        mjpeg_backend = getattr(cv2, "CAP_OPENCV_MJPEG", None)
        if mjpeg_backend is not None:
            backends.append(mjpeg_backend)
        backends.append(None)
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


def _capture_snapshot_bytes(source: str):
    """嘗試以較短超時擷取單張影像，避免 RTSP 長時間阻塞。"""
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
):
    """產生即時 MJPEG 串流，可選擇是否繪製辨識疊加。"""
    http_source = resolve_local_api_source(source)
    use_http_mjpeg = _is_http_mjpeg_source(http_source)
    source = resolve_capture_source(source)
    cap = None if use_http_mjpeg else _open_capture(source)
    if cap is not None:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    zones = zones or []
    detection_config = detection_config or {}
    det_zones = select_zones(zones, scope=SCOPE_TRAFFIC, allowed_types=("detection", "flow_detection"))
    speed_zones = select_zones(zones, scope=SCOPE_SPEED, allowed_types=("speed", "speed_roi"))
    # 粗略像素速度轉換係數（可在 detection_config.speed_kmh_per_pxps 調整）
    speed_kmh_per_pxps = float(detection_config.get("speed_kmh_per_pxps", 0.12))
    speed_smooth_alpha = float(detection_config.get("speed_smooth_alpha", 0.35))
    tracks = {}
    next_track_id = 1
    track_ttl_sec = 1.2
    detector = None
    try:
        from detection.vehicle_detector import VehicleDetector
        detector = VehicleDetector(conf_threshold=0.45)
    except Exception as e:
        add_log("warning", f"疊加串流偵測器初始化失敗，改回原始串流: {e}", "stream")
    
    last_ok = time.time()
    last_placeholder_ts = 0.0
    had_frame = False
    http_frames = None
    http_state = _ensure_http_mjpeg_worker(http_source) if use_http_mjpeg else None
    last_http_ts = 0.0
    while True:
        if use_http_mjpeg:
            _touch_http_mjpeg_worker(http_state)
            ts = float(http_state.get("last_ts") or 0.0) if http_state else 0.0
            jpg = http_state.get("last_jpeg") if http_state else None
            if jpg and ts > 0 and ts != last_http_ts:
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                ret = frame is not None and getattr(frame, "size", 0) > 0
                last_http_ts = ts
            else:
                frame = None
                ret = False
        else:
            ret, frame = cap.read()
        if not ret:
            if time.time() - last_ok > 2.0:
                if cap is not None:
                    cap.release()
                    time.sleep(0.1)
                    cap = _open_capture(source)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

        frame = cv2.resize(frame, (640, 360))
        if not render_overlay:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        if render_roi_labels:
            _draw_roi_labels(frame, zones)
        if detector is not None:
            try:
                detections = detector.detect(frame)
                fh, fw = frame.shape[:2]
                now_ts = time.time()
                valid_dets = []
                for det in detections:
                    if det.get("class_name") not in ["car", "motorcycle", "truck", "bus"]:
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

                # 清理過期追蹤
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
                    tracks[track_id] = {
                        "center": (cx, cy),
                        "t": now_ts,
                        "class_name": det["class_name"],
                        "speed_kmh": speed_kmh,
                    }

                    b = det["bbox"]
                    if render_overlay:
                        color = (0, 200, 0)
                        cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        if in_speed_roi:
                            label += f" | {int(speed_kmh)} km/h"
                        else:
                            label += " | ROI"
                        cv2.putText(
                            frame,
                            label,
                            (b["x1"], max(20, b["y1"] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2
                        )
            except Exception:
                pass
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)
    
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
            camera.source,
            camera.zones or [],
            camera.detection_config or {},
            render_overlay=False,
            render_roi_labels=False,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/{camera_id}/live-overlay")
async def live_stream_overlay(camera_id: int, db: Session = Depends(get_db)):
    """即時影像串流 (MJPEG + AI 辨識框)"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")
    overlay_source = camera.source if "/api/nx/stream/" in str(camera.source or "") else resolve_analysis_source(camera)
    return StreamingResponse(
        generate_frames_overlay(
            overlay_source,
            camera.zones or [],
            camera.detection_config or {},
            render_roi_labels=False,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/{camera_id}/snapshot")
async def snapshot(camera_id: int, db: Session = Depends(get_db)):
    """取得單張截圖"""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="攝影機不存在")
    if not bool(camera.enabled):
        raise HTTPException(status_code=409, detail="攝影機已關閉")
    
    # 若最近已有快照，直接回傳，避免前端縮圖連續請求造成 RTSP 阻塞
    cached = snapshot_cache.get(camera_id)
    if cached and (time.time() - cached.get("ts", 0) <= 2):
        return StreamingResponse(iter([cached.get("image")]), media_type="image/jpeg")

    lock = snapshot_locks.setdefault(camera_id, asyncio.Lock())
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
        cached_after_wait = snapshot_cache.get(camera_id)
        if cached_after_wait and (time.time() - cached_after_wait.get("ts", 0) <= 600):
            return StreamingResponse(iter([cached_after_wait.get("image")]), media_type="image/jpeg")

    async with lock:
        # Keep snapshot latency very short for UI usage; slow sources should fallback fast
        # and let warm-up task fill cache for subsequent requests.
        attempts = [(4.0, 1)]
        for timeout_sec, count in attempts:
            if image:
                break
            for _ in range(count):
                if image:
                    break
                try:
                    image = await asyncio.wait_for(
                        asyncio.to_thread(_capture_snapshot_bytes, camera.source),
                        timeout=timeout_sec,
                    )
                except asyncio.TimeoutError:
                    image = None

    if image:
        snapshot_cache[camera_id] = {"image": image, "ts": time.time()}
    else:
        if cached and (time.time() - cached.get("ts", 0) <= 600):
            image = cached.get("image")
        else:
            # No immediate snapshot/cached image: trigger background warm-up for next request.
            task = snapshot_warm_tasks.get(camera_id)
            if task is None or task.done():
                snapshot_warm_tasks[camera_id] = asyncio.create_task(
                    _warm_snapshot(camera_id, camera.source)
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
        return detection_services[camera_id]
    return {"running": False}


@router.get("/detection/all")
async def all_detection_status():
    """取得所有偵測服務狀態"""
    return detection_services


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
    
    detector = VehicleDetector(conf_threshold=0.5)
    add_log("info", f"偵測器 device: {getattr(detector, 'runtime_device', 'unknown')}", "detection")
    cap = cv2.VideoCapture(source)
    speed_kmh_per_pxps = float(detection_config.get("speed_kmh_per_pxps", 0.12) or 0.12)
    speed_smooth_alpha = float(detection_config.get("speed_smooth_alpha", 0.35) or 0.35)
    track_ttl_sec = float(detection_config.get("speed_track_ttl_sec", 1.2) or 1.2)
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
    
    det_zones = select_zones(zones, scope=SCOPE_TRAFFIC, allowed_types=("detection", "flow_detection"))
    speed_zones = select_zones(zones, scope=SCOPE_SPEED, allowed_types=("speed", "speed_roi"))
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
    
    while detection_services.get(camera_id, {}).get('running', False):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        if frame_count % 10 != 0:  # 每 10 幀處理一次
            continue
        
        detections = detector.detect(frame)
        vehicles = [d for d in detections if d['class_name'] in ['car', 'motorcycle', 'truck', 'bus']]
        
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
            for v in vehicles:
                b = v.get("bbox", {}) or {}
                cx = int((b.get("x1", 0) + b.get("x2", 0)) / 2)
                cy = int((b.get("y1", 0) + b.get("y2", 0)) / 2)
                cls = str(v.get("class_name") or "")
                track_id = _nearest_track_id((cx, cy), cls, tracks)
                if track_id is None:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {"center": (cx, cy), "t": now_ts, "class_name": cls, "speed_kmh": None}
                prev = tracks.get(track_id, {})
                prev_center = prev.get("center", (cx, cy))
                prev_t = prev.get("t", now_ts)
                dt = max(1e-3, now_ts - prev_t)
                px_dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                pxps = px_dist / dt
                raw_kmh = pxps * speed_kmh_per_pxps
                prev_kmh = prev.get("speed_kmh")
                speed_kmh = raw_kmh if prev_kmh is None else ((speed_smooth_alpha * raw_kmh) + ((1.0 - speed_smooth_alpha) * prev_kmh))
                speed_kmh = max(0.0, min(220.0, float(speed_kmh)))
                tracks[track_id] = {"center": (cx, cy), "t": now_ts, "class_name": cls, "speed_kmh": speed_kmh}
                v["speed_kmh"] = speed_kmh
        
        if vehicles:
            detection_count += 1
            
            # 更新服務狀態
            detection_services[camera_id]['detections'] = detection_count
            detection_services[camera_id]['last_detection'] = datetime.now().isoformat()
            db = None
            try:
                db = SessionLocal()
                rows = []
                zone_occupancy_map = {
                    _zone_key(z): _zone_occupancy(z, vehicles)
                    for z in det_zones
                } if det_zones else {}
                for v in vehicles:
                    bbox = v.get("bbox", {}) or {}
                    hit_zones = _vehicle_hit_zones(v, det_zones)
                    pick_zone = hit_zones[0] if hit_zones else {}
                    occupancy_val = zone_occupancy_map.get(_zone_key(pick_zone)) if pick_zone else None
                    speed_raw = v.get("speed_kmh")
                    try:
                        speed_num = float(speed_raw)
                    except Exception:
                        speed_num = None
                    speed_val = speed_num if (isinstance(speed_num, float) and speed_num > 0) else None
                    rows.append(TrafficEvent(
                        camera_id=int(camera_id),
                        label=str(v.get("class_name", "unknown")).lower(),
                        speed_kmh=speed_val,
                        occupancy=occupancy_val,
                        lane_no=_parse_lane_no(pick_zone),
                        direction=_normalize_event_direction(pick_zone.get("direction")),
                        entered_zones=[str(z.get("name") or "") for z in hit_zones if str(z.get("name") or "")],
                        bbox=[bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")],
                        source="roi_detection",
                    ))
                if rows:
                    db.add_all(rows)
                    db.commit()
                db.close()
            except Exception:
                try:
                    if db is not None:
                        db.close()
                except Exception:
                    pass
            
            # 每 50 次偵測記錄一次違規 (模擬)
            if detection_count % 50 == 1 and enabled_types:
                import random
                candidates = list(enabled_types)
                if any(t[0] == 'SPEEDING' for t in candidates):
                    has_speed_zone = len(speed_zones) > 0
                    if has_speed_zone and not speed_zone_vehicles:
                        candidates = [t for t in candidates if t[0] != 'SPEEDING']
                if not candidates:
                    continue
                v_type, v_name, v_fine = random.choice(candidates)
                lpr_hit = _latest_lpr_plate(camera_id, max_age_sec=20)
                plate = (lpr_hit or {}).get("plate", "")
                speed_limit_kmh = float(detection_config.get("speed_limit", 50) or 50)
                overspeed_threshold_kmh = 10.0
                if speed_zones:
                    zone_cfg = speed_zones[0] or {}
                    speed_limit_kmh = float(zone_cfg.get("speed_limit") or speed_limit_kmh or 50)
                    overspeed_threshold_kmh = float(
                        zone_cfg.get("overspeed_kmh")
                        or zone_cfg.get("speed_margin")
                        or overspeed_threshold_kmh
                    )
                
                # 繪製標註
                annotated = frame.copy()
                for det in vehicles:
                    bbox = det['bbox']
                    cv2.rectangle(annotated, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 2)
                
                # 儲存截圖
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = f"{v_type}_{timestamp_str}.jpg"
                image_path = output_dir / image_name
                cv2.imwrite(str(image_path), annotated)
                
                # 發送到 API
                target_vehicle = speed_zone_vehicles[0] if (v_type == 'SPEEDING' and speed_zone_vehicles) else vehicles[0]
                speed_kmh = None
                overspeed_kmh = None
                if v_type == "SPEEDING":
                    raw_speed = target_vehicle.get("speed_kmh")
                    if isinstance(raw_speed, (int, float)) and raw_speed > 0:
                        speed_kmh = float(raw_speed)
                    else:
                        speed_kmh = speed_limit_kmh + max(overspeed_threshold_kmh, 5.0) + random.uniform(3.0, 15.0)
                    overspeed_kmh = max(0.0, speed_kmh - speed_limit_kmh)
                data = {
                    "violation_type": v_type,
                    "violation_name": v_name,
                    "vehicle_type": target_vehicle['class_name'],
                    "license_plate": plate or None,
                    "location": location,
                    "camera_id": camera_id,
                    "confidence": target_vehicle['confidence'],
                    "fine_amount": v_fine,
                    "points": 1,
                    "image_path": f"/files/violations/{image_name}",
                    "bbox": target_vehicle.get("bbox"),
                    "speed_kmh": round(speed_kmh, 1) if isinstance(speed_kmh, (int, float)) else None,
                    "speed_limit_kmh": round(speed_limit_kmh, 1) if v_type == "SPEEDING" else None,
                    "overspeed_kmh": round(overspeed_kmh, 1) if isinstance(overspeed_kmh, (int, float)) else None,
                    "flow_roi_hit": _vehicle_in_any_zone(target_vehicle, det_zones),
                    "speed_roi_hit": _vehicle_in_any_zone(target_vehicle, speed_zones),
                }
                
                try:
                    requests.post("http://localhost:8000/api/violations", json=data, timeout=5)
                    plate_text = plate or "NODATA(LPR)"
                    speed_text = f" | {data['speed_kmh']}km/h (限{data['speed_limit_kmh']})" if data.get("speed_kmh") else ""
                    print(f"🚨 違規記錄: {v_name} | {plate_text}{speed_text}")
                    add_log("warning", f"偵測到違規: {v_name} | 車牌 {plate_text}{speed_text} | 攝影機 ID={camera_id}", "detection")
                except:
                    pass
        
        time.sleep(0.03)
    
    cap.release()
    add_log("info", f"偵測背景任務結束: camera_id={camera_id}", "detection")
    print(f"⏹️ 偵測服務停止: camera_id={camera_id}")
