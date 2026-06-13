"""
停車場車位佔用判定 — ROI 多邊形 + YOLO car detection.

兩種 source:
- twipcam_demo: 拉 TwiPcam 公開 snapshot URL (https://c01.twipcam.com/...)
- cam:{id}:     走 lpr_visual frame fallback chain (frigate/go2rtc/RTSP)

對每個 slot (config 內定義的多邊形) 跑 IoU 判定:
- 任一 detected vehicle bbox 中心點落在 slot polygon 內 → occupied
- 或 slot polygon 跟 vehicle bbox IoU > 0.15 → occupied
- 否則 → empty
"""
from __future__ import annotations

import json
import os
import time
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "parking_slots.json")

# 預設 source 配置 — 只列出 source meta,slots 留空讓 user 用「自動偵測」/
# 「背景累積」/ 拖拉 自己決定 (不再 hardcode 預設車位框)
_DEFAULT_CONFIG = {
    "twipcam:tpe-005013": {
        "name": "百齡橋停車場 (台北市政府 PB048)",
        "image_url": "https://c01.twipcam.com/cam/snapshot/tpe-005013.jpg",
        "note": "請點「自動偵測車位」或「背景累積偵測」標 polygon",
        "slots": [],
    }
}

_LATEST_FRAME_CACHE: Dict[str, Tuple[float, np.ndarray]] = {}
_FRAME_CACHE_TTL = 2.0  # 同 source 2 秒內共用 frame,降低 fetch 次數
_CACHE_LOCK = threading.Lock()

# 歷史寫入 throttle (per-source 5 分鐘記一筆)
_HISTORY_LAST_AT: Dict[str, float] = {}
_HISTORY_THROTTLE_SEC = 300.0
_HISTORY_LOCK = threading.Lock()

# Phase 3 交叉校正: |車位偵測 - 進出累計| >= 此值 → 升級 VLM 仲裁範圍
CROSS_GAP_ESCALATE = 5

# 自動偵測 background session (per-source)
_AUTO_SESSIONS: Dict[str, Dict] = {}
_AUTO_LOCK = threading.Lock()


def auto_session_start(source: str, conf: float = 0.15, expand: float = 0.08,
                       interval_sec: float = 30.0, max_frames: int = 60) -> Dict:
    """啟動背景自動偵測 — thread 每 interval 拉 frame yolo + merge,
    max_frames 達到自動停 (避免無限跑)."""
    with _AUTO_LOCK:
        old = _AUTO_SESSIONS.get(source)
        if old and old.get("running"):
            return {"ok": False, "error": "session already running",
                    "frames_processed": old.get("frames_processed", 0),
                    "merged_slots": len(old.get("merged_boxes", []))}
        sess = {
            "running": True, "frames_processed": 0, "merged_boxes": [],
            "all_raw_count": 0, "started_at": time.time(),
            "conf": conf, "expand": expand, "interval_sec": interval_sec,
            "max_frames": max_frames, "frame_w": 0, "frame_h": 0,
            "last_frame_at": 0.0, "stop_event": threading.Event(),
        }
        _AUTO_SESSIONS[source] = sess

    def _runner():
        try:
            from detection.vehicle_detector import VehicleDetector
            yolo_local = VehicleDetector(conf_threshold=float(conf))
        except Exception as e:
            sess["error"] = f"yolo init: {e}"
            sess["running"] = False
            return
        vehicle_classes = {"car", "truck", "bus", "heavy_truck", "light_truck", "non_truck"}
        first = True
        while sess["frames_processed"] < max_frames and not sess["stop_event"].is_set():
            if not first:
                if sess["stop_event"].wait(interval_sec):
                    break
            first = False
            frame = fetch_frame(source, bypass_cache=True)
            if frame is None:
                continue
            if sess["frame_w"] == 0:
                sess["frame_h"], sess["frame_w"] = frame.shape[:2]
            try:
                dets = yolo_local.detect(frame)
            except Exception:
                continue
            for det in dets or []:
                cls = str(det.get("class_name") or "").lower()
                if cls not in vehicle_classes:
                    continue
                bb = det.get("bbox", {})
                x1 = int(bb.get("x1", 0)); y1 = int(bb.get("y1", 0))
                x2 = int(bb.get("x2", 0)); y2 = int(bb.get("y2", 0))
                if x2 <= x1 or y2 <= y1:
                    continue
                sess["all_raw_count"] += 1
                _merge_box_into(sess["merged_boxes"], [x1, y1, x2, y2])
            sess["frames_processed"] += 1
            sess["last_frame_at"] = time.time()
        sess["running"] = False

    t = threading.Thread(target=_runner, daemon=True, name=f"parking_auto_{source}")
    sess["thread"] = t
    t.start()
    return {"ok": True, "source": source, "started_at": sess["started_at"]}


def _merge_box_into(merged: list, bx: list) -> None:
    """IoU > 0.5 合併 (加權平均座標 + count++)"""
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        ar = (a[2]-a[0]) * (a[3]-a[1])
        br = (b[2]-b[0]) * (b[3]-b[1])
        return inter / max(1, ar + br - inter)

    for i, m in enumerate(merged):
        if _iou(bx, m[:4]) > 0.5:
            cnt = m[4] + 1
            nx1 = (m[0] * m[4] + bx[0]) / cnt
            ny1 = (m[1] * m[4] + bx[1]) / cnt
            nx2 = (m[2] * m[4] + bx[2]) / cnt
            ny2 = (m[3] * m[4] + bx[3]) / cnt
            merged[i] = [int(nx1), int(ny1), int(nx2), int(ny2), cnt]
            return
    merged.append([bx[0], bx[1], bx[2], bx[3], 1])


def auto_session_status(source: str) -> Dict:
    sess = _AUTO_SESSIONS.get(source)
    if not sess:
        return {"running": False, "frames_processed": 0, "merged_slots": 0,
                "total_detections": 0}
    return {
        "running": sess.get("running", False),
        "frames_processed": sess.get("frames_processed", 0),
        "merged_slots": len(sess.get("merged_boxes", [])),
        "total_detections": sess.get("all_raw_count", 0),
        "frame_w": sess.get("frame_w", 0),
        "frame_h": sess.get("frame_h", 0),
        "started_at": sess.get("started_at"),
        "last_frame_at": sess.get("last_frame_at"),
        "max_frames": sess.get("max_frames"),
        "interval_sec": sess.get("interval_sec"),
        "error": sess.get("error", ""),
    }


def auto_session_stop_and_get(source: str) -> Dict:
    """停止 session 並回回 merged slots (轉成 polygon 4 點 + 外擴)"""
    sess = _AUTO_SESSIONS.get(source)
    if not sess:
        return {"slots": [], "frames_processed": 0, "merged_slots": 0}
    sess.get("stop_event") and sess["stop_event"].set()
    expand = float(sess.get("expand", 0.08))
    w_img = sess.get("frame_w", 1)
    h_img = sess.get("frame_h", 1)
    slots_out = []
    for idx, m in enumerate(sess.get("merged_boxes", [])):
        x1, y1, x2, y2, _cnt = m
        w = x2 - x1; h = y2 - y1
        dx = int(w * expand); dy = int(h * expand)
        nx1 = max(0, x1 - dx); ny1 = max(0, y1 - dy)
        nx2 = min(w_img - 1, x2 + dx); ny2 = min(h_img - 1, y2 + dy)
        lbl = f"P{idx + 1}"
        slots_out.append({
            "id": lbl, "label": lbl,
            "polygon": [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]],
            "samples": _cnt,
        })
    return {
        "slots": slots_out,
        "frames_processed": sess.get("frames_processed", 0),
        "merged_slots": len(sess.get("merged_boxes", [])),
        "total_detections": sess.get("all_raw_count", 0),
        "frame_w": w_img, "frame_h": h_img,
    }


def record_to_history(result: Dict) -> bool:
    """寫一筆 ParkingSample (per-source 5 分鐘 throttle).
    Return True = 已寫入, False = 跳過 (throttle 中或無 total)."""
    if not result or not result.get("total"):
        return False
    src = result.get("source") or ""
    if not src:
        return False
    now = time.time()
    with _HISTORY_LOCK:
        last = _HISTORY_LAST_AT.get(src, 0.0)
        if (now - last) < _HISTORY_THROTTLE_SEC:
            return False
        _HISTORY_LAST_AT[src] = now
    try:
        from api.models import SessionLocal, ParkingSample
        db = SessionLocal()
        try:
            row = ParkingSample(
                source=src,
                source_name=result.get("source_name", src),
                total=int(result.get("total") or 0),
                occupied=int(result.get("occupied") or 0),
                available=int(result.get("available") or 0),
                occupancy_rate=float(result.get("occupancy_rate") or 0.0),
                detected_vehicles=int(result.get("detected_vehicles") or 0),
                vehicles_in_area=int(result.get("vehicles_in_area") or 0),
            )
            db.add(row)
            db.commit()
            return True
        finally:
            db.close()
    except Exception as e:
        print(f"[parking] record_to_history err: {e}", flush=True)
        return False


def load_slots(source_key: str) -> List[Dict]:
    """讀 parking_slots.json,回傳 source_key 對應的 slots list.
    每個 slot: {id, label, polygon: [[x,y], ...] (pixel coord)}"""
    if not os.path.exists(_CONFIG_PATH):
        return _DEFAULT_CONFIG.get(source_key, {}).get("slots", [])
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d.get(source_key, {}).get("slots", [])
    except Exception as e:
        print(f"[parking] load_slots failed: {e}", flush=True)
        return []


def get_source_meta(source_key: str) -> Dict:
    """讀 source 設定 (name / image_url / frame_size 等)"""
    if not os.path.exists(_CONFIG_PATH):
        meta = dict(_DEFAULT_CONFIG.get(source_key, {}))
        meta.pop("slots", None)
        return meta
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = dict(d.get(source_key, {}))
        meta.pop("slots", None)
        return meta
    except Exception:
        return {}


def fetch_frame(source_key: str, bypass_cache: bool = False) -> Optional[np.ndarray]:
    """從 source 拉一張 frame (BGR ndarray).有 2 秒 cache (multi-frame 累積時可 bypass)."""
    now = time.time()
    if not bypass_cache:
        with _CACHE_LOCK:
            cached = _LATEST_FRAME_CACHE.get(source_key)
            if cached and (now - cached[0]) < _FRAME_CACHE_TTL:
                return cached[1].copy()

    frame = None
    meta = get_source_meta(source_key)
    # 優先 stream_url (RTSP / MJPEG / HLS — cv2.VideoCapture 直拉)
    stream_url = meta.get("stream_url", "") if meta else ""
    if stream_url:
        try:
            cap = cv2.VideoCapture(stream_url)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ret, frame_in = cap.read()
            cap.release()
            if ret and frame_in is not None:
                frame = frame_in
            else:
                print(f"[parking] stream_url 拉失敗 {stream_url}", flush=True)
        except Exception as e:
            print(f"[parking] stream_url err: {e}", flush=True)

    if frame is None and source_key.startswith("twipcam:"):
        import requests as _req
        url = source_key.split(":", 1)[1]
        if not url.startswith("http"):
            url = meta.get("image_url", "")
        try:
            if bypass_cache:
                sep = "&" if "?" in url else "?"
                fetch_url = f"{url}{sep}_t={int(now*1000)}"
                headers = {"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"}
            else:
                fetch_url = url
                headers = {}
            r = _req.get(fetch_url, timeout=8, headers=headers)
            content = r.content
            # 驗 JPEG 完整性再 decode: 殘缺/截斷的 buffer 餵 cv2.imdecode 會 native SEGV
            # 拉垮整個 traffic-api (try/except 攔不到 SIGSEGV),只能 decode 前擋
            valid_jpeg = (len(content) > 5000 and content[:2] == b"\xff\xd8"
                          and b"\xff\xd9" in content[-16:])
            if r.status_code == 200 and valid_jpeg:
                frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            elif r.status_code == 200:
                print(f"[parking] twipcam JPEG 不完整 (len={len(content)}),跳過 decode 避免 SEGV", flush=True)
        except Exception as e:
            print(f"[parking] twipcam fetch err: {e}", flush=True)
    elif source_key.startswith("cam:"):
        cam_id = source_key.split(":", 1)[1]
        try:
            from api.routes.vision_eye import _fetch_frame as _ve_fetch
            from api.models import SessionLocal, Camera
            db = SessionLocal()
            try:
                cam = db.query(Camera).filter(Camera.id == int(cam_id)).first()
                if cam:
                    frame = _ve_fetch(cam)
            finally:
                db.close()
        except Exception as e:
            print(f"[parking] cam fetch err: {e}", flush=True)

    if frame is not None:
        with _CACHE_LOCK:
            _LATEST_FRAME_CACHE[source_key] = (now, frame.copy())
    return frame


def _point_in_polygon(px: float, py: float, polygon: List[List[float]]) -> bool:
    """ray-casting,polygon 是 [[x,y], ...]"""
    if len(polygon) < 3:
        return False
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_bbox(polygon: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _bbox_iou_with_polygon(bbox: Tuple[int, int, int, int],
                           polygon: List[List[float]]) -> float:
    """簡化:用 polygon bbox 跟 vehicle bbox 算 IoU."""
    px1, py1, px2, py2 = _polygon_bbox(polygon)
    bx1, by1, bx2, by2 = bbox
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = max(1, (px2 - px1) * (py2 - py1))
    a2 = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a1 + a2 - inter)


# 自動車位累積 (zero-config 模式) — per-source in-memory dict:
# {source: [[x1,y1,x2,y2,seen_count,last_seen_ts], ...]}
_AUTO_POSITIONS: Dict[str, list] = {}
_AUTO_POS_LOCK = threading.Lock()
# IO trigger throttle (per source 60 秒只觸發一次,避免狂發 pulse)
_IO_TRIGGER_LAST_AT: Dict[str, float] = {}
_IO_TRIGGER_THROTTLE_SEC = 60.0
_IO_TRIGGER_LOCK = threading.Lock()


def maybe_trigger_io(result: Dict, meta: Dict) -> Dict:
    """meta 內若有 io_trigger 設定且 rate >= threshold,觸發 io_tcp pulse.
    return {triggered, reason}"""
    cfg = meta.get("io_trigger") or {}
    if not cfg.get("enabled"):
        return {"triggered": False, "reason": "io_trigger 未啟用"}
    module_id = cfg.get("module_id")
    do_ch = int(cfg.get("do_ch", 0))
    threshold = float(cfg.get("threshold_rate", 95.0))
    pulse_ms = int(cfg.get("pulse_ms", 1000))
    rate = float(result.get("occupancy_rate") or 0)
    if rate < threshold:
        return {"triggered": False, "reason": f"rate {rate}% < threshold {threshold}%"}
    if not module_id:
        return {"triggered": False, "reason": "未指定 module_id"}
    src = result.get("source") or ""
    now = time.time()
    with _IO_TRIGGER_LOCK:
        last = _IO_TRIGGER_LAST_AT.get(src, 0.0)
        if (now - last) < _IO_TRIGGER_THROTTLE_SEC:
            return {"triggered": False, "reason": "60 秒內已觸發過"}
        _IO_TRIGGER_LAST_AT[src] = now
    try:
        from services.modbus_tcp_io import get_module
        mod = get_module(module_id)
        if mod is None:
            return {"triggered": False, "reason": f"module {module_id} 未配置"}
        ok = mod.pulse_output(do_ch, pulse_ms)
        return {"triggered": ok, "module_id": module_id, "do_ch": do_ch,
                "pulse_ms": pulse_ms, "rate": rate,
                "reason": f"rate {rate}% >= {threshold}%" if ok else mod.error}
    except Exception as e:
        return {"triggered": False, "reason": str(e)}
_AUTO_POS_TTL_SEC = 3600.0
_AUTO_POS_CONF = 0.12         # 低 threshold,盡量抓到所有車 (sliced 後 false pos 在 NMS 過濾)
_AUTO_POS_MERGE_IOU = 0.6     # IoU 0.6 才 merge,相鄰車不誤合
_AUTO_POS_OCC_IOU = 0.4
# Sliced inference — 3×3 切片 + 全圖各跑一次,對「俯角遠處小車」(停車場典型) 偵測率提升明顯
_SLICE_ROWS = 3
_SLICE_COLS = 3
_SLICE_OVERLAP = 0.2


def _yolo_sliced_detect(yolo_local, frame: np.ndarray) -> list:
    """sliced inference — 預設手寫 (停車場場景效果較好),env PARKING_USE_SAHI=1 可切 SAHI.
    切 frame 成 R×C tile 各自跑 yolo,coord 加 offset 回 full-frame."""
    if os.getenv("PARKING_USE_SAHI", "0") == "1":
        try:
            from services.parking_sahi import is_available, sahi_detect
            if is_available():
                sahi_results = sahi_detect(frame, confidence=_AUTO_POS_CONF,
                                            slice_size=256, overlap=0.2)
                if sahi_results:
                    return sahi_results
        except Exception as e:
            print(f"[parking] sahi fallback: {e}", flush=True)
    # 預設手寫 sliced
    h, w = frame.shape[:2]
    tile_w = max(64, w // _SLICE_COLS)
    tile_h = max(64, h // _SLICE_ROWS)
    ov_w = int(tile_w * _SLICE_OVERLAP)
    ov_h = int(tile_h * _SLICE_OVERLAP)
    all_dets = []
    for r in range(_SLICE_ROWS):
        for c in range(_SLICE_COLS):
            x0 = max(0, c * tile_w - ov_w)
            y0 = max(0, r * tile_h - ov_h)
            x1 = min(w, (c + 1) * tile_w + ov_w)
            y1 = min(h, (r + 1) * tile_h + ov_h)
            tile = frame[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            try:
                dets = yolo_local.detect(tile)
            except Exception:
                continue
            for d in dets or []:
                bb = d.get("bbox") or {}
                all_dets.append({
                    "class_name": d.get("class_name", ""),
                    "conf": float(d.get("confidence") or d.get("conf") or 0.0),
                    "bbox": {
                        "x1": int(bb.get("x1", 0)) + x0,
                        "y1": int(bb.get("y1", 0)) + y0,
                        "x2": int(bb.get("x2", 0)) + x0,
                        "y2": int(bb.get("y2", 0)) + y0,
                    },
                })
    # 全圖補大型物 (sliced 可能切斷貨車/巴士)
    try:
        for d in (yolo_local.detect(frame) or []):
            bb = d.get("bbox") or {}
            all_dets.append({
                "class_name": d.get("class_name", ""),
                "conf": float(d.get("confidence") or d.get("conf") or 0.0),
                "bbox": {
                    "x1": int(bb.get("x1", 0)), "y1": int(bb.get("y1", 0)),
                    "x2": int(bb.get("x2", 0)), "y2": int(bb.get("y2", 0)),
                },
            })
    except Exception:
        pass

    # NMS
    def _iou_d(a, b):
        ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        ar = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        br = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        return inter / max(1, ar + br - inter)
    all_dets.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
    kept = []
    for d in all_dets:
        # NMS IoU > 0.3 抑制 (從 0.5 降到 0.3 — 同台車跨 tile 切斷後 2 個 bbox
        # 通常 IoU 落 0.3-0.5 區間,更嚴 NMS 才合併)
        if any(_iou_d(d["bbox"], k["bbox"]) > 0.3 for k in kept):
            continue
        kept.append(d)
    return kept


def reset_auto_positions(source_key: str) -> int:
    """清掉 _AUTO_POSITIONS[source],回 cleared 數"""
    with _AUTO_POS_LOCK:
        old = _AUTO_POSITIONS.pop(source_key, [])
    return len(old)


def _eval_auto_mode(source_key: str, frame: np.ndarray, meta: Dict) -> Dict:
    """Zero-config — 偵測到車的位置就是車位.每次 evaluate:
    1. yolo detect 當前 cars
    2. 對每車 bbox merge 進 _AUTO_POSITIONS (IoU>0.5 → 加權平均座標 + count)
    3. 算 occupancy: 累積每位置看是否有 current detection match
    4. 30 分沒見過自動 GC"""
    h_img, w_img = frame.shape[:2]
    try:
        from detection.vehicle_detector import VehicleDetector
        yolo_local = VehicleDetector(conf_threshold=_AUTO_POS_CONF)
        detections = _yolo_sliced_detect(yolo_local, frame)
    except Exception as e:
        return {"source": source_key, "source_name": meta.get("name", source_key),
                "error": f"yolo: {e}", "total": 0, "occupied": 0, "available": 0,
                "occupancy_rate": 0.0, "slots": []}

    vehicle_classes = {"car", "truck", "bus", "heavy_truck", "light_truck", "non_truck"}
    current_boxes = []
    for det in detections or []:
        cls = str(det.get("class_name") or "").lower()
        if cls not in vehicle_classes:
            continue
        bb = det.get("bbox", {})
        x1 = int(bb.get("x1", 0)); y1 = int(bb.get("y1", 0))
        x2 = int(bb.get("x2", 0)); y2 = int(bb.get("y2", 0))
        if x2 <= x1 or y2 <= y1:
            continue
        current_boxes.append([x1, y1, x2, y2])

    def _iou(a, b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
        inter=iw*ih
        ar=(a[2]-a[0])*(a[3]-a[1]); br=(b[2]-b[0])*(b[3]-b[1])
        return inter/max(1, ar+br-inter)

    now = time.time()
    with _AUTO_POS_LOCK:
        positions = _AUTO_POSITIONS.setdefault(source_key, [])
        # 對每個 current detection 累積進 positions
        for bx in current_boxes:
            absorbed = False
            for i, p in enumerate(positions):
                if _iou(bx, p[:4]) > _AUTO_POS_MERGE_IOU:
                    cnt = p[4] + 1
                    nx1 = (p[0]*p[4] + bx[0]) / cnt
                    ny1 = (p[1]*p[4] + bx[1]) / cnt
                    nx2 = (p[2]*p[4] + bx[2]) / cnt
                    ny2 = (p[3]*p[4] + bx[3]) / cnt
                    positions[i] = [int(nx1), int(ny1), int(nx2), int(ny2), cnt, now]
                    absorbed = True
                    break
            if not absorbed:
                positions.append([bx[0], bx[1], bx[2], bx[3], 1, now])
        # TTL GC
        positions[:] = [p for p in positions if (now - p[5]) <= _AUTO_POS_TTL_SEC]

        # Mask filter (inclusion + exclusion) — 同 PKLot mode 邏輯
        area_mask = meta.get("parking_area_mask") or []
        exclusion_mask = meta.get("exclusion_mask") or []
        try:
            from services.parking_pklot_model import _point_in_poly
        except Exception:
            _point_in_poly = None

        # 算佔用: 對每 position 看是否有 current detection match,
        # 車位 polygon 用車身大小 (不外擴)
        slot_results = []
        idx = 0
        for p in positions:
            x1, y1, x2, y2, _cnt, last = p
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            # mask 過濾
            if _point_in_poly:
                if area_mask and not _point_in_poly(cx, cy, area_mask):
                    continue
                if exclusion_mask and _point_in_poly(cx, cy, exclusion_mask):
                    continue
            poly = [[max(0,x1), max(0,y1)],
                    [min(w_img-1,x2), max(0,y1)],
                    [min(w_img-1,x2), min(h_img-1,y2)],
                    [max(0,x1), min(h_img-1,y2)]]
            occ = False
            for cb in current_boxes:
                if _iou([x1,y1,x2,y2], cb) > _AUTO_POS_OCC_IOU:
                    occ = True
                    break
            idx += 1
            lbl = f"P{idx}"
            slot_results.append({
                "id": lbl, "label": lbl,
                "occupied": occ, "conf": round(min(1.0, _cnt/5.0), 3),
                "polygon": poly,
            })
    occupied = sum(1 for s in slot_results if s["occupied"])
    total = len(slot_results)
    result = {
        "source": source_key, "source_name": meta.get("name", source_key),
        "frame_w": w_img, "frame_h": h_img,
        "total": total, "occupied": occupied,
        "available": total - occupied,
        "occupancy_rate": round((occupied/total*100.0) if total else 0.0, 1),
        "detected_vehicles": len(current_boxes),
        "slots": slot_results,
        "mode": "auto",
    }
    record_to_history(result)
    result["io_trigger"] = maybe_trigger_io(result, meta)
    return result


def evaluate_occupancy(source_key: str) -> Dict:
    """跑佔用判定 — return:
    {
      source, source_name, frame_w, frame_h, total, occupied, available,
      occupancy_rate, slots: [{id, label, occupied, conf, polygon}],
      detected: N (yolo 偵測到車數),
      error?: str
    }
    """
    meta = get_source_meta(source_key)
    slots_cfg = load_slots(source_key)

    frame = fetch_frame(source_key)
    if frame is None:
        return {"source": source_key,
                "source_name": meta.get("name", source_key),
                "error": "frame unavailable", "total": 0, "occupied": 0,
                "available": 0, "occupancy_rate": 0.0, "slots": [], "mode": "auto"}

    # 沒標 slot → mode 由 meta.detection_mode 控制 (預設 pklot,直接識空/有車車位)
    # mode='auto' 走 YOLO 累積推車位 / mode='pklot' 走 PKLot 二分類
    if not slots_cfg:
        mode = (meta.get("detection_mode") or os.getenv("PARKING_MODE", "pklot")).lower()
        if mode == "pklot":
            try:
                from services.parking_pklot_model import is_available as pklot_avail, evaluate_pklot
                if pklot_avail():
                    result = evaluate_pklot(source_key, frame, meta)
                    if result.get("total", 0) > 0:
                        record_to_history(result)
                        result["io_trigger"] = maybe_trigger_io(result, meta)
                        return result
            except Exception as e:
                print(f"[parking] pklot fail: {e}", flush=True)
        # mode='auto' 或 PKLot 失敗 fallback
        auto_result = _eval_auto_mode(source_key, frame, meta)
        auto_result["io_trigger"] = maybe_trigger_io(auto_result, meta)
        return auto_result

    h, w = frame.shape[:2]
    # 跑 yolo
    try:
        from detection.vehicle_detector import VehicleDetector
        yolo_local_roi = VehicleDetector(conf_threshold=0.12)
        detections = _yolo_sliced_detect(yolo_local_roi, frame)
    except Exception as e:
        return {"source": source_key, "error": f"yolo detect err: {e}",
                "total": len(slots_cfg), "occupied": 0, "available": 0,
                "occupancy_rate": 0.0, "slots": []}

    # 過濾車輛類別
    vehicle_classes = {"car", "truck", "bus", "heavy_truck", "light_truck",
                       "non_truck", "motorcycle"}
    vehicles = []
    for det in detections or []:
        cls = str(det.get("class_name") or "").lower()
        if cls not in vehicle_classes:
            continue
        bb = det.get("bbox", {})
        x1, y1 = int(bb.get("x1", 0)), int(bb.get("y1", 0))
        x2, y2 = int(bb.get("x2", 0)), int(bb.get("y2", 0))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        vehicles.append({"bbox": (x1, y1, x2, y2), "cx": cx, "cy": cy, "cls": cls})

    # 對每 slot 判定 + VLM 仲裁 hook (borderline conf 0.3~0.6 排入背景 queue)
    try:
        from services.parking_vlm_hook import enqueue as vlm_enqueue, get_verdict as vlm_get_verdict, start_worker as vlm_start_worker
        vlm_start_worker()  # idempotent
    except Exception:
        vlm_enqueue = None
        vlm_get_verdict = None
    slot_results = []
    for slot in slots_cfg:
        poly = slot.get("polygon") or []
        sid = str(slot["id"])
        if len(poly) < 3:
            slot_results.append({"id": slot["id"], "label": slot.get("label", str(slot["id"])),
                                 "occupied": False, "conf": 0.0, "polygon": poly})
            continue
        best_conf = 0.0
        max_iou_low = 0.0  # 追蹤「不到 0.15 但 > 0」的 overlap,作 borderline 訊號
        for v in vehicles:
            # 1. center-in-polygon
            if _point_in_polygon(v["cx"], v["cy"], poly):
                best_conf = max(best_conf, 0.95)
                continue
            # 2. polygon-bbox IoU > 0.15
            iou = _bbox_iou_with_polygon(v["bbox"], poly)
            if iou > 0.15:
                best_conf = max(best_conf, min(1.0, iou * 4))
            elif iou > 0:
                max_iou_low = max(max_iou_low, iou)
        # borderline → 排入 VLM 背景仲裁:
        # 情況 A: conf 0.30-0.60 真實 borderline
        # 情況 B: conf < 0.30 但有 vehicle overlap > 0.03 — 車邊緣壓到/部份 occ
        if vlm_enqueue and (
            (0.30 <= best_conf < 0.60) or
            (best_conf < 0.30 and max_iou_low > 0.03)
        ):
            try:
                vlm_enqueue(source_key, sid, poly)
            except Exception:
                pass
        # 取 cache 結果 (若有,附在 slot 給 UI 看,但不覆寫 occupied 判定 — VLM 2B 信心不夠)
        vlm_verdict = vlm_get_verdict(source_key, sid) if vlm_get_verdict else None
        slot_results.append({
            "id": slot["id"],
            "label": slot.get("label", str(slot["id"])),
            "occupied": best_conf >= 0.5,
            "conf": round(best_conf, 3),
            "polygon": poly,
            "vlm_verdict": vlm_verdict,
        })

    occupied = sum(1 for s in slot_results if s["occupied"])
    total = len(slot_results)
    available = total - occupied
    rate = (occupied / total * 100.0) if total else 0.0
    result = {
        "source": source_key,
        "source_name": meta.get("name", source_key),
        "frame_w": w, "frame_h": h,
        "total": total, "occupied": occupied, "available": available,
        "occupancy_rate": round(rate, 1),
        "detected_vehicles": len(vehicles),
        "slots": slot_results,
        "mode": "roi",
    }
    # counting hook: 用 YOLO 偵測到的 vehicles 餵 counter (line cross 判 enter/exit)
    counting_line = meta.get("counting_line") or None
    enter_normal = meta.get("counting_enter_normal") or "right"
    in_lot_now = None
    if counting_line and vehicles:
        try:
            from services.parking_counter import feed as counter_feed
            vehicles_for_counter = [{"bbox": {
                "x1": v["bbox"][0], "y1": v["bbox"][1],
                "x2": v["bbox"][2], "y2": v["bbox"][3]}} for v in vehicles]
            cnt = counter_feed(source_key, vehicles_for_counter, counting_line, enter_normal)
            result["counting"] = cnt.get("status")
            in_lot_now = int((cnt.get("status") or {}).get("in_lot") or 0)
        except Exception as e:
            print(f"[parking] counter err: {e}", flush=True)
    elif counting_line:
        # 本 frame 沒 vehicles 也讀目前累積 state (供交叉校正)
        try:
            from services.parking_counter import get_status as _counter_status
            in_lot_now = int(_counter_status(source_key).get("in_lot") or 0)
        except Exception:
            in_lot_now = None

    # Phase 3 交叉校正 — 兩個獨立「場內停放車數」估計差距大 → 偵測可疑 → 升級 VLM 仲裁
    # 主基準: vehicles_in_area (YOLO 看到、中心落在停車場區域內的車) vs slot_occupied
    #   兩者都不需計數線,靜態畫好停車場區域 mask 即可隨時比對 (車流計數不一定有/不一定抓得到進出)
    # 次基準: in_lot (進出累計) — 只有計數線校準後才納入
    # 註: detected_vehicles 是整幀 YOLO 原始(含路過/相鄰),會超估,只當參考不入校正
    area_mask = meta.get("parking_area_mask") or []
    exclusion_mask = meta.get("exclusion_mask") or []

    def _veh_in_area(cx: float, cy: float) -> bool:
        if area_mask and not _point_in_polygon(cx, cy, area_mask):
            return False
        if exclusion_mask and _point_in_polygon(cx, cy, exclusion_mask):
            return False
        return True

    vehicles_in_area = sum(1 for v in vehicles if _veh_in_area(v["cx"], v["cy"]))
    has_area_mask = bool(area_mask)
    # area 基準只在 mask 真的涵蓋到車時可信: vehicles_in_area=0 但有佔用 = mask 沒畫對,
    # 不是真背離 (避免假升級),標 suspect 提示去 editor 重畫區域
    area_mask_suspect = bool(has_area_mask and vehicles_in_area == 0 and occupied > 0)
    area_reliable = bool(has_area_mask and vehicles_in_area > 0)
    area_gap = abs(vehicles_in_area - occupied) if area_reliable else None
    calibrated = bool(counting_line) and in_lot_now is not None and in_lot_now > 0
    count_gap = abs(occupied - in_lot_now) if calibrated else None
    escalated = bool(
        (area_reliable and area_gap >= CROSS_GAP_ESCALATE) or
        (count_gap is not None and count_gap >= CROSS_GAP_ESCALATE)
    )
    if escalated and vlm_enqueue:
        for s in slot_results:
            poly_s = s.get("polygon")
            conf_s = s.get("conf", 0.0)
            if poly_s and len(poly_s) >= 3 and 0.15 <= conf_s <= 0.85:
                try:
                    vlm_enqueue(source_key, str(s["id"]), poly_s)
                except Exception:
                    pass
    result["vehicles_in_area"] = vehicles_in_area
    result["cross_validation"] = {
        "slot_occupied": occupied,
        "vehicles_in_area": vehicles_in_area,
        "detected_vehicles": len(vehicles),
        "area_gap": area_gap,
        "has_area_mask": has_area_mask,
        "area_reliable": area_reliable,
        "area_mask_suspect": area_mask_suspect,
        "in_lot": in_lot_now,
        "count_gap": count_gap,
        "calibrated": calibrated,
        "escalated": escalated,
        "escalate_threshold": CROSS_GAP_ESCALATE,
    }
    record_to_history(result)
    result["io_trigger"] = maybe_trigger_io(result, meta)
    return result
