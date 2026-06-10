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

# 預設 demo 配置 (當 data/parking_slots.json 不存在時 fallback,跟 code 一起 deploy)
# 實際生產建議寫 data/parking_slots.json 覆蓋 (per-機器自訂)
_DEFAULT_CONFIG = {
    "twipcam:tpe-005013": {
        "name": "百齡橋停車場 (台北市政府 PB048)",
        "image_url": "https://c01.twipcam.com/cam/snapshot/tpe-005013.jpg",
        "note": "預設 8 格示範,進「編輯車位」可補齊 80+ 格",
        "slots": [
            {"id": "A1", "label": "A1", "polygon": [[5, 250], [105, 230], [120, 360], [10, 395]]},
            {"id": "A2", "label": "A2", "polygon": [[110, 235], [200, 220], [215, 350], [125, 365]]},
            {"id": "A3", "label": "A3", "polygon": [[205, 220], [285, 210], [305, 340], [220, 350]]},
            {"id": "B1", "label": "B1", "polygon": [[235, 165], [305, 158], [320, 215], [248, 222]]},
            {"id": "B2", "label": "B2", "polygon": [[310, 158], [380, 152], [395, 215], [325, 220]]},
            {"id": "B3", "label": "B3", "polygon": [[385, 150], [450, 145], [465, 210], [400, 215]]},
            {"id": "C1", "label": "C1", "polygon": [[355, 95], [415, 92], [428, 145], [368, 148]]},
            {"id": "C2", "label": "C2", "polygon": [[420, 92], [480, 88], [495, 145], [435, 148]]},
        ],
    }
}

_LATEST_FRAME_CACHE: Dict[str, Tuple[float, np.ndarray]] = {}
_FRAME_CACHE_TTL = 2.0  # 同 source 2 秒內共用 frame,降低 fetch 次數
_CACHE_LOCK = threading.Lock()

# 歷史寫入 throttle (per-source 5 分鐘記一筆)
_HISTORY_LAST_AT: Dict[str, float] = {}
_HISTORY_THROTTLE_SEC = 300.0
_HISTORY_LOCK = threading.Lock()

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
    if source_key.startswith("twipcam:"):
        import requests as _req
        url = source_key.split(":", 1)[1]
        if not url.startswith("http"):
            meta = get_source_meta(source_key)
            url = meta.get("image_url", "")
        try:
            # bypass_cache 時加 ts query + no-cache header 強制拿新 frame
            if bypass_cache:
                sep = "&" if "?" in url else "?"
                fetch_url = f"{url}{sep}_t={int(now*1000)}"
                headers = {"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"}
            else:
                fetch_url = url
                headers = {}
            r = _req.get(fetch_url, timeout=8, headers=headers)
            if r.status_code == 200 and len(r.content) > 5000:
                frame = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
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
    if not slots_cfg:
        return {"source": source_key, "error": "no slots configured", "total": 0,
                "occupied": 0, "available": 0, "occupancy_rate": 0.0, "slots": []}

    frame = fetch_frame(source_key)
    if frame is None:
        # frame 拿不到 → 全 unknown,但仍 return slots 結構
        return {
            "source": source_key, "source_name": meta.get("name", source_key),
            "error": "frame unavailable", "total": len(slots_cfg),
            "occupied": 0, "available": 0, "occupancy_rate": 0.0,
            "slots": [{"id": s["id"], "label": s.get("label", str(s["id"])),
                       "occupied": None, "conf": 0.0, "polygon": s["polygon"]}
                      for s in slots_cfg],
        }

    h, w = frame.shape[:2]
    # 跑 yolo
    try:
        from api.routes.lpr_visual import get_yolo
        yolo = get_yolo()
        detections = yolo.detect(frame)
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

    # 對每 slot 判定
    slot_results = []
    for slot in slots_cfg:
        poly = slot.get("polygon") or []
        if len(poly) < 3:
            slot_results.append({"id": slot["id"], "label": slot.get("label", str(slot["id"])),
                                 "occupied": False, "conf": 0.0, "polygon": poly})
            continue
        best_conf = 0.0
        for v in vehicles:
            # 1. center-in-polygon
            if _point_in_polygon(v["cx"], v["cy"], poly):
                best_conf = max(best_conf, 0.95)
                continue
            # 2. polygon-bbox IoU > 0.15
            iou = _bbox_iou_with_polygon(v["bbox"], poly)
            if iou > 0.15:
                best_conf = max(best_conf, min(1.0, iou * 4))
        slot_results.append({
            "id": slot["id"],
            "label": slot.get("label", str(slot["id"])),
            "occupied": best_conf >= 0.5,
            "conf": round(best_conf, 3),
            "polygon": poly,
        })

    occupied = sum(1 for s in slot_results if s["occupied"])
    total = len(slot_results)
    available = total - occupied
    rate = (occupied / total * 100.0) if total else 0.0
    return {
        "source": source_key,
        "source_name": meta.get("name", source_key),
        "frame_w": w, "frame_h": h,
        "total": total, "occupied": occupied, "available": available,
        "occupancy_rate": round(rate, 1),
        "detected_vehicles": len(vehicles),
        "slots": slot_results,
    }
