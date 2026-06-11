"""
停車場車位佔用 endpoints.

- GET  /api/parking/sources                    列已配置 source
- GET  /api/parking/occupancy?source=<key>     回 total/occupied/available/rate/slots[]
- GET  /api/parking/snapshot?source=<key>      回 overlay 後 JPEG (slot polygon 著色)
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import time
import threading
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from services.parking_occupancy import (
    evaluate_occupancy, fetch_frame, load_slots, get_source_meta,
    auto_session_start, auto_session_status, auto_session_stop_and_get,
    reset_auto_positions,
    _CONFIG_PATH,
)


class SlotPolygon(BaseModel):
    id: str
    label: Optional[str] = None
    polygon: List[List[float]]


class SlotsSaveBody(BaseModel):
    source: str
    name: Optional[str] = None
    image_url: Optional[str] = None
    stream_url: Optional[str] = None
    slots: List[SlotPolygon]


class MaskSaveBody(BaseModel):
    source: str
    parking_area_mask: List[List[float]]   # polygon [[x,y], ...] or [] to clear


class ExclusionMaskBody(BaseModel):
    source: str
    exclusion_mask: List[List[float]]      # 不偵測區 polygon or [] to clear


class IOTriggerBody(BaseModel):
    source: str
    enabled: bool = True
    module_id: str                          # io_tcp module id (例 site_106)
    do_ch: int = 0
    threshold_rate: float = 95.0            # 佔用率 >= 此值觸發
    pulse_ms: int = 1000


router = APIRouter(prefix="/api/parking", tags=["parking"])


_TWIPCAM_LIST_CACHE: dict = {"ts": 0.0, "data": []}
_TWIPCAM_TTL = 3600.0  # 1 小時 cache


def _load_twipcam_list() -> list:
    import time as _t
    import requests as _req
    now = _t.time()
    if _TWIPCAM_LIST_CACHE["data"] and (now - _TWIPCAM_LIST_CACHE["ts"]) < _TWIPCAM_TTL:
        return _TWIPCAM_LIST_CACHE["data"]
    try:
        r = _req.get("https://www.twipcam.com/api/v1/cam-list.json", timeout=15)
        if r.status_code == 200:
            data = r.json()
            _TWIPCAM_LIST_CACHE["data"] = data
            _TWIPCAM_LIST_CACHE["ts"] = now
            return data
    except Exception as e:
        print(f"[parking] twipcam list fetch err: {e}", flush=True)
    return _TWIPCAM_LIST_CACHE["data"]


@router.get("/twipcam/search")
def twipcam_search(q: str = Query("", description="關鍵字 (名稱/id 模糊比對)"),
                   limit: int = Query(50, ge=1, le=500)):
    """從 TwiPcam cam-list.json 搜尋 cam (cache 1h).回 id/name/lat/lon/cam_url + snapshot_url"""
    data = _load_twipcam_list()
    q = (q or "").strip().lower()
    out = []
    for c in data:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", ""))
        name = str(c.get("name", ""))
        if q and (q not in cid.lower()) and (q not in name.lower()):
            continue
        out.append({
            "id": cid,
            "name": name,
            "lat": c.get("lat"),
            "lon": c.get("lon"),
            "cam_url": c.get("cam_url", ""),
            "snapshot_url": f"https://c01.twipcam.com/cam/snapshot/{cid}.jpg",
        })
        if len(out) >= limit:
            break
    return {"total_in_list": len(data), "matched": len(out), "items": out}


@router.get("/source_meta")
def get_source_meta_route(source: str = Query(...)):
    """回 source meta (name/image_url/stream_url/parking_area_mask/slots count) — 供編輯器讀取"""
    meta = get_source_meta(source) or {}
    return {
        "source": source,
        "name": meta.get("name", ""),
        "image_url": meta.get("image_url", ""),
        "stream_url": meta.get("stream_url", ""),
        "parking_area_mask": meta.get("parking_area_mask", []),
        "exclusion_mask": meta.get("exclusion_mask", []),
        "slot_count": len(meta.get("slots", [])) if isinstance(meta.get("slots"), list) else 0,
    }


@router.get("/sources")
def list_sources():
    """已配置 source 列表 (key/name/slot 數)"""
    if not os.path.exists(_CONFIG_PATH):
        from services.parking_occupancy import _DEFAULT_CONFIG
        d = _DEFAULT_CONFIG
    else:
      try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
      except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    out = []
    for key, meta in d.items():
        out.append({
            "key": key,
            "name": meta.get("name", key),
            "image_url": meta.get("image_url", ""),
            "slot_count": len(meta.get("slots", [])),
        })
    return {"sources": out}


@router.get("/occupancy")
def get_occupancy(source: str = Query(..., description="source key e.g. twipcam:tpe-005013")):
    """跑 yolo 判定每 slot occupied/empty + 統計"""
    return evaluate_occupancy(source)


@router.post("/auto/reset")
def auto_reset(source: str = Query(...)):
    """重置累積位置 (auto + PKLot 都清)"""
    n = reset_auto_positions(source)
    try:
        from services.parking_pklot_model import reset_pklot_positions
        n_pklot = reset_pklot_positions(source)
    except Exception:
        n_pklot = 0
    return {"source": source, "cleared_auto": n, "cleared_pklot": n_pklot}


@router.get("/history")
def get_history(source: str = Query(..., description="source key"),
                hours: int = Query(24, ge=1, le=720, description="近 N 小時")):
    """佔用率歷史時序 — 從 ParkingSample table 撈"""
    from datetime import datetime, timedelta
    from api.models import SessionLocal, ParkingSample
    start = datetime.utcnow() - timedelta(hours=hours)
    db = SessionLocal()
    try:
        rows = (db.query(ParkingSample)
                  .filter(ParkingSample.source == source,
                          ParkingSample.created_at >= start)
                  .order_by(ParkingSample.created_at.asc())
                  .all())
        return {
            "source": source,
            "hours": hours,
            "count": len(rows),
            "samples": [
                {
                    "ts": r.created_at.isoformat() if r.created_at else None,
                    "total": r.total,
                    "occupied": r.occupied,
                    "available": r.available,
                    "occupancy_rate": r.occupancy_rate,
                    "detected_vehicles": r.detected_vehicles,
                }
                for r in rows
            ],
        }
    finally:
        db.close()


@router.get("/stream")
def stream_parking(source: str = Query(..., description="source key"),
                    show_boxes: bool = Query(True),
                    show_vehicles: bool = Query(True),
                    eval_interval_sec: float = Query(2.0, ge=0.5, le=30.0,
                        description="跑 evaluate (PKLot+YOLO) 間隔,期間用 cached overlay")):
    """MJPEG 串流 (multipart/x-mixed-replace):
    - 高 fps 拉 frame (stream_url cv2.VideoCapture 開一次持續 read)
    - 低 fps 跑 evaluate (cache slots + car bboxes)
    - 每 frame 用 cached overlay 渲染,不卡頓"""
    from services.parking_pklot_model import _yolo_car_centers as _yolo_cars

    def render_overlay(frame: np.ndarray, slots: list, cars: list):
        if show_vehicles:
            for c in cars:
                cx, cy, x1, y1, x2, y2 = c
                cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 210, 30), 1)
                cv2.circle(frame, (cx, cy), 3, (210, 210, 30), -1)
        if show_boxes:
            for slot in slots:
                poly = slot.get("polygon") or []
                if len(poly) < 3:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cx = int(sum(p[0] for p in poly) / len(poly))
                cy = int(sum(p[1] for p in poly) / len(poly))
                label = str(slot.get("label", slot.get("id", "")))
                if not slot.get("occupied"):
                    slot_overlay = frame.copy()
                    cv2.fillPoly(slot_overlay, [pts], (60, 180, 80))
                    cv2.addWeighted(slot_overlay, 0.40, frame, 0.60, 0, frame)
                    cv2.polylines(frame, [pts], True, (80, 220, 100), 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th // 2 - 4),
                                  (cx + tw // 2 + 4, cy + th // 2 + 4),
                                  (80, 220, 100), -1)
                    cv2.putText(frame, label, (cx - tw // 2, cy + th // 2 + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        return frame

    def gen():
        cached_slots = []
        cached_cars = []
        last_eval = 0.0
        # 開 cap (stream_url 模式) 持續拉 frame,沒設則用 fetch_frame fallback
        meta = get_source_meta(source) or {}
        stream_url = meta.get("stream_url", "")
        cap = None
        if stream_url:
            try:
                cap = cv2.VideoCapture(stream_url)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            except Exception:
                cap = None
        try:
            while True:
                # fetch frame (cap 模式快 / fetch_frame fallback 慢)
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # cap 斷線 retry
                        cap.release()
                        cap = cv2.VideoCapture(stream_url)
                        time.sleep(0.5)
                        continue
                else:
                    frame = fetch_frame(source)
                    if frame is None:
                        time.sleep(1.0)
                        continue
                # 低 fps evaluate (跨 frame 用 cache)
                now = time.time()
                if (now - last_eval) >= eval_interval_sec:
                    try:
                        result = evaluate_occupancy(source)
                        cached_slots = result.get("slots", []) or []
                        cached_cars = _yolo_cars(frame, conf=0.12) if show_vehicles else []
                    except Exception as e:
                        print(f"[parking stream] eval err: {e}", flush=True)
                    last_eval = now
                # render overlay (cheap)
                frame = render_overlay(frame, cached_slots, cached_cars)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + buf.tobytes() + b'\r\n')
                time.sleep(0.04)   # ~25 fps cap
        finally:
            if cap is not None:
                try: cap.release()
                except Exception: pass

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@router.get("/snapshot/raw")
def get_snapshot_raw(source: str = Query(..., description="source key")):
    """回 raw frame (沒 overlay) — 供 ROI 編輯器當底圖"""
    frame = fetch_frame(source)
    if frame is None:
        ph = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(ph, "FRAME UNAVAILABLE", (24, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        ok, buf = cv2.imencode(".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


def _hough_lines_to_slots(frame: np.ndarray) -> list:
    """Hough lines 找白色車格線 → 推 parking slot polygon.
    流程:
    1. 灰階 + adaptive threshold 找白色線條
    2. HoughLinesP 抓線段
    3. 依角度分群 (水平 / 垂直)
    4. 相鄰平行線配對成矩形車格"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 增強白線對比 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # adaptive threshold 找白色 (背景柏油暗,線條亮)
    binary = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 25, -10)
    # 膨脹一下接斷線
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)
    # Hough 找線段
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 180,
                             threshold=30, minLineLength=20, maxLineGap=8)
    if lines is None:
        return []

    # 分群: 水平 (angle 接近 0 / 180) vs 垂直 (接近 90)
    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 25 or angle > 155:
            horizontal.append((min(x1, x2), max(x1, x2), (y1 + y2) // 2))
        elif 65 < angle < 115:
            vertical.append((min(y1, y2), max(y1, y2), (x1 + x2) // 2))

    # 相鄰垂直線配對成車格 (簡化:每對相鄰垂直線 + 上下邊 → 矩形)
    vertical.sort(key=lambda v: v[2])  # 按 x 排
    slots = []
    for i in range(len(vertical) - 1):
        v1 = vertical[i]
        for j in range(i + 1, min(i + 4, len(vertical))):
            v2 = vertical[j]
            x1, x2 = v1[2], v2[2]
            width = x2 - x1
            if width < 20 or width > 80:  # 車格寬度範圍 px
                continue
            # 取兩條線 y 重疊區域
            y_top = max(v1[0], v2[0])
            y_bot = min(v1[1], v2[1])
            height = y_bot - y_top
            if height < 25 or height > 120:
                continue
            # ratio check (車格通常高 > 寬,俯角拍攝)
            if height / width < 0.6:
                continue
            slots.append({
                "polygon": [[int(x1), int(y_top)], [int(x2), int(y_top)],
                            [int(x2), int(y_bot)], [int(x1), int(y_bot)]],
                "_w": int(width), "_h": int(height),
            })

    # 去重 (IoU > 0.4)
    def _iou(a, b):
        ax1, ay1 = a["polygon"][0]; ax2, ay2 = a["polygon"][2]
        bx1, by1 = b["polygon"][0]; bx2, by2 = b["polygon"][2]
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        ar = (ax2 - ax1) * (ay2 - ay1)
        br = (bx2 - bx1) * (by2 - by1)
        return inter / max(1, ar + br - inter)
    kept = []
    for s in slots:
        if any(_iou(s, k) > 0.4 for k in kept):
            continue
        kept.append(s)
    return kept


@router.post("/slots/auto")
def auto_detect_slots(source: str = Query(..., description="source key"),
                      mode: str = Query("pklot", description="pklot | yolo | hough"),
                      expand: float = Query(0.0, ge=0.0, le=0.5),
                      conf: float = Query(0.08, ge=0.05, le=0.9),
                      frames: int = Query(5, ge=1, le=20),
                      interval_sec: float = Query(10.0, ge=1.0, le=120.0),
                      merge_overlap: bool = Query(True)):
    """一鍵偵測車位 polygon — 給 user 用 ROI 編輯器當基底.
    mode=pklot (預設): 用 PKLot model 找所有 parking slot (空+有車) — 對「車位位置」最準
    mode=yolo: 用 YOLO car detect 找車輛 bbox 當車位 — 只能找到有車的位置.
    回 slots list 後 user 在編輯器可調整 / 刪除 / 儲存."""
    if mode.lower() == "hough":
        # OpenCV Hough lines 找白色車格線,跳過 model domain shift
        frame = fetch_frame(source)
        if frame is None:
            raise HTTPException(status_code=503, detail="frame unavailable")
        hough_slots = _hough_lines_to_slots(frame)
        h_img, w_img = frame.shape[:2]
        slots_out = []
        for idx, s in enumerate(hough_slots):
            lbl = f"P{idx + 1}"
            slots_out.append({"id": lbl, "label": lbl, "polygon": s["polygon"]})
        return {
            "source": source, "mode": "hough",
            "frame_w": w_img, "frame_h": h_img,
            "detected_slots": len(slots_out),
            "slots": slots_out,
            "note": f"Hough lines 找到 {len(slots_out)} 個白線車格,在編輯器內修正後儲存",
        }

    if mode.lower() == "pklot":
        # 直接拉一張 frame 跑 sliced PKLot,結果就是所有車位 polygon
        try:
            from services.parking_pklot_model import is_available, detect_slots
            if is_available():
                frame = fetch_frame(source)
                if frame is None:
                    raise HTTPException(status_code=503, detail="frame unavailable")
                h_img, w_img = frame.shape[:2]
                dets = detect_slots(frame, conf=conf)
                slots_out = []
                for idx, d in enumerate(dets):
                    x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                    lbl = f"P{idx + 1}"
                    slots_out.append({
                        "id": lbl, "label": lbl,
                        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    })
                return {
                    "source": source, "mode": "pklot",
                    "frame_w": w_img, "frame_h": h_img,
                    "detected_slots": len(slots_out),
                    "slots": slots_out,
                    "note": f"PKLot 找到 {len(slots_out)} 個車位 polygon — 在編輯器內刪除誤判 / 拖拉調整 / 儲存,儲存後 evaluate 會走 ROI mode 對每格跑 YOLO,空車位偵測最準",
                }
        except Exception as e:
            print(f"[parking] pklot auto detect fall back yolo: {e}", flush=True)
    # mode='yolo' 或 PKLot 失敗 → 沿用 YOLO car detect 流程
    """多 frame 累積自動偵測車位 — 抓即時影像 N 張,每張間隔 K 秒,
    YOLO detect 全部車輛 bbox 累積後合併 (IoU > 0.5 視為同位置).
    停車場滿時車輛位置變化 → 多 frame 涵蓋更多車位."""
    import time as _t
    import numpy as np

    vehicle_classes = {"car", "truck", "bus", "heavy_truck", "light_truck", "non_truck"}
    all_raw_boxes: list = []
    frame_info = []
    w_img, h_img = 0, 0
    yolo_local = None
    try:
        from detection.vehicle_detector import VehicleDetector
        yolo_local = VehicleDetector(conf_threshold=float(conf))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"yolo init err: {e}")

    for fi in range(frames):
        if fi > 0:
            _t.sleep(interval_sec)
        # 第 2 張之後 bypass cache 強制重拉即時影像
        frame = fetch_frame(source, bypass_cache=(fi > 0))
        if frame is None:
            continue
        if w_img == 0:
            h_img, w_img = frame.shape[:2]
        try:
            detections = yolo_local.detect(frame)
        except Exception as e:
            frame_info.append({"frame": fi, "err": str(e)})
            continue
        n_this = 0
        for det in detections or []:
            cls = str(det.get("class_name") or "").lower()
            if cls not in vehicle_classes:
                continue
            bb = det.get("bbox", {})
            x1 = int(bb.get("x1", 0)); y1 = int(bb.get("y1", 0))
            x2 = int(bb.get("x2", 0)); y2 = int(bb.get("y2", 0))
            if x2 <= x1 or y2 <= y1:
                continue
            all_raw_boxes.append([x1, y1, x2, y2])
            n_this += 1
        frame_info.append({"frame": fi, "vehicles": n_this})

    if w_img == 0:
        raise HTTPException(status_code=503, detail="所有 frame 都 fetch 失敗")

    raw_boxes = all_raw_boxes

    # 合併過度重疊 (IoU > 0.6)
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        ar = (a[2]-a[0]) * (a[3]-a[1])
        br = (b[2]-b[0]) * (b[3]-b[1])
        return inter / max(1, ar + br - inter)

    if merge_overlap:
        # 多 frame 累積後 merge: IoU > 0.5 視為同位置 (合併用平均座標而非聯集,
        # 避免框越合越大),累積出現越多次的位置可信度越高
        merged = []   # [[x1,y1,x2,y2,count], ...]
        for bx in raw_boxes:
            absorbed = False
            for i, m in enumerate(merged):
                if _iou(bx, m[:4]) > 0.5:
                    cnt = m[4] + 1
                    # 加權移動平均 (新框 vs 累積)
                    nx1 = (m[0] * m[4] + bx[0]) / cnt
                    ny1 = (m[1] * m[4] + bx[1]) / cnt
                    nx2 = (m[2] * m[4] + bx[2]) / cnt
                    ny2 = (m[3] * m[4] + bx[3]) / cnt
                    merged[i] = [int(nx1), int(ny1), int(nx2), int(ny2), cnt]
                    absorbed = True
                    break
            if not absorbed:
                merged.append([bx[0], bx[1], bx[2], bx[3], 1])
        # 過濾僅出現一次 (frames > 1 時) — 減少 false positive
        if frames > 1:
            merged = [m for m in merged if m[4] >= 1]   # 暫不過濾,讓 user 自己刪
        boxes = [m[:4] for m in merged]
    else:
        boxes = raw_boxes

    # 外擴 + 轉 polygon
    slots_out = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        w = x2 - x1; h = y2 - y1
        dx = int(w * expand); dy = int(h * expand)
        nx1 = max(0, x1 - dx); ny1 = max(0, y1 - dy)
        nx2 = min(w_img - 1, x2 + dx); ny2 = min(h_img - 1, y2 + dy)
        lbl = f"P{idx + 1}"
        slots_out.append({
            "id": lbl,
            "label": lbl,
            "polygon": [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]],
        })
    return {
        "source": source,
        "frame_w": w_img, "frame_h": h_img,
        "frames_used": frames,
        "interval_sec": interval_sec,
        "detected_vehicles": len(raw_boxes),
        "merged_slots": len(boxes),
        "slots": slots_out,
        "frame_info": frame_info,
        "note": f"從 {frames} 張即時影像累積 yolo 偵測產生,可在編輯器調整或刪除",
    }


@router.post("/slots/auto_session/start")
def slots_auto_start(source: str = Query(...),
                     conf: float = Query(0.15, ge=0.05, le=0.9),
                     expand: float = Query(0.08, ge=0.0, le=0.5),
                     interval_sec: float = Query(30.0, ge=5.0, le=300.0),
                     max_frames: int = Query(60, ge=1, le=500)):
    """啟動背景累積偵測 — thread 每 interval_sec 拉即時影像 + yolo + merge,
    max_frames 後自動停 (60 frames × 30s = 30 分鐘 max)."""
    return auto_session_start(source, conf=conf, expand=expand,
                               interval_sec=interval_sec, max_frames=max_frames)


@router.get("/slots/auto_session/status")
def slots_auto_status(source: str = Query(...)):
    return auto_session_status(source)


@router.post("/slots/auto_session/stop")
def slots_auto_stop(source: str = Query(...)):
    """停止背景 session 並回 merged slots polygon list"""
    return auto_session_stop_and_get(source)


@router.post("/io_trigger/save")
def save_io_trigger(body: IOTriggerBody):
    """寫入 source 的 io_trigger 設定 (高佔用率自動觸發 io_tcp DO pulse).
    enabled=false 或 module_id 空 → 清除."""
    existing: dict = {}
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    else:
        from services.parking_occupancy import _DEFAULT_CONFIG
        existing = json.loads(json.dumps(_DEFAULT_CONFIG))

    entry = existing.get(body.source) or {}
    if body.enabled and body.module_id:
        entry["io_trigger"] = {
            "enabled": True,
            "module_id": body.module_id,
            "do_ch": int(body.do_ch),
            "threshold_rate": float(body.threshold_rate),
            "pulse_ms": int(body.pulse_ms),
        }
    else:
        entry.pop("io_trigger", None)
    existing[body.source] = entry

    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return {"ok": True, "source": body.source,
            "io_trigger": entry.get("io_trigger") or None}


@router.post("/exclusion_mask/save")
def save_exclusion_mask(body: ExclusionMaskBody):
    """不偵測區 polygon — 在裡面的 slot detection 過濾掉.
    polygon=[] 清除."""
    existing: dict = {}
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    else:
        from services.parking_occupancy import _DEFAULT_CONFIG
        existing = json.loads(json.dumps(_DEFAULT_CONFIG))

    entry = existing.get(body.source) or {}
    if body.exclusion_mask:
        entry["exclusion_mask"] = body.exclusion_mask
    else:
        entry.pop("exclusion_mask", None)
    existing[body.source] = entry

    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return {"ok": True, "source": body.source,
            "exclusion_points": len(body.exclusion_mask)}


@router.post("/mask/save")
def save_mask(body: MaskSaveBody):
    """寫入 source 的停車場區域 mask polygon (用來過濾非停車場區域誤判).
    polygon=[] 清除 mask."""
    existing: dict = {}
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    else:
        from services.parking_occupancy import _DEFAULT_CONFIG
        existing = json.loads(json.dumps(_DEFAULT_CONFIG))

    entry = existing.get(body.source) or {}
    if body.parking_area_mask:
        entry["parking_area_mask"] = body.parking_area_mask
    else:
        entry.pop("parking_area_mask", None)
    existing[body.source] = entry

    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return {"ok": True, "source": body.source,
            "mask_points": len(body.parking_area_mask)}


@router.post("/slots/save")
def save_slots(body: SlotsSaveBody):
    """覆寫 data/parking_slots.json 內該 source 的 slots."""
    # 讀現有 (或從 default 起手)
    existing: dict = {}
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    else:
        from services.parking_occupancy import _DEFAULT_CONFIG
        existing = json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy

    entry = existing.get(body.source) or {}
    if body.name:
        entry["name"] = body.name
    if body.image_url is not None:
        entry["image_url"] = body.image_url
    if body.stream_url is not None:
        entry["stream_url"] = body.stream_url
    entry["slots"] = [s.model_dump() for s in body.slots]
    existing[body.source] = entry

    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return {"ok": True, "source": body.source, "saved": len(body.slots)}


@router.get("/snapshot")
def get_snapshot(source: str = Query(..., description="source key"),
                  show_boxes: bool = Query(True, description="畫車位 polygon (空車位 + 號碼)"),
                  show_vehicles: bool = Query(True, description="畫車輛 YOLO bbox (青框)")):
    """回 frame + slot polygon overlay (綠=空 / 紅=佔用) 的 JPEG"""
    result = evaluate_occupancy(source)
    frame = fetch_frame(source)
    if frame is None:
        ph = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(ph, "PARKING FRAME UNAVAILABLE", (24, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        ok, buf = cv2.imencode(".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    # 暗化 18% 突顯 overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # 車輛 bbox (青框) — 可 toggle
    if show_vehicles:
        try:
            from services.parking_pklot_model import _yolo_car_centers
            car_centers = _yolo_car_centers(frame, conf=0.12)
            for c in car_centers:
                cx, cy, x1, y1, x2, y2 = c
                cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 210, 30), 1)
                cv2.circle(frame, (cx, cy), 3, (210, 210, 30), -1)
        except Exception as e:
            print(f"[parking] snapshot yolo render err: {e}", flush=True)

    if not show_boxes:
        # 不畫車位 ROI,直接 encode 回傳
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    for slot in result.get("slots") or []:
        poly = slot.get("polygon") or []
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cx = int(sum(p[0] for p in poly) / len(poly))
        cy = int(sum(p[1] for p in poly) / len(poly))
        label = str(slot.get("label", slot.get("id", "")))

        if not slot.get("occupied"):
            # 空車位: 畫綠色框 + 號碼 (空格少所以顯示)
            fill_color = (60, 180, 80)
            stroke_color = (80, 220, 100)
            slot_overlay = frame.copy()
            cv2.fillPoly(slot_overlay, [pts], fill_color)
            cv2.addWeighted(slot_overlay, 0.40, frame, 0.60, 0, frame)
            cv2.polylines(frame, [pts], True, stroke_color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th // 2 - 4),
                          (cx + tw // 2 + 4, cy + th // 2 + 4),
                          stroke_color, -1)
            cv2.putText(frame, label, (cx - tw // 2, cy + th // 2 + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        # 有車的車位:不畫框也不畫號碼 (避免畫面太擠),已用 YOLO 青框標示車輛

    # HUD 文字移除 — 影像保持乾淨,統計顯示在 UI 旁邊
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")
