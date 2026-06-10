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
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
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
    """重置 auto mode 累積位置 (例如停車場攝影機角度變了想重來)"""
    n = reset_auto_positions(source)
    return {"source": source, "cleared": n}


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


@router.get("/snapshot/raw")
def get_snapshot_raw(source: str = Query(..., description="source key")):
    """回 raw frame (沒 overlay) — 供 ROI 編輯器當底圖"""
    import numpy as np
    frame = fetch_frame(source)
    if frame is None:
        ph = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(ph, "FRAME UNAVAILABLE", (24, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        ok, buf = cv2.imencode(".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/slots/auto")
def auto_detect_slots(source: str = Query(..., description="source key"),
                      expand: float = Query(0.08, ge=0.0, le=0.5,
                                            description="bbox 外擴比例"),
                      conf: float = Query(0.15, ge=0.05, le=0.9,
                                          description="YOLO conf threshold"),
                      frames: int = Query(5, ge=1, le=20,
                                          description="抓即時影像 frame 數累積"),
                      interval_sec: float = Query(10.0, ge=1.0, le=120.0,
                                                  description="frame 間隔秒 (TwiPcam 通常 30-60 秒才會更新一次,間隔太短拿到相同 frame)"),
                      merge_overlap: bool = Query(True)):
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
def get_snapshot(source: str = Query(..., description="source key")):
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

    for slot in result.get("slots") or []:
        poly = slot.get("polygon") or []
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        if slot.get("occupied"):
            fill_color = (0, 60, 200)        # 紅 (BGR)
            stroke_color = (0, 80, 240)
            label_color = (255, 255, 255)
        else:
            fill_color = (60, 180, 80)        # 綠
            stroke_color = (80, 220, 100)
            label_color = (0, 0, 0)
        # 半透明填色
        slot_overlay = frame.copy()
        cv2.fillPoly(slot_overlay, [pts], fill_color)
        cv2.addWeighted(slot_overlay, 0.40, frame, 0.60, 0, frame)
        cv2.polylines(frame, [pts], True, stroke_color, 2)
        # label 在 polygon 中心
        cx = int(sum(p[0] for p in poly) / len(poly))
        cy = int(sum(p[1] for p in poly) / len(poly))
        label = str(slot.get("label", slot.get("id", "")))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (cx - tw // 2 - 3, cy - th // 2 - 3),
                      (cx + tw // 2 + 3, cy + th // 2 + 3),
                      stroke_color, -1)
        cv2.putText(frame, label, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

    # HUD — 中文用 PIL (cv2.putText 不支援)
    name = result.get('source_name', '')
    occ = result.get('occupied', 0)
    tot = result.get('total', 0)
    rate = result.get('occupancy_rate', 0)
    hud_line1 = name
    hud_line2 = f"已佔用 {occ} / 總車位 {tot}  ·  佔用率 {rate}%"
    cv2.rectangle(frame, (8, 8), (440, 56), (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (440, 56), (60, 220, 130), 1)
    try:
        from api.routes.vision_eye import _draw_zh_batch
        frame = _draw_zh_batch(frame, [
            (hud_line1, (18, 12), 15, (60, 220, 130)),
            (hud_line2, (18, 33), 13, (240, 240, 240)),
        ])
    except Exception:
        cv2.putText(frame, hud_line2, (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")
