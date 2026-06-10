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
    slots: List[SlotPolygon]


router = APIRouter(prefix="/api/parking", tags=["parking"])


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
    if body.image_url:
        entry["image_url"] = body.image_url
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
