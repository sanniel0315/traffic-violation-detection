#!/usr/bin/env python3
"""VisionEye 風格雷達 overlay — 在 live frame 上從鏡頭視角 (frame 底部中心)
拉線到每個 detected vehicle bbox 中心,加 class label + speed。

對應 Ultralytics solutions.VisionEye 視覺語彙,但用本專案既有 YOLO detector
跟 frame fallback chain (frigate / go2rtc / RTSP)。

endpoint:
- GET /api/vision_eye/snapshot/{camera_id} → 單張 overlay 後 JPEG
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/workspace')

from api.utils.camera_stream import resolve_analysis_source

router = APIRouter(prefix="/api/vision_eye", tags=["vision-eye"])

# ─── 中文字體 (PIL — cv2.putText 不支援中文,會輸出 ???) ───
_FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_FONT_PATH = next((p for p in _FONT_PATH_CANDIDATES if os.path.exists(p)), None)
_FONT_CACHE: dict = {}


def _font(size: int):
    if not _FONT_PATH:
        return None
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            _FONT_CACHE[size] = None
    return _FONT_CACHE[size]


def _draw_zh_batch(frame: np.ndarray, items: list) -> np.ndarray:
    """batch 寫中文 — items = [(text, (x,y), size, (B,G,R)), ...]
    回傳新 frame (in-place 改寫 BGR np.ndarray)。"""
    if not items or not _FONT_PATH:
        return frame
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, pos, size, c_bgr in items:
        font = _font(size)
        if not font or not text:
            continue
        rgb = (int(c_bgr[2]), int(c_bgr[1]), int(c_bgr[0]))
        draw.text(pos, text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# 中文 class label
_CLS_ZH = {
    'car': '小客車', 'non_truck': '小客車',
    'truck': '貨車', 'heavy_truck': '大貨車', 'light_truck': '小貨車',
    'bus': '大客車',
    'motorcycle': '機車', 'bike': '腳踏車', 'bicycle': '腳踏車',
    'person': '行人',
}

# 類別配色 (BGR for opencv)
_CLS_COLOR = {
    'car': (255, 210, 60), 'non_truck': (255, 210, 60),       # 青藍
    'truck': (60, 200, 250), 'heavy_truck': (60, 200, 250),    # 黃
    'light_truck': (60, 200, 250),
    'bus': (220, 130, 220),                                    # 紫
    'motorcycle': (180, 110, 240), 'bike': (180, 110, 240),    # 粉
    'person': (60, 220, 130),                                  # 綠
}


def _fetch_frame(camera) -> np.ndarray:
    """frame fallback chain (跟 lpr_visual 對齊):
    1. frigate latest.jpg > 80KB
    2. go2rtc /api/frame.jpeg
    3. cap.read RTSP
    都失敗 → None (caller 處理)。
    """
    import requests as _req
    camera_id = int(camera.id)
    frame = None
    try:
        r = _req.get(f"http://127.0.0.1:5000/api/cam_{camera_id}/latest.jpg", timeout=3)
        if r.status_code == 200 and len(r.content) > 80000:
            frame = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        frame = None
    if frame is None:
        try:
            r = _req.get(f"http://127.0.0.1:1984/api/frame.jpeg?src=cam_{camera_id}", timeout=5)
            if r.status_code == 200 and len(r.content) > 10000:
                frame = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = None
    if frame is None:
        try:
            cap = cv2.VideoCapture(resolve_analysis_source(camera))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                frame = None
        except Exception:
            frame = None
    return frame


def _placeholder(text: str, w: int = 960, h: int = 540) -> bytes:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, "VISION EYE", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 130), 2)
    cv2.putText(frame, text, (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else b""


def _track_speed_map(camera_id: int) -> dict:
    """從 _VEHICLE_TRACK_SNAPSHOTS 取 {track_id: speed_kmh}"""
    try:
        from api.routes.stream import _VEHICLE_TRACK_SNAPSHOTS
        snap = _VEHICLE_TRACK_SNAPSHOTS.get(camera_id) or []
        return {int(t.get("track_id", 0)): math.hypot(float(t.get("vx", 0)),
                                                       float(t.get("vy", 0))) * 3.6
                for t in snap}
    except Exception:
        return {}


@router.get("/snapshot/{camera_id}")
def vision_eye_snapshot(camera_id: int):
    """單張 VisionEye 風格 overlay — 雷達式線條 + class + speed"""
    from api.models import SessionLocal, Camera

    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="攝影機不存在")
        if not bool(camera.enabled):
            return Response(content=_placeholder(f"攝影機 {camera.name} 已關閉"),
                            media_type="image/jpeg")

        frame = _fetch_frame(camera)
        if frame is None:
            return Response(content=_placeholder(f"攝影機 {camera.name} 訊號中斷"),
                            media_type="image/jpeg")

        # yolo detect
        try:
            from api.routes.lpr_visual import get_yolo
            yolo = get_yolo()
            detections = yolo.detect(frame)
        except Exception as e:
            cv2.putText(frame, f"detector error: {e}", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return Response(content=buf.tobytes(), media_type="image/jpeg")

        fh, fw = frame.shape[:2]
        # 鏡頭視角中心 = frame 底部中心 (車道前方延伸點)
        cam_cx = fw // 2
        cam_cy = fh - 24

        speed_map = _track_speed_map(int(camera_id))

        # 暗化背景 30% (突顯 overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # 蒐集中文 text items 最後 PIL batch render (一次 BGR↔RGB conversion)
        text_items: list = []
        n_drawn = 0
        for det in detections or []:
            cls_name = str(det.get('class_name') or '').lower()
            if cls_name not in _CLS_ZH:
                continue
            bbox = det.get('bbox', {})
            x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
            x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
            if x2 <= x1 or y2 <= y1:
                continue
            bcx, bcy = (x1 + x2) // 2, (y1 + y2) // 2
            color = _CLS_COLOR.get(cls_name, (200, 200, 200))

            # 從鏡頭中心拉線到物件 (vision eye 視覺語彙) — 細白 line + glow
            cv2.line(frame, (cam_cx, cam_cy), (bcx, bcy), (255, 255, 255), 3)
            cv2.line(frame, (cam_cx, cam_cy), (bcx, bcy), color, 1)

            # bbox 細框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 端點圓 (對應物件)
            cv2.circle(frame, (bcx, bcy), 6, color, -1)
            cv2.circle(frame, (bcx, bcy), 6, (255, 255, 255), 1)

            # 標籤: 中文類別 + 速度 + 距離 (px) — 用 PIL 寫中文
            dist_px = int(math.hypot(bcx - cam_cx, bcy - cam_cy))
            label = _CLS_ZH.get(cls_name, cls_name)
            tid = det.get('track_id')
            speed_kmh = speed_map.get(int(tid) if tid is not None else -1, 0.0)
            label_speed = f"  {speed_kmh:.0f} km/h" if speed_kmh > 0.5 else ""
            text1 = f"{label}{label_speed}"
            text2 = f"距離 {dist_px} px"

            # 標籤背景 (cv2 — 純色矩形 ok)
            label_w = max(120, len(text1) * 14)
            lx = x1
            ly = max(y1 - 30, 4)
            cv2.rectangle(frame, (lx - 2, ly - 2), (lx + label_w, ly + 38), color, -1)
            # 黑字 (中文用 PIL)
            text_items.append((text1, (lx + 4, ly + 2), 16, (0, 0, 0)))
            text_items.append((text2, (lx + 4, ly + 20), 12, (40, 40, 40)))
            n_drawn += 1

        # 中心點 (鏡頭視角) — pulse-style 三層圓
        cv2.circle(frame, (cam_cx, cam_cy), 14, (60, 220, 130), 2)
        cv2.circle(frame, (cam_cx, cam_cy), 8, (60, 220, 130), -1)
        # VISION EYE 標籤 (英文用 cv2 即可)
        cv2.putText(frame, "VISION EYE", (cam_cx - 56, cam_cy - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 130), 1)

        # HUD 左上 — 中文 cam name 用 PIL,其他英文用 cv2
        cv2.rectangle(frame, (10, 10), (560, 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (560, 50), (60, 220, 130), 1)
        text_items.append((f"攝影機  {camera.name}", (20, 14), 16, (60, 220, 130)))
        text_items.append((f"追蹤 {n_drawn} 個物件 · AI 語義化視覺", (20, 32), 14, (220, 220, 220)))

        # 一次性 PIL render 中文
        frame = _draw_zh_batch(frame, text_items)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise HTTPException(status_code=500, detail="encode 失敗")
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    finally:
        db.close()
