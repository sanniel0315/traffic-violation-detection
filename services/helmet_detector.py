"""
機車安全帽偵測 — YOLOv8 helmet detection.

預設用 keremberke/yolov8n-helmet-detection (HuggingFace 公開 2-class:
With Helmet / Without Helmet, ~5MB).

模型不存在 → return [] 不掛 service (允許 user 後續手動 download).

權重路徑: models/helmet/yolov8n-helmet.pt
download 命令 (Jetson 上跑一次):
    cd ~/traffic-violation-detection
    mkdir -p models/helmet
    huggingface-cli download keremberke/yolov8n-helmet-detection \\
      best.pt --local-dir models/helmet/
    mv models/helmet/best.pt models/helmet/yolov8n-helmet.pt
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_MODEL = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED = False

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "helmet" / "yolov8n-helmet.pt"

# class id → label (keremberke 模型)
CLASS_LABELS = {0: "with_helmet", 1: "without_helmet"}


def is_available() -> bool:
    """權重檔存在 + 沒 load 失敗即可用"""
    return WEIGHTS_PATH.exists() and not _LOAD_FAILED


def _ensure_loaded() -> bool:
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return True
    if _LOAD_FAILED or not WEIGHTS_PATH.exists():
        return False
    with _LOAD_LOCK:
        if _MODEL is not None:
            return True
        try:
            from ultralytics import YOLO
            _MODEL = YOLO(str(WEIGHTS_PATH))
            print(f"[helmet] model loaded from {WEIGHTS_PATH}", flush=True)
            return True
        except Exception as e:
            _LOAD_FAILED = True
            print(f"[helmet] load fail: {e}", flush=True)
            return False


def detect(frame: np.ndarray, conf: float = 0.30) -> List[Dict]:
    """跑整張 frame 的 helmet 偵測,return [{bbox: (x1,y1,x2,y2), label, conf}]"""
    if frame is None or frame.size == 0 or not _ensure_loaded():
        return []
    try:
        results = _MODEL.predict(frame, conf=conf, verbose=False)
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(b.cls[0])
                conf_v = float(b.conf[0])
                out.append({
                    "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    "label": CLASS_LABELS.get(cls_id, str(cls_id)),
                    "conf": conf_v,
                })
        return out
    except Exception as e:
        print(f"[helmet] detect err: {e}", flush=True)
        return []


def _rider_region(motorcycle_bbox: Dict, frame_h: int, frame_w: int) -> Optional[tuple]:
    """機車 bbox 上方延伸 80% 高度作為 rider 區域 (頭+上半身).
    Return (x1,y1,x2,y2) clipped 到 frame.
    """
    try:
        x1 = int(motorcycle_bbox.get("x1", 0))
        y1 = int(motorcycle_bbox.get("y1", 0))
        x2 = int(motorcycle_bbox.get("x2", 0))
        y2 = int(motorcycle_bbox.get("y2", 0))
        mh = y2 - y1
        if mh <= 0 or x2 <= x1:
            return None
        # 往上延 80% 機車高度抓 rider 頭部
        ry1 = max(0, y1 - int(mh * 0.8))
        ry2 = y1 + int(mh * 0.3)  # 跨 1/3 摩托車本體確保 rider 在內
        # 左右各加 10%
        mw = x2 - x1
        rx1 = max(0, x1 - int(mw * 0.10))
        rx2 = min(frame_w, x2 + int(mw * 0.10))
        ry2 = min(frame_h, ry2)
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        return (rx1, ry1, rx2, ry2)
    except Exception:
        return None


def check_motorcycle_no_helmet(frame: np.ndarray, motorcycle_bbox: Dict,
                                conf: float = 0.30) -> Dict:
    """對單台機車 bbox 判斷 rider 有無戴 helmet.

    Return:
      {
        "has_helmet": True/False/None,  # None = 偵測不到 head
        "rider_region": (x1,y1,x2,y2),
        "without_helmet_conf": float,   # 沒戴 helmet 的最高信心 (若有)
        "helmet_conf": float,           # 有戴 helmet 的最高信心 (若有)
      }
    """
    out = {"has_helmet": None, "rider_region": None,
           "without_helmet_conf": 0.0, "helmet_conf": 0.0}
    if frame is None or frame.size == 0 or not motorcycle_bbox:
        return out
    if not _ensure_loaded():
        return out
    H, W = frame.shape[:2]
    region = _rider_region(motorcycle_bbox, H, W)
    if region is None:
        return out
    out["rider_region"] = region
    rx1, ry1, rx2, ry2 = region
    crop = frame[ry1:ry2, rx1:rx2]
    if crop is None or crop.size == 0:
        return out
    dets = detect(crop, conf=conf)
    if not dets:
        return out
    for d in dets:
        if d["label"] == "with_helmet" and d["conf"] > out["helmet_conf"]:
            out["helmet_conf"] = d["conf"]
        elif d["label"] == "without_helmet" and d["conf"] > out["without_helmet_conf"]:
            out["without_helmet_conf"] = d["conf"]
    # 判定: 沒戴 helmet 信心比有戴高 → no helmet
    if out["without_helmet_conf"] >= 0.40 and out["without_helmet_conf"] > out["helmet_conf"]:
        out["has_helmet"] = False
    elif out["helmet_conf"] >= 0.40:
        out["has_helmet"] = True
    return out
