"""本地車位逐格分類器 (P4) — 取代不泛化的巴西 PKLot detector。

crop 每個車位 ROI → 二分類 空(empty)/有車(occupied)。模型由 tools/train_parking_classifier.py
用本地標註裁切 (data/parking_train) fine-tune 產出 models/parking/local_cls.pt。
有此檔才啟用 (is_available),否則 hook 退回原 PKLot detect。
"""
from __future__ import annotations

import os
import threading
from typing import Optional, Tuple, List

import numpy as np

WEIGHTS_PATH = os.path.join("models", "parking", "local_cls.pt")

_MODEL = None
_LOCK = threading.Lock()
_LOAD_FAILED = False


def is_available() -> bool:
    return os.path.exists(WEIGHTS_PATH) and not _LOAD_FAILED


def get_model():
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED or not os.path.exists(WEIGHTS_PATH):
        return None
    with _LOCK:
        if _MODEL is None and not _LOAD_FAILED:
            try:
                from ultralytics import YOLO
                _MODEL = YOLO(WEIGHTS_PATH)
                print(f"[slot_cls] local classifier loaded: {_MODEL.names}", flush=True)
            except Exception as e:
                _LOAD_FAILED = True
                print(f"[slot_cls] load fail: {e}", flush=True)
    return _MODEL


def crop_polygon(frame: np.ndarray, polygon: List, pad: float = 0.08) -> Optional[np.ndarray]:
    """從 frame crop 車位多邊形的外接矩形 (含 pad 邊)。"""
    if frame is None or not polygon or len(polygon) < 3:
        return None
    H, W = frame.shape[:2]
    xs = [p[0] for p in polygon]; ys = [p[1] for p in polygon]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    dx = int((x2 - x1) * pad); dy = int((y2 - y1) * pad)
    x1 = max(0, int(x1) - dx); y1 = max(0, int(y1) - dy)
    x2 = min(W, int(x2) + dx); y2 = min(H, int(y2) + dy)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def classify_crop(crop: np.ndarray) -> Tuple[Optional[bool], float]:
    """回 (occupied: bool|None, conf)。None=無法判斷。"""
    m = get_model()
    if m is None or crop is None or crop.size == 0:
        return (None, 0.0)
    try:
        r = m.predict(crop, verbose=False)
        if not r:
            return (None, 0.0)
        probs = r[0].probs
        top = int(probs.top1)
        conf = float(probs.top1conf)
        name = str(m.names.get(top, "")).lower()
        return (name.startswith("o"), conf)   # 'occupied' → True
    except Exception as e:
        print(f"[slot_cls] predict err: {e}", flush=True)
        return (None, 0.0)
