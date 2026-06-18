"""MobileSAM 點選分割 — 車位編輯器「點一下自動框車格」(Phase B)。

懶載入,只在編輯車位時呼叫;模型 models/mobile_sam.pt (NVMe symlink)。
不影響即時偵測:模型只在第一次 /slots/sam_point 被呼叫時才載入。
回傳該點物件的最小旋轉外接矩形 4 點 (跟車位 polygon 格式一致)。
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

import numpy as np

WEIGHTS_PATH = os.path.join("models", "mobile_sam.pt")

_MODEL = None
_LOCK = threading.Lock()
_LOAD_FAILED = False


def is_available() -> bool:
    return os.path.exists(WEIGHTS_PATH) and not _LOAD_FAILED


def get_model():
    """懶載入 MobileSAM (ultralytics SAM)。"""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED or not os.path.exists(WEIGHTS_PATH):
        return None
    with _LOCK:
        if _MODEL is None and not _LOAD_FAILED:
            try:
                from ultralytics import SAM
                _MODEL = SAM(WEIGHTS_PATH)
                print(f"[parking_sam] MobileSAM loaded: {WEIGHTS_PATH}", flush=True)
            except Exception as e:
                _LOAD_FAILED = True
                print(f"[parking_sam] load fail: {e}", flush=True)
    return _MODEL


def segment_point(frame: np.ndarray, x: float, y: float) -> Optional[List[List[int]]]:
    """以點提示 (x,y) 分割,回最小旋轉外接矩形 4 點 [[x,y],...] (frame px)。

    回 None = 該點分不出物件。
    """
    m = get_model()
    if m is None or frame is None:
        return None
    import cv2

    H, W = frame.shape[:2]
    xi = max(0, min(W - 1, int(round(x))))
    yi = max(0, min(H - 1, int(round(y))))
    try:
        res = m.predict(frame, points=[[xi, yi]], labels=[1], verbose=False)
    except Exception as e:
        print(f"[parking_sam] predict err: {e}", flush=True)
        return None
    if not res or res[0].masks is None:
        return None
    masks = res[0].masks.data.cpu().numpy()  # (N, H, W) float
    if masks.size == 0:
        return None

    # 選「包含點擊位置」中面積最小的 mask (避免選到整片地面);沒包含就取面積最大
    best = None
    best_area = -1.0
    contain_best = None
    contain_area = 1e18
    for mk in masks:
        mb = (mk > 0.5).astype("uint8")
        area = float(mb.sum())
        if area <= 0:
            continue
        if mb[yi, xi] > 0:
            if area < contain_area:
                contain_area = area
                contain_best = mb
        if area > best_area:
            best_area = area
            best = mb
    chosen = contain_best if contain_best is not None else best
    if chosen is None:
        return None

    # 面積守門: 車格/車輛應該很小,選到 >12% 畫面 = SAM 抓到整片地面,拒絕
    if float(chosen.sum()) > 0.12 * H * W:
        return None

    cnts, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return [[int(round(p[0])), int(round(p[1]))] for p in box]
