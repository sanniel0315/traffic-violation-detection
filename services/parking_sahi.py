"""
SAHI (Slicing Aided Hyper Inference) wrapper — 取代手寫 sliced inference.

SAHI 優點:
- NMM (Non-Maximum Merging) 處理 tile 邊界切斷的 bbox 更穩
- 業界 mature 方案
- 對小物 detection 比裸 YOLO 更好

用 cache global model instance 避免每次 evaluate 重 init.
"""
from __future__ import annotations

import os
import threading
from typing import List, Dict, Optional

import numpy as np

_SAHI_AVAILABLE: Optional[bool] = None
_SAHI_MODEL = None
_SAHI_LOCK = threading.Lock()


def is_available() -> bool:
    global _SAHI_AVAILABLE
    if _SAHI_AVAILABLE is None:
        try:
            from sahi import AutoDetectionModel  # noqa: F401
            from sahi.predict import get_sliced_prediction  # noqa: F401
            _SAHI_AVAILABLE = True
        except Exception as e:
            print(f"[sahi] unavailable: {e}", flush=True)
            _SAHI_AVAILABLE = False
    return _SAHI_AVAILABLE


def get_model(confidence: float = 0.15):
    """單例 SAHI model — 共用 ultralytics YOLOv8 weights."""
    global _SAHI_MODEL
    if not is_available():
        return None
    if _SAHI_MODEL is not None:
        return _SAHI_MODEL
    with _SAHI_LOCK:
        if _SAHI_MODEL is not None:
            return _SAHI_MODEL
        try:
            from sahi import AutoDetectionModel
            # 找 yolo weights — 沿用既有 vehicle_detector path
            from detection.vehicle_detector import VehicleDetector
            # VehicleDetector init 時 load weights;取 model 路徑
            vd = VehicleDetector(conf_threshold=confidence)
            weights_path = getattr(vd, "model_path", None) or "yolov8n.pt"
            if not os.path.exists(weights_path):
                # fallback 路徑
                candidates = [
                    "models/yolov8n.pt",
                    "/home/ubuntu/traffic-violation-detection/models/yolov8n.pt",
                ]
                for c in candidates:
                    if os.path.exists(c):
                        weights_path = c
                        break
            device = "cuda:0"
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
            _SAHI_MODEL = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=weights_path,
                confidence_threshold=confidence,
                device=device,
            )
            print(f"[sahi] model loaded: {weights_path} device={device}", flush=True)
        except Exception as e:
            print(f"[sahi] get_model err: {e}", flush=True)
            return None
    return _SAHI_MODEL


def sahi_detect(frame: np.ndarray, confidence: float = 0.15,
                slice_size: int = 256, overlap: float = 0.2) -> List[Dict]:
    """跑 SAHI sliced inference,回 detection list.
    每筆 {class_name, conf, bbox: {x1,y1,x2,y2}}"""
    model = get_model(confidence)
    if model is None:
        return []
    try:
        from sahi.predict import get_sliced_prediction
        result = get_sliced_prediction(
            frame,
            model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            perform_standard_pred=True,        # 全圖再跑一次補大型物
            postprocess_type="NMM",            # Non-Maximum Merging 比 NMS 好
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        out = []
        for pred in result.object_prediction_list or []:
            try:
                bb = pred.bbox
                out.append({
                    "class_name": str(pred.category.name) if pred.category else "",
                    "conf": float(pred.score.value) if pred.score else 0.0,
                    "bbox": {
                        "x1": int(bb.minx), "y1": int(bb.miny),
                        "x2": int(bb.maxx), "y2": int(bb.maxy),
                    },
                })
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"[sahi] detect err: {e}", flush=True)
        return []
