"""
PKLot pretrained YOLOv8 model — 直接識別「空車位 / 有車車位」二分類.

Source: pokhrelapar/yolov8-pklot (raw GitHub URL, 公開無需 API key)
- 6.2 MB YOLOv8n, fine-tuned on PKLot dataset (2766 真實停車場標註圖)
- Classes: 0='e' (empty), 1='o' (occupied)

優點 (對比 car detection + ROI):
- 直接識「空車位」(不靠 car 存在判定)
- 不用手動標 ROI / 累積位置
- 每張 frame 獨立計算

用 single cached model instance 避免重複 load.
"""
from __future__ import annotations

import os
import threading
from typing import List, Dict, Optional

import numpy as np

PKLOT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "parking", "yolov8n_pklot.pt"
)

_PKLOT_MODEL = None
_PKLOT_LOCK = threading.Lock()
_PKLOT_AVAILABLE: Optional[bool] = None


def is_available() -> bool:
    global _PKLOT_AVAILABLE
    if _PKLOT_AVAILABLE is None:
        _PKLOT_AVAILABLE = os.path.exists(PKLOT_WEIGHTS_PATH)
        if not _PKLOT_AVAILABLE:
            print(f"[pklot] weights not found: {PKLOT_WEIGHTS_PATH}", flush=True)
    return _PKLOT_AVAILABLE


def get_model():
    global _PKLOT_MODEL
    if not is_available():
        return None
    if _PKLOT_MODEL is not None:
        return _PKLOT_MODEL
    with _PKLOT_LOCK:
        if _PKLOT_MODEL is not None:
            return _PKLOT_MODEL
        try:
            from ultralytics import YOLO
            _PKLOT_MODEL = YOLO(PKLOT_WEIGHTS_PATH)
            print(f"[pklot] model loaded: classes={_PKLOT_MODEL.names}", flush=True)
        except Exception as e:
            print(f"[pklot] load err: {e}", flush=True)
            return None
    return _PKLOT_MODEL


def _predict_on(model, image: np.ndarray, conf: float) -> List[Dict]:
    try:
        results = model.predict(image, conf=conf, verbose=False)
    except Exception as e:
        print(f"[pklot] predict err: {e}", flush=True)
        return []
    out = []
    for r in results or []:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for i in range(len(boxes)):
            try:
                cls_id = int(boxes.cls[i].item())
                cls_name = str(model.names.get(cls_id, "")).lower()
                occupied = cls_name.startswith("o")
                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "occupied": occupied,
                    "conf": float(boxes.conf[i].item()),
                    "cls": cls_name,
                })
            except Exception:
                continue
    return out


def detect_slots(frame: np.ndarray, conf: float = 0.05,
                  rows: int = 3, cols: int = 3, overlap: float = 0.25) -> List[Dict]:
    """Sliced PKLot inference — 切 R×C tile + 全圖,合併後 NMS 去重.
    PKLot model 對「close-up 俯角」識別率高,sliced 後在 tile 區域內近距更佳."""
    model = get_model()
    if model is None:
        return []
    h, w = frame.shape[:2]
    tile_w = max(64, w // cols)
    tile_h = max(64, h // rows)
    ov_w = int(tile_w * overlap)
    ov_h = int(tile_h * overlap)
    all_dets = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * tile_w - ov_w)
            y0 = max(0, r * tile_h - ov_h)
            x1 = min(w, (c + 1) * tile_w + ov_w)
            y1 = min(h, (r + 1) * tile_h + ov_h)
            tile = frame[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            for d in _predict_on(model, tile, conf):
                all_dets.append({
                    **d,
                    "x1": d["x1"] + x0, "y1": d["y1"] + y0,
                    "x2": d["x2"] + x0, "y2": d["y2"] + y0,
                })
    # 全圖補大型 slot (PKLot model 偶爾看大區域好)
    for d in _predict_on(model, frame, conf):
        all_dets.append(d)
    # NMS — IoU > 0.4 抑制 (同 slot 重複)
    def _iou(a, b):
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
        if any(_iou(d, k) > 0.4 for k in kept):
            continue
        kept.append(d)
    return kept


def evaluate_pklot(source_key: str, frame: np.ndarray, meta: Dict) -> Dict:
    """用 PKLot model 直接評估 — 每個 detected box 就是一個 slot,
    class 區分空/滿."""
    if not is_available():
        return {"source": source_key, "source_name": meta.get("name", source_key),
                "error": "PKLot model 未安裝", "total": 0, "occupied": 0,
                "available": 0, "occupancy_rate": 0.0, "slots": [], "mode": "pklot"}
    h_img, w_img = frame.shape[:2]
    detections = detect_slots(frame, conf=0.05)  # 低 conf + sliced 涵蓋率高
    slot_results = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        lbl = f"P{idx + 1}"
        slot_results.append({
            "id": lbl, "label": lbl,
            "occupied": det["occupied"],
            "conf": round(det["conf"], 3),
            "polygon": polygon,
        })
    occupied = sum(1 for s in slot_results if s["occupied"])
    total = len(slot_results)
    return {
        "source": source_key, "source_name": meta.get("name", source_key),
        "frame_w": w_img, "frame_h": h_img,
        "total": total, "occupied": occupied,
        "available": total - occupied,
        "occupancy_rate": round((occupied / total * 100.0) if total else 0.0, 1),
        "detected_vehicles": occupied,
        "slots": slot_results,
        "mode": "pklot",
    }
