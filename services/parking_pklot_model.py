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


def _yolo_car_centers(frame: np.ndarray, conf: float = 0.15) -> List[tuple]:
    """跑 YOLO car detect (sliced) 回傳所有 car/truck bbox 中心點 (x,y).
    用來確認 PKLot slot 是否真的有車 (排除草地誤判)."""
    try:
        from services.parking_occupancy import _yolo_sliced_detect
        from detection.vehicle_detector import VehicleDetector
        yolo = VehicleDetector(conf_threshold=conf)
        dets = _yolo_sliced_detect(yolo, frame)
    except Exception as e:
        print(f"[pklot] yolo confirm err: {e}", flush=True)
        return []
    vc = {"car", "truck", "bus", "heavy_truck", "light_truck", "non_truck"}
    centers = []
    for d in dets or []:
        cls = str(d.get("class_name") or "").lower()
        if cls not in vc:
            continue
        bb = d.get("bbox", {})
        x1 = int(bb.get("x1", 0)); y1 = int(bb.get("y1", 0))
        x2 = int(bb.get("x2", 0)); y2 = int(bb.get("y2", 0))
        if x2 <= x1 or y2 <= y1:
            continue
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2, y2))
    return centers


def _bbox_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, ar + br - inter)


def evaluate_pklot(source_key: str, frame: np.ndarray, meta: Dict) -> Dict:
    """PKLot + YOLO 雙確認:
    1. PKLot 給「車位 polygon」(空+滿)
    2. YOLO car 給「真實有車輛的 bbox」
    3. 每個 slot 覆蓋判定: 任一 YOLO car bbox 跟 slot IoU > 0.3 → occupied
       (排除 PKLot 把草地誤判為「空車位」+ 排除 PKLot 把車身誤判為「空」)
    4. PKLot empty 且 YOLO 也沒車 → 真空車位
    5. PKLot empty 但 YOLO 該位置有車 → 確認 occupied (PKLot 誤判)
    6. 完全沒車的 slot (PKLot empty + YOLO 無) 若 conf < 0.1 → 過濾掉 (可能誤判草地)"""
    if not is_available():
        return {"source": source_key, "source_name": meta.get("name", source_key),
                "error": "PKLot model 未安裝", "total": 0, "occupied": 0,
                "available": 0, "occupancy_rate": 0.0, "slots": [], "mode": "pklot"}
    h_img, w_img = frame.shape[:2]
    pklot_dets = detect_slots(frame, conf=0.05)
    car_centers = _yolo_car_centers(frame, conf=0.12)
    car_bboxes = [(c[2], c[3], c[4], c[5]) for c in car_centers]

    slot_results = []
    for idx, det in enumerate(pklot_dets):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        slot_bbox = (x1, y1, x2, y2)
        # 雙確認: YOLO 任一 car bbox 跟 slot IoU > 0.3 → occupied
        yolo_occ = False
        for cb in car_bboxes:
            if _bbox_iou(slot_bbox, cb) > 0.3:
                yolo_occ = True
                break
        # 任一 YOLO car center 落 slot 內也算
        if not yolo_occ:
            for c in car_centers:
                cx, cy = c[0], c[1]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    yolo_occ = True
                    break

        pklot_occ = det["occupied"]
        final_occ = yolo_occ

        # 嚴格過濾 — 排除 PKLot 把道路/草地/路口誤判為車位:
        # 規則 1: YOLO 沒看到車 + PKLot conf < 0.18 → 八成不是真車位,過濾
        # 規則 2: YOLO 沒看到車 + PKLot 自己也說 occupied → 嚴重誤判,過濾
        # 規則 3: 通過 = YOLO 有車 (高信度有車位) OR PKLot conf >= 0.18 且 PKLot 說 empty (model 對空車位確信)
        if not yolo_occ:
            if det["conf"] < 0.18:
                continue
            if pklot_occ:  # PKLot 說「有車」但 YOLO 看不到 → 八成是把樹/車道誤認車
                continue

        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        lbl = f"P{idx + 1}"
        slot_results.append({
            "id": lbl, "label": lbl,
            "occupied": final_occ,
            "conf": round(det["conf"], 3),
            "polygon": polygon,
            "pklot_says_occupied": pklot_occ,
            "yolo_confirms_car": yolo_occ,
        })

    # 重新編號 (過濾後)
    for i, s in enumerate(slot_results):
        s["id"] = f"P{i+1}"
        s["label"] = f"P{i+1}"

    occupied = sum(1 for s in slot_results if s["occupied"])
    total = len(slot_results)
    return {
        "source": source_key, "source_name": meta.get("name", source_key),
        "frame_w": w_img, "frame_h": h_img,
        "total": total, "occupied": occupied,
        "available": total - occupied,
        "occupancy_rate": round((occupied / total * 100.0) if total else 0.0, 1),
        "detected_vehicles": len(car_centers),
        "slots": slot_results,
        "mode": "pklot+yolo",
    }
