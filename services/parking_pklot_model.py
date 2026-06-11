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

# Temporal accumulation — 跨多 frame 累積 slot 位置 (提升精確度過濾偶發誤判)
# per-source: [[x1,y1,x2,y2, n_seen, n_occupied, last_seen_ts], ...]
_PKLOT_POSITIONS: Dict[str, List] = {}
_PKLOT_POS_LOCK = threading.Lock()
_PKLOT_MIN_SEEN = 3        # 跨 3 frame 都見才算有效車位 (過濾偶發誤判)
_PKLOT_TTL_SEC = 1800.0    # 30 分鐘沒見過自動刪
_PKLOT_MERGE_IOU = 0.55    # IoU > 0.55 視為同位置累積
_PKLOT_OCC_RATIO = 0.4     # n_occupied / n_seen > 0.4 → 視為 occupied


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


def _point_in_poly(px: float, py: float, poly: List) -> bool:
    """ray-casting polygon point-in-poly,poly = [[x,y], ...]"""
    if not poly or len(poly) < 3:
        return True   # 沒設 mask 視為都通過
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


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
    import time as _t
    now = _t.time()
    h_img, w_img = frame.shape[:2]
    pklot_dets = detect_slots(frame, conf=0.05)
    car_centers = _yolo_car_centers(frame, conf=0.12)
    car_bboxes = [(c[2], c[3], c[4], c[5]) for c in car_centers]
    area_mask = meta.get("parking_area_mask") or []

    # 當下這 frame 通過嚴格雙確認的 candidate slots
    candidates = []  # [(bbox, final_occupied)]
    for idx, det in enumerate(pklot_dets):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        slot_bbox = (x1, y1, x2, y2)
        # mask 過濾
        if area_mask:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not _point_in_poly(cx, cy, area_mask):
                continue
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
            if pklot_occ:
                continue

        candidates.append((slot_bbox, final_occ))

    # Temporal accumulation — 累積進 _PKLOT_POSITIONS,只有跨多 frame 持續見的才算有效車位
    with _PKLOT_POS_LOCK:
        positions = _PKLOT_POSITIONS.setdefault(source_key, [])
        for bbox, occ in candidates:
            matched = False
            for p in positions:
                if _bbox_iou(bbox, (p[0], p[1], p[2], p[3])) > _PKLOT_MERGE_IOU:
                    # 加權平均更新座標 (穩定 bbox 位置)
                    n = p[4] + 1
                    p[0] = int((p[0] * p[4] + bbox[0]) / n)
                    p[1] = int((p[1] * p[4] + bbox[1]) / n)
                    p[2] = int((p[2] * p[4] + bbox[2]) / n)
                    p[3] = int((p[3] * p[4] + bbox[3]) / n)
                    p[4] = n
                    if occ:
                        p[5] += 1
                    p[6] = now
                    matched = True
                    break
            if not matched:
                positions.append([bbox[0], bbox[1], bbox[2], bbox[3],
                                   1, 1 if occ else 0, now])
        # TTL GC
        positions[:] = [p for p in positions if (now - p[6]) <= _PKLOT_TTL_SEC]
        # 取「穩定」的位置 (跨 _PKLOT_MIN_SEEN frame 都見過)
        stable = [p for p in positions if p[4] >= _PKLOT_MIN_SEEN]

    slot_results = []
    for idx, p in enumerate(stable):
        x1, y1, x2, y2, n_seen, n_occ, last = p
        occupied_ratio = n_occ / max(1, n_seen)
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        lbl = f"P{idx+1}"
        slot_results.append({
            "id": lbl, "label": lbl,
            "occupied": occupied_ratio > _PKLOT_OCC_RATIO,
            "conf": round(min(1.0, n_seen / 5.0), 3),  # 穩定度當 conf
            "polygon": polygon,
            "n_seen": n_seen,
            "n_occupied": n_occ,
            "occupied_ratio": round(occupied_ratio, 2),
        })

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
        "mode": "pklot+yolo+temporal",
        "candidates_this_frame": len(candidates),
        "stable_positions": total,
        "min_seen_threshold": _PKLOT_MIN_SEEN,
    }


def reset_pklot_positions(source_key: str) -> int:
    """清掉 _PKLOT_POSITIONS[source] 累積"""
    with _PKLOT_POS_LOCK:
        old = _PKLOT_POSITIONS.pop(source_key, [])
    return len(old)
