#!/usr/bin/env python3
"""Dedicated license-plate detector wrapper with a heuristic fallback."""
import os
from typing import List, Dict, Any

import cv2
import numpy as np

from model_paths import get_plate_model_engine, get_plate_model_pt


class PlateDetector:
    """Plate detector backed by a dedicated YOLO model or heuristic fallback."""

    def __init__(self):
        model_path = get_plate_model_engine()
        if not os.path.exists(model_path):
            model_path = get_plate_model_pt()
        self.model_path = model_path if os.path.exists(model_path) else None
        self.model = None
        self.backend = "heuristic"
        self.allowed_class_ids = None
        if self.model_path:
            try:
                from ultralytics import YOLO

                self.model = YOLO(self.model_path, task="detect")
                self.backend = "yolo"
                names = getattr(self.model.model, "names", None) or getattr(self.model, "names", None) or {}
                allowed = []
                for cls_id, name in dict(names).items():
                    label = str(name or "").strip().lower()
                    if "plate" in label:
                        allowed.append(int(cls_id))
                self.allowed_class_ids = set(allowed) if allowed else None
            except Exception:
                self.model = None
                self.backend = "heuristic"

    def detect(self, image, conf: float = 0.25) -> List[Dict[str, Any]]:
        if image is None or getattr(image, "size", 0) == 0:
            return []

        if self.model is None:
            return self._detect_heuristic(image)

        h, w = image.shape[:2]
        results = self.model(image, verbose=False, conf=conf)
        out: List[Dict[str, Any]] = []
        for res in results:
            for box in getattr(res, "boxes", []):
                cls_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else None
                if self.allowed_class_ids is not None and cls_id not in self.allowed_class_ids:
                    continue
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0,
                        "backend": self.backend,
                        "class_id": cls_id,
                    }
                )
        if out:
            return self._nms(out, h, w)
        return self._detect_heuristic(image)

    def _detect_heuristic(self, image) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        if h < 48 or w < 120:
            return []

        scale = 1.0
        if w < 640:
            scale = 640.0 / float(w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        work_h, work_w = gray.shape[:2]
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        masks = []
        blackhat = cv2.morphologyEx(
            enhanced,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)),
        )
        _, bh_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(cv2.morphologyEx(bh_mask, cv2.MORPH_CLOSE, np.ones((3, 11), np.uint8)))

        grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
        _, gx_mask = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(cv2.morphologyEx(gx_mask, cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8)))

        edges = cv2.Canny(enhanced, 60, 180)
        masks.append(cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1))

        candidates: List[Dict[str, Any]] = []
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < work_w * 0.12 or ch < work_h * 0.06:
                    continue
                if cw > work_w * 0.88 or ch > work_h * 0.45:
                    continue
                aspect = cw / max(1.0, float(ch))
                if not (2.0 <= aspect <= 6.8):
                    continue
                y_center = (y + (ch / 2.0)) / max(1.0, float(work_h))
                if not (0.25 <= y_center <= 0.92):
                    continue

                pad_x = int(cw * 0.04)
                pad_y = int(ch * 0.10)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(work_w, x + cw + pad_x)
                y2 = min(work_h, y + ch + pad_y)
                patch = enhanced[y1:y2, x1:x2]
                score = self._plate_score(patch, x1, y1, x2, y2, work_w, work_h)
                if score < 0.28:
                    continue
                candidates.append(
                    {
                        "bbox": [
                            int(x1 / scale),
                            int(y1 / scale),
                            int(x2 / scale),
                            int(y2 / scale),
                        ],
                        "confidence": float(min(0.99, score)),
                        "backend": self.backend,
                    }
                )

        if not candidates:
            return []
        return self._nms(candidates, h, w)

    @staticmethod
    def _character_score(gray_patch) -> float:
        if gray_patch is None or getattr(gray_patch, "size", 0) == 0:
            return 0.0
        try:
            h, w = gray_patch.shape[:2]
            _, binary = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            scores = []
            for mask in (binary, 255 - binary):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                count = 0
                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw < 3 or ch < max(8, int(h * 0.35)):
                        continue
                    if ch > int(h * 0.95):
                        continue
                    aspect = cw / max(1.0, float(ch))
                    if not (0.10 <= aspect <= 1.0):
                        continue
                    if y + ch < int(h * 0.25):
                        continue
                    count += 1
                scores.append(max(0.0, 1.0 - abs(count - 6) / 6.0))
            return max(scores) if scores else 0.0
        except Exception:
            return 0.0

    def _plate_score(self, gray_patch, x1: int, y1: int, x2: int, y2: int, full_w: int, full_h: int) -> float:
        if gray_patch is None or getattr(gray_patch, "size", 0) == 0:
            return 0.0
        h, w = gray_patch.shape[:2]
        area = float(max(1, h * w))
        aspect = w / max(1.0, float(h))
        edges = cv2.Canny(gray_patch, 60, 180)
        edge_density = float(cv2.countNonZero(edges)) / area
        sobel_x = cv2.Sobel(gray_patch, cv2.CV_32F, 1, 0, ksize=3)
        stroke_energy = float(np.mean(np.abs(sobel_x))) / 255.0
        _, binary = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio = float(cv2.countNonZero(255 - binary)) / area
        char_score = self._character_score(gray_patch)
        cx = ((x1 + x2) / 2.0) / max(1.0, float(full_w))
        cy = ((y1 + y2) / 2.0) / max(1.0, float(full_h))
        center_score = max(0.0, 1.0 - abs(cx - 0.5) / 0.5)
        lower_bias = max(0.0, 1.0 - abs(cy - 0.68) / 0.45)
        aspect_score = max(0.0, 1.0 - abs(aspect - 3.4) / 3.4)
        edge_score = max(0.0, 1.0 - abs(edge_density - 0.14) / 0.14)
        fg_score = max(0.0, 1.0 - abs(fg_ratio - 0.34) / 0.34)
        stroke_score = min(1.0, stroke_energy / 0.22)
        return (
            (char_score * 0.34)
            + (aspect_score * 0.18)
            + (edge_score * 0.14)
            + (fg_score * 0.10)
            + (stroke_score * 0.10)
            + (center_score * 0.06)
            + (lower_bias * 0.08)
        )

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        return inter / max(1.0, area_a + area_b - inter)

    def _nms(self, candidates: List[Dict[str, Any]], full_h: int, full_w: int) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for item in sorted(candidates, key=lambda d: d["confidence"], reverse=True):
            x1, y1, x2, y2 = item["bbox"]
            x1 = max(0, min(full_w - 1, int(x1)))
            y1 = max(0, min(full_h - 1, int(y1)))
            x2 = max(0, min(full_w, int(x2)))
            y2 = max(0, min(full_h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = [x1, y1, x2, y2]
            if any(self._iou(bbox, kept_item["bbox"]) > 0.45 for kept_item in kept):
                continue
            item["bbox"] = bbox
            kept.append(item)
            if len(kept) >= 3:
                break
        return kept

    @staticmethod
    def crop(image, bbox):
        if image is None or getattr(image, "size", 0) == 0:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    @staticmethod
    def draw_debug(image, bbox):
        if image is None or getattr(image, "size", 0) == 0:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
