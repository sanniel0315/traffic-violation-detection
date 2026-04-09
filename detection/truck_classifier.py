#!/usr/bin/env python3
"""
大貨車/小貨車/大客車 分類器
===========================
對 YOLO 偵測出的 truck / bus 做二階段細分類。

類別對應:
  0: bus         (大客車)
  1: heavy_truck (大貨車)
  2: light_truck (小貨車)
  3: non_truck   (非目標)
"""

import os
import warnings
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

from ultralytics import YOLO
from model_paths import get_model_dir


# 分類結果 → 顯示用中文標籤 & 車輛等效長度
CLASS_META = {
    "heavy_truck": {"label": "大貨車", "length_m": 12.0, "group": "large"},
    "light_truck": {"label": "小貨車", "length_m": 6.0, "group": "small"},
    "bus":         {"label": "大客車", "length_m": 12.0, "group": "large"},
    "non_truck":   {"label": "非目標", "length_m": 6.0, "group": "other"},
}


def get_truck_cls_model_path() -> str:
    model_dir = get_model_dir()
    value = os.getenv("TRUCK_CLS_MODEL", "truck_cls_yolo26s.pt")
    if os.path.isabs(value):
        return value
    return os.path.join(model_dir, value)


class TruckClassifier:
    """
    大型車輛細分類器

    用法:
        classifier = TruckClassifier()
        result = classifier.classify(frame, bbox)
        # result = {"class_name": "heavy_truck", "label": "大貨車",
        #           "confidence": 0.92, "group": "large", "length_m": 12.0}
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        imgsz: int = 224,
    ):
        model_path = model_path or get_truck_cls_model_path()
        if not os.path.exists(model_path):
            print(f"⚠️  分類模型不存在: {model_path}，TruckClassifier 停用")
            self.model = None
            return

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        self.device = os.getenv("DEVICE", "cuda:0")
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"

        # 建立 class index → name 的映射
        self.class_names = self.model.names  # {0: 'bus', 1: 'heavy_truck', ...}
        print(f"✅ 大型車分類器初始化完成 (模型: {model_path}, 類別: {self.class_names})")

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def classify(
        self,
        frame: np.ndarray,
        bbox: dict,
        pad_ratio: float = 0.1,
    ) -> dict:
        """
        對 bounding box 區域做分類

        Args:
            frame: 完整影像 (BGR)
            bbox: {"x1": int, "y1": int, "x2": int, "y2": int}
            pad_ratio: bbox 外擴比例，避免裁太緊

        Returns:
            {"class_name": str, "label": str, "confidence": float,
             "group": str, "length_m": float}
        """
        if not self.enabled:
            return self._default_result()

        # 裁切 + padding
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._default_result()

        # 推論
        results = self.model.predict(
            source=crop,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )

        if not results or results[0].probs is None:
            return self._default_result()

        probs = results[0].probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = self.class_names[top1_idx]

        # 信心度不足 → 回傳預設
        if top1_conf < self.conf_threshold:
            return self._default_result()

        meta = CLASS_META.get(class_name, CLASS_META["non_truck"])
        return {
            "class_name": class_name,
            "label": meta["label"],
            "confidence": round(top1_conf, 3),
            "group": meta["group"],
            "length_m": meta["length_m"],
        }

    def classify_batch(
        self,
        frame: np.ndarray,
        bboxes: list[dict],
        pad_ratio: float = 0.1,
    ) -> list[dict]:
        """批次分類多個 bbox"""
        return [self.classify(frame, bb, pad_ratio) for bb in bboxes]

    @staticmethod
    def _default_result() -> dict:
        return {
            "class_name": "unknown",
            "label": "未知",
            "confidence": 0.0,
            "group": "other",
            "length_m": 6.0,
        }
