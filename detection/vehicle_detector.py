#!/usr/bin/env python3
"""車輛偵測模組 - 使用 YOLOv8"""
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import numpy as np
import os
from model_paths import get_detect_model_pt


class VehicleDetector:
    """車輛偵測器"""

    # COCO 預設類別對照；若模型有自帶 names，會動態覆蓋。
    DEFAULT_VEHICLE_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    # 需要做二階段細分類的類別
    _RECLASSIFY_CLASSES = {'truck', 'bus'}
    CLASS_NAME_ALIASES = {
        'person': {'person', 'pedestrian', 'people'},
        'bicycle': {'bicycle', 'cycle'},
        'car': {'car', 'vehicle', 'sedan', 'suv', 'van', 'auto', 'automobile', 'taxi', 'jeep'},
        'motorcycle': {'motorcycle', 'motorbike', 'scooter', 'moped'},
        'bus': {'bus', 'coach', 'minibus'},
        'truck': {'truck', 'lorry', 'pickup', 'pickup truck', 'pickup_truck', 'trailer'},
    }
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5,
                 enable_truck_cls: bool = True):
        """
        初始化偵測器

        Args:
            model_path: YOLOv8 模型路徑
            conf_threshold: 信心度閾值
            enable_truck_cls: 是否啟用大型車細分類
        """
        model_path = model_path or get_detect_model_pt()
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = os.getenv("DEVICE", "cuda:0")
        self.runtime_device = "cpu"
        try:
            self.model.to(self.device)
            self.runtime_device = self.device
        except Exception:
            self.runtime_device = "cpu"
        self.vehicle_classes = self._resolve_vehicle_classes()

        # 大型車細分類器（可選）
        self.truck_classifier = None
        if enable_truck_cls:
            try:
                from detection.truck_classifier import TruckClassifier
                tc = TruckClassifier()
                if tc.enabled:
                    self.truck_classifier = tc
            except Exception as e:
                print(f"⚠️  大型車分類器載入失敗: {e}")

        print(f"✅ 車輛偵測器初始化完成 (模型: {model_path}, device: {self.runtime_device})")
        print(f"✅ 車種類別映射: {self.vehicle_classes}")
        if self.truck_classifier:
            print(f"✅ 大型車細分類: 啟用")

    @classmethod
    def _normalize_label(cls, value: str) -> str:
        text = str(value or "").strip().lower()
        for ch in ("_", "-"):
            text = text.replace(ch, " ")
        return " ".join(text.split())

    @classmethod
    def _match_canonical_label(cls, raw_name: str) -> Optional[str]:
        name = cls._normalize_label(raw_name)
        compact = name.replace(" ", "")
        for canonical, aliases in cls.CLASS_NAME_ALIASES.items():
            for alias in aliases:
                alias_norm = cls._normalize_label(alias)
                alias_compact = alias_norm.replace(" ", "")
                if name == alias_norm or compact == alias_compact:
                    return canonical
                if alias_norm and alias_norm in name:
                    return canonical
        return None

    def _resolve_vehicle_classes(self) -> Dict[int, str]:
        names = getattr(self.model, "names", None)
        items = []
        if isinstance(names, dict):
            items = list(names.items())
        elif isinstance(names, (list, tuple)):
            items = list(enumerate(names))

        resolved: Dict[int, str] = {}
        for class_id, class_name in items:
            canonical = self._match_canonical_label(str(class_name))
            if canonical:
                resolved[int(class_id)] = canonical

        if resolved:
            return resolved
        return dict(self.DEFAULT_VEHICLE_CLASSES)
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        偵測影像中的車輛
        
        Args:
            frame: BGR 影像 (numpy array)
            
        Returns:
            偵測結果列表
        """
        # 執行推論
        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.runtime_device,
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                # 只保留交通相關類別
                if class_id not in self.vehicle_classes:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                det = {
                    'class_id': class_id,
                    'class_name': self.vehicle_classes[class_id],
                    'confidence': confidence,
                    'bbox': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    }
                }

                # 大型車細分類
                if (self.truck_classifier
                        and det['class_name'] in self._RECLASSIFY_CLASSES):
                    cls_result = self.truck_classifier.classify(frame, det['bbox'])
                    if cls_result['class_name'] != 'unknown':
                        det['class_name'] = cls_result['class_name']
                        det['truck_cls'] = cls_result

                detections.append(det)
        
        return detections
    
    def detect_with_draw(self, frame: np.ndarray) -> tuple:
        """
        偵測並繪製標註框
        
        Returns:
            (標註後的影像, 偵測結果)
        """
        import cv2
        
        detections = self.detect(frame)
        annotated = frame.copy()
        
        # 顏色定義 (BGR)
        colors = {
            'person': (0, 255, 0),        # 綠色
            'car': (255, 0, 0),           # 藍色
            'motorcycle': (0, 255, 255),  # 黃色
            'bus': (255, 165, 0),         # 橙色
            'truck': (128, 0, 128),       # 紫色
            'bicycle': (0, 128, 255),     # 橘色
            'heavy_truck': (0, 0, 255),   # 紅色
            'light_truck': (255, 128, 0), # 淺藍
            'non_truck': (180, 180, 180), # 灰色
        }
        
        for det in detections:
            bbox = det['bbox']
            color = colors.get(det['class_name'], (255, 255, 255))
            
            # 繪製矩形框
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color, 2
            )
            
            # 繪製標籤（若有細分類，顯示中文標籤）
            truck_cls = det.get('truck_cls')
            if truck_cls:
                label = f"{truck_cls['label']} {truck_cls['confidence']:.2f}"
            else:
                label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(
                annotated, label,
                (bbox['x1'], bbox['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        return annotated, detections


# 測試
if __name__ == '__main__':
    detector = VehicleDetector()
    print(f"支援類別: {list(detector.VEHICLE_CLASSES.values())}")
