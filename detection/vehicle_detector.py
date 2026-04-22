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

    # 車種中文標籤對照（統一所有車種顯示）
    CLASS_LABEL_ZH = {
        'car':         '小客車',
        'motorcycle':  '機車',
        'bicycle':     '自行車',
        'person':      '行人',
        'truck':       '貨車',
        'bus':         '大客車',
        'heavy_truck': '大貨車',
        'light_truck': '小貨車',
        'non_truck':   '小客車',
    }

    @classmethod
    def get_zh_label(cls, class_name: str) -> str:
        return cls.CLASS_LABEL_ZH.get(str(class_name or ''), str(class_name or ''))

    # 共用 truck classifier（一個 instance 給所有 cam），classify() 有內部 lock
    _shared_truck_classifier = None
    _shared_truck_classifier_lock = None

    @classmethod
    def _get_shared_truck_classifier(cls):
        import threading as _th
        if cls._shared_truck_classifier_lock is None:
            cls._shared_truck_classifier_lock = _th.Lock()
        if cls._shared_truck_classifier is not None:
            return cls._shared_truck_classifier
        with cls._shared_truck_classifier_lock:
            if cls._shared_truck_classifier is not None:
                return cls._shared_truck_classifier
            from detection.truck_classifier import TruckClassifier
            tc = TruckClassifier()
            if tc.enabled:
                cls._shared_truck_classifier = tc
                print("♻️  共用 TruckClassifier 載入完成", flush=True)
            return cls._shared_truck_classifier

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
        # 優先使用同名 .engine（TensorRT 加速 ~3-4x），存在才切換
        engine_path = os.path.splitext(model_path)[0] + ".engine"
        if os.path.exists(engine_path) and os.getenv("DISABLE_TRT", "").lower() not in ("1", "true", "yes"):
            print(f"⚡ 偵測到 TensorRT engine，切換到 {engine_path}")
            model_path = engine_path
        self.model = YOLO(model_path, task='detect')
        self.conf_threshold = conf_threshold
        self.device = os.getenv("DEVICE", "cuda:0")
        self.runtime_device = "cpu"
        # TensorRT engine 已綁定 device，不需 .to()
        if not model_path.endswith(".engine"):
            try:
                self.model.to(self.device)
                self.runtime_device = self.device
            except Exception:
                self.runtime_device = "cpu"
        else:
            self.runtime_device = self.device
        self.vehicle_classes = self._resolve_vehicle_classes()

        # 大型車細分類器（可選）— 所有 VehicleDetector 共用單一 instance，避免 4 cam 各一份吃 GPU
        self.truck_classifier = None
        if enable_truck_cls:
            try:
                self.truck_classifier = VehicleDetector._get_shared_truck_classifier()
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
        # 第一輪：精確比對（避免 bicycle 的 alias "cycle" 子字串命中 "motorcycle"）
        for canonical, aliases in cls.CLASS_NAME_ALIASES.items():
            for alias in aliases:
                alias_norm = cls._normalize_label(alias)
                alias_compact = alias_norm.replace(" ", "")
                if name == alias_norm or compact == alias_compact:
                    return canonical
        # 第二輪：子字串比對（容忍 "pickup_truck" 之類的變體）
        for canonical, aliases in cls.CLASS_NAME_ALIASES.items():
            for alias in aliases:
                alias_norm = cls._normalize_label(alias)
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

                # 大型車細分類（共用 instance，加 lock 保護避免多 cam 併發）
                if (self.truck_classifier
                        and det['class_name'] in self._RECLASSIFY_CLASSES):
                    _tc_lock = VehicleDetector._shared_truck_classifier_lock
                    if _tc_lock is not None:
                        with _tc_lock:
                            cls_result = self.truck_classifier.classify(frame, det['bbox'])
                    else:
                        cls_result = self.truck_classifier.classify(frame, det['bbox'])
                    if cls_result['class_name'] == 'non_truck':
                        det['class_name'] = 'car'
                        det['truck_cls'] = cls_result
                    elif cls_result['class_name'] != 'unknown':
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
            
            # 繪製標籤（統一顯示中文，不帶信心度）
            truck_cls = det.get('truck_cls')
            if truck_cls:
                label = str(truck_cls['label'])
            else:
                label = self.get_zh_label(det['class_name'])
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
