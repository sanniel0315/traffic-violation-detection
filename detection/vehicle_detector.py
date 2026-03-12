#!/usr/bin/env python3
"""車輛偵測模組 - 使用 YOLOv8"""
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from typing import List, Dict, Any
import numpy as np
import os
from model_paths import get_detect_model_pt


class VehicleDetector:
    """車輛偵測器"""
    
    # 交通相關類別
    VEHICLE_CLASSES = {
        0: 'person',
        1: 'bicycle', 
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):
        """
        初始化偵測器
        
        Args:
            model_path: YOLOv8 模型路徑
            conf_threshold: 信心度閾值
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
        print(f"✅ 車輛偵測器初始化完成 (模型: {model_path}, device: {self.runtime_device})")
    
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
                if class_id not in self.VEHICLE_CLASSES:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                detections.append({
                    'class_id': class_id,
                    'class_name': self.VEHICLE_CLASSES[class_id],
                    'confidence': confidence,
                    'bbox': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    }
                })
        
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
            'person': (0, 255, 0),      # 綠色
            'car': (255, 0, 0),         # 藍色
            'motorcycle': (0, 255, 255), # 黃色
            'bus': (255, 165, 0),       # 橙色
            'truck': (128, 0, 128),     # 紫色
            'bicycle': (0, 128, 255)    # 橘色
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
            
            # 繪製標籤
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
