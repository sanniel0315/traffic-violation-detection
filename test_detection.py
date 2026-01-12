#!/usr/bin/env python3
"""測試 YOLOv8 物件偵測"""
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

# 載入模型
model = YOLO('yolov8n.pt')

# 偵測目標類別 (交通相關)
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'person']

print("=" * 50)
print("YOLOv8 物件偵測測試")
print("=" * 50)
print(f"模型類別數: {len(model.names)}")
print(f"交通相關類別: {VEHICLE_CLASSES}")

# 取得類別 ID
class_ids = {v: k for k, v in model.names.items() if v in VEHICLE_CLASSES}
print(f"類別 ID 對應: {class_ids}")

print("\n✅ 偵測模組準備就緒！")
print("=" * 50)
