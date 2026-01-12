#!/usr/bin/env python3
"""即時違規偵測服務"""
import sys
sys.path.insert(0, '/workspace')

import cv2
import time
import requests
import os
from datetime import datetime
from pathlib import Path

from detection.vehicle_detector import VehicleDetector


class DetectionService:
    """偵測服務"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.output_dir = Path("./output/violations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🚀 初始化偵測服務...")
        self.detector = VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.5)
        print("✅ 偵測服務初始化完成")
    
    def process_video(self, source: str, camera_id: int = 1, location: str = "測試路口"):
        """處理影片"""
        print(f"\n📹 開始處理: {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ 無法開啟: {source}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"✅ 影片開啟成功 (FPS: {fps:.1f}, 總幀數: {total_frames})")
        
        # 更新攝影機狀態
        self._update_camera_status(camera_id, "online")
        
        frame_count = 0
        detection_count = 0
        process_interval = max(1, int(fps / 3))  # 每秒處理 3 幀
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 跳幀
                if frame_count % process_interval != 0:
                    continue
                
                # 偵測
                detections = self.detector.detect(frame)
                vehicles = [d for d in detections if d['class_name'] in ['car', 'motorcycle', 'truck', 'bus']]
                
                if vehicles:
                    detection_count += 1
                    print(f"\r📊 幀 {frame_count}/{total_frames} | 偵測到 {len(vehicles)} 輛車", end="", flush=True)
                    
                    # 每 30 次偵測儲存一次截圖作為範例
                    if detection_count % 30 == 1:
                        self._save_sample(frame, vehicles, camera_id, location)
                
        except KeyboardInterrupt:
            print("\n⏹️ 使用者中斷")
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")
        finally:
            cap.release()
            self._update_camera_status(camera_id, "offline")
            print(f"\n📊 處理完成: {frame_count} 幀, {detection_count} 次偵測")
    
    def _save_sample(self, frame, detections, camera_id: int, location: str):
        """儲存範例"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 繪製標註
        annotated = frame.copy()
        for det in detections:
            bbox = det['bbox']
            cv2.rectangle(annotated, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(annotated, label, (bbox['x1'], bbox['y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 儲存圖片
        image_name = f"detection_{timestamp_str}.jpg"
        image_path = self.output_dir / image_name
        cv2.imwrite(str(image_path), annotated)
        
        # 模擬違規 (每次偵測都記錄)
        violation_types = ["RED_LIGHT", "SPEEDING", "ILLEGAL_PARKING", "WRONG_WAY"]
        v_type = violation_types[hash(timestamp_str) % len(violation_types)]
        v_names = {"RED_LIGHT": "闖紅燈", "SPEEDING": "超速", "ILLEGAL_PARKING": "違規停車", "WRONG_WAY": "逆向行駛"}
        v_fines = {"RED_LIGHT": 2700, "SPEEDING": 1800, "ILLEGAL_PARKING": 600, "WRONG_WAY": 900}
        
        # 模擬車牌
        import random
        plate = f"{random.choice('ABCDEFGH')}{random.choice('ABCDEFGH')}{random.choice('ABCDEFGH')}-{random.randint(1000,9999)}"
        
        data = {
            "violation_type": v_type,
            "violation_name": v_names[v_type],
            "vehicle_type": detections[0]['class_name'],
            "license_plate": plate,
            "location": location,
            "camera_id": camera_id,
            "confidence": detections[0]['confidence'],
            "fine_amount": v_fines[v_type],
            "points": 1,
            "image_path": f"/files/violations/{image_name}"
        }
        
        try:
            resp = requests.post(f"{self.api_url}/api/violations", json=data, timeout=5)
            if resp.status_code == 200:
                print(f"\n🚨 已記錄: {v_names[v_type]} | {plate} | {image_name}")
        except Exception as e:
            print(f"\n⚠️ API 錯誤: {e}")
    
    def _update_camera_status(self, camera_id: int, status: str):
        try:
            requests.put(f"{self.api_url}/api/cameras/{camera_id}", json={"status": status}, timeout=5)
        except:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', default='./source/test.mp4')
    parser.add_argument('--camera-id', '-c', type=int, default=3)
    parser.add_argument('--location', '-l', default='測試路口')
    parser.add_argument('--api', default='http://localhost:8000')
    args = parser.parse_args()
    
    service = DetectionService(api_url=args.api)
    service.process_video(args.source, args.camera_id, args.location)
