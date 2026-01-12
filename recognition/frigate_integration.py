#!/usr/bin/env python3
"""Frigate NVR 整合模組"""
import os
import json
import requests
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path


class FrigateClient:
    """Frigate API 客戶端"""
    
    def __init__(self, host: str = "localhost", port: int = 5000):
        """
        初始化 Frigate 客戶端
        
        Args:
            host: Frigate 主機
            port: Frigate 端口
        """
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"
        
    def _get(self, endpoint: str) -> Optional[Dict]:
        """GET 請求"""
        try:
            response = requests.get(f"{self.api_url}/{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[Frigate] GET 錯誤 {endpoint}: {e}")
            return None
    
    def _post(self, endpoint: str, data: Dict = None) -> Optional[Dict]:
        """POST 請求"""
        try:
            response = requests.post(
                f"{self.api_url}/{endpoint}",
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[Frigate] POST 錯誤 {endpoint}: {e}")
            return None
    
    # ============ 系統狀態 ============
    
    def get_stats(self) -> Optional[Dict]:
        """取得系統統計"""
        return self._get("stats")
    
    def get_config(self) -> Optional[Dict]:
        """取得配置"""
        return self._get("config")
    
    def get_version(self) -> Optional[str]:
        """取得版本"""
        result = self._get("version")
        return result.get("version") if result else None
    
    # ============ 攝影機 ============
    
    def get_cameras(self) -> Dict[str, Dict]:
        """取得所有攝影機狀態"""
        stats = self.get_stats()
        if stats:
            return stats.get("cameras", {})
        return {}
    
    def get_camera_snapshot(self, camera_name: str) -> Optional[bytes]:
        """取得攝影機快照"""
        try:
            url = f"{self.api_url}/{camera_name}/latest.jpg"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"[Frigate] 快照錯誤 {camera_name}: {e}")
            return None
    
    # ============ 事件 ============
    
    def get_events(
        self,
        camera: str = None,
        label: str = None,
        zone: str = None,
        after: float = None,
        before: float = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        查詢事件
        
        Args:
            camera: 攝影機名稱
            label: 物件標籤 (car, motorcycle, person)
            zone: 區域名稱
            after: 開始時間戳
            before: 結束時間戳
            limit: 數量限制
        """
        params = []
        if camera:
            params.append(f"camera={camera}")
        if label:
            params.append(f"label={label}")
        if zone:
            params.append(f"zone={zone}")
        if after:
            params.append(f"after={after}")
        if before:
            params.append(f"before={before}")
        params.append(f"limit={limit}")
        
        query = "&".join(params)
        result = self._get(f"events?{query}")
        return result if result else []
    
    def get_event(self, event_id: str) -> Optional[Dict]:
        """取得單一事件詳情"""
        return self._get(f"events/{event_id}")
    
    def get_event_snapshot(self, event_id: str) -> Optional[bytes]:
        """取得事件快照"""
        try:
            url = f"{self.api_url}/events/{event_id}/snapshot.jpg"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"[Frigate] 事件快照錯誤 {event_id}: {e}")
            return None
    
    def get_event_clip(self, event_id: str) -> Optional[bytes]:
        """取得事件影片"""
        try:
            url = f"{self.api_url}/events/{event_id}/clip.mp4"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"[Frigate] 事件影片錯誤 {event_id}: {e}")
            return None
    
    def delete_event(self, event_id: str) -> bool:
        """刪除事件"""
        try:
            response = requests.delete(
                f"{self.api_url}/events/{event_id}",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"[Frigate] 刪除事件錯誤 {event_id}: {e}")
            return False
    
    # ============ 錄影 ============
    
    def get_recordings(
        self,
        camera: str,
        after: float = None,
        before: float = None
    ) -> List[Dict]:
        """查詢錄影"""
        params = []
        if after:
            params.append(f"after={after}")
        if before:
            params.append(f"before={before}")
        
        query = "&".join(params) if params else ""
        endpoint = f"{camera}/recordings"
        if query:
            endpoint += f"?{query}"
        
        result = self._get(endpoint)
        return result if result else []


class FrigateEventProcessor:
    """Frigate 事件處理器 - 整合車牌辨識"""
    
    def __init__(
        self,
        frigate_client: FrigateClient,
        plate_recognizer,
        violation_detector,
        output_dir: str = "./output/violations"
    ):
        """
        初始化事件處理器
        
        Args:
            frigate_client: Frigate 客戶端
            plate_recognizer: 車牌辨識器
            violation_detector: 違規偵測器
            output_dir: 輸出目錄
        """
        self.frigate = frigate_client
        self.recognizer = plate_recognizer
        self.detector = violation_detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 已處理事件
        self.processed_events = set()
    
    def process_event(self, event: Dict) -> Optional[Dict]:
        """
        處理單一 Frigate 事件
        
        Args:
            event: Frigate 事件資料
            
        Returns:
            違規記錄 (如果偵測到違規)
        """
        event_id = event.get("id")
        
        # 檢查是否已處理
        if event_id in self.processed_events:
            return None
        
        # 取得事件資訊
        camera = event.get("camera")
        label = event.get("label")  # car, motorcycle, etc.
        zones = event.get("zones", [])
        start_time = event.get("start_time")
        end_time = event.get("end_time")
        
        # 只處理車輛事件
        if label not in ["car", "motorcycle", "bus", "truck"]:
            return None
        
        # 取得事件快照
        snapshot = self.frigate.get_event_snapshot(event_id)
        if not snapshot:
            return None
        
        # 轉換為 OpenCV 格式
        import cv2
        import numpy as np
        nparr = np.frombuffer(snapshot, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        # 車牌辨識
        plate_result = self._recognize_plate(frame, event)
        
        # 違規檢測
        violation = self._check_violation(event, zones)
        
        if violation:
            # 建立違規記錄
            violation_record = {
                "frigate_event_id": event_id,
                "camera": camera,
                "vehicle_type": label,
                "plate_number": plate_result.get("plate_number"),
                "plate_confidence": plate_result.get("confidence", 0),
                "violation_type": violation["type"],
                "zone": violation.get("zone"),
                "start_time": datetime.fromtimestamp(start_time).isoformat() if start_time else None,
                "end_time": datetime.fromtimestamp(end_time).isoformat() if end_time else None,
                "snapshot_path": None,
                "clip_path": None
            }
            
            # 儲存快照
            snapshot_path = self._save_snapshot(frame, violation_record)
            violation_record["snapshot_path"] = str(snapshot_path)
            
            # 標記為已處理
            self.processed_events.add(event_id)
            
            return violation_record
        
        return None
    
    def _recognize_plate(self, frame, event: Dict) -> Dict:
        """辨識車牌"""
        if not self.recognizer:
            return {"plate_number": None, "confidence": 0}
        
        # 從事件取得車輛位置
        box = event.get("box", {})
        if box:
            h, w = frame.shape[:2]
            x1 = int(box.get("x", 0) * w)
            y1 = int(box.get("y", 0) * h)
            x2 = x1 + int(box.get("width", 0) * w)
            y2 = y1 + int(box.get("height", 0) * h)
            
            vehicle_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            return self.recognizer.process_vehicle(frame, vehicle_bbox)
        
        return {"plate_number": None, "confidence": 0}
    
    def _check_violation(self, event: Dict, zones: List[str]) -> Optional[Dict]:
        """檢查是否違規"""
        # 定義違規區域對應
        violation_zones = {
            "sidewalk_zone": "駕車行駛人行道或騎樓",
            "crosswalk_zone": "不停讓行人",
            "no_parking_zone": "違規停車",
            "red_light_zone": "闖紅燈"
        }
        
        for zone in zones:
            if zone in violation_zones:
                return {
                    "type": violation_zones[zone],
                    "zone": zone
                }
        
        return None
    
    def _save_snapshot(self, frame, violation: Dict) -> Path:
        """儲存違規快照"""
        import cv2
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plate = violation.get("plate_number") or "unknown"
        filename = f"{timestamp}_{plate}_{violation['violation_type']}.jpg"
        
        # 清理檔名
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), frame)
        
        return filepath
    
    async def poll_events(self, interval: float = 5.0):
        """
        輪詢 Frigate 事件
        
        Args:
            interval: 輪詢間隔 (秒)
        """
        print(f"[Frigate] 開始輪詢事件 (間隔: {interval}秒)")
        
        last_check = datetime.now().timestamp() - 60  # 從1分鐘前開始
        
        while True:
            try:
                # 查詢新事件
                events = self.frigate.get_events(
                    after=last_check,
                    limit=20
                )
                
                for event in events:
                    violation = self.process_event(event)
                    if violation:
                        print(f"[違規] {violation['violation_type']} - {violation.get('plate_number', '未知')}")
                        # 這裡可以加入資料庫儲存
                
                last_check = datetime.now().timestamp()
                
            except Exception as e:
                print(f"[Frigate] 輪詢錯誤: {e}")
            
            await asyncio.sleep(interval)


# 測試
if __name__ == "__main__":
    client = FrigateClient(host="localhost", port=5000)
    
    # 測試連線
    version = client.get_version()
    if version:
        print(f"✅ Frigate 連線成功，版本: {version}")
        
        # 取得攝影機
        cameras = client.get_cameras()
        print(f"📹 攝影機: {list(cameras.keys())}")
        
        # 取得最近事件
        events = client.get_events(limit=5)
        print(f"📝 最近事件: {len(events)} 筆")
    else:
        print("❌ 無法連線到 Frigate")
