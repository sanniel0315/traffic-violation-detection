#!/usr/bin/env python3
"""感測器融合 API: 雷達 + 視覺 fusion 結果查詢

雷達硬體未到貨,用 MockRadarDriver 從現有 YOLO visual tracks 偽造 radar
detection,fuse 後給 UI 顯示 demo。真雷達到貨換 RadarDriver 子類即可。

endpoint:
- GET /api/sensor_fusion/{camera_id}/tracks → 當前 fused tracks
- GET /api/sensor_fusion/{camera_id}/summary → class / source 分布統計
"""
from typing import Dict

from fastapi import APIRouter, HTTPException

from detection.radar_track import MockRadarDriver
from detection.sensor_fusion import fuse, summarize

router = APIRouter(prefix="/api/sensor_fusion", tags=["雷達融合"])

# per-camera Mock driver (有狀態:per-camera 不同 random seed,track_id 連續性)
_MOCK_DRIVERS: Dict[int, MockRadarDriver] = {}


def _get_driver(camera_id: int) -> MockRadarDriver:
    drv = _MOCK_DRIVERS.get(camera_id)
    if drv is None:
        drv = MockRadarDriver(camera_id=camera_id, seed=camera_id * 17)
        _MOCK_DRIVERS[camera_id] = drv
    return drv


@router.get("/{camera_id}/tracks")
def get_fused_tracks(camera_id: int):
    """取當前 fused tracks: 雷達 + 視覺融合結果。"""
    from api.routes.stream import _VEHICLE_TRACK_SNAPSHOTS
    visual = list(_VEHICLE_TRACK_SNAPSHOTS.get(int(camera_id)) or [])

    drv = _get_driver(int(camera_id))
    drv.feed_visual_tracks(visual)
    radar = drv.get_detections()

    fused = fuse(radar, visual)
    return {
        "camera_id": int(camera_id),
        "visual_count": len(visual),
        "radar_count": len(radar),
        "tracks": [
            {
                "id": t.track_id,
                "world_x": round(t.world_x, 3),
                "world_y": round(t.world_y, 3),
                "vx": round(t.vx, 3),
                "vy": round(t.vy, 3),
                "speed_kmh": round(t.speed_kmh, 1),
                "class": t.vehicle_class,
                "class_confidence": round(t.class_confidence, 3),
                "source": t.source,
                "radar_track_id": t.radar_track_id,
                "visual_track_id": t.visual_track_id,
                "rcs": round(t.rcs, 2),
                "bbox": t.bbox,
            }
            for t in fused
        ],
        "summary": summarize(fused),
    }


@router.get("/{camera_id}/summary")
def get_fusion_summary(camera_id: int):
    """純 stats — class 分布 / source 分布,給 dashboard 卡片用"""
    from api.routes.stream import _VEHICLE_TRACK_SNAPSHOTS
    visual = list(_VEHICLE_TRACK_SNAPSHOTS.get(int(camera_id)) or [])
    drv = _get_driver(int(camera_id))
    drv.feed_visual_tracks(visual)
    radar = drv.get_detections()
    fused = fuse(radar, visual)
    return {
        "camera_id": int(camera_id),
        "visual_count": len(visual),
        "radar_count": len(radar),
        **summarize(fused),
    }
