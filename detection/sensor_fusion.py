#!/usr/bin/env python3
"""感測器融合: 雷達 track + 視覺 (YOLO) bbox 在 BEV 平面 association

核心算法:
1. 視覺 bbox 用 pixel→world calibration 投到 BEV
2. 雷達 track 直接在 BEV (從 driver 拿)
3. 同 timestamp,計算 visual_BEV ↔ radar 兩兩距離
4. Hungarian matching 找最佳配對 (scipy.optimize.linear_sum_assignment)
5. 距離 < threshold 才算 match,否則 radar-only / visual-only

輸出 FusedTrack:
- 有 match → class from visual + speed from radar (radar 速度準) + position 兩者平均
- radar-only → 知道有東西但 class 'unknown'
- visual-only → 沒被雷達看到,可能離太遠或雷達遺漏
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np

from detection.radar_track import RadarDetection


@dataclass
class FusedTrack:
    """融合 track,代表一個被感知的物體"""
    track_id: int
    world_x: float
    world_y: float
    vx: float
    vy: float
    vehicle_class: str = "unknown"
    class_confidence: float = 0.0
    source: str = "fused"           # 'fused' | 'radar_only' | 'visual_only'
    radar_track_id: Optional[int] = None
    visual_track_id: Optional[int] = None
    rcs: float = 0.0
    bbox: Optional[Dict[str, int]] = None  # 視覺 bbox (pixel coord)
    timestamp: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def speed_kmh(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy) * 3.6


# 視覺 track 跟雷達 track 距離超過此 threshold (m) 視為不同物
ASSOCIATION_DISTANCE_THRESHOLD_M = 3.0


def _hungarian_match(
    cost_matrix: np.ndarray,
    max_cost: float,
) -> List[Tuple[int, int]]:
    """跑 scipy Hungarian,回傳 cost <= max_cost 的 (row, col) pair。
    沒 scipy 就 fallback 貪婪 nearest neighbor。"""
    if cost_matrix.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)
                 if cost_matrix[r, c] <= max_cost]
        return pairs
    except ImportError:
        # 貪婪 fallback:每次取 cost 最小的 pair,佔用後 mask 掉
        pairs = []
        used_rows = set()
        used_cols = set()
        flat = [(cost_matrix[r, c], r, c)
                for r in range(cost_matrix.shape[0])
                for c in range(cost_matrix.shape[1])]
        flat.sort()
        for cost, r, c in flat:
            if cost > max_cost:
                break
            if r in used_rows or c in used_cols:
                continue
            pairs.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
        return pairs


def fuse(
    radar_detections: List[RadarDetection],
    visual_tracks: List[Dict[str, Any]],
    *,
    distance_threshold_m: float = ASSOCIATION_DISTANCE_THRESHOLD_M,
    timestamp: Optional[float] = None,
) -> List[FusedTrack]:
    """主 fusion 入口。

    visual_tracks 每筆需含:
        track_id (int), world_x (m), world_y (m), vx (m/s), vy (m/s),
        class_name (str), confidence (float), bbox (dict, pixel coord)
    """
    import time as _t
    ts = timestamp if timestamp is not None else _t.time()
    if not radar_detections and not visual_tracks:
        return []

    # 構 cost matrix (距離)
    n_radar = len(radar_detections)
    n_visual = len(visual_tracks)
    cost = np.full((n_radar, n_visual), 1e9, dtype=np.float32)
    for i, r in enumerate(radar_detections):
        for j, v in enumerate(visual_tracks):
            dx = r.x - float(v.get("world_x", 0.0))
            dy = r.y - float(v.get("world_y", 0.0))
            cost[i, j] = math.sqrt(dx * dx + dy * dy)

    pairs = _hungarian_match(cost, distance_threshold_m)
    matched_radar = {p[0] for p in pairs}
    matched_visual = {p[1] for p in pairs}

    fused: List[FusedTrack] = []
    next_id = 1

    # 配對成功 → fused (class 用 visual, speed 用 radar, position 兩者平均)
    for r_idx, v_idx in pairs:
        r = radar_detections[r_idx]
        v = visual_tracks[v_idx]
        wx = (r.x + float(v.get("world_x", 0.0))) * 0.5
        wy = (r.y + float(v.get("world_y", 0.0))) * 0.5
        fused.append(FusedTrack(
            track_id=next_id,
            world_x=wx, world_y=wy,
            vx=r.vx, vy=r.vy,             # 雷達速度準,優先
            vehicle_class=str(v.get("class_name", "unknown")),
            class_confidence=float(v.get("confidence", 0.0)),
            source="fused",
            radar_track_id=r.track_id,
            visual_track_id=int(v.get("track_id", 0)),
            rcs=r.rcs,
            bbox=v.get("bbox"),
            timestamp=ts,
        ))
        next_id += 1

    # 雷達 only — 知道有物體但沒視覺 (可能遮擋、太遠、太小)
    for i, r in enumerate(radar_detections):
        if i in matched_radar:
            continue
        fused.append(FusedTrack(
            track_id=next_id,
            world_x=r.x, world_y=r.y,
            vx=r.vx, vy=r.vy,
            vehicle_class="unknown",
            class_confidence=0.0,
            source="radar_only",
            radar_track_id=r.track_id,
            rcs=r.rcs,
            timestamp=ts,
        ))
        next_id += 1

    # 視覺 only — 雷達沒看到 (可能雷達遺漏、靜止難測、行人 RCS 太小)
    for j, v in enumerate(visual_tracks):
        if j in matched_visual:
            continue
        fused.append(FusedTrack(
            track_id=next_id,
            world_x=float(v.get("world_x", 0.0)),
            world_y=float(v.get("world_y", 0.0)),
            vx=float(v.get("vx", 0.0)),
            vy=float(v.get("vy", 0.0)),
            vehicle_class=str(v.get("class_name", "unknown")),
            class_confidence=float(v.get("confidence", 0.0)),
            source="visual_only",
            visual_track_id=int(v.get("track_id", 0)),
            bbox=v.get("bbox"),
            timestamp=ts,
        ))
        next_id += 1

    return fused


def summarize(fused: List[FusedTrack]) -> Dict[str, Any]:
    """產生統計摘要,給 dashboard / log 用"""
    from collections import Counter
    cls_dist = Counter(t.vehicle_class for t in fused)
    src_dist = Counter(t.source for t in fused)
    return {
        "total": len(fused),
        "by_class": dict(cls_dist),
        "by_source": dict(src_dist),
        "radar_visual_fused_count": src_dist.get("fused", 0),
    }
