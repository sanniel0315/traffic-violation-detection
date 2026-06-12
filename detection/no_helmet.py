#!/usr/bin/env python3
"""機車未戴安全帽 detector (§31).

判定邏輯:
1. 對每台 motorcycle (vehicle_class='motorcycle'),取機車 bbox 上方延伸 rider 區
2. 跑 helmet detection model 看 rider 頭上有沒有 helmet
3. without_helmet conf >= MIN_CONF + 連續 N frame 都沒戴 → 觸發
4. dedup 60 秒同 (cam, track) 不重複

避免雜訊:
- 軌跡 < MIN_TRACK_FRAMES 不算 (剛偵測到不穩)
- motorcycle bbox 太小 (< MIN_MOTORCYCLE_HEIGHT px) 不算 (太遠看不清頭)
- without_helmet conf < MIN_CONF 不算

仿 WrongWayEvaluator / PedestrianYieldEvaluator pattern: per camera state.
"""
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


MIN_CONF = 0.40                  # without_helmet 信心 < 此值不算
MIN_TRACK_FRAMES = 8             # track frame 數 < 此不檢查 (剛偵測到不穩)
MIN_MOTORCYCLE_HEIGHT = 80       # 機車 bbox 高度 < 此值跳過 (太遠看不清 rider)
CONSECUTIVE_FRAMES = 4           # 連續這麼多 frame 都 no helmet 才觸發
DEDUP_WINDOW_SEC = 60.0


@dataclass
class NoHelmetTrigger:
    vehicle_track_id: int
    vehicle_bbox: dict
    vehicle_class: str
    rider_region: tuple             # (x1,y1,x2,y2) crop 區域
    without_helmet_conf: float      # 觸發時最後一 frame 的信心


class NoHelmetEvaluator:
    """機車未戴安全帽 evaluator. per camera, 跨 frame 連續沒戴判定."""

    VIOLATION_TYPE = "NO_HELMET"
    VIOLATION_NAME = "未戴安全帽"

    def __init__(self, camera_id: int):
        self.camera_id = int(camera_id)
        # track_id → 連續 no_helmet frame 計數
        self._no_helmet_counts: Dict[int, int] = {}
        # track_id → 最後一 frame 的 without_helmet_conf (給 trigger 用)
        self._last_conf: Dict[int, float] = {}
        # track_id → 最後 rider_region
        self._last_region: Dict[int, tuple] = {}
        # dedup: (cam, track_id) → last fire_t
        self._fired: Dict[Tuple[int, int], float] = {}

    def feed(
        self,
        vehicles: List[dict],
        tracks: Dict[Any, Dict[str, Any]],
        frame,
        now_ts: float,
    ) -> List[NoHelmetTrigger]:
        """vehicles: detector 當前 frame 結果 (帶 bbox + track_id + class_name)
        tracks: stream.py tracks dict (含 frames count)
        frame: 當前 BGR frame (給 helmet detector 跑)
        """
        triggered: List[NoHelmetTrigger] = []
        if not vehicles or frame is None:
            return triggered
        # lazy import (避免循環 + helmet detector 載入慢)
        try:
            from services.helmet_detector import is_available as _hav, check_motorcycle_no_helmet as _chk
        except Exception:
            return triggered
        if not _hav():
            return triggered

        active_tids = set()
        for v in vehicles:
            cls = str(v.get("class_name") or "").lower()
            if cls != "motorcycle":
                continue
            tid = v.get("track_id")
            if tid is None:
                continue
            tid = int(tid)
            active_tids.add(tid)
            tr = tracks.get(tid) or {}
            frames = int(tr.get("frames", 0) or 0)
            if frames < MIN_TRACK_FRAMES:
                continue
            bb = v.get("bbox") or {}
            mh = float(bb.get("y2", 0)) - float(bb.get("y1", 0))
            if mh < MIN_MOTORCYCLE_HEIGHT:
                continue
            # dedup check
            fired_key = (self.camera_id, tid)
            last_fire = self._fired.get(fired_key, 0.0)
            if now_ts - last_fire < DEDUP_WINDOW_SEC:
                continue
            # 跑 helmet detector
            res = _chk(frame, bb, conf=0.30)
            wo_conf = float(res.get("without_helmet_conf") or 0.0)
            has_helmet = res.get("has_helmet")
            region = res.get("rider_region")

            if has_helmet is False and wo_conf >= MIN_CONF:
                self._no_helmet_counts[tid] = self._no_helmet_counts.get(tid, 0) + 1
                self._last_conf[tid] = wo_conf
                if region:
                    self._last_region[tid] = region
                if self._no_helmet_counts[tid] >= CONSECUTIVE_FRAMES:
                    # 觸發
                    triggered.append(NoHelmetTrigger(
                        vehicle_track_id=tid,
                        vehicle_bbox=bb,
                        vehicle_class=cls,
                        rider_region=region or (0, 0, 0, 0),
                        without_helmet_conf=wo_conf,
                    ))
                    self._fired[fired_key] = now_ts
                    # 觸發後 reset 計數,避免同 track 之後又連觸
                    self._no_helmet_counts[tid] = 0
            else:
                # 戴 helmet 或不確定 → reset 計數
                self._no_helmet_counts.pop(tid, None)
                self._last_conf.pop(tid, None)
                self._last_region.pop(tid, None)

        # 不在 active_tids 內的 track 也清掉
        for tid in list(self._no_helmet_counts.keys()):
            if tid not in active_tids:
                self._no_helmet_counts.pop(tid, None)
                self._last_conf.pop(tid, None)
                self._last_region.pop(tid, None)

        # GC dedup table — 過 5 分鐘的 entries 清掉
        for k in list(self._fired.keys()):
            if now_ts - self._fired[k] > 300:
                self._fired.pop(k, None)

        return triggered
