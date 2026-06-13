"""停車場車輛 counting — per source state, line-cross detection.

設計目的:
- entry/exit 共用同一條道路 (沒實體閘門),靠**方向**判進/出
- user 在 editor 畫 1 條 counting line (2 點)
- 法線方向標 enter (UI 提供翻轉 button)
- 每 evaluate frame 的 vehicles 跟上一 frame 做簡易 IoU + center 配對
  → 對 matched track 看 center 是否跨過 line
  → cross + 方向與 enter normal 同向 → enter_count++,反向 → exit_count++

cumulative state:
- enter_today / exit_today / in_lot 每天午夜 reset (in_lot 帶到隔天)

每 5 分鐘 sample 一次寫 DB (ParkingCountSample).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

# per source state
_STATE_LOCK = threading.Lock()
_STATE: Dict[str, Dict] = {}

# 配對參數
_MATCH_IOU_MIN = 0.20
_MATCH_CENTER_DIST_MAX = 80  # px, IoU 不夠時用 center distance fallback
_TRACK_MAX_AGE_SEC = 5.0

# DB throttle
_DB_THROTTLE_SEC = 300  # 5 分鐘 sample 一次


def _today_key() -> str:
    """台北日切 key (YYYY-MM-DD)"""
    tw = datetime.now(timezone(timedelta(hours=8)))
    return tw.strftime("%Y-%m-%d")


def _ensure_state(source: str) -> Dict:
    with _STATE_LOCK:
        st = _STATE.get(source)
        if st is None:
            st = {
                "tracks": [],         # list of {bbox, center, last_seen, last_side}
                "enter_today": 0,
                "exit_today": 0,
                "in_lot": 0,
                "day_key": _today_key(),
                "last_db_ts": 0.0,
            }
            _STATE[source] = st
        # 跨日 reset
        today = _today_key()
        if st["day_key"] != today:
            st["enter_today"] = 0
            st["exit_today"] = 0
            st["day_key"] = today
            # in_lot 保留 (帶到隔天)
        return st


def get_status(source: str) -> Dict:
    """回 counting 即時 state (給 API)"""
    st = _ensure_state(source)
    return {
        "source": source,
        "enter_today": st["enter_today"],
        "exit_today": st["exit_today"],
        "in_lot": st["in_lot"],
        "day_key": st["day_key"],
        "active_tracks": len(st["tracks"]),
    }


def reset_today(source: str) -> Dict:
    st = _ensure_state(source)
    st["enter_today"] = 0
    st["exit_today"] = 0
    return get_status(source)


def reset_all(source: str) -> Dict:
    """重設全部 (含 in_lot) — 給 user 重新 calibrate 用"""
    st = _ensure_state(source)
    st["enter_today"] = 0
    st["exit_today"] = 0
    st["in_lot"] = 0
    return get_status(source)


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / max(1, aa + bb - inter)


def _which_side(line: List[List[int]], cx: float, cy: float) -> float:
    """回 cross product 正負代表 point 在 line 的哪一邊.
    line = [[x1,y1],[x2,y2]]
    cross = (x2-x1)*(cy-y1) - (y2-y1)*(cx-x1)
    > 0 一邊, < 0 另一邊, == 0 在線上
    """
    x1, y1 = line[0]; x2, y2 = line[1]
    return (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)


def _line_normal_sign(line: List[List[int]], enter_normal: str = "right") -> int:
    """enter_normal 'right' = 線方向的右側為 enter 側 → cross > 0 = enter
    'left' = 反過來"""
    return 1 if enter_normal == "right" else -1


def feed(source: str,
         vehicles: List[Dict],
         counting_line: Optional[List[List[int]]],
         enter_normal: str = "right",
         now_ts: Optional[float] = None) -> Dict:
    """每 frame 餵入當前 vehicles list (YOLO 偵測).
    vehicles: [{bbox: {x1,y1,x2,y2}}, ...]
    counting_line: [[x1,y1],[x2,y2]] 或 None (沒設則略過,只更新 tracks)
    enter_normal: 'right' or 'left' 線方向哪邊是入場
    Return: status dict + 本次 frame 觸發的 events
    """
    if now_ts is None:
        now_ts = time.time()
    st = _ensure_state(source)
    events = []

    # 1. 算當前 frame 每車 center
    curr = []
    for v in vehicles:
        bb = v.get("bbox") or {}
        x1 = float(bb.get("x1", 0)); y1 = float(bb.get("y1", 0))
        x2 = float(bb.get("x2", 0)); y2 = float(bb.get("y2", 0))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        curr.append({"bbox": (x1, y1, x2, y2), "center": (cx, cy), "matched": False})

    # 2. 跟上一 frame tracks 配對 (greedy IoU first, then center distance)
    norm_sign = _line_normal_sign(counting_line or [], enter_normal) if counting_line else 0

    for tr in st["tracks"]:
        if now_ts - tr.get("last_seen", 0) > _TRACK_MAX_AGE_SEC:
            continue
        best_idx = -1; best_iou = 0.0; best_dist = 1e9
        for i, c in enumerate(curr):
            if c["matched"]:
                continue
            iou = _bbox_iou(tr["bbox"], c["bbox"])
            if iou > best_iou:
                best_iou = iou; best_idx = i
        if best_iou < _MATCH_IOU_MIN:
            # IoU 太弱,改用 center distance
            tcx, tcy = tr["center"]
            for i, c in enumerate(curr):
                if c["matched"]:
                    continue
                ccx, ccy = c["center"]
                d = ((ccx - tcx) ** 2 + (ccy - tcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d; best_idx = i if d < _MATCH_CENTER_DIST_MAX else best_idx
            if best_dist >= _MATCH_CENTER_DIST_MAX:
                continue
        if best_idx < 0:
            continue
        c = curr[best_idx]
        c["matched"] = True
        prev_center = tr["center"]
        tr["bbox"] = c["bbox"]; tr["center"] = c["center"]; tr["last_seen"] = now_ts
        # 線跨越判定
        if counting_line:
            prev_side = _which_side(counting_line, prev_center[0], prev_center[1])
            curr_side = _which_side(counting_line, c["center"][0], c["center"][1])
            if prev_side * curr_side < 0:  # 跨過線
                # curr_side * norm_sign > 0 → 進入 enter 側 = enter event
                if curr_side * norm_sign > 0:
                    st["enter_today"] += 1; st["in_lot"] += 1
                    events.append({"type": "enter", "center": c["center"]})
                else:
                    st["exit_today"] += 1; st["in_lot"] = max(0, st["in_lot"] - 1)
                    events.append({"type": "exit", "center": c["center"]})

    # 3. 沒被配對的 curr 變新 track
    for c in curr:
        if c["matched"]:
            continue
        st["tracks"].append({
            "bbox": c["bbox"], "center": c["center"], "last_seen": now_ts,
        })

    # 4. GC 太久沒見的 tracks
    st["tracks"] = [t for t in st["tracks"]
                    if now_ts - t.get("last_seen", 0) <= _TRACK_MAX_AGE_SEC]

    # 5. DB throttle
    if now_ts - st["last_db_ts"] >= _DB_THROTTLE_SEC:
        _write_db_sample(source, st)
        st["last_db_ts"] = now_ts

    return {
        "status": {
            "enter_today": st["enter_today"],
            "exit_today": st["exit_today"],
            "in_lot": st["in_lot"],
        },
        "events": events,
    }


def _write_db_sample(source: str, st: Dict) -> None:
    try:
        from api.models import SessionLocal, ParkingCountSample
        db = SessionLocal()
        try:
            row = ParkingCountSample(
                source=source,
                enter_today=int(st["enter_today"]),
                exit_today=int(st["exit_today"]),
                in_lot=int(st["in_lot"]),
            )
            db.add(row); db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"[parking_counter] db write fail: {e}", flush=True)


def history(source: str, hours: int = 24) -> List[Dict]:
    """24h 內 sample 歷史"""
    try:
        from api.models import SessionLocal, ParkingCountSample
        db = SessionLocal()
        try:
            start = datetime.utcnow() - timedelta(hours=hours)
            rows = (db.query(ParkingCountSample)
                      .filter(ParkingCountSample.source == source,
                              ParkingCountSample.created_at >= start)
                      .order_by(ParkingCountSample.created_at.asc())
                      .all())
            return [
                {"ts": r.created_at.isoformat() if r.created_at else None,
                 "enter": r.enter_today, "exit": r.exit_today, "in_lot": r.in_lot}
                for r in rows
            ]
        finally:
            db.close()
    except Exception:
        return []
