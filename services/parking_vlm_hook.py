"""
VLM 仲裁 hook — 在 evaluate_occupancy 偵測到 borderline slot (conf 0.3~0.6)
或 PKLot/YOLO disagree 時,把 slot 排入背景 queue,worker 每 30s 跑 N 個
VLM query,結果存 cache 5 分鐘.

評估主路徑不卡 VLM 推論時間 (~5s/格),只取最近 cache 結果附在 slot.

API:
- enqueue(source_key, slot_id, polygon): 排入候選
- get_verdict(source_key, slot_id) -> dict | None: 取 cache
- start_worker(): 啟動背景 thread (一次即可)
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

_QUEUE_LOCK = threading.Lock()
_QUEUE: "OrderedDict[Tuple[str, str], List[List[int]]]" = OrderedDict()

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[Tuple[str, str], Dict] = {}
_CACHE_TTL_SEC = 300

_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()

WORKER_INTERVAL_SEC = 30
MAX_PER_BATCH = 5


def enqueue(source_key: str, slot_id: str, polygon: List[List[int]]) -> None:
    if not source_key or not slot_id or not polygon:
        return
    key = (source_key, str(slot_id))
    with _QUEUE_LOCK:
        # OrderedDict — 若已存在,移到尾端 (LRU)
        if key in _QUEUE:
            _QUEUE.move_to_end(key)
        _QUEUE[key] = polygon


def get_verdict(source_key: str, slot_id: str) -> Optional[Dict]:
    key = (source_key, str(slot_id))
    now = time.time()
    with _CACHE_LOCK:
        v = _CACHE.get(key)
        if v is None:
            return None
        if now - v.get("ts", 0) > _CACHE_TTL_SEC:
            _CACHE.pop(key, None)
            return None
        return dict(v)


def list_verdicts(source_key: str) -> Dict[str, Dict]:
    """回傳該 source 所有未過期 cache,key=slot_id"""
    out = {}
    now = time.time()
    with _CACHE_LOCK:
        for (s, sid), v in list(_CACHE.items()):
            if s != source_key:
                continue
            if now - v.get("ts", 0) > _CACHE_TTL_SEC:
                _CACHE.pop((s, sid), None)
                continue
            out[sid] = dict(v)
    return out


def start_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True
    t = threading.Thread(target=_worker_loop, daemon=True, name="vlm_arbiter")
    t.start()
    print("[vlm_hook] background arbiter started", flush=True)


def _worker_loop() -> None:
    while True:
        try:
            _process_batch()
        except Exception as e:
            print(f"[vlm_hook] worker err: {e}", flush=True)
        time.sleep(WORKER_INTERVAL_SEC)


def _process_batch() -> None:
    # 只有 VLM 載入完才跑
    try:
        from services.parking_vlm import is_loaded, query_slot, crop_slot_from_frame
    except Exception:
        return
    if not is_loaded():
        return
    # 取 batch
    batch: List[Tuple[Tuple[str, str], List[List[int]]]] = []
    with _QUEUE_LOCK:
        while _QUEUE and len(batch) < MAX_PER_BATCH:
            k, v = _QUEUE.popitem(last=False)  # FIFO
            batch.append((k, v))
    if not batch:
        return
    # 同 source 共用 frame
    from services.parking_occupancy import fetch_frame
    frame_cache: Dict[str, "object"] = {}
    for (source_key, slot_id), polygon in batch:
        frame = frame_cache.get(source_key)
        if frame is None:
            frame = fetch_frame(source_key, bypass_cache=True)
            frame_cache[source_key] = frame
        if frame is None:
            continue
        try:
            crop = crop_slot_from_frame(frame, polygon, padding=1.0, highlight=True)
            if crop is None:
                continue
            result = query_slot(crop)
            if not result.get("ok"):
                continue
            with _CACHE_LOCK:
                _CACHE[(source_key, slot_id)] = {
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "confidence": result.get("confidence"),
                    "occupied": result.get("occupied"),
                    "latency_sec": result.get("latency_sec"),
                    "ts": time.time(),
                }
            print(f"[vlm_hook] verdict {source_key}/{slot_id}: {result.get('status')}",
                  flush=True)
        except Exception as e:
            print(f"[vlm_hook] query err {source_key}/{slot_id}: {e}", flush=True)
