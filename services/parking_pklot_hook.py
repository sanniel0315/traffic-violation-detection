"""PKLot 逐格分類 — 背景 hook (P2)。

目的: 補 YOLO 漏抓造成的「假空位」。PKLot model 直接判每格 空(e)/有車(o),
抓得到 YOLO 漏的車。但 detect_slots ~2s/張太慢,不能每次 poll 跑 →
照 VLM hook 模式: 背景 worker 每 INTERVAL 對 active source 跑一次,
快取每格 e/o 結果;主 evaluate 只讀快取融合 (不卡)。

GPU 推論用共用 INFER_LOCK 跟 vehicle YOLO 序列化,避免併發 native SEGV。
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

# 共用 GPU 推論鎖 (evaluate 的 vehicle YOLO 也要 acquire,序列化避免併發 SEGV)
INFER_LOCK = threading.Lock()

_LOCK = threading.Lock()
_VERDICTS: Dict[Tuple[str, str], Dict] = {}   # (source, slot_id) -> {occupied, conf, ts}
_ACTIVE: Dict[str, float] = {}                # source -> last_request_ts
_TTL = 120.0          # 快取有效秒數
_INTERVAL = 45.0      # worker 每 N 秒掃一輪
_ACTIVE_WINDOW = 300  # source 最近 5 分鐘有被查才跑
_OCC_CONF = 0.25      # PKLot 'o' 達此 conf 才視為「確認有車」(PKLot conf 偏低)

_WORKER_STARTED = False
_WSTART_LOCK = threading.Lock()


def mark_active(source: str) -> None:
    with _LOCK:
        _ACTIVE[str(source)] = time.time()


def get_slot_verdict(source: str, slot_id) -> Optional[Dict]:
    """回該格最近 PKLot 結果 {occupied, conf, ts} 或 None (無/過期)。"""
    key = (str(source), str(slot_id))
    now = time.time()
    with _LOCK:
        v = _VERDICTS.get(key)
        if v and (now - v["ts"]) <= _TTL:
            return dict(v)
    return None


def start_worker() -> None:
    global _WORKER_STARTED
    with _WSTART_LOCK:
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True
    threading.Thread(target=_loop, daemon=True, name="pklot_hook").start()
    print("[pklot_hook] background worker started", flush=True)


def _loop() -> None:
    while True:
        try:
            _process()
        except Exception as e:
            print(f"[pklot_hook] worker err: {e}", flush=True)
        time.sleep(_INTERVAL)


def _process() -> None:
    try:
        from services.parking_pklot_model import is_available, detect_slots
    except Exception:
        return
    if not is_available():
        return
    now = time.time()
    with _LOCK:
        sources = [s for s, ts in _ACTIVE.items() if (now - ts) <= _ACTIVE_WINDOW]
    if not sources:
        return
    from services.parking_occupancy import (
        fetch_frame, load_slots, _point_in_polygon, _bbox_iou_with_polygon,
    )
    for source in sources:
        try:
            slots = load_slots(source) or []
            if not slots:
                continue
            frame = fetch_frame(source)
            if frame is None:
                continue
            with INFER_LOCK:
                dets = detect_slots(frame) or []
            # 每格找覆蓋它的 PKLot 偵測 (中心在格內優先,否則 IoU>0.2),取 conf 最高
            for s in slots:
                poly = s.get("polygon") or []
                sid = str(s["id"])
                if len(poly) < 3:
                    continue
                best = None
                for d in dets:
                    cx = (d["x1"] + d["x2"]) // 2
                    cy = (d["y1"] + d["y2"]) // 2
                    hit = _point_in_polygon(cx, cy, poly)
                    if not hit:
                        iou = _bbox_iou_with_polygon((d["x1"], d["y1"], d["x2"], d["y2"]), poly)
                        if iou <= 0.2:
                            continue
                    if best is None or d["conf"] > best["conf"]:
                        best = d
                if best is not None:
                    with _LOCK:
                        _VERDICTS[(source, sid)] = {
                            "occupied": bool(best["occupied"]),
                            "conf": float(best["conf"]),
                            "ts": now,
                        }
        except Exception as e:
            print(f"[pklot_hook] {source} err: {e}", flush=True)


def is_occupied_signal(source: str, slot_id) -> bool:
    """PKLot 是否確認該格有車 (conf 達門檻)。給 evaluate 補 YOLO 漏抓用。"""
    v = get_slot_verdict(source, slot_id)
    return bool(v and v["occupied"] and v["conf"] >= _OCC_CONF)
