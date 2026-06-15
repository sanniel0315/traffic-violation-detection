"""車位佔用 — per-slot 時間遲滯 (hysteresis) 狀態機。

目的: 讓「空位」判定準確。原本 ROI 模式「單幀 YOLO 偵測不到車 = 空位」很脆弱,
低解析/雨/遮擋下漏抓停著的車 → 有車誤判成空位 (最糟)。

做法: 每個車位維護一個穩定狀態,**狀態翻面需要連續 K 幀一致** (非對稱):
- 空 → 有車: 連續 ON_FRAMES 幀偵測到車才翻 (新車進場合理快)
- 有車 → 空: 連續 OFF_FRAMES 幀都沒車才翻 (較嚴,避免單幀漏抓造成假空位)

每次 evaluate_occupancy 對某 source 跑一次 = 一「幀」。
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

_LOCK = threading.Lock()
# (source, slot_id) -> {"stable": bool, "run_val": bool, "run": int}
_STATE: Dict[Tuple[str, str], Dict] = {}

ON_FRAMES = 2    # 空→有車: 連續 2 幀偵測到車
OFF_FRAMES = 3   # 有車→空: 連續 3 幀都沒車 (寧可慢宣告空,因假空位最傷)


def smooth(source: str, slot_id, raw_occupied: bool) -> bool:
    """餵入單幀原始判定 raw_occupied,回傳遲滯後的穩定佔用狀態。"""
    key = (str(source), str(slot_id))
    raw = bool(raw_occupied)
    with _LOCK:
        st = _STATE.get(key)
        if st is None:
            # 冷啟動: 直接採用首次判定,不延遲
            _STATE[key] = {"stable": raw, "run_val": raw, "run": 1}
            return raw
        # 更新連續同值計數
        if raw == st["run_val"]:
            st["run"] += 1
        else:
            st["run_val"] = raw
            st["run"] = 1
        # 非對稱遲滯翻面
        if st["stable"]:            # 目前: 有車
            if (not raw) and st["run"] >= OFF_FRAMES:
                st["stable"] = False
        else:                       # 目前: 空
            if raw and st["run"] >= ON_FRAMES:
                st["stable"] = True
        return st["stable"]


def reset(source: Optional[str] = None) -> int:
    """清狀態 (source=None 清全部)。車位重標/重設後呼叫,避免舊狀態殘留。"""
    with _LOCK:
        if source is None:
            n = len(_STATE)
            _STATE.clear()
            return n
        keys = [k for k in _STATE if k[0] == str(source)]
        for k in keys:
            _STATE.pop(k, None)
        return len(keys)
