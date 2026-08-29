# -*- coding: utf-8 -*-
"""動態號誌基準/fallback 查表

讀 config/system/ramp_timing_baseline.json(中央官方時制表 0814),提供決策引擎與
fallback 用的唯讀查詢:

    now → 當下該跑哪組計畫(時段查表,也是規範 E 失效回退目標)
    plan_id → 綠燈/週期/min·max/黃全紅(計畫查表)
    分相 → 匝道角色/儲車上限/測點/優先權(時相查表)

⚠ 純唯讀資料層,不下任何號誌命令。決策引擎(Phase 2)與 fallback(Phase 4/E)在此之上建。
"""
from __future__ import annotations

import datetime as _dt
import json as _json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_BASELINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "config" / "system" / "ramp_timing_baseline.json"
)


@lru_cache(maxsize=1)
def load_baseline() -> dict:
    """載入基準時制表(結果快取)。"""
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        return _json.load(f)


def _day_key(dt: _dt.datetime) -> str:
    """星期 → 排程分類鍵(weekday / saturday / sunday)。"""
    wd = dt.weekday()  # 0=一 … 5=六 6=日
    if wd == 5:
        return "saturday"
    if wd == 6:
        return "sunday"
    return "weekday"


def current_base_plan(dt: Optional[_dt.datetime] = None) -> int:
    """回傳 dt 當下依官方排程「應該在跑」的計畫編號。

    這同時是規範 (E) 動態失效時的 fallback 目標 —— 回退到當下時段的排程計畫,
    不是隨便一組。dt 省略時用現在時間。
    """
    dt = dt or _dt.datetime.now()
    sched = load_baseline()["schedule"][_day_key(dt)]
    hhmm = dt.strftime("%H:%M")
    # 取「開始時間 <= 現在」中最晚的一筆;若都比現在晚(凌晨),取當日最後一筆(前一段延續)
    chosen = sched[-1][1]
    for start, plan_id in sched:
        if start <= hhmm:
            chosen = plan_id
    return int(chosen)


def fallback_plan(dt: Optional[_dt.datetime] = None) -> int:
    """規範 (E) 失效回退目標 = 當下時段的排程計畫。"""
    return current_base_plan(dt)


def plan_params(plan_id: int) -> Optional[dict]:
    """計畫查表:綠燈/週期/min·max/黃全紅。查無回 None。"""
    return load_baseline()["plans"].get(str(plan_id))


def phase_role(phase_no: int) -> Optional[dict]:
    """時相查表:匝道角色/儲車上限/測點/是否主線保護優先。查無回 None。"""
    return load_baseline()["phases"].get(str(phase_no))


def storage_limit_m(phase_no: int) -> Optional[int]:
    """該分相對應匝道的儲車上限(公尺)。分相2(下匝道)=600、分相1(上匝道)=210。"""
    p = phase_role(phase_no)
    return int(p["storage_m"]) if p else None


def priority_phase() -> int:
    """主線保護優先的分相(下匝道)。"""
    for no, p in load_baseline()["phases"].items():
        if p.get("priority"):
            return int(no)
    return 2


def base_split(plan_id: int) -> Optional[tuple]:
    """該計畫的基準綠燈分配 (分相1綠, 分相2綠) —— 動態微調的起點。"""
    p = plan_params(plan_id)
    return (p["phase1_green"], p["phase2_green"]) if p else None


if __name__ == "__main__":
    # 簡易自我測試(唯讀,安全)
    b = load_baseline()
    print("站點:", b["site"], "|", b["direction"], "| 來源:", b["source"])
    print("分相優先(主線保護):分相", priority_phase())
    for wd, label in [(0, "平日一"), (5, "週六"), (6, "週日")]:
        for h in (2, 7, 9, 12, 17, 20):
            dt = _dt.datetime(2026, 8, 24 + wd, h, 0)  # 2026-08-24 是週一
            pid = current_base_plan(dt)
            sp = base_split(pid)
            print(f"  {label} {h:02d}:00 → 計畫 {pid} 分相綠 {sp}")
        break  # 只示範平日
    print("計畫 1 參數:", plan_params(1))
    print("分相2(下匝道)儲車上限:", storage_limit_m(2), "m")
