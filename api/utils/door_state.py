"""兩道門的聚合狀態。

規則(現場需求):
  - 只要**任何一道門是開的** → 開門
  - **兩道門都確認關上** → 關上
  - 其餘(有鎖離線讀不到,無法確認全關) → 狀態不明

🛑 為什麼離線不當成「關上」:規則是「兩個門關上才顯示關上」,讀不到就無法確認
   那一道是關的,報「關上」等於憑空宣稱一件沒驗證過的事。但另一道若確定是開的,
   仍然顯示開門 —— 開門這件事只要一道成立就成立,不需要看另一道。

聚合邏輯獨立在這裡(不 import 硬體層),方便測試與給對外/內部兩個端點共用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

STATE_OPEN = "open"
STATE_CLOSED = "closed"
STATE_UNKNOWN = "unknown"

STATE_LABELS = {
    STATE_OPEN: "開門",
    STATE_CLOSED: "關上",
    STATE_UNKNOWN: "狀態不明",
}


def door_state_of(lock: Dict[str, Any]) -> str:
    """單一顆鎖的門狀態。

    `status.door.closed` 是 True(關)/False(開)/None(沒讀到)。
    鎖斷線時 status 可能整個是舊的,所以 connected=False 一律視為不明 ——
    寧可說不知道,也不要拿過期的值當現況。
    """
    if not lock.get("connected"):
        return STATE_UNKNOWN
    closed = ((lock.get("status") or {}).get("door") or {}).get("closed")
    if closed is True:
        return STATE_CLOSED
    if closed is False:
        return STATE_OPEN
    return STATE_UNKNOWN


def aggregate_door_state(lock_block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把 io_service.lock_status_block() 聚合成一個對外狀態。

    回傳 state / label / 各門明細 / 計數,呼叫端直接丟給客戶端顯示。
    """
    block = lock_block or {}
    locks: List[Dict[str, Any]] = list(block.get("locks") or [])

    doors = []
    for lk in locks:
        st = door_state_of(lk)
        doors.append({
            "addr": lk.get("addr"),
            "name": lk.get("name"),
            "online": bool(lk.get("connected")),
            "state": st,
            "label": STATE_LABELS[st],
        })

    open_count = sum(1 for d in doors if d["state"] == STATE_OPEN)
    closed_count = sum(1 for d in doors if d["state"] == STATE_CLOSED)
    unknown_count = sum(1 for d in doors if d["state"] == STATE_UNKNOWN)

    if not block.get("enabled") or not doors:
        state = STATE_UNKNOWN          # 功能沒啟用/沒有鎖 → 不能宣稱關上
    elif open_count > 0:
        state = STATE_OPEN             # 一道開就是開,不必看另一道
    elif unknown_count > 0:
        state = STATE_UNKNOWN          # 沒有開的,但也無法確認全關
    else:
        state = STATE_CLOSED           # 每一道都確認關上

    return {
        "state": state,
        "label": STATE_LABELS[state],
        "enabled": bool(block.get("enabled")),
        "door_count": len(doors),
        "open_count": open_count,
        "closed_count": closed_count,
        "unknown_count": unknown_count,
        "doors": doors,
    }
