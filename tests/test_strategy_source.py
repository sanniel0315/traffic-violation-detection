"""「現在是什麼控制策略」要以 5FC0(查詢回報)為準,不是 5F00(變化通知)。

🛑 2026-09-17 現場問「切了動態控制為什麼還在定時控制」:續約瞬間控制器會連報
   01 → 10 的暫態,只取最後一則 5F00 會停在「定時控制」,但實際是 14 時相控制。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture
def T(monkeypatch):
    from api.routes import signal_tc3 as T
    monkeypatch.setitem(T._safety, "strategy", None)
    monkeypatch.setitem(T._safety, "strategy_ts", 0.0)
    monkeypatch.setitem(T._safety, "strategy_confirmed", None)
    monkeypatch.setitem(T._safety, "strategy_confirmed_ts", 0.0)
    return T


def _rec(code, strategy, ts=1000.0):
    return {"code": code, "strategy": strategy, "ts": ts, "addr": 0xFFFF, "cks_ok": True}


def test_query_reply_sets_the_confirmed_strategy(T):
    T._safety_watch(_rec("5FC0", 0x14))
    assert T._safety["strategy_confirmed"] == 0x14


def test_change_notice_does_not_overwrite_the_confirmed_value(T):
    """續約瞬間的 5F00 暫態不可以蓋掉實際策略。"""
    T._safety_watch(_rec("5FC0", 0x14))
    T._safety_watch(_rec("5F00", 0x01, ts=1001.0))     # 暫態
    T._safety_watch(_rec("5F00", 0x10, ts=1002.0))     # 暫態
    assert T._safety["strategy_confirmed"] == 0x14, "實際策略要維持 5FC0 的值"
    assert T._safety["strategy"] == 0x10, "變化通知仍照記(除錯用)"


def test_intersections_payload_uses_the_confirmed_strategy(T):
    """演算法讀的是這個欄位 —— 用暫態會讓它誤判『沒有時相控制』而擋下下發。"""
    T._safety_watch(_rec("5FC0", 0x14))
    T._safety_watch(_rec("5F00", 0x01, ts=1001.0))
    mode = T._control_mode(T._safety["strategy_confirmed"] if
                           T._safety.get("strategy_confirmed") is not None
                           else T._safety["strategy"])
    assert mode.get("code") == "external_dynamic"


def test_falls_back_to_notice_before_any_query_reply(T):
    """剛啟動還沒查過 5FC0 時,退回用 5F00,不要變成 None。"""
    T._safety_watch(_rec("5F00", 0x05))
    v = (T._safety.get("strategy_confirmed")
         if T._safety.get("strategy_confirmed") is not None else T._safety["strategy"])
    assert v == 0x05
