"""兩道門的聚合狀態。

現場規則:只要任一門開就顯示「開門」,兩門都關上才顯示「關上」。
讀不到的鎖不可以被當成「關上」—— 那等於憑空宣稱一件沒驗證過的事。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.utils.door_state import aggregate_door_state, door_state_of


def _lock(addr, name, closed, connected=True):
    return {"addr": addr, "name": name, "connected": connected,
            "status": {"door": {"closed": closed}} if closed is not None else {}}


def _block(*locks, enabled=True):
    return {"enabled": enabled, "locks": list(locks)}


def test_both_closed_reports_closed():
    r = aggregate_door_state(_block(_lock(2, "後門", True), _lock(3, "前門", True)))
    assert r["state"] == "closed" and r["label"] == "關上"
    assert r["open_count"] == 0 and r["closed_count"] == 2


def test_any_open_reports_open():
    """一道開就是開,不必看另一道。"""
    r = aggregate_door_state(_block(_lock(2, "後門", False), _lock(3, "前門", True)))
    assert r["state"] == "open" and r["label"] == "開門"
    r = aggregate_door_state(_block(_lock(2, "後門", True), _lock(3, "前門", False)))
    assert r["state"] == "open"


def test_both_open_reports_open():
    r = aggregate_door_state(_block(_lock(2, "後門", False), _lock(3, "前門", False)))
    assert r["state"] == "open" and r["open_count"] == 2


def test_offline_lock_is_not_treated_as_closed():
    """「兩個門關上才顯示關上」—— 讀不到就無法確認,不可以報關上。"""
    r = aggregate_door_state(_block(_lock(2, "後門", True),
                                    _lock(3, "前門", None, connected=False)))
    assert r["state"] == "unknown" and r["label"] == "狀態不明"
    assert r["unknown_count"] == 1


def test_offline_lock_does_not_hide_an_open_door():
    """另一道確定開著,就算有鎖離線也要回開門。"""
    r = aggregate_door_state(_block(_lock(2, "後門", False),
                                    _lock(3, "前門", None, connected=False)))
    assert r["state"] == "open"


def test_stale_status_on_disconnected_lock_is_unknown():
    """斷線的鎖 status 可能是舊值,不可以拿過期的當現況。"""
    lk = _lock(3, "前門", True, connected=False)      # 舊值說「關」
    assert door_state_of(lk) == "unknown"


def test_missing_door_field_is_unknown():
    r = aggregate_door_state(_block(_lock(2, "後門", None), _lock(3, "前門", True)))
    assert r["state"] == "unknown"


def test_disabled_or_no_lock_is_unknown():
    """功能沒啟用或一顆鎖都沒有,不能宣稱關上。"""
    assert aggregate_door_state(_block(enabled=False))["state"] == "unknown"
    assert aggregate_door_state(_block(_lock(2, "後門", True), enabled=False))["state"] == "unknown"
    assert aggregate_door_state({"enabled": True, "locks": []})["state"] == "unknown"
    assert aggregate_door_state(None)["state"] == "unknown"


def test_single_lock_still_works():
    """只掛一顆鎖時規則不變。"""
    assert aggregate_door_state(_block(_lock(2, "後門", True)))["state"] == "closed"
    assert aggregate_door_state(_block(_lock(2, "後門", False)))["state"] == "open"


def test_doors_detail_carries_name_and_online():
    r = aggregate_door_state(_block(_lock(2, "後門", False),
                                    _lock(3, "前門", None, connected=False)))
    by_addr = {d["addr"]: d for d in r["doors"]}
    assert by_addr[2]["name"] == "後門" and by_addr[2]["state"] == "open" and by_addr[2]["online"]
    assert by_addr[3]["name"] == "前門" and by_addr[3]["state"] == "unknown"
    assert by_addr[3]["online"] is False


def test_static_api_key_has_door_scope():
    """87 用的是 .env 固定 key,scope 寫在程式碼裡 —— 少了它現場要改設定才能用。"""
    from api.utils.api_key_auth import _STATIC_KEY_OBJ
    assert "door_status" in _STATIC_KEY_OBJ.scopes
