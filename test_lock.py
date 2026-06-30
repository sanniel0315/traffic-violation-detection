"""電子鎖邏輯自動化測試 (不需真鎖)。

用 IOService.__new__ 繞過 __init__(不連串口/daemon) + mock IOModule,
純驗證:協議 action 定義、手柄/門/斷電警報計時觸發、狀態 label 協議用詞。
跑法: cd 專案根 → python3 test_lock.py
"""
import time
import sys

from services.io_service import (
    IOService,
    _LOCK_ACTION_NAMES,
    _HANDLE_OFF_ALARM_SEC,
    _LOCK_OFFLINE_ALARM_SEC,
    _DOOR_OPEN_ALARM_SEC,
)

results: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    results.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def new_svc() -> IOService:
    """造一個只夠測方法的 IOService,_fire_lock_event 改成收集器。"""
    s = IOService.__new__(IOService)
    s._fired = []
    s._fire_lock_event = lambda et, label, *a, **k: s._fired.append((et, label))
    return s


print("=== 1. 協議 0x0023 action 定義 ===")
check("action 4 = 鑰匙轉動", _LOCK_ACTION_NAMES.get(4) == "鑰匙轉動")
check("移除協議沒有的 5(手柄開)", 5 not in _LOCK_ACTION_NAMES)
check("_is_real_action 1..4 為真", all(IOService._is_real_action(a) for a in (1, 2, 3, 4)))
check("_is_real_action 0/5/250/None 為假",
      not any(IOService._is_real_action(a) for a in (0, 5, 250, None)))

print("=== 2. 手柄長時間不在位警報 ===")
s = new_svc()
s._lock_status = {"handle": {"in_place": False}, "door": {"closed": True}, "key": {"in_place": True}}
s._lock_prev_states = {"door": True, "handle": False, "key": True}
s._door_open_since = None
s._door_alarmed = False
s._handle_off_since = time.time() - (_HANDLE_OFF_ALARM_SEC + 2)
s._handle_alarmed = False
s._detect_state_events()
check("手柄超時觸發 alarm", any(et == "alarm" and "手柄長時間不在位" in l for et, l in s._fired))
check("觸發後標記 alarmed(只發一次)", s._handle_alarmed is True)
s._fired.clear()
s._lock_status["handle"]["in_place"] = True
s._lock_prev_states = {"door": True, "handle": False, "key": True}
s._detect_state_events()
check("手柄復位後重置 alarmed", s._handle_alarmed is False)

print("=== 3. 門長時間未關警報(回歸,確保沒被改壞) ===")
s = new_svc()
s._lock_status = {"handle": {"in_place": True}, "door": {"closed": False}, "key": {"in_place": True}}
s._lock_prev_states = {"door": False, "handle": True, "key": True}
s._handle_off_since = None
s._handle_alarmed = False
s._door_open_since = time.time() - (_DOOR_OPEN_ALARM_SEC + 2)
s._door_alarmed = False
s._detect_state_events()
check("門開超時觸發 alarm", any(et == "alarm" and "門長時間未關" in l for et, l in s._fired))

print("=== 4. 斷電/失聯告警 + 恢復 ===")
s = new_svc()
s._lock_connected = False
s._offline_since = time.time() - (_LOCK_OFFLINE_ALARM_SEC + 2)
s._offline_alarmed = False
s._detect_offline()
check("斷電超時觸發 alarm", any(et == "alarm" and "斷電" in l for et, l in s._fired))
check("斷電觸發後標記 alarmed", s._offline_alarmed is True)
s._fired.clear()
s._lock_connected = True
s._detect_offline()
check("恢復連線記一筆", any("恢復連線" in l for et, l in s._fired))
check("恢復後重置 since/alarmed", s._offline_since is None and s._offline_alarmed is False)

print("=== 5. 狀態 label 協議用詞 ===")
s = IOService.__new__(IOService)


class _Mod0:
    def read_holding(self, addr, reg, cnt):
        return [0]


s._mod = _Mod0()
s._lock_status = {}
s._lock_prev_action = 0
s._poll_lock_states(action=0)
check("raw0 手柄在位", s._lock_status["handle"]["label"] == "手柄在位")
check("raw0 門磁閉合", s._lock_status["door"]["label"] == "門磁閉合(門關)")
check("raw0 鑰匙在位", s._lock_status["key"]["label"] == "鑰匙在位")
check("action0 無", s._lock_status["action"]["label"] == "無")


class _Mod1:
    def read_holding(self, addr, reg, cnt):
        return [1]


s._mod = _Mod1()
s._poll_lock_states(action=1)
check("raw1 手柄不在位", s._lock_status["handle"]["label"] == "手柄不在位")
check("raw1 門磁斷開", s._lock_status["door"]["label"] == "門磁斷開(門開)")
check("action1 刷卡", s._lock_status["action"]["label"] == "刷卡")

print()
passed = sum(1 for _, c in results if c)
total = len(results)
print(f"==== {passed}/{total} PASS ====")
sys.exit(0 if passed == total else 1)
