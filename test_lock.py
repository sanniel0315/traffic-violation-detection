"""電子鎖邏輯自動化測試 (不需真鎖)。

用 IOService.__new__ 繞過 __init__(不連串口/daemon) + mock IOModule,
純驗證:協議 action 定義、箱門未關告警(門磁)、上鎖/解鎖事件(0x0020)、
斷電告警、多鎖互不干擾、狀態 label。
跑法: cd 專案根 → python3 test_lock.py
"""
import time
import sys

from services.io_service import (
    IOService,
    _LockState,
    _LOCK_ACTION_NAMES,
    _LOCK_OFFLINE_ALARM_SEC,
    _DOOR_OPEN_ALARM_SEC,
)

results: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    results.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def new_svc() -> IOService:
    """mock 掉 _fire_lock_event,只收 (event_type, label) 進 _fired。
    多鎖版第一個參數是 _LockState,測試要跟著吃掉。"""
    s = IOService.__new__(IOService)
    s._fired = []
    s._fire_lock_event = lambda lk, et, label, *a, **k: s._fired.append((et, label))
    return s


def new_lock(addr: int = 2) -> _LockState:
    return _LockState(addr)


print("=== 1. 協議 0x0023 action 定義 ===")
check("action 4 = 鑰匙轉動", _LOCK_ACTION_NAMES.get(4) == "鑰匙轉動")
check("移除協議沒有的 5(手柄開)", 5 not in _LOCK_ACTION_NAMES)
check("_is_real_action 1..4 為真", all(IOService._is_real_action(a) for a in (1, 2, 3, 4)))
check("_is_real_action 0/5/250/None 為假",
      not any(IOService._is_real_action(a) for a in (0, 5, 250, None)))

print("=== 2. 箱門未關告警(門磁 0x0021) ===")
s = new_svc()
lk = new_lock()
lk.status = {"door": {"closed": False}, "handle": {"in_place": True}}
lk.prev_states = {"door": False, "handle": True}
lk.door_open_since = time.time() - (_DOOR_OPEN_ALARM_SEC + 2)
s._detect_state_events(lk)
check("門開超時觸發箱門未關告警", any(et == "alarm" and "箱門未關" in l for et, l in s._fired))
check("觸發後標記 alarmed(只發一次)", lk.door_alarmed is True)
s2 = new_svc()
lk2 = new_lock()
lk2.status = {"door": {"closed": True}, "handle": {"in_place": True}}
lk2.prev_states = {"door": False, "handle": True}
lk2.door_open_since = time.time()
lk2.door_alarmed = True
s2._detect_state_events(lk2)
check("門關後重置 door_alarmed", lk2.door_alarmed is False)

print("=== 3. 鎖定狀態(0x0020) 上鎖/解鎖事件 ===")
s = new_svc()
lk = new_lock()
lk.status = {"door": {"closed": True}, "handle": {"in_place": False}}   # 解鎖
lk.prev_states = {"door": True, "handle": True}                        # 前次上鎖
s._detect_state_events(lk)
check("轉鑰匙解鎖 → lock事件『解鎖』", any(et == "lock" and l == "解鎖" for et, l in s._fired))
s = new_svc()
lk = new_lock()
lk.status = {"door": {"closed": True}, "handle": {"in_place": True}}    # 上鎖
lk.prev_states = {"door": True, "handle": False}                       # 前次解鎖
s._detect_state_events(lk)
check("轉鑰匙上鎖 → lock事件『上鎖』", any(et == "lock" and l == "上鎖" for et, l in s._fired))

print("=== 4. 斷電/失聯告警 + 恢復 ===")
s = new_svc()
lk = new_lock()
lk.connected = False
lk.offline_since = time.time() - (_LOCK_OFFLINE_ALARM_SEC + 2)
s._detect_offline(lk)
check("斷電超時觸發 alarm", any(et == "alarm" and "斷電" in l for et, l in s._fired))
check("斷電觸發後標記 alarmed", lk.offline_alarmed is True)
s._fired.clear()
lk.connected = True
s._detect_offline(lk)
check("恢復連線記一筆", any("恢復連線" in l for et, l in s._fired))
check("恢復後重置 since/alarmed", lk.offline_since is None and lk.offline_alarmed is False)

print("=== 4b. 多鎖:一顆斷線不影響另一顆 ===")
s = new_svc()
a, b = new_lock(2), new_lock(3)
a.connected = False
a.offline_since = time.time() - (_LOCK_OFFLINE_ALARM_SEC + 2)
b.connected = True
b.status = {"door": {"closed": True}, "handle": {"in_place": True}}
b.prev_states = {"door": True, "handle": True}
s._detect_offline(a)
s._detect_offline(b)
s._detect_state_events(b)
check("#2 失聯有告警", any(et == "alarm" and "斷電" in l for et, l in s._fired))
check("#3 沒被連累(仍 connected)", b.connected is True)
check("#3 沒有多餘事件", not any(et == "alarm" for et, l in s._fired[1:]))
check("兩顆告警旗標各自獨立", a.offline_alarmed is True and b.offline_alarmed is False)

print("=== 5. 狀態 label ===")
s = IOService.__new__(IOService)


class _Mod0:
    def read_holding(self, addr, reg, cnt):
        return [0]


s._mod = _Mod0()
s._lock_dev = None
lk = new_lock()
s._poll_lock_states(lk, action=0)
check("raw0 鎖定=已上鎖", lk.status["handle"]["label"] == "已上鎖")
check("raw0 門磁斷開(NO接點=門開)", lk.status["door"]["label"] == "門磁斷開(門開)")
check("action0 無", lk.status["action"]["label"] == "無")


class _Mod1:
    def read_holding(self, addr, reg, cnt):
        return [1]


s._mod = _Mod1()
lk = new_lock()
s._poll_lock_states(lk, action=1)
check("raw1 鎖定=已解鎖", lk.status["handle"]["label"] == "已解鎖")
check("raw1 門磁閉合(NO接點=門關)", lk.status["door"]["label"] == "門磁閉合(門關)")
check("action1 刷卡", lk.status["action"]["label"] == "刷卡")

print("=== 6. 失效卡/未授權刷卡(0xD000)解析 ===")
sd = IOService.__new__(IOService)


class _ModD:
    def read_holding(self, addr, reg, cnt):
        # 0xD000 失敗記錄: [剩餘方式, 卡號hi, 卡號lo, 年月, 日時, 分秒]
        return [0x0001, 0xBF7B, 0x7CB2, 0x1404, 0x0B0D, 0x3938]


sd._mod = _ModD()
sd._lock_dev = None
fr = sd._read_lock_fail_record(3)
check("0xD000 解析出卡號 BF7B7CB2", bool(fr) and fr["card"] == "BF7B7CB2")
check("0xD000 方式=1(卡片)", bool(fr) and fr["way"] == 1)

print()
passed = sum(1 for _, c in results if c)
total = len(results)
print(f"==== {passed}/{total} PASS ====")
sys.exit(0 if passed == total else 1)
