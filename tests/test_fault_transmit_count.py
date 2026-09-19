"""故障檢核的「指令傳輸錯誤」要以**命令**計數,不是以檢核輪次計數(2026-09-20)。

舊寫法每 2 秒把最後一則命令重判一次:一則被拒 → 60 秒內累積約 30 次 → 降階。
35 次傳輸故障降階中 30 次是這樣來的;降階後不再送命令,計數永遠不歸零,只能重啟。
"""
import os
import time

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _mod(monkeypatch):
    from api.routes import signal_shadow as S
    from api.routes import signal_tc3 as T
    monkeypatch.setattr(S, "_fault", {"pending": {}, "active": {}, "clear_since": None, "send_fails": 0,
                                      "logic_fails": 0, "nack_fails": 0, "judged_ts": None, "events": []})
    monkeypatch.setattr(T, "enter_degraded", lambda *a, **k: None)
    monkeypatch.setattr(T, "exit_degraded", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(S, "_degrade_persist", lambda *a, **k: None)
    monkeypatch.setattr(S, "add_log", lambda *a, **k: None)
    monkeypatch.setattr(S, "push_alert", lambda *a, **k: None, raising=False)
    return S


M = {"queue_m": 5.0, "flow_vpm": 3.0}


def test_one_rejected_command_rechecked_many_ticks_counts_once(monkeypatch):
    S = _mod(monkeypatch)
    monkeypatch.setattr(S, "_ack_of_last_send", lambda since_ts=None: False)
    monkeypatch.setitem(S._act, "last_ts", time.time() - 70)
    for _ in range(32):                          # 64 秒的檢核輪次,同一則命令
        S._fault_check({}, M, M)
    assert S._fault["nack_fails"] == 1
    assert "transmit" not in S._fault["pending"] and "transmit" not in S._fault["active"]


def test_three_distinct_rejected_commands_raise_fault(monkeypatch):
    S = _mod(monkeypatch)
    monkeypatch.setattr(S, "_ack_of_last_send", lambda since_ts=None: False)
    t = time.time() - 100
    for k in range(3):
        monkeypatch.setitem(S._act, "last_ts", t + k * 10)
        S._fault_check({}, M, M)
    assert S._fault["nack_fails"] == 3 and "transmit" in S._fault["pending"]


def test_accepted_command_resets_and_pending_ack_waits(monkeypatch):
    S = _mod(monkeypatch)
    res = iter([False, False, None, True])
    monkeypatch.setattr(S, "_ack_of_last_send", lambda since_ts=None: next(res))
    t = time.time() - 100
    for k, ts in enumerate((t, t + 10, t + 20, t + 20)):   # 第三則還在等回覆,下一輪才判
        monkeypatch.setitem(S._act, "last_ts", ts)
        S._fault_check({}, M, M)
    assert S._fault["nack_fails"] == 0


def test_transmit_degrade_resets_counter_so_it_can_recover(monkeypatch):
    S = _mod(monkeypatch)
    monkeypatch.setattr(S, "FAULT_HOLD_SEC", 0.0)
    monkeypatch.setattr(S, "_ack_of_last_send", lambda since_ts=None: False)
    t = time.time() - 100
    for k in range(3):
        monkeypatch.setitem(S._act, "last_ts", t + k * 10)
        S._fault_check({}, M, M)
    assert "transmit" in S._fault["active"]          # 降階
    assert S._fault["nack_fails"] == 0                 # 已歸零
    S._fault_check({}, M, M)                           # 沒有新命令 → 故障不再掛起,可以進入復歸計時
    assert "transmit" not in S._fault["pending"]
