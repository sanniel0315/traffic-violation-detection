"""延長綠燈 P1 限時試行:時段開關與自動停止(2026-09-18 使用者核准 16:30-19:00)。

🛑 試行在傍晚,不能假設有人盯著 —— 停止條件必須由程式自己判,
   任一成立就退回影子模式(照算照記,不再下發),重啟才解除。
"""
import os
import time
from datetime import datetime, timedelta

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _mod(monkeypatch, since=None, until=None, mode="1"):
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "EXTEND_GREEN", mode not in ("0", ""))
    monkeypatch.setattr(S, "EXTEND_SHADOW", mode == "shadow")
    monkeypatch.setattr(S, "EXTEND_SINCE", since or "")
    monkeypatch.setattr(S, "EXTEND_UNTIL", until or "")
    monkeypatch.setattr(S, "_ext_trip", {"tripped": False, "at": None, "why": "", "naks": 0,
                                         "sends": 0, "mismatch": 0, "pending": None})
    monkeypatch.setattr(S, "add_log", lambda *a, **k: None)
    return S


def _iso(dt):
    return dt.isoformat(timespec="seconds")


def test_window_before_during_after(monkeypatch):
    now = datetime.now()
    S = _mod(monkeypatch, since=_iso(now + timedelta(hours=1)),
             until=_iso(now + timedelta(hours=2)))
    on, why = S.extend_live_now(time.time())
    assert on is False and "尚未開始" in why

    S = _mod(monkeypatch, since=_iso(now - timedelta(hours=1)),
             until=_iso(now + timedelta(hours=1)))
    assert S.extend_live_now(time.time())[0] is True

    S = _mod(monkeypatch, since=_iso(now - timedelta(hours=2)),
             until=_iso(now - timedelta(hours=1)))
    on, why = S.extend_live_now(time.time())
    assert on is False and "已結束" in why


def test_shadow_mode_never_live(monkeypatch):
    S = _mod(monkeypatch, mode="shadow")
    assert S.extend_live_now(time.time())[0] is False


def test_bad_time_format_is_not_live(monkeypatch):
    """格式錯要當成不下發,不能當成「沒設時段=永遠開」。"""
    S = _mod(monkeypatch, since="16:30")
    on, why = S.extend_live_now(time.time())
    assert on is False and "格式錯" in why


def test_trip_on_storage(monkeypatch):
    """任一側排隊 ≥ 儲車 80% 就停(下匝道 480 m)。"""
    S = _mod(monkeypatch)
    from detection.signal_timing_lookup import phase_of_role
    off = phase_of_role("off_ramp")
    q = {1: 10.0, 2: 10.0}
    q[off] = 490.0
    S._ext_watch(off, {"step_id": 1}, q)
    assert S._ext_trip["tripped"] is True
    assert "排隊" in S._ext_trip["why"]
    assert S.extend_live_now(time.time())[0] is False, "停了之後要退回影子"


def test_trip_when_controller_ignores_T(monkeypatch):
    """送了 T=36,已亮 20 → 剩餘應為 16;控制器回報剩 5 → 不照命令走,兩次就停。"""
    S = _mod(monkeypatch)
    for _ in range(2):
        S._ext_trip["pending"] = {"phase": 1, "T": 36, "ts": time.time() - 3}
        S._ext_watch(1, {"step_id": 1, "step_remain_sec": 5, "phase_elapsed_sec": 20},
                     {1: 10.0, 2: 10.0})
    assert S._ext_trip["tripped"] is True
    assert "沒有照" in S._ext_trip["why"]


def test_no_trip_when_controller_honours_T(monkeypatch):
    S = _mod(monkeypatch)
    for _ in range(3):
        S._ext_trip["pending"] = {"phase": 1, "T": 36, "ts": time.time() - 3}
        S._ext_watch(1, {"step_id": 1, "step_remain_sec": 16, "phase_elapsed_sec": 20},
                     {1: 10.0, 2: 10.0})
    assert S._ext_trip["tripped"] is False


def test_trip_on_repeated_nak(monkeypatch):
    S = _mod(monkeypatch)
    monkeypatch.setattr(S, "_ack_of_last_send", lambda since_ts=None: False)
    for _ in range(3):
        S._ext_trip["ack_check"] = time.time() - 5
        S._ext_watch(1, {"step_id": 1}, {1: 10.0, 2: 10.0})
    assert S._ext_trip["tripped"] is True
    assert "拒絕" in S._ext_trip["why"]


def test_watch_does_nothing_outside_window(monkeypatch):
    """時段外不下發,也就沒有東西要停 —— 不可以因為排隊高就把「還沒開始」標成停止。"""
    now = datetime.now()
    S = _mod(monkeypatch, since=_iso(now + timedelta(hours=1)))
    S._ext_watch(1, {"step_id": 1}, {1: 590.0, 2: 200.0})
    assert S._ext_trip["tripped"] is False
