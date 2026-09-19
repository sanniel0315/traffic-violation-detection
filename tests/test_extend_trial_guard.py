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
    """送了 T=36,已亮 20 → 剩餘應為 16;控制器回報剩 5 → 不照命令走,兩次就停(要明確開啟)。"""
    S = _mod(monkeypatch)
    monkeypatch.setattr(S, "EXT_TRIP_MISMATCH_MAX", 2)
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
    monkeypatch.setattr(S, "_ext_ack_result", lambda ts, seq: False)          # 明確 0F81
    for _ in range(3):
        S._ext_trip["ack_check"] = {"ts": time.time() - 5, "seq": 1}
        S._ext_watch(1, {"step_id": 1}, {1: 10.0, 2: 10.0})
    assert S._ext_trip["tripped"] is True
    assert "拒絕" in S._ext_trip["why"]


def test_late_or_missing_ack_is_not_a_reject(monkeypatch):
    """09-19 18:17 誤停:延長命令的 ACK 要 20~51 秒才回。還在等 / 等太久沒回,都不可算被拒。"""
    S = _mod(monkeypatch)
    for res in (None, "unknown", True, "unknown", True):
        monkeypatch.setattr(S, "_ext_ack_result", lambda ts, seq, r=res: r)
        S._ext_trip["ack_check"] = {"ts": time.time() - 5, "seq": 1}
        S._ext_watch(1, {"step_id": 1}, {1: 10.0, 2: 10.0})
    assert S._ext_trip["tripped"] is False and S._ext_trip["naks"] == 0
    assert S._ext_trip.get("acks") == 2 and S._ext_trip.get("ack_unknown") == 2


def test_ext_ack_matches_own_seq_even_when_late(monkeypatch, tmp_path):
    """用序號配對:51 秒後才回的 0F80 算接受;別則命令(不同序號)的回覆不算。"""
    import sqlite3
    S = _mod(monkeypatch)
    db = tmp_path / "v.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE signal_frames(ts REAL, src TEXT, code TEXT, seq INTEGER, raw TEXT)")
    t0 = time.time() - 100
    c.execute("INSERT INTO signal_frames VALUES(?,?,?,?,?)", (t0 + 0.1, "controller", "0F80", 50, "AA BB 32 FF FF 00 0E 0F 80 5F 1C AA CC 00"))
    c.execute("INSERT INTO signal_frames VALUES(?,?,?,?,?)", (t0 + 51.2, "controller", "0F80", 49, "AA BB 31 FF FF 00 0E 0F 80 5F 1C AA CC 84"))
    c.commit(); c.close()
    monkeypatch.setattr(S, "_VIOL_DB", str(db))
    assert S._ext_ack_result(t0, 49) is True
    assert S._ext_ack_result(t0, 77) == "unknown"                         # 90 秒內沒有自己的回覆
    assert S._ext_ack_result(time.time() - 3, 77) is None                  # 還在等


def test_watch_does_nothing_outside_window(monkeypatch):
    """時段外不下發,也就沒有東西要停 —— 不可以因為排隊高就把「還沒開始」標成停止。"""
    now = datetime.now()
    S = _mod(monkeypatch, since=_iso(now + timedelta(hours=1)))
    S._ext_watch(1, {"step_id": 1}, {1: 590.0, 2: 200.0})
    assert S._ext_trip["tripped"] is False


def test_default_only_counts_not_applied(monkeypatch):
    """2026-09-19:沒照新總長走 = 延長沒生效、照原時制結束,沒有危險;預設只記錄,不停止。"""
    S = _mod(monkeypatch)
    for rem in (5, 16, 5):
        S._ext_trip["pending"] = {"phase": 1, "T": 36, "ts": time.time() - 3}
        S._ext_watch(1, {"step_id": 1, "step_remain_sec": rem, "phase_elapsed_sec": 20}, {1: 10.0, 2: 10.0})
    assert S._ext_trip["tripped"] is False
    assert S._ext_trip["not_applied"] == 2 and S._ext_trip["applied"] == 1
