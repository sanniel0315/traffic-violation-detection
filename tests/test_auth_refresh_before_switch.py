"""換相(5F1C)送出前,授權太舊就先補一次 5F10。

🛑 2026-09-17 現場實測(12 小時)的依據:
   5F1C 被拒(0F81 ErrorCode=02)與「距上次 5F10 多久」完全對得上 ——
     0~10 秒 送出 36 則,被拒 0%
     15~20 秒 39 則,被拒 41%
     20~25 秒 16 則,被拒 50%
   同一份資料裡時相控制保持率也隨這個 age 衰減(≤12 秒 99%、20~24 秒 86%)。
   續約迴圈即使設 10 秒,實際仍會被序列排隊拉出 28~30 秒的空檔,
   所以要在**送出前**補,不能只靠背景續約。
"""
import os
import asyncio

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


class _Sock:
    def __init__(self):
        self.sent = []

    def sendall(self, b):
        self.sent.append(b)


def _send(T, code="5F1C"):
    out = T._finish_prepare(b"\xaa\xbb\x01", code, int(code[2:], 16), int(code[:2], 16), 1, 1, "algorithm")
    return asyncio.get_event_loop().run_until_complete(
        T.control_send({"token": out["token"]}, _user="tester"))


def _setup(monkeypatch, age_sec):
    from api.routes import signal_tc3 as T

    calls = []
    monkeypatch.setattr(T, "_do_reassert", lambda kind="": calls.append(kind))
    monkeypatch.setattr(T, "_control_guard", lambda *a, **k: "")
    monkeypatch.setitem(T._sock_ref, "sock", _Sock())
    monkeypatch.setitem(T._dyn, "enabled", True)
    monkeypatch.setitem(T._dyn, "level", "L0")
    monkeypatch.setitem(T._safety, "strategy", 0x14)
    import time as _t
    monkeypatch.setitem(T._auth, "last", _t.time() - age_sec)
    monkeypatch.setattr(T, "_persist_control", lambda rec: None)
    monkeypatch.setattr(T, "_enqueue_frame", lambda rec: None)
    monkeypatch.setattr(T, "add_log", lambda *a, **k: None)
    return T, calls


def test_refreshes_authority_when_stale(monkeypatch):
    T, calls = _setup(monkeypatch, age_sec=30)
    _send(T)
    assert calls, "授權已 30 秒沒續,換相前應該先補 5F10"


def test_does_not_refresh_when_authority_is_fresh(monkeypatch):
    T, calls = _setup(monkeypatch, age_sec=2)
    _send(T)
    assert calls == [], "剛續過就不必再補 —— 補了只是多佔序列線"


def test_only_switch_commands_trigger_refresh(monkeypatch):
    """查詢或其他設定類不補 —— 補授權是為了讓換相被接受,不是每則命令都要。"""
    T, calls = _setup(monkeypatch, age_sec=30)
    _send(T, code="5F40")
    assert calls == []


def test_refresh_disabled_by_zero(monkeypatch):
    from api.routes import signal_tc3 as T
    T2, calls = _setup(monkeypatch, age_sec=30)
    monkeypatch.setattr(T, "AUTH_REFRESH_BEFORE_SEC", 0.0)
    _send(T2)
    assert calls == [], "設 0 要能關掉這個行為"


def test_seq_assigned_at_send_after_refresh(monkeypatch):
    """🛑 2026-09-18 16:30:19 現場:補授權 5F10 與延長 5F1C 同為序號 0x28,
    控制器只回 5F10,5F1C 被當成重複框丟掉。序號要在送出當下重新分配。"""
    T, calls = _setup(monkeypatch, age_sec=30)
    sock = T._sock_ref["sock"]

    def fake_reassert(kind=""):
        calls.append(kind)
        T._seq_next["n"] = (int(T._seq_next.get("n", 0)) + 1) & 0xFF   # 補授權用掉一個序號
    monkeypatch.setattr(T, "_do_reassert", fake_reassert)
    monkeypatch.setattr(T, "AUTH_REFRESH_GAP_SEC", 0.0)
    T._seq_next["n"] = 0x27
    frame = T.build_frame(0xFFFF, 0x28, bytes([0x5F, 0x1C, 0x01, 0x01, 0x19]))
    out = T._finish_prepare(frame, "5F1C", 0x1C, 0x5F, 0xFFFF, 0x28, "algorithm-extend")
    import asyncio
    asyncio.get_event_loop().run_until_complete(T.control_send({"token": out["token"]}, _user="t"))
    assert calls, "應該先補授權"
    sent = sock.sent[-1]
    assert sent[2] == 0x29, "5F1C 不可以沿用 prepare 時的 0x28(已被補授權用掉)"
    d = T.decode_frame(sent)
    assert d and d["cks_ok"] and d["code"] == "5F1C", "重新組框後檢查碼要正確"
