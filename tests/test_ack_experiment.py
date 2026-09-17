"""限時實驗:對控制器的主動回報回 0F80,看重複會不會停(2026-09-17 現場授權)。

🛑 預設關閉、必須設到期時間、到期自動停 —— 這是改變我方對控制器行為的開關,
   不可以因為忘了關就一直在送。
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture
def T(monkeypatch):
    from api.routes import signal_tc3 as T
    sent = []
    monkeypatch.setattr(T, "_controller_send", lambda b: sent.append(b) or True)
    monkeypatch.setattr(T, "_enqueue_frame", lambda rec: None)
    monkeypatch.setattr(T, "_ack_stat", {"sent": 0, "last": "", "last_error": ""})
    T._sent_frames = sent
    return T


def _rec(code="5F03", seq=7):
    return {"code": code, "seq": seq, "addr": 0xFFFF, "cks_ok": True}


def test_off_by_default(T, monkeypatch):
    monkeypatch.setattr(T, "ACK_CONTROLLER", False)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "")
    T._ack_controller_frame(_rec())
    assert T._sent_frames == [] and T._ack_stat["sent"] == 0


def test_requires_an_expiry_time(T, monkeypatch):
    """只開開關、沒設到期時間 → 不送(避免忘了關)。"""
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "")
    T._ack_controller_frame(_rec())
    assert T._sent_frames == []


def test_stops_after_expiry(T, monkeypatch):
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL",
                        (datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds"))
    T._ack_controller_frame(_rec())
    assert T._sent_frames == []


def test_acks_autonomous_reports_within_window(T, monkeypatch):
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL",
                        (datetime.now() + timedelta(minutes=30)).isoformat(timespec="seconds"))
    T._ack_controller_frame(_rec("5F03", 9))
    assert len(T._sent_frames) == 1
    f = T._sent_frames[0]
    assert f[:2] == b"\xaa\xbb" and f[2] == 9        # 序號沿用對方的
    assert f[7:11] == bytes((0x0F, 0x80, 0x5F, 0x03))  # 0F80 + 被確認的碼
    assert T._ack_stat["sent"] == 1


@pytest.mark.parametrize("code", ["0F80", "0F81", "5FC0", "0FC1", "5FC8"])
def test_never_acks_replies_or_acks(T, monkeypatch, code):
    """回覆類與 ACK/NAK 本身不回,避免互相回應的迴圈。"""
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL",
                        (datetime.now() + timedelta(minutes=30)).isoformat(timespec="seconds"))
    T._ack_controller_frame(_rec(code, 3))
    assert T._sent_frames == []


def test_never_acks_a_bad_checksum_frame(T, monkeypatch):
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL",
                        (datetime.now() + timedelta(minutes=30)).isoformat(timespec="seconds"))
    r = _rec(); r["cks_ok"] = False
    T._ack_controller_frame(r)
    assert T._sent_frames == []
