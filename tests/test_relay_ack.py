"""中間層對控制器回 0F80(2026-09-17 實測證實:不回就會被每 2 秒重送整段)。

我方是中間層:控制器的訊框先到我們這裡再上傳中央,所以收訊端的 ACK 由我方回。
🛑 永遠不回 0F80/0F81 本身(會變成互相回應的迴圈);到期時間是選填,給限時實驗用。
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


def test_no_expiry_means_always_on(T, monkeypatch):
    """沒設到期時間 = 持續生效(正式功能,不是實驗)。"""
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "")
    monkeypatch.setattr(T, "ACK_ONLY", set())
    T._ack_controller_frame(_rec())
    assert len(T._sent_frames) == 1


def test_bad_expiry_format_does_not_send(T, monkeypatch):
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "not-a-time")
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
    monkeypatch.setattr(T, "ACK_ONLY", set())
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL",
                        (datetime.now() + timedelta(minutes=30)).isoformat(timespec="seconds"))
    T._ack_controller_frame(_rec("5F03", 9))
    assert len(T._sent_frames) == 1
    f = T._sent_frames[0]
    assert f[:2] == b"\xaa\xbb" and f[2] == 9        # 序號沿用對方的
    assert f[7:11] == bytes((0x0F, 0x80, 0x5F, 0x03))  # 0F80 + 被確認的碼
    assert T._ack_stat["sent"] == 1


@pytest.mark.parametrize("code", ["0F80", "0F81"])
def test_never_acks_an_ack(T, monkeypatch, code):
    """ACK/NAK 本身不回,否則兩邊會互相回應到天荒地老。"""
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "")
    monkeypatch.setattr(T, "ACK_ONLY", set())
    T._ack_controller_frame(_rec(code, 3))
    assert T._sent_frames == []


@pytest.mark.parametrize("code", ["5FC0", "0FC1", "5FC8", "5F03", "0F04"])
def test_acks_everything_else_including_replies(T, monkeypatch, code):
    """回覆中央查詢的那幾種也要回 —— 實驗中沒回的就繼續被重送 60% 以上。"""
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "")
    monkeypatch.setattr(T, "ACK_ONLY", set())
    T._ack_controller_frame(_rec(code, 5))
    assert len(T._sent_frames) == 1


def test_ack_codes_env_can_narrow_the_scope(T, monkeypatch):
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "")
    monkeypatch.setattr(T, "ACK_ONLY", {"5F03"})
    T._ack_controller_frame(_rec("5F03", 1))
    T._ack_controller_frame(_rec("0FC1", 2))
    assert len(T._sent_frames) == 1


def test_never_acks_a_bad_checksum_frame(T, monkeypatch):
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL",
                        (datetime.now() + timedelta(minutes=30)).isoformat(timespec="seconds"))
    monkeypatch.setattr(T, "ACK_ONLY", set())
    r = _rec(); r["cks_ok"] = False
    T._ack_controller_frame(r)
    assert T._sent_frames == []


def _ack_rec(inner="5F10", seq=7):
    """控制器回的 0F80,內容是被確認的碼。"""
    raw = "AA BB %02X FF FF 00 0E 0F 80 %s %s AA CC 00" % (seq, inner[:2], inner[2:])
    return {"code": "0F80", "seq": seq, "addr": 0xFFFF, "cks_ok": True, "raw": raw}


def test_acks_the_controller_ack_of_our_command(T, monkeypatch):
    """控制器對**我方命令**回的 0F80 也要確認 —— 中央沒送過那道命令,不該由它收尾。"""
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "")
    monkeypatch.setattr(T, "ACK_ONLY", set())
    T._ack_controller_frame(_ack_rec("5F10", 11))
    assert len(T._sent_frames) == 1
    assert T._sent_frames[0][7:11] == bytes((0x0F, 0x80, 0x0F, 0x80))


def test_does_not_ack_an_ack_of_an_ack(T, monkeypatch):
    """只回一層:對方確認的若本身是 ACK 就停,否則會互相確認到天荒地老。"""
    monkeypatch.setattr(T, "ACK_CONTROLLER", True)
    monkeypatch.setattr(T, "ACK_CONTROLLER_UNTIL", "")
    monkeypatch.setattr(T, "ACK_ONLY", set())
    T._ack_controller_frame(_ack_rec("0F80", 12))
    T._ack_controller_frame(_ack_rec("0F81", 13))
    assert T._sent_frames == []
