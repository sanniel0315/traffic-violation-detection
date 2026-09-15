"""控制器重送的舊 5F03 不可覆蓋目前燈態(2026-09-16 戰情「上下匝道紅綠燈相反」)。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rec(seq, ph, step, ts, rem=5):
    raw = "AA BB %02X FF FF 00 13 5F 03 00 5F 06 %02X %02X 00 %02X" % (seq, ph, step, rem)
    return {"seq": seq, "raw": raw, "ts": ts, "phase": {"sub_phase_id": ph, "step_id": step}}


def test_same_frame_resent_is_repeat():
    from api.routes.signal_tc3 import frame_is_stale_repeat
    st = {}
    assert not frame_is_stale_repeat(st, _rec(230, 1, 2, 100.0))
    for t in (102.1, 104.2, 106.2, 108.3):          # 現場:每 2 秒重送一次
        assert frame_is_stale_repeat(st, _rec(230, 1, 2, t))


def test_late_old_frame_after_newer_is_ignored():
    """17:23 實例:166(分相1 綠)→ 167(分相1 黃)→ 又收到 166 的重送。"""
    from api.routes.signal_tc3 import frame_is_stale_repeat
    st = {}
    assert not frame_is_stale_repeat(st, _rec(166, 1, 1, 34.0))
    assert not frame_is_stale_repeat(st, _rec(167, 1, 4, 45.0))
    assert frame_is_stale_repeat(st, _rec(166, 1, 1, 47.0))
    # 首次傳送遺失、只收到晚到的重送:序號比最新舊 → 也不採用
    assert frame_is_stale_repeat(st, _rec(165, 2, 5, 48.0))


def test_new_frames_and_seq_wrap_pass():
    from api.routes.signal_tc3 import frame_is_stale_repeat
    st = {}
    assert not frame_is_stale_repeat(st, _rec(254, 1, 1, 1.0))
    assert not frame_is_stale_repeat(st, _rec(255, 1, 2, 5.0))
    assert not frame_is_stale_repeat(st, _rec(0, 1, 4, 10.0))      # 序號繞回 0
    assert not frame_is_stale_repeat(st, _rec(1, 1, 5, 13.0))


def test_same_content_long_after_is_new():
    """序號一圈 256 則,事件驅動約 40 分鐘才繞回 —— 30 秒外同內容視為新框。"""
    from api.routes.signal_tc3 import frame_is_stale_repeat
    st = {}
    assert not frame_is_stale_repeat(st, _rec(10, 1, 1, 1.0))
    assert not frame_is_stale_repeat(st, _rec(10, 1, 1, 3000.0))
