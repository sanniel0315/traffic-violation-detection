"""電子鎖斷線判定去抖動測試。

守住 2026-09-07 查出的問題:刷卡當下鎖在驅動馬達,常來不及回應 _poll_lock_states
三次連讀中的一次;舊版單次失敗就 connected=False,接著 _lock_loop 罰退避 5 秒
(_LOCK_RETRY_SEC),剛好跨過 _detect_offline 的 5 秒門檻 →「每刷一次卡就離線一次」。
現場 45 次「電子鎖恢復連線」中,38% 發生在同一顆鎖刷卡後 60 秒內。

重點:去抖動不能把「真的沒接的鎖」也一起放過 —— 那顆要照樣被判離線,
否則退避機制失效,一顆沒接的鎖會每 tick 吃掉 serial timeout 拖垮另一顆。
"""
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import io_service as S


def _lk():
    lk = S._LockState(2, "測試鎖")
    lk.connected = True
    lk.fail_streak = 0
    return lk


def _svc():
    """只取 _poll_lock_states,不啟動真的序列埠。"""
    svc = S.IOService.__new__(S.IOService)
    return svc


def test_單次讀取失敗不算斷線():
    """刷卡瞬間的無回應不可以被當成離線。"""
    svc, lk = _svc(), _lk()
    with mock.patch.object(svc, "_lock_read", side_effect=OSError("timeout")):
        svc._poll_lock_states(lk)
    assert lk.fail_streak == 1
    assert lk.connected is True, "單次失敗就標離線 → 會被 5 秒退避放大成完整離線事件"


def test_連續失敗到門檻才判離線():
    """真的沒接/斷電的鎖仍要被判離線,退避機制才有效。"""
    svc, lk = _svc(), _lk()
    with mock.patch.object(svc, "_lock_read", side_effect=OSError("timeout")):
        for _ in range(S._LOCK_FAIL_STREAK):
            svc._poll_lock_states(lk)
    assert lk.fail_streak == S._LOCK_FAIL_STREAK
    assert lk.connected is False, "連續失敗達門檻仍應判離線"


def test_中途讀成功會清空計數():
    """偶發抖動不可以累積 —— 兩次相隔很久的單次失敗不該湊成一次離線。"""
    svc, lk = _svc(), _lk()
    with mock.patch.object(svc, "_lock_read", side_effect=OSError("timeout")):
        svc._poll_lock_states(lk)
    assert lk.fail_streak == 1
    with mock.patch.object(svc, "_lock_read", return_value=[0]):
        svc._poll_lock_states(lk)
    assert lk.fail_streak == 0, "讀成功要清空連續失敗計數"
    assert lk.connected is True


def test_讀成功後狀態欄位齊全():
    """去抖動不可以改變正常路徑的行為。"""
    svc, lk = _svc(), _lk()
    with mock.patch.object(svc, "_lock_read", return_value=[0]):
        svc._poll_lock_states(lk)
    assert set(lk.status) == {"handle", "door", "key", "action"}
    assert lk.err == ""


def test_門檻可用環境變數調整():
    assert S._LOCK_FAIL_STREAK >= 1
    # 1 = 舊行為(單次即斷),預設應該大於 1 才有去抖動效果
    assert S._LOCK_FAIL_STREAK > 1, "預設門檻要 >1,否則等於沒去抖動"


def test_去抖動期間不會誤觸退避():
    """fail_streak 未達門檻時 connected 維持 True → _lock_loop 不會設 next_probe。"""
    svc, lk = _svc(), _lk()
    lk.next_probe = 0.0
    with mock.patch.object(svc, "_lock_read", side_effect=OSError("timeout")):
        svc._poll_lock_states(lk)
    assert lk.connected is True
    assert lk.next_probe == 0.0, "還沒判離線就不該進入 5 秒退避"
