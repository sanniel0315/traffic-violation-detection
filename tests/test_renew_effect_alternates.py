"""續約的 EffectTime 要交替,否則控制器不重新計時(2026-09-17 實測)。

每 20 秒送一模一樣的 5F10,授權仍平均只撐 41 秒就掉回 0x05(定時+殘留手動),
時相控制只佔約一半時間。交替 1↔2 分鐘讓每一則都是新命令。
"""
import os
import sys
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
    monkeypatch.setattr(T, "add_log", lambda *a, **k: None)
    monkeypatch.setattr(T, "_target_addr", lambda: 0xFFFF)
    monkeypatch.setitem(T._dyn, "enabled", True)
    monkeypatch.setitem(T._dyn, "level", "L0")
    monkeypatch.setitem(T._safety, "strategy", 0x14)
    monkeypatch.setitem(T._reassert, "n", 0)
    T._sent = sent
    return T


def _eff_of(frame):
    """5F10 之後第二個位元組 = EffectTime。"""
    return frame[10]


def _strategy_of(frame):
    return frame[9]


def test_effect_time_alternates_between_renewals(T):
    for _ in range(4):
        T._do_reassert(kind="續約")
    effs = [_eff_of(f) for f in T._sent]
    assert len(effs) == 4
    assert effs[0] != effs[1] and effs[1] != effs[2] and effs[2] != effs[3], effs
    assert set(effs) == {T.REASSERT_EFFECT, T.REASSERT_EFFECT + 1}


def test_strategy_value_never_changes(T):
    """只交替有效時間,策略值一律 0x14(含路口手動位元),不可以動。"""
    for _ in range(4):
        T._do_reassert(kind="續約")
    assert {_strategy_of(f) for f in T._sent} == {T.reassert_strategy()}
    assert T.reassert_strategy() & 0x10, "必須含 bit4 時相控制"


def test_window_always_longer_than_renew_interval(T):
    """最短的那個有效時間也要大於續約間隔,中間才不會有空檔。"""
    assert T.REASSERT_EFFECT * 60 > T.AUTH_RENEW_SEC


def test_does_not_send_when_dynamic_control_is_off(T, monkeypatch):
    monkeypatch.setitem(T._dyn, "enabled", False)
    T._do_reassert(kind="續約")
    assert T._sent == []
