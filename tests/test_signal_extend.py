"""延長綠燈(規範 16613 K(C)d):5F1C(分相, 1, T),T = 步階1 總長(09-15 現場驗證)。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


class _D:
    def __init__(self, action="KEEP", by="cost", green=2.0, red=1.0):
        self.action, self.decided_by, self.reason = action, by, "x"
        self.detail = {"green_remain": green, "red_veh": red}


def _live(step=1, rem=5, el=30.0, **kw):
    d = {"control_mode": "external_dynamic", "clearance": False, "stale": False,
         "step_id": step, "step_remain_sec": rem, "phase_elapsed_sec": el}
    d.update(kw)
    return d


@pytest.fixture
def S(monkeypatch):
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "EXTEND_GREEN", True)
    monkeypatch.setattr(S, "EXTEND_UNTIL", "")
    monkeypatch.setitem(S._act, "enabled", True)
    monkeypatch.setitem(S._act, "last_ts", 0)
    monkeypatch.setitem(S._act, "ext_last_ts", 0)
    return S


def test_extends_when_keep_with_demand_near_end(S):
    why, T = S._extend_decision(_D(), 1, _live(rem=5, el=30), 1000.0)
    assert why == "" and T == 40          # 30 + 5 + 5


def test_capped_at_max_step1(S):
    why, T = S._extend_decision(_D(), 1, _live(rem=5, el=88), 1000.0)
    assert T == 95
    why, T = S._extend_decision(_D(), 1, _live(rem=5, el=90), 1000.0)
    assert T is None and "上限" in why


@pytest.mark.parametrize("live,why", [
    (_live(rem=10), "還沒快結束"),
    (_live(rem=3), "黃燈保護"),
    (_live(step=2), "不在主綠燈"),
    (_live(clearance=True), "不在主綠燈"),
    (_live(rem=None), "不知道"),
])
def test_never_extends_outside_safe_window(S, live, why):
    w, T = S._extend_decision(_D(), 1, live, 1000.0)
    assert T is None and why in w


def test_needs_keep_decision_and_demand(S):
    assert S._extend_decision(_D(action="SWITCH"), 1, _live(), 1000.0)[1] is None
    assert S._extend_decision(_D(by="min_green"), 1, _live(), 1000.0)[1] is None
    assert S._extend_decision(_D(green=0.5), 1, _live(), 1000.0)[1] is None


def test_no_back_to_back_commands(S, monkeypatch):
    monkeypatch.setitem(S._act, "last_ts", 998.0)
    assert "不到" in S._extend_decision(_D(), 1, _live(), 1000.0)[0]


def test_protects_offramp_mainline(S):
    """上匝道綠燈、對向下匝道排隊已達儲車一半(300 m)→ 不延長。"""
    w, T = S._extend_decision(_D(red=60), 1, _live(), 1000.0)   # 60 台 × 6 m = 360 m
    assert T is None and "下匝道" in w


def test_disabled_or_after_until(S, monkeypatch):
    import time
    monkeypatch.setattr(S, "EXTEND_UNTIL", "2020-01-01T00:00:00")
    assert "結束" in S._extend_decision(_D(), 1, _live(), time.time())[0]
    monkeypatch.setattr(S, "EXTEND_GREEN", False)
    assert S._extend_decision(_D(), 1, _live(), 1000.0)[1] is None


def test_actuate_sends_extend_frame(S, monkeypatch):
    calls = []

    def post(path, body):
        calls.append((path, body))
        return {"token": "T"} if path.endswith("/prepare") else {"sent": {"seq": 1, "raw": "x"}}
    monkeypatch.setattr(S, "_daemon_post", post)
    S._actuate(_D(), 2, _live(rem=5, el=20))
    assert calls[0][1] == {"code": "5F1C", "info_hex": "02011E", "by": "algorithm-extend"}   # T = 20+5+5 = 30 = 0x1E
    assert S._act["ext_last_ts"] > 0


def test_end_command_blocked_right_after_extend(S, monkeypatch):
    import time
    monkeypatch.setattr(S, "ab_firstgreen_side", lambda now=None: "A")
    monkeypatch.setitem(S._act, "ext_last_ts", time.time())
    assert "不連送" in S._actuate_gates(_live(rem=20), time.time())
