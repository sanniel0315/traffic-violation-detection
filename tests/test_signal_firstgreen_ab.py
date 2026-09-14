"""步階1 閘門 A/B:A 段放開、B 段擋;A 段步階1 快結束時仍不送。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture
def S(monkeypatch):
    from api.routes import signal_shadow as S
    monkeypatch.setitem(S._act, "enabled", True)
    monkeypatch.setitem(S._act, "last_ts", 0)
    return S


def _live(step, remain):
    return {"control_mode": "external_dynamic", "clearance": False, "stale": False,
            "step_id": step, "step_remain_sec": remain}


def test_side_b_blocks_step1(S, monkeypatch):
    monkeypatch.setattr(S, "ab_firstgreen_side", lambda now=None: "B")
    assert "第一個綠階" in S._actuate_gates(_live(1, 20), 1000.0)


def test_side_a_allows_step1_with_time_left(S, monkeypatch):
    monkeypatch.setattr(S, "ab_firstgreen_side", lambda now=None: "A")
    assert S._actuate_gates(_live(1, 20), 1000.0) is None


def test_side_a_blocks_step1_near_end(S, monkeypatch):
    """步階1 剩 5 秒:送了可能被推進約 5 秒的延長段,反而更長。"""
    monkeypatch.setattr(S, "ab_firstgreen_side", lambda now=None: "A")
    assert "延長段反而更長" in S._actuate_gates(_live(1, 5), 1000.0)


def test_step2_unaffected_by_step1_rule(S, monkeypatch):
    monkeypatch.setattr(S, "ab_firstgreen_side", lambda now=None: "B")
    assert S._actuate_gates(_live(2, 4), 1000.0) is None


def test_side_alternates_by_slot(S, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(S, "AB_FIRSTGREEN_MIN", 30.0)
    t = lambda h, m: datetime(2026, 9, 14, h, m).timestamp()
    assert S.ab_firstgreen_side(t(10, 5)) == "A"
    assert S.ab_firstgreen_side(t(10, 35)) == "B"
    assert S.ab_firstgreen_side(t(11, 5)) == "A"


def test_peak_includes_1630(S):
    from datetime import datetime
    t = lambda h, m: datetime(2026, 9, 14, h, m).timestamp()
    assert S._is_peak(t(16, 30)) and S._is_peak(t(19, 59))
    assert not S._is_peak(t(16, 29)) and not S._is_peak(t(20, 0))


def test_report_says_disabled_when_off(S, monkeypatch):
    monkeypatch.setattr(S, "AB_FIRSTGREEN_MIN", 0.0)
    assert S.ab_firstgreen_report(since="", until="", _user=None)["enabled"] is False
