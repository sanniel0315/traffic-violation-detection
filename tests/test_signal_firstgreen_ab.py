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


def _slots(n, **kv):
    return [dict({"slot": "s%d" % i}, **kv) for i in range(n)]


def _tier(sA, sB, cyc_p=0.001, spillA=0, spillB=0, belowA=0):
    return {"slots_A": sA, "slots_B": sB, "spillback_A": spillA, "spillback_B": spillB,
            "green_below_min_A": belowA,
            "compare_slots": {"cycle_sec": {"p": cyc_p}}}


def test_verdict_needs_three_slots_each(S):
    v = S._fg_verdict({1: {"peak": _tier(_slots(2, cycle_sec=60), _slots(4, cycle_sec=70))},
                       2: {"peak": _tier(_slots(2), _slots(4))}}, "peak")
    assert v["result"] == "樣本不足"


def test_verdict_passes_when_all_hold(S):
    good = dict(cycle_sec=62, queue_max_m=30, queue_avg_m=6)
    base = dict(cycle_sec=73, queue_max_m=31, queue_avg_m=6)
    v = S._fg_verdict({1: {"peak": _tier(_slots(3, **good), _slots(4, **base))},
                       2: {"peak": _tier(_slots(3, **good), _slots(4, **base))}}, "peak")
    assert v["result"] == "通過", v


def test_verdict_fails_when_onramp_queue_worse(S):
    """上匝道最大排隊超過 B 的 +20% → 不通過,即使週期縮短了。"""
    v = S._fg_verdict({1: {"peak": _tier(_slots(3, cycle_sec=62, queue_max_m=50, queue_avg_m=6),
                                          _slots(4, cycle_sec=73, queue_max_m=30, queue_avg_m=6))},
                       2: {"peak": _tier(_slots(3, queue_max_m=30, queue_avg_m=6),
                                          _slots(4, queue_max_m=30, queue_avg_m=6))}}, "peak")
    assert v["result"] == "不通過" and "上匝道最大排隊" in v["text"]


def test_verdict_fails_on_below_min_green(S):
    v = S._fg_verdict({1: {"peak": _tier(_slots(3, cycle_sec=62), _slots(4, cycle_sec=73), belowA=1)},
                       2: {"peak": _tier(_slots(3), _slots(4))}}, "peak")
    assert v["result"] == "不通過" and "最小綠" in v["text"]


def test_slot_means_group_by_half_hour(S, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(S, "AB_FIRSTGREEN_MIN", 30.0)
    t = lambda h, m: datetime(2026, 9, 14, h, m).timestamp()
    monkeypatch.setattr(S, "FG_MIN_CYCLES_PER_SLOT", 1)
    rows = [{"start": t(17, 1), "cycle_sec": 60}, {"start": t(17, 20), "cycle_sec": 64},
            {"start": t(18, 5), "cycle_sec": 70}]
    got = S._fg_slot_means(rows, ("cycle_sec",))
    assert [g["cycle_sec"] for g in got] == [62, 70] and [g["cycles"] for g in got] == [2, 1]


def test_partial_slot_is_not_a_slot(S, monkeypatch):
    """進行中的時段(2~3 個週期)不可當成完整一段 —— 19:30 那段就這樣把 B 拉低。"""
    from datetime import datetime
    monkeypatch.setattr(S, "AB_FIRSTGREEN_MIN", 30.0)
    base = datetime(2026, 9, 14, 18, 30).timestamp()
    full = [{"start": base + 73 * i, "cycle_sec": 73} for i in range(24)]
    part = [{"start": datetime(2026, 9, 14, 19, 30).timestamp() + 50 * i, "cycle_sec": 52} for i in range(3)]
    got = S._fg_slot_means(full + part, ("cycle_sec",))
    assert [g["cycle_sec"] for g in got] == [73]


def test_report_refuses_without_start_time(S, monkeypatch):
    """沒有 A/B 開始時刻就不判讀 —— 之前的週期全是 B,會被時鐘規則誤標成 A。"""
    monkeypatch.setattr(S, "AB_FIRSTGREEN_MIN", 30.0)
    monkeypatch.setattr(S, "AB_FIRSTGREEN_SINCE", "")
    assert "不判讀" in S.ab_firstgreen_report(since="", until="", _user=None)["note"]


def test_report_never_reads_before_start(S, monkeypatch):
    seen = {}
    monkeypatch.setattr(S, "AB_FIRSTGREEN_MIN", 30.0)
    monkeypatch.setattr(S, "AB_FIRSTGREEN_SINCE", "2026-09-14T12:27:00")
    monkeypatch.setattr(S, "_actual_runs_from_frames", lambda a, b: seen.setdefault("a", a) and [])
    monkeypatch.setattr(S, "_phase_lanes", lambda p: {})
    from detection import signal_eval as E
    monkeypatch.setattr(E, "load_congestion", lambda *a, **k: {})
    r = S.ab_firstgreen_report(since="2026-09-13T00:00:00", until="2026-09-14T13:00:00", _user=None)
    assert seen["a"] == "2026-09-14T12:27:00" and r["since"] == "2026-09-14T12:27:00"
