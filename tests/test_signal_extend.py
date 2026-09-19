"""延長綠燈(規範 16613 K(C)d):考慮整個分相時間的嚴謹算法(extend_plan)。

5F1C(分相, 1, T):T = 步階1 總長(09-15 現場驗證)。延長 Δ 秒,對向就多等 Δ 秒、週期多 Δ 秒,
所以 Δ = min(需求, 最大綠, 對向儲車, 週期, 單次上限)。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

BASE = dict(el=30.0, rem=5.0, q_green_m=60.0, q_red_m=12.0, arr_red_vpm=3.0, storage_red_m=600.0,
            min_green_other=20.0, mpv=6.0, sat_vph=974.0, yellow=3.0, all_red=2.0,
            ped_flash=5.0, max_green=100.0, cycle_max=120.0, opp_ratio=0.8, min_delta=3, max_delta=10)


def _plan(**kw):
    from api.routes.signal_shadow import extend_plan
    a = dict(BASE); a.update(kw)
    return extend_plan(**a)


def test_need_is_queue_discharge_minus_remaining_green_including_ped_flash():
    """10 台 × 3.7 秒 = 37 秒;剩 5 秒主綠 + 5 秒行閃 → 還差約 27 秒,被單次上限 10 卡住。"""
    p = _plan()
    assert p["caps"]["需求"] == pytest.approx(10 * 3600 / 974 - 10, abs=0.1)
    assert p["delta"] == 10 and p["binding"] == "單次上限" and p["T"] == 45


def test_no_extension_when_remaining_green_clears_queue():
    """2 台 × 3.7 = 7.4 秒 < 剩 5 + 行閃 5 → 放得完,不延。"""
    p = _plan(q_green_m=12.0)
    assert p["T"] is None and p["binding"] == "需求" and "放得完" in p["why"]


def test_no_extension_without_queue():
    assert _plan(q_green_m=0.0)["T"] is None


def test_max_green_counts_ped_flash():
    """已亮 83 + 剩 5 + 行閃 5 = 93 → 最大綠 100 只剩 7 秒。"""
    p = _plan(el=83.0, cycle_max=300.0)          # 放寬週期,單獨看最大綠
    assert p["binding"] == "最大綠" and p["delta"] == 7 and p["T"] == 95
    # 同一狀況在週期 120 下,週期先卡住(本相 83+5+5+3+2 = 98,+ 對向最短 25 = 123 > 120)
    assert _plan(el=83.0)["binding"] == "週期"


def test_opposing_storage_counts_whole_remaining_phase_time():
    """對向(下匝道)已 470 m、儲車 480 m 上限只剩 10 m ≈ 1.7 台;每分鐘 30 台 → 3.3 秒就滿,
    扣掉剩餘 + 行閃 + 黃 + 全紅(15 秒)已經不夠 → 不延。"""
    p = _plan(q_red_m=470.0, arr_red_vpm=30.0)
    assert p["T"] is None and p["binding"] == "對向儲車"


def test_opposing_already_over_storage_blocks():
    p = _plan(q_red_m=500.0, arr_red_vpm=0.0)
    assert p["T"] is None and p["binding"] == "對向儲車"


def test_cycle_cap_uses_whole_phase_and_other_min():
    """已亮 80:本相總長 80+5+5+3+2 = 95,對向最短 20+3+2 = 25 → 已 120,週期不能再加。"""
    p = _plan(el=80.0, max_green=210.0)
    assert p["T"] is None and p["binding"] == "週期"


# ── _extend_decision:時機與狀態閘門 ──

class _D:
    def __init__(self, action="KEEP", by="cost"):
        self.action, self.decided_by, self.reason = action, by, "x"
        self.detail = {}


def _live(step=1, rem=5, el=30.0, **kw):
    d = {"control_mode": "external_dynamic", "clearance": False, "stale": False,
         "step_id": step, "step_remain_sec": rem, "phase_elapsed_sec": el}
    d.update(kw)
    return d


@pytest.fixture
def S(monkeypatch):
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "EXTEND_GREEN", True)
    monkeypatch.setattr(S, "EXTEND_SHADOW", False)
    monkeypatch.setattr(S, "EXTEND_UNTIL", "")
    monkeypatch.setattr(S, "EXTEND_TIMING", "late")      # 這組測試驗的是舊行為(快結束才評估)
    monkeypatch.setitem(S._act, "enabled", True)
    monkeypatch.setitem(S._act, "last_ts", 0)
    monkeypatch.setitem(S._act, "ext_last_ts", 0)
    monkeypatch.setattr(S, "_events_flow_vpm", lambda ph, **k: 3.0)
    monkeypatch.setattr(S, "_last_meas", {"q": {1: 60.0, 2: 12.0}})
    return S


def test_decision_extends_with_queue(S):
    why, T, plan = S._extend_decision(_D(), 1, _live(rem=5, el=30), 1000.0)
    assert why == "" and T == plan["T"] and T > 35


@pytest.mark.parametrize("live,why", [
    (_live(rem=10), "還沒快結束"),
    (_live(rem=3), "黃燈保護"),
    (_live(step=2), "不在主綠燈"),
    (_live(clearance=True), "不在主綠燈"),
    (_live(rem=None), "不知道"),
])
def test_never_extends_outside_safe_window(S, live, why):
    w, T, _ = S._extend_decision(_D(), 1, live, 1000.0)
    assert T is None and why in w


def test_needs_cost_keep(S):
    assert S._extend_decision(_D(action="SWITCH"), 1, _live(), 1000.0)[1] is None
    assert S._extend_decision(_D(by="min_green"), 1, _live(), 1000.0)[1] is None


def test_no_back_to_back_commands(S, monkeypatch):
    monkeypatch.setitem(S._act, "last_ts", 998.0)
    assert "不到" in S._extend_decision(_D(), 1, _live(), 1000.0)[0]


def test_shadow_mode_never_sends(S, monkeypatch):
    calls = []
    monkeypatch.setattr(S, "EXTEND_SHADOW", True)
    monkeypatch.setattr(S, "_daemon_post", lambda p, b: calls.append(p) or {"token": "T"})
    monkeypatch.setattr(S, "_extend_shadow_record", lambda *a, **k: None)
    S._actuate(_D(), 1, _live(rem=5, el=30))
    assert calls == []


def test_actuate_sends_total_length(S, monkeypatch):
    calls = []

    def post(path, body):
        calls.append((path, body))
        return {"token": "T"} if path.endswith("/prepare") else {"sent": {"seq": 1, "raw": "x"}}
    monkeypatch.setattr(S, "_daemon_post", post)
    monkeypatch.setattr(S, "_extend_shadow_record", lambda *a, **k: None)
    _, T, plan = S._extend_decision(_D(), 1, _live(rem=5, el=30), 1000.0)   # 先算預期(送出後會被不連送擋)
    S._actuate(_D(), 1, _live(rem=5, el=30))
    body = calls[0][1]
    assert body["code"] == "5F1C" and body["info_hex"].startswith("0101") and body["by"] == "algorithm-extend"
    assert int(body["info_hex"][4:], 16) == T == 30 + 5 + plan["delta"]


def test_end_command_blocked_right_after_extend(S, monkeypatch):
    import time
    monkeypatch.setattr(S, "ab_firstgreen_side", lambda now=None: "A")
    monkeypatch.setitem(S._act, "ext_last_ts", time.time())
    assert "不連送" in S._actuate_gates(_live(rem=20), time.time())


def test_disabled_or_after_until(S, monkeypatch):
    """🛑 2026-09-18 契約變更:試行時段外**照算照記、只是不送**。

    舊版截止後 _extend_decision 直接回「結束」不再計算,時段外的影子紀錄就斷了。
    現在「送不送」統一由 extend_live_now 判,_extend_decision 只管「該不該延、延幾秒」。
    """
    import time
    monkeypatch.setattr(S, "EXTEND_SHADOW", False)
    monkeypatch.setattr(S, "EXTEND_UNTIL", "2020-01-01T00:00:00")
    on, why = S.extend_live_now(time.time())
    assert on is False and "結束" in why, "截止後不可再下發"
    monkeypatch.setattr(S, "EXTEND_GREEN", False)
    assert S._extend_decision(_D(), 1, _live(), 1000.0)[1] is None


# ── early(預設):最小綠燈結束前決定 —— 2026-09-19 受控驗證:最小綠內 10/12 生效、之後 2/11 ──
@pytest.fixture
def E(S, monkeypatch):
    import detection.signal_timing_lookup as L
    monkeypatch.setattr(S, "EXTEND_TIMING", "early")
    monkeypatch.setattr(S, "EXTEND_EARLY_LEAD", 6.0)
    monkeypatch.setitem(S._ext_trip, "last_green_key", None)
    monkeypatch.setattr(S, "_last_meas", {"q": {1: 150.0, 2: 150.0}})   # 剩下的綠燈放不完的排隊
    monkeypatch.setattr(L, "plan_params", lambda pid: {"min_green": [10, 20], "yellow": 3, "all_red": 2})
    monkeypatch.setattr(L, "current_base_plan", lambda *a, **k: 1)
    return S


def test_early_evaluates_inside_min_green_even_when_engine_says_min_green(E):
    why, T, plan = E._extend_decision(_D(by="min_green"), 1, _live(rem=22, el=7.0), 1000.0)
    assert why == "" and T is not None and plan["timing"] == "early"


@pytest.mark.parametrize("ph,el,frag", [
    (1, 3.0, "還沒到評估時機"),      # 分相1 最小綠 10:4~9 秒才評估
    (1, 9.5, "已過最小綠"),
    (2, 12.0, "還沒到評估時機"),     # 分相2 最小綠 20:14~19 秒
    (2, 19.5, "已過最小綠"),
])
def test_early_window_follows_each_phase_min_green(E, ph, el, frag):
    w, T, _ = E._extend_decision(_D(by="min_green"), ph, _live(rem=20, el=el), 1000.0)
    assert T is None and frag in w


def test_early_decides_once_per_green(E):
    assert E._extend_decision(_D(by="min_green"), 1, _live(rem=22, el=5.0), 1000.0)[1] is not None
    w, T, _ = E._extend_decision(_D(by="min_green"), 1, _live(rem=18, el=9.0), 1004.0)
    assert T is None and "已評估過" in w
    # 下一段綠燈(開始時刻不同)可以再評估
    assert E._extend_decision(_D(by="min_green"), 1, _live(rem=22, el=5.0), 1100.0)[1] is not None


def test_early_demand_counts_arrivals_during_remaining_green(E):
    """提早決定時,剩下的綠燈裡還會有車到:到達率越高,要延的越多。"""
    base = dict(el=7.0, rem=20.0, q_green_m=60.0, q_red_m=0.0, arr_red_vpm=0.0, storage_red_m=210.0,
                min_green_other=20.0, mpv=6.0, sat_vph=1200.0, yellow=3.0, all_red=2.0, max_delta=30)
    lo = E.extend_plan(arr_green_vpm=0.0, **base)["caps"]["需求"]
    hi = E.extend_plan(arr_green_vpm=12.0, **base)["caps"]["需求"]
    assert hi - lo == pytest.approx(12.0 / 60.0 * 20.0 * 3.0, abs=0.2)   # 4 台 × 3 秒
