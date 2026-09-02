"""影子決策引擎:安全約束與切換規則。

規則來源:2026-09-02 從 OPAC decision.log 41 筆樣本反推,SWITCH ⟺ pn1 > pn2。
本引擎只算不送 —— 任何下發都要走 signal_tc3 的 control/send(預設關閉)。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.signal_decision_engine import ApproachState, decide, compare


def _st(phase, queue_m=None, arrivals=0.0, storage_m=None, priority=False):
    return ApproachState(phase_no=phase, queue_m=queue_m, arrivals=arrivals,
                         storage_m=storage_m, priority=priority)


def _decide(green_q=None, red_q=None, elapsed=30, mn=15, mx=100,
            green_arr=0.0, red_arr=0.0, g_priority=False, g_storage=None):
    return decide(
        green_phase=2, green_elapsed_sec=elapsed,
        green_side=_st(2, green_q, green_arr, g_storage, g_priority),
        red_side=_st(1, red_q, red_arr),
        min_green_sec=mn, max_green_sec=mx,
    )


# ---- 安全約束(優先於一般規則) ----

def test_min_green_blocks_switch():
    """未滿最小綠一律 KEEP,即使紅側需求爆表。"""
    d = _decide(green_q=0, red_q=700, elapsed=10, mn=15)
    assert d.action == "KEEP" and "最小綠" in d.reason


def test_max_green_forces_switch():
    """達最大綠強制切,即使綠側還有需求。"""
    d = _decide(green_q=700, red_q=0, elapsed=100, mx=100)
    assert d.action == "SWITCH" and d.forced_by_max_green


def test_max_green_priority_over_mainline_protection():
    """max-green 檢查在主線保護之前 —— 到頂就是要切,不能無限長綠。"""
    d = _decide(green_q=700, red_q=0, elapsed=120, mx=100,
                g_priority=True, g_storage=600)
    assert d.action == "SWITCH" and d.forced_by_max_green


def test_mainline_protection_blocks_switch():
    """主線保護:下匝道(優先相)排隊逼近儲車上限,不可被切走。"""
    # 儲車 600m,排隊 500m = 83% ≥ 80% 門檻
    d = _decide(green_q=500, red_q=700, elapsed=30,
                g_priority=True, g_storage=600)
    assert d.action == "KEEP" and d.blocked_by_priority


def test_non_priority_phase_no_protection():
    """非優先相沒有這個保護 —— 排隊再滿也可被切。"""
    d = _decide(green_q=500, red_q=700, elapsed=30,
                g_priority=False, g_storage=600)
    assert d.action == "SWITCH" and not d.blocked_by_priority


# ---- 一般規則 SWITCH ⟺ pn1 > pn2 ----

def test_switch_when_red_demand_exceeds_green_residual():
    """紅側排隊多、綠燈已放行久 → 綠側殘量被消化完 → 切。"""
    d = _decide(green_q=14, red_q=70, elapsed=30)   # 綠側2輛,已放行15輛→殘0
    assert d.action == "SWITCH" and d.pn1 > d.pn2


def test_keep_when_green_still_has_backlog():
    """綠側還有大量未消化 → 續綠。"""
    d = _decide(green_q=700, red_q=7, elapsed=20)   # 綠側100輛,放行10輛→殘90
    assert d.action == "KEEP" and d.pn2 >= d.pn1


def test_tie_keeps():
    """平手不切(與 OPAC 一致:pn1 ≤ pn2 才 KEEP)。"""
    d = _decide(green_q=None, red_q=None, elapsed=30)
    assert d.pn1 == d.pn2 == 0 and d.action == "KEEP"


def test_arrivals_counted_into_demand():
    """到達車數要計入需求,不是只看排隊。"""
    a = _decide(green_q=None, red_q=None, elapsed=30, red_arr=5)
    assert a.pn1 == 5 and a.action == "SWITCH"


def test_green_residual_decays_with_elapsed():
    """同樣綠側需求,綠燈亮越久殘量越少(飽和流消化) —— pn2 的核心語意。"""
    short = _decide(green_q=140, red_q=0, elapsed=16)
    long_ = _decide(green_q=140, red_q=0, elapsed=39)
    assert short.pn2 > long_.pn2


def test_none_queue_treated_as_zero_not_crash():
    """排隊量測為 None(未量到)不可當機,以 0 計。"""
    d = _decide(green_q=None, red_q=None, elapsed=30)
    assert d.action == "KEEP"


# ---- 比對 ----

def test_compare_match_and_mismatch():
    d = _decide(green_q=0, red_q=70, elapsed=30)
    assert compare(d, "SWITCH")["match"] is True
    assert compare(d, "KEEP")["match"] is False
    assert compare(d, None)["match"] is None
