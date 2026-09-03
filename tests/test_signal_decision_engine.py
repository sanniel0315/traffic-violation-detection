"""影子決策引擎:我方自己的延滯成本模型 + 安全約束。

不複製 OPAC 的 pn1/pn2(2026-09-02 用 400 筆測過 15 種假設全失敗,
那是它內部狀態機的耦合計數器,無法外部重建)。改用交通工程標準目標:
最小化總延滯 —— 比較「紅側繼續等的成本」與「綠側續綠的價值 + 換相成本」。

本引擎只算不送。任何下發都要走 signal_tc3 的 control/send(預設關閉)。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.signal_decision_engine import (
    ApproachState, decide, compare, evaluate_outcome,
)


def _st(phase, queue_m=None, arrivals=0.0, storage_m=None, priority=False, waiting=0.0):
    return ApproachState(phase_no=phase, queue_m=queue_m, arrivals=arrivals,
                         storage_m=storage_m, priority=priority, waiting_sec=waiting)


def _decide(green_q=None, red_q=None, elapsed=30, mn=15, mx=100,
            green_arr=0.0, red_arr=0.0, g_priority=False, g_storage=None,
            red_wait=0.0):
    return decide(
        green_phase=2, green_elapsed_sec=elapsed,
        green_side=_st(2, green_q, green_arr, g_storage, g_priority),
        red_side=_st(1, red_q, red_arr, waiting=red_wait),
        min_green_sec=mn, max_green_sec=mx,
    )


# ---- 安全約束(優先於成本比較) ----

def test_min_green_blocks_switch():
    """未滿最小綠一律 KEEP,即使紅側延滯爆表。"""
    d = _decide(green_q=0, red_q=700, elapsed=10, mn=15)
    assert d.action == "KEEP" and "最小綠" in d.reason


def test_max_green_forces_switch():
    d = _decide(green_q=700, red_q=0, elapsed=100, mx=100)
    assert d.action == "SWITCH" and d.forced_by_max_green


def test_max_green_beats_mainline_protection():
    """max-green 檢查在主線保護之前 —— 不可無限長綠。"""
    d = _decide(green_q=700, red_q=0, elapsed=120, mx=100,
                g_priority=True, g_storage=600)
    assert d.action == "SWITCH" and d.forced_by_max_green


def test_mainline_protection_blocks_switch():
    """主線保護:下匝道(優先相)排隊逼近儲車上限,不可被切走。"""
    d = _decide(green_q=500, red_q=700, elapsed=30,
                g_priority=True, g_storage=600)
    assert d.action == "KEEP" and d.blocked_by_priority


def test_non_priority_no_protection():
    d = _decide(green_q=500, red_q=700, elapsed=30,
                g_priority=False, g_storage=600)
    assert d.action == "SWITCH"


# ---- 延滯成本模型 ----

def test_long_waiting_red_side_triggers_switch():
    """紅側等越久,切換效益越高 —— 這是延滯模型的核心。"""
    short = _decide(green_q=70, red_q=70, elapsed=20, red_wait=5)
    long_ = _decide(green_q=70, red_q=70, elapsed=20, red_wait=300)
    assert long_.switch_gain > short.switch_gain
    assert long_.action == "SWITCH"


def test_more_red_vehicles_higher_switch_gain():
    """紅側車越多,切換效益越高。"""
    few = _decide(green_q=0, red_q=7, elapsed=30, red_wait=60)
    many = _decide(green_q=0, red_q=70, elapsed=30, red_wait=60)
    assert many.switch_gain > few.switch_gain


def test_green_backlog_keeps_green():
    """綠側還有大量未消化 → 續綠。"""
    d = _decide(green_q=1400, red_q=7, elapsed=20, red_wait=20)
    assert d.action == "KEEP" and d.keep_gain > d.switch_gain


def test_keep_gain_follows_measured_queue_not_elapsed_time():
    """★ keep_gain 由「當下實測排隊」決定,不隨綠燈時間自行衰減。

    🛑 不可以用「需求 − 飽和流×綠燈秒」推剩餘:飽和流是車連續通過時的速率,
       不能假設綠燈全程滿載放行。那樣推會讓長綠燈的 keep_gain 恆為 0 →
       「明明還在排隊卻判該切」(2026-09-02 實測:下匝道 77m/11台排隊、
       綠燈已亮 60 秒,被算成剩餘 0 而誤判 SWITCH)。
    """
    # 同樣排隊量,綠燈長短不影響 keep_gain —— 排隊沒少就是沒少
    short = _decide(green_q=140, red_q=0, elapsed=16)
    long_ = _decide(green_q=140, red_q=0, elapsed=90)
    assert short.keep_gain == long_.keep_gain

    # 排隊真的變少了,keep_gain 才下降
    less = _decide(green_q=70, red_q=0, elapsed=16)
    assert less.keep_gain < short.keep_gain


def test_long_green_with_real_queue_keeps():
    """現場情境:下匝道排隊 77m(11台)、綠燈已亮 60 秒 → 應續綠不是切走。"""
    d = _decide(green_q=77, red_q=14, elapsed=60, red_wait=60)
    assert d.action == "KEEP"


def test_change_cost_prevents_marginal_switch():
    """換相成本是防抖動:微小優勢不值得付換相代價。"""
    d = _decide(green_q=0, red_q=1, elapsed=20, red_wait=1)
    # 紅側只有 0.14 輛等 1 秒,遠不足以抵換相成本
    assert d.action == "KEEP" and d.change_cost > 0


def test_empty_both_sides_keeps():
    """兩側都沒車 → 不切(切了也沒意義,只浪費換相損失)。"""
    d = _decide(green_q=None, red_q=None, elapsed=30)
    assert d.action == "KEEP"


def test_none_queue_not_crash():
    d = _decide(green_q=None, red_q=None, elapsed=30)
    assert d.switch_gain == 0


# ---- 成效評估(取代逐筆一致率) ----

def test_evaluate_outcome_basic():
    samples = [
        {"queue_m_1": 70, "queue_m_2": 140, "storage_2": 600,
         "interval_sec": 5, "switched": False},
        {"queue_m_1": 0, "queue_m_2": 500, "storage_2": 600,
         "interval_sec": 5, "switched": True},
    ]
    r = evaluate_outcome(samples)
    assert r["samples"] == 2
    assert r["switch_count"] == 1
    assert r["max_queue_m_2"] == 500
    assert r["spillback_events_2"] == 1      # 500/600 = 83% ≥ 80%
    assert r["total_delay_veh_sec"] > 0


def test_evaluate_outcome_empty():
    assert evaluate_outcome([]) == {}


def test_lower_delay_is_better():
    """成效比較的核心:延滯低的那套比較好。"""
    good = evaluate_outcome([{"queue_m_1": 10, "queue_m_2": 10,
                              "interval_sec": 5, "switched": False}])
    bad = evaluate_outcome([{"queue_m_1": 200, "queue_m_2": 200,
                             "interval_sec": 5, "switched": False}])
    assert good["total_delay_veh_sec"] < bad["total_delay_veh_sec"]


def test_compare_still_works():
    d = _decide(green_q=0, red_q=70, elapsed=30, red_wait=60)
    assert compare(d, d.action)["match"] is True
    assert compare(d, None)["match"] is None


def test_green_side_flow_counts_even_when_queue_cleared():
    """綠側排隊已清空但車流仍在到達時，不可判成「綠燈沒價值」。

    2026-09-03 實測 13.5 小時：我方提早切的 1074 筆岐異中 67% 的 keep_gain=0,
    reason 都是「綠側價值 0」—— 因為只看靜態排隊,車一放行就歸零。
    但綠燈期間車流仍持續到達被消化,那才是綠燈的價值。
    """
    common = dict(green_phase=1, green_elapsed_sec=40.0,
                  min_green_sec=10.0, max_green_sec=210.0)
    # 紅側 10m(約 1.4 輛)等 40 秒 → 延滯 57 車·秒
    red = ApproachState(2, queue_m=10.0, waiting_sec=40.0)

    # 綠側排隊 0、也沒量到流量 → 舊行為:綠燈確實沒價值,該切
    idle = decide(green_side=ApproachState(1, queue_m=0.0), red_side=red, **common)
    assert idle.keep_gain == 0.0
    assert idle.action == "SWITCH"

    # 綠側排隊同樣是 0,但實測每分鐘 30 輛還在進來 → 綠燈有價值
    busy = decide(green_side=ApproachState(1, queue_m=0.0, flow_vpm=30.0),
                  red_side=red, **common)
    assert busy.keep_gain > 0
    assert busy.detail["stranded_arrivals"] > 0
    # 30 輛/分 = 0.5 輛/秒,被切走要等 lost_time(5) + min_green(10) = 15 秒
    # → 攔下 7.5 輛,keep_gain = 7.5 × 15 = 112.5 車·秒
    assert busy.keep_gain == pytest.approx(112.5, rel=0.01)
    # 57(紅側延滯) < 112.5(綠側價值) + 12.5(換相成本) → 續綠
    assert busy.action == "KEEP"

    # 但綠側流量不是免死金牌:紅側夠塞就仍該切
    heavy_red = decide(green_side=ApproachState(1, queue_m=0.0, flow_vpm=30.0),
                       red_side=ApproachState(2, queue_m=60.0, waiting_sec=60.0),
                       **common)
    assert heavy_red.action == "SWITCH"
