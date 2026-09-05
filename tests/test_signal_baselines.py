"""基準控制器與 benchmark 彙整 —— 我方演算法的對照組(不含 OPAC)。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.signal_baselines import (  # noqa: E402
    actuated, fixed_time, max_pressure, metrics, run_benchmark, webster_split,
)
from detection.signal_decision_engine import ApproachState, decide  # noqa: E402
from detection.signal_sim import SimConfig  # noqa: E402

CFG = SimConfig(dt_sec=5.0, min_green_sec={1: 10.0, 2: 20.0}, max_green_sec=100.0,
                lost_time_sec=5.0, meters_per_vehicle=6.0,
                saturation_by_phase={1: 0.139, 2: 0.139})
ROLES = {1: {"storage_m": 210, "priority": False},
         2: {"storage_m": 600, "priority": True}}


def _ours(state):
    g = state["green_phase"]
    r = 2 if g == 1 else 1
    qv = state["queue_veh"]
    d = decide(
        green_phase=g, green_elapsed_sec=state["green_elapsed"],
        green_side=ApproachState(g, queue_m=qv[g] * 6.0,
                                 storage_m=ROLES[g]["storage_m"],
                                 priority=ROLES[g]["priority"]),
        red_side=ApproachState(r, queue_m=qv[r] * 6.0,
                               storage_m=ROLES[r]["storage_m"],
                               priority=ROLES[r]["priority"],
                               waiting_sec=state["green_elapsed"]),
        min_green_sec=CFG.min_green_sec[g], max_green_sec=CFG.max_green_sec,
        saturation_vph=500.0, meters_per_vehicle=6.0, lost_time_sec=5.0)
    return d.action == "SWITCH"


def _rate(t, p):
    return 0.045 if p == 1 else 0.032


# ── Webster ──────────────────────────────────────────────────────────

def test_webster_normal_demand():
    w = webster_split({1: 0.045, 2: 0.032}, {1: 0.139, 2: 0.139}, 5.0,
                      min_green={1: 10.0, 2: 20.0})
    assert w["feasible"] is True
    # y1 = .045/.139 = .324, y2 = .23, Y = .554
    assert abs(w["flow_ratio"][1] - 0.324) < 0.01
    assert abs(w["Y"] - 0.554) < 0.01
    # C0 = (1.5*10 + 5) / (1 - .554) = 20/.446 = 44.8
    assert abs(w["cycle"] - 44.8) < 0.5
    # 最小綠會把分相2 頂到 20
    assert w["green"][2] >= 20.0


def test_webster_over_capacity_says_no():
    """需求超過容量時要照實說無解,不可以硬套出一個看起來合理的數字。"""
    w = webster_split({1: 0.10, 2: 0.08}, {1: 0.139, 2: 0.139}, 5.0)
    assert w["feasible"] is False
    assert w["cycle"] is None
    assert "無解" in w["note"]


# ── 各基準的行為 ──────────────────────────────────────────────────────

def test_fixed_time_switches_at_split():
    fn = fixed_time({1: 35.0, 2: 40.0})
    base = {"t": 0, "green_phase": 1, "queue_veh": {1: 0, 2: 0}}
    assert fn({**base, "green_elapsed": 30.0}) is False
    assert fn({**base, "green_elapsed": 35.0}) is True
    assert fn({**base, "green_phase": 2, "green_elapsed": 35.0}) is False
    assert fn({**base, "green_phase": 2, "green_elapsed": 40.0}) is True


def test_actuated_gaps_out_only_after_queue_empties():
    fn = actuated(gap_sec=3.0)
    st = {"green_phase": 1, "green_elapsed": 30.0}
    # 還有車 → 不切
    assert fn({**st, "t": 0.0, "queue_veh": {1: 4.0, 2: 2.0}}) is False
    # 剛清空 → 開始計 gap,還不切
    assert fn({**st, "t": 5.0, "queue_veh": {1: 0.0, 2: 2.0}}) is False
    # 過了 gap → 切
    assert fn({**st, "t": 9.0, "queue_veh": {1: 0.0, 2: 2.0}}) is True
    # 中途又來車 → gap 計時重來
    assert fn({**st, "t": 12.0, "queue_veh": {1: 2.0, 2: 2.0}}) is False
    assert fn({**st, "t": 13.0, "queue_veh": {1: 0.0, 2: 2.0}}) is False


def test_max_pressure_follows_the_bigger_queue():
    fn = max_pressure({1: 0.139, 2: 0.139})
    st = {"t": 0, "green_phase": 1, "green_elapsed": 30.0}
    assert fn({**st, "queue_veh": {1: 5.0, 2: 2.0}}) is False   # 綠側壓力大 → 續
    assert fn({**st, "queue_veh": {1: 1.0, 2: 6.0}}) is True    # 紅側壓力大 → 切
    # 飽和流不同的話,壓力要跟著加權
    fn2 = max_pressure({1: 0.05, 2: 0.20})
    assert fn2({**st, "queue_veh": {1: 5.0, 2: 2.0}}) is True


# ── 彙整 ──────────────────────────────────────────────────────────────

def test_benchmark_runs_all_controllers_on_same_demand():
    b = run_benchmark(_rate, 3600.0, CFG, _ours, start_phase=1,
                      plan_green={1: 35.0, 2: 40.0},
                      flow_veh_per_sec={1: 0.045, 2: 0.032})
    assert set(b["results"]) == {"ours", "fixed_plan", "webster",
                                 "actuated", "max_pressure"}
    # 同一份需求
    assert abs(b["demand_veh"] - (0.045 + 0.032) * 3600) < 1.0
    for key, m in b["results"].items():
        assert m["delay_per_veh_sec"] is not None, key
        assert m["throughput_vph"] is not None, key
    # 對照組不含現行控制
    assert "opac" not in b["results"] and "actual" not in b["results"]
    assert "OPAC" in b["note"] and "校準" in b["note"]
    for key, row in b["vs_baseline"].items():
        assert "delay_per_veh_sec" in row, key
        assert row["delay_per_veh_sec"]["ours"] == b["results"]["ours"]["delay_per_veh_sec"]


def test_fixed_plan_is_worse_than_ours_under_asymmetric_demand():
    """固定時制把綠燈平均給兩邊,需求不對稱時必然吃虧 —— 這是動態控制的基本盤,
    連這個都贏不了就不必談接管。"""
    b = run_benchmark(_rate, 3600.0, CFG, _ours, start_phase=1,
                      plan_green={1: 35.0, 2: 40.0},
                      flow_veh_per_sec={1: 0.045, 2: 0.032})
    assert b["vs_baseline"]["fixed_plan"]["delay_per_veh_sec"]["pct"] < 0


def test_metrics_counts_residual_queue():
    """控制器把車擋著不放時延滯會低估,殘留排隊必須一起回報。"""
    sim = {"duration_sec": 100.0, "total_delay_veh_sec": 500.0,
           "avg_queue_m_1": 10.0, "avg_queue_m_2": 20.0,
           "trajectory": [{"t": 100.0, "q1": 3.0, "q2": 2.0}]}
    m = metrics(sim, demand_veh=50.0, cfg=CFG)
    assert m["residual_queue_veh"] == 5.0
    assert m["delay_per_veh_sec"] == round(500.0 / 45.0, 2)   # 分母是放行數不是需求
    assert m["avg_queue_m"] == 15.0


def test_benchmark_is_deterministic():
    a = run_benchmark(_rate, 1800.0, CFG, _ours, start_phase=1,
                      plan_green={1: 35.0, 2: 40.0},
                      flow_veh_per_sec={1: 0.045, 2: 0.032})
    b = run_benchmark(_rate, 1800.0, CFG, _ours, start_phase=1,
                      plan_green={1: 35.0, 2: 40.0},
                      flow_veh_per_sec={1: 0.045, 2: 0.032})
    assert a["results"]["ours"] == b["results"]["ours"]
    assert a["results"]["actuated"] == b["results"]["actuated"]
