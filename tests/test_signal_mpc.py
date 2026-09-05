"""滾動時域最佳化(MPC)—— 安全約束、rollout 忠實度、候選集合。

🛑 現況:MPC 在合成的起伏需求下**還沒有贏過規則版**(見模組 docstring 的
   實測紀錄)。這裡的測試鎖的是「行為正確」與「不會違反安全約束」,
   不是「比較好」—— 沒贏之前不會上線,也不會寫進報告。
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.signal_mpc import (  # noqa: E402
    MPCConfig, _candidates, _rollout, decide_mpc, mpc_switch_fn,
)
from detection.signal_sim import SimConfig, simulate  # noqa: E402

SAT = {1: 1184 / 3600, 2: 909 / 3600}
CFG = MPCConfig(horizon_sec=120.0, dt_sec=5.0, lost_time_sec=5.0,
                min_green_sec={1: 10.0, 2: 20.0}, max_green_sec=100.0,
                saturation_veh_per_sec=SAT, terminal_weight_sec=0.0)


def _rate(t, p):
    return 0.11 if p == 1 else 0.09


# ── 安全約束:最佳化繞不過去 ──────────────────────────────────────────

def test_min_green_blocks_switch():
    d = decide_mpc(green_phase=2, green_elapsed_sec=10.0,
                   queue_veh={1: 20.0, 2: 0.0}, arrival_rate=_rate, cfg=CFG)
    assert d["action"] == "KEEP" and d["forced"] is True
    assert "最小綠" in d["reason"]


def test_max_green_forces_switch():
    d = decide_mpc(green_phase=1, green_elapsed_sec=100.0,
                   queue_veh={1: 30.0, 2: 0.0}, arrival_rate=_rate, cfg=CFG)
    assert d["action"] == "SWITCH" and d["forced"] is True
    assert "最大綠" in d["reason"]


def test_mainline_protection_blocks_switch():
    d = decide_mpc(green_phase=2, green_elapsed_sec=40.0,
                   queue_veh={1: 30.0, 2: 85.0}, arrival_rate=_rate, cfg=CFG,
                   storage_veh={1: 35.0, 2: 100.0}, priority_phase=2)
    assert d["action"] == "KEEP" and d["forced"] is True
    assert "主線保護" in d["reason"]


def test_candidates_respect_min_green_of_both_phases():
    cands = _candidates(1, 5.0, CFG)          # 已亮 5 秒,最小綠 10
    assert all((not c) or c[0] >= 5.0 for c in cands), "第一次換相不可早於最小綠"
    # 第二次換相要滿足對向(分相2)最小綠 20
    for c in cands:
        if len(c) == 2:
            assert c[1] - c[0] >= 20.0


def test_no_empty_candidate():
    """空方案在前推時其實會照尾段策略換相,卻被讀成 KEEP —— 換相會被無限延後。"""
    assert [] not in _candidates(1, 30.0, CFG)


def test_switch_times_are_distinguishable():
    """t1=0 與 t1=5 必須產生不同軌跡;否則等於每次換相都晚一步。"""
    q = {1: 1.0, 2: 4.0}
    c0 = _rollout(q, 1, 30.0, [0.0], _rate, 0.0, CFG)
    c5 = _rollout(q, 1, 30.0, [5.0], _rate, 0.0, CFG)
    assert c0 != c5


# ── rollout 要忠實反映模擬器的動態 ────────────────────────────────────

def test_rollout_matches_simulator_within_tolerance():
    """rollout 與 signal_sim.simulate 是同一套物理,差異只能來自邊界處理。"""
    sim_cfg = SimConfig(dt_sec=5.0, min_green_sec={1: 10.0, 2: 20.0},
                        max_green_sec=100.0, lost_time_sec=5.0,
                        meters_per_vehicle=6.0, saturation_by_phase=SAT)
    q = {1: 1.0, 2: 4.0}
    # 尾段策略 = 滿最小綠就換;simulate 的 switch_fn 恆真等同此策略
    roll = _rollout(q, 2, 0.0, [], _rate, 0.0, CFG)
    sim = simulate(_rate, lambda s: True, 120.0, sim_cfg,
                   init_queue_veh=q, start_phase=2)
    assert abs(roll - sim["total_delay_veh_sec"]) / sim["total_delay_veh_sec"] < 0.05


def test_switch_fn_never_violates_min_green_in_simulation():
    sim_cfg = SimConfig(dt_sec=5.0, min_green_sec={1: 10.0, 2: 20.0},
                        max_green_sec=100.0, lost_time_sec=5.0,
                        meters_per_vehicle=6.0, saturation_by_phase=SAT)
    def rate(t, p):
        f = 0.5 + 0.5 * math.sin(2 * math.pi * t / 1800.0 + (0 if p == 1 else math.pi))
        return (0.16 if p == 1 else 0.13) * (0.35 + 0.65 * f)
    sim = simulate(rate, mpc_switch_fn(CFG, rate), 1800.0, sim_cfg, start_phase=1)
    traj = sim["trajectory"]
    for a, b in zip(traj, traj[1:]):
        if b["green_phase"] != a["green_phase"]:
            assert a["green_elapsed"] >= CFG.min_green(a["green_phase"]), a


def test_decide_exposes_candidate_costs_for_audit():
    d = decide_mpc(green_phase=1, green_elapsed_sec=30.0,
                   queue_veh={1: 0.5, 2: 5.0}, arrival_rate=_rate, cfg=CFG)
    assert d["n_candidates"] > 10
    assert d["candidates"] and "cost" in d["candidates"][0]
    costs = [c["cost"] for c in d["candidates"]]
    assert costs == sorted(costs), "候選要依成本排序,方便稽核"
