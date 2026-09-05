"""演算法最佳化的驗證平台:天花板(beam search)、多情境、訓練/驗證分離。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.signal_baselines import _demand_veh, metrics  # noqa: E402
from detection.signal_optimize import (  # noqa: E402
    DEFAULT_PARAMS, evaluate, optimal_policy, rule_switch_fn, scenario_set, tune,
)
from detection.signal_sim import SimConfig, simulate  # noqa: E402

SAT = {1: 1184 / 3600, 2: 909 / 3600}
CFG = SimConfig(dt_sec=5.0, min_green_sec={1: 10.0, 2: 20.0}, max_green_sec=100.0,
                lost_time_sec=5.0, meters_per_vehicle=6.0, saturation_by_phase=SAT)
ROLES = {1: {"storage_m": 210, "priority": False},
         2: {"storage_m": 600, "priority": True}}


# ── 情境:必須是決定性的,否則任何比較都不可重現 ──────────────────────

def test_scenarios_are_deterministic():
    a = scenario_set("train")
    b = scenario_set("train")
    assert [s["name"] for s in a] == [s["name"] for s in b]
    for x, y in zip(a, b):
        assert [x["rate"](t, 1) for t in range(0, 600, 50)] == \
               [y["rate"](t, 1) for t in range(0, 600, 50)]


def test_train_and_test_sets_do_not_overlap():
    """訓練集調過的情境不能拿來當驗證 —— 那樣的『改進』證明不了任何事。"""
    tr = {s["name"] for s in scenario_set("train")}
    te = {s["name"] for s in scenario_set("test")}
    assert tr and te and not (tr & te)


# ── 天花板 ────────────────────────────────────────────────────────────

def test_optimal_respects_min_and_max_green():
    sc = scenario_set("train")[1]
    opt = optimal_policy(sc["rate"], 900.0, CFG, beam=80)
    ts = opt["switch_times"]
    # 相鄰換相間隔 = 某一相的綠燈長度,必須 >= 該相最小綠、<= 最大綠
    for a, b in zip(ts, ts[1:]):
        gap = b - a
        assert gap >= min(CFG.min_green_sec.values()), (a, b)
        assert gap <= CFG.max_green_sec + CFG.dt_sec, (a, b)


def test_optimal_beam_converges():
    """beam 加大結果幾乎不動,才能當上界用。"""
    sc = scenario_set("train")[1]
    a = optimal_policy(sc["rate"], 900.0, CFG, beam=50)["total_delay_veh_sec"]
    b = optimal_policy(sc["rate"], 900.0, CFG, beam=200)["total_delay_veh_sec"]
    assert abs(a - b) / b < 0.02


def test_optimal_beats_every_heuristic():
    """離線最佳解看得到未來,任何線上控制都不該贏過它;贏了就是模型不一致。"""
    sc = scenario_set("train")[1]
    dur = 900.0
    dem = _demand_veh(sc["rate"], dur, CFG.dt_sec)
    opt = metrics(optimal_policy(sc["rate"], dur, CFG, beam=200), dem, CFG)
    sim = simulate(sc["rate"], rule_switch_fn(CFG, ROLES), dur, CFG, start_phase=1)
    ours = metrics(sim, dem, CFG)
    assert opt["delay_per_veh_sec"] <= ours["delay_per_veh_sec"] + 1e-6


# ── 評估與調參 ────────────────────────────────────────────────────────

def test_evaluate_reports_every_scenario_and_gap():
    res = evaluate(scenario_set("train")[:2], CFG, ROLES,
                   duration_sec=900.0, with_optimal=True, beam=80)
    assert len(res["rows"]) == 2
    for r in res["rows"]:
        assert r["ours"] is not None and r["optimal"] is not None
    assert "ours" in res["gap_to_optimal_pct"]
    assert res["gap_to_optimal_pct"]["ours"] >= -1e-6, "不可能贏過離線最佳解"


def test_keep_weight_reduces_switching():
    """綠側價值加權變大 → 更不願換相。參數的作用方向要說得出來。"""
    sc = scenario_set("train")[1]
    a = simulate(sc["rate"], rule_switch_fn(CFG, ROLES, {"keep_weight": 1.0}),
                 1800.0, CFG, start_phase=1)["switch_count"]
    b = simulate(sc["rate"], rule_switch_fn(CFG, ROLES, {"keep_weight": 3.0}),
                 1800.0, CFG, start_phase=1)["switch_count"]
    assert b < a


def test_tune_improves_training_score_and_reports_history():
    t = tune(CFG, ROLES, grid={"keep_weight": [1.0, 3.0]}, duration_sec=900.0)
    base = evaluate(scenario_set("train"), CFG, ROLES, DEFAULT_PARAMS,
                    duration_sec=900.0)["mean_delay_per_veh"]["ours"]
    assert t["train_delay_per_veh"] <= base + 1e-6
    assert t["history"] and "params" in t["history"][0]
