"""滾動時程決策:看未來一個週期挑切換時機(2026-09-16 新增)。

現行 decide() 只比當下成本,看不到「再等幾秒這波車就放完」。
這支用校準過的模型往前推,選總延滯最小的切換時機。只算不送。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.signal_rolling import (forward_delay, plan_extra_green,  # noqa: E402
                                      rolling_horizon, _clear_time)
from detection.signal_sim import SimConfig, simulate  # noqa: E402


def _cfg(min1=10.0, min2=10.0, maxg=100.0):
    c = SimConfig(dt_sec=5.0, min_green_sec={1: min1, 2: min2}, max_green_sec=maxg,
                  lost_time_sec=5.0, meters_per_vehicle=6.0)
    c.saturation_by_phase = {1: 0.3, 2: 0.3}
    return c


SAT = {1: 0.3, 2: 0.3}


def test_clear_time_grows_with_queue_and_is_capped():
    assert abs(_clear_time(3.0, 0.3, 0.1, 10.0, 100.0) - 15.0) < 1e-6   # 3/(0.3-0.1)
    assert _clear_time(0.0, 0.3, 0.1, 10.0, 100.0) == 10.0     # 不低於最小綠
    assert _clear_time(99.0, 0.1, 0.2, 10.0, 60.0) == 60.0     # 放不完 → 夾在最大綠


def test_keeps_green_while_its_own_queue_is_still_discharging():
    """綠側還有 10 台在放、對向只有 1 台 → 應該續綠,不是馬上切。"""
    plan = plan_extra_green(lambda t, p: 0.05, SAT, _cfg(), 0.0, 1,
                            {1: 10.0, 2: 1.0}, elapsed=12.0)
    assert plan["switch_now"] is False
    assert plan["best_sec"] >= 10.0


def test_switches_when_own_queue_is_empty_and_other_side_waits():
    """綠側放完、對向 8 台在等 → 現在就該切(這正是我方輸給感應控制的情境)。"""
    plan = plan_extra_green(lambda t, p: 0.02, SAT, _cfg(), 0.0, 1,
                            {1: 0.0, 2: 8.0}, elapsed=15.0)
    assert plan["switch_now"] is True and plan["best_sec"] == 0.0


def test_never_switches_before_min_green():
    plan = plan_extra_green(lambda t, p: 0.02, SAT, _cfg(min1=20.0), 0.0, 1,
                            {1: 0.0, 2: 8.0}, elapsed=5.0)
    assert plan["switch_now"] is False and plan["best_sec"] >= 15.0


def test_forward_delay_prefers_clearing_the_bigger_queue():
    """同樣的未來,先放大排隊那一相的總延滯比較低 —— 模型方向要對。"""
    cfg = _cfg()
    keep_big = forward_delay(lambda t, p: 0.02, SAT, cfg, 0.0, 1, {1: 12.0, 2: 1.0}, 40.0, 120.0)
    switch_now = forward_delay(lambda t, p: 0.02, SAT, cfg, 0.0, 1, {1: 12.0, 2: 1.0}, 0.0, 120.0)
    assert keep_big < switch_now


def test_switch_fn_is_usable_in_the_simulator():
    cfg = _cfg()
    fn = rolling_horizon(lambda t, p: 0.05, SAT, cfg, horizon_sec=60.0)
    sim = simulate(lambda t, p: 0.05, fn, 600.0, cfg, init_queue_veh={1: 2.0, 2: 2.0})
    assert sim["switch_count"] >= 1
    assert sim["total_delay_veh_sec"] > 0


def test_defaults_do_not_enable_the_switch_margin():
    """🛑 2026-09-16 踩過:只改了內層函式的預設,rolling_horizon 仍留 8.0,
       整輪驗證都跑成「幾乎不切、全部撞最大綠」,數字全錯。兩處要一致。"""
    import inspect
    from detection.signal_rolling import plan_extra_green as pe, rolling_horizon as rh
    for fn in (pe, rh):
        assert inspect.signature(fn).parameters["switch_margin_veh_sec"].default == 0.0
        assert inspect.signature(fn).parameters["grid_sec"].default == 2.0


def test_switch_margin_blocks_switching_when_enabled():
    """門檻真的會抑制切換(留著參數就要有測試說明它的作用)。"""
    cfg = _cfg()
    q = {1: 0.0, 2: 8.0}
    assert plan_extra_green(lambda t, p: 0.02, SAT, cfg, 0.0, 1, q, 15.0,
                            switch_margin_veh_sec=0.0)["switch_now"] is True
    assert plan_extra_green(lambda t, p: 0.02, SAT, cfg, 0.0, 1, q, 15.0,
                            switch_margin_veh_sec=1e6)["switch_now"] is False
