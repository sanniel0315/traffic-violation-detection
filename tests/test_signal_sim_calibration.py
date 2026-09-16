"""校準要驗「模型能不能重現週期內的排隊」,不是累積誤差(2026-09-16)。

整段連跑從排隊 0 開始跑好幾小時,到達率只要略高於放行,模擬排隊就一路長大 ——
09-14 實測 MAE 108 公尺、r 0.12;同一批資料改成逐週期錨定是 MAE 7.9、r 0.55。
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from detection.signal_sim import (SimConfig, calibrate, calibrate_by_cycle,  # noqa: E402
                                  replay_actual, _phase_segments)

MPV = 6.0


def _rows(cycles=40, green=40, dt=5, arr=0.08, sat=0.3, drift=0.0):
    """做一段「紅燈累積、綠燈消散」的假資料,drift 模擬長期漂移。"""
    t = datetime(2026, 9, 14, 8, 0, 0)
    q = {1: 3.0, 2: 3.0}
    out = []
    ph = 1
    for c in range(cycles):
        for k in range(green // dt):
            for p in (1, 2):
                q[p] += arr * dt + drift * dt
                if p == ph:
                    q[p] = max(0.0, q[p] - sat * dt)
            out.append((t.isoformat(timespec="seconds"), ph,
                        round(q[1] * MPV, 1), round(q[2] * MPV, 1)))
            t += timedelta(seconds=dt)
        ph = 2 if ph == 1 else 1
    return out


def _cfg():
    return SimConfig(dt_sec=5.0, min_green_sec={1: 10.0, 2: 20.0}, max_green_sec=100.0,
                     lost_time_sec=0.0, meters_per_vehicle=MPV,
                     saturation_by_phase={1: 0.3, 2: 0.3})


def _arr(t, p):
    return 0.08


def test_segments_split_on_phase_change():
    segs = _phase_segments(_rows(cycles=4))
    assert len(segs) == 4 and all(len({r[1] for r in s}) == 1 for s in segs)


def test_cycle_anchored_reproduces_a_consistent_world():
    rows = _rows()
    cal = calibrate_by_cycle(rows, _arr, _cfg(), MPV, align_sec=0.0)
    assert cal["usable"], cal
    assert cal["method"] == "cycle" and cal["cycles"] >= 30
    assert cal["phase_1"]["r"] > 0.9 and cal["phase_1"]["mae"] < 3


def test_small_arrival_error_ruins_whole_run_but_not_cycle_anchoring():
    """接近飽和時,到達率只要高估 11%,整段連跑就一路偏掉;逐週期錨定不受影響。

    這正是現場發生的事:同一批 09-14 資料,整段連跑 MAE 108 公尺、r 0.12,
    逐週期錨定 MAE 7.9 公尺、r 0.55。差別不在模型好壞,在誤差會不會累積。
    """
    rows = _rows(cycles=60, arr=0.14)                      # 真實世界(約 93% 飽和)
    cfg = _cfg()

    def model_arr(t, p):                                   # 模型高估一點點
        return 0.155

    whole = calibrate(rows, replay_actual(rows, model_arr, cfg), MPV)
    cycle = calibrate_by_cycle(rows, model_arr, cfg, MPV, align_sec=0.0)
    assert not whole["usable"], "整段連跑在到達率略偏時本來就該失真"
    assert cycle["usable"], cycle
    assert cycle["phase_1"]["mae"] < whole["phase_1"]["mae"] / 3


def test_random_world_still_fails():
    """模型跟現場無關時一定要擋下來(不能因為改判準就變成都會過)。"""
    import random
    random.seed(7)
    rows = [(r[0], r[1], round(random.uniform(0, 60), 1), round(random.uniform(0, 60), 1))
            for r in _rows()]
    cal = calibrate_by_cycle(rows, _arr, _cfg(), MPV, align_sec=0.0)
    assert not cal["usable"] and "相關係數" in cal["reason"]


@pytest.mark.parametrize("field", ["usable", "phase_1", "phase_2", "reason"])
def test_verdict_shape(field):
    assert field in calibrate_by_cycle(_rows(), _arr, _cfg(), MPV)
