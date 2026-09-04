# -*- coding: utf-8 -*-
"""模擬器測試 —— 重點在「校準關卡真的會擋」。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.signal_sim import (  # noqa: E402
    SimConfig, calibrate, estimate_arrivals, replay_actual, simulate,
)


def _rows(n=400, dt=5.0):
    """造一段假資料:分相每 60 秒輪替,紅燈時排隊線性成長、綠燈時遞減。"""
    from datetime import datetime, timedelta
    t0 = datetime(2026, 9, 4, 8, 0, 0)
    rows = []
    q = {1: 0.0, 2: 0.0}
    green = 1
    since_sw = 0.0
    for i in range(n):
        ts = (t0 + timedelta(seconds=i * dt)).isoformat(timespec="seconds")
        for p in (1, 2):
            q[p] += (0.08 if p == 1 else 0.13) * dt * 7.0   # 到達(公尺)
        q[green] = max(0.0, q[green] - 0.5 * dt * 7.0)      # 放行
        rows.append((ts, green, round(q[1], 1), round(q[2], 1)))
        since_sw += dt
        if since_sw >= 60:
            green = 2 if green == 1 else 1
            since_sw = 0.0
    return rows


def test_estimate_arrivals_from_red_growth():
    """到達率要從紅燈期間的排隊成長估 —— 綠燈期間分不出到達與離開。"""
    rows = _rows()
    a = estimate_arrivals(rows)
    assert a[1]["samples"] > 3 and a[2]["samples"] > 3
    # 造資料時分相1 用 0.08、分相2 用 0.13 輛/秒
    assert a[1]["veh_per_sec"] == pytest.approx(0.08, abs=0.03)
    assert a[2]["veh_per_sec"] == pytest.approx(0.13, abs=0.03)


def test_calibration_gate_blocks_bad_model():
    """🛑 校準是整條模擬路線的關卡:重現不了現場就必須擋下來。

    這裡故意餵一個錯很多的到達率,模擬排隊必然對不上實際,
    calibrate 必須回 usable=False 並說明原因。
    """
    rows = _rows()
    wrong = {1: {"veh_per_sec": 0.9, "samples": 50},
             2: {"veh_per_sec": 0.9, "samples": 50}}
    sim = replay_actual(rows, wrong, SimConfig(dt_sec=5.0))
    cal = calibrate(rows, sim)
    assert cal["usable"] is False
    assert "MAE" in cal["reason"] or "相關係數" in cal["reason"]


def test_simulator_enforces_min_and_max_green():
    """兩套控制受到的安全約束必須一致,否則比較不公平。"""
    cfg = SimConfig(dt_sec=1.0, min_green_sec={1: 10.0, 2: 10.0},
                    max_green_sec=30.0)
    arr = {1: {"veh_per_sec": 0.1}, 2: {"veh_per_sec": 0.1}}

    # 每一步都喊要換 → 仍不得早於最小綠
    out = simulate(arr, lambda st: True, 120.0, cfg)
    assert out["switch_count"] <= 120 / cfg.min_green_sec[1] + 1

    # 永遠不換 → 仍會被最大綠強制切
    out2 = simulate(arr, lambda st: False, 120.0, cfg)
    assert out2["forced_switch_count"] >= 3
