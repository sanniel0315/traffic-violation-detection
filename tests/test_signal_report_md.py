"""Markdown 報告渲染:三等級欄位、方法標記、A/B 差值、配對區塊、無 B/無 TDX 時不炸。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _blk(delay, queue, tp, cycles=10):
    core = {"avg_delay_sec": {"value": delay, "method": "approx"},
            "avg_queue_m": {"value": queue, "method": "measured"},
            "throughput_vph": {"value": tp, "method": "measured"},
            "avg_travel_sec": {"value": 11.3, "method": "approx", "confidence": "low"},
            "avg_stops": {"value": 0.68, "method": "approx"},
            "delay_p95_sec": {"value": 9.7, "method": "approx"}}
    adv = {"queue_max_m": {"value": 77.0, "method": "measured"}, "spillback_cycles": {"value": 0, "method": "measured"},
           "cycle_sec_avg": {"value": 73.4, "method": "measured"}, "cycle_sec_std": {"value": 26.0, "method": "measured"},
           "green_sec_avg": {"value": 25.7, "method": "measured"}, "green_util": {"value": 0.81, "method": "measured"},
           "speed_avg_kmh": {"value": 34.5, "method": "measured"}}
    return {"cycles": cycles, "core": core, "advanced": adv}


def _report(tier="full", with_b=True):
    r = {"tier": tier, "a": {"since": "2026-09-04T09:00:00", "until": "2026-09-04T12:00:00",
                             "all": _blk(4.61, 7.91, 1228), "by_phase": {"1": _blk(5.6, 7.6, 944), "2": _blk(3.6, 8.2, 1510)},
                             "travel_time_tdx": {"n": 0}, "peak": _blk(4.61, 7.91, 1228, 293), "offpeak": _blk(3.0, 5.0, 900, 40)}}
    if with_b:
        r["b"] = {"since": "2026-09-05T09:00:00", "until": "2026-09-05T12:00:00",
                  "all": _blk(4.0, 7.0, 1300), "by_phase": {"1": _blk(5.0, 7.0, 1000), "2": _blk(3.0, 7.0, 1600)},
                  "travel_time_tdx": {"n": 2, "source": "TDX eTag 站間旅行時間(國道主線,實測)", "avg_travel_time_sec": 118.0,
                                      "pairs": [{"pair_id": "P1", "direction": "E", "n": 2, "travel_time_sec": 118.0, "speed_kmh": 88.0, "vehicles": 300}]}}
        r["ab_test"] = {"all": {"delay_per_veh": {"mean_a": 4.61, "mean_b": 4.0, "diff": -0.61, "t": -1.2, "p": 0.23, "cohen_d": -0.1, "effect": "negligible", "n_a": 293, "n_b": 290},
                                "queue_avg_m": {"mean_a": 7.91, "mean_b": 7.0, "diff": -0.91, "t": -3.9, "p": 0.0001, "cohen_d": -0.33, "effect": "small", "n_a": 293, "n_b": 290}}}
    return r


def _paired():
    return {"source": "controller_5F03", "runs_usable": 293, "earlier": 14, "earlier_meaningful": 14, "same": 9, "hold": 270,
            "later": 0, "delta_meaningful": {"avg": -16.1}, "waste_sec_total": 225.0,
            "hold_compare": {"margin_at_switch": {"avg": 179.7}, "red_waiting_ratio": 0.33}}


def test_min_tier_only_three_core_rows_and_no_advanced():
    from detection.signal_report_md import render
    md = render(_report("min", with_b=False))
    assert "工程報告" in md and "平均延滯" in md and "每小時通過車輛數" in md
    assert "平均旅行時間" not in md and "進階指標" not in md and "統計檢定" not in md
    assert "| 4.61 | 近似 |" in md and "| 7.91 | 實測 |" in md


def test_standard_tier_has_six_core_with_low_confidence_tag():
    from detection.signal_report_md import render
    md = render(_report("standard", with_b=False))
    assert "技術報告" in md and "95 分位延滯" in md and "近似(低信心)" in md
    assert "尚無 TDX 資料" in md and "進階指標" not in md


def test_full_tier_with_ab_paired_and_tdx():
    from detection.signal_report_md import render
    md = render(_report("full"), _paired(), _paired(), {"a_label": "舊參數", "b_label": "新參數"})
    for s in ("完整報告", "進階指標", "統計檢定", "Cohen's d", "分時段", "決策配對",
              "| 平均延滯 | 4.61 | 4.00 | -0.61 |", "small", "| A | 293 | 14（有車 14） | 9 | 270 | 0 | -16.1 | 225 | 179.7 | 33% | 控制器 5F03 秒數 |",
              "118.0", "P1", "舊參數", "新參數"):
        assert s in md, s
    # A/B 差值欄:B−A
    assert "| 平均排隊長度(公尺) | 7.91 | 7.00 | -0.91 | 實測 |" in md
