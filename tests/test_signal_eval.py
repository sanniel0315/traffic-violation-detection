"""成效評估模組:三等級指標、Welch t-test / Cohen's d、逐週期指標、method 標示。"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_welch_t_and_cohen_d_known_values():
    from detection.signal_eval import welch_t
    a = [10, 11, 9, 10, 12, 8, 10, 11]
    b = [14, 15, 13, 16, 14, 15, 13, 14]
    r = welch_t(a, b)
    assert r["diff"] > 0 and r["p"] < 0.001 and r["effect"] == "large" and r["cohen_d"] > 2
    same = welch_t(a, a)
    assert same["cohen_d"] == 0.0 and same["p"] > 0.99
    assert welch_t([1], [2, 3])["p"] is None            # 樣本不足要講,不能瞎算


def _cycles(n, start=1000.0, green=30.0, cyc=70.0):
    return [{"phase": 1, "start": start + i * cyc, "green_end": start + i * cyc + green,
             "end": start + (i + 1) * cyc, "green_sec": green} for i in range(n)]


def test_per_cycle_metrics_integrates_stopped_delay_and_counts_passes():
    from detection.signal_eval import per_cycle_metrics
    cyc = _cycles(2)
    # 每 5 秒一筆:第一個週期停等 2 台持續 20 秒(4 筆),其餘 0;第二週期全 0
    cong = []
    t = 1000.0
    while t < 1140.0:
        stopped = 2.0 if (1000.0 <= t < 1020.0) else 0.0
        cong.append((t, stopped, stopped + 1.0, stopped * 7.0))
        t += 5.0
    passes = [(1000.0 + k * 3.0, 40.0) for k in range(10)] + [(1070.0 + k * 7.0, 35.0) for k in range(5)]
    rows = per_cycle_metrics(cyc, cong, passes, storage_m=210.0, approach_len_m=52.7)
    assert len(rows) == 2
    r0, r1 = rows
    assert r0["passes"] == 10 and r1["passes"] == 5
    assert r0["delay_veh_sec"] == 40.0            # 2 台 × 20 秒
    assert r0["delay_per_veh"] == 4.0             # 40 車·秒 ÷ 10 台
    assert r1["delay_per_veh"] == 0.0
    assert r0["queue_max_m"] == 14.0 and r0["spillback"] is False
    assert r0["throughput_vph"] == round(10 * 3600 / 70.0, 1)
    assert r0["travel_sec"] is not None and r0["travel_sec"] > 4.0    # 距離/車速 + 停等延滯


def test_summarize_tiers_and_method_labels():
    from detection.signal_eval import per_cycle_metrics, summarize_cycles
    cyc = _cycles(3)
    cong = [(1000.0 + 5 * k, 1.0, 2.0, 7.0) for k in range(42)]
    passes = [(1000.0 + 2.0 * k, 30.0) for k in range(100)]
    rows = per_cycle_metrics(cyc, cong, passes, 210.0, 52.7)
    mn = summarize_cycles(rows, "min")
    assert set(mn["core"]) == {"avg_delay_sec", "avg_queue_m", "throughput_vph"} and mn["advanced"] == {}
    st = summarize_cycles(rows, "standard")
    assert {"avg_travel_sec", "avg_stops", "delay_p95_sec"} <= set(st["core"])
    fu = summarize_cycles(rows, "full")
    assert len(fu["advanced"]) == 7
    # 近似值必須標 approx,實測值標 measured —— 報告不可以混
    assert fu["core"]["avg_delay_sec"]["method"] == "approx"
    assert fu["core"]["avg_queue_m"]["method"] == "measured"
    assert fu["core"]["throughput_vph"]["method"] == "measured"
    assert fu["core"]["avg_travel_sec"]["confidence"] == "low"


def test_compare_returns_test_per_metric():
    from detection.signal_eval import per_cycle_metrics, compare
    cyc = _cycles(6)
    # 每週期給一點變異,否則兩組都是常數,t 無定義(那條路徑另有 note,不在這裡驗)
    cong_a = [(1000.0 + 5 * k, 3.0 + (k // 14) % 2, 4.0 + (k // 14) % 2, 21.0 + 3 * ((k // 14) % 3)) for k in range(84)]
    cong_b = [(1000.0 + 5 * k, 1.0 + (k // 14) % 2, 2.0 + (k // 14) % 2, 7.0 + 2 * ((k // 14) % 3)) for k in range(84)]
    passes = [(1000.0 + 2.0 * k, 30.0) for k in range(210)]
    ra = per_cycle_metrics(cyc, cong_a, passes, 210.0, 52.7)
    rb = per_cycle_metrics(cyc, cong_b, passes, 210.0, 52.7)
    c = compare(ra, rb)
    assert c["queue_avg_m"]["diff"] < 0 and c["queue_avg_m"]["p"] is not None
    assert c["delay_per_veh"]["diff"] < 0
