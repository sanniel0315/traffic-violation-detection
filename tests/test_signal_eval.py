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


def test_report_window_runs_without_name_errors(tmp_path, monkeypatch):
    """★ 2026-09-05 教訓:_eval_window 引用的 _VIOL_DB 沒定義,部署後端點直接 500。
    單元測試只測 signal_eval 純函式抓不到 —— 這裡真的呼叫 _eval_window,
    指向空的臨時 DB,任何 NameError / 匯入錯都會在這裡炸。"""
    import sqlite3
    from api.routes import signal_shadow as S
    db = tmp_path / "v.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE congestion_samples (camera_id INT, created_at TEXT, stopped_vehicle_count INT, vehicle_count INT, estimated_queue_length_m REAL, is_overall INT)")
    conn.execute("CREATE TABLE traffic_events (camera_id INT, created_at TEXT, speed_kmh REAL, direction TEXT)")
    conn.commit(); conn.close()
    monkeypatch.setattr(S, "_VIOL_DB", str(db))
    monkeypatch.setattr(S, "_actual_runs_from_frames", lambda a, b: [])
    out = S._eval_window("2026-09-04T09:00:00", "2026-09-04T09:10:00")
    assert out == {1: [], 2: []}


def test_load_passes_counts_only_exit_rows(tmp_path):
    """traffic_events 混了每幀偵測列與別的進場道;只有 EXIT 是一車一筆的通過。"""
    import sqlite3
    from detection.signal_eval import load_passes
    db = tmp_path / "v.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE traffic_events (camera_id INT, created_at TEXT, "
                 "speed_kmh REAL, direction TEXT)")
    rows = []
    for i in range(5):
        rows.append((3, "2026-09-05 01:%02d:00" % i, 30.0, "EXIT"))
    for i in range(40):
        rows.append((3, "2026-09-05 01:%02d:%02d" % (i // 6, i % 6), 30.0, "INOUT"))
    for i in range(12):
        rows.append((3, "2026-09-05 01:00:%02d" % i, 30.0, "straight"))
    for i in range(7):
        rows.append((3, "2026-09-05 01:%02d:30" % i, 30.0, "IN"))
    conn.executemany("INSERT INTO traffic_events VALUES (?,?,?,?)", rows)
    conn.commit(); conn.close()
    got = load_passes(str(db), [3], "2026-09-05T09:00:00", "2026-09-05T11:00:00")
    assert len(got[3]) == 5, "只能算 EXIT,不可把每幀偵測列或別的進場道算進來"


def test_load_passes_empty_when_no_exit_rows(tmp_path):
    """沒有 EXIT 列就回空(量不到),不可退回全部列湊數字。"""
    import sqlite3
    from detection.signal_eval import load_passes
    db = tmp_path / "v2.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE traffic_events (camera_id INT, created_at TEXT, "
                 "speed_kmh REAL, direction TEXT)")
    conn.executemany("INSERT INTO traffic_events VALUES (?,?,?,?)",
                     [(3, "2026-09-05 01:00:%02d" % i, 30.0, "INOUT") for i in range(20)])
    conn.commit(); conn.close()
    got = load_passes(str(db), [3], "2026-09-05T09:00:00", "2026-09-05T11:00:00")
    assert got[3] == []


def test_load_flow_reads_shadow_log_flow_columns(tmp_path):
    """使用者:「in out 不看,我們看流量」—— 流量取 signal_shadow_log 的 flow_vpm。"""
    import sqlite3
    from detection.signal_eval import load_flow
    db = tmp_path / "s.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE signal_shadow_log (ts TEXT, flow_vpm_1 REAL, flow_vpm_2 REAL)")
    conn.executemany("INSERT INTO signal_shadow_log VALUES (?,?,?)",
                     [("2026-09-05T17:%02d:00" % i, 20.0 + i, 10.0) for i in range(10)])
    conn.commit(); conn.close()
    f = load_flow(str(db), 1, "2026-09-05T17:00:00", "2026-09-05T18:00:00")
    assert f["samples"] == 10
    assert abs(f["flow_vpm"] - 24.5) < 1e-6
    assert abs(f["throughput_vph"] - 24.5 * 60) < 1e-6
    assert "caveat" in f, "已知的高估/低估方向要一起帶出去"
    f2 = load_flow(str(db), 2, "2026-09-05T17:00:00", "2026-09-05T18:00:00")
    assert abs(f2["flow_vpm"] - 10.0) < 1e-6


def test_load_flow_empty_window_returns_none_not_zero(tmp_path):
    import sqlite3
    from detection.signal_eval import load_flow
    db = tmp_path / "s2.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE signal_shadow_log (ts TEXT, flow_vpm_1 REAL, flow_vpm_2 REAL)")
    conn.commit(); conn.close()
    f = load_flow(str(db), 1, "2026-09-05T17:00:00", "2026-09-05T18:00:00")
    assert f["samples"] == 0 and f["flow_vpm"] is None and f["throughput_vph"] is None


def test_flow_capacity_check_flags_impossible_flow():
    """流量高於容量就是有一邊量錯,必須標出來而不是照登。"""
    from detection.signal_eval import flow_capacity_check
    bad = flow_capacity_check(1596.0, 1184.0, 0.45)      # 09-05 17-20 分相1 實況
    assert bad["checked"] and bad["over_capacity"] is True
    assert bad["ratio"] > 2.9
    ok = flow_capacity_check(400.0, 1184.0, 0.45)
    assert ok["checked"] and ok["over_capacity"] is False
    none = flow_capacity_check(None, 1184.0, 0.45)
    assert none["checked"] is False
