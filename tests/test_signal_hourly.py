"""逐時評估:每小時一列(配對/一致率/排隊/參數/成效),存表、當前小時標 partial、報告逐時段。"""
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _shadow_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE signal_shadow_log (id INTEGER PRIMARY KEY, ts TEXT, green_phase INT, green_elapsed REAL,
        queue_m_1 REAL, queue_m_2 REAL, ours TEXT, actual TEXT, agree INT, switch_gain REAL, keep_gain REAL, change_cost REAL,
        forced INT, blocked INT, reason TEXT, step_id INT, clearance INT, control_mode TEXT, flow_vpm_1 REAL, flow_vpm_2 REAL)""")
    conn.executemany("INSERT INTO signal_shadow_log(ts,green_phase,green_elapsed,queue_m_1,queue_m_2,ours,actual,agree,"
                     "switch_gain,keep_gain,change_cost,forced,blocked,reason,step_id,clearance,control_mode) "
                     "VALUES(?,?,?,?,?,?,?,?,?,?,?,0,0,'',1,0,'external_dynamic')", rows)
    conn.commit(); conn.close()


def test_hourly_compute_store_and_rows(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S
    db = str(tmp_path / "shadow.db")
    # 09:00 起 5 秒一筆 30 分鐘;一段分相1 綠燈 0..55s 我方 30s 判切(紅側有車);其餘續綠
    rows = []
    t = datetime(2026, 9, 4, 9, 0, 0)
    for i in range(360):
        el = (i % 12) * 5.0
        ph = 1 if (i // 12) % 2 == 0 else 2
        ours = "SWITCH" if (ph == 1 and el >= 30) else "KEEP"
        actual = "SWITCH" if el == 0 else "KEEP"
        agree = None if el == 0 else (1 if ours == "KEEP" else 0)
        q1, q2 = (0.0, 14.0) if ph == 1 else (0.0, 0.0)
        rows.append(((t + timedelta(seconds=5 * i)).isoformat(timespec="seconds"), ph, el, q1, q2, ours, actual, agree,
                     4.0 * el if ph == 1 else 0.0, 20.0, 3.5))
    _shadow_db(db, rows)
    monkeypatch.setattr(S, "_db", lambda: sqlite3.connect(db, timeout=5))
    monkeypatch.setattr(S, "_actual_runs_from_frames", lambda a, b: None)     # 沒抄錄框 → 退回取樣法
    monkeypatch.setattr(S, "_eval_window", lambda a, b: {1: [], 2: []})
    r = S._hourly_compute("2026-09-04T09")
    assert r["samples"] == 360 and r["change_cost_avg"] == 3.5 and r["source"] == "shadow_sampling_fallback"
    assert r["runs"] == 30 and r["earlier"] == 15 and r["earlier_meaningful"] == 15 and r["later"] == 0
    assert r["hold"] == 15 and r["agree_rate"] is not None and r["partial"] is False
    S._hourly_store(r)
    got = S.hourly_rows("2026-09-04", compute_missing=False)["rows"]
    assert len(got) == 1 and got[0]["hour"] == "2026-09-04T09" and got[0]["earlier"] == 15
    # 存過的不重算:把 compute 換成會炸的,還是拿得到那一列
    monkeypatch.setattr(S, "_hourly_compute", lambda h: (_ for _ in ()).throw(RuntimeError("不該重算")))
    got2 = S.hourly_rows("2026-09-04", compute_missing=False)["rows"]
    assert got2[0]["runs"] == 30


def test_hourly_current_hour_is_partial(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S
    db = str(tmp_path / "shadow.db"); _shadow_db(db, [])
    monkeypatch.setattr(S, "_db", lambda: sqlite3.connect(db, timeout=5))
    monkeypatch.setattr(S, "_actual_runs_from_frames", lambda a, b: None)
    monkeypatch.setattr(S, "_eval_window", lambda a, b: {1: [], 2: []})
    now = datetime.now().strftime("%Y-%m-%dT%H")
    r = S._hourly_compute(now)
    assert r["partial"] is True and r["samples"] == 0 and r["runs"] == 0
    # 一天缺很多小時:同步只算最近 2 個,其餘丟背景(pending 會 >0 或已算完)
    calls = []
    monkeypatch.setattr(S, "_hourly_compute", lambda k: (calls.append(k) or {"hour": k, "partial": False, "runs": 0}))
    out = S.hourly_rows(datetime.now().strftime("%Y-%m-%d"), compute_missing=True, max_sync=2)
    # 同步算的是最近兩個小時(當前小時在內);之後背景執行緒才接著算其餘的,
    # 所以只能看前兩個,不能看 calls[-1](那可能已經是回填的第一個)。
    assert now in calls[:2] and len(calls) >= 2
    assert "backfilling" in out and "rows" in out


def test_report_md_hourly_section():
    from detection.signal_report_md import render
    report = {"tier": "full", "a": {"since": "2026-09-04T09:00:00", "until": "2026-09-04T12:00:00",
                                    "all": {"cycles": 3, "core": {}, "advanced": {}}, "by_phase": {}}}
    hourly = {"A 基準": [{"hour": "2026-09-04T09", "samples": 720, "agree_rate": 91.3, "runs": 98, "earlier": 5,
                           "earlier_meaningful": 5, "same": 3, "hold": 90, "later": 0, "delta_avg": -16.1, "waste_sec": 80.0,
                           "delay_per_veh": 4.6, "queue_eval_m": 7.9, "throughput_vph": 1228.0, "change_cost_avg": 12.5, "partial": False},
                          {"hour": "2026-09-04T10", "samples": 100, "partial": True}]}
    md = render(report, None, None, {}, hourly)
    assert "## 六、逐時評估" in md and "| 09:00 | 720 | 91.3% | 98 | 5(5) | 3 | 90 | 0 | -16.1 | 80 | 4.60 | 7.90 | 1,228 | 12.50 |" in md
    assert "10:00（進行中）" in md
