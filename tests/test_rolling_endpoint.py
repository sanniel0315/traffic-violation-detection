"""預測要看得到也查得到:/rolling 回傳即時建議、逐筆紀錄與一致率。

🛑 使用者 2026-09-18:「預測的要有地方顯示動態跟紀錄」。
🛑 一致率不是成效:現在下發的是成本式,rolling 只在旁邊算 ——
   說中只代表兩者看法相同。這一點必須寫在回傳裡,不能只寫在文件。
"""
import os
import json
import sqlite3
import time
from datetime import datetime, timedelta

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _seed(db, now):
    c = sqlite3.connect(str(db))
    c.execute("""CREATE TABLE signal_rule_shadow (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, epoch REAL, rule TEXT,
        green_phase INTEGER, green_elapsed REAL, step_id INTEGER, would INTEGER,
        reason TEXT, queue_green REAL, queue_red REAL)""")
    c.execute("""CREATE TABLE signal_shadow_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, green_phase INTEGER)""")

    def iso(t):
        return datetime.fromtimestamp(t).isoformat(timespec="seconds")

    # 預測:t-100 說「現在切」,t-60 說「再給 20 秒」
    c.execute("INSERT INTO signal_rule_shadow(ts,epoch,rule,green_phase,green_elapsed,"
              "step_id,would,reason,queue_green,queue_red) VALUES(?,?,?,?,?,?,?,?,?,?)",
              (iso(now - 100), now - 100, "rolling", 1, 30.0, 1, 1,
               json.dumps({"best_sec": 0.0, "horizon_sec": 100.0}), 10.0, 40.0))
    c.execute("INSERT INTO signal_rule_shadow(ts,epoch,rule,green_phase,green_elapsed,"
              "step_id,would,reason,queue_green,queue_red) VALUES(?,?,?,?,?,?,?,?,?,?)",
              (iso(now - 60), now - 60, "rolling", 2, 12.0, 1, 0,
               json.dumps({"best_sec": 20.0, "horizon_sec": 100.0}), 30.0, 5.0))
    # 實際換相:t-95(命中前者)、t-35(後者實際跑了 25 秒 → 誤差 5 秒)
    for t, ph in ((now - 300, 1), (now - 95, 2), (now - 35, 1)):
        c.execute("INSERT INTO signal_shadow_log(ts,green_phase) VALUES(?,?)", (iso(t), ph))
    c.commit()
    c.close()


def test_rolling_endpoint_reports_live_history_and_agreement(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S

    now = time.time()
    db = tmp_path / "s.db"
    _seed(db, now)
    monkeypatch.setattr(S, "_db", lambda: sqlite3.connect(str(db)))

    out = S.rolling_shadow(minutes=60, limit=100, _user="t")
    assert out["available"] is True
    assert out["samples"] == 2
    # 動態:最新一筆就是「目前建議」
    assert out["live"]["advice"] in ("現在就切", "再給 20 秒")
    # 紀錄:每一筆都要帶匝道名(主鍵是匝道)與實際多久後換相
    assert all(i.get("ramp") for i in out["items"])
    assert any(i["actual_switch_in_sec"] is not None for i in out["items"])
    # 一致率
    acc = out["accuracy"]
    assert acc["said_switch_now"] == 1 and acc["hit"] == 1
    assert acc["hit_rate_pct"] == 100.0
    # 種子資料的秒數會被 isoformat 截掉小數,容忍 ±0.5 秒
    assert abs(acc["keep_err_median_sec"] - 5.0) <= 0.5
    assert "一致率" in acc["note"] and "成效" in acc["note"]
    assert "不下發" in out["caveat"]


def test_empty_history_is_not_an_error(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S

    db = tmp_path / "empty.db"
    c = sqlite3.connect(str(db))
    c.execute("""CREATE TABLE signal_rule_shadow (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, epoch REAL, rule TEXT,
        green_phase INTEGER, green_elapsed REAL, step_id INTEGER, would INTEGER,
        reason TEXT, queue_green REAL, queue_red REAL)""")
    c.execute("CREATE TABLE signal_shadow_log (id INTEGER PRIMARY KEY, ts TEXT, green_phase INTEGER)")
    c.commit(); c.close()
    monkeypatch.setattr(S, "_db", lambda: sqlite3.connect(str(db)))

    out = S.rolling_shadow(minutes=60, limit=10, _user="t")
    assert out["available"] is True and out["samples"] == 0
    assert out["live"] is None
    assert out["accuracy"]["hit_rate_pct"] is None
