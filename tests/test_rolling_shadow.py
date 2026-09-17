"""rolling horizon 進影子模式:只算不送,而且要用各相自己的最小綠。

🛑 為什麼要影子跑:2026-09-16 模擬對照顯示現行瞬時成本式贏固定時制 68%、
   贏 Webster 72%,但輸感應控制 97% —— 差別就在「看不看得到未來」。
   線上先只記錄,累積夠多樣本再談要不要換掉決策式。
"""
import os
import json
import sqlite3

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _prep(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S

    db = tmp_path / "shadow.db"
    monkeypatch.setattr(S, "_db", lambda: sqlite3.connect(str(db)))
    monkeypatch.setattr(S, "_mpv", lambda: 6.0)
    monkeypatch.setattr(S, "_sat_for", lambda ph: 1800.0)
    monkeypatch.setattr(S, "_lost_time_for", lambda ph: 5.0)
    return S, db


def test_records_a_row_and_never_sends(tmp_path, monkeypatch):
    S, db = _prep(tmp_path, monkeypatch)

    sent = []
    monkeypatch.setattr(S, "_daemon_post", lambda *a, **k: sent.append(a) or {})
    monkeypatch.setattr(S, "_actuate", lambda *a, **k: sent.append(a))

    S._rolling_shadow_record(
        g_no=1, green_elapsed=12.0, min_green_map={1: 10.0, 2: 20.0},
        max_green=100.0, q_map={1: 30.0, 2: 90.0}, f_map={1: 2.0, 2: 12.0},
        live={"step_id": 1})

    rows = list(sqlite3.connect(str(db)).execute(
        "SELECT rule,green_phase,would,reason FROM signal_rule_shadow"))
    assert len(rows) == 1 and rows[0][0] == "rolling"
    r = json.loads(rows[0][3])
    assert "best_sec" in r and r["horizon_sec"] > 0
    assert not sent, "影子評估不可以下發任何命令"


def test_skips_during_clearance(tmp_path, monkeypatch):
    """清道(黃燈/全紅)期間不評估 —— 那時候沒有「要不要續綠」這個問題。"""
    S, db = _prep(tmp_path, monkeypatch)
    S._rolling_shadow_record(1, 12.0, {1: 10.0, 2: 20.0}, 100.0,
                             {1: 30.0, 2: 90.0}, {1: 2.0, 2: 12.0},
                             {"step_id": 2, "clearance": True})
    conn = sqlite3.connect(str(db))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    assert "signal_rule_shadow" not in tables or not list(
        conn.execute("SELECT 1 FROM signal_rule_shadow"))


def test_uses_each_phase_own_min_green(tmp_path, monkeypatch):
    """🛑 兩相最小綠不同(現場 10 / 20 秒)。共用綠側那一個會讓模型以為對向
       也能 10 秒切走,最佳時機整個偏移 —— 這裡直接驗設定物件收到兩個值。"""
    S, _ = _prep(tmp_path, monkeypatch)
    seen = {}

    def _fake(rate_fn, sat, cfg, t0, green, q0, elapsed, horizon_sec=120.0, **kw):
        seen["min_green"] = dict(cfg.min_green_sec)
        return {"best_sec": 0.0, "switch_now": True, "delay_by_sec": {}}

    import detection.signal_rolling as R
    monkeypatch.setattr(R, "plan_extra_green", _fake)
    S._rolling_shadow_record(1, 12.0, {1: 10.0, 2: 20.0}, 100.0,
                             {1: 30.0, 2: 90.0}, {1: 2.0, 2: 12.0}, {"step_id": 1})
    assert seen["min_green"] == {1: 10.0, 2: 20.0}


def test_failure_does_not_break_the_loop(tmp_path, monkeypatch):
    """影子評估壞掉只能記錯誤,不可以往上拋 —— 主迴圈還要繼續控制。"""
    S, _ = _prep(tmp_path, monkeypatch)
    import detection.signal_rolling as R

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(R, "plan_extra_green", _boom)
    S._rolling_shadow_record(1, 12.0, {1: 10.0, 2: 20.0}, 100.0,
                             {1: 30.0, 2: 90.0}, {1: 2.0, 2: 12.0}, {"step_id": 1})
    assert "boom" in (S._stats.get("rolling_shadow_error") or "")
