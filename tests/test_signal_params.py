"""決策參數實測落地:損失時間(控制器 5F03 清道秒數)、每車佔用長度(停止線排隊÷停等車數)。
量到才更新、超出範圍不採信、落地檔重啟載回、沒量到退回預設。"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reset(S):
    S._measured_params.update({"lost_time_sec": {}, "meters_per_vehicle": None, "ts": None,
                               "source": "default", "samples": {}})


def test_refresh_measures_lost_time_and_mpv(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S
    _reset(S)
    monkeypatch.setattr(S, "_PARAMS_FILE", str(tmp_path / "p.json"))
    # 控制器框重建的段:分相1 清道 5s(30 段)、分相2 清道 6s(30 段)、一段 40s 的異常值要被丟掉
    segs = []
    for i in range(30):
        segs.append({"phase": 1, "start": 0, "green_end": 100, "end": 105.0 + (0.4 if i % 2 else 0)})
        segs.append({"phase": 2, "start": 0, "green_end": 100, "end": 106.0})
    segs.append({"phase": 1, "start": 0, "green_end": 100, "end": 140.0})
    monkeypatch.setattr(S, "_actual_runs_from_frames", lambda a, b: segs)
    monkeypatch.setattr(S, "_frames_spacing", lambda a, b: 1.0)        # 每秒一框才准用框量
    # 停止線相機:排隊 21m/3 台、14m/2 台、一筆 60m/2 台(30 m/車,超出上限要丟)
    cong = {3: [(0, 3.0, 4.0, 21.0)] * 60 + [(0, 2.0, 3.0, 14.0)] * 60 + [(0, 2.0, 2.0, 60.0)] * 5,
            5: [(0, 2.0, 2.0, 14.0)] * 20 + [(0, 1.0, 1.0, 7.0)] * 50}     # 停等 1 台不算
    from detection import signal_eval as E
    monkeypatch.setattr(E, "load_congestion", lambda db, cams, a, b: cong)
    S._refresh_params()
    p = S._measured_params
    assert p["source"] == "measured" and p["lost_time_sec"][1] == 5.2 and p["lost_time_sec"][2] == 6.0
    assert p["meters_per_vehicle"] == 7.0 and p["samples"]["mpv"] == 140
    assert S._lost_time_for(1) == 5.2 and S._lost_time_for(2) == 6.0 and S._mpv() == 7.0
    # 落地 + 清空 + 載回
    assert json.load(open(tmp_path / "p.json", encoding="utf-8"))["meters_per_vehicle"] == 7.0
    _reset(S)
    assert S._lost_time_for(1) == 5.0 and S._mpv() == 7.0          # 預設(引擎常數)
    assert S._load_params() is True and S._measured_params["source"] == "measured(restored)"
    assert S._lost_time_for(2) == 6.0


def test_refresh_keeps_previous_when_too_few_samples(monkeypatch, tmp_path):
    from api.routes import signal_shadow as S
    _reset(S)
    monkeypatch.setattr(S, "_PARAMS_FILE", str(tmp_path / "p.json"))
    S._measured_params["lost_time_sec"] = {1: 4.8}
    monkeypatch.setattr(S, "_frames_spacing", lambda a, b: 1.0)
    monkeypatch.setattr(S, "_actual_runs_from_frames", lambda a, b: [{"phase": 1, "start": 0, "green_end": 10, "end": 15}] * 5)
    from detection import signal_eval as E
    monkeypatch.setattr(E, "load_congestion", lambda db, cams, a, b: {3: [(0, 2.0, 2.0, 14.0)] * 10})
    S._refresh_params()
    assert S._measured_params["lost_time_sec"][1] == 4.8 and S._measured_params["meters_per_vehicle"] is None


def test_load_rejects_out_of_range(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S
    _reset(S)
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"lost_time_sec": {"1": 40.0}, "meters_per_vehicle": 30.0}), encoding="utf-8")
    monkeypatch.setattr(S, "_PARAMS_FILE", str(f))
    assert S._load_params() is False and S._mpv() == 7.0 and S._lost_time_for(1) == 5.0


def test_lost_time_falls_back_to_timing_plan_when_frames_are_sparse(tmp_path, monkeypatch):
    """2026-09-05 教訓:5F03 變成每 6~7 秒一框後,用框量出 8.5 秒(真值 3+2=5)。
    框距 >1.5 秒不可以用框量,改用控制器回報的時制設定(5FC4 黃燈+全紅)。"""
    from api.routes import signal_shadow as S
    _reset(S)
    monkeypatch.setattr(S, "_PARAMS_FILE", str(tmp_path / "p.json"))
    monkeypatch.setattr(S, "_frames_spacing", lambda a, b: 6.6)
    monkeypatch.setattr(S, "_actual_runs_from_frames", lambda a, b: [{"phase": 1, "start": 0, "green_end": 100, "end": 108.5}] * 50)
    import detection.signal_timing_lookup as L
    monkeypatch.setattr(L, "plan_params", lambda pid: {"yellow": 3, "all_red": 2})
    monkeypatch.setattr(L, "current_base_plan", lambda: 36)
    from detection import signal_eval as E
    monkeypatch.setattr(E, "load_congestion", lambda db, cams, a, b: {})
    S._refresh_params()
    assert S._lost_time_for(1) == 5.0 and S._lost_time_for(2) == 5.0
    assert S._measured_params["samples"]["lost_time_source"] == "timing_plan_5FC4"
    assert S._measured_params["samples"]["frame_interval_sec"] == 6.6
