"""驗收清單:決策參數皆實測、控制器回報週期、逐時評估無缺漏 —— 三組項目的判定。"""
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _plan(sat_src="measured", lost_src="控制器時制設定 5FC4(黃燈+全紅;5F03 框距 1.82 秒不足以逐秒量)",
          mpv_src="實測(停止線排隊÷停等車數,20082 筆)", fi=1.82):
    return {"constants": {"saturation_vph": 527.0, "saturation_source": sat_src, "saturation_measured_at": "2026-09-05T11:58:28",
                          "lost_time_sec": 5.0, "lost_time_source": lost_src, "meters_per_vehicle": 6.0,
                          "meters_per_vehicle_source": mpv_src, "frame_interval_sec": fi}}


def _hourly(n_done, partial=True):
    rows = [{"hour": f"2026-09-05T{h:02d}", "partial": False} for h in range(n_done)]
    if partial:
        rows.append({"hour": f"2026-09-05T{n_done:02d}", "partial": True})
    return {"rows": rows}


def test_all_measured_and_sparse_frames(monkeypatch):
    from api.routes import acceptance as A
    now_h = datetime.now().hour
    monkeypatch.setattr(A, "_shadow_constants", lambda: _plan()["constants"])
    monkeypatch.setattr(A, "_shadow_hourly", lambda day: _hourly(now_h)["rows"])
    items = {i["key"]: i for i in A._check_params()}
    assert items["param_saturation"]["state"] == A.PASS
    assert items["param_lost_time"]["state"] == A.PASS and "5FC4" in items["param_lost_time"]["evidence"]
    assert items["param_mpv"]["state"] == A.PASS
    assert items["frame_interval"]["state"] == A.WARN and "TransmitCycle" in items["frame_interval"]["evidence"]
    assert items["hourly_eval"]["state"] == A.PASS


def test_default_saturation_and_missing_hours_warn(monkeypatch):
    from api.routes import acceptance as A
    now_h = datetime.now().hour
    monkeypatch.setattr(A, "_shadow_constants", lambda: _plan(sat_src="default", lost_src="預設 5 s(未量到)", mpv_src="預設 7 m(未量到)", fi=1.0)["constants"])
    monkeypatch.setattr(A, "_shadow_hourly", lambda day: _hourly(max(0, now_h - 3))["rows"])   # 缺 3 個小時
    items = {i["key"]: i for i in A._check_params()}
    assert items["param_saturation"]["state"] == A.WARN
    assert items["param_lost_time"]["state"] == A.WARN
    assert items["param_mpv"]["state"] == A.WARN
    assert items["frame_interval"]["state"] == A.PASS
    assert items["hourly_eval"]["state"] == (A.WARN if now_h >= 3 else A.PASS)


def test_no_plan_is_unknown_not_pass(monkeypatch):
    """量不到一律 unknown,不當成通過 —— 缺資料和通過是兩件事。"""
    from api.routes import acceptance as A
    monkeypatch.setattr(A, "_shadow_constants", lambda: {})
    monkeypatch.setattr(A, "_shadow_hourly", lambda day: [])
    items = {i["key"]: i for i in A._check_params()}
    assert items["param_saturation"]["state"] == A.UNKNOWN
    assert items["frame_interval"]["state"] == A.UNKNOWN
    assert items["hourly_eval"]["state"] == A.UNKNOWN
