"""相機斷線不可被當成「0 台車」(2026-09-15 WN-2 斷線事件)。

斷線時:戰情顯示「斷線」、不給量測值;聚合跳過該台;偵測不拿 Frigate 警示圖當畫面。
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


def _res(q, age=0.0, **kw):
    r = {"timestamp": datetime.fromtimestamp(time.time() - age).isoformat(),
         "estimated_queue_length_m": q, "occupancy": 0.3, "level": "medium", "level_name": "車多"}
    r.update(kw)
    return r


@pytest.fixture
def cr(monkeypatch):
    from api.routes import congestion as C
    from api.routes import signal_shadow as S
    d = {}
    monkeypatch.setattr(C, "congestion_results", d)
    monkeypatch.setattr(S, "ARRIVAL_FROM_EVENTS", False)
    monkeypatch.setitem(S.PHASE_CAMERAS, 2, [4, 5])
    return d


def test_cam_ok_rejects_no_frame_and_stale():
    from api.routes.signal_shadow import _cam_ok
    assert _cam_ok(_res(10.0))
    assert not _cam_ok({})
    assert not _cam_ok(_res(None, no_frame=True))
    assert not _cam_ok(_res(10.0, age=60))          # 連不上時最後一筆會一直掛著


def test_offline_camera_shows_disconnected_without_values(cr):
    from api.routes.signal_shadow import _camera_live
    cr[4] = _res(20.0)
    cr[5] = _res(None, no_frame=True, level="offline", level_name="斷線")
    by = {c["camera_id"]: c for c in _camera_live()}
    assert by[4]["online"] and by[4]["queue_m"] == 20.0
    assert by[5]["online"] is False and by[5]["level_name"] == "斷線"
    assert by[5]["queue_m"] is None and by[5]["occupancy"] is None


def test_stale_result_is_offline_not_last_value(cr):
    from api.routes.signal_shadow import _camera_live
    cr[4] = _res(20.0)
    cr[5] = _res(35.0, age=120)
    by = {c["camera_id"]: c for c in _camera_live()}
    assert by[5]["online"] is False and by[5]["queue_m"] is None


def test_phase_measure_skips_offline_camera(cr):
    from api.routes.signal_shadow import _phase_measure
    cr[4] = _res(20.0)
    cr[5] = _res(None, no_frame=True)
    m = _phase_measure(2)
    assert m["queue_m"] == 20.0 and m["cameras"] == 1


def test_phase_measure_all_offline_is_none_not_zero(cr):
    from api.routes.signal_shadow import _phase_measure
    cr[4] = _res(20.0, age=60)
    cr[5] = _res(None, no_frame=True)
    m = _phase_measure(2)
    assert m["queue_m"] is None and m["cameras"] == 0


def test_mark_no_frame_uses_none_not_zero(monkeypatch):
    from api.routes import congestion as C
    d = {}
    monkeypatch.setattr(C, "congestion_results", d)
    C._mark_no_frame(5)
    r = d[5]
    assert r["no_frame"] and r["level_name"] == "斷線"
    assert r["estimated_queue_length_m"] is None and r["vehicle_count"] is None
    C.draw_congestion(__import__("numpy").zeros((90, 160, 3), dtype="uint8"), r)   # 不可格式化失敗


def test_detection_fallback_skips_frigate_placeholder(monkeypatch):
    """Frigate 那台沒在收畫面(camera_fps=0)時,latest.jpg 是警示圖 —— 偵測不可取用。"""
    from api.routes import stream as ST
    calls = []

    class _R:
        status_code = 200
        content = b"x" * 5000

        def json(self):
            return {"cameras": {"cam_5": {"camera_fps": 0.0}, "cam_4": {"camera_fps": 1.0}}}

    def get(url, timeout=None):
        calls.append(url)
        return _R()
    monkeypatch.setattr(ST.requests, "get", get)
    monkeypatch.setitem(ST._frigate_fps_cache, "ts", 0.0)
    assert ST._try_frigate_snapshot("rtsp://10.42.40.25/x", camera_id=5, require_live=True) is None
    assert ST._try_frigate_snapshot("rtsp://10.42.40.24/x", camera_id=4, require_live=True)
    # 顯示用(預設)照舊取圖 —— 畫面上看到 Frigate 的「沒有畫面」是正確的
    assert ST._try_frigate_snapshot("rtsp://10.42.40.25/x", camera_id=5)
