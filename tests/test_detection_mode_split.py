#!/usr/bin/env python3
"""車流偵測 / 車速偵測 可以各自啟用與停止。

兩者共用同一支偵測 worker(YOLO 推論只跑一次),所以「停掉其中一項」不是把
執行緒殺掉,而是把該子功能的旗標關掉;兩項都關才真的停 worker。
這支測試釘住的行為:
  1. 只啟動車流時,車速不會跟著被啟用(反之亦然)
  2. 關掉其中一項,另一項與 worker 都還活著(串流不會斷)
  3. 兩項都關 → worker running=False
  4. 舊版 feature_state 只有 `detection` 一個 key 時,兩個子功能都跟著它 → 升級不改行為
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from api.utils import feature_state as fs  # noqa: E402
import api.routes.stream as st  # noqa: E402

# 狀態檔導到暫存路徑。要在 fixture 內指派 —— 其他測試模組也會蓋 fs.STATE_PATH,
# import 時設會被後 import 的模組搶走,整個檔案就跑在別人的狀態檔上。
_TMP = Path(tempfile.mkdtemp()) / "feature_state.json"

CAM_ID = 77


class _FakeCamera:
    def __init__(self):
        self.id = CAM_ID
        self.name = "測試攝影機"
        self.enabled = True
        self.detection_enabled = False
        self.status = "offline"
        self.detection_config = {}
        self.zones = []
        self.location = "測試路口"


class _FakeQuery:
    def __init__(self, cam):
        self._cam = cam

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._cam


class _FakeDB:
    def __init__(self, cam):
        self._cam = cam

    def query(self, _model):
        return _FakeQuery(self._cam)

    def commit(self):
        pass


def _fake_start(camera, modes=None):
    """不真的開執行緒,只把 detection_services 填成 worker 已啟動的樣子。"""
    _modes = {m: True for m in st.DETECTION_MODES} if modes is None else {
        m: bool((modes or {}).get(m)) for m in st.DETECTION_MODES
    }
    st.detection_services[camera.id] = {
        "running": True, "camera_name": camera.name, "modes": _modes,
    }
    return True


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(fs, "STATE_PATH", _TMP)
    if _TMP.exists():
        _TMP.unlink()
    st.detection_services.pop(CAM_ID, None)
    monkeypatch.setattr(st, "_start_detection_service", _fake_start)
    yield
    st.detection_services.pop(CAM_ID, None)


def _start(mode, db):
    return asyncio.run(st.start_detection(CAM_ID, mode=mode, db=db))


def _stop(mode, db):
    return asyncio.run(st.stop_detection(CAM_ID, mode=mode, db=db))


def _status():
    return asyncio.run(st.detection_status(CAM_ID))


# ---- 參數解析 ----

def test_parse_mode():
    assert st._parse_detection_mode("all") == st.DETECTION_MODES
    assert st._parse_detection_mode("") == st.DETECTION_MODES
    assert st._parse_detection_mode("traffic") == ("traffic",)
    assert st._parse_detection_mode("speed") == ("speed",)
    with pytest.raises(HTTPException) as e:
        st._parse_detection_mode("congestion")
    assert e.value.status_code == 400


# ---- 啟動 / 停止 ----

def test_start_traffic_does_not_start_speed():
    cam = _FakeCamera()
    db = _FakeDB(cam)
    _start("traffic", db)
    s = _status()
    assert s["traffic_running"] is True
    assert s["speed_running"] is False, "只啟動車流時車速不該被一起打開"
    assert fs.get_feature_enabled("detection_speed", CAM_ID, default=True) is False
    assert fs.get_feature_enabled("detection", CAM_ID) is True, "worker 要跑,主旗標應為 True"


def test_start_speed_on_running_worker_keeps_traffic():
    cam = _FakeCamera()
    db = _FakeDB(cam)
    _start("traffic", db)
    svc_before = st.detection_services[CAM_ID]
    _start("speed", db)
    s = _status()
    assert s["traffic_running"] is True and s["speed_running"] is True
    assert st.detection_services[CAM_ID] is svc_before, "worker 不該被重建(串流會斷)"


def test_stop_one_keeps_the_other_running():
    cam = _FakeCamera()
    db = _FakeDB(cam)
    _start("all", db)
    r = _stop("speed", db)
    assert r["status"] == "partial" and r["running"] is True
    s = _status()
    assert s["running"] is True, "另一項還開著,worker 要繼續跑"
    assert s["traffic_running"] is True
    assert s["speed_running"] is False
    assert cam.detection_enabled is True


def test_stop_both_stops_worker():
    cam = _FakeCamera()
    db = _FakeDB(cam)
    _start("all", db)
    _stop("speed", db)
    r = _stop("traffic", db)
    assert r["status"] == "stopped" and r["running"] is False
    assert st.detection_services[CAM_ID]["running"] is False
    assert cam.detection_enabled is False
    assert fs.get_feature_enabled("detection", CAM_ID, default=True) is False


def test_stop_mode_on_never_started_camera_does_not_enable_the_other():
    """沒啟動過的攝影機被「只停車流」,不能反而把車速標成啟用。"""
    cam = _FakeCamera()
    db = _FakeDB(cam)
    r = _stop("traffic", db)
    assert r["running"] is False
    assert fs.get_feature_enabled("detection", CAM_ID, default=True) is False


# ---- 狀態回傳 ----

def test_status_without_service():
    s = _status()
    assert s == {"running": False,
                 "modes": {"traffic": False, "speed": False},
                 "traffic_running": False, "speed_running": False}


def test_legacy_worker_without_modes_counts_as_both_on():
    """升級前就在跑的 worker 沒有 modes key,要視為兩項都開,不能顯示成全停。"""
    st.detection_services[CAM_ID] = {"running": True, "camera_name": "舊"}
    s = _status()
    assert s["traffic_running"] is True and s["speed_running"] is True


# ---- 重啟還原 ----

def test_resume_intent_falls_back_to_legacy_master_key():
    fs.set_feature_state("detection", CAM_ID, True)
    modes = st.resolve_detection_modes_intent(CAM_ID, default=False)
    assert modes == {"traffic": True, "speed": True}, "舊設定檔沒有子項時要跟著主旗標"


def test_resume_intent_uses_sub_keys_when_present():
    fs.set_feature_state("detection", CAM_ID, True)
    fs.set_feature_state("detection_traffic", CAM_ID, True)
    fs.set_feature_state("detection_speed", CAM_ID, False)
    modes = st.resolve_detection_modes_intent(CAM_ID, default=True)
    assert modes == {"traffic": True, "speed": False}
