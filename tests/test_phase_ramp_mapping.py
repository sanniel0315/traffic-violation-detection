"""分相 ↔ 匝道的對應只有一份真相:ramp_timing_baseline.json 的 phases。

🛑 現場改號誌線路就會對調(2026-09-11、09-12、09-16 各一次)。任何地方寫死
   「分相1 = 上匝道」都會在下一次對調時漏掉,變成「畫面說下匝道、演算法拿
   上匝道的排隊」。這個測試守的是「改一個檔就整組搬」。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture
def LK():
    from detection import signal_timing_lookup as L
    return L


def test_roles_are_one_per_phase(LK):
    on, off = LK.phase_of_role("on_ramp"), LK.phase_of_role("off_ramp")
    assert {on, off} == {1, 2}
    assert LK.ramp_name(on) == "上匝道" and LK.ramp_name(off) == "下匝道"


def test_storage_and_priority_follow_the_ramp(LK):
    """儲車上限與主線保護是匝道的屬性:下匝道 600m 且優先,上匝道 210m。"""
    on, off = LK.phase_of_role("on_ramp"), LK.phase_of_role("off_ramp")
    assert LK.storage_limit_m(off) == 600 and LK.storage_limit_m(on) == 210
    assert LK.priority_phase() == off


def test_cameras_and_stopline_follow_the_ramp(LK):
    """上匝道相 = NE-1/NE-2(2,3)、停止線 NE-2;下匝道相 = WN-1/WN-2(4,5)、停止線 WN-2。"""
    from api.routes import signal_shadow as S
    on, off = LK.phase_of_role("on_ramp"), LK.phase_of_role("off_ramp")
    assert sorted(S.PHASE_CAMERAS[on]) == [2, 3] and sorted(S.PHASE_CAMERAS[off]) == [4, 5]
    assert S.PHASE_STOPLINE[on] == 3 and S.PHASE_STOPLINE[off] == 5
    assert S.PHASE_CAMERA[on] == 3 and S.PHASE_CAMERA[off] == 4          # 基準測點
    assert S.APPROACH_LEN_M[on] == 52.7 and S.APPROACH_LEN_M[off] == 16.0


def test_signal_heads_follow_the_ramp(LK):
    """燈頭綁匝道:1、2 號燈在下匝道,3、5、6 號燈在上匝道。"""
    from api.routes import signal_tc3 as T
    on, off = LK.phase_of_role("on_ramp"), LK.phase_of_role("off_ramp")
    ph = T._head_phases()
    assert ph[1] == off and ph[2] == off
    assert ph[3] == on and ph[5] == on and ph[6] == on


def test_plain_text_uses_the_baseline_names(LK):
    """白話依據講的匝道名要跟基準表一致,不可寫死。"""
    from api.routes import signal_shadow as S
    t = S._basis_plain({"green_phase": 1, "action": "KEEP", "green_elapsed": 20.0,
                        "queue_m_1": 10.0, "queue_m_2": 5.0})
    assert LK.ramp_name(1) in t and LK.ramp_name(2) in t


def test_swapping_the_json_swaps_everything(LK, monkeypatch, tmp_path):
    """把基準表的兩相對調,推導出來的相機/儲車/優先/燈頭要整組跟著換。"""
    import importlib
    import json
    base = json.loads(json.dumps(LK.load_baseline()))
    base["phases"]["1"], base["phases"]["2"] = base["phases"]["2"], base["phases"]["1"]
    path = tmp_path / "swapped.json"
    path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("RAMP_TIMING_BASELINE", str(path))
    LK.load_baseline.cache_clear()
    try:
        import api.routes.signal_shadow as S
        S = importlib.reload(S)
        on = LK.phase_of_role("on_ramp")
        assert sorted(S.PHASE_CAMERAS[on]) == [2, 3]          # 上匝道相永遠看 NE 兩台
        assert LK.storage_limit_m(LK.priority_phase()) == 600  # 主線保護永遠在下匝道
    finally:
        monkeypatch.delenv("RAMP_TIMING_BASELINE", raising=False)
        LK.load_baseline.cache_clear()
        importlib.reload(importlib.import_module("api.routes.signal_shadow"))
