#!/usr/bin/env python3
"""攝影機改名後,報表的「設備編號」必須跟著現行名稱,不可分裂成多個設備。

回歸案例:104 上同一台攝影機因為改過名,VD 報表設備編號下拉同時出現
台24 / 24_01 / 24_兩車道0708 三個項目 —— 聚合表把 camera_name 當欄位
存進每一列(寫入當下的快照),舊時間桶永遠帶著舊名字。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.utils.report_aggregation import _device_id_for  # noqa: E402


class Agg:
    """模擬聚合列:camera_id + 寫入當下的名稱快照。"""

    def __init__(self, camera_id, camera_name):
        self.camera_id = camera_id
        self.camera_name = camera_name


# 現行 cameras 表：cam2 現在叫「24_兩車道0708」
LIVE = {
    2: {"camera_name": "24_兩車道0708"},
    3: {"camera_name": "24_三車道_0710"},
}


def test_old_snapshots_resolve_to_current_name():
    """同一台攝影機的三個歷史名稱,都要解析成同一個現行名稱。"""
    names = {
        _device_id_for(Agg(2, "台24"), LIVE),
        _device_id_for(Agg(2, "24_01"), LIVE),
        _device_id_for(Agg(2, "24_兩車道0708"), LIVE),
    }
    assert names == {"24_兩車道0708"}, f"應合併為單一設備，實際 {names}"


def test_different_cameras_stay_separate():
    a = _device_id_for(Agg(2, "台24"), LIVE)
    b = _device_id_for(Agg(3, "24_02"), LIVE)
    assert a != b, "不同攝影機不可被合併"
    assert a == "24_兩車道0708" and b == "24_三車道_0710"


def test_deleted_camera_falls_back_to_snapshot():
    """攝影機已刪除(查不到 id)→ 退回快照名稱,至少看得出是哪一台。"""
    assert _device_id_for(Agg(99, "已刪除的攝影機"), LIVE) == "已刪除的攝影機"


def test_deleted_camera_without_snapshot():
    assert _device_id_for(Agg(99, ""), LIVE) == "cam_99"
    assert _device_id_for(Agg(99, None), LIVE) == "cam_99"


def test_bad_camera_id_is_safe():
    assert _device_id_for(Agg(None, "某台"), LIVE) == "某台"
    assert _device_id_for(Agg("abc", "某台"), LIVE) == "某台"


def test_live_name_wins_over_snapshot():
    """關鍵:現行名稱優先於快照(修正前是相反的)。"""
    assert _device_id_for(Agg(2, "舊名字"), LIVE) == "24_兩車道0708"


def test_blank_live_name_falls_back():
    """現行名稱是空字串時不可回傳空,要退回快照。"""
    live = {2: {"camera_name": "   "}}
    assert _device_id_for(Agg(2, "快照名"), live) == "快照名"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{'全部通過' if not failed else str(failed) + ' 項失敗'}")
    sys.exit(1 if failed else 0)
