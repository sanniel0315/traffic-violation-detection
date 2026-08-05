#!/usr/bin/env python3
"""攝影機刪除後,功能啟停狀態不可殘留給下一台重用同 id 的攝影機。

回歸案例:上傳影片新建的攝影機拿到被回收的 id(該 id 上次是 false),
get_feature_enabled 直接回 false、default=True 用不到,watchdog 不拉起它,
畫面永遠停在第一幀不會播。
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.utils import feature_state as fs  # noqa: E402

# 把狀態檔導到暫存路徑，不動到真實設定
_TMP = Path(tempfile.mkdtemp()) / "feature_state.json"
fs.STATE_PATH = _TMP


def _reset():
    if _TMP.exists():
        _TMP.unlink()


def test_clear_removes_all_features():
    _reset()
    fs.set_feature_state("detection", 4, False)
    fs.set_feature_state("congestion", 4, False)
    fs.set_feature_state("lpr", 4, True)
    fs.set_feature_state("detection", 5, True)

    fs.clear_camera_state(4)

    assert fs.get_feature_enabled("detection", 4, default=True) is True, \
        "清除後應回落到 default，而不是舊的 False"
    assert fs.get_feature_enabled("congestion", 4, default=True) is True
    assert fs.get_feature_enabled("lpr", 4, default=False) is False
    assert fs.get_feature_enabled("detection", 5, default=False) is True, \
        "不可誤刪其他攝影機的狀態"


def test_clear_on_absent_id_is_safe():
    _reset()
    fs.set_feature_state("detection", 1, True)
    fs.clear_camera_state(99)          # 不存在的 id
    assert fs.get_feature_enabled("detection", 1, default=False) is True


def test_stale_false_blocks_default_before_fix():
    """證明問題本體:殘留的 False 會蓋掉 default=True。"""
    _reset()
    fs.set_feature_state("detection", 4, False)
    assert fs.get_feature_enabled("detection", 4, default=True) is False, \
        "這就是 bug 的成因 — 殘留值優先於 default"
    fs.clear_camera_state(4)
    assert fs.get_feature_enabled("detection", 4, default=True) is True


def test_explicit_set_overrides_stale():
    """建立攝影機時明確寫入,可蓋掉既有殘留值。"""
    _reset()
    fs.set_feature_state("detection", 4, False)      # 前一台留下的
    fs.set_feature_state("detection", 4, True)       # create_camera 明確寫入
    assert fs.get_feature_enabled("detection", 4, default=False) is True


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
