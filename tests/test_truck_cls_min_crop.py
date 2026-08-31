"""細分類最小框門檻:太小的框不可以送進 GPU。

遠處小框放大到 imgsz 只是插值出來的糊圖,實測信心必定低於門檻 → 回 unknown,
而 unknown 不進快取 → 下一幀再算一次。這條門檻就是要把這些必定白花的
GPU 呼叫擋在外面(2026-08-31 87 現場:細分類吃掉 52% GPU、分析率掉到 0.3 fps)。
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tc():
    """載入 truck_classifier,不真的載 YOLO。"""
    # 🛑 別的測試檔可能已經先塞過同名 stub(缺我們要的屬性),所以是「補屬性」
    #    不是「不存在才建」—— 否則 pytest 的執行順序會決定這個測試會不會爆。
    for name in ("ultralytics", "model_paths"):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        if name == "ultralytics":
            if not hasattr(mod, "YOLO"):
                mod.YOLO = object
        else:
            if not hasattr(mod, "get_model_dir"):
                mod.get_model_dir = lambda *a, **k: "."
            if not hasattr(mod, "get_truck_cls_model_path"):
                mod.get_truck_cls_model_path = lambda *a, **k: "x.pt"
    import detection.truck_classifier as m
    return m


class _SpyModel:
    """記錄 predict 有沒有被呼叫。"""

    def __init__(self):
        self.calls = 0

    def predict(self, *a, **k):
        self.calls += 1
        raise AssertionError("小框不應該送進 GPU")


def _make(tc, min_px=48):
    obj = tc.TruckClassifier.__new__(tc.TruckClassifier)
    obj.model = _SpyModel()
    obj.conf_threshold = 0.5
    obj.imgsz = 224
    obj.device = "cpu"
    obj.class_names = {0: "non_truck"}
    tc.MIN_CROP_PX = min_px
    return obj


def _bbox(w, h):
    return {"x1": 100, "y1": 100, "x2": 100 + w, "y2": 100 + h}


@pytest.mark.parametrize("w,h", [(20, 20), (47, 200), (200, 47), (10, 500)])
def test_small_bbox_skips_gpu(tc, w, h):
    """小框直接回 unknown,而且完全不呼叫模型。"""
    obj = _make(tc)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = obj.classify(frame, _bbox(w, h))
    assert res["class_name"] == "unknown"
    assert obj.model.calls == 0


def test_threshold_uses_short_side(tc):
    """10x500 面積夠大但短邊只有 10px —— 一樣判不出來,要擋掉。"""
    obj = _make(tc)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert obj.classify(frame, _bbox(10, 500))["class_name"] == "unknown"
    assert obj.model.calls == 0


def test_large_bbox_still_classified(tc):
    """門檻只擋小框,不可以改變原本會被分類的框的行為。"""
    obj = _make(tc)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        obj.classify(frame, _bbox(120, 90))
    assert obj.model.calls == 1


def test_zero_threshold_disables_guard(tc):
    """留退路:env 設 0 要能完全回到舊行為。"""
    obj = _make(tc, min_px=0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        obj.classify(frame, _bbox(5, 5))
    assert obj.model.calls == 1
