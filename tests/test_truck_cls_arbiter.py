"""大小貨車仲裁(規則 A)邏輯測試。

規則:主模型先判;只有它判 light_truck 時才叫仲裁模型,若仲裁說 heavy_truck 就改判大貨。
依據 2026-09-06 695 條盲標歧異抽樣:主模型單獨 -2.46pp,擋掉「大貨誤判小貨」這一格後 +1.62pp。

重點驗證:
  1. 主模型判非小貨 → 不可呼叫仲裁(成本控制的關鍵,叫了就等於兩顆全跑)
  2. 主模型判小貨 + 仲裁判大貨 → 改判大貨
  3. 主模型判小貨 + 仲裁不判大貨 → 維持主模型的小貨
  4. 沒有主模型 → 行為與啟用前完全相同(安全退路)
  5. 兩顆模型 index→name 順序不同時仍各用各的(共用會判錯類別)
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _Probs:
    def __init__(self, top1, conf):
        self.top1 = top1
        self.top1conf = types.SimpleNamespace(item=lambda: conf)


class _Result:
    def __init__(self, top1, conf):
        self.probs = _Probs(top1, conf)


class _FakeModel:
    """回傳指定 top1 index 的假模型,並記錄被呼叫幾次。"""

    def __init__(self, names, top1, conf=0.9):
        self.names = names
        self._top1 = top1
        self._conf = conf
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        return [_Result(self._top1, self._conf)]


NAMES = {0: "bus", 1: "heavy_truck", 2: "light_truck", 3: "non_truck"}
# 故意用不同順序,驗證不會共用 class_names
NAMES_ALT = {0: "light_truck", 1: "non_truck", 2: "bus", 3: "heavy_truck"}


def _make(primary_model, arbiter_model):
    """繞過 __init__(會載真模型),直接組出要測的物件。"""
    from detection.truck_classifier import TruckClassifier

    obj = TruckClassifier.__new__(TruckClassifier)
    obj.model = arbiter_model
    obj.class_names = arbiter_model.names
    obj.primary = primary_model
    obj.primary_names = primary_model.names if primary_model else None
    obj.conf_threshold = 0.5
    obj.imgsz = 224
    obj.device = "cpu"
    return obj


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _bbox():
    return {"x1": 100, "y1": 100, "x2": 260, "y2": 260}


def test_主模型判大貨時不呼叫仲裁():
    """成本控制的關鍵:非小貨就不該多跑第二顆。"""
    primary = _FakeModel(NAMES, top1=1)      # heavy_truck
    arbiter = _FakeModel(NAMES, top1=1)
    obj = _make(primary, arbiter)

    res = obj.classify(_frame(), _bbox())

    assert res["class_name"] == "heavy_truck"
    assert primary.calls == 1
    assert arbiter.calls == 0, "主模型判非小貨卻叫了仲裁 → 成本會變成兩顆全跑"


def test_主模型判小客車時不呼叫仲裁():
    primary = _FakeModel(NAMES, top1=3)      # non_truck
    arbiter = _FakeModel(NAMES, top1=1)
    obj = _make(primary, arbiter)

    assert obj.classify(_frame(), _bbox())["class_name"] == "non_truck"
    assert arbiter.calls == 0


def test_主模型判小貨而仲裁判大貨時改判大貨():
    """這就是規則 A 修掉的那一格(246 條中真值 226 條是大貨)。"""
    primary = _FakeModel(NAMES, top1=2)      # light_truck
    arbiter = _FakeModel(NAMES, top1=1, conf=0.88)   # heavy_truck
    obj = _make(primary, arbiter)

    res = obj.classify(_frame(), _bbox())

    assert res["class_name"] == "heavy_truck"
    assert res["confidence"] == pytest.approx(0.88)
    assert arbiter.calls == 1


def test_主模型判小貨而仲裁不判大貨時維持小貨():
    primary = _FakeModel(NAMES, top1=2)      # light_truck
    arbiter = _FakeModel(NAMES, top1=3)      # non_truck → 不推翻
    obj = _make(primary, arbiter)

    assert obj.classify(_frame(), _bbox())["class_name"] == "light_truck"
    assert arbiter.calls == 1


def test_沒有主模型時行為與啟用前相同():
    """安全退路:主模型載入失敗絕不能變成只跑主模型。"""
    arbiter = _FakeModel(NAMES, top1=2)      # light_truck
    obj = _make(None, arbiter)

    assert obj.classify(_frame(), _bbox())["class_name"] == "light_truck"
    assert arbiter.calls == 1


def test_兩顆模型類別順序不同時各用各的映射():
    """共用 class_names 會把類別對錯 —— 這是最容易靜默出錯的地方。"""
    primary = _FakeModel(NAMES_ALT, top1=0)          # NAMES_ALT[0] = light_truck
    arbiter = _FakeModel(NAMES, top1=1, conf=0.77)   # NAMES[1]     = heavy_truck
    obj = _make(primary, arbiter)

    res = obj.classify(_frame(), _bbox())

    # 主模型判小貨(用 NAMES_ALT)→ 仲裁判大貨(用 NAMES)→ 改判大貨
    assert res["class_name"] == "heavy_truck"
    assert res["confidence"] == pytest.approx(0.77)


def test_仲裁結果信心不足時回未知():
    """門檻要套用在最終採用的那個答案上。"""
    primary = _FakeModel(NAMES, top1=2)
    arbiter = _FakeModel(NAMES, top1=1, conf=0.3)    # 低於 0.5
    obj = _make(primary, arbiter)

    assert obj.classify(_frame(), _bbox())["class_name"] == "unknown"
