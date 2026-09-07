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
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _Probs:
    def __init__(self, top1, conf, n=4):
        self.top1 = top1
        self.top1conf = types.SimpleNamespace(item=lambda: conf)
        # LIGHT_BIAS != 1 時 _infer 會走機率向量路徑,假模型也要提供 data
        vec = [(1.0 - conf) / (n - 1)] * n
        vec[top1] = conf
        self.data = types.SimpleNamespace(tolist=lambda: list(vec))


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
    arbiter = _FakeModel(NAMES, top1=1, conf=0.995)  # heavy_truck(要 >= ARBITER_MIN_CONF)
    obj = _make(primary, arbiter)

    res = obj.classify(_frame(), _bbox())

    assert res["class_name"] == "heavy_truck"
    assert res["confidence"] == pytest.approx(0.995)
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
    arbiter = _FakeModel(NAMES, top1=1, conf=0.995)  # NAMES[1] = heavy_truck
    obj = _make(primary, arbiter)

    res = obj.classify(_frame(), _bbox())

    # 主模型判小貨(用 NAMES_ALT)→ 仲裁判大貨(用 NAMES)→ 改判大貨
    assert res["class_name"] == "heavy_truck"
    assert res["confidence"] == pytest.approx(0.995)


def test_主模型信心不足時回未知():
    """conf_threshold 要套用在最終採用的那個答案上。"""
    primary = _FakeModel(NAMES, top1=2, conf=0.3)    # 低於 conf_threshold 0.5
    arbiter = _FakeModel(NAMES, top1=3)              # 非大貨,不推翻
    obj = _make(primary, arbiter)

    assert obj.classify(_frame(), _bbox())["class_name"] == "unknown"


# ── 判定工作點(2026-09-07 留半驗證後加)────────────────────────────────
class _ProbModel:
    """回傳指定機率向量的假模型(names 依 index 對應)。"""

    def __init__(self, names, probs, calls=None):
        self.names = names
        self._p = probs
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        pr = types.SimpleNamespace(
            top1=max(range(len(self._p)), key=lambda i: self._p[i]),
            top1conf=types.SimpleNamespace(item=lambda: max(self._p)),
            data=types.SimpleNamespace(tolist=lambda: list(self._p)),
        )
        return [types.SimpleNamespace(probs=pr)]


def test_仲裁信心不足時不推翻():
    """舊行為只要 argmax 是大貨就推翻,會誤殺真小貨(留出驗證小貨 F1 67.1%→74.1%)。"""
    from detection import truck_classifier as TC

    primary = _ProbModel(NAMES, [0.0, 0.30, 0.70, 0.0])   # light
    arbiter = _ProbModel(NAMES, [0.0, 0.55, 0.45, 0.0])   # heavy 但只有 0.55
    obj = _make(primary, arbiter)
    with mock.patch.object(TC, "ARBITER_MIN_CONF", 0.99):
        res = obj.classify(_frame(), _bbox())
    assert res["class_name"] == "light_truck", "仲裁沒把握就不該推翻"
    assert arbiter.calls == 1, "仍要問仲裁(要拿到它的信心才能判斷)"


def test_仲裁高信心時才推翻():
    from detection import truck_classifier as TC

    primary = _ProbModel(NAMES, [0.0, 0.30, 0.70, 0.0])
    arbiter = _ProbModel(NAMES, [0.0, 0.995, 0.005, 0.0])
    obj = _make(primary, arbiter)
    with mock.patch.object(TC, "ARBITER_MIN_CONF", 0.99):
        res = obj.classify(_frame(), _bbox())
    assert res["class_name"] == "heavy_truck"


# 🛑 曾經加過 LIGHT_BIAS(把 light 機率乘倍率偏向判小貨),已移除。
# 原因:bias 只在 heavy 機率 > light 時翻轉,那必然代表 light < 0.5 = conf_threshold,
# 翻轉後一律被門檻擋成「未知」—— 只會把「判大貨」變成「判不出來」。
# 離線掃參數看到的 +1.9pp 是假的(那個算法沒模擬 conf_threshold)。詳見該檔註解。
