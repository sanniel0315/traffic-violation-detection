"""大型車細分類快取:結果必須跟不用快取時一模一樣。

快取只是省掉「同一台車重複推論」,不可以改變任何一筆標籤。
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def vd(monkeypatch):
    """載入 vehicle_detector,但不要真的去載 YOLO 模型。"""
    for name in ("ultralytics", "model_paths"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if name == "ultralytics":
                mod.YOLO = object
            else:
                mod.get_detect_model_pt = lambda *a, **k: "x.pt"
            sys.modules[name] = mod
    import detection.vehicle_detector as m
    return m


def _bbox(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class _FakeDetector:
    """只借用 VehicleDetector 的快取方法,不碰模型。"""

    def __init__(self, m, results):
        self._tc_cache = []
        self._tc_cache_on = True      # 對應 _get_per_cam_detector 的 cls_cache=True
        self._results = list(results)
        self.calls = 0
        self._m = m
        self.truck_classifier = object()

    def _truck_cls_raw(self, frame, bbox):
        self.calls += 1
        return self._results[min(self.calls - 1, len(self._results) - 1)]


def _cached(m, det, frame, bbox):
    return m.VehicleDetector._truck_cls_cached(det, frame, bbox)


def test_同一台車連續幀只推論一次(vd):
    r = {"class_name": "heavy_truck", "label": "大貨車"}
    det = _FakeDetector(vd, [r])
    # 同一台車緩慢往前開,每幀位移 5px(重疊仍高)
    for i in range(10):
        res, hit = _cached(vd, det, None, _bbox(100 + i * 5, 200, 300 + i * 5, 400))
        assert res == r, "快取回傳的結果必須跟原本一致"
        assert hit == (i > 0), "第一幀該推論,之後該命中"
    assert det.calls == 1, f"應該只推論 1 次,實際 {det.calls} 次"


def test_不同車不會互相污染(vd):
    a = {"class_name": "heavy_truck", "label": "大貨車"}
    b = {"class_name": "bus", "label": "大客車"}
    det = _FakeDetector(vd, [a, b])
    r1, h1 = _cached(vd, det, None, _bbox(100, 200, 300, 400))
    r2, h2 = _cached(vd, det, None, _bbox(900, 200, 1100, 400))   # 完全沒重疊
    assert (r1, h1) == (a, False)
    assert (r2, h2) == (b, False), "另一台車必須自己推論,不能沿用"
    assert det.calls == 2


def test_重疊不足就重新推論(vd):
    a = {"class_name": "heavy_truck"}
    b = {"class_name": "light_truck"}
    det = _FakeDetector(vd, [a, b])
    _cached(vd, det, None, _bbox(0, 0, 100, 100))
    # 只重疊 1/4 面積 → IoU 約 0.14,低於門檻 0.6
    res, hit = _cached(vd, det, None, _bbox(50, 50, 150, 150))
    assert hit is False
    assert res == b
    assert det.calls == 2


def test_過了TTL會重新驗證(vd, monkeypatch):
    a = {"class_name": "heavy_truck"}
    b = {"class_name": "bus"}
    det = _FakeDetector(vd, [a, b])
    t = [1000.0]
    monkeypatch.setattr(vd._time_stats, "perf_counter", lambda: t[0])

    box = _bbox(100, 200, 300, 400)
    assert _cached(vd, det, None, box) == (a, False)
    t[0] += vd.TRUCK_CLS_CACHE_SEC * 0.5
    assert _cached(vd, det, None, box) == (a, True), "TTL 內應命中"
    t[0] += vd.TRUCK_CLS_CACHE_SEC * 0.6      # 累計超過 TTL
    res, hit = _cached(vd, det, None, box)
    assert hit is False, "超過 TTL 必須重驗,否則分錯的標籤會永遠留著"
    assert res == b
    assert det.calls == 2


def test_命中不會延長壽命(vd, monkeypatch):
    """連續命中不可以無限續命 —— 每台車至少每 TTL 秒重驗一次。"""
    det = _FakeDetector(vd, [{"class_name": "bus"}])
    t = [0.0]
    monkeypatch.setattr(vd._time_stats, "perf_counter", lambda: t[0])
    box = _bbox(100, 200, 300, 400)
    _cached(vd, det, None, box)
    for _ in range(20):                        # 高頻命中
        t[0] += vd.TRUCK_CLS_CACHE_SEC * 0.1
        _cached(vd, det, None, box)
    assert det.calls >= 2, "跨過 TTL 後必須重驗過至少一次"


def test_關掉快取就完全照舊(vd, monkeypatch):
    monkeypatch.setattr(vd, "TRUCK_CLS_CACHE", False)
    det = _FakeDetector(vd, [{"class_name": "bus"}])
    box = _bbox(100, 200, 300, 400)
    for _ in range(5):
        res, hit = _cached(vd, det, None, box)
        assert hit is False
    assert det.calls == 5, "關掉快取時每幀都要推論"


def test_iou算式(vd):
    assert vd._bbox_iou(_bbox(0, 0, 10, 10), _bbox(0, 0, 10, 10)) == 1.0
    assert vd._bbox_iou(_bbox(0, 0, 10, 10), _bbox(20, 20, 30, 30)) == 0.0
    # 一半重疊:交集 50、聯集 150
    assert abs(vd._bbox_iou(_bbox(0, 0, 10, 10), _bbox(5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_共用偵測器不可以開快取(vd):
    """跨來源共用的 instance 開快取會讓兩台相機同位置的車互相沿用標籤。"""
    a = {"class_name": "heavy_truck"}
    b = {"class_name": "bus"}
    det = _FakeDetector(vd, [a, b])
    det._tc_cache_on = False          # 共用 singleton 的設定
    box = _bbox(100, 200, 300, 400)
    assert _cached(vd, det, None, box) == (a, False)
    assert _cached(vd, det, None, box) == (b, False), "沒開快取就必須每次重算"
    assert det.calls == 2


def test_沒有結論的不要存進快取(vd):
    """_default_result()(unknown/信心0)是「這次判不出來」,不是判斷結果。

    存了會讓這台車在 TTL 內都拿不到細分類 —— 下一幀角度變一下本來就可能判得出來。
    """
    unknown = {"class_name": "unknown", "label": "未知", "confidence": 0.0}
    good = {"class_name": "heavy_truck", "confidence": 0.95}
    det = _FakeDetector(vd, [unknown, good])
    box = _bbox(100, 200, 300, 400)
    r1, h1 = _cached(vd, det, None, box)
    assert (r1["class_name"], h1) == ("unknown", False)
    # 同一台車下一幀:不可以沿用 unknown,要重新判
    r2, h2 = _cached(vd, det, None, box)
    assert h2 is False, "unknown 不該被快取"
    assert r2["class_name"] == "heavy_truck"
    # 這次有結論了,才該進快取
    r3, h3 = _cached(vd, det, None, box)
    assert h3 is True and r3["class_name"] == "heavy_truck"
