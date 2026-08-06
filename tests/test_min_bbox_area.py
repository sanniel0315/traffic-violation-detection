#!/usr/bin/env python3
"""最小 bbox 面積過濾:砍掉把路面標線誤判成車的小框,且不誤殺真車。

回歸案例:某支五車道攝影機 33% 的偵測是 27x21 之類的小框,
反覆出現在完全相同的座標((286,369) 10 次) —— 靜止物特徵。
實測該攝影機真車面積 p05=10395,誤判多在 567~3000。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")


def _filter(vehicles, min_area):
    """複製 stream.py 內的過濾邏輯（同一份運算式）。"""
    if not vehicles or min_area <= 0:
        return vehicles
    return [
        v for v in vehicles
        if ((v.get('bbox') or {}).get('x2', 0) - (v.get('bbox') or {}).get('x1', 0))
           * ((v.get('bbox') or {}).get('y2', 0) - (v.get('bbox') or {}).get('y1', 0))
           >= min_area
    ]


def veh(x1, y1, w, h, cls="car"):
    return {"class_name": cls, "bbox": {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h}}


# 取自實測:誤判的固定座標小框 vs 真車
FP = [veh(166, 354, 27, 21), veh(234, 371, 35, 17), veh(286, 369, 30, 22),
      veh(110, 377, 34, 20), veh(87, 395, 33, 18)]
REAL = [veh(440, 405, 135, 85), veh(700, 500, 180, 120), veh(300, 450, 90, 60)]


def test_default_zero_keeps_everything():
    """預設 0 = 不過濾,既有攝影機行為完全不變。"""
    allv = FP + REAL
    assert _filter(allv, 0) == allv
    assert len(_filter(allv, 0)) == 8


def test_threshold_2000_removes_false_positives():
    kept = _filter(FP + REAL, 2000)
    assert len(kept) == len(REAL), f"應只剩真車 {len(REAL)} 台，實際 {len(kept)}"
    for v in kept:
        b = v["bbox"]
        assert (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) >= 2000


def test_all_known_false_positives_are_cut():
    """實測到的誤判樣本，在門檻 2000 下要全部被砍掉。"""
    assert _filter(FP, 2000) == []


def test_real_vehicles_survive():
    """真車一台都不能少。"""
    assert len(_filter(REAL, 2000)) == len(REAL)


def test_boundary_is_inclusive():
    """剛好等於門檻要保留（>=，不是 >）。"""
    exact = [veh(0, 0, 50, 40)]        # 面積正好 2000
    assert len(_filter(exact, 2000)) == 1
    just_under = [veh(0, 0, 50, 39)]   # 1950
    assert _filter(just_under, 2000) == []


def test_empty_and_malformed_are_safe():
    assert _filter([], 2000) == []
    assert _filter([{"class_name": "car"}], 2000) == []          # 沒有 bbox
    assert _filter([{"class_name": "car", "bbox": {}}], 2000) == []


def test_large_truck_not_affected():
    big = [veh(100, 400, 400, 300, "heavy_truck")]
    assert len(_filter(big, 2000)) == 1


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
