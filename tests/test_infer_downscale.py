#!/usr/bin/env python3
"""縮圖推論後,bbox 必須正確換算回原圖座標。

推論前把 1080p 縮小可省下大量 CPU(memcpy + ultralytics 的 CPU letterbox),
但下游全部以原圖座標運作(ROI 多邊形判定、違規 bbox、LPR 車牌裁切),
換算錯了 ROI 就會整個錯位、車全部判定在框外。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.routes.stream import _scale_detections  # noqa: E402


def det(x1, y1, x2, y2, cls="car"):
    return {"class_name": cls, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def test_scale_back_to_original():
    """1920 縮到 960 推論(scale=0.5)，bbox 要乘回 2.0。"""
    d = _scale_detections([det(100, 50, 200, 150)], 2.0)
    b = d[0]["bbox"]
    assert (b["x1"], b["y1"], b["x2"], b["y2"]) == (200, 100, 400, 300)


def test_width_height_recomputed():
    """width/height 也要跟著換算 — 面積過濾與車型判定會用到。"""
    d = _scale_detections([det(100, 50, 200, 150)], 2.0)
    b = d[0]["bbox"]
    assert b["width"] == 200 and b["height"] == 200


def test_scale_one_is_noop():
    """scale=1.0（未縮圖）必須原封不動回傳。"""
    orig = [det(10, 20, 30, 40)]
    out = _scale_detections(orig, 1.0)
    assert out is orig
    assert out[0]["bbox"]["x1"] == 10


def test_multiple_detections():
    ds = _scale_detections([det(10, 10, 20, 20), det(100, 100, 200, 200)], 3.0)
    assert ds[0]["bbox"]["x2"] == 60
    assert ds[1]["bbox"]["y1"] == 300


def test_roi_containment_preserved():
    """核心保證:縮圖前後,車輛中心點與 ROI 的相對關係不變。"""
    # 原圖 1920x1080，ROI 是 x 800-1200 的區域
    roi_x1, roi_x2 = 800, 1200
    # 原圖上車在 x 900-1000（在 ROI 內）
    # 縮到 960 寬（scale 0.5）後推論，模型回報 x 450-500
    scaled_back = _scale_detections([det(450, 250, 500, 300)], 2.0)
    b = scaled_back[0]["bbox"]
    cx = (b["x1"] + b["x2"]) / 2
    assert roi_x1 <= cx <= roi_x2, f"換算後中心 {cx} 應落在 ROI {roi_x1}-{roi_x2} 內"
    assert (b["x1"], b["x2"]) == (900, 1000)


def test_malformed_bbox_is_safe():
    assert _scale_detections([{"class_name": "car"}], 2.0)          # 沒有 bbox
    assert _scale_detections([{"class_name": "car", "bbox": {}}], 2.0)
    assert _scale_detections([{"class_name": "car",
                               "bbox": {"x1": None, "y1": 1, "x2": 2, "y2": 3}}], 2.0)


def test_empty_input():
    assert _scale_detections([], 2.0) == []
    assert _scale_detections(None, 2.0) is None


def test_rounding_is_stable():
    """非整數倍縮放要四捨五入，不可截斷造成系統性偏移。"""
    d = _scale_detections([det(100, 100, 101, 101)], 1.5)
    b = d[0]["bbox"]
    assert b["x1"] == 150 and b["x2"] == 152   # 101*1.5=151.5 → 152


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
