#!/usr/bin/env python3
"""車流 ROI 進出線(in_edge / out_edge)判定測試。

驗證重點:車輛必須「實際跨越指定的那條邊」才計 IN/OUT,
從其他邊進出的車不能被算進去。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# api.routes 匯入時會過 AUTH_SECRET 資安 gate,測試只用純幾何函式,給個測試用值即可
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.routes.stream import _seg_intersect, _vehicle_center, _zone_edge_segment  # noqa: E402

# 測試用 ROI:400x400 的正方形框
#   points[0]=(100,100) 左上  points[1]=(300,100) 右上
#   points[2]=(300,300) 右下  points[3]=(100,300) 左下
# 邊0 = 上邊, 邊1 = 右邊, 邊2 = 下邊, 邊3 = 左邊
ZONE = {"points": [[100, 100], [300, 100], [300, 300], [100, 300]]}


def _veh(cx, cy):
    """做一台中心點在 (cx,cy) 的車。"""
    return {"bbox": {"x1": cx - 10, "y1": cy - 10, "x2": cx + 10, "y2": cy + 10}}


def test_edge_segment_indexing():
    assert _zone_edge_segment(ZONE, 0) == ((100.0, 100.0), (300.0, 100.0))  # 上邊
    assert _zone_edge_segment(ZONE, 2) == ((300.0, 300.0), (100.0, 300.0))  # 下邊
    assert _zone_edge_segment(ZONE, 3) == ((100.0, 300.0), (100.0, 100.0))  # 左邊(收尾回起點)


def test_edge_segment_unset_returns_none():
    """沒設 / 空字串 / 超界 → None,代表沿用整框進出的舊行為。"""
    for bad in (None, "", -1, 4, 99, "abc", {}):
        assert _zone_edge_segment(ZONE, bad) is None, f"{bad!r} 應視為未指定"


def test_edge_segment_needs_polygon():
    assert _zone_edge_segment({"points": [[0, 0], [1, 1]]}, 0) is None  # 只有 2 點不成多邊形
    assert _zone_edge_segment({}, 0) is None


def test_vehicle_center():
    assert _vehicle_center(_veh(200, 150)) == (200, 150)


def test_cross_top_edge_counts_as_in():
    """車由上往下開進框 → 有跨過上邊(邊0)。"""
    top = _zone_edge_segment(ZONE, 0)
    prev_pt, cur_pt = _vehicle_center(_veh(200, 80)), _vehicle_center(_veh(200, 120))
    assert _seg_intersect(prev_pt, cur_pt, top[0], top[1]) is True


def test_enter_from_other_edge_not_counted():
    """車由左往右從『左邊』進框 → 上邊沒被跨過,不該算成 IN。

    這是這次改動的核心:同一個 ROI 不再兩側都灌數。
    """
    top = _zone_edge_segment(ZONE, 0)
    prev_pt, cur_pt = _vehicle_center(_veh(80, 200)), _vehicle_center(_veh(120, 200))
    assert _seg_intersect(prev_pt, cur_pt, top[0], top[1]) is False


def test_cross_bottom_edge_counts_as_out():
    """車由上往下開出框 → 跨過下邊(邊2)。"""
    bottom = _zone_edge_segment(ZONE, 2)
    prev_pt, cur_pt = _vehicle_center(_veh(200, 280)), _vehicle_center(_veh(200, 320))
    assert _seg_intersect(prev_pt, cur_pt, bottom[0], bottom[1]) is True


def test_move_inside_zone_crosses_nothing():
    """在框內移動不該觸發任何一條邊。"""
    prev_pt, cur_pt = _vehicle_center(_veh(150, 150)), _vehicle_center(_veh(250, 250))
    for i in range(4):
        e = _zone_edge_segment(ZONE, i)
        assert _seg_intersect(prev_pt, cur_pt, e[0], e[1]) is False, f"邊{i} 不該被跨過"


def test_full_pass_through_in_then_out():
    """一台車從上方直穿到下方:先跨上邊(IN)、再跨下邊(OUT),各一次。"""
    track = [_veh(200, y) for y in (60, 90, 120, 200, 280, 310, 340)]
    top, bottom = _zone_edge_segment(ZONE, 0), _zone_edge_segment(ZONE, 2)
    in_count = out_count = 0
    for prev, cur in zip(track, track[1:]):
        p, c = _vehicle_center(prev), _vehicle_center(cur)
        if _seg_intersect(p, c, top[0], top[1]):
            in_count += 1
        if _seg_intersect(p, c, bottom[0], bottom[1]):
            out_count += 1
    assert (in_count, out_count) == (1, 1)


def test_reverse_direction_still_crosses():
    """相交測試不分方向 — 逆向車跨同一條邊也會被偵到。

    方向性由『框內/框外狀態轉換』把關(IN 只在框外→框內時判定),
    這裡確認幾何層本身是雙向的。
    """
    top = _zone_edge_segment(ZONE, 0)
    p, c = _vehicle_center(_veh(200, 120)), _vehicle_center(_veh(200, 80))
    assert _seg_intersect(p, c, top[0], top[1]) is True


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
