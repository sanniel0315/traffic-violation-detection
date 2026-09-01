"""停等判定改用速度(px/秒)而非每幀位移。

🛑 為什麼:每幀基準的門檻會隨分析率漂移 ——
   8 fps 時「3 幀位移 < 45px」等於 120 px/秒;
   0.7 fps 時同一組參數等於 10 px/秒。
   分析率隨車流量變動,排隊量測的地基不能建在會漂移的門檻上。
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def det():
    from collections import defaultdict
    from detection.congestion_detector import CongestionDetector
    d = CongestionDetector.__new__(CongestionDetector)
    d.track_meta_map = defaultdict(dict)
    d.static_spot_map = defaultdict(list)
    d.prev_center_map = {}
    d.tracker_map = {}
    return d


def _veh(track_id, cx, cy, w=100, h=80):
    return {"track_id": track_id, "class_name": "car",
            "bbox": {"x1": cx - w // 2, "y1": cy - h // 2,
                     "x2": cx + w // 2, "y2": cy + h // 2}}


def _feed(det, positions, *, dt_sec, speed_thr, window_sec=1.0, cam="cam1"):
    """依序餵入各時間點的位置,回傳最後一次的 stopped ids。"""
    t0 = datetime(2026, 9, 1, 12, 0, 0)
    stopped = set()
    for i, (cx, cy) in enumerate(positions):
        stopped, _static = det._update_track_motion(
            cam, [_veh(1, cx, cy)],
            stop_distance_px=45.0, stop_min_frames=3,
            stop_speed_px_per_sec=speed_thr, stop_min_window_sec=window_sec,
            static_object_sec=99999.0, static_object_px=12.0,
            now=t0 + timedelta(seconds=i * dt_sec),
        )
    return stopped


def test_stationary_vehicle_is_stopped(det):
    """完全不動 → 停等。"""
    assert 1 in _feed(det, [(500, 500)] * 5, dt_sec=1.4, speed_thr=20)


def test_fast_vehicle_is_not_stopped(det):
    """每 1.4 秒移動 200px(≈143 px/秒) → 不是停等。"""
    pos = [(300 + i * 200, 500) for i in range(5)]
    assert 1 not in _feed(det, pos, dt_sec=1.4, speed_thr=20)


def test_creeping_vehicle_under_threshold_is_stopped(det):
    """緩行:每 1.4 秒移動 14px(=10 px/秒) → 門檻 20 之下,算停等。"""
    pos = [(300 + i * 14, 500) for i in range(5)]
    assert 1 in _feed(det, pos, dt_sec=1.4, speed_thr=20)


def test_threshold_is_independent_of_frame_rate(det):
    """同一台車、同樣的實際速度,取樣率不同,判定必須一致 —— 這是這次改動的重點。

    速度固定 50 px/秒:高取樣率(0.35秒/幀)與低取樣率(2.8秒/幀)都要判成「非停等」。
    舊的每幀位移門檻在這兩種取樣率下會給出相反的答案。
    """
    fast_sampling = [(300 + int(i * 50 * 0.35), 500) for i in range(12)]
    slow_sampling = [(300 + int(i * 50 * 2.8), 500) for i in range(5)]
    assert 1 not in _feed(det, fast_sampling, dt_sec=0.35, speed_thr=20, cam="a")
    assert 1 not in _feed(det, slow_sampling, dt_sec=2.8, speed_thr=20, cam="b")


def test_slow_speed_consistent_across_frame_rates(det):
    """反向:實際速度 5 px/秒,兩種取樣率都要判成停等。"""
    fast = [(300 + int(i * 5 * 0.35), 500) for i in range(12)]
    slow = [(300 + int(i * 5 * 2.8), 500) for i in range(5)]
    assert 1 in _feed(det, fast, dt_sec=0.35, speed_thr=20, cam="c")
    assert 1 in _feed(det, slow, dt_sec=2.8, speed_thr=20, cam="d")


def test_window_too_short_does_not_conclude(det):
    """觀察時間不足時不下結論 —— 寧可晚一點認定,也不要用一兩點翻轉判定。"""
    # 只餵 2 點、間隔 0.2 秒,湊不滿 1 秒的觀察窗
    assert 1 not in _feed(det, [(500, 500), (500, 500)], dt_sec=0.2,
                          speed_thr=20, window_sec=1.0)


def test_zero_threshold_falls_back_to_legacy(det):
    """0 = 退回舊的每幀位移判定,行為不變(可退回)。"""
    # 每幀移動 10px、3 幀共 20px < 45px → 舊邏輯判停等
    pos = [(300 + i * 10, 500) for i in range(5)]
    assert 1 in _feed(det, pos, dt_sec=1.4, speed_thr=0)
    # 每幀移動 100px、3 幀共 200px > 45px → 舊邏輯判非停等
    pos2 = [(300 + i * 100, 500) for i in range(5)]
    assert 1 not in _feed(det, pos2, dt_sec=1.4, speed_thr=0, cam="cam2")
