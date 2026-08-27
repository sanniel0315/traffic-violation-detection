"""壅塞偵測固定物抑制:終生沒動過的 track(地上標線誤判)要被濾掉,真車不可誤殺。

實例:cam_3 (CCTV-N8-E-9-L-NE-2-SIG) 地上白色轉彎箭頭被低信心偵測(conf 0.12)
判成 car,同一 track_id 靜止數小時,壅塞面板長期顯示「暢通車輛 1、佔用率 1.6%」。
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import defaultdict

from detection.congestion_detector import CongestionDetector


def _detector() -> CongestionDetector:
    """不載模型,只建 _update_track_motion 需要的狀態。"""
    det = CongestionDetector.__new__(CongestionDetector)
    det.history_map = defaultdict(list)
    det.tracker_map = {}
    det.track_meta_map = defaultdict(dict)
    det.queue_state_map = defaultdict(dict)
    return det


def _vehicle(track_id: int, cx: int, cy: int, w: int = 76, h: int = 59):
    return {
        "track_id": track_id,
        "class_name": "car",
        "bbox": {"x1": cx - w // 2, "y1": cy - h // 2, "x2": cx + w // 2, "y2": cy + h // 2},
    }


def _step(det, vehicles, now):
    return det._update_track_motion(
        "cam_3",
        vehicles,
        stop_distance_px=45.0,
        stop_min_frames=3,
        static_object_sec=300.0,
        static_object_px=12.0,
        now=now,
    )


def test_static_marking_suppressed_after_grace():
    """固定標線:滿 300 秒前照舊(頂多算停等),滿了之後進 static、退出 stopped。"""
    det = _detector()
    t0 = datetime(2026, 8, 27, 23, 0, 0)
    # 前 299 秒:位置只抖動 ±1px
    for sec in (0, 60, 120, 299):
        stopped, static = _step(det, [_vehicle(5384, 793 + (sec % 2), 616)], t0 + timedelta(seconds=sec))
    assert 5384 not in static  # 未滿門檻不可提前抑制(排隊車保護)
    assert 5384 in stopped     # 這段期間仍是「停等」語意
    # 滿 300 秒:變固定物
    stopped, static = _step(det, [_vehicle(5384, 793, 616)], t0 + timedelta(seconds=301))
    assert 5384 in static
    assert 5384 not in stopped


def test_real_queued_car_never_suppressed():
    """真車:移動進畫面後停等,就算停超過 300 秒也不可被當固定物。"""
    det = _detector()
    t0 = datetime(2026, 8, 27, 23, 0, 0)
    # 從遠處開進來(位移遠大於 12px)
    for i, cy in enumerate((400, 450, 500, 560, 616)):
        _step(det, [_vehicle(77, 793, cy)], t0 + timedelta(seconds=i))
    # 停在原地超過 300 秒(stopped 判定看最近 3 幀,要多餵幾幀靜止)
    for sec in (200, 300, 400):
        stopped, static = _step(det, [_vehicle(77, 793, 616)], t0 + timedelta(seconds=sec))
    assert 77 not in static
    assert 77 in stopped


def test_two_tracks_independent():
    """固定物與真車同框:只濾固定物。"""
    det = _detector()
    t0 = datetime(2026, 8, 27, 23, 0, 0)
    for i, cy in enumerate((400, 500, 616)):
        _step(
            det,
            [_vehicle(5384, 793, 300), _vehicle(77, 900, cy)],
            t0 + timedelta(seconds=i * 150),
        )
    stopped, static = _step(
        det,
        [_vehicle(5384, 793, 300), _vehicle(77, 900, 616)],
        t0 + timedelta(seconds=500),
    )
    assert static == {5384}
    assert 77 not in static


if __name__ == "__main__":
    test_static_marking_suppressed_after_grace()
    test_real_queued_car_never_suppressed()
    test_two_tracks_independent()
    print("OK")
