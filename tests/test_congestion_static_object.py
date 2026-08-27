"""壅塞偵測固定物抑制:被誤判成車的地上標線要被濾掉,真車不可誤殺。

實例:cam_3 (CCTV-N8-E-9-L-NE-2-SIG) 地上白色轉彎箭頭被低信心偵測(conf 0.12)
判成 car,壅塞面板長期顯示「暢通車輛 1、佔用率 1.6%」。
🛑 誤判會閃爍、track id 一直換(實測 6 分鐘 id 1→21),所以存在時間掛在
「固定點(位置)」上跨 track 累積,不能掛在 track 上。
"""
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.congestion_detector import CongestionDetector


def _detector() -> CongestionDetector:
    """不載模型,只建 _update_track_motion 需要的狀態。"""
    det = CongestionDetector.__new__(CongestionDetector)
    det.history_map = defaultdict(list)
    det.tracker_map = {}
    det.track_meta_map = defaultdict(dict)
    det.queue_state_map = defaultdict(dict)
    det.static_spot_map = defaultdict(list)
    det.prev_center_map = {}
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


T0 = datetime(2026, 8, 27, 23, 0, 0)


def test_static_marking_same_track():
    """固定標線(track id 不變):滿 300 秒前照舊,滿了之後進 static、退出 stopped。"""
    det = _detector()
    stopped = static = set()
    # 每 10 秒一幀,位置抖 ±1px;固定點在第二幀(有上一幀可比)誕生,年齡從那時起算
    for sec in range(0, 310, 10):
        stopped, static = _step(det, [_vehicle(5384, 793 + (sec // 10 % 2), 616)], T0 + timedelta(seconds=sec))
    assert 5384 not in static  # 未滿門檻不可提前抑制(排隊車保護)
    assert 5384 in stopped     # 這段期間仍是「停等」語意
    stopped, static = _step(det, [_vehicle(5384, 793, 616)], T0 + timedelta(seconds=315))
    assert 5384 in static
    assert 5384 not in stopped


def test_static_marking_flickering_track_ids():
    """固定標線(閃爍、track id 一直換):存在時間仍要跨 track 累積到抑制。"""
    det = _detector()
    static = set()
    for i, sec in enumerate(range(0, 320, 10)):
        # 每幀都換一個新 track id,模擬低信心偵測閃斷重生
        _, static = _step(det, [_vehicle(100 + i, 793, 616)], T0 + timedelta(seconds=sec))
    assert static  # 最後一個 track id 已被抑制
    assert 100 + 31 in static


def test_real_queued_car_not_suppressed_within_grace():
    """真車:移動進畫面後停等,300 秒內不可被當固定物、且要算停等。
    (刻意的 tradeoff:凍在原地「連續」超過 300 秒才會被視為固定物;
    紅燈/儀控週期遠短於 300 秒,正常停等不受影響。)"""
    det = _detector()
    for i, cy in enumerate((400, 450, 500, 560, 616)):
        _step(det, [_vehicle(77, 793, cy)], T0 + timedelta(seconds=i * 5))
    stopped = static = set()
    for sec in range(30, 300, 10):
        stopped, static = _step(det, [_vehicle(77, 793, 616)], T0 + timedelta(seconds=sec))
    assert 77 not in static
    assert 77 in stopped


def test_red_light_cycles_do_not_accumulate():
    """停止線頭車:每紅燈停 60 秒、綠燈空 90 秒(>30 秒 gap),固定點計時要歸零,
    多個週期不可累積到誤殺。track id 每週期不同、都直接生在停止線(最壞情況)。"""
    det = _detector()
    static = set()
    t = 0
    for cycle in range(6):  # 6 個週期共 900 秒,單點累積早超過 300 秒
        tid = 200 + cycle
        for sec in range(0, 60, 10):
            _, static = _step(det, [_vehicle(tid, 793, 616)], T0 + timedelta(seconds=t + sec))
            assert tid not in static
        t += 60 + 90  # 綠燈 90 秒沒車在該點
    assert not static


def test_hijacked_track_resumes_suppression():
    """真車開過標線,tracker 把標線 track 短暫接到車上再跳回來:
    抑制不可因 track 位移史被污染而失效(87 實測踩過)。"""
    det = _detector()
    static = set()
    for sec in range(0, 320, 10):  # 標線靜止滿 300 秒 → 已被抑制
        _, static = _step(det, [_vehicle(21, 793, 616)], T0 + timedelta(seconds=sec))
    assert 21 in static
    # track 21 被接到路過的車上(位置大跳),10 秒後又跳回標線
    _, static = _step(det, [_vehicle(21, 400, 300)], T0 + timedelta(seconds=325))
    assert 21 not in static  # 在車上時是真車,不可抑制
    _, static = _step(det, [_vehicle(21, 793, 616)], T0 + timedelta(seconds=335))
    assert 21 in static  # 回到標線立刻恢復抑制


def test_static_and_real_car_coexist():
    """固定物與真車同框:只抑制固定物。"""
    det = _detector()
    static = set()
    for sec in range(0, 320, 10):
        cy = min(616, 300 + sec * 2)  # 真車持續移動
        _, static = _step(
            det,
            [_vehicle(5384, 793, 200), _vehicle(88, 1100, cy)],
            T0 + timedelta(seconds=sec),
        )
    assert 5384 in static
    assert 88 not in static


if __name__ == "__main__":
    test_static_marking_same_track()
    test_static_marking_flickering_track_ids()
    test_real_queued_car_not_suppressed_within_grace()
    test_red_light_cycles_do_not_accumulate()
    test_hijacked_track_resumes_suppression()
    test_static_and_real_car_coexist()
    print("OK")
