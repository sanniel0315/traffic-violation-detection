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


# ── 偵測信心門檻(2026-09-07 實測後定案)────────────────────────────────
def test_信心門檻不可低於雜訊帶():
    """cam_3 地上左轉箭頭被判成 car 的信心是 0.131。門檻低於它就擋不住標線。

    2026-09-07 現場實測 3603 幀 / 9822 筆:
      0.12(舊值) 誤報 3572 : 真車 3105 → 誤報佔 53.5%
      0.25       誤報   67 : 真車 1633 → 誤報佔  3.9%
    既有的固定物抑制擋不住這種「閃爍出現」的擦邊偵測(每次中斷就把 300 秒計時歸零),
    只能從源頭擋。
    """
    from detection.congestion_detector import CongestionDetector

    assert CongestionDetector.DEFAULT_DETECT_CONF >= 0.20, (
        "壅塞偵測門檻低於 0.20 → 地上標線/標誌/路緣會被當成車"
    )


def test_fallback_門檻也要在雜訊之上():
    """fallback 只在主偵測器一台都沒抓到時啟用 —— 空曠路面必定觸發它。

    舊值 0.05:主門檻拉高後,空路會落到 fallback,標線照樣被抓進來,
    誤報反而更明顯。兩個門檻要一起守。
    """
    from detection.congestion_detector import CongestionDetector

    # cam_3 箭頭標線信心分佈(9017 幀/778 次):P95 0.181、P99 0.218。
    # 低於 P99 就會在空路上反覆冒出假車(實測 0.15 時約 25% 過關)。
    assert CongestionDetector.DEFAULT_FALLBACK_CONF >= 0.22, (
        "fallback 門檻低於標線信心的 P99(0.218) → 空曠路面會用標線湊出佔用率"
    )
    assert (CongestionDetector.DEFAULT_FALLBACK_CONF
            <= CongestionDetector.DEFAULT_DETECT_CONF), "fallback 不該比主門檻嚴"


# ── 單車不算壅塞(2026-09-07 實測後定案)────────────────────────────────
def _analyze_one(det, cls_name, occ_boxes):
    """跑一次 analyze,回傳 level。occ_boxes 決定佔用率高低。"""
    import numpy as np
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    class _FakeDet:
        def detect(self, f):
            # 🛑 bbox 必須帶 width/height:analyze 的面積過濾讀的是這兩個鍵,
            #    只給 x1/y1/x2/y2 會被算成面積 0 → 被最小面積門檻濾光。
            return [{"bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                              "width": x2 - x1, "height": y2 - y1},
                     "class_name": cls_name, "confidence": 0.9}
                    for x1, y1, x2, y2 in occ_boxes]

    det.detector = _FakeDet()
    det.fallback_detector = None
    res = None
    for _ in range(6):          # 跑幾輪讓 tracker/停等判定穩定
        res = det.analyze(frame, zones=None, camera_key="t_single",
                          params={"stop_speed_px_per_sec": 0, "stop_min_frames": 2})
    return res


def test_單一小客車停著不算壅塞():
    """cam_3 上匝道實測:ROI 94,101 px²,單一小客車依距離佔 10.2%~25.8%,
    中等門檻 0.20 正好卡在中間 → 同一台車停近一點就「中等」、遠一點就「暢通」。
    紅燈時一台小客車在匝道停等是正常現象,不是壅塞。"""
    from detection.congestion_detector import CongestionDetector

    det = CongestionDetector.__new__(CongestionDetector)
    CongestionDetector.__init__(det, vehicle_detector=object())
    # 一台佔畫面約 25% 的靜止小客車
    big = [(200, 200, 840, 660)]
    res = _analyze_one(det, "car", big)
    assert res["level"] == "low", f"單一小客車不該判壅塞,實得 {res['level']}"


def test_單一大貨車停著仍算壅塞():
    """保留原設計:單一大貨/大客卡住前方 != 短暫路過,仍要升級。"""
    from detection.congestion_detector import CongestionDetector

    det = CongestionDetector.__new__(CongestionDetector)
    CongestionDetector.__init__(det, vehicle_detector=object())
    big = [(200, 200, 840, 660)]
    res = _analyze_one(det, "heavy_truck", big)
    assert res["level"] != "low", "單一大貨車卡住仍應判為壅塞"
