"""壅塞判級的流量與車輛數封頂。

要擋的兩種誤報:
  1. 一台大貨車停在近鏡頭 → 佔用率 60%+ → 舊邏輯喊「嚴重壅塞」。
     流量條件救不了它(停著的車流量趨近 0),要靠最少車輛數。
  2. 一堆車佔用率高但正在順暢通過 → 不是壅塞。這才是流量條件負責的。
封頂只能往下修,不可以往上升級。
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
    from detection.congestion_detector import CongestionDetector
    obj = CongestionDetector.__new__(CongestionDetector)
    from collections import defaultdict
    obj.flow_state_map = defaultdict(lambda: {"seen": {}, "passed": []})
    return obj


def _veh(track_id):
    return {"track_id": track_id, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10,
                                           "width": 10, "height": 10}}


def test_stopped_vehicle_not_counted(det):
    """同一台車一直在畫面上 → 流量 0。停等車不該被算成流量。"""
    t0 = datetime(2026, 8, 31, 12, 0, 0)
    for i in range(10):
        vpm = det._update_flow_vpm("cam1", [_veh(1)], now=t0 + timedelta(seconds=i))
    assert vpm == 0.0


def test_disappeared_track_counts_as_passed(det):
    """track 出現又消失 = 通過一台。"""
    t0 = datetime(2026, 8, 31, 12, 0, 0)
    det._update_flow_vpm("cam1", [_veh(1)], now=t0)
    vpm = det._update_flow_vpm("cam1", [], now=t0 + timedelta(seconds=1))
    assert vpm == pytest.approx(1.0)      # 60 秒窗、1 台 → 1 輛/分


def test_flow_window_expires(det):
    """超過 window_sec 的通過紀錄要被丟掉,不然流量會一路累積。"""
    t0 = datetime(2026, 8, 31, 12, 0, 0)
    det._update_flow_vpm("cam1", [_veh(1)], now=t0)
    det._update_flow_vpm("cam1", [], now=t0 + timedelta(seconds=1))
    vpm = det._update_flow_vpm("cam1", [], now=t0 + timedelta(seconds=120))
    assert vpm == 0.0


def test_multiple_vehicles_flow(det):
    """10 台車在 60 秒窗內先後通過 → 10 輛/分。"""
    t0 = datetime(2026, 8, 31, 12, 0, 0)
    for i in range(10):
        det._update_flow_vpm("cam1", [_veh(i)], now=t0 + timedelta(seconds=i))
    vpm = det._update_flow_vpm("cam1", [], now=t0 + timedelta(seconds=10))
    assert vpm == pytest.approx(10.0)


def test_flow_isolated_per_camera(det):
    t0 = datetime(2026, 8, 31, 12, 0, 0)
    det._update_flow_vpm("cam1", [_veh(1)], now=t0)
    det._update_flow_vpm("cam2", [_veh(1)], now=t0)
    det._update_flow_vpm("cam1", [], now=t0 + timedelta(seconds=1))
    assert det._update_flow_vpm("cam2", [_veh(1)], now=t0 + timedelta(seconds=1)) == 0.0


# ---- 封頂邏輯:直接打真的 _cap_level,不重寫一份規則 ----


def _apply_cap(det, level, n_veh, flow_vpm, *, min_high=2, min_critical=3, free_flow=0.0,
               large=True, queue=True, need_queue=True):
    lv, _reason = det._cap_level(
        level,
        vehicle_count=n_veh,
        large_vehicle_present=large,
        flow_vpm=flow_vpm,
        queue_active=queue,
        min_vehicles_high=min_high,
        min_vehicles_critical=min_critical,
        free_flow_vpm=free_flow,
        critical_requires_queue=need_queue,
    )
    return lv


def test_single_truck_capped_at_medium(det):
    """一台車佔用率再高也不該是嚴重壅塞。"""
    assert _apply_cap(det, "critical", n_veh=1, flow_vpm=0.0) == "medium"


def test_two_vehicles_capped_at_high(det):
    """『單一大車卡住前方』仍看得到,只是不喊最高級。"""
    assert _apply_cap(det, "critical", n_veh=2, flow_vpm=0.0) == "high"


def test_three_vehicles_allows_critical(det):
    assert _apply_cap(det, "critical", n_veh=3, flow_vpm=0.0) == "critical"


def test_high_flow_caps_at_medium(det):
    """車多、佔用率高,但正在順暢通過 → 不是壅塞。"""
    assert _apply_cap(det, "critical", n_veh=10, flow_vpm=20.0, free_flow=12.0) == "medium"


def test_low_flow_no_cap(det):
    assert _apply_cap(det, "critical", n_veh=10, flow_vpm=3.0, free_flow=12.0) == "critical"


def test_flow_gate_disabled_is_noop(det):
    """free_flow_vpm=0 必須等於舊行為。"""
    assert _apply_cap(det, "critical", n_veh=10, flow_vpm=999.0, free_flow=0.0) == "critical"


def test_cap_never_upgrades(det):
    """低等級不可以因為封頂規則被拉高。"""
    for lv in ("low", "medium"):
        assert _apply_cap(det, lv, n_veh=1, flow_vpm=0.0) == lv
        assert _apply_cap(det, lv, n_veh=10, flow_vpm=50.0, free_flow=12.0) == lv


def test_strictest_cap_wins(det):
    """車輛數封在 high、流量封在 medium → 取 medium。"""
    assert _apply_cap(det, "critical", n_veh=2, flow_vpm=20.0, free_flow=12.0) == "medium"


def test_cap_only_applies_to_large_vehicles(det):
    """幾台小客車把 ROI 塞到 60% 是真的擠,不可以被車輛數封頂壓下來。

    會「一台就灌爆佔用率」的只有大貨車/大客車。
    """
    assert _apply_cap(det, "critical", n_veh=1, flow_vpm=0.0, large=False) == "critical"
    assert _apply_cap(det, "critical", n_veh=2, flow_vpm=0.0, large=False) == "critical"


def test_large_vehicle_still_capped(det):
    """有大型車在場才套車輛數封頂。"""
    assert _apply_cap(det, "critical", n_veh=1, flow_vpm=0.0, large=True) == "medium"
    assert _apply_cap(det, "critical", n_veh=2, flow_vpm=0.0, large=True) == "high"


def test_flow_cap_ignores_vehicle_class(det):
    """流量封頂跟車種無關 —— 順暢通過就不是壅塞,不管是什麼車。"""
    assert _apply_cap(det, "critical", n_veh=10, flow_vpm=20.0,
                      free_flow=12.0, large=False) == "medium"


def test_large_vehicle_classes_cover_bus_and_truck(det):
    """大貨車(heavy_truck/truck)與大客車(bus)都要算大型車。

    細分類沒跑或判不出來時類別會停在 truck,前端顯示就是「大貨車」,
    漏掉它等於這條規則在細分類降頻後大半時間失效。
    """
    from detection.congestion_detector import CongestionDetector
    assert {"heavy_truck", "bus", "truck"} <= set(CongestionDetector.LARGE_VEHICLE_CLASSES)
    assert "car" not in CongestionDetector.LARGE_VEHICLE_CLASSES
    assert "motorcycle" not in CongestionDetector.LARGE_VEHICLE_CLASSES


def test_reset_clears_zone_flow_state():
    """檔案來源 loop 回開頭時,車道層的流量狀態也要清掉。

    只清相機層的 key 會讓上一輪的通過紀錄灌到新一輪的判級。
    """
    from detection.congestion_detector import CongestionDetector
    from collections import defaultdict
    det = CongestionDetector.__new__(CongestionDetector)
    for attr in ("history_map", "track_meta_map", "queue_state_map", "static_spot_map"):
        setattr(det, attr, defaultdict(dict))
    det.tracker_map = {}
    det.prev_center_map = {}
    det.flow_state_map = defaultdict(lambda: {"seen": {}, "passed": []})
    det.flow_state_map["cam1"]["passed"].append(1.0)
    det.flow_state_map["cam1::zone::車道1"]["passed"].append(1.0)
    det.flow_state_map["cam2::zone::車道1"]["passed"].append(1.0)

    det.reset_camera_state("cam1")

    assert "cam1" not in det.flow_state_map
    assert "cam1::zone::車道1" not in det.flow_state_map
    assert "cam2::zone::車道1" in det.flow_state_map      # 別台不受影響


def test_no_queue_caps_critical(det):
    """排隊 0 公尺卻判嚴重壅塞在物理上是矛盾的 —— 封頂在擁擠。

    這條不分車種:車道 ROI 小,一台小客車就能佔掉六成,
    車輛數封頂(只看大型車)擋不到它。
    """
    assert _apply_cap(det, "critical", n_veh=1, flow_vpm=0.0,
                      large=False, queue=False) == "high"
    assert _apply_cap(det, "critical", n_veh=10, flow_vpm=0.0,
                      large=False, queue=False) == "high"


def test_no_queue_does_not_touch_lower_levels(det):
    """只封 critical,「擁擠」以下不受影響。"""
    for lv in ("low", "medium", "high"):
        assert _apply_cap(det, lv, n_veh=10, flow_vpm=0.0,
                          large=False, queue=False) == lv


def test_queue_active_allows_critical(det):
    """真的有排隊就放行。"""
    assert _apply_cap(det, "critical", n_veh=10, flow_vpm=0.0,
                      large=False, queue=True) == "critical"


def test_critical_requires_queue_can_be_disabled(det):
    """留退路:關掉要能回到舊行為。"""
    assert _apply_cap(det, "critical", n_veh=10, flow_vpm=0.0,
                      large=False, queue=False, need_queue=False) == "critical"


def test_strictest_cap_wins_with_no_queue(det):
    """沒排隊封在 high、流量高封在 medium → 取 medium。"""
    assert _apply_cap(det, "critical", n_veh=10, flow_vpm=30.0, free_flow=12.0,
                      large=False, queue=False) == "medium"
