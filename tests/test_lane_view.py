"""壅塞量測按車道挑(2026-09-19 使用者:「是分車道的,我的流量也要分車道」)。

NE-2 畫面裡有「上匝道前停等區」(車道 1,分相 2)和「下匝道後平面道路」(車道 2,不屬上匝道)。
整台相機的排隊會把車道 2 混進來 —— 實測有排隊時平均 16.3 m 對車道 10.0 m。
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")

R = {"estimated_queue_length_m": 18.0, "raw_occupancy": 0.4, "occupancy": 0.6, "flow_vpm": 35.0,
     "vehicle_count": 5, "stopped_vehicle_count": 4, "level": "high", "level_name": "擁擠",
     "zone_results": [
         {"lane_no": 2, "estimated_queue_length_m": 18.0, "raw_occupancy": 0.4, "occupancy": 0.6,
          "flow_vpm": 22.0, "vehicle_count": 3, "stopped_vehicle_count": 3, "level": "high", "level_name": "擁擠"},
         {"lane_no": 1, "estimated_queue_length_m": 10.0, "raw_occupancy": 0.2, "occupancy": 0.3,
          "flow_vpm": 13.0, "vehicle_count": 2, "stopped_vehicle_count": 1, "level": "medium", "level_name": "中等"},
     ]}


def test_lane_view_picks_phase_lanes(monkeypatch):
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "_phase_lanes", lambda ph: {3: [1]})
    monkeypatch.setattr(S, "QUEUE_BY_LANE", True)
    v = S._lane_view(3, R, 2)
    assert v["scope"] == "lane" and v["lanes"] == [1]
    assert v["queue_m"] == 10.0 and v["raw_occupancy"] == 0.2 and v["flow_vpm"] == 13.0
    assert v["level"] == "medium"


def test_lane_view_multi_lane_max_and_sum(monkeypatch):
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "_phase_lanes", lambda ph: {3: [1, 2]})
    monkeypatch.setattr(S, "QUEUE_BY_LANE", True)
    v = S._lane_view(3, R, 2)
    assert v["queue_m"] == 18.0 and v["flow_vpm"] == 35.0 and v["vehicle_count"] == 5 and v["level"] == "high"


def test_lane_view_falls_back_to_camera(monkeypatch):
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "_phase_lanes", lambda ph: {3: [1]})
    monkeypatch.setattr(S, "QUEUE_BY_LANE", True)
    v = S._lane_view(3, {k: x for k, x in R.items() if k != "zone_results"}, 2)
    assert v["scope"] == "camera" and v["queue_m"] == 18.0
    monkeypatch.setattr(S, "QUEUE_BY_LANE", False)
    assert S._lane_view(3, R, 2)["queue_m"] == 18.0          # 回退開關


def test_phase_measure_uses_lane_queue(monkeypatch):
    from api.routes import signal_shadow as S
    from api.routes import congestion
    monkeypatch.setattr(S, "_phase_lanes", lambda ph: {3: [1]})
    monkeypatch.setattr(S, "QUEUE_BY_LANE", True)
    monkeypatch.setattr(S, "PHASE_CAMERAS", {2: [3]})
    monkeypatch.setattr(S, "_cam_ok", lambda r, now=None: True)
    monkeypatch.setattr(S, "ARRIVAL_FROM_EVENTS", False)
    monkeypatch.setattr(S, "QUEUE_CLAMP_ENABLED", False)
    monkeypatch.setitem(congestion.congestion_results, 3, R)
    m = S._phase_measure(2)
    assert m["queue_m"] == 10.0


def test_lane_view_no_phase_lane_reports_nothing(monkeypatch):
    """NE-2 09-19 14:59 重畫後只剩別條路的壅塞區 —— 不可退回整台相機(那是別條路的排隊)。"""
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "_phase_lanes", lambda ph: {3: [1]})
    monkeypatch.setattr(S, "QUEUE_BY_LANE", True)
    r = dict(R, zone_results=[z for z in R["zone_results"] if z["lane_no"] == 2])
    v = S._lane_view(3, r, 2)
    assert v["scope"] == "none" and v["queue_m"] is None and v["raw_occupancy"] is None
