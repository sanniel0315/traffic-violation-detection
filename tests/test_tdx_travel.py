"""TDX eTag 旅行時間:里程解析、時區、Flows 彙整與配對篩選、落庫去重、時段彙總。
不打真的 TDX(需要金鑰、有配額),HTTP 層全部 monkeypatch。"""
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_km_and_local_time_parsing():
    from services import tdx_travel as T
    assert T._km("7K+500") == 7.5 and T._km("12K+0") == 12.0 and T._km("3.25") == 3.25
    assert T._km(None) is None and T._km("abc") is None
    # +08:00 → 本地(不帶時區),跟號誌側一致
    assert T._local_iso("2026-09-05T10:05:00+08:00").startswith("2026-09-05T10:05:00")
    assert T._local_iso(None) is None


def test_fetch_live_aggregates_flows_and_filters_pairs(monkeypatch):
    from services import tdx_travel as T
    payload = [
        {"ETagPairID": "P1", "Direction": "E", "DataCollectTime": "2026-09-05T10:00:00+08:00",
         "Flows": [{"VehicleType": 31, "TravelTime": 100, "SpaceMeanSpeed": 90, "VehicleCount": 40},
                   {"VehicleType": 32, "TravelTime": 120, "SpaceMeanSpeed": 80, "VehicleCount": 10}]},
        {"ETagPairID": "P9", "Direction": "W", "DataCollectTime": "2026-09-05T10:00:00+08:00",
         "TravelTime": 300, "SpaceMeanSpeed": 60, "VehicleCount": 5},
    ]
    monkeypatch.setattr(T, "_get_json", lambda path, params: payload)
    monkeypatch.setattr(T, "PAIRS", [])
    rows = T.fetch_live()
    assert len(rows) == 2
    p1 = rows[0]
    assert p1["travel_time_sec"] == 110 and p1["speed_kmh"] == 85 and p1["vehicle_count"] == 50
    assert rows[1]["travel_time_sec"] == 300
    monkeypatch.setattr(T, "PAIRS", ["P9"])
    assert [r["pair_id"] for r in T.fetch_live()] == ["P9"]


def test_store_dedups_same_data_time_and_summarize(tmp_path, monkeypatch):
    from services import tdx_travel as T
    db = str(tmp_path / "v.db")
    rows = [{"pair_id": "P1", "direction": "E", "travel_time_sec": 110.0, "speed_kmh": 85.0,
             "vehicle_count": 50, "data_time": "2026-09-05T10:00:00+08:00"},
            {"pair_id": "P1", "direction": "E", "travel_time_sec": 130.0, "speed_kmh": 75.0,
             "vehicle_count": 60, "data_time": "2026-09-05T10:05:00+08:00"}]
    assert T.store(rows, db) == 2
    assert T.store(rows, db) == 0                 # 同配對同資料時間不重複寫
    n = sqlite3.connect(db).execute("SELECT COUNT(*) FROM tdx_travel_time").fetchone()[0]
    assert n == 2
    monkeypatch.setattr(T, "PAIRS", [])
    sm = T.summarize("2026-09-05T09:00:00", "2026-09-05T12:00:00", db)
    assert sm["method"] == "measured" and sm["n"] == 2
    assert sm["avg_travel_time_sec"] == 120.0 and sm["pairs"][0]["vehicles"] == 110
    # 沒表的 DB 不可以炸(現場沒申請 TDX 時 /report 照常)
    empty = str(tmp_path / "e.db"); sqlite3.connect(empty).close()
    assert T.summarize("2026-09-05T09:00:00", "2026-09-05T12:00:00", empty)["n"] == 0


def test_discover_ranks_pairs_spanning_interchange_first(monkeypatch):
    from services import tdx_travel as T
    pairs = [{"pair_id": "A", "start_km": 2.0, "end_km": 5.0},
             {"pair_id": "B", "start_km": 5.0, "end_km": 9.0},     # 跨過 7.5
             {"pair_id": "C", "start_km": 9.0, "end_km": 14.0},
             {"pair_id": "D", "start_km": 30.0, "end_km": 35.0}]   # 太遠
    monkeypatch.setattr(T, "list_pairs", lambda road_id=None: [dict(p) for p in pairs])
    out = T.discover(7.5, span_km=6.0)
    assert [p["pair_id"] for p in out] == ["B", "C", "A"] and out[0]["spans_center"] is True
    assert all(p["pair_id"] != "D" for p in out)


def test_start_is_noop_without_credentials(monkeypatch):
    from services import tdx_travel as T
    monkeypatch.setitem(T._state, "enabled", False)
    assert T.start() is False


def test_discover_by_coord_prefers_pair_spanning_the_site(monkeypatch):
    from services import tdx_travel as T
    site = (23.063772, 120.279169)      # 新市交流道路口(使用者地圖頁座標)
    pairs = [
        # 西邊門架 → 東邊門架,站點落在中間(跨過)
        {"pair_id": "SPAN", "start_lat": 23.0620, "start_lon": 120.2600, "end_lat": 23.0660, "end_lon": 120.3000},
        # 兩門架都在東邊,沒跨過但很近
        {"pair_id": "EAST", "start_lat": 23.0660, "start_lon": 120.3000, "end_lat": 23.0700, "end_lon": 120.3400},
        # 20 公里外
        {"pair_id": "FAR", "start_lat": 23.2000, "start_lon": 120.5000, "end_lat": 23.2100, "end_lon": 120.5400},
    ]
    monkeypatch.setattr(T, "list_pairs", lambda road_id=None: [dict(p) for p in pairs])
    out = T.discover_by_coord(*site, radius_km=8.0)
    assert [p["pair_id"] for p in out] == ["SPAN", "EAST"]
    assert out[0]["spans_site"] is True and out[0]["offset_km"] is not None and out[0]["offset_km"] < 0.5
    assert out[1]["spans_site"] is False


def test_list_pairs_does_not_send_server_side_filter(monkeypatch):
    """TDX 的 ETagPair 端點不吃 RoadID 的 $filter —— 加了會回 400(2026-09-06 實測)。"""
    from services import tdx_travel as T
    seen = {}

    def fake_get(path, params):
        seen["path"], seen["params"] = path, dict(params)
        return [{"ETagPairID": "01F0-01F1", "RoadID": "000010",
                 "StartETag": {}, "EndETag": {}},
                {"ETagPairID": "08F0-08F1", "RoadID": "000080",
                 "StartETag": {}, "EndETag": {}}]

    monkeypatch.setattr(T, "_get_json", fake_get)
    out = T.list_pairs("000080")
    assert "$filter" not in seen["params"], "不可以下 $filter,TDX 會回 400"
    # 過濾改在本地做:只留該國道
    assert [x["pair_id"] for x in out] == ["08F0-08F1"]


def test_fetch_live_remembers_the_path_that_worked(monkeypatch):
    """即時路徑用候選清單試,試到通就記住 —— TDX 有流量限制,不能每次重試一輪。"""
    from services import tdx_travel as T
    T._live_path["value"] = None
    calls = []

    def fake_get(path, params):
        calls.append(path)
        if path != T.LIVE_PATH_CANDIDATES[1]:
            return []                      # 前面的候選都沒資料
        return [{"ETagPairID": "08F0-08F1", "RoadID": "000080",
                 "TravelTime": 60, "SpaceMeanSpeed": 80, "VehicleCount": 5}]

    monkeypatch.setattr(T, "_get_json", fake_get)
    T.fetch_live("000080")
    assert T._live_path["value"] == T.LIVE_PATH_CANDIDATES[1]
    n_first = len(calls)
    T.fetch_live("000080")
    assert len(calls) == n_first + 1, "記住之後只該打一次,不可以再試一輪"


def test_fetch_live_stops_on_rate_limit(monkeypatch):
    """打到 429 就停手,不要把剩下的候選也試完 —— 那會直接把配額打光。"""
    from services import tdx_travel as T
    T._live_path["value"] = None
    calls = []

    def fake_get(path, params):
        calls.append(path)
        raise RuntimeError("TDX 429 流量限制,暫停抓取: %s" % path)

    monkeypatch.setattr(T, "_get_json", fake_get)
    try:
        T.fetch_live("000080")
    except RuntimeError as e:
        assert "429" in str(e)
    assert len(calls) == 1, "第一個候選就撞到 429,不該繼續試"
