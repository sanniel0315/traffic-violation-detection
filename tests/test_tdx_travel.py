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
