"""人工計數對照表 —— 90% 準確度條款的證據產生器。

🛑 只列我方量測。進出線(IN/OUT/EXIT)是給 OPAC 的,不列、不對照(使用者 2026-09-14)。
"""
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


def _mk_viol_db(path):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE traffic_events (camera_id INT, created_at TEXT, "
                 "speed_kmh REAL, direction TEXT, lane_no INT)")
    rows = []
    # 本地 10:00~10:10 = UTC 02:00~02:10;我方計數 12 筆,只有 02:00~02:09 的 10 筆在窗內
    for i in range(12):
        rows.append((3, "2026-09-06 02:%02d:00" % i, 30.0, "INOUT", 1))
    # 同台別條匝道的 lane —— 不算
    for i in range(8):
        rows.append((3, "2026-09-06 02:%02d:20" % i, 30.0, "straight", 2))
    # 進出線 —— 不算
    for d in ("IN", "EXIT"):
        for i in range(9):
            rows.append((3, "2026-09-06 02:%02d:30" % i, 30.0, d, 1))
    conn.executemany("INSERT INTO traffic_events VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


@pytest.fixture
def S(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S
    db = tmp_path / "v.db"
    _mk_viol_db(db)
    monkeypatch.setattr(S, "_VIOL_DB", str(db))
    monkeypatch.setattr(S, "_phase_lanes", lambda p: {3: [1]} if p == 1 else {5: [1]})
    return S


def test_count_check_lists_only_our_measurements(S):
    r = S.count_check(camera_id=3, since="2026-09-06T10:00:00",
                      until="2026-09-06T10:10:00", manual=-1, _user=None)
    keys = {m["key"] for m in r["methods"]}
    assert "flow_events" in keys
    assert not keys & {"events_all", "events_exit", "events_in"}, "進出線不列"
    by = {m["key"]: m for m in r["methods"]}
    assert by["flow_events"]["count"] == 10
    assert "尚未輸入人工計數" in r["verdict"]
    assert r["playback"]["camera"] == "cam_3"
    assert "EXIT" not in r["playback"]["hint"]


def test_count_check_judges_against_manual(S):
    """人工計數是唯一的裁判。"""
    r = S.count_check(camera_id=3, since="2026-09-06T10:00:00",
                      until="2026-09-06T10:10:00", manual=10, _user=None)
    by = {m["key"]: m for m in r["methods"]}
    assert by["flow_events"]["accuracy_pct"] == 100.0 and by["flow_events"]["meets_90"] is True
    assert "我方車流計數" in r["verdict"]


def test_count_check_says_so_when_nothing_meets_90(S):
    """沒有任何一種達標時要明講,不可以挑一個最接近的當答案。"""
    r = S.count_check(camera_id=3, since="2026-09-06T10:00:00",
                      until="2026-09-06T10:10:00", manual=100, _user=None)
    assert all(not m.get("meets_90") for m in r["methods"])
    assert "沒有任何一種量測達到" in r["verdict"]
