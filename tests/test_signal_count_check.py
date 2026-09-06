"""人工計數對照表 —— 90% 準確度條款的證據產生器。"""
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
                 "speed_kmh REAL, direction TEXT)")
    rows = []
    # 本地 10:00~10:10 = UTC 02:00~02:10
    for i in range(40):
        rows.append((3, "2026-09-06 02:%02d:%02d" % (i // 6, (i % 6) * 10), 30.0, "INOUT"))
    for i in range(12):
        rows.append((3, "2026-09-06 02:%02d:00" % i, 30.0, "EXIT"))
    for i in range(11):
        rows.append((3, "2026-09-06 02:%02d:30" % i, 30.0, "IN"))
    conn.executemany("INSERT INTO traffic_events VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_count_check_lists_every_machine_measurement(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S
    db = tmp_path / "v.db"
    _mk_viol_db(db)
    monkeypatch.setattr(S, "_VIOL_DB", str(db))
    r = await S.count_check(camera_id=3, since="2026-09-06T10:00:00",
                            until="2026-09-06T10:10:00", manual=-1, _user=None)
    keys = {m["key"] for m in r["methods"]}
    assert {"events_all", "events_exit", "events_in"} <= keys
    by = {m["key"]: m for m in r["methods"]}
    # 視窗是 10:00~10:10(本地)= 02:00~02:10(UTC),跨出視窗的列要被排除:
    # INOUT 40 筆全在 02:00~02:06;EXIT 造 12 筆但只有 02:00~02:09 的 10 筆在內;
    # IN 造 11 筆同理只有 10 筆在內。
    assert by["events_all"]["count"] == 60
    assert by["events_exit"]["count"] == 10
    assert by["events_in"]["count"] == 10
    assert "尚未輸入人工計數" in r["verdict"]
    assert r["playback"]["camera"] == "cam_3"


@pytest.mark.asyncio
async def test_count_check_judges_against_manual(tmp_path, monkeypatch):
    """人工計數是唯一的裁判:10 台 → 只取 EXIT 完全命中,全部列差 6 倍。"""
    from api.routes import signal_shadow as S
    db = tmp_path / "v2.db"
    _mk_viol_db(db)
    monkeypatch.setattr(S, "_VIOL_DB", str(db))
    r = await S.count_check(camera_id=3, since="2026-09-06T10:00:00",
                            until="2026-09-06T10:10:00", manual=10, _user=None)
    by = {m["key"]: m for m in r["methods"]}
    assert by["events_exit"]["accuracy_pct"] == 100.0 and by["events_exit"]["meets_90"] is True
    assert by["events_all"]["meets_90"] is False
    assert by["events_in"]["meets_90"] is True      # 10 vs 10 → 100%
    assert "只取 EXIT" in r["verdict"]


@pytest.mark.asyncio
async def test_count_check_says_so_when_nothing_meets_90(tmp_path, monkeypatch):
    """沒有任何一種達標時要明講,不可以挑一個最接近的當答案。"""
    from api.routes import signal_shadow as S
    db = tmp_path / "v3.db"
    _mk_viol_db(db)
    monkeypatch.setattr(S, "_VIOL_DB", str(db))
    r = await S.count_check(camera_id=3, since="2026-09-06T10:00:00",
                            until="2026-09-06T10:10:00", manual=100, _user=None)
    assert all(not m.get("meets_90") for m in r["methods"])
    assert "沒有任何一種量測達到" in r["verdict"]
