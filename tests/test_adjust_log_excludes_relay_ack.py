"""中間層代控制器補回的 0F80(user='relay-ack')不可被當成「我方下發的調整」。

🛑 2026-09-17 現場問題:啟用中繼 ACK 之後,我方每收到控制器一則主動框就回一則
   0F80,這些框同樣寫成 src='self',結果:
     - /adjust-log 把它們列成「時制調整」,筆數灌大好幾十倍
     - 診斷卡片的來源統計把它們歸進 manual(人工),看起來像有人在狂下命令
   兩邊都要排掉;真正的下發(5F1C / 5F15 …)不受影響。
"""
import os
import sqlite3
import time

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE signal_frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, src TEXT, code TEXT, seq INTEGER, addr INTEGER,
        len INTEGER, cks_ok INTEGER, raw TEXT, user TEXT, sent_hw INTEGER)""")
    now = time.time()
    rows = [
        # 真正的下發:演算法換相
        (now - 60, "self", "5F1C", 10, 1, 4, 1, "AA BB 0A 01 04 5F 1C 02 23 AA CC 00", "algorithm", 1),
        # 中繼 ACK:每 2 秒一則,量遠大於真正的下發
        *[(now - 50 + i, "self", "0F80", 20 + i, 1, 4, 1,
           "AA BB 14 01 04 0F 80 5F 03 AA CC 00", "relay-ack", 1) for i in range(30)],
    ]
    conn.executemany(
        "INSERT INTO signal_frames (ts,src,code,seq,addr,len,cks_ok,raw,user,sent_hw) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_adjust_log_and_source_stats_ignore_relay_ack(tmp_path, monkeypatch):
    from api.routes import signal_shadow as sh

    db = tmp_path / "v.db"
    _make_db(str(db))
    monkeypatch.setattr(sh, "_VIOL_DB", str(db), raising=False)

    # include_blocked=0:只看真正送出的框,不要混進「判該換相但沒送」的補列
    res = sh.adjust_log(hours=1, include_blocked=0, _user={"username": "t"})
    items = res.get("items") or res.get("rows") or []
    codes = [it.get("code") for it in items]
    assert "0F80" not in codes, "中繼 ACK 不是時制調整,不可列入"
    assert codes.count("5F1C") == 1, "真正的下發要保留"


def test_source_stats_do_not_count_relay_ack_as_manual(tmp_path, monkeypatch):
    """診斷用的來源統計(ours.manual / ours.algorithm)同樣要排掉中繼 ACK。"""
    import re
    import pathlib

    src = pathlib.Path("api/routes/signal_shadow.py").read_text(encoding="utf-8")
    # 兩支查詢都必須帶上排除條件 —— 這是 SQL 字串,只能用來源檢查守住
    hits = re.findall(r"src='self'[^\"]*user<>'relay-ack'", src)
    assert len(hits) >= 2, "adjust-log 與來源統計兩支查詢都要排除 relay-ack"
