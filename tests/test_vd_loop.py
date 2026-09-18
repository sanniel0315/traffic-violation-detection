"""VD 線圈協定解析 —— 用 2026-09-18 現場實際收到的框驗證。

協定:《高快速公路交通控制系統中央電腦軟體雲端化通訊協定》3.2(框)、10H(週期性資料)。
🛑 本協定**沒有 DLE 跳脫、沒有框尾**,框長完全靠 LEN。資料裡的 0x10 是一般資料。
"""
import os
import time
from datetime import datetime

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")

# 使用者提供的一筆 + 探測時實際收到的兩筆
F_USER = bytes.fromhex("10018fffff00161610000000000012 0b000105312201394b0000000064 0493".replace(" ", ""))
# 🛑 直接貼現場探測輸出的原字串(含空白),不要手打 —— 第一版手打多了一個 00,三支測試因此誤報
F1 = bytes.fromhex("10 01 f9 ff ff 00 16 16 10 00 00 00 00 00 12 0c 2e 01 04 3f 25 00 00 00 00 00 00 00 ac 02 79".replace(" ", ""))
F2 = bytes.fromhex("10 01 fa ff ff 00 16 16 10 00 00 00 00 00 12 0c 2f 01 04 42 26 02 2e 3e 00 00 00 00 68 03 d2".replace(" ", ""))


def test_frames_split_and_lrc():
    from api.routes.vd_loop import split_frames, lrc
    frames, rest, junk = split_frames(F1 + F2)
    assert frames == [F1, F2] and rest == b"" and junk == 0
    for f in (F_USER, F1, F2):
        assert lrc(f[3:7]) == f[7], "表頭 LRC"
        assert lrc(f[:-1]) == f[-1], "整框 LRC"


def test_split_handles_partial_and_noise():
    """半截的框要留到下一次;前面的雜訊要丟掉但不可吃掉後面的好框。"""
    from api.routes.vd_loop import split_frames
    frames, rest, junk = split_frames(b"\x00\x33" + F1 + F2[:10])
    assert frames == [F1]
    assert rest == F2[:10]
    assert junk == 2
    frames2, rest2, _ = split_frames(rest + F2[10:])
    assert frames2 == [F2] and rest2 == b""


def test_data_0x10_is_not_a_frame_boundary():
    """表頭 LRC=16 後面緊接指令碼 10 —— 若誤當跳脫字元,整框會解錯。"""
    from api.routes.vd_loop import split_frames, decode_10h
    frames, _, _ = split_frames(F_USER)
    assert len(frames) == 1
    d = decode_10h(frames[0][8:-1])
    assert d is not None and d["hour"] == 11 and d["minute"] == 0


def test_decode_user_frame_matches_protocol():
    from api.routes.vd_loop import decode_10h
    d = decode_10h(F_USER[8:-1])
    assert (d["day"], d["hour"], d["minute"]) == (18, 11, 0)
    assert d["hw_status"] == "00000000"
    L = d["lanes"][0]
    assert (L["small_n"], L["small_kmh"], L["small_len_m"]) == (5, 49, 3.4)
    assert (L["large_n"], L["large_kmh"], L["large_len_m"]) == (1, 57, 7.5)
    assert L["trailer_n"] == 0
    assert L["headway_s"] == 10.0 and L["occ_pct"] == 4


def test_ack_echoes_seq_and_checksum():
    """ACK 要帶回同一個 SEQ;代理依文件算出使用者那筆的 ACK = 10 06 8F FF FF 99。"""
    from api.routes.vd_loop import ack_for
    assert ack_for(F_USER) == bytes.fromhex("10068fffff99")
    assert ack_for(F1) == bytes.fromhex("1006f9ffffef")
    assert ack_for(F2) == bytes.fromhex("1006faffffec")


def test_length_mismatch_is_rejected_not_guessed():
    from api.routes.vd_loop import decode_10h
    assert decode_10h(F1[8:-2]) is None


def test_fault_lane_is_marked_not_counted():
    """文件:線圈/偵測器故障時整車道填 FF —— 不可當成 255 輛。"""
    from api.routes.vd_loop import decode_10h
    text = bytes([0x10, 0, 0, 0, 0, 0, 18, 11, 0, 1]) + b"\xff" * 12
    L = decode_10h(text)["lanes"][0]
    assert L["fault"] is True and L["small_n"] is None


def test_clock_offset_detects_slow_device():
    """實測:13:16:00 收到、時間戳 12:46 → 設備慢 30 分鐘。"""
    from api.routes.vd_loop import device_clock_offset
    recv = datetime(2026, 9, 18, 13, 16, 0).timestamp()
    assert device_clock_offset(recv, 18, 12, 46) == 1800.0


def test_parse_devices():
    from api.routes.vd_loop import parse_devices
    assert parse_devices("WN下匝道@10.42.39.60:1002, 壞格式, x@h:abc") == [
        ("WN下匝道", "10.42.39.60", 1002)]
    assert parse_devices("") == []


def test_seq_step_wraps_in_terminal_range():
    """終端設備序號在 0x80~0xFF 循環:FF 的下一個是 80。"""
    from api.routes.vd_loop import seq_step
    assert seq_step(0xAA, 0xAB) == 1
    assert seq_step(0xAA, 0xAC) == 2, "中間漏一框"
    assert seq_step(0xAA, 0xAA) == 0, "重送同一框"
    assert seq_step(0xFF, 0x80) == 1, "循環回 80 不是漏框"
    assert seq_step(0xFE, 0x81) == 3


def test_completeness_finds_todays_real_gaps():
    """重現 2026-09-18 現場:14:06(AB)與 14:10(AF)兩框設備有送、我們沒存到。"""
    from datetime import datetime
    from api.routes.vd_loop import completeness
    seqs = [(3, 0xA8), (4, 0xA9), (5, 0xAA), (7, 0xAC), (8, 0xAD), (9, 0xAE), (11, 0xB0)]
    rows = [(datetime(2026, 9, 18, 14, m).timestamp(), "10 01 %02X FF FF" % s) for m, s in seqs]
    out = completeness(rows)
    assert out["missed"] == 2
    assert out["missed_minutes"] == ["14:06", "14:10"]
    assert out["received"] == 7 and out["rate_pct"] == 77.8


def test_store_failure_spools_instead_of_losing(tmp_path, monkeypatch):
    """寫庫失敗要先暫存、之後補寫 —— 不可以丟資料。"""
    import api.routes.vd_loop as V
    monkeypatch.setattr(V, "_SPOOL", str(tmp_path / "spool.jsonl"))
    V._spool("WN", 1789700000.0, F1)
    stored = []
    monkeypatch.setattr(V, "_store", lambda name, ts, fr, d, off: stored.append((name, fr)))
    assert V._flush_spool("WN") == 1
    assert stored == [("WN", F1)]
    assert (tmp_path / "spool.jsonl").read_text() == "", "補寫成功後要從暫存移除"


def test_spool_keeps_frame_when_db_still_failing(tmp_path, monkeypatch):
    import api.routes.vd_loop as V
    monkeypatch.setattr(V, "_SPOOL", str(tmp_path / "spool.jsonl"))
    V._spool("WN", 1789700000.0, F1)

    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(V, "_store", boom)
    assert V._flush_spool("WN") == 0
    assert "WN" in (tmp_path / "spool.jsonl").read_text(), "還寫不進去就要留著"


def _seed_vd(tmp_path, monkeypatch, minutes):
    """種 N 分鐘的線圈資料:每分鐘 小 5、大 1;攝影機每分鐘 4 輛(其中大 2)。"""
    import sqlite3
    from datetime import datetime, timedelta
    import api.routes.vd_loop as V
    monkeypatch.setattr(V, "_DB", str(tmp_path / "vd.db"))
    monkeypatch.setattr(V, "_DEVICES_RAW", "WN@1.2.3.4:1002")
    c = V._conn()
    t0 = datetime(2026, 9, 18, 10, 0)
    cam = {}
    for i in range(minutes):
        recv = t0 + timedelta(minutes=i + 1)          # hh:mm+1 收到 = hh:mm 那一分鐘
        c.execute("INSERT INTO vd_minute(device,recv_ts,recv_iso,dev_time,clock_offset_sec,hw_status,lane,fault,"
                  "small_n,small_kmh,large_n,large_kmh,trailer_n,headway_s,occ_pct,raw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  ("WN", recv.timestamp(), recv.isoformat(), "x", 1800.0, "00000000", 1, 0,
                   5, 50, 1, 40, 0, 10.0, 5, "10 01 %02X FF FF" % (0x80 + i)))
        cam[(t0 + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M")] = {"n": 4, "large": 2}
    c.commit(); c.close()
    monkeypatch.setattr(V, "_camera_minutes", lambda a, b: {"camera": 4, "lanes": [1], "minutes": cam})
    return V, t0


def test_vd_status_range_and_bucket(tmp_path, monkeypatch):
    """查詢功能:since/until 取區間、bucket 分組;大小車分開對照。"""
    V, t0 = _seed_vd(tmp_path, monkeypatch, 30)
    out = V.vd_status(device="", minutes=60, since=t0.isoformat(), until=(t0.replace(minute=30)).isoformat(),
                      bucket=5, _user="t")
    assert out["available"] and out["bucket"] == 5
    assert len(out["items"]) == 6, "30 分鐘 / 每 5 分一組 = 6 組"
    first = out["items"][-1]                          # 最早那組
    assert first["vd_small"] == 25 and first["vd_large"] == 5 and first["vd_total"] == 30
    assert first["cam_small"] == 10 and first["cam_large"] == 10
    sm = out["summary"]
    assert sm["vd_small"] == 150 and sm["cam_small"] == 60 and sm["small_pct"] == -60.0
    assert sm["vd_large"] == 30 and sm["cam_large"] == 60 and sm["large_pct"] == 100.0


def test_vd_status_rejects_bad_range(tmp_path, monkeypatch):
    V, t0 = _seed_vd(tmp_path, monkeypatch, 5)
    out = V.vd_status(device="", minutes=60, since="2026-09-18T12:00:00", until="2026-09-18T11:00:00",
                      bucket=1, _user="t")
    assert out["available"] is False


def test_camera_minute_key_includes_date():
    """跨日查詢:攝影機分鐘鍵要含日期,不然兩天同一分鐘會加在一起。"""
    import inspect
    import api.routes.vd_loop as V
    assert "%%Y-%%m-%%d %%H:%%M" in inspect.getsource(V._camera_minutes)


def test_time_set_frame_follows_protocol():
    """02H 對時:格式照協定文件頁 19;框/LRC 的算法用文件頁 12 的範例(04 07 → LRC 31)核對。"""
    from api.routes.vd_loop import time_set_frame, lrc
    f = time_set_frame(0x23, datetime(2026, 9, 18, 19, 30, 5))
    assert f.hex(" ") == "10 01 23 ff ff 00 08 08 02 07 ea 09 12 13 1e 05 ce"
    assert f[2] <= 0x7F                                   # 中央端 SEQ 範圍 0~127
    assert time_set_frame(0xA3, datetime(2026, 1, 1))[2] == 0x23
    h = b"\xff\xff\x00\x02"
    g = bytes([0x10, 0x01, 0x23]) + h + bytes([lrc(h)]) + b"\x04\x07"
    assert (g + bytes([lrc(g)])).hex(" ") == "10 01 23 ff ff 00 02 02 04 07 31"


def test_split_recognizes_device_ack_and_nak():
    """設備對我方命令回的 ACK(文件頁 14 範例 10 06 24 FF FF 32)/ NAK 要切得出來,資料框照舊。"""
    from api.routes.vd_loop import split_frames, lrc
    ack = bytes.fromhex("100624ffff32")
    nak = bytes([0x10, 0x15, 0x25, 0xFF, 0xFF, 0x00, 0x20])
    nak += bytes([lrc(nak)])
    frames, rest, junk = split_frames(ack + F1 + nak + F2)
    assert [fr[1] for fr in frames] == [0x06, 0x01, 0x15, 0x01]
    assert frames[1] == F1 and frames[3] == F2 and rest == b"" and junk == 0
    # LRC 錯的「假 ACK」不收,當雜訊
    frames, _, _ = split_frames(bytes.fromhex("100624ffff33") + F1)
    assert [fr[1] for fr in frames] == [0x01]


def test_timesync_refuses_when_not_connected(monkeypatch):
    from api.routes import vd_loop
    monkeypatch.setattr(vd_loop, "_DEVICES_RAW", "X@127.0.0.1:1")
    monkeypatch.setattr(vd_loop, "_state", {})
    r = vd_loop.vd_timesync(device="", _user=None)
    assert r["ok"] is False and "X" not in vd_loop._timesync_req


# 2026-09-18 15:44:56 現場收到的 1FH+02H(逆向車事件)原框
F_1F = bytes.fromhex("10 01 92 ff ff 00 11 11 1f 02 00 00 00 00 00 01 12 0f 0e 39 00 53 ff fa 02 e1".replace(" ", ""))


def test_decode_wrong_way_event_frame():
    from api.routes.vd_loop import decode_1f02, decode_10h, lrc, split_frames
    frames, rest, junk = split_frames(F_1F)
    assert frames == [F_1F] and lrc(F_1F[:-1]) == F_1F[-1]
    ev = decode_1f02(F_1F[8:-1])
    assert ev == [{"day": 18, "hour": 15, "minute": 14, "second": 57, "lane_id": 0,
                   "car_length_m": 8.3, "car_interval_s": 6553.0, "car_type": 2}]
    assert decode_10h(F_1F[8:-1]) is None
    assert decode_1f02(F_1F[8:-2]) is None                 # 長度不符不猜


def test_completeness_counts_event_frames_as_received():
    """1FH 事件框也佔序號:10H 序號 91→93 中間夾 92 的事件框,不是漏收。"""
    from api.routes.vd_loop import completeness
    t0 = datetime(2026, 9, 18, 15, 44).timestamp()
    rows = [(t0, "10 01 91 FF FF"), (t0 + 56, F_1F.hex(" ").upper()), (t0 + 60, "10 01 93 FF FF")]
    assert completeness(rows)["missed"] == 0
    assert completeness([rows[0], rows[2]])["missed"] == 1   # 舊算法(只看 10H)會誤判
