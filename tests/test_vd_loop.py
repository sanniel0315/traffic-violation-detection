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
