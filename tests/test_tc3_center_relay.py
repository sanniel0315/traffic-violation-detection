"""中央電腦中繼(都三透明中繼 + 注入鉤子)本機 loopback 測試,不需硬體。

驗證:
  1. 中央→我們→控制器:中央送的 frame 原封轉給控制器 socket
  2. 控制器→我們→中央:_tee_to_center 把控制器 bytes 原封轉給中央
  3. 我方自報:_send_to_center 把我們自己組的 frame 上報中央(非寫死純通透)
  4. 中央下傳的 frame 有被側錄進 _frames 且標 src=center
"""
import os
import socket
import time

# env 必須在 import 前設好(模組載入時讀)
os.environ.setdefault("AUTH_SECRET", "test_secret_center_relay_only")
os.environ["SIGNAL_TC3_CENTER_RELAY"] = "1"
os.environ["SIGNAL_TC3_CENTER_LISTEN_HOST"] = "127.0.0.1"
os.environ["SIGNAL_TC3_CENTER_LISTEN_PORT"] = "51701"
os.environ.setdefault("SIGNAL_TC3_ENABLED", "0")   # 不要起真的抄錄器(不連現場)

from api.routes import signal_tc3 as S  # noqa: E402


def _connect_center(port: int, tries: int = 20):
    last = None
    for _ in range(tries):
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=2)
        except OSError as e:            # server 還沒 listen 起來 → 重試
            last = e
            time.sleep(0.1)
    raise last


def test_center_relay_bidirectional_and_inject():
    # 🛑 不能只靠 import 前設 env:整包 pytest 一起跑時,api.routes.signal_tc3
    #    往往已經被前面的測試載過了,模組常數在那時就定死成預設的 0.0.0.0:1001,
    #    這裡再設 env 也不會生效(單獨跑這一支才會過 —— 典型的測試隔離陷阱)。
    #    直接覆寫模組常數,兩種跑法都成立。
    port = int(os.environ["SIGNAL_TC3_CENTER_LISTEN_PORT"])
    S.CENTER_RELAY_ENABLED = True
    S.CENTER_LISTEN_HOST = "127.0.0.1"
    S.CENTER_LISTEN_PORT = port
    # 中繼迴圈實際看的是 _conn["center_relay"](為了執行期能開關),
    # 只改 CENTER_RELAY_ENABLED 常數的話迴圈會一直閒置、不 bind port。
    S._conn["center_relay"] = True
    # 假控制器:socketpair,一端塞進 _sock_ref 當「控制器連線」,另一端我們檢查收到什麼
    ctrl_ours, ctrl_far = socket.socketpair()
    ctrl_far.settimeout(2)
    S._sock_ref["sock"] = ctrl_ours

    S.start_center_relay()
    center = _connect_center(port)
    center.settimeout(2)
    # 等 server accept 掛好 _center_sock_ref
    for _ in range(20):
        if S._center_sock_ref.get("sock") is not None:
            break
        time.sleep(0.1)
    assert S._center_state["connected"] is True

    # 1) 中央→控制器:中央送 5F45 查詢,控制器端應原封收到
    q = S.build_frame(0xFFFF, 1, bytes([0x5F, 0x45, 0x05]))
    center.sendall(q)
    got = ctrl_far.recv(1024)
    assert got == q, f"中央→控制器轉發不符: {got.hex()} != {q.hex()}"

    # 2) 控制器→中央:tee 原封轉發
    rep = S.build_frame(0xFFFF, 2, bytes([0x5F, 0xC5, 0x00]))
    S._tee_to_center(rep)
    got2 = center.recv(1024)
    assert got2 == rep, f"控制器→中央 tee 不符: {got2.hex()} != {rep.hex()}"

    # 3) 我方自報(不是轉發控制器的):_send_to_center
    mine = S.build_frame(0xFFFF, 3, bytes([0x5F, 0x03, 0x01, 0x02]))
    assert S._send_to_center(mine) is True
    got3 = center.recv(1024)
    assert got3 == mine, "我方自報上中央不符"

    # 4) 中央下傳的 frame 有被側錄且標 src=center
    time.sleep(0.2)
    center_frames = [f for f in list(S._frames) if f.get("src") == "center"]
    assert center_frames, "中央下傳的 frame 沒被側錄"
    assert S._center_state["center_frames"] >= 1

    # 收尾:關 socket + 讓 daemon 中繼執行緒乾淨退出(避免 interpreter 關閉噪音)
    center.close()
    ctrl_far.close()
    ctrl_ours.close()
    S._close_center()
    S.shutdown_event.set()
    time.sleep(1.2)          # 讓 _center_relay_loop 的 accept/recv 逾時後看到旗標退出
    S.shutdown_event.clear()


def test_hardwarestatus_raw_is_default():
    """🛑 預設必須是純通透:控制器報 0x4000,中央就要收到 0x4000。

    2026-09-07 現場實證:翻掉 bit14 送 0x0000 → 中央硬體狀態顯示異常;
    原封送 0x4000(bit14=1 控制器就緒)→ 中央顯示正常。所以「不竄改」才是
    安全的預設值。這條測試就是防止預設值被改回 flip14。
    """
    import socket as _s
    info = bytes([0x0F, 0x04, 0x40, 0x00])
    frame = S.build_frame(0xFFFF, 0x63, info)
    rec = S.decode_frame(frame)
    old = dict(S._hw_center_mode)
    S._hw_center_mode["mode"] = "raw"        # 預設值,顯式寫出來讓測試自足
    ours, far = _s.socketpair()
    far.settimeout(2)
    S._center_sock_ref["sock"] = ours
    try:
        S._forward_controller_frame_to_center(frame, rec)
        got = far.recv(1024)
    finally:
        S._close_center()
        far.close()
        S._hw_center_mode.update(old)
    assert got == frame, "raw 模式必須原封轉發,一個位元都不能改"
    print("test_hardwarestatus_raw_is_default: PASS")


def test_hardwarestatus_bit14_flip_to_center():
    """flip14 這條退路本身要能動(顯式指定才會用到,已不是預設值)。

    🛑 這條驗的是「程式碼路徑正確」,不是「應該這樣送中央」——
       實證顯示翻轉會害中央顯示異常,見 test_hardwarestatus_raw_is_default。
    """
    import socket as _s
    # 造一個 0F04 主動回報:INFO = 0F 04 + HardwareStatus(0x4000 big-endian)
    info = bytes([0x0F, 0x04, 0x40, 0x00])
    frame = S.build_frame(0xFFFF, 0x63, info)
    rec = S.decode_frame(frame)
    assert rec is not None and rec.get("code") == "0F04" and rec.get("cks_ok")
    # 假中央 socket
    ours, far = _s.socketpair()
    far.settimeout(2)
    S._center_sock_ref["sock"] = ours
    _old_mode = dict(S._hw_center_mode)
    S._hw_center_mode["mode"] = "flip14"     # 預設已改 raw,要測這條得顯式指定
    try:
        S._forward_controller_frame_to_center(frame, rec)
        got = far.recv(1024)
    finally:
        S._close_center()
        far.close()
        S._hw_center_mode.update(_old_mode)
    # 中央收到的框:解出來 HardwareStatus 應為 0x0000(bit14 被翻掉)
    out = S.decode_frame(got)
    assert out is not None and out.get("cks_ok"), "校正後的框 CKS 不合法"
    fields = S._decode_fields("0F04", got.hex(" ").upper())
    if fields is None:
        # 🛑 utc-tc3 解碼庫只裝在現場(UTC_TC3_PATH 預設 /home/ubuntu/utc-tc3),
        #    開發機沒有 → _decode_fields 回 None。這不是程式壞掉,是環境沒有庫。
        #    先前這裡直接斷言,導致本機跑測試永遠紅一個,久了就會習慣性忽略 ——
        #    真的壞掉時反而看不出來。改成明確 skip,並且退而驗原始位元組。
        assert got[9:11] == bytes(2), (
            f"bit14 沒被翻:HardwareStatus 位元組={got[9:11].hex()}")
        import pytest as _pt
        _pt.skip("utc-tc3 解碼庫不在本機(UTC_TC3_PATH=%s);已改驗原始位元組"
                 % S.UTC_TC3_PATH)
    hs = next((x["value"] for x in fields if x["name"] == "HardwareStatus"), None)
    assert hs == 0x0000, f"bit14 沒被翻:HardwareStatus={hs}"
    # seq/addr 不變
    assert out.get("seq") == 0x63 and out.get("addr") == 0xFFFF
    print("test_hardwarestatus_bit14_flip_to_center: PASS")


def test_cabinet_open_sets_bit9_to_center(monkeypatch):
    """🛑 我方電子鎖的門一開,中央必須看得到「機箱門開啟」。

    號誌機自己的 bit9 對不上現場(機箱開著它仍為 0),中央要看到「有人開箱」
    只能靠我方的門磁。

    🛑 2026-09-11 更正位元位置:中央產生告警名稱時是**反讀**這個欄位。
       raw 模式下線路值就是我方寫的值,所以要讓中央讀到 bit9,線路上必須是
       **bit1**。先前放 bit9 → 中央反讀成 bit1,畫面顯示「記憶體異常」——
       機箱門開啟變成記憶體故障(現場實證,已報備中央/廠商)。
       bit9 與 bit1 在位元組交換下剛好互換,所以這不是偶發。

    這條測試釘住三件事:
      · 門開 → 線路上 bit1 = 1,且中央反讀後看得到 bit9
      · 只加不減 → 控制器原本報的位元(這裡是 bit14)不能被蓋掉
      · 門關 → 不加任何東西(raw 模式應原封轉發)
    """
    import socket as _s
    info = bytes([0x0F, 0x04, 0x40, 0x00])          # HardwareStatus = 0x4000(bit14)
    frame = S.build_frame(0xFFFF, 0x63, info)
    rec = S.decode_frame(frame)
    old_mode = dict(S._hw_center_mode)
    S._hw_center_mode["mode"] = "raw"

    def _send_once():
        ours, far = _s.socketpair()
        far.settimeout(2)
        S._center_sock_ref["sock"] = ours
        try:
            S._forward_controller_frame_to_center(frame, dict(rec))
            return far.recv(1024)
        finally:
            S._close_center()
            far.close()

    try:
        # 門開
        S._cab_cache.update({"open": True, "ts": 9e9})
        got = _send_once()
        out = S.decode_frame(got)
        assert out and out.get("cks_ok"), "補完 bit9 的框 CKS 不合法"
        hs = (S._unstuff(got[7:-3])[2] << 8) | S._unstuff(got[7:-3])[3]
        assert hs & (1 << 1), "門開了但線路上的 bit1 沒被設起來"
        swapped = ((hs & 0xFF) << 8) | ((hs >> 8) & 0xFF)
        assert swapped & (1 << 9), (
            "中央反讀之後看不到 bit9 機箱門開啟 —— 位元擺錯位置")
        assert not (swapped & (1 << 1)), (
            "中央反讀後不該出現 bit1 記憶體錯誤 —— 那正是先前的誤報")
        assert hs & (1 << 14), "只加不減:控制器原本的 bit14 被蓋掉了"

        # 門關 → raw 應原封轉發
        S._cab_cache.update({"open": False, "ts": 9e9})
        assert _send_once() == frame, "門關著卻改了轉發內容"
    finally:
        S._hw_center_mode.update(old_mode)
        S._cab_cache.update({"open": False, "ts": 0.0})


if __name__ == "__main__":
    test_center_relay_bidirectional_and_inject()
    test_hardwarestatus_raw_is_default()
    test_hardwarestatus_bit14_flip_to_center()
    test_cabinet_open_sets_bit9_to_center(None)
    print("ALL PASS")
