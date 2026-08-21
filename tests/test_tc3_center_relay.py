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
    port = int(os.environ["SIGNAL_TC3_CENTER_LISTEN_PORT"])
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


def test_hardwarestatus_bit14_flip_to_center():
    """0F04(HardwareStatus=0x4000) 轉給中央前,bit14 應被翻成 0(補償廠商寫反),
    且重組的碼框 CKS 合法、其他欄位不變。"""
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
    try:
        S._forward_controller_frame_to_center(frame, rec)
        got = far.recv(1024)
    finally:
        S._close_center()
        far.close()
    # 中央收到的框:解出來 HardwareStatus 應為 0x0000(bit14 被翻掉)
    out = S.decode_frame(got)
    assert out is not None and out.get("cks_ok"), "校正後的框 CKS 不合法"
    fields = S._decode_fields("0F04", got.hex(" ").upper())
    hs = next((x["value"] for x in (fields or []) if x["name"] == "HardwareStatus"), None)
    assert hs == 0x0000, f"bit14 沒被翻:HardwareStatus={hs}"
    # seq/addr 不變
    assert out.get("seq") == 0x63 and out.get("addr") == 0xFFFF
    print("test_hardwarestatus_bit14_flip_to_center: PASS")


if __name__ == "__main__":
    test_center_relay_bidirectional_and_inject()
    test_hardwarestatus_bit14_flip_to_center()
    print("ALL PASS")
