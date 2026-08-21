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


if __name__ == "__main__":
    test_center_relay_bidirectional_and_inject()
    print("test_tc3_center_relay: PASS")
