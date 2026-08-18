#!/usr/bin/env python3
"""證明停擺看門狗會動:假來源送幾框後閉嘴,抄錄器要斷線重連。

同時回歸驗證 byte stuffing 的 _find_etx —— INFO 內含 AA AA CC 不可被切短。
"""
import importlib.util
import os
import socket
import sys
import threading
import time
import types

sys.stdout.reconfigure(encoding='utf-8')

# 假來源要先起來才知道 port,所以環境變數在載入模組前設定
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", 0))
srv.listen(4)
PORT = srv.getsockname()[1]

os.environ["SIGNAL_TC3_HOST"] = "127.0.0.1"
os.environ["SIGNAL_TC3_PORT"] = str(PORT)
os.environ["SIGNAL_TC3_STALL_TIMEOUT"] = "4"     # 測試縮短
os.environ["SIGNAL_TC3_PEER_TIMEOUT"] = "6"

for name in ("api", "api.routes", "api.utils"):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
au = types.ModuleType("api.routes.auth")
au.get_current_user = lambda: None
sys.modules["api.routes.auth"] = au
sd = types.ModuleType("api.utils.shutdown")


class _Ev:
    def __init__(self):
        self._e = threading.Event()

    def is_set(self):
        return self._e.is_set()

    def set(self):
        self._e.set()

    def wait(self, t=None):
        return self._e.wait(t)


sd.shutdown_event = _Ev()
sys.modules["api.utils.shutdown"] = sd

spec = importlib.util.spec_from_file_location("sig", "api/routes/signal_tc3.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def build(info: bytes, seq: int = 1) -> bytes:
    """組一個合法碼框(含 byte stuffing 與 CKS)。"""
    stuffed = bytearray()
    for b in info:
        stuffed.append(b)
        if b == 0xAA:
            stuffed.append(0xAA)      # 協定 2-8:INFO 內的 AA 要重複
    body = bytes([0xAA, 0xBB, seq, 0xFF, 0xFF]) + \
        (10 + len(stuffed)).to_bytes(2, "big") + bytes(stuffed) + b"\xaa\xcc"
    cks = 0
    for b in body:
        cks ^= b
    return body + bytes([cks])


# 5F03 時相:PhaseOrder SignalMap SignalCount SubPhaseID StepID StepSec(2B) 6 燈
PHASE = build(bytes([0x5F, 0x03, 0x00, 0x5F, 0x06, 0x01, 0x01]) +
              (0x2D).to_bytes(2, "big") + bytes([0x44, 0x44, 0x81, 0x81, 0x44, 0x81]))
# StepSec = 0xAACC —— 這正是會讓天真 find(AA CC) 切短的值
TRICKY = build(bytes([0x5F, 0x03, 0x00, 0x5F, 0x06, 0x01, 0x01]) +
               (0xAACC).to_bytes(2, "big") + bytes([0x44, 0x44, 0x81, 0x81, 0x44, 0x81]),
               seq=2)

accepts = []


def handle(c, n):
    try:
        c.sendall(PHASE)
        time.sleep(0.2)
        c.sendall(TRICKY)
        print(f"  [假來源] 第 {n} 次連線:送 2 框後閉嘴")
        # 之後什麼都不送,也不關 socket —— 就是要重現「連線還在但沒資料」
        while True:
            time.sleep(1)
    except Exception:
        pass


def server():
    while True:
        try:
            c, _ = srv.accept()
        except OSError:
            return
        accepts.append(time.time())
        threading.Thread(target=handle, args=(c, len(accepts)), daemon=True).start()


threading.Thread(target=server, daemon=True).start()

S.start_recorder()
print(f"  抄錄器啟動,假來源 127.0.0.1:{PORT},stall timeout 4 秒")

t0 = time.time()
while time.time() - t0 < 20 and len(accepts) < 2:
    time.sleep(0.5)

with S._lock:
    st = dict(S._state)
sd.shutdown_event.set()
srv.close()
time.sleep(0.5)

print(f"  連線次數 {len(accepts)}  frames_total {st['frames_total']} "
      f"cks_bad {st['cks_bad']} stalls {st['stalls']} reconnects {st['reconnects']}")
print(f"  peer_note: {st['peer_note']}")

fail = []
if len(accepts) < 2:
    fail.append("停擺後沒有重連")
if st["stalls"] < 1:
    fail.append("stalls 沒有累加")
if st["frames_total"] < 2:
    fail.append(f"訊框數 {st['frames_total']} < 2,byte stuffing 那框被切掉了")
if st["cks_bad"]:
    fail.append(f"cks_bad={st['cks_bad']},切框位置錯了")
lat = st.get("latest")
if not lat or lat.get("phase", {}).get("step_sec") != 0xAACC:
    fail.append(f"StepSec=0xAACC 的框沒正確解出 latest={lat and lat.get('phase')}")

print("  結果:", "❌ " + " / ".join(fail) if fail else "✅ 全過")
sys.exit(1 if fail else 0)
