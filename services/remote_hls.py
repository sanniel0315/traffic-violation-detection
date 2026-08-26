#!/usr/bin/env python3
"""遠端省流大緩衝 HLS 重打包器 + 獨立 HTTP 服務。

為什麼:go2rtc 的 HLS 視窗只有 ~1 秒(低延遲設計),對高延遲外網線(實測 592ms、
抖動到 700ms+)緩衝不夠 → hls.js 追不上就凍。這裡用 ffmpeg 把 cam_X_lofi(已是
350k H.264)以 -c copy(不重編碼、便宜)重打包成「大視窗 HLS(預設 10 秒)」,
hls.js 就能緩衝 ~5 秒吃掉抖動 → 順(代價:延遲幾秒,遠端監看可接受)。

🛑 用獨立行程 + ThreadingHTTPServer 服務,不經 traffic-api —— traffic-api 的
   async 事件迴圈被吃滿 CPU 的分析卡住,連 HLS segment 都會延遲(見 go2rtc.py 註解)。

用法(systemd):python3 services/remote_hls.py
env:
  REMOTE_HLS_CAMS=2,3,4,5     要打包哪幾台(對應 cam_X_lofi)
  REMOTE_HLS_PORT=8013        HTTP 服務埠
  REMOTE_HLS_DIR=/dev/shm/hls_remote
  REMOTE_HLS_LIST_SIZE=10     視窗segment數(x hls_time=秒數)
"""
import os
import shutil
import signal
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

CAMS = [c.strip() for c in os.getenv("REMOTE_HLS_CAMS", "2,3,4,5").split(",") if c.strip()]
PORT = int(os.getenv("REMOTE_HLS_PORT", "8013"))
ROOT = os.getenv("REMOTE_HLS_DIR", "/dev/shm/hls_remote")
LIST_SIZE = int(os.getenv("REMOTE_HLS_LIST_SIZE", "10"))
HLS_TIME = os.getenv("REMOTE_HLS_TIME", "1")
RTSP = os.getenv("GO2RTC_RTSP", "rtsp://127.0.0.1:8554").rstrip("/")
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

_procs: dict = {}
_stop = threading.Event()


def _src_name(cam: str) -> str:
    return f"cam_{cam}_lofi"


def _rtsp_timeout_args() -> list:
    """RTSP socket 逾時參數(μs),依 ffmpeg 版本挑選項名。
    🛑 ffmpeg 4.x 叫 -stimeout;5.0+ 改名 -timeout。4.x 的 -timeout 是
    RTSP「listen 模式」逾時 —— 誤用會去佔 8554 監聽 → Address already in use
    秒退 → 無限重生(2026-08-26 實測)。"""
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-h", "demuxer=rtsp"],
                             capture_output=True, text=True, timeout=10).stdout
        opt = "-stimeout" if "stimeout" in out else "-timeout"
        return [opt, "10000000"]
    except Exception:
        return []


_TIMEOUT_ARGS = _rtsp_timeout_args()


def _spawn(cam: str) -> subprocess.Popen:
    name = _src_name(cam)
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    cmd = [
        FFMPEG, "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-fflags", "+genpts",
        # 🛑 RTSP socket 逾時。沒有這個的話 TCP 半死(對端重啟/斷線未 FIN)
        #    read 會永久 block:2026-08-26 實測四支從 12:00 卡到 23:00,
        #    行程活著所以 watchdog 不重啟 → 遠端監控全部轉圈圈。
        *_TIMEOUT_ARGS,
        "-i", f"{RTSP}/{name}",
        "-c", "copy", "-f", "hls",
        "-hls_time", HLS_TIME,
        "-hls_list_size", str(LIST_SIZE),
        "-hls_flags", "delete_segments+omit_endlist+independent_segments",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", os.path.join(d, "seg_%05d.ts"),
        os.path.join(d, "index.m3u8"),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=open(f"/tmp/remote_hls_{name}.err", "wb"))


def _repack_loop():
    spawn_ts: dict = {}
    stall_sec = float(os.getenv("REMOTE_HLS_STALL_SEC", "20") or 20)
    while not _stop.is_set():
        for cam in CAMS:
            p = _procs.get(cam)
            # 🛑 卡住偵測:行程活著但 index.m3u8 太久沒更新(= 上游斷流且 read
            #    卡死)就殺掉,下一輪重生。只看行程死活不夠(2026-08-26 教訓)。
            if p is not None and p.poll() is None:
                m3u8 = os.path.join(ROOT, _src_name(cam), "index.m3u8")
                try:
                    last = os.stat(m3u8).st_mtime
                except OSError:
                    last = 0.0
                fresh = max(last, spawn_ts.get(cam, 0.0))
                if time.time() - fresh > stall_sec:
                    print(f"[remote_hls] {_src_name(cam)} 卡住 {int(time.time()-fresh)}s,砍掉重生", flush=True)
                    try:
                        p.kill()
                    except Exception:
                        pass
                    p = _procs[cam] = None
            if p is None or p.poll() is not None:
                try:
                    _procs[cam] = _spawn(cam)
                    spawn_ts[cam] = time.time()
                    print(f"[remote_hls] (re)spawn {_src_name(cam)}", flush=True)
                except Exception as e:
                    print(f"[remote_hls] spawn {cam} failed: {e}", flush=True)
        _stop.wait(3.0)


class Handler(SimpleHTTPRequestHandler):
    # 🛑 HTTP/1.1 keep-alive:高延遲外網線(592ms)上,每個 segment 若重開 TCP,
    #    每次都要等一個 RTT 握手 → segment 抓不贏播放 → 卡。keep-alive 讓 hls.js
    #    重用連線,免掉每段握手 → 高延遲線也追得上。
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True   # 客戶端中斷,靜默收掉,別噴 traceback

    def log_message(self, *a):
        pass


def main():
    os.makedirs(ROOT, exist_ok=True)
    for cam in CAMS:
        shutil.rmtree(os.path.join(ROOT, _src_name(cam)), ignore_errors=True)
    threading.Thread(target=_repack_loop, daemon=True).start()

    def _sig(*_a):
        _stop.set()
        for p in _procs.values():
            try:
                p.terminate()
            except Exception:
                pass
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[remote_hls] serving {ROOT} on :{PORT}, cams={CAMS} window={LIST_SIZE}x{HLS_TIME}s", flush=True)
    try:
        srv.serve_forever()
    finally:
        _sig()


if __name__ == "__main__":
    main()
