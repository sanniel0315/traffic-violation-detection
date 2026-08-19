"""把偵測過的 frame（含 bbox）即時編碼推 RTSP 給 go2rtc，前端走 WebRTC 拉。

為什麼需要(2026-08-18 量出來的)
────────────────────────────────────────────────────────────────────────
疊加原本走 MJPEG。現場 87 對外上行實測只有 1.2~1.5 Mbps(下行 28.6),而一路
1280 寬的疊加 MJPEG 就要 15.8~19.8 Mbps —— 遠端看必定不順,連 /api/health 都
擠不出去。MJPEG 想同時「順」又「看得清」在這條線上做不到:保 15fps 要壓進
1.4 Mbps 只能降到 w=320,那是縮圖。
H.264 同畫質約只要 MJPEG 的 1/20 —— 1 Mbps 就能給 720p 順暢畫面。

曾經被停用的原因,以及現在憑什麼重開
────────────────────────────────────────────────────────────────────────
stream.py 原本寫死 `_ANNOTATED_STREAM_CAM_IDS = set()`,註解只留一句
「confirmed annotated_streamer triggers SEGV race」。
本次重開做了兩件對應處理:
  ① _spawn() 的 subprocess.Popen 在 fork/exec 時要讀整份 environ,而偵測執行緒
     重連時會寫 os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"](= setenv)。
     glibc 的 setenv 可能 realloc environ 陣列,跟併發的讀就是 use-after-free
     → 原生 SEGV。2026-08-18 已在 api/utils/camera_stream.py 用
     capture_open_guard 把「寫」序列化,這裡把「spawn」也納入同一把鎖。
  ② 繪製(resize + 畫框 + tobytes)原本在 reader thread 做,等於每一幀都在
     偵測的取像路徑上加工 —— 既拖慢分析,也讓 reader 與 ffmpeg 生命週期糾纏。
     改成 reader 只放最新畫面的參照,全部加工搬到 pacer thread。
🛑 這兩項都不是「猜測的修法」,①有今天實際修掉同類 race 的證據,②是把工作
   移出關鍵路徑。但仍必須在 104 壓測數小時確認不再 SEGV 才可上 87。

架構(不變)
- reader thread 每幀 push_frame → 只存參照,不做加工
- worker 每次推論後 update_detections → bbox 不卡 detection
- pacer 用 wall-clock 加工並餵 ffmpeg → 跟 detection/reader 都解耦

環境變數
    ANNOTATED_STREAM_CAMS=7,8      要開哪幾台(空 = 全部關閉,預設關閉)
    ANNOTATED_STREAM_BITRATE=1M    編碼位元率
    ANNOTATED_STREAM_WIDTH=1280 / _HEIGHT=720 / _FPS=25
    ANNOTATED_STREAM_ENCODER=libx264   Jetson 可試 h264_v4l2m2m(硬體編碼)
"""

import logging
import os
from collections import deque
import subprocess
import threading
import time
from typing import Optional

from api.utils.camera_stream import capture_open_guard

import cv2
import numpy as np

log = logging.getLogger(__name__)

_streamers: dict = {}
_streamers_lock = threading.Lock()

STREAM_WIDTH = int(os.getenv("ANNOTATED_STREAM_WIDTH", "1280") or 1280)
STREAM_HEIGHT = int(os.getenv("ANNOTATED_STREAM_HEIGHT", "720") or 720)
# 0 = 自動(啟動時先量供幀率再決定,建議);>0 = 固定值
# 🛑 編碼率必須等於供幀率,否則同一張會被不均勻重複送出 = 抖動。
#    而供幀率是「來源 fps ÷ (decode_skip_frames+1)」,每台不一樣 ——
#    87 實測 cam_2 來源 60fps→供幀 20,cam_3/4/5 來源 30fps→供幀 10。
#    用同一個設定值一定有台會抖,所以改成各自量各自的。
STREAM_FPS = int(os.getenv("ANNOTATED_STREAM_FPS", "0") or 0)
STREAM_BITRATE = os.getenv("ANNOTATED_STREAM_BITRATE", "1M") or "1M"
# libx264 = 軟體(到處都能跑);hw = Jetson 硬體編碼(nvv4l2h264enc)
# 87 實測 960x540@20fps,只算編碼器本身:
#     軟體 libx264            20% 一顆核
#     硬體 nvv4l2h264enc      9% 一顆核   ← BGRx 直入,不經 videoconvert
# 🛑 一定要餵 BGRx。餵 BGR 就得插 videoconvert,那層 CPU 會把省下來的吃光
#    (先前量到「硬體只差 16%」就是踩這個)。
# 🛑 gst-plugins-bad 的 rtspclientsink 在 87 上沒有,所以 GStreamer 只負責編碼,
#    再用管線交給 ffmpeg 純封裝成 RTSP(-c copy,不重編碼,CPU 可忽略)。
STREAM_ENCODER = os.getenv("ANNOTATED_STREAM_ENCODER", "libx264") or "libx264"


def _hw_available() -> bool:
    """實際跑一次極小的編碼管線來判斷硬體路徑可不可用。

    🛑 不要用「/dev/v4l2-nvenc 存不存在/是不是真裝置」來判斷 ——
       那個節點在 Jetson host 上本來就是 1,3(等於 /dev/null)的 stub,
       但硬體編碼是好的(87 實測產出 383KB、log 顯示 NvVideo: NVENC)。
       用節點判斷會在 87 誤判成不可用。
    環境差異很大,設定值不能當保證:
       87  traffic-api 跑在 host(systemd)→ 有 gst-launch-1.0,硬體可用
       104 traffic-api 跑在 Docker      → 容器裡連 gst-launch-1.0 都沒有
    設了 hw 卻沒退回的話,結果是「完全沒有串流」而不是「慢一點」,所以一定要探。
    """
    import shutil
    if not shutil.which("gst-launch-1.0"):
        return False
    probe = ("gst-launch-1.0 -q videotestsrc num-buffers=3 ! "
             "video/x-raw,width=320,height=240,format=BGRx ! nvvidconv ! "
             "'video/x-raw(memory:NVMM),format=NV12' ! nvv4l2h264enc ! fakesink")
    try:
        r = subprocess.run(["sh", "-c", probe], timeout=20,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


_want_hw = STREAM_ENCODER.lower() in ("hw", "nvv4l2h264enc", "nvenc")
USE_HW = _want_hw and _hw_available()
if _want_hw and not USE_HW:
    print("⚠️ [annot] 要求硬體編碼但環境不支援(缺 gst-launch-1.0 或 /dev/v4l2-nvenc)"
          " → 退回 libx264 軟體編碼", flush=True)
    STREAM_ENCODER = "libx264"
# 畫面延遲多久才送出。🛑 這是「框追得準」的關鍵:
#   pacer 若拿最新畫面配最新偵測,那組偵測是好幾百毫秒前那張畫面算出來的,
#   框就會系統性落後車子(2026-08-18 實測 shared_frames age_sec 0.5,
#   在 30fps 順暢畫面上非常明顯;MJPEG 只有 10fps 時比較看不出來)。
#   把畫面壓在延遲線裡等偵測追上,再用時間戳配對,框就畫在「它自己那張」上。
#   代價是畫面延遲這麼久 —— 監看用途可以接受,換來的是框準。
# 0 = 自動(依實測偵測延遲自己調,建議);>0 = 固定值
STREAM_DELAY_SEC = float(os.getenv("ANNOTATED_STREAM_DELAY", "0") or 0)
DELAY_MIN, DELAY_MAX = 0.25, 1.20
# 配到的偵測比這張畫面舊超過這麼久就不畫框。0 = 不啟用(預設)。
# 🛑 預設不啟用是產品決定,不是技術取捨:
#    「要不要看框」是使用者用「原始畫面 / 辨識疊加」自己選的,
#    系統不該在他選了疊加之後又自作主張把框藏起來。
#    這個旋鈕留著給「寧可沒有框也不要錯位框」的場景,要用再開。
# 參考數據(87 2026-08-19 長時間 A/B):框誤差中位約 98ms,但 p95 常 500~2000ms
#    —— 分析率低且忽高忽低(10 次取樣 6.6~17.3),一卡住最近鄰只能配到很舊那組。
#    也就是說 87 的疊加框「有時候會嚴重落後」,這是分析率的物理結果,
#    使用者選疊加時應該知道,但不該由系統替他決定要不要看。
MAX_GAP_SEC = float(os.getenv("ANNOTATED_STREAM_MAX_GAP", "0") or 0)
# 同時再推一條「不畫框」的低頻寬串流 cam_N_lite。
# 🛑 為什麼需要:87 對外上行只有 1.4 Mbps,而原始 H.264 是原生 1080p 5.08 Mbps
#    —— 使用者切「原始畫面」遠端還是會頓,只有疊加那條(0.81 Mbps)塞得下。
#    「完整的影像順暢」代表兩種模式都要順,所以原始也要有低頻寬版本。
#    重用已經解碼並縮好的畫面,只多一次編碼(硬體約 13% 一顆核),
#    比讓 go2rtc 另外 decode+encode 便宜得多。
#    區網使用者不受影響:前端只在窄頻時才改用 lite,平常仍是原生 1080p。
# 🛑 預設關閉:這條沒有被驗證過。2026-08-19 在 87 開啟時兩條串流一起變得
#    不穩(0.00~0.61 Mbps、間隔跳到 6 秒),而且 go2rtc 必須先宣告 cam_N_lite
#    空串流才收得到 —— 104 沒宣告,結果子行程一直死掉重生(寫入只有 8.1 fps)。
#    要用之前必須先把「單條就穩」這件事做到。
LITE_ENABLED = os.getenv("ANNOTATED_STREAM_LITE", "0") != "0"


def enabled_camera_ids() -> set:
    """要開 H.264 疊加串流的相機。預設空 = 全部關閉,必須明確指定才會啟用。"""
    raw = (os.getenv("ANNOTATED_STREAM_CAMS", "") or "").strip()
    out = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out

_LABELS = {
    "car": "Car", "motorcycle": "Moto", "truck": "Truck", "bus": "Bus",
    "heavy_truck": "Heavy", "light_truck": "Light",
    "person": "Person", "bicycle": "Bike",
}
_BBOX_COLOR = (0, 216, 100)


def _draw_overlay(frame: np.ndarray, detections: list, sx: float = 1.0, sy: float = 1.0) -> np.ndarray:
    if not detections:
        return frame
    out = frame
    drawn = False
    for det in detections:
        cls = det.get("class_name", "")
        label = _LABELS.get(cls)
        if not label:
            continue
        b = det.get("bbox") or {}
        if not all(k in b for k in ("x1", "y1", "x2", "y2")):
            continue
        if not drawn:
            out = frame.copy()
            drawn = True
        x1 = int(b["x1"] * sx); y1 = int(b["y1"] * sy)
        x2 = int(b["x2"] * sx); y2 = int(b["y2"] * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), _BBOX_COLOR, 2)
        cv2.putText(out, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BBOX_COLOR, 2)
    return out


class AnnotatedStreamer:
    def __init__(self, camera_id: int, width: int = 1280, height: int = 720, fps: int = 25):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        # fps=0 → 先不決定,等 _pacer_loop 量到供幀率再定案(見 _resolve_fps)
        self.fps = max(1, int(fps)) if fps else 0
        self.frame_interval = 1.0 / (self.fps or 20)

        # 每個變體一個編碼子行程:annotated=畫框,lite=不畫框
        self._variants = ["annotated"] + (["lite"] if LITE_ENABLED else [])
        self.procs: dict = {v: None for v in self._variants}
        self._proc_lock = threading.Lock()
        # 🛑 每個變體一條寫入執行緒,各自一個「最新畫面」插槽。
        #    先前是 pacer 同一條執行緒依序寫兩個管線 —— 其中一個阻塞(RTSP 推送
        #    卡住很常見)就把另一個也拖住,兩條串流一起變 0.00~0.40 Mbps(實測)。
        #    分開之後一條卡住不影響另一條,而且 pacer 永遠不會被寫入阻塞。
        self._slots: dict = {v: None for v in self._variants}
        self._slot_lock = threading.Lock()

        # reader 只放「最新原始畫面的參照」,加工全部在 pacer 做 ——
        # 不可以在 reader thread 做 resize/畫框,那是偵測的取像路徑。
        # 延遲線:(ts, frame, sx, sy),依時間排序。長度剛好蓋住延遲 + 一點餘裕。
        # 環形緩衝要蓋得住最大延遲 + 餘裕
        self._frames = deque(maxlen=max(4, int(self.fps * (DELAY_MAX + 0.5)) + 2))
        self._frame_lock = threading.Lock()
        self._last_sent_ts = 0.0

        # 偵測結果也要帶時間戳,才配得起來。留幾組,配對時取「不晚於該幀」的最新一組。
        self._dets: deque = deque(maxlen=32)      # (ts, detections)
        self._dets_lock = threading.Lock()

        # 供幀率監看。🛑 pacer 是定速餵 ffmpeg 的(rawvideo 的 -r 是固定值),
        #    所以 reader 供幀率一旦低於這個速率,同一張畫面就會被不均勻地重複送出
        #    —— 使用者看到的就是「抖動」。2026-08-18 實測:來源 30fps、
        #    decode_skip_frames=2 只解 1/3 → 供幀 10fps、pacer 送 25fps,
        #    每張重複 2~3 次,畫面明顯抖。改成 decode_skip=0 + fps 30 才 1:1 順。
        #    這裡把供幀率量出來並在偏低時明講,不要讓人再從畫面去猜。
        self._supply_n = 0
        self._supply_t0 = time.monotonic()
        self._supply_fps = 0.0
        self._warned_supply = False
        # 對齊品質統計:偵測延遲(結果何時才追上那張畫面)與實際配對誤差
        self._dropped = 0                     # 因為太舊而不畫框的幀數
        # 寫入端統計:實際送出幾幀、單次 write 花多久。
        # 🛑 這是分辨「我們產不出幀」與「下游送不出去」的關鍵:
        #    write_fps 掉下來 = pacer/writer 被卡(GIL 或 CPU);
        #    write_fps 正常但觀看端有大間隔 = 卡在 gst/ffmpeg/go2rtc 那側。
        #    87 實測疊加串流間隔會跳到 1.6~5.3 秒,但 104 同一份程式連跑 20 小時穩定,
        #    沒有這個數字就只能猜。
        self._wr = {}                         # variant -> (n, t0, fps, max_write_ms)
        self._det_lat = deque(maxlen=120)     # update_detections 時 now - 該幀 ts
        self._match_gap = deque(maxlen=300)   # 送出時 該幀 ts - 配到的偵測 ts
        self._stopped = False
        self._restart_count = 0
        self._max_restart = 200

        self._pacer = threading.Thread(target=self._pacer_loop, name=f"annot-pacer-{camera_id}", daemon=True)
        self._pacer.start()
        for _v in self._variants:
            threading.Thread(target=self._writer_loop, args=(_v,),
                             name=f"annot-{_v}-{camera_id}", daemon=True).start()

    def _spawn(self, variant: str = "annotated"):
        rtsp_url = f"rtsp://127.0.0.1:8554/cam_{self.camera_id}_{variant}"
        if USE_HW:
            gst = (
                f"gst-launch-1.0 -q fdsrc ! "
                f"rawvideoparse width={self.width} height={self.height} "
                f"format=bgrx framerate={self.fps}/1 ! "
                # 🛑 caps 一定要加引號:括號在 sh -c 裡會被當成子 shell → 語法錯誤
                f"nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! "
                f"nvv4l2h264enc bitrate={self._bitrate_bps()} insert-sps-pps=true "
                # 🛑 關鍵幀間隔 1 秒。設 2 秒的話 MSE 觀看端要等最多 2 秒才拿得到
                #    第一個可解碼的段,看起來就像「連上了但沒畫面」。
                f"iframeinterval={self.fps} ! h264parse ! "
                # 🛑 一定要用 mpegts 承載,不要送裸 H.264。裸流沒有時間戳,
                #    ffmpeg 只能照 -r 硬編 PTS,時序一亂 go2rtc 的 MSE 分段就壞掉
                #    —— 症狀是 go2rtc 明明持續收到 1.0 Mbps,送給觀看端卻只有
                #    0.0~0.4 Mbps 且忽有忽無(2026-08-19 在 87 實測)。
                f"mpegtsmux ! fdsink"
            )
            ff = (
                f"ffmpeg -hide_banner -loglevel error -f mpegts -i - "
                f"-c copy -f rtsp -rtsp_transport tcp {rtsp_url}"
            )
            cmd = ["sh", "-c", f"{gst} | {ff}"]
        else:
            cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", STREAM_ENCODER,
            "-preset", "ultrafast", "-tune", "zerolatency",
            "-b:v", STREAM_BITRATE, "-maxrate", STREAM_BITRATE,
            "-bufsize", STREAM_BITRATE,
            "-g", str(self.fps * 2),
            "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp",
            rtsp_url,
            ]
        try:
            # 🛑 進閘門:fork/exec 會讀整份 environ,而偵測重連那條路徑會 setenv。
            #    兩者併發就是 use-after-free → 原生 SEGV(這正是當初停用的原因)。
            with capture_open_guard():
                self.procs[variant] = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
            log.info(f"AnnotatedStreamer cam_{self.camera_id}/{variant} "
                     f"spawned pid={self.procs[variant].pid}")
        except Exception as e:
            log.error(f"AnnotatedStreamer cam_{self.camera_id}/{variant} spawn failed: {e}")
            self.procs[variant] = None

    def _bitrate_bps(self) -> int:
        """nvv4l2h264enc 的 bitrate 要 bps 整數,ffmpeg 吃 "1M" 這種字串。"""
        v = str(STREAM_BITRATE).strip().upper()
        try:
            if v.endswith("M"):
                return int(float(v[:-1]) * 1_000_000)
            if v.endswith("K"):
                return int(float(v[:-1]) * 1_000)
            return int(float(v))
        except Exception:
            return 1_000_000

    def _ensure_proc(self, variant: str = "annotated") -> bool:
        with self._proc_lock:
            proc = self.procs.get(variant)
            if proc is not None and proc.poll() is None:
                return True
            if self._restart_count >= self._max_restart:
                return False
            self._restart_count += 1
            self._spawn(variant)
            return self.procs.get(variant) is not None

    def _resolve_fps(self) -> bool:
        """還沒定案就先量供幀率。量到才開編碼器,避免編碼率與供幀率不匹配。"""
        if self.fps:
            return True
        if self._supply_fps <= 0:
            return False
        self.fps = max(1, min(30, int(round(self._supply_fps))))
        self.frame_interval = 1.0 / self.fps
        print(f"🎬 [annot] cam_{self.camera_id} 供幀 {self._supply_fps:.1f} fps "
              f"→ 編碼率定為 {self.fps} fps", flush=True)
        return True

    def _pacer_loop(self):
        next_tick = time.monotonic()
        while not self._stopped:
            next_tick += self.frame_interval
            if not self._resolve_fps():
                time.sleep(0.5)          # 還沒量到供幀率,先不要開編碼器
                next_tick = time.monotonic()
                continue
            item = self._take_delayed()
            if item is not None and item[0] != self._last_sent_ts:
                bufs = self._encode_frame(item)
                self._last_sent_ts = item[0]
                self._last_buf = bufs
            else:
                # 還沒有新的「等夠久」的畫面 → 重送上一張,維持定速餵給編碼器
                bufs = getattr(self, "_last_buf", None)
            if bufs:
                with self._slot_lock:
                    for variant in self._variants:
                        b = bufs.get(variant)
                        if b is not None:
                            self._slots[variant] = b
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def _writer_loop(self, variant: str):
        """把最新畫面定速餵給該變體的編碼子行程。

        每個變體一條 —— 寫入阻塞只會影響自己,不會拖累別條或 pacer。
        """
        while not self._stopped and not self.fps:
            time.sleep(0.5)              # 等 pacer 把 fps 定案
        next_tick = time.monotonic()
        while not self._stopped:
            next_tick += self.frame_interval
            with self._slot_lock:
                buf = self._slots.get(variant)
            if buf is not None and self._ensure_proc(variant):
                t_w = time.monotonic()
                try:
                    self.procs[variant].stdin.write(
                        memoryview(buf).cast("B") if hasattr(buf, "flags") else buf)
                except (BrokenPipeError, IOError, ValueError, AttributeError):
                    with self._proc_lock:
                        self.procs[variant] = None
                w_ms = (time.monotonic() - t_w) * 1000
                st = self._wr.get(variant) or [0, time.monotonic(), 0.0, 0.0]
                st[0] += 1
                st[3] = max(st[3], w_ms)
                dt = time.monotonic() - st[1]
                if dt >= 10.0:
                    st[2] = st[0] / dt
                    st[0], st[1], st[3] = 0, time.monotonic(), 0.0
                self._wr[variant] = st
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def update_detections(self, detections: list, ts: float = 0.0):
        """worker 呼叫。ts 是「這組結果所屬那張畫面」的時間戳,配對用。"""
        d_ts = float(ts) if ts else time.time()
        self._det_lat.append(max(0.0, time.time() - d_ts))
        with self._dets_lock:
            self._dets.append((d_ts, list(detections or [])))
        try:
            if detections:
                self._dbg_n = getattr(self, '_dbg_n', 0) + 1
                if self._dbg_n % 30 == 1:
                    d0 = detections[0]
                    print(f"[ann_dbg] cam={self.camera_id} n={len(detections)} keys={list(d0.keys())} cls={d0.get('class_name')!r} bbox={d0.get('bbox')}", flush=True)
        except Exception:
            pass

    def _dets_for(self, ts: float) -> list:
        """取「時間上最接近這張畫面」的一組偵測結果。

        🛑 為什麼是最接近而不是「不晚於」:
           偵測只有約 6.6 次/秒,而影像是 30 幀/秒 —— 兩次偵測之間的畫面只能沿用
           鄰近那組。若只往回看,誤差是 0~150ms 且永遠落後(2026-08-18 實測
           中位 70.5ms、p95 401.8ms);取最接近的話誤差變成 ±75ms 左右,
           而且不再是單向落後。延遲線本來就壓著畫面等,所以「稍後那組」通常已經到了。
        配不到任何一組(剛啟動)才回空 —— 寧可暫時沒有框,也不要亂畫。
        """
        with self._dets_lock:
            items = list(self._dets)
        best, best_ts = [], None
        best_d = None
        for d_ts, dets in items:
            d = abs(d_ts - ts)
            if best_d is None or d < best_d:
                best, best_ts, best_d = dets, d_ts, d
            elif d_ts > ts:
                break      # 已經越走越遠(deque 依時間遞增),不必再看
        # 配到的偵測跟這張畫面差多久 —— 這就是「框比車慢多少」
        # 記絕對誤差:現在可能配到稍後那組,負號沒有意義
        gap = abs(ts - best_ts) if best_ts is not None else None
        self._match_gap.append(gap if gap is not None else -1.0)
        if gap is None:
            return []          # 完全沒有偵測結果(剛啟動)
        if MAX_GAP_SEC > 0 and gap > MAX_GAP_SEC:
            self._dropped += 1
            return []
        return best

    def align_stats(self) -> dict:
        lat = sorted(self._det_lat)
        gap = sorted(g for g in self._match_gap if g >= 0)
        miss = sum(1 for g in self._match_gap if g < 0)
        def pct(a, q):
            return round(a[min(len(a) - 1, int(len(a) * q))] * 1000, 1) if a else None
        return {
            "delay_sec": round(self._effective_delay(), 3),
            "delay_mode": "fixed" if STREAM_DELAY_SEC > 0 else "auto",
            "supply_fps": round(self._supply_fps, 1),
            # 偵測結果要多久才追上那張畫面 → 延遲線至少要大於這個
            "det_latency_ms_med": pct(lat, 0.5), "det_latency_ms_p95": pct(lat, 0.95),
            # 實際送出時,框比畫面舊多少 → 這就是使用者看到的「框比車慢」
            "match_gap_ms_med": pct(gap, 0.5), "match_gap_ms_p95": pct(gap, 0.95),
            "unmatched": miss,
            # 因為配到的偵測太舊而選擇不畫框的幀數(門檻 MAX_GAP_SEC)
            "dropped_stale": self._dropped,
            # 實際送進編碼器的幀率(應等於設定 fps);單次 write 最久多少毫秒
            "write_fps": {k: round(v[2], 1) for k, v in self._wr.items()},
            "write_ms_max": {k: round(v[3], 1) for k, v in self._wr.items()},
            "max_gap_sec": MAX_GAP_SEC,
        }

    def _effective_delay(self) -> float:
        """延遲線要壓多久。

        🛑 這個值是「框準不準」與「畫面慢不慢」的唯一旋鈕,手動猜不準:
           太短 → 稍後那組偵測還沒到,只能沿用舊的,尾端誤差變大
                 (104 實測 0.45s:中位 28.7ms 但 p95 210.6ms)
           太長 → 畫面白白變慢
                 (0.6s:中位 44.8ms、p95 134.9ms)
        所以改成依實測偵測延遲自己調:取 p95 再加 100ms 餘裕。
        機器閒時自動變短(畫面更即時),忙時自動變長(框仍然準)。
        ANNOTATED_STREAM_DELAY 設 >0 就改用固定值。
        """
        if STREAM_DELAY_SEC > 0:
            return STREAM_DELAY_SEC
        lat = sorted(self._det_lat)
        if len(lat) < 20:
            return 0.6                      # 樣本不足時先用保守值
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        return min(DELAY_MAX, max(DELAY_MIN, p95 + 0.10))

    def _take_delayed(self):
        """從延遲線取一張「已經等夠久」的畫面。沒有就回 None(pacer 會重送上一張)。"""
        target = time.time() - self._effective_delay()
        with self._frame_lock:
            pick = None
            for item in self._frames:
                if item[0] <= target:
                    pick = item
                else:
                    break
            return pick

    def _encode_frame(self, item) -> Optional[bytes]:
        """畫框 + 轉 bytes。只在 pacer thread 呼叫。

        收到的畫面已經在 push_frame 裡脫離過 cap.read() 的緩衝(見那裡的說明)。
        """
        try:
            ts, frame, sx, sy = item
            out = {}
            if "lite" in self._variants:
                # 不畫框那條:直接用原畫面(_draw_overlay 只有真的要畫才複製,
                # 所以這裡拿到的 frame 不會被下面污染)
                lite = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) if USE_HW else frame
                if not lite.flags['C_CONTIGUOUS']:
                    lite = np.ascontiguousarray(lite)
                out["lite"] = lite
            dets = self._dets_for(ts)
            annotated = _draw_overlay(frame, dets, sx, sy)
            if USE_HW:
                # 硬體路徑吃 BGRx。在這裡轉,GStreamer 就不必插 videoconvert。
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2BGRA)
            if not annotated.flags['C_CONTIGUOUS']:
                annotated = np.ascontiguousarray(annotated)
            # 🛑 回傳 ndarray 而不是 tobytes():那是每幀 1.5~2MB 的複製
            #    (20fps = 31MB/s),而且複製時抓著 GIL,會跟偵測執行緒搶。
            #    寫入時直接用 buffer protocol,零複製。
            out["annotated"] = annotated
            return out
        except Exception:
            return None

    def push_frame(self, frame: np.ndarray, ts: float = 0.0):
        """reader thread 呼叫。

        🛑 這裡「一定」要複製或 resize,不可以只存參照。
           傳進來的 ndarray 是 cv2.VideoCapture.read() 的輸出,那塊記憶體會被
           解碼器回收重用;pacer 最多 40ms 後才碰它,就是跨執行緒 use-after-free。
           2026-08-18 實測:改成只存參照後 104 出現兩次
               Fatal Python error: Segmentation fault
               reader 停在 stream.py cap.read()、pacer 停在用那個 ndarray
           resize 本身就會配置新陣列,所以尺寸不同時不必再多複製一次;
           尺寸剛好相同時才走 copy()。
           繪製與 tobytes(較貴的部分)仍留在 pacer,不佔偵測的取像路徑。
        """
        if self._stopped or frame is None:
            return
        try:
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                out = cv2.resize(frame, (self.width, self.height))
                sx = self.width / w if w else 1.0
                sy = self.height / h if h else 1.0
            else:
                out = frame.copy()
                sx = sy = 1.0
        except Exception:
            return
        with self._frame_lock:
            self._frames.append((float(ts) if ts else time.time(), out, sx, sy))
        self._supply_n += 1
        dt = time.monotonic() - self._supply_t0
        # 一開始先用 5 秒快速定出一個值,之後才改成 10 秒視窗
        if dt >= (5.0 if self._supply_fps <= 0 else 10.0):
            self._supply_fps = self._supply_n / dt
            self._supply_n = 0
            self._supply_t0 = time.monotonic()
            if self._supply_fps < self.fps * 0.8 and not self._warned_supply:
                self._warned_supply = True
                log.warning(
                    "AnnotatedStreamer cam_%s 供幀 %.1f fps < 編碼 %d fps —— "
                    "同一張會被重複送出,畫面會抖。把該台的 decode_skip_frames 設成 0 "
                    "(或把 ANNOTATED_STREAM_FPS 調到與供幀相同)",
                    self.camera_id, self._supply_fps, self.fps)
                print(f"⚠️ [annot] cam_{self.camera_id} 供幀 {self._supply_fps:.1f} fps "
                      f"< 編碼 {self.fps} fps → 畫面會抖,請把 decode_skip_frames 設 0",
                      flush=True)

    def push(self, frame: np.ndarray):
        self.push_frame(frame)

    def close(self):
        self._stopped = True
        with self._proc_lock:
            for variant, proc in list(self.procs.items()):
                if not proc:
                    continue
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                self.procs[variant] = None


def get_streamer(camera_id: int, width: int = 0, height: int = 0, fps: int = 0) -> AnnotatedStreamer:
    """0 = 用環境變數的預設值(呼叫端不必知道尺寸)。"""
    with _streamers_lock:
        s = _streamers.get(camera_id)
        if s is None:
            s = AnnotatedStreamer(camera_id,
                                  width or STREAM_WIDTH,
                                  height or STREAM_HEIGHT,
                                  fps if fps else STREAM_FPS)
            _streamers[camera_id] = s
        return s


def update_detections(camera_id: int, detections: list, ts: float = 0.0):
    """Worker thread call after each inference。ts = 該組結果所屬畫面的時間戳。"""
    get_streamer(camera_id).update_detections(detections, ts)


def push_frame(camera_id: int, frame: np.ndarray, ts: float = 0.0):
    """Reader thread call per cap.read() — high cadence (~30 fps)。ts = 取得該幀的時間。"""
    if frame is None:
        return
    get_streamer(camera_id).push_frame(frame, ts)


def push_annotated(camera_id: int, frame: np.ndarray, detections: list):
    """Backward-compat: 一次更新 dets + 推 frame。"""
    if frame is None:
        return
    s = get_streamer(camera_id)
    s.update_detections(detections)
    s.push_frame(frame)
