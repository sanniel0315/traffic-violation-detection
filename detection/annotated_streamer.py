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
STREAM_FPS = int(os.getenv("ANNOTATED_STREAM_FPS", "25") or 25)
STREAM_BITRATE = os.getenv("ANNOTATED_STREAM_BITRATE", "1M") or "1M"
STREAM_ENCODER = os.getenv("ANNOTATED_STREAM_ENCODER", "libx264") or "libx264"


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
        self.fps = max(1, int(fps))
        self.frame_interval = 1.0 / self.fps

        self.proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()

        # reader 只放「最新原始畫面的參照」,加工全部在 pacer 做 ——
        # 不可以在 reader thread 做 resize/畫框,那是偵測的取像路徑。
        self._latest_raw = None
        self._frame_lock = threading.Lock()

        self._latest_dets: list = []
        self._dets_lock = threading.Lock()

        self._stopped = False
        self._restart_count = 0
        self._max_restart = 200

        self._pacer = threading.Thread(target=self._pacer_loop, name=f"annot-pacer-{camera_id}", daemon=True)
        self._pacer.start()

    def _spawn(self):
        rtsp_url = f"rtsp://127.0.0.1:8554/cam_{self.camera_id}_annotated"
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
                self.proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
            log.info(f"AnnotatedStreamer cam_{self.camera_id} spawned pid={self.proc.pid}")
        except Exception as e:
            log.error(f"AnnotatedStreamer cam_{self.camera_id} spawn failed: {e}")
            self.proc = None

    def _ensure_proc(self) -> bool:
        with self._proc_lock:
            if self.proc is not None and self.proc.poll() is None:
                return True
            if self._restart_count >= self._max_restart:
                return False
            self._restart_count += 1
            self._spawn()
            return self.proc is not None

    def _pacer_loop(self):
        next_tick = time.monotonic()
        while not self._stopped:
            next_tick += self.frame_interval
            with self._frame_lock:
                raw = self._latest_raw
            buf = self._encode_frame(raw) if raw is not None else None
            if buf is not None and self._ensure_proc():
                try:
                    self.proc.stdin.write(buf)
                except (BrokenPipeError, IOError, ValueError, AttributeError):
                    with self._proc_lock:
                        self.proc = None
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def update_detections(self, detections: list):
        with self._dets_lock:
            self._latest_dets = list(detections or [])
        try:
            if detections:
                self._dbg_n = getattr(self, '_dbg_n', 0) + 1
                if self._dbg_n % 30 == 1:
                    d0 = detections[0]
                    print(f"[ann_dbg] cam={self.camera_id} n={len(detections)} keys={list(d0.keys())} cls={d0.get('class_name')!r} bbox={d0.get('bbox')}", flush=True)
        except Exception:
            pass

    def _encode_frame(self, item) -> Optional[bytes]:
        """畫框 + 轉 bytes。只在 pacer thread 呼叫。

        收到的畫面已經在 push_frame 裡脫離過 cap.read() 的緩衝(見那裡的說明)。
        """
        try:
            frame, sx, sy = item
            with self._dets_lock:
                dets = self._latest_dets
            annotated = _draw_overlay(frame, dets, sx, sy)
            if not annotated.flags['C_CONTIGUOUS']:
                annotated = np.ascontiguousarray(annotated)
            return annotated.tobytes()
        except Exception:
            return None

    def push_frame(self, frame: np.ndarray):
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
            self._latest_raw = (out, sx, sy)

    def push(self, frame: np.ndarray):
        self.push_frame(frame)

    def close(self):
        self._stopped = True
        with self._proc_lock:
            if self.proc:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=2)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                self.proc = None


def get_streamer(camera_id: int, width: int = 0, height: int = 0, fps: int = 0) -> AnnotatedStreamer:
    """0 = 用環境變數的預設值(呼叫端不必知道尺寸)。"""
    with _streamers_lock:
        s = _streamers.get(camera_id)
        if s is None:
            s = AnnotatedStreamer(camera_id,
                                  width or STREAM_WIDTH,
                                  height or STREAM_HEIGHT,
                                  fps or STREAM_FPS)
            _streamers[camera_id] = s
        return s


def update_detections(camera_id: int, detections: list):
    """Worker thread call after each inference."""
    get_streamer(camera_id).update_detections(detections)


def push_frame(camera_id: int, frame: np.ndarray):
    """Reader thread call per cap.read() — high cadence (~30 fps)."""
    if frame is None:
        return
    get_streamer(camera_id).push_frame(frame)


def push_annotated(camera_id: int, frame: np.ndarray, detections: list):
    """Backward-compat: 一次更新 dets + 推 frame。"""
    if frame is None:
        return
    s = get_streamer(camera_id)
    s.update_detections(detections)
    s.push_frame(frame)
