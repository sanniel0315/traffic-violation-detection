"""把偵測過的 frame（含 bbox）即時編碼推 RTSP 給 go2rtc，前端走 WebRTC 拉。
v3：分離畫面源 (30 fps) 跟 detection (8 fps)。
- reader thread 每幀都 push (push_frame) → 畫面順
- worker 只更新最新 dets (update_detections) → bbox 不卡 detection
- pacer 用 wall-clock 餵 ffmpeg → 跟 detection/reader 都解耦"""

import logging
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_streamers: dict = {}
_streamers_lock = threading.Lock()

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

        self._latest_frame: Optional[bytes] = None
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
            "-c:v", "libx264",
            "-preset", "ultrafast", "-tune", "zerolatency",
            "-b:v", "2M", "-maxrate", "2M", "-bufsize", "2M",
            "-g", str(self.fps * 2),
            "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp",
            rtsp_url,
        ]
        try:
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
                buf = self._latest_frame
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

    def push_frame(self, frame: np.ndarray):
        if self._stopped or frame is None:
            return
        h, w = frame.shape[:2]
        sx = self.width / w if w else 1.0
        sy = self.height / h if h else 1.0
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        with self._dets_lock:
            dets = self._latest_dets
        annotated = _draw_overlay(frame, dets, sx, sy)
        if not annotated.flags['C_CONTIGUOUS']:
            annotated = np.ascontiguousarray(annotated)
        buf = annotated.tobytes()
        with self._frame_lock:
            self._latest_frame = buf

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


def get_streamer(camera_id: int, width: int = 1280, height: int = 720, fps: int = 25) -> AnnotatedStreamer:
    with _streamers_lock:
        s = _streamers.get(camera_id)
        if s is None:
            s = AnnotatedStreamer(camera_id, width, height, fps)
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
