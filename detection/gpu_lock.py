"""Process-wide GPU inference lock.

Jetson Orin 上 ultralytics + TensorRT 多 thread 並發 inference 不同 detector
instance 仍會共用 GPU，CUDA stream race 觸發 `torch.cuda.synchronize` SEGV
(~每 1-2 分鐘一次)。所有 detector inference call 都序列化過這個 lock，
trade throughput for stability。

涵蓋：
  - VehicleDetector.detect (vehicle_detector.py)
  - TruckClassifier.classify (truck_classifier.py)
  - PlateDetector.detect (recognition/plate_detector.py)

用 `RLock` 而非 `Lock`：VehicleDetector.detect 在 lock 內會 call truck_classifier
做大型車細分類，同一 thread 嵌套 acquire 必須允許。

── 為什麼要量測(2026-08-18)──────────────────────────────────────────────
「分析率不可降低」是硬約束,但這把鎖是全 process 唯一的推論通道 —— 分析率究竟
是「機器跑不動」還是「排隊排不到」,只看 fps 分不出來,而這兩者的解法完全相反。
所以把鎖本身包一層,直接記兩個數:
    wait_ms  拿到鎖之前等了多久   → 大 = 通道塞住,要降低需求或提高吞吐
    hold_ms  拿到之後真正跑多久   → 大 = 模型/前後處理本身慢
utilization = 總 hold 時間 / 觀測時間,逼近 1 就代表這把鎖已經飽和,
再多給 CPU 也沒用。

包一層的成本是每次 acquire 多兩個 perf_counter 與一次 deque append(次微秒級),
相對於一次推論(數十毫秒)可以忽略。巢狀(truck classifier 在 vehicle detect 內)
只記最外層,不重複計。
"""
import re
import threading
import time
from collections import deque

_CAM_SUFFIX = re.compile(r"[-_]\d+$")


class _InstrumentedRLock:
    """RLock 的薄包裝,附滾動視窗統計。用法與 threading.RLock 完全相同。"""

    def __init__(self, window_sec: float = 60.0):
        self._lock = threading.RLock()
        self._local = threading.local()
        self._win = window_sec
        self._buf = deque()          # (release_ts, wait_sec, hold_sec)
        self._buf_lock = threading.Lock()

    # ── 與 threading.RLock 相容的介面 ──────────────────────────────────
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        t0 = time.perf_counter()
        ok = self._lock.acquire(blocking, timeout)
        if not ok:
            return ok
        depth = getattr(self._local, "depth", 0)
        if depth == 0:               # 只有最外層才計時,巢狀不重複計
            self._local.wait = time.perf_counter() - t0
            self._local.t_in = time.perf_counter()
        self._local.depth = depth + 1
        return ok

    def release(self) -> None:
        depth = getattr(self._local, "depth", 1) - 1
        self._local.depth = depth
        if depth == 0:
            hold = time.perf_counter() - getattr(self._local, "t_in", time.perf_counter())
            self._observe(getattr(self._local, "wait", 0.0), hold,
                          threading.current_thread().name)
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    # ── 統計 ──────────────────────────────────────────────────────────
    def _observe(self, wait: float, hold: float, who: str = "?") -> None:
        now = time.time()
        with self._buf_lock:
            self._buf.append((now, wait, hold, who))
            cut = now - self._win
            while self._buf and self._buf[0][0] < cut:
                self._buf.popleft()

    def stats(self) -> dict:
        with self._buf_lock:
            items = list(self._buf)
        if len(items) < 2:
            return {"samples": len(items)}
        span = items[-1][0] - items[0][0]
        if span <= 0:
            return {"samples": len(items)}
        n = len(items)
        holds = sorted(i[2] for i in items)
        waits = sorted(i[1] for i in items)
        # 依呼叫端 thread 名歸戶 —— 通道被誰吃掉,直接看得出來
        by_caller: dict = {}
        for _ts, w, h, who in items:
            # detection-7 / congestion-8 → detection / congestion,
            # 要看的是「哪一類工作」吃掉通道,不是哪一台
            key = _CAM_SUFFIX.sub("", str(who or "?"))
            b = by_caller.setdefault(key, {"calls": 0, "hold_sec": 0.0})
            b["calls"] += 1
            b["hold_sec"] += h
        callers = sorted(
            ({"caller": k,
              "calls_per_sec": round(v["calls"] / span, 2),
              "hold_ms_avg": round(v["hold_sec"] / v["calls"] * 1000, 1),
              "share": round(v["hold_sec"] / max(1e-9, sum(holds)), 3)}
             for k, v in by_caller.items()),
            key=lambda x: -x["share"])

        def p(sorted_vals, q):
            return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]

        return {
            "samples": n,
            "window_sec": round(span, 1),
            # 這把鎖每秒放行幾次推論 —— 全 process 的推論總吞吐
            "inferences_per_sec": round(n / span, 2),
            "wait_ms_avg": round(sum(waits) / n * 1000, 1),
            "wait_ms_p95": round(p(waits, 0.95) * 1000, 1),
            "hold_ms_avg": round(sum(holds) / n * 1000, 1),
            "hold_ms_p95": round(p(holds, 0.95) * 1000, 1),
            # 1.0 = 鎖被佔滿,再多 CPU 也提不了分析率
            "utilization": round(min(1.0, sum(holds) / span), 3),
            "by_caller": callers,
        }


GPU_INFERENCE_LOCK = _InstrumentedRLock()


def gpu_lock_stats() -> dict:
    """給監控端點用:目前 GPU 推論通道的吞吐與排隊狀況。"""
    return GPU_INFERENCE_LOCK.stats()
