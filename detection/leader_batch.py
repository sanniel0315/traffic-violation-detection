"""車輛偵測的 leader-follower 合批。

背景:為什麼不是「中央批次執行緒」
────────────────────────────────────────────────────────────────────────
2026-08-18 先做過中央批次器(所有呼叫丟進佇列,一條執行緒統一合批),104 A/B
在負載相當下分析率反而掉約 35%:
    批次開啟  3.17 / 3.01   detection 佔通道 30%
    批次關閉  5.66 / 4.49   detection 佔通道 39%
原因是它把 N 條偵測執行緒收斂成 1 條,跟 LPR 搶 GPU_INFERENCE_LOCK 時
從「N 對 1」變成「1 對 1」,detection 分到的通道直接變少;而到達率低於
處理量時佇列又不會積,批次大小實測就是 1.0,沒有合批來補償。

leader-follower 怎麼避開這件事
────────────────────────────────────────────────────────────────────────
🛑 每個呼叫端「仍然各自去搶 GPU_INFERENCE_LOCK」—— 競爭者數量完全不變,
   對 LPR 的相對佔比不變,上面那個公平性損失就不存在。
差別只在拿到鎖之後:
    先拿到的那條(leader)把「此刻已登記、參數相同」的其他畫面一起做掉,
    後面那些(follower)輪到自己拿到鎖時發現結果已經有了,立刻放掉鎖。
於是鎖被實際佔用的總時間下降,通道吞吐上升,而排隊順序與競爭者數量不變。

離線量測(87)的空間:4 張分開跑 121.0ms、一次批次 68.1ms = 1.78x,
差距是 ultralytics 每次呼叫的固定開銷(predictor 設定 / Results 物件 / NMS),
那是 per-call 不是 per-image,所以合批才有意義。

環境變數
    DETECT_LEADER_BATCH=0   關閉,完全走原本的單張路徑
    DETECT_BATCH_MAX=8      單批最多幾張
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Callable

from detection.gpu_lock import GPU_INFERENCE_LOCK

LEADER_BATCH_ENABLED = os.getenv("DETECT_LEADER_BATCH", "1") != "0"
BATCH_MAX = max(1, int(os.getenv("DETECT_BATCH_MAX", "8") or 8))


class _Req:
    __slots__ = ("key", "model", "conf", "device", "frame", "parse",
                 "out", "err", "done", "by_leader")

    def __init__(self, key, model, conf, device, frame, parse):
        self.key = key
        self.model = model
        self.conf = conf
        self.device = device
        self.frame = frame
        self.parse = parse
        self.out: list | None = None
        self.err: BaseException | None = None
        self.done = False           # 只在持有 GPU_INFERENCE_LOCK 時讀寫
        self.by_leader = False      # 這筆是被別人的批次做掉的


class _LeaderBatcher:
    def __init__(self) -> None:
        self._pending: dict[int, _Req] = {}
        self._pend_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._buf: deque = deque()      # (ts, batch_size, sec, followers)

    def run(self, model, conf: float, device: Any, frame,
            parse: Callable[[Any, Any], list], model_key: Any = None) -> list:
        """model_key:決定「哪些請求可以合成一批」。

        🛑 一定要傳「權重相同就相同」的鍵(實務上是模型檔路徑),不能用
           id(model)。每台相機各自 new 一個 VehicleDetector,用 id 的話四台
           就是四個鍵,永遠湊不成批 —— 2026-08-18 在 87 實測 batch_size
           恆為 1.0 就是這個原因。權重相同時,由任何一台的 model 一次跑完
           整批,結果與各自跑完全相同。
        """
        if not LEADER_BATCH_ENABLED:
            return self._single(model, conf, device, frame, parse)

        req = _Req((model_key if model_key is not None else id(model),
                    float(conf), str(device)),
                   model, conf, device, frame, parse)
        rid = id(req)
        with self._pend_lock:
            self._pending[rid] = req
        try:
            # 🛑 照舊自己搶鎖。這一行就是與中央批次器的根本差別。
            with GPU_INFERENCE_LOCK:
                if not req.done:
                    self._lead(req)
        finally:
            with self._pend_lock:
                self._pending.pop(rid, None)

        if req.err is not None:
            raise req.err
        if not req.done:
            # 理論上不會走到(leader 一定會把自己做完),留著當保險絲
            return self._single(model, conf, device, frame, parse)
        return req.out if req.out is not None else []

    # ── 內部 ────────────────────────────────────────────────────────────
    def _lead(self, me: _Req) -> None:
        """在持有 GPU_INFERENCE_LOCK 的情況下,把自己與同組的待處理一起做掉。"""
        batch = [me]
        with self._pend_lock:
            for r in self._pending.values():
                if len(batch) >= BATCH_MAX:
                    break
                if r is me or r.done or r.key != me.key:
                    continue
                batch.append(r)
        t0 = time.perf_counter()
        try:
            if len(batch) == 1:
                results = me.model(me.frame, conf=me.conf, verbose=False,
                                   device=me.device)
            else:
                results = me.model([r.frame for r in batch], conf=me.conf,
                                   verbose=False, device=me.device)
            # 模型呼叫本身的時間要跟後處理分開記,否則看不出合批到底省了什麼
            model_sec = time.perf_counter() - t0
            for r, res in zip(batch, results):
                try:
                    r.out = r.parse(res, r.frame)
                except BaseException as exc:      # noqa: BLE001
                    r.err = exc
                r.by_leader = r is not me
                r.done = True
        except BaseException as exc:              # noqa: BLE001
            # 整批失敗:自己往上丟,其他人標記失敗讓他們各自重試單張
            for r in batch:
                if r is me:
                    r.err = exc
                    r.done = True
                # follower 保持 done=False → 它拿到鎖時會自己當 leader 重跑
            raise
        finally:
            self._observe(len(batch), time.perf_counter() - t0,
                          locals().get("model_sec", 0.0))

    def _single(self, model, conf, device, frame, parse) -> list:
        with GPU_INFERENCE_LOCK:
            results = model(frame, conf=conf, verbose=False, device=device)
            out: list = []
            for res in results:
                out = parse(res, frame)
            return out

    def _observe(self, size: int, sec: float, model_sec: float = 0.0) -> None:
        now = time.time()
        with self._stats_lock:
            self._buf.append((now, size, sec, model_sec))
            cut = now - 60.0
            while self._buf and self._buf[0][0] < cut:
                self._buf.popleft()

    def stats(self) -> dict:
        with self._stats_lock:
            items = list(self._buf)
        if len(items) < 2:
            return {"enabled": LEADER_BATCH_ENABLED, "samples": len(items)}
        span = items[-1][0] - items[0][0]
        if span <= 0:
            return {"enabled": LEADER_BATCH_ENABLED, "samples": len(items)}
        n = len(items)
        imgs = sum(i[1] for i in items)
        return {
            "enabled": LEADER_BATCH_ENABLED,
            "samples": n,
            "window_sec": round(span, 1),
            # 1.0 = 完全沒有合批(沒有競爭);越大代表越多推論被攤掉
            "batch_size_avg": round(imgs / n, 2),
            "batch_size_max": max(i[1] for i in items),
            "batches_per_sec": round(n / span, 2),
            "images_per_sec": round(imgs / span, 2),
            "ms_per_batch_avg": round(sum(i[2] for i in items) / n * 1000, 1),
            "ms_per_image_avg": round(sum(i[2] for i in items) / max(1, imgs) * 1000, 1),
            # 只算模型那一呼叫,不含後處理。合批有效的話這個數字會明顯下降 ——
            # 因為 ultralytics 的固定開銷被整批攤掉了。
            "model_ms_per_image": round(sum(i[3] for i in items) / max(1, imgs) * 1000, 1),
            "model_ms_per_batch": round(sum(i[3] for i in items) / n * 1000, 1),
        }


LEADER = _LeaderBatcher()


def leader_batch_stats() -> dict:
    return LEADER.stats()
