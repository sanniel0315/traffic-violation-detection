#!/usr/bin/env python3
"""capture_open_guard / open_capture 的不變式測試。

背景:2026-08-18 08:33 在 87 發生 Fatal Python error: Segmentation fault。
四條串流在同一秒 30 秒逾時,四條 reader 同時走重連路徑,同時做
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ...   # setenv
    cv2.VideoCapture(...)                               # 內部 getenv
glibc 的 setenv 可能 realloc environ 陣列,跟併發的 getenv 就是 use-after-free。

這裡驗兩件事(都是當時缺的保證):
  ① 互斥   —— 任何時刻只有一條 thread 在「設參數 + 開啟」區段內
  ② 不被蓋 —— 區段內讀到的參數一定是自己要的那組,不會被別的模組蓋掉
"""
import os
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')

from api.utils.camera_stream import capture_open_guard, open_capture  # noqa: E402

ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
N_THREADS = 12
ROUNDS = 25

inside = [0]
max_inside = [0]
clobbered = []
counter_lock = threading.Lock()


def worker(idx: int) -> None:
    opts = f"marker;thread{idx}"
    for _ in range(ROUNDS):
        with capture_open_guard(opts):
            with counter_lock:
                inside[0] += 1
                max_inside[0] = max(max_inside[0], inside[0])
            # 在區段內停留一下,沒有鎖的話一定會重疊
            time.sleep(0.002)
            seen = os.environ.get(ENV)
            if seen != opts:
                clobbered.append((opts, seen))
            with counter_lock:
                inside[0] -= 1


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

fail = []
if max_inside[0] != 1:
    fail.append(f"同時進入區段的 thread 最多 {max_inside[0]} 條(應為 1)—— 沒有互斥")
if clobbered:
    fail.append(f"參數被蓋掉 {len(clobbered)} 次,例:要 {clobbered[0][0]} 讀到 {clobbered[0][1]}")

print(f"  {N_THREADS} 條 thread x {ROUNDS} 輪:區段內最大並行 {max_inside[0]},"
      f"參數被蓋 {len(clobbered)} 次")

# ── open_capture 也必須走同一個閘門 ──────────────────────────────────────
# 用假的 cv2 攔截:open_capture 內部是延遲 import cv2,塞一個假模組就能觀察。
seen_env = []


class _FakeCap:
    def __init__(self, *a):
        seen_env.append(os.environ.get(ENV))


fake_cv2 = types.ModuleType("cv2")
fake_cv2.VideoCapture = _FakeCap
sys.modules["cv2"] = fake_cv2

open_capture("rtsp://x/y", 1900, "marker;from-open-capture")
if seen_env != ["marker;from-open-capture"]:
    fail.append(f"open_capture 沒有把參數帶進去:{seen_env}")

# 沒給 options 就不該動環境變數(只上鎖)
before = os.environ.get(ENV)
seen_env.clear()
open_capture("/tmp/a.mp4")
if seen_env != [before]:
    fail.append(f"open_capture(options=None) 不該改動環境變數:{before} -> {seen_env}")

print("  open_capture 帶參數 / 不帶參數 行為正確" if not fail else "")
print("  結果:", "❌ " + " / ".join(fail) if fail else "✅ 全過")
sys.exit(1 if fail else 0)
