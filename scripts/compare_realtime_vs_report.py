#!/usr/bin/env python3
"""即時查詢 vs 報表:同一分鐘的數字必須一致(現場機執行)。

公平比較的前提:
  realtime  = 現在往回推 60 秒的滾動視窗
  vd-report = 對齊的分鐘桶
只有在「整分那一刻」呼叫 realtime,它的視窗才會剛好等於前一個分鐘桶。
所以先等到整分邊界再打,否則兩邊時間範圍不同,數字本來就不會一樣。

報表讀聚合表、即時讀原始表 —— 這個測試同時驗證兩條路徑算出同一個答案。
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

KEY = "tvd_hwacom_traffic_2026"
BASE = "http://127.0.0.1:8000/api/v1/external"


def api(path):
    out = subprocess.run(
        ["curl", "-s", "--max-time", "30", "-H", f"X-API-Key: {KEY}", f"{BASE}{path}"],
        capture_output=True, text=True, timeout=60).stdout
    return json.loads(out)["data"]


# ── 等到整分邊界 ─────────────────────────────────────────────────────
now = time.time()
wait = 60 - (now % 60)
print(f"  等 {wait:.0f} 秒到整分邊界…", flush=True)
time.sleep(wait + 0.3)

rt = api("/realtime?window_sec=60")
rt_start = datetime.fromisoformat(rt["period"]["start"])
print(f"  即時視窗 {rt['period']['start'][11:19]} ~ {rt['period']['end'][11:19]}")

# ── 等聚合跟上,再抓同一個桶 ──────────────────────────────────────────
print("  等聚合處理該分鐘…", flush=True)
target = rt_start.astimezone(timezone.utc).replace(tzinfo=None)
vd = None
for _ in range(12):
    time.sleep(10)
    got = api("/vd-report/latest?minutes=5&interval=1m")
    hit = [r for r in got["records"]
           if datetime.fromisoformat(r["time_start"]).astimezone(timezone.utc)
           .replace(tzinfo=None) == target]
    if hit:
        vd = hit
        break
if not vd:
    print("  ✗ 聚合遲遲沒處理到該分鐘,無法比對")
    sys.exit(1)
print(f"  報表桶   {vd[0]['time_start'][11:19]} ~ {vd[0]['time_end'][11:19]}")

# ── 逐台逐欄比對 ─────────────────────────────────────────────────────
FIELDS = ["total_flow", "small_vehicle_flow", "large_vehicle_flow", "in_flow", "out_flow"]
rt_by = {r["detector_id"]: r for r in rt["records"]}
vd_by = {r["detector_id"]: r for r in vd}

print()
print("  %-22s %-20s %-12s %-12s %s" % ("偵測器", "欄位", "即時", "報表", ""))
bad = 0
for det in sorted(set(rt_by) | set(vd_by)):
    a, b = rt_by.get(det, {}), vd_by.get(det, {})
    if not a or not b:
        print("  %-22s 只出現在 %s" % (det[:20], "即時" if a else "報表"))
        bad += 1
        continue
    for f in FIELDS:
        if f not in a and f not in b:
            continue
        va, vb = a.get(f, "(無)"), b.get(f, "(無)")
        ok = va == vb
        if not ok:
            bad += 1
        print("  %-22s %-20s %-12s %-12s %s"
              % (det[:20], f, va, vb, "" if ok else "← 不符"))
    print()

print(f"  不符 {bad} 項")
sys.exit(1 if bad else 0)
