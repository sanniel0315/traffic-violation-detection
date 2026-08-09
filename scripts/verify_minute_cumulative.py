#!/usr/bin/env python3
"""mode=minute 的累積,在分鐘結束那一刻必須等於 /vd-report 該分鐘桶的值。

兩條路徑不同:累積讀原始表、報表讀聚合表。同一分鐘算出不同答案就是 bug。
做法:在整分的前一瞬間抓累積(此時已涵蓋整分鐘),等聚合跟上後抓同一個桶比對。
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

KEY = "tvd_hwacom_traffic_2026"
BASE = "http://127.0.0.1:8000/api/v1/external"


def api(path):
    out = subprocess.run(["curl", "-s", "--max-time", "30", "-H", f"X-API-Key: {KEY}",
                          f"{BASE}{path}"], capture_output=True, text=True, timeout=60).stdout
    return json.loads(out)["data"]


# 等到「整分前 0.5 秒」—— 此時 mode=minute 幾乎涵蓋完整一分鐘
now = time.time()
wait = 60 - (now % 60) - 0.5
if wait < 0:
    wait += 60
print(f"  等 {wait:.1f} 秒到整分前一瞬…", flush=True)
time.sleep(wait)

cum = api("/realtime?mode=minute")
target = datetime.fromisoformat(cum["period"]["start"]).astimezone(timezone.utc).replace(tzinfo=None)
print(f"  累積視窗 {cum['period']['start'][11:19]} ~ {cum['period']['end'][11:19]}"
      f"  經過 {cum['elapsed_sec']} 秒")
if cum["elapsed_sec"] < 58:
    print(f"  ⚠️ 只涵蓋 {cum['elapsed_sec']} 秒,未涵蓋完整分鐘,比對會有落差")

print("  等聚合處理該分鐘…", flush=True)
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
    print("  ✗ 聚合沒處理到該分鐘")
    sys.exit(1)
print(f"  報表桶   {vd[0]['time_start'][11:19]} ~ {vd[0]['time_end'][11:19]}")

FIELDS = ["total_flow", "small_vehicle_flow", "large_vehicle_flow", "in_flow", "out_flow"]
a_by = {r["detector_id"]: r for r in cum["records"]}
b_by = {r["detector_id"]: r for r in vd}
print()
print("  %-22s %-20s %-10s %-10s" % ("偵測器", "欄位", "分鐘累積", "報表桶"))
bad = 0
for det in sorted(set(a_by) | set(b_by)):
    a, b = a_by.get(det, {}), b_by.get(det, {})
    for f in FIELDS:
        if f not in a and f not in b:
            continue
        va, vb = a.get(f, "(無)"), b.get(f, "(無)")
        ok = va == vb
        if not ok:
            bad += 1
        print("  %-22s %-20s %-10s %-10s %s" % (det[:20], f, va, vb, "" if ok else "← 不符"))
    print()
print(f"  不符 {bad} 項")
sys.exit(1 if bad else 0)
