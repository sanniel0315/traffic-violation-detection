#!/usr/bin/env python3
"""VD 報表三個粒度(1m/5m/1h)全面驗證(現場機執行,唯讀)。

驗三件事:
  A. 每個粒度的聚合值 == 原始表算出來的值(逐相機、逐欄位)
  B. 跨粒度自洽:同一小時內 60 個 1m 桶 == 12 個 5m 桶 == 1 個 1h 桶
  C. 桶覆蓋完整:該時段每一個桶都存在,沒有缺漏

比對欄位:total_flow / small / large / in / out
"""
import sqlite3
import sys
from datetime import datetime, timedelta

DB = "data/violations.db"
LARGE = ("heavy_truck", "bus", "trailer", "tractor")
LARGE_SQL = " OR ".join("LOWER(COALESCE(label,'')) LIKE '%%%s%%'" % l for l in LARGE)
STEP = {"1m": 60, "5m": 300, "1h": 3600}

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.execute("PRAGMA busy_timeout = 30000")
names = dict(conn.execute("select id, name from cameras"))

# 取一個「已經完全結束」的整點小時當比對區間
now = datetime.utcnow()
hour_end = now.replace(minute=0, second=0, microsecond=0)
hour_start = hour_end - timedelta(hours=1)
S, E = hour_start.isoformat(sep=" "), hour_end.isoformat(sep=" ")
print(f"  比對區間(UTC) {S} ~ {E}")
print()

fails = []


def raw_by_bucket(step: int):
    """直接從原始表算出每個桶、每台相機的各項數字。"""
    rows = conn.execute(f"""
        SELECT datetime(strftime('%s', created_at) - (strftime('%s', created_at) % {step}),
                        'unixepoch') AS b,
               camera_id,
               UPPER(COALESCE(direction,'')) AS dir,
               COUNT(*) AS n,
               SUM(CASE WHEN ({LARGE_SQL}) THEN 1 ELSE 0 END) AS large_n
        FROM traffic_events INDEXED BY ix_traffic_events_created_at
        WHERE created_at >= ? AND created_at < ? AND camera_id IS NOT NULL
        GROUP BY b, camera_id, dir
    """, (S, E)).fetchall()
    out = {}
    for b, cam, d, n, large in rows:
        k = (b, cam)
        rec = out.setdefault(k, {"total": 0, "small": 0, "large": 0, "in": 0, "out": 0})
        if d == "IN":
            rec["in"] += n
        elif d in ("OUT", "EXIT"):
            rec["out"] += n
        else:
            rec["total"] += n
            rec["large"] += large
            rec["small"] += n - large
    return out


def agg_by_bucket(size: str):
    rows = conn.execute("""
        SELECT bucket_start, camera_id, UPPER(COALESCE(direction,'')),
               SUM(total_flow), SUM(small_vehicle_flow), SUM(large_vehicle_flow)
        FROM traffic_report_aggs
        WHERE bucket_size = ? AND bucket_start >= ? AND bucket_start < ?
        GROUP BY bucket_start, camera_id, 3
    """, (size, S, E)).fetchall()
    out = {}
    for b, cam, d, tot, small, large in rows:
        k = (str(b)[:19], cam)
        rec = out.setdefault(k, {"total": 0, "small": 0, "large": 0, "in": 0, "out": 0})
        if d == "IN":
            rec["in"] += tot or 0
        elif d in ("OUT", "EXIT"):
            rec["out"] += tot or 0
        else:
            rec["total"] += tot or 0
            rec["small"] += small or 0
            rec["large"] += large or 0
    return out


# ── A + C:每個粒度對原始表 ───────────────────────────────────────────
totals = {}
for size, step in STEP.items():
    raw, agg = raw_by_bucket(step), agg_by_bucket(size)
    expect_buckets = 3600 // step
    got_buckets = len({b for b, _ in agg})
    miss_keys = sorted(set(raw) - set(agg))
    extra_keys = sorted(set(agg) - set(raw))
    bad = []
    for k in sorted(set(raw) & set(agg)):
        for f in ("total", "small", "large", "in", "out"):
            if raw[k][f] != agg[k][f]:
                bad.append((k, f, raw[k][f], agg[k][f]))
    print(f"  === {size} ===")
    print(f"    桶數 {got_buckets}/{expect_buckets}   有資料的(桶,相機)組合 {len(raw)}")
    print(f"    原始有但聚合缺 {len(miss_keys)}   聚合有但原始無 {len(extra_keys)}   數值不符 {len(bad)}")
    for k, f, a, b in bad[:5]:
        print(f"      {k[0]} {names.get(k[1], k[1])} {f}: 原始 {a} 聚合 {b}")
    for k in miss_keys[:3]:
        print(f"      缺桶: {k[0]} {names.get(k[1], k[1])} 原始={raw[k]}")
    if miss_keys or bad:
        fails.append(size)
    # 累計該粒度的總量,供跨粒度比對
    t = {"total": 0, "small": 0, "large": 0, "in": 0, "out": 0}
    for rec in agg.values():
        for f in t:
            t[f] += rec[f]
    totals[size] = t
    print()

# ── B:跨粒度自洽 ─────────────────────────────────────────────────────
print("  === 跨粒度自洽(同一小時,三種粒度的總量應相同) ===")
print("    %-8s %-9s %-9s %-9s %-7s %-7s" % ("粒度", "total", "small", "large", "in", "out"))
for size in ("1m", "5m", "1h"):
    t = totals[size]
    print("    %-8s %-9s %-9s %-9s %-7s %-7s"
          % (size, t["total"], t["small"], t["large"], t["in"], t["out"]))
ref = totals["1h"]
for size in ("1m", "5m"):
    diff = {f: totals[size][f] - ref[f] for f in ref if totals[size][f] != ref[f]}
    if diff:
        print(f"    ✗ {size} 與 1h 不一致: {diff}")
        fails.append(f"{size}-vs-1h")
if not fails:
    print("    ✓ 三種粒度總量完全相同")

print()
print(f"  有問題的項目: {fails or '無'}")
sys.exit(1 if fails else 0)
