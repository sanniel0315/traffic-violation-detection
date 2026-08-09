#!/usr/bin/env python3
"""對外 API 的 in_flow / out_flow 不可把 INOUT 一般流量算進去。

背景:commit 5fede7c 讓「進出計數」與「一般流量計數」並存後,
一個 INOUT 的 ROI 會同時產生三種 direction:
    INOUT      → 一般流量(計入 total_flow)
    IN         → 進場轉場
    EXIT / OUT → 離場轉場
舊公式 in_flow = IN + INOUT、out_flow = OUT + EXIT + INOUT 會把一般流量
重複加進進出流量。87 實測 directionCounts={straight:67,IN:40,INOUT:40,EXIT:39}
舊公式得 in=80/out=79,灌水約 100%。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.routes.external import _vd_rows_to_records, _vd_stats  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def row(direction_counts, total=0, **kw):
    base = {
        "deviceId": "cam_2", "roadName": "台62", "timeKey": 1785916800000,
        "direction": "INOUT", "directionText": "進出",
        "totalFlow": total, "smallFlow": 0, "largeFlow": 0,
        "avgSpeed": 0, "avgOccupancyPct": 0,
        "directionCounts": direction_counts, "laneCount": 1, "lanes": {},
        # 有畫進出線的相機才會輸出 in_flow/out_flow(2026-08-09 起)。
        # 預設 True 讓既有案例維持原本要驗的重點:進出流量怎麼算。
        "inoutEnabled": kw.pop("inout_enabled", True),
    }
    base.update(kw)
    return base


print("[1] 87 實測那一筆(關鍵回歸)")
recs = _vd_rows_to_records(
    [row({"straight": 67, "IN": 40, "INOUT": 40, "EXIT": 39}, total=107)], "5m")
r = recs[0]
check("total_flow 不變", r["total_flow"], 107)
check("in_flow  = IN            (不含 INOUT)", r["in_flow"], 40)
check("out_flow = OUT + EXIT    (不含 INOUT)", r["out_flow"], 39)
check("舊公式的 80 不可再出現", r["in_flow"] != 80, True)
check("舊公式的 79 不可再出現", r["out_flow"] != 79, True)

print("\n[2] 只有一般流量、沒有轉場 → 進出皆 0")
r = _vd_rows_to_records([row({"straight": 100}, total=100)], "5m")[0]
check("total_flow", r["total_flow"], 100)
check("in_flow", r["in_flow"], 0)
check("out_flow", r["out_flow"], 0)

print("\n[3] OUT 與 EXIT 兩種寫法都要算進 out_flow")
r = _vd_rows_to_records([row({"IN": 10, "OUT": 4, "EXIT": 6}, total=10)], "5m")[0]
check("in_flow", r["in_flow"], 10)
check("out_flow = OUT 4 + EXIT 6", r["out_flow"], 10)

print("\n[4] 純 INOUT(整框進出但轉場被邊線擋掉) → 進出 0,總流量仍在")
r = _vd_rows_to_records([row({"INOUT": 55}, total=55)], "5m")[0]
check("total_flow", r["total_flow"], 55)
check("in_flow 不可被 INOUT 灌成 55", r["in_flow"], 0)
check("out_flow 不可被 INOUT 灌成 55", r["out_flow"], 0)

print("\n[5] 統計摘要沿用同一份數字")
recs = _vd_rows_to_records([
    row({"straight": 67, "IN": 40, "INOUT": 40, "EXIT": 39}, total=107),
    row({"IN": 10, "EXIT": 8, "INOUT": 12}, total=12),
], "5m")
st = _vd_stats(recs)
ov = st["overall"] if "overall" in st else st
check("摘要 total_flow = 107+12", ov["total_flow"], 119)
check("摘要 in_flow  = 40+10", ov["in_flow"], 50)
check("摘要 out_flow = 39+8", ov["out_flow"], 47)

print("\n[6] direction_counts 原封不動帶出去(呼叫端要自己拆時仍拿得到)")
r = _vd_rows_to_records([row({"straight": 1, "IN": 2, "INOUT": 3, "EXIT": 4}, total=4)], "5m")[0]
check("direction_counts", r["direction_counts"],
      {"straight": 1, "IN": 2, "INOUT": 3, "EXIT": 4})

print("\n" + (f"FAILED ({len(fails)}): {fails}" if fails else "ALL PASS"))
sys.exit(1 if fails else 0)
