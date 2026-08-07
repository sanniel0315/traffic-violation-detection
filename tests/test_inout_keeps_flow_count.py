#!/usr/bin/env python3
"""進出(IN/OUT)框必須「兩者並存」：進出流量歸進出，原本的流量計數還是要。

回歸案例：舊版把 direction=INOUT 的 zone 從一般流量迴圈排除
（stream.py 的 `hit_zones = [z for z in hit_zones if key not in _inout_keys]`），
使用者一旦在 ROI 邊界標了進出線，那個車道的流量計數就整個消失，
只剩轉場數 —— 而轉場數會漏掉「第一次被偵測時就已經在框內」的車，
設了 in_edge 之後更嚴格，總流量會塌掉。

現在：
  一般流量事件 direction='INOUT'  → 進 totalFlow（30 秒同車冷卻，語意同其他車流區）
  轉場事件     direction='IN'/'EXIT' → 只進 directionCounts，不進 totalFlow
兩條路徑各司其職，不重複計數。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.utils.report_aggregation import normalize_direction  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


class Agg:
    """模擬 traffic_report_aggs 的一列。"""

    def __init__(self, direction, total_flow, small=0, large=0, lane_no=1):
        self.direction = direction
        self.total_flow = total_flow
        self.small_vehicle_flow = small
        self.large_vehicle_flow = large
        self.lane_no = lane_no
        self.event_count = total_flow
        self.avg_speed = None
        self.avg_occupancy = None


def accumulate(aggs):
    """複製 build_vd_report_rows 內的計數分支（同一份判斷式）。"""
    total = 0
    counts = {}
    lane_flow = 0
    for a in aggs:
        d = normalize_direction(a.direction)
        counts[d] = counts.get(d, 0) + int(a.total_flow or 0)
        if d in ("IN", "EXIT"):
            continue
        total += int(a.total_flow or 0)
        if a.lane_no and int(a.lane_no) > 0:
            lane_flow += int(a.total_flow or 0)
    return total, counts, lane_flow


print("[1] 一般車流區(未設進出) — 不受影響")
t, c, lf = accumulate([Agg("straight", 100)])
check("總流量", t, 100)
check("車道流量", lf, 100)

print("\n[2] 進出框:一般流量 + 進出轉場並存")
aggs = [Agg("INOUT", 100), Agg("IN", 87), Agg("EXIT", 85)]
t, c, lf = accumulate(aggs)
check("總流量只算一般流量事件", t, 100)
check("車道流量同樣是 100", lf, 100)
check("IN 有獨立計數", c.get("IN"), 87)
check("EXIT 有獨立計數", c.get("EXIT"), 85)
check("INOUT 也在 directionCounts", c.get("INOUT"), 100)

print("\n[3] 不會重複計數(關鍵)")
check("總流量 != 100+87 (IN 不重複進總量)", t != 187, True)
check("總流量 != 100+87+85", t != 272, True)

print("\n[4] 轉場數塌掉時,總流量仍然正確")
# 設了 in_edge → 嚴格跨線,大多數車不計入 IN;總流量不該跟著塌
aggs = [Agg("INOUT", 100), Agg("IN", 3), Agg("EXIT", 0)]
t, c, lf = accumulate(aggs)
check("總流量仍是 100", t, 100)
check("IN 只有 3", c.get("IN"), 3)

print("\n[5] 舊行為對照:若總流量靠 IN,轉場塌掉就會低估")
old_total = sum(a.total_flow for a in aggs if normalize_direction(a.direction) == "IN")
check("舊算法會得到 3(低估 97%)", old_total, 3)
check("新算法得到 100", t, 100)

print("\n[6] 多車道彙總")
aggs = [Agg("INOUT", 40, lane_no=1), Agg("INOUT", 60, lane_no=2),
        Agg("IN", 35, lane_no=1), Agg("EXIT", 33, lane_no=1)]
t, c, lf = accumulate(aggs)
check("兩車道總流量 40+60", t, 100)
check("車道流量也是 100", lf, 100)

print("\n[7] normalize_direction 不可把 INOUT 併進 IN")
check("'INOUT'", normalize_direction("INOUT"), "INOUT")
check("'IN'", normalize_direction("IN"), "IN")
check("'EXIT'", normalize_direction("EXIT"), "EXIT")
check("'straight'", normalize_direction("straight"), "straight")

print("\n[8] stream.py 不可再把 INOUT 框排除在一般迴圈外")
import pathlib  # noqa: E402
src = pathlib.Path(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "api", "routes", "stream.py")).read_text(encoding="utf-8")
check("不存在 _inout_keys 過濾", "_inout_keys" in src, False)
check("仍有 _inout_zones(轉場計數還在)", "_inout_zones" in src, True)

print("\n" + (f"FAILED ({len(fails)}): {fails}" if fails else "ALL PASS"))
sys.exit(1 if fails else 0)
