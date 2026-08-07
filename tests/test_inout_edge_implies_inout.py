#!/usr/bin/env python3
"""標了進出線的車流區,方向一律視為 INOUT(防呆)。

回歸案例:使用者在 ROI 編輯器標好進線(邊1)/出線(邊3)後,又到主頁把
「行車方向」選成「進場(IN)」。結果:
  1. direction='IN' → _normalize_event_direction 回 'IN' 不是 'INOUT'
     → 該 zone 沒被放進 _inout_zones → 轉場邏輯整個不跑,標的邊白標
     (實測 lane2 只有 IN 事件、EXIT 永遠 0)
  2. 該框的一般流量事件 direction 也是 'IN',報表端
     `if direction in ("IN","EXIT"): continue` 會排除它
     → lane2 的 29 筆完全沒進 totalFlow(實測 totalFlow=67 只剩 lane1)

修法:偵測迴圈啟動時,只要 zone 有 in_edge 或 out_edge,就把 direction
正規化成 INOUT。前端也移除了車流區的單獨 IN/OUT 選項。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

import pathlib  # noqa: E402
import re  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


# ── 取出 stream.py 裡真正的正規化函式與防呆邏輯 ──────────────────
SRC = pathlib.Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "api", "routes", "stream.py")).read_text(encoding="utf-8")

m = re.search(r"def _normalize_event_direction\(raw\):(.*?)(?=\n    def |\n\n    #|\nclass )", SRC, re.S)
assert m, "抽不到 _normalize_event_direction"
fn_src = "def _normalize_event_direction(raw):" + m.group(1)
fn_src = "\n".join(line[4:] if line.startswith("    ") else line
                   for line in fn_src.split("\n"))
ns = {}
exec(fn_src, ns)
norm = ns["_normalize_event_direction"]


def coerce(zones):
    """複製 stream.py 內的防呆邏輯(同一份判斷式)。"""
    for z in zones:
        if z.get("in_edge") in (None, "") and z.get("out_edge") in (None, ""):
            continue
        if norm(z.get("direction")) != "INOUT":
            z["direction"] = "INOUT"
    return [z for z in zones if norm(z.get("direction")) == "INOUT"]


print("[1] 正規化函式本身")
check("'IN'    不等於 INOUT", norm("IN"), "IN")
check("'OUT'   不等於 INOUT", norm("OUT"), "OUT")
check("'INOUT' 是 INOUT", norm("INOUT"), "INOUT")
check("'straight'", norm("straight"), "straight")

print("\n[2] 回歸:標了邊但方向是 IN → 必須被當成進出框")
z = {"name": "車流區 2", "lane_no": 2, "direction": "IN", "in_edge": 0, "out_edge": 2}
inout = coerce([z])
check("direction 被正規化成 INOUT", z["direction"], "INOUT")
check("有被放進 _inout_zones", len(inout), 1)

print("\n[3] 只標進線(出線留空)也算")
z = {"name": "A", "direction": "IN", "in_edge": 1, "out_edge": None}
inout = coerce([z])
check("direction", z["direction"], "INOUT")
check("在 _inout_zones", len(inout), 1)

print("\n[4] 只標出線也算")
z = {"name": "B", "direction": "OUT", "in_edge": "", "out_edge": 3}
inout = coerce([z])
check("direction", z["direction"], "INOUT")
check("在 _inout_zones", len(inout), 1)

print("\n[5] 沒標邊 → 不可被亂改方向")
for d in ("straight", "left", "right", ""):
    z = {"name": "C", "direction": d, "in_edge": None, "out_edge": ""}
    coerce([z])
    check(f"direction={d!r} 維持原樣", z["direction"], d)

print("\n[6] 沒標邊但方向本來就是 INOUT → 仍是進出框(整框進出)")
z = {"name": "D", "direction": "INOUT", "in_edge": None, "out_edge": None}
inout = coerce([z])
check("在 _inout_zones", len(inout), 1)
check("direction 不變", z["direction"], "INOUT")

print("\n[7] 混合:一個標邊、一個沒標")
zs = [
    {"name": "lane1", "direction": "straight", "in_edge": None, "out_edge": None},
    {"name": "lane2", "direction": "IN", "in_edge": 0, "out_edge": 2},
]
inout = coerce(zs)
check("只有 lane2 是進出框", [z["name"] for z in inout], ["lane2"])
check("lane1 方向不受影響", zs[0]["direction"], "straight")
check("lane2 方向被修正", zs[1]["direction"], "INOUT")

print("\n[8] 前端不可再提供單獨的 IN / OUT 給車流區")
IDX = pathlib.Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "web", "index.html")).read_text(encoding="utf-8")
blk = re.search(r"const flowDirectionOptions=\[(.*?)\];", IDX, re.S)
assert blk, "找不到 flowDirectionOptions"
opts = re.findall(r"value:'([^']*)'", blk.group(1))
check("選項清單", opts, ["", "INOUT", "left", "straight", "right"])
check("沒有單獨的 IN", "IN" in opts, False)
check("沒有單獨的 OUT", "OUT" in opts, False)

print("\n" + (f"FAILED ({len(fails)}): {fails}" if fails else "ALL PASS"))
sys.exit(1 if fails else 0)
