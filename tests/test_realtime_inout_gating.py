#!/usr/bin/env python3
"""即時查詢端點：滾動視窗、以及「沒畫進出線就不給 in/out」。

兩個需求（使用者 2026-08-09）：
1. **每 20 秒查要有不同數據** —— 報表端點最小是 1 分鐘桶，20 秒輪詢會連拿三次
   同一份。即時端點改回「現在往回推 N 秒」的滾動視窗，每次呼叫視窗都不同。
2. **有畫 in/out 才給** —— 沒設進出線的相機回 `in_flow: 0` 的話，
   「沒有車進出」和「這支根本不算進出」長得一模一樣，呼叫端分不出來。
   所以沒畫的相機不輸出這兩個欄位。

🛑 查詢一定要 `INDEXED BY` 釘住 created_at 索引。SQLite planner 在有低選擇性
   條件（`camera_id IS NOT NULL`、`is_overall=1`）時會挑錯索引，現場實測：
     traffic_events     滾動 60 秒   364 ms → 0.3 ms
     congestion_samples 滾動 60 秒  5663 ms → 0.5 ms
   差一萬倍；沒釘住的話 20 秒輪詢會把系統拖垮。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.utils.report_aggregation import _TRANSITION_DIRECTIONS, is_vd_zone, normalize_direction  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


# ── 1. inout_enabled 判定（_camera_meta 用的規則） ───────────────────
def inout_enabled(zones):
    return any(
        is_vd_zone(z) and normalize_direction(z.get("direction")) in _TRANSITION_DIRECTIONS
        for z in zones
    )


FLOW = "flow_detection"
print("哪些相機該給 in/out")
check("有一個 INOUT zone → 給",
      inout_enabled([{"type": FLOW, "direction": "INOUT"}]), True)
check("只有 straight zone → 不給",
      inout_enabled([{"type": FLOW, "direction": "straight"}]), False)
check("混合(一個 straight 一個 INOUT) → 給",
      inout_enabled([{"type": FLOW, "direction": "straight"}, {"type": FLOW, "direction": "INOUT"}]), True)
check("完全沒有 zone → 不給", inout_enabled([]), False)
check("非 VD zone 的 INOUT 不算",
      inout_enabled([{"type": "parking", "direction": "INOUT"}]), False)
check("方向留空 → 不給",
      inout_enabled([{"type": FLOW, "direction": ""}]), False)

# ── 2. 端點的 SQL 必須釘住索引 ───────────────────────────────────────
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "api", "routes", "external.py"), encoding="utf-8").read()
rt = src.split("def external_realtime(", 1)[1].split("\n@router", 1)[0]
print("\n即時端點的查詢必須釘住 created_at 索引")
check("traffic_events 有 INDEXED BY",
      "traffic_events INDEXED BY ix_traffic_events_created_at" in rt, True)
check("congestion_samples 有 INDEXED BY",
      "congestion_samples INDEXED BY ix_congestion_samples_created_at" in rt, True)
check("視窗有上限(避免有人傳超大值)", "le=600" in rt, True)
check("視窗有下限", "ge=10" in rt, True)

# ── 3. 進出流量的歸屬規則不可與 total 混算 ───────────────────────────
print("\n進出事件不可計入一般車流")
check("IN/OUT/EXIT 事件不進 totalFlow",
      'if direction in ("IN", "EXIT", "OUT"):' in rt, True)

# ── 4. 即時與報表必須是同一種記錄格式（不可分岔成兩套） ──────────────
# 客戶同時要打即時與報表，欄位名稱不一致等於逼他寫兩套解析。
print("\n即時與報表共用同一個輸出函式")
check("realtime 用 _vd_rows_to_records 產生記錄", "_vd_rows_to_records(out_rows" in rt, True)
check("realtime 用 _vd_stats 產生摘要", "_vd_stats(records)" in rt, True)
check("realtime 帶出 inoutEnabled 交給共用函式判斷", '"inoutEnabled": bool(' in rt, True)
check("對外時間不帶微秒", "replace(microsecond=0)" in rt, True)

conv = src.split("def _vd_rows_to_records(", 1)[1].split("\ndef ", 1)[0]
check("『有畫才給』的判斷收斂在共用函式裡", 'if row.get("inoutEnabled"):' in conv, True)
check("共用函式支援自訂視窗長度（即時用）", "span or _BUCKET_INTERVALS" in conv, True)
check("共用函式不再無條件輸出 in_flow", conv.count('"in_flow"'), 1)

# ── 5. mode=minute：分鐘內累積（使用者 2026-08-09 要求） ──────────────
# 需求：每 20 秒查一次，要拿到「這一分鐘從整分到現在的累積」，跨分歸零。
#   18:49:14 → 起點 18:49:00 經過 14 秒
#   18:49:54 → 起點 18:49:00 經過 54 秒（同一起點，只增不減）
#   18:50:15 → 起點 18:50:00 經過 15 秒（跨分歸零）
# 🛑 與滑動視窗(mode=window)是不同語意，不可混用：
#    滑動視窗的起點會跟著移動，累積模式的起點釘在整分。
print("\nmode=minute 分鐘內累積")
check("有 mode 參數且限定兩種值", 'pattern="^(window|minute)$"' in rt, True)
check("minute 模式起點釘在整分", 'start = now.replace(second=0)' in rt, True)
check("window 模式維持往回推", "start = now - span" in rt, True)
check("回應標明經過秒數", '"elapsed_sec": elapsed' in rt, True)
check("每小時車流率用實際經過秒數換算，不是 window_sec",
      "rec[\"total_flow\"] * 3600.0 / elapsed" in rt, True)
check("經過 0 秒時不做除法（整分那一刻）", "if elapsed > 0 else None" in rt, True)
check("window_sec 只在 window 模式回報",
      'int(window_sec) if mode == "window" else None' in rt, True)

print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
