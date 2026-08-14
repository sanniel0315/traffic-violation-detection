#!/usr/bin/env python3
"""對外 VD 報表的 `direction` 必須是「行車方向」，而且逐桶穩定。

回歸案例（2026-08-09 實測，外部每 20 秒輪詢 /api/v1/external/vd-report/latest）：
cam_2「台62基隆段隧道口」連續三個 1 分鐘桶回報的 direction 分別是
`INOUT` → `EXIT` → `IN`，15 筆記錄裡有 5 筆的 direction 是進出場事件。
上層照 (detector, time_start) upsert，這一欄就變成無意義又會跳動的值。

兩個成因：
1. `_camera_meta` 取代表方向時，cam_2 有兩個 VD zone（車流區 1=straight、
   車流區 2=INOUT）各 1 票，排序 `(-票數, 名稱)` 平手時按字母排 →
   'INOUT' < 'straight' → 進出模式勝出。
2. `build_vd_report_rows` 從 `directionCounts` 挑最高票，而 IN/EXIT/INOUT
   都在裡面，於是每個桶依當下事件組成而變。

規則：真實行車方向（straight/left/right/N2S…）一律優先於進出場的假方向
（IN/OUT/EXIT/INOUT）；桶內只有進出場事件時，保留 camera meta 的方向。
🛑 進出的「數量」由 in_flow / out_flow 表達，不靠 direction 這個欄位。
   本測試不得改動任何流量數字。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.utils.report_aggregation import _TRANSITION_DIRECTIONS, normalize_direction  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


# ── 1. camera meta：真實方向要贏過平手的 INOUT ────────────────────────
# 🛑 直接呼叫真正的 _camera_meta（配記憶體 DB），不要在測試裡「複製一份規則」——
#    複製的話邏輯改了測試照樣綠，等於沒測到。
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from api.models import Base, Camera  # noqa: E402
from api.utils.report_aggregation import _camera_meta  # noqa: E402

_engine = create_engine("sqlite://")
Base.metadata.create_all(_engine)
_Session = sessionmaker(bind=_engine)
_seq = [0]


def meta_direction(zones):
    """建一台只有這些 zone 的相機，回傳 _camera_meta 算出來的代表方向。

    zones 可以是字串（只給 direction）或 dict（可含 travel_direction）。
    """
    db = _Session()
    try:
        _seq[0] += 1
        cam = Camera(
            name=f"t{_seq[0]}",
            zones=[
                ({"type": "flow_detection", "lane_no": 1, "direction": z}
                 if isinstance(z, str) else {"type": "flow_detection", "lane_no": 1, **z})
                for z in zones
            ],
        )
        db.add(cam)
        db.commit()
        by_id, _ = _camera_meta(db)
        return by_id[int(cam.id)]["direction"]
    finally:
        db.close()


print("camera meta 代表方向")
check("cam_2 的 straight 與 INOUT 平手時取 straight", meta_direction(["straight", "INOUT"]), "straight")
check("多個 straight 也是 straight", meta_direction(["straight"] * 4), "straight")
check("只有 INOUT 時才會是 INOUT", meta_direction(["INOUT"]), "INOUT")
check("票數多的真實方向勝出", meta_direction(["left", "straight", "straight"]), "straight")
check("真實方向即使票數少也贏過 INOUT", meta_direction(["INOUT", "INOUT", "INOUT", "left"]), "left")
check("沒有 zone 時 unknown", meta_direction([]), "unknown")

print("\n整支相機的 zone 全部綁 IN/OUT（使用者明確說一定會出現的情況）")
all_inout = [{"direction": "INOUT"}, {"direction": "INOUT"}]
check("沒設行進方向 → 只能回 INOUT（就是要修掉的狀況）", meta_direction(all_inout), "INOUT")
check("設了行進方向 → 回真實方向",
      meta_direction([{"direction": "INOUT", "travel_direction": "S2N"},
                      {"direction": "INOUT", "travel_direction": "S2N"}]), "S2N")
check("只有部分 zone 設行進方向,真實方向仍勝出",
      meta_direction([{"direction": "INOUT"},
                      {"direction": "INOUT", "travel_direction": "N2S"}]), "N2S")
check("行進方向被誤填成 IN/OUT 時忽略,退回 direction",
      meta_direction([{"direction": "INOUT", "travel_direction": "OUT"}]), "INOUT")
check("行進方向留空不影響原本的 straight",
      meta_direction([{"direction": "straight", "travel_direction": ""}]), "straight")
check("行進方向可覆寫轉向欄位",
      meta_direction([{"direction": "straight", "travel_direction": "N2S"}]), "N2S")

# ── 2. build_vd_report_rows：逐桶代表方向 ─────────────────────────────
def bucket_direction(meta_dir, direction_counts):
    """複製 build_vd_report_rows 的挑選規則。"""
    direction = meta_dir or "unknown"
    if direction_counts:
        ordered = sorted(
            direction_counts.items(),
            key=lambda item: (-int(item[1] or 0), item[0] == "unknown", str(item[0])),
        )
        best = next(
            (k for k, _ in ordered if k != "unknown" and k not in _TRANSITION_DIRECTIONS),
            None,
        )
        if best:
            direction = best
    return direction


print("\n逐桶代表方向（cam_2 實際觀測到的三個桶）")
# 實測資料：00:02 {'EXIT':2,'IN':3,'INOUT':3,'straight':1} → 舊版給 'IN'
check("桶內有 straight 就取 straight",
      bucket_direction("straight", {"EXIT": 2, "IN": 3, "INOUT": 3, "straight": 1}), "straight")
# 00:03 {'EXIT':1,'IN':1,'INOUT':1} → 舊版給 'EXIT'
check("桶內只有進出場事件 → 保留 meta 方向",
      bucket_direction("straight", {"EXIT": 1, "IN": 1, "INOUT": 1}), "straight")
# 00:04~00:06 {} → 舊版給 meta 的 'INOUT'
check("空桶 → 保留 meta 方向", bucket_direction("straight", {}), "straight")
check("三個桶的方向完全一致（不再逐桶跳）",
      len({bucket_direction("straight", c) for c in
           ({"EXIT": 2, "IN": 3, "INOUT": 3, "straight": 1}, {"EXIT": 1, "IN": 1, "INOUT": 1}, {})}), 1)
check("meta 也不知道方向時維持 unknown",
      bucket_direction("unknown", {"IN": 5, "EXIT": 4}), "unknown")
check("真實方向照樣正常運作", bucket_direction("straight", {"left": 9, "straight": 2}), "left")

# ── 3. 流量數字一律不受影響 ──────────────────────────────────────────
from api.routes.external import _vd_rows_to_records  # noqa: E402

row = {
    "deviceId": "台62基隆段隧道口", "roadName": "台62", "timeKey": 1786000000000,
    "direction": "straight", "directionText": "直行",
    "directionCounts": {"straight": 1, "IN": 3, "INOUT": 3, "EXIT": 2},
    "totalFlow": 4, "smallFlow": 4, "largeFlow": 0,
    "avgSpeed": None, "avgOccupancyPct": None,
    "avgQueueLengthM": None, "maxQueueLengthM": None,
    "queueDurationSec": None, "maxQueueDurationSec": None,
    "laneCount": 1, "lanes": {},
    "inoutEnabled": True,   # 有畫進出線才會輸出 in_flow/out_flow
}
rec = _vd_rows_to_records([row], "1m")[0]
print("\n流量數字不得被方向修正影響")
check("in_flow 只算 IN", rec["in_flow"], 3)
check("out_flow 算 OUT+EXIT", rec["out_flow"], 2)
check("total_flow 不含 IN/EXIT", rec["total_flow"], 4)
check("direction_counts 原封不動", rec["direction_counts"],
      {"straight": 1, "IN": 3, "INOUT": 3, "EXIT": 2})
check("direction 是行車方向", rec["direction"], "straight")

# 沒畫進出線的相機:同一批 directionCounts,但不該輸出 in/out 欄位
no_io = dict(row, inoutEnabled=False)
rec2 = _vd_rows_to_records([no_io], "1m")[0]
check("沒畫進出線 → 不輸出 in_flow", "in_flow" in rec2, False)
check("沒畫進出線 → 不輸出 out_flow", "out_flow" in rec2, False)
check("沒畫進出線 → total_flow 照常", rec2["total_flow"], 4)
check("沒畫進出線 → direction_counts 照常給", rec2["direction_counts"],
      {"straight": 1, "IN": 3, "INOUT": 3, "EXIT": 2})

# ── 5. 偵測器名單固定 + 狀態標示（2026-08-09 使用者要求） ─────────────
# 兩個實際問題：
#   a) realtime 有 4 台、vd-report 只有 3 台 —— 報表端的名單原本只在
#      「完全沒資料」時才補齊，於是停用的相機在有資料的時段就消失，
#      同一台在兩支端點看得到/看不到，呼叫端會以為系統壞了。
#   b) 停用相機的數值恆 0，與「真的沒有車」長得一模一樣 → 用 status 分辨。
print("\n偵測器名單固定 + 狀態標示")
row_disabled = dict(row, status="disabled", totalFlow=0, smallFlow=0, largeFlow=0,
                    directionCounts={})
rec_d = _vd_rows_to_records([row_disabled], "1m")[0]
check("停用相機標成 disabled", rec_d["status"], "disabled")
check("停用相機仍然出現在記錄裡（不是藏起來）", rec_d["detector_id"], "台62基隆段隧道口")
check("停用相機的流量是 0", rec_d["total_flow"], 0)

rec_a = _vd_rows_to_records([dict(row, status="active")], "1m")[0]
check("正常相機標成 active", rec_a["status"], "active")
check("沒帶 status 時預設 active",
      _vd_rows_to_records([{k: v for k, v in row.items() if k != "status"}], "1m")[0]["status"],
      "active")

# 名單來源：有畫車流區(vd_eligible)就要列出，與是否啟用無關
agg_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "api", "utils", "report_aggregation.py"), encoding="utf-8").read()
pick = agg_src.split("device_ids = {device_id", 1)[1].split("current = start", 1)[0]
check("名單不再只在『完全沒資料』時才補齊", "elif not device_ids:" in pick, False)
check("有畫車流區的一律列出", 'if meta.get("vd_eligible"):' in pick, True)

# ── 6. lane_count 與 lanes 長度必須一致 ──────────────────────────────
# 回歸案例（2026-08-09 使用者發現）：沒有車的時段回 lane_count: 4 但 lanes: []。
# 呼叫端照 lane_count 跑迴圈讀 lanes[i] 會直接爆掉。
# 治法：把該相機設定好的車道先建出來（值 0），不是等有事件才建。
print("\nlane_count 與 lanes 必須對得起來")
lane_meta = agg_src.split("def _camera_meta(", 1)[1].split("\ndef ", 1)[0]
check("meta 帶出車道編號清單而不只數量", '"lane_nos": sorted(lane_set)' in lane_meta, True)

rows_src = agg_src.split("def build_vd_report_rows(", 1)[1]
check("報表建 row 時先補齊車道", "prefill_lanes(buckets[key], meta)" in rows_src, True)
check("補齊用的是 meta 的車道編號", 'meta.get("lane_nos")' in rows_src, True)

ext_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "api", "routes", "external.py"), encoding="utf-8").read()
# 即時查詢的實作已抽成 _realtime_rows()(逐分鐘查詢重複呼叫同一段),
# 補齊車道那段在 helper 裡 —— 只切端點本體會假失敗。兩段合起來檢查。
rt_src = (ext_src.split("def _realtime_rows(", 1)[1].split("\n@router", 1)[0]
          + ext_src.split("def external_realtime(", 1)[1].split("\n@router", 1)[0])
check("即時端點也先補齊車道", 'for ln in (meta.get("lane_nos") or [])' in rt_src, True)

print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
