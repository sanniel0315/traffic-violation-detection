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
}
rec = _vd_rows_to_records([row], "1m")[0]
print("\n流量數字不得被方向修正影響")
check("in_flow 只算 IN", rec["in_flow"], 3)
check("out_flow 算 OUT+EXIT", rec["out_flow"], 2)
check("total_flow 不含 IN/EXIT", rec["total_flow"], 4)
check("direction_counts 原封不動", rec["direction_counts"],
      {"straight": 1, "IN": 3, "INOUT": 3, "EXIT": 2})
check("direction 是行車方向", rec["direction"], "straight")

print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
