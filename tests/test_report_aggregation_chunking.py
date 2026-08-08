#!/usr/bin/env python3
"""聚合分段(chunk)必須與「一次算完整個範圍」得到完全相同的結果。

背景：`refresh_congestion_aggregates` / `refresh_lpr_aggregates` 是
`query.all()` 把整個時間範圍載成 ORM 物件。平常背景 job 只跑 1 小時沒事，
但範圍一大就 OOM —— 2026-05-08~06-15 那 41 天壅塞聚合從來沒產生過，
就是首次全量聚合一次拉 1100 萬列直接掛掉（實測 congestion_report_aggs
1h bucket 最早只到 2026-06-16，比原始資料晚了 39 天）。

修法是在 `refresh_report_aggregates_for_range` 切成 1 小時一段，
聚合邏輯本身完全不動。這個測試就是證明「切段不改變結果」：
同一批樣本，一次算完 vs 分段算，每一列的每個欄位都必須一致。

🛑 段長必須是所有 bucket size 的整數倍，否則 bucket 會被切斷。
測試特地涵蓋跨 3 小時、且樣本落在 bucket 邊界上的情形。
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from api.models import Base, CongestionSample, CongestionReportAgg, TrafficEvent, TrafficReportAgg  # noqa: E402
from api.utils import report_aggregation as ra  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={_brief(got)} want={_brief(want)}")
    if not ok:
        fails.append(name)


def _brief(value, limit=90):
    """聚合表整張比對時值會有幾十萬字元,只印摘要,失敗時才需要細節。"""
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"<{type(value).__name__} len={len(value)}> {text[:limit]}..."


def check_rows(name, got_rows, want_rows):
    """逐列比對兩張聚合表,只印出第一筆不同的那列(不要整張倒出來)。"""
    if got_rows == want_rows:
        print(f"  PASS  {name}  {len(got_rows)} 列完全相同")
        return
    fails.append(name)
    print(f"  FAIL  {name}  分段 {len(got_rows)} 列 vs 一次算完 {len(want_rows)} 列")
    for idx, (a, b) in enumerate(zip(got_rows, want_rows)):
        if a != b:
            print(f"         第 {idx} 列起不同\n           分段 = {a}\n           一次 = {b}")
            break


BASE = datetime(2026, 5, 10, 8, 0, 0)          # 對齊整點
SPAN_HOURS = 3


def make_session():
    engine = create_engine("sqlite://")          # 記憶體 DB，每次乾淨
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def seed(db):
    """塞入跨 3 小時的樣本，刻意包含 bucket 邊界(整點/整分)與各種 None。"""
    rows_c, rows_t = [], []
    for i in range(SPAN_HOURS * 60):             # 每分鐘一筆，共 180 分鐘
        ts = BASE + timedelta(minutes=i)
        for cam, zone, lane, overall in ((1, "車流區 1", 1, False), (1, "", 0, True), (2, "車流區 1", 2, False)):
            rows_c.append(
                CongestionSample(
                    camera_id=cam,
                    camera_name=f"cam_{cam}",
                    zone_name=zone,
                    lane_no=lane or None,
                    direction="straight",
                    movement="through",
                    is_overall=overall,
                    vehicle_count=i % 7,
                    stopped_vehicle_count=i % 3,
                    occupancy=(i % 50) / 100.0,
                    raw_occupancy=(i % 50) / 100.0,
                    # 一半沒有排隊(0/None)，驗證「只算 >0」那條分母規則不被切段影響
                    estimated_queue_length_m=(float(i % 20) if i % 2 else 0.0),
                    queue_active=bool(i % 4),
                    queue_duration_sec=float(i % 11),
                    sample_interval_sec=2.0,
                    created_at=ts,
                )
            )
        rows_t.append(
            TrafficEvent(
                camera_id=1,
                lane_no=1,
                direction="straight",
                label="car" if i % 3 else "bus",
                speed_kmh=float(30 + (i % 25)),
                occupancy=0.4,
                created_at=ts,
            )
        )
    db.add_all(rows_c)
    db.add_all(rows_t)
    db.commit()


AGG_FIELDS_C = [
    "bucket_start", "bucket_size", "camera_id", "camera_name", "zone_name", "lane_no",
    "direction", "movement", "is_overall", "avg_occupancy", "max_occupancy",
    "avg_vehicle_count", "avg_stopped_vehicle_count", "avg_queue_length_m",
    "max_queue_length_m", "queue_active_duration_sec", "max_queue_duration_sec", "sample_count",
]
AGG_FIELDS_T = [
    "bucket_start", "bucket_size", "camera_id", "camera_name", "direction", "lane_no",
    "total_flow", "avg_speed", "max_speed", "avg_occupancy",
    "small_vehicle_flow", "large_vehicle_flow", "event_count",
]


def snapshot(db, model, fields):
    """把聚合表整張拉出來排序，updated_at 不比(每次執行必然不同)。"""
    out = []
    for row in db.query(model).all():
        out.append(tuple(getattr(row, f) for f in fields))
    return sorted(out, key=repr)


start, end = BASE, BASE + timedelta(hours=SPAN_HOURS)

# --- A. 一次算完整個範圍(修改前的行為) ---
db_a = make_session()
seed(db_a)
for size in ("1m", "5m", "1h"):
    ra.refresh_congestion_aggregates(db_a, start, end, size)
    ra.refresh_traffic_aggregates(db_a, start, end, size)
db_a.commit()
once_c = snapshot(db_a, CongestionReportAgg, AGG_FIELDS_C)
once_t = snapshot(db_a, TrafficReportAgg, AGG_FIELDS_T)

# --- B. 分段(修改後的行為，1 小時一段) ---
db_b = make_session()
seed(db_b)
ra.refresh_report_aggregates_for_range(db_b, start, end)
chunk_c = snapshot(db_b, CongestionReportAgg, AGG_FIELDS_C)
chunk_t = snapshot(db_b, TrafficReportAgg, AGG_FIELDS_T)

print("聚合分段等價性")
check("段長是每個 bucket 的整數倍", [ra._AGG_CHUNK_SEC % s for s in (60, 300, 3600)], [0, 0, 0])
check("壅塞聚合列數相同", len(chunk_c), len(once_c))
check("交通聚合列數相同", len(chunk_t), len(once_t))
check_rows("壅塞聚合每一欄都相同", chunk_c, once_c)
check_rows("交通聚合每一欄都相同", chunk_t, once_t)
check("確實有算出東西(非兩邊都空)", len(once_c) > 0 and len(once_t) > 0, True)

# --- C. 跨越多段時 1h bucket 沒有被切斷 ---
hour_rows = [r for r in chunk_c if r[1] == "1h"]
check("1h bucket 數 = 3 小時 x 3 個 zone/lane 組合", len(hour_rows), SPAN_HOURS * 3)
# 每個 1h bucket 都該蒐滿 60 筆樣本，被切斷的話會變成不足 60
check("每個 1h bucket 都收滿 60 筆樣本", sorted({r[AGG_FIELDS_C.index("sample_count")] for r in hour_rows}), [60])

# --- D. 起點不對齊整點也不能漏資料 ---
db_d = make_session()
seed(db_d)
ra.refresh_report_aggregates_for_range(db_d, start + timedelta(minutes=23), end)
misaligned = [r for r in snapshot(db_d, CongestionReportAgg, AGG_FIELDS_C) if r[1] == "1h"]
check("起點 08:23 仍從 08:00 整點算起,不漏第一個 bucket", len(misaligned), SPAN_HOURS * 3)

print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
