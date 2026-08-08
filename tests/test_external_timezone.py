#!/usr/bin/env python3
"""對外報表端點:帶時區與不帶時區指到同一時刻，就必須查到同一批資料。

回歸案例（2026-08-09 實測）：`bucket_start` 在 DB 是 naive UTC，
`congestion-report` 卻把使用者傳進來的 tz-aware datetime 直接綁進 SQL。
SQLite 方言只會把它格式化成字串、把 tzinfo 丟掉 →
台北的牆上時間被當成 UTC，整個視窗往後位移 8 小時。

    start_time=2026-08-08T16:00:00        (naive，視為 UTC) → 7 筆
    start_time=2026-08-09T00:00:00+08:00  (同一時刻)        → 0 筆  ✗

而端點自己的說明就是叫人「查台北時間請帶 +08:00」，等於照文件用一定查不到。
vd-report 沒踩到是因為 build_vd_report_rows 內部有做 to_utc_naive。
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from api.models import Base, CongestionReportAgg  # noqa: E402
from api.utils.report_aggregation import to_utc_naive  # noqa: E402

fails = []
TZ_TAIPEI = timezone(timedelta(hours=8))


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


# ── to_utc_naive 本身 ────────────────────────────────────────────────
print("to_utc_naive")
check("+08:00 轉成 naive UTC",
      to_utc_naive(datetime(2026, 8, 9, 0, 0, tzinfo=TZ_TAIPEI)),
      datetime(2026, 8, 8, 16, 0))
check("naive 原樣不動（視為 UTC）",
      to_utc_naive(datetime(2026, 8, 8, 16, 0)), datetime(2026, 8, 8, 16, 0))
check("UTC 時區去掉 tzinfo",
      to_utc_naive(datetime(2026, 8, 8, 16, 0, tzinfo=timezone.utc)), datetime(2026, 8, 8, 16, 0))
check("None 回 None", to_utc_naive(None), None)

# ── 真的用 SQLAlchemy 查一次：兩種寫法必須拿到同一批 ──────────────────
# 🛑 用真的 query，不是自己算 —— 這個 bug 的成因就在 SQLAlchemy/SQLite
#    綁定 tz-aware datetime 的行為，只驗轉換函式抓不到它。
engine = create_engine("sqlite://")
Base.metadata.create_all(engine)
db = sessionmaker(bind=engine)()
for hour in (15, 16, 17):
    db.add(CongestionReportAgg(
        bucket_start=datetime(2026, 8, 8, hour, 0), bucket_size="1h",
        camera_id=2, camera_name="cam_2", zone_name="", lane_no=None,
        direction="INOUT", movement="", is_overall=True, sample_count=10,
    ))
db.commit()


def query_count(start, end):
    return db.query(CongestionReportAgg).filter(
        CongestionReportAgg.bucket_size == "1h",
        CongestionReportAgg.bucket_start >= to_utc_naive(start),
        CongestionReportAgg.bucket_start < to_utc_naive(end),
    ).count()


naive = query_count(datetime(2026, 8, 8, 16, 0), datetime(2026, 8, 8, 17, 0))
aware = query_count(datetime(2026, 8, 9, 0, 0, tzinfo=TZ_TAIPEI),
                    datetime(2026, 8, 9, 1, 0, tzinfo=TZ_TAIPEI))
utc_aware = query_count(datetime(2026, 8, 8, 16, 0, tzinfo=timezone.utc),
                        datetime(2026, 8, 8, 17, 0, tzinfo=timezone.utc))

print("\n同一時刻的三種寫法（DB 內 16:00 UTC 那個桶只有 1 列）")
check("naive（視為 UTC）", naive, 1)
check("+08:00（台北）", aware, 1)
check("+00:00（UTC）", utc_aware, 1)
check("三種寫法結果一致", len({naive, aware, utc_aware}), 1)

# 沒轉換的話會查到什麼 —— 證明這個 bug 真的會讓資料消失
broken = db.query(CongestionReportAgg).filter(
    CongestionReportAgg.bucket_size == "1h",
    CongestionReportAgg.bucket_start >= datetime(2026, 8, 9, 0, 0, tzinfo=TZ_TAIPEI),
    CongestionReportAgg.bucket_start < datetime(2026, 8, 9, 1, 0, tzinfo=TZ_TAIPEI),
).count()
check("不轉換就會查不到（bug 重現）", broken, 0)

db.close()
print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
