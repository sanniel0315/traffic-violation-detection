#!/usr/bin/env python3
"""壅塞原始樣本保留政策 —— 安全連鎖必須成立：聚合沒覆蓋的日期絕對不能刪。

為什麼要有這條連鎖：報表讀的是 congestion_report_aggs，原始樣本刪掉沒關係 ——
「前提是那段時間真的有聚合」。2026-05-08~06-15 那 41 天，聚合因為首次全量
aggregation OOM 而從來沒產生過（實測 1h bucket 最早只到 06-16），
如果保留政策只看天數就硬刪，這 41 天的壅塞歷史會永久消失、且沒有任何告警。

所以規則是：超過保留天數 **且** 該日在 congestion_report_aggs 有 1h 聚合 → 才刪。
"""
import os
import sqlite3
import sys
import tempfile
import time

# 被測程式會印 emoji（⚠️），Windows 開發機 console 是 cp950 會直接 UnicodeEncodeError。
# 現場機是 UTF-8 沒這問題，但測試要能在兩邊都跑。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from cleanup_storage import cleanup_congestion_samples  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def day_ago(n: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() - n * 86400))


def build_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE congestion_samples (id INTEGER PRIMARY KEY, created_at TEXT);
        CREATE TABLE congestion_report_aggs (id INTEGER PRIMARY KEY, bucket_start TEXT, bucket_size TEXT);
        """
    )
    # 三個日期各 50 列：
    #   d_old_covered   90 天前、有聚合   → 該刪
    #   d_old_uncovered 80 天前、沒聚合   → 🛑 不可刪
    #   d_recent         5 天前、有聚合   → 在保留期內，不該刪
    for day in (DAY_OLD_COVERED, DAY_OLD_UNCOVERED, DAY_RECENT):
        conn.executemany(
            "INSERT INTO congestion_samples (created_at) VALUES (?)",
            [(f"{day} 08:{i:02d}:00",) for i in range(50)],
        )
    for day in (DAY_OLD_COVERED, DAY_RECENT):
        conn.execute(
            "INSERT INTO congestion_report_aggs (bucket_start, bucket_size) VALUES (?, '1h')",
            (f"{day} 08:00:00",),
        )
    # 干擾項：同一天只有 5m 聚合不算覆蓋（1h 才是報表長期查詢用的）
    conn.execute(
        "INSERT INTO congestion_report_aggs (bucket_start, bucket_size) VALUES (?, '5m')",
        (f"{DAY_OLD_UNCOVERED} 08:00:00",),
    )
    conn.commit()
    conn.close()


DAY_OLD_COVERED = day_ago(90)
DAY_OLD_UNCOVERED = day_ago(80)
DAY_RECENT = day_ago(5)


def count(path: str, day: str) -> int:
    conn = sqlite3.connect(path)
    n = conn.execute(
        "SELECT COUNT(*) FROM congestion_samples WHERE substr(created_at,1,10) = ?", (day,)
    ).fetchone()[0]
    conn.close()
    return n


print("壅塞樣本保留政策安全連鎖")

# --- A. dry-run 不能真的刪 ---
with tempfile.TemporaryDirectory() as tmp:
    db = os.path.join(tmp, "t.db")
    build_db(db)
    rows, skipped = cleanup_congestion_samples(db, keep_days=30, dry_run=True)
    check("dry-run 回報要刪的列數", rows, 50)
    check("dry-run 跳過未覆蓋的天數", skipped, 1)
    check("dry-run 沒有真的刪掉任何列", count(db, DAY_OLD_COVERED), 50)

# --- B. 實際刪除 ---
with tempfile.TemporaryDirectory() as tmp:
    db = os.path.join(tmp, "t.db")
    build_db(db)
    rows, skipped = cleanup_congestion_samples(db, keep_days=30, dry_run=False, batch=7)
    check("刪除列數", rows, 50)
    check("有聚合覆蓋的舊日期被清空", count(db, DAY_OLD_COVERED), 0)
    check("🛑 沒有聚合覆蓋的舊日期一列都不能少", count(db, DAY_OLD_UNCOVERED), 50)
    check("保留期內的日期不動", count(db, DAY_RECENT), 50)
    check("回報跳過 1 天", skipped, 1)

# --- C. 分批要能把整天刪完（batch 小於當日列數也不能只刪一批） ---
with tempfile.TemporaryDirectory() as tmp:
    db = os.path.join(tmp, "t.db")
    build_db(db)
    cleanup_congestion_samples(db, keep_days=30, dry_run=False, batch=3)
    check("batch=3 仍把 50 列全部刪完", count(db, DAY_OLD_COVERED), 0)

# --- D. 只有 5m 聚合不算覆蓋 ---
with tempfile.TemporaryDirectory() as tmp:
    db = os.path.join(tmp, "t.db")
    build_db(db)
    cleanup_congestion_samples(db, keep_days=30, dry_run=False)
    check("只有 5m 聚合的日期不算覆蓋,不刪", count(db, DAY_OLD_UNCOVERED), 50)

print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
