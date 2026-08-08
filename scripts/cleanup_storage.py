#!/usr/bin/env python3
"""儲存空間清理 job — 刪除超過保留天數的違規快照 / LPR 快照 / 事件快照 cache。

背景：
- output/violations（含 snapshots/ 子目錄）與 storage/lpr_snapshots 無 retention →
  NVMe 塞到 100% 導致 SQLite 偶發 database or disk is full。
- /tmp/event_snapshots（事件快照 cache，在 eMMC 上，UI 縮圖用）每天新增 ~1.3 萬檔、
  從不清 → eMMC 一路爬滿（實測 6G / 27 萬檔）。此 cache 短保留 3 天即可。
- congestion_samples（DB 資料列）每天 ~50 萬列、從不清 → 實測 4470 萬列 / DB 19 GB，
  是唯一失控成長的表。報表讀的是聚合表，原始樣本只有近期查詢用得到 → 短保留 30 天。
  🛑 只刪「聚合表已覆蓋」的日期，聚合沒跑成功的那幾天一律跳過（見函式註解）。

用法（專案根目錄執行）：
    python3 scripts/cleanup_storage.py --dry-run   # 只列統計不刪
    python3 scripts/cleanup_storage.py             # NVMe 快照留 30 天、eMMC 事件 cache 留 3 天
    python3 scripts/cleanup_storage.py --days 60
    python3 scripts/cleanup_storage.py --event-snapshot-days 5

由 systemd timer（traffic-cleanup.timer）每日 02:30 執行。
"""
from __future__ import annotations

import argparse
import datetime
import os
import time

# 相對專案根目錄；data/output/storage 在現場機是 symlink → /mnt/nvme/traffic
DEFAULT_TARGETS = [
    "output/violations",
    "storage/lpr_snapshots",
]


def cleanup_dir(root: str, cutoff_ts: float, dry_run: bool) -> tuple[int, int, int]:
    """遞迴刪除 root 下 mtime 早於 cutoff_ts 的檔案。

    回傳 (刪除檔案數, 釋放 bytes, 錯誤數)。只刪檔案不刪目錄。
    """
    deleted = freed = errors = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
                if st.st_mtime < cutoff_ts:
                    if not dry_run:
                        os.unlink(path)
                    deleted += 1
                    freed += st.st_size
            except OSError:
                errors += 1
    return deleted, freed, errors


def cleanup_camera_media(
    db_path: str, camera_id: int, keep_days: int, dry_run: bool
) -> tuple[int, int]:
    """指定相機的違規「媒體檔」短保留：刪除該相機超過 keep_days 的
    主圖(image_path)與 snapshots/{id}_* 衍生檔(時間軸圖/組合圖/影片cache)。

    ⚠️ 只刪媒體檔，violations 資料列不動（報表/紀錄依政策保留半年）。
    用途：cam_8 測試灌單每天 ~7000 筆 SPEEDING，快照佔 400G+。

    回傳 (刪除檔案數, 釋放 bytes)。
    """
    import sqlite3

    cutoff = time.time() - keep_days * 86400
    cutoff_dt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(cutoff))  # DB 存 UTC naive
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout = 5000")
    rows = conn.execute(
        "SELECT id, image_path FROM violations WHERE camera_id = ? AND created_at < ?",
        (int(camera_id), cutoff_dt),
    ).fetchall()
    conn.close()

    deleted = freed = 0

    def _unlink(path: str) -> None:
        nonlocal deleted, freed
        try:
            size = os.path.getsize(path)
            if not dry_run:
                os.unlink(path)
            deleted += 1
            freed += size
        except OSError:
            pass

    # 主圖：image_path 形如 /files/violations/XXX.jpg → output/violations/XXX.jpg
    ids = set()
    for vid, image_path in rows:
        ids.add(int(vid))
        p = str(image_path or "")
        if p.startswith("/files/"):
            _unlink(p.replace("/files/", "output/", 1))

    # 衍生檔：snapshots/{id}_*.*（單次 scandir 掃全目錄，用 id 前綴比對）
    snap_dir = "output/violations/snapshots"
    if ids and os.path.isdir(snap_dir):
        with os.scandir(snap_dir) as it:
            for entry in it:
                head = entry.name.split("_", 1)[0]
                if head.isdigit() and int(head) in ids and entry.is_file():
                    _unlink(entry.path)
    return deleted, freed


def cleanup_congestion_samples(
    db_path: str, keep_days: int, dry_run: bool, batch: int = 20000
) -> tuple[int, int]:
    """壅塞原始樣本短保留 —— 報表讀的是聚合表，原始樣本只在近期查詢用得到。

    congestion_samples 每天新增約 50 萬列、從不清理（實測 4470 萬列 / DB 19 GB），
    是 DB 唯一的失控成長來源。traffic_events 不在這裡清：api/routes/traffic.py 與
    analytics.py 直接讀原始表，刪了歷史查詢會少資料。

    🛑 安全連鎖：某一天在 congestion_report_aggs 沒有 1h 聚合就「不刪那天」。
    原始樣本刪掉而聚合又沒跑成功 = 那段時間永久消失。2026-05-08~06-15 就是
    聚合從沒跑成功（首次全量聚合 OOM），當時若照天數硬刪，41 天壅塞歷史會全滅。

    分批刪除（預設 2 萬列一批）避免長時間鎖表 —— 服務是邊跑邊清的，
    一次 DELETE 2600 萬列會把寫入卡到 timeout（VD 報表 0/500 就是撞鎖來的）。

    回傳 (刪除列數, 跳過的天數)。
    """
    import sqlite3

    cutoff = time.time() - keep_days * 86400
    cutoff_dt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(cutoff))  # DB 存 UTC naive

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        # 🛑 條件一律寫成 created_at >= 日 AND < 隔日,不要用 substr(created_at,1,10)=日。
        # substr() 讓索引失效:實測 explain query plan 是 SCAN 整條 created_at 索引
        # (4470 萬筆),範圍比較才是 SEARCH 索引定位。逐日刪就是逐日全掃 vs 逐日定位。
        first = conn.execute("SELECT MIN(created_at) FROM congestion_samples").fetchone()[0]
        days = []
        if first:
            day = datetime.date.fromisoformat(str(first)[:10])
            last = datetime.date.fromisoformat(cutoff_dt[:10])
            while day < last:
                days.append(day.isoformat())
                day += datetime.timedelta(days=1)
        covered = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT substr(bucket_start,1,10) FROM congestion_report_aggs "
                "WHERE bucket_size = '1h'"
            ).fetchall()
        }
        deleted = 0
        skipped = []
        for day in days:
            nxt = (datetime.date.fromisoformat(day) + datetime.timedelta(days=1)).isoformat()
            if day not in covered:
                # 只有真的還有資料的日子才值得警告(整段沒資料的空日不用吵)
                exists = conn.execute(
                    "SELECT 1 FROM congestion_samples WHERE created_at >= ? AND created_at < ? LIMIT 1",
                    (day, nxt),
                ).fetchone()
                if exists:
                    skipped.append(day)
                continue
            while True:
                if dry_run:
                    n = conn.execute(
                        "SELECT COUNT(*) FROM congestion_samples "
                        "WHERE created_at >= ? AND created_at < ?",
                        (day, nxt),
                    ).fetchone()[0]
                    deleted += n
                    break
                cur = conn.execute(
                    "DELETE FROM congestion_samples WHERE id IN ("
                    "  SELECT id FROM congestion_samples"
                    "  WHERE created_at >= ? AND created_at < ? LIMIT ?"
                    ")",
                    (day, nxt, int(batch)),
                )
                conn.commit()
                if not cur.rowcount:
                    break
                deleted += cur.rowcount
        if skipped:
            print(f"  ⚠️ 跳過 {len(skipped)} 天（聚合表沒有覆蓋，刪了會永久丟資料）: "
                  f"{skipped[0]}~{skipped[-1]}")
        return deleted, len(skipped)
    finally:
        conn.close()


def _parse_camera_days(items: list[str]) -> dict[int, int]:
    """解析 --camera-days 參數，格式 '8:3' = camera_id 8 保留 3 天"""
    result: dict[int, int] = {}
    for item in items:
        cam_s, _, days_s = str(item).partition(":")
        if cam_s.strip().isdigit() and days_s.strip().isdigit() and int(days_s) >= 1:
            result[int(cam_s)] = int(days_s)
        else:
            print(f"  ⏭️ 忽略無效 --camera-days 項目: {item}（格式應為 8:3 且天數>=1）")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="刪除超過保留天數的快照檔案")
    parser.add_argument("--days", type=int, default=30, help="保留天數（預設 30）")
    parser.add_argument("--paths", nargs="*", default=DEFAULT_TARGETS, help="要清理的目錄")
    parser.add_argument(
        "--camera-days", nargs="*", default=[],
        help="指定相機媒體短保留，格式 camera_id:days（例：8:3 = cam_8 測試快照只留 3 天）",
    )
    parser.add_argument("--db", default="data/violations.db", help="violations DB 路徑")
    parser.add_argument(
        "--event-snapshot-dir", default="/tmp/event_snapshots",
        help="事件快照 cache 目錄（eMMC /tmp，短保留；設空字串可停用）",
    )
    parser.add_argument(
        "--event-snapshot-days", type=int, default=3,
        help="事件快照 cache 保留天數（預設 3；設 0 停用）",
    )
    parser.add_argument(
        "--congestion-sample-days", type=int, default=30,
        help="壅塞原始樣本保留天數（預設 30；設 0 停用）。只刪聚合表已覆蓋的日期",
    )
    parser.add_argument("--dry-run", action="store_true", help="只統計不刪除")
    args = parser.parse_args()

    if args.days < 1:
        print("❌ --days 必須 >= 1（安全防呆）")
        return 1

    cutoff_ts = time.time() - args.days * 86400
    mode = "DRY-RUN" if args.dry_run else "刪除"
    print(f"🧹 清理開始（{mode}，保留 {args.days} 天）")

    total_deleted = total_freed = total_errors = 0
    for target in args.paths:
        if not os.path.isdir(target):
            print(f"  ⏭️ 跳過（不存在）: {target}")
            continue
        deleted, freed, errors = cleanup_dir(target, cutoff_ts, args.dry_run)
        total_deleted += deleted
        total_freed += freed
        total_errors += errors
        print(f"  {target}: {deleted} 檔 / {freed / 1e9:.1f} GB" + (f" / {errors} 錯誤" if errors else ""))

    # 指定相機媒體短保留（cam_8 測試灌單等）
    for cam_id, keep_days in _parse_camera_days(args.camera_days).items():
        if not os.path.exists(args.db):
            print(f"  ⏭️ 跳過 camera-days（DB 不存在）: {args.db}")
            break
        deleted, freed = cleanup_camera_media(args.db, cam_id, keep_days, args.dry_run)
        total_deleted += deleted
        total_freed += freed
        print(f"  cam_{cam_id} 媒體（保留 {keep_days} 天）: {deleted} 檔 / {freed / 1e9:.1f} GB")

    # 事件快照 cache（/tmp/event_snapshots，在 eMMC，成長快 → 獨立短保留，預設 3 天）
    esd = str(args.event_snapshot_dir or "").strip()
    if esd and args.event_snapshot_days >= 1 and os.path.isdir(esd):
        es_cutoff = time.time() - args.event_snapshot_days * 86400
        deleted, freed, errors = cleanup_dir(esd, es_cutoff, args.dry_run)
        total_deleted += deleted
        total_freed += freed
        total_errors += errors
        print(f"  {esd}（保留 {args.event_snapshot_days} 天）: {deleted} 檔 / {freed / 1e9:.1f} GB" + (f" / {errors} 錯誤" if errors else ""))

    # 壅塞原始樣本（DB 資料列，唯一失控成長的表；報表讀聚合表不受影響）
    if args.congestion_sample_days >= 1 and os.path.exists(args.db):
        rows, skipped_days = cleanup_congestion_samples(
            args.db, args.congestion_sample_days, args.dry_run
        )
        note = f" / 跳過 {skipped_days} 天" if skipped_days else ""
        print(f"  congestion_samples（保留 {args.congestion_sample_days} 天）: {rows} 列{note}")

    print(f"✅ 合計: {total_deleted} 檔 / {total_freed / 1e9:.1f} GB" + (f" / {total_errors} 錯誤" if total_errors else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
