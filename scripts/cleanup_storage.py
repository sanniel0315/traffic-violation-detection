#!/usr/bin/env python3
"""儲存空間清理 job — 刪除超過保留天數的違規快照 / LPR 快照。

背景：output/violations（含 snapshots/ 子目錄）與 storage/lpr_snapshots
無 retention 機制，NVMe 塞到 100% 導致 SQLite 偶發 database or disk is full。

用法（專案根目錄執行）：
    python3 scripts/cleanup_storage.py --dry-run   # 只列統計不刪
    python3 scripts/cleanup_storage.py             # 預設保留 30 天
    python3 scripts/cleanup_storage.py --days 60

由 systemd timer（traffic-cleanup.timer）每日 02:30 執行。
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="刪除超過保留天數的快照檔案")
    parser.add_argument("--days", type=int, default=30, help="保留天數（預設 30）")
    parser.add_argument("--paths", nargs="*", default=DEFAULT_TARGETS, help="要清理的目錄")
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

    print(f"✅ 合計: {total_deleted} 檔 / {total_freed / 1e9:.1f} GB" + (f" / {total_errors} 錯誤" if total_errors else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
