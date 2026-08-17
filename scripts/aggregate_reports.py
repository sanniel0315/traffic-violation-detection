#!/usr/bin/env python3
"""報表聚合 job"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from api.models import SessionLocal
from api.utils.report_aggregation import run_incremental_report_aggregation, to_utc_naive


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    dt = datetime.fromisoformat(raw)
    return to_utc_naive(dt)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate traffic / congestion / LPR report tables")
    parser.add_argument("--start", help="ISO datetime, e.g. 2026-03-24T00:00:00+08:00")
    parser.add_argument("--end", help="ISO datetime, e.g. 2026-03-24T23:59:59+08:00")
    parser.add_argument("--chunk-hours", type=int, default=12,
                        help="回填時每塊的時數(預設 12)。整段跑在單一交易會長時間佔住"
                             "寫鎖,與即時事件寫入互卡 → database is locked。0=不分塊")
    args = parser.parse_args()

    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    chunk = max(0, int(args.chunk_hours or 0))

    # 沒給區間、或不分塊 → 維持原本的單次行為(增量 job 就是這樣用的)
    if start is None or end is None or chunk <= 0:
        db = SessionLocal()
        try:
            result = run_incremental_report_aggregation(db, start_time=start, end_time=end)
            print("aggregation_result", result)
            return 0
        finally:
            db.close()

    # 回填長區間:切塊逐段跑,每塊自己一個 session/交易。
    # 聚合表的洞就是這樣來的 —— 服務首次啟動只回填近 7 天
    # (api/main.py _REPORT_AGG_BACKFILL_DAYS),而且只在沒有 job state 時跑一次,
    # 更早的資料永遠不會被聚合。要補洞就用這個模式指定完整區間。
    # run_incremental_report_aggregation 回的是巢狀 dict(traffic/congestion/lpr
    # 各自再分 bucket_size),不是單層計數,所以要遞迴合併
    def _merge(acc: dict, src: dict) -> None:
        for k, v in (src or {}).items():
            if isinstance(v, dict):
                _merge(acc.setdefault(k, {}), v)
            else:
                try:
                    acc[k] = acc.get(k, 0) + int(v or 0)
                except (TypeError, ValueError):
                    acc[k] = v

    total: dict = {}
    cur = start
    n_chunks = 0
    while cur < end:
        nxt = min(cur + timedelta(hours=chunk), end)
        db = SessionLocal()
        try:
            res = run_incremental_report_aggregation(db, start_time=cur, end_time=nxt)
        finally:
            db.close()
        _merge(total, res or {})
        n_chunks += 1
        print(f"  [{n_chunks}] {cur} ~ {nxt}  {res}", flush=True)
        cur = nxt
    print("aggregation_result", total, f"({n_chunks} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
