#!/usr/bin/env python3
"""報表聚合 job"""
from __future__ import annotations

import argparse
from datetime import datetime

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
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = run_incremental_report_aggregation(
            db,
            start_time=_parse_dt(args.start),
            end_time=_parse_dt(args.end),
        )
        print("aggregation_result", result)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
