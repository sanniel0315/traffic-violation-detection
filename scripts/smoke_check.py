#!/usr/bin/env python3
"""Basic smoke checks for API + VD report integration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request


TZ_TAIPEI = dt.timezone(dt.timedelta(hours=8))


class CheckError(RuntimeError):
    pass


def _http_get(url: str, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"Accept": "application/json,text/html"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body


def _get_json(url: str, timeout: float) -> dict:
    code, body = _http_get(url, timeout)
    if code != 200:
        raise CheckError(f"GET {url} failed with HTTP {code}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise CheckError(f"GET {url} returned non-JSON payload") from exc


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise CheckError(message)


def _range_query(base_url: str, start: dt.datetime, end: dt.datetime) -> dict:
    params = {
        "page": "1",
        "page_size": "1",
        "include_total": "1",
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
    }
    url = f"{base_url}/api/traffic/events?{urllib.parse.urlencode(params)}"
    return _get_json(url, timeout=10.0)


def run(base_url: str, timeout: float) -> None:
    print("[1/5] health check")
    health = _get_json(f"{base_url}/api/health", timeout=timeout)
    _assert(str(health.get("status", "")).lower() == "healthy", "health status is not healthy")

    print("[2/5] cameras API shape check")
    cameras = _get_json(f"{base_url}/api/cameras", timeout=timeout)
    items = cameras.get("items", [])
    _assert(isinstance(items, list), "/api/cameras.items is not a list")

    print("[3/5] traffic range query checks (today/24h/7d)")
    now_local = dt.datetime.now(TZ_TAIPEI)
    today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    d24_start = now_local - dt.timedelta(hours=24)
    d7_start = now_local - dt.timedelta(days=7)

    q_today = _range_query(base_url, today_start, now_local)
    q_24h = _range_query(base_url, d24_start, now_local)
    q_7d = _range_query(base_url, d7_start, now_local)

    for name, payload in (("today", q_today), ("24h", q_24h), ("7d", q_7d)):
        _assert(isinstance(payload.get("items"), list), f"{name} query payload.items is not a list")
        _assert("has_more" in payload, f"{name} query payload missing has_more")

    n_today = int(q_today.get("total") or 0)
    n_24h = int(q_24h.get("total") or 0)
    n_7d = int(q_7d.get("total") or 0)
    _assert(n_24h >= n_today, f"24h events({n_24h}) < today events({n_today})")
    _assert(n_7d >= n_24h, f"7d events({n_7d}) < 24h events({n_24h})")

    print("[4/5] verify deployed VD frontend markers")
    code, html = _http_get(f"{base_url}/web/index.html", timeout=timeout)
    _assert(code == 200, "cannot load /web/index.html")
    markers = [
        "loadVdReport(true)",
        "const parseVdTimeRange=()=>",
        "VD 報表查詢完成",
    ]
    for marker in markers:
        _assert(marker in html, f"frontend marker missing: {marker}")

    print("[5/5] summary")
    print(
        "PASS",
        json.dumps(
            {
                "health": health.get("status"),
                "camera_count": len(items),
                "events_today": n_today,
                "events_24h": n_24h,
                "events_7d": n_7d,
                "today_has_more": bool(q_today.get("has_more")),
                "h24_has_more": bool(q_24h.get("has_more")),
                "d7_has_more": bool(q_7d.get("has_more")),
            },
            ensure_ascii=False,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke checks for traffic-violation-detection")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    try:
        run(args.base_url.rstrip("/"), timeout=args.timeout)
        return 0
    except CheckError as exc:
        print(f"FAIL {exc}")
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"FAIL unexpected error: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
