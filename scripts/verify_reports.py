#!/usr/bin/env python3
"""VD 車流報表 + 統計報表 端到端回歸驗證。

逐一打「統計分析頁 / VD 車道報表頁」實際用的內部端點,以及對外 API
(`/api/v1/external/*`),檢查 HTTP 狀態與回應結構,並附真實資料量佐證。

用法(專案根目錄執行,服務需已啟動):
    python3 scripts/verify_reports.py                      # 打 http://127.0.0.1:8000
    python3 scripts/verify_reports.py --base-url http://100.92.17.87:8000

對外端點需 API key:優先讀環境變數 EXTERNAL_API_KEY,否則讀 ./.env 同名值。
缺 key 時對外測試標 SKIP(不失敗),仍會驗證「缺 key → 401」的認證防線。

離開碼 = 失敗項數(全過為 0),方便接 CI / 部署後 smoke。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

EXT_PREFIX = "/api/v1/external"


def load_api_key() -> str:
    """API key:先環境變數 EXTERNAL_API_KEY,再 ./.env。找不到回空字串。"""
    key = os.getenv("EXTERNAL_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(".env", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                m = re.match(r"\s*EXTERNAL_API_KEY\s*=\s*(.+?)\s*$", line)
                if m:
                    return m.group(1).strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def iso(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def q(**kw) -> str:
    """組 query string,urlencode 確保 '+00:00' 的 '+' 不被當成空格(422 常見坑)。"""
    return "?" + urllib.parse.urlencode({k: v for k, v in kw.items() if v is not None})


def call(base: str, path: str, headers: dict | None = None):
    req = urllib.request.Request(base + path, headers=headers or {})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()
    except Exception as e:  # noqa: BLE001 — 連不上也算一筆失敗
        return -1, "", str(e).encode()


def js(body: bytes):
    try:
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="VD + 統計報表端到端回歸驗證")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    key = load_api_key()

    now = datetime.datetime.now(datetime.timezone.utc)
    t2h = iso(now - datetime.timedelta(hours=2))
    t1h = iso(now - datetime.timedelta(hours=1))
    tnow = iso(now)

    results: list[tuple[str, bool, str]] = []  # (name, ok, detail); ok=None → SKIP

    def rec(name, ok, detail):
        results.append((name, ok, detail))

    # ===== 內部(統計分析頁 / VD 頁實際呼叫,無需 auth) =====
    s, ct, b = call(base, "/api/traffic/events/trend" + q(
        bucket_sec=3600, mode="count", start_time=t2h, end_time=tnow, max_buckets=100))
    d = js(b); nb = len(d.get("buckets", [])) if d else 0
    rec("內部 events/trend (事件趨勢·count)", s == 200 and bool(d) and "buckets" in d,
        "status=%s buckets=%s" % (s, nb))

    s, ct, b = call(base, "/api/traffic/events/trend" + q(
        bucket_sec=3600, mode="speed", start_time=t2h, end_time=tnow, max_buckets=100))
    d = js(b); nb = len(d.get("buckets", [])) if d else 0
    rec("內部 events/trend (車速趨勢·speed)", s == 200 and d is not None,
        "status=%s buckets=%s" % (s, nb))

    s, ct, b = call(base, "/api/traffic/events" + q(
        start_time=t1h, end_time=tnow, include_total="true", page_size=5))
    d = js(b); tot = (d or {}).get("total")
    rec("內部 events (事件清單/摘要磚)", s == 200 and bool(d) and "items" in d,
        "status=%s total=%s" % (s, tot))

    s, ct, b = call(base, "/api/traffic/vd-report" + q(
        start_time=t2h, end_time=tnow, bucket_size="5m"))
    d = js(b); items = (d or {}).get("items", [])
    rec("內部 vd-report (VD 車道報表頁)", s == 200 and bool(d) and "items" in d,
        "status=%s rows=%s" % (s, len(items)))

    s, ct, b = call(base, "/api/congestion/samples" + q(start_time=t2h, end_time=tnow))
    d = js(b)
    n_samp = len((d or {}).get("items", []) if isinstance(d, dict) else (d or []))
    rec("內部 congestion/samples (車流趨勢資料源)", s == 200 and d is not None,
        "status=%s samples=%s" % (s, n_samp))

    s, ct, b = call(base, "/api/traffic/events" + q(start_time=t1h, end_time=tnow, page_size=1))
    d = js(b); ev = (d or {}).get("items", []); eid = ev[0].get("id") if ev else None
    if eid:
        s2, ct2, b2 = call(base, "/api/traffic/events/%s/snapshot.jpg" % eid)
        isimg = ct2.startswith("image/") or (len(b2) > 500 and b2[:3] == b"\xff\xd8\xff")
        rec("內部 event snapshot.jpg", s2 in (200, 404) and (s2 == 404 or isimg),
            "status=%s ct=%s bytes=%s" % (s2, ct2, len(b2)))
    else:
        rec("內部 event snapshot.jpg", None, "跳過(近 1h 無事件可取樣)")

    # ===== 對外(需 X-API-Key,prefix /api/v1/external) =====
    H = {"X-API-Key": key}
    if key:
        s, ct, b = call(base, EXT_PREFIX + "/vd-report" + q(
            start_time=t2h, end_time=tnow, interval="5m"), H)
        d = js(b); recs = ((d or {}).get("data") or {}).get("records", [])
        rec("對外 external/vd-report (JSON)", s == 200 and (d or {}).get("status") == "success",
            "status=%s records=%s" % (s, len(recs)))

        s, ct, b = call(base, EXT_PREFIX + "/vd-report" + q(
            start_time=t2h, end_time=tnow, interval="5m", format="csv"), H)
        iscsv = "text/csv" in ct or b[:11] == b"detector_id"
        rec("對外 external/vd-report (CSV)", s == 200 and iscsv, "status=%s ct=%s" % (s, ct))

        s, ct, b = call(base, EXT_PREFIX + "/vd-report/latest" + q(minutes=5, interval="1m"), H)
        d = js(b); st = ((d or {}).get("data") or {}).get("stats")
        rec("對外 external/vd-report/latest", s == 200 and st is not None, "status=%s" % s)

        s, ct, b = call(base, EXT_PREFIX + "/congestion-report" + q(
            start_time=t2h, end_time=tnow, interval="5m"), H)
        d = js(b); recs = ((d or {}).get("data") or {}).get("records", [])
        rec("對外 external/congestion-report", s == 200 and (d or {}).get("status") == "success",
            "status=%s records=%s" % (s, len(recs)))

        s, ct, b = call(base, EXT_PREFIX + "/congestion-report/latest" + q(minutes=5, interval="1m"), H)
        d = js(b)
        rec("對外 external/congestion-report/latest", s == 200 and (d or {}).get("status") == "success",
            "status=%s" % s)

        s, ct, b = call(base, EXT_PREFIX + "/streams", H)
        d = js(b); data = (d or {}).get("data")
        strs = data.get("streams", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        rec("對外 external/streams", s == 200 and (d or {}).get("status") == "success",
            "status=%s streams=%s" % (s, len(strs) if isinstance(strs, list) else strs))
    else:
        for name in ("vd-report (JSON)", "vd-report (CSV)", "vd-report/latest",
                     "congestion-report", "congestion-report/latest", "streams"):
            rec("對外 external/" + name, None, "跳過(未設 EXTERNAL_API_KEY)")

    # 負向:認證防線(不需 key 也能驗)
    s, ct, b = call(base, EXT_PREFIX + "/vd-report" + q(start_time=t2h, end_time=tnow))
    rec("負向 external 缺 key → 401", s == 401, "status=%s" % s)

    s, ct, b = call(base, EXT_PREFIX + "/streams", {"X-API-Key": "tvd_invalid_xxx"})
    rec("負向 external 亂 key → 401", s == 401, "status=%s" % s)

    # ===== 輸出 =====
    print("base:", base, "| API key:", "loaded" if key else "NOT set(對外測試 SKIP)")
    print("=" * 64)
    passed = failed = skipped = 0
    for name, ok, detail in results:
        if ok is None:
            tag = "[SKIP]"; skipped += 1
        elif ok:
            tag = "[PASS]"; passed += 1
        else:
            tag = "[FAIL]"; failed += 1
        print("%s %s  %s" % (tag, name, detail))
    print("=" * 64)
    print("PASS %s / FAIL %s / SKIP %s" % (passed, failed, skipped))
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
