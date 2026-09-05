#!/usr/bin/env python3
"""產出動態號誌成效報告(Markdown)。

用法(在 Jetson 上,API 跑在 :8000):
  python3 scripts/signal_report.py --a 2026-09-04T09:00:00 2026-09-04T12:00:00 \\
      --b 2026-09-05T09:00:00 2026-09-05T12:00:00 --tier full \\
      --a-label "09-04 週五 舊參數" --b-label "09-05 新參數" -o docs/reports/20260905_0912.md

不帶 --b 就是單一時段報告。--tier min|standard|full 對應 工程/技術/完整。
認證:讀 AUTH_SECRET 環境變數(沒有就從 `systemctl cat traffic-api` 找),
自簽 tvd_session cookie,跟前端同一套。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _secret() -> str:
    s = os.getenv("AUTH_SECRET", "").strip()
    if s:
        return s
    try:
        out = subprocess.run(["systemctl", "cat", "traffic-api"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "AUTH_SECRET=" in line:
                return line.split("AUTH_SECRET=", 1)[1].strip().strip('"').split()[0]
    except Exception:
        pass
    raise SystemExit("找不到 AUTH_SECRET(環境變數或 systemctl cat traffic-api)")


def _cookie(secret: str) -> str:
    p = base64.urlsafe_b64encode(json.dumps({"u": "admin", "r": "admin", "exp": int(time.time()) + 600}).encode()).rstrip(b"=")
    sig = base64.urlsafe_b64encode(hmac.new(secret.encode(), p, hashlib.sha256).digest()).rstrip(b"=")
    return "tvd_session=" + (p + b"." + sig).decode()


def _get(base: str, path: str, params: dict, cookie: str, timeout: float = 60) -> dict:
    url = base + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"cookie": cookie})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, required=True, metavar=("SINCE", "UNTIL"))
    ap.add_argument("--b", nargs=2, metavar=("SINCE", "UNTIL"))
    ap.add_argument("--tier", default="full", choices=["min", "standard", "full"])
    ap.add_argument("--a-label", default="基準")
    ap.add_argument("--b-label", default="對照")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    cookie = _cookie(_secret())
    q = {"since": args.a[0], "until": args.a[1], "tier": args.tier}
    if args.b:
        q.update({"b_since": args.b[0], "b_until": args.b[1]})
    report = _get(args.base, "/api/signal/shadow/report", q, cookie)
    paired_a = paired_b = None
    sources = [("成效 A/B", args.base + "/api/signal/shadow/report?" + urllib.parse.urlencode(q))]
    if args.tier == "full":
        # include_runs=1:把每一段綠燈的配對明細一起拉,存成 CSV —— 報告上的每個統計都能回到逐段
        paired_a = _get(args.base, "/api/signal/shadow/paired", {"since": args.a[0], "until": args.a[1], "include_runs": 1}, cookie)
        sources.append(("配對 A", args.base + "/api/signal/shadow/paired?since=%s&until=%s" % (args.a[0], args.a[1])))
        if args.b:
            paired_b = _get(args.base, "/api/signal/shadow/paired", {"since": args.b[0], "until": args.b[1], "include_runs": 1}, cookie)
            sources.append(("配對 B", args.base + "/api/signal/shadow/paired?since=%s&until=%s" % (args.b[0], args.b[1])))

    hourly = None
    if args.tier == "full":
        hourly = {}
        for label, win in (("A " + args.a_label, args.a), ("B " + args.b_label, args.b)):
            if not win:
                continue
            day = win[0][:10]
            h0, h1 = int(win[0][11:13]), int(win[1][11:13]) or 24
            try:
                rows = _get(args.base, "/api/signal/shadow/hourly", {"date": day}, cookie).get("rows", [])
                hourly[label] = [r for r in rows if h0 <= int(str(r["hour"])[11:13]) < h1]
            except Exception as e:
                hourly[label] = []
                print(f"[hourly] {label} 取得失敗: {e}", file=sys.stderr)
    if hourly:
        for label in hourly:
            d0 = (args.a if label.startswith("A ") else args.b)[0][:10]
            sources.append(("逐時 " + label, args.base + "/api/signal/shadow/hourly?date=" + d0))
    from detection.signal_report_md import render
    md = render(report, paired_a, paired_b, {"a_label": args.a_label, "b_label": args.b_label, "sources": sources}, hourly)
    if args.out:
        import csv
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"寫入 {out}({len(md)} 字元)")
        # 逐段明細與逐時列存 CSV(報告旁),讓每個統計都能回到原始列
        for tag, pr in (("a", paired_a), ("b", paired_b)):
            runs = (pr or {}).get("runs") or []
            if runs:
                fp = out.with_name(out.stem + f".runs_{tag}.csv")
                with open(fp, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=list(runs[0].keys())); w.writeheader(); w.writerows(runs)
                print(f"寫入 {fp}({len(runs)} 段)")
        for label, rows in (hourly or {}).items():
            if rows:
                fp = out.with_name(out.stem + ".hourly_" + ("a" if label.startswith("A ") else "b") + ".csv")
                with open(fp, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
                print(f"寫入 {fp}({len(rows)} 小時)")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
