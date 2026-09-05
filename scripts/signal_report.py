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
    if args.tier == "full":
        paired_a = _get(args.base, "/api/signal/shadow/paired", {"since": args.a[0], "until": args.a[1]}, cookie)
        if args.b:
            paired_b = _get(args.base, "/api/signal/shadow/paired", {"since": args.b[0], "until": args.b[1]}, cookie)

    from detection.signal_report_md import render
    md = render(report, paired_a, paired_b, {"a_label": args.a_label, "b_label": args.b_label})
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"寫入 {out}({len(md)} 字元)")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
