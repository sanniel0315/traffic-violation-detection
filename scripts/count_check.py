#!/usr/bin/env python3
"""進出計數準確度驗證:錄一段影片人工數,跟系統同時窗的數字比。

沒有真值就沒有「精確」可言 —— 調參數只是把一組數字換成另一組。這支腳本
把驗證流程固定下來:

  1) record  錄 N 分鐘原始影片(畫上 ROI 與進出線),並記下精確的起訖時間
  2) 人工看影片數:各框進了幾台、出了幾台
  3) report  讀 DB 取同一時窗的系統計數,算誤差

🛑 不碰執行中的 traffic-api:自己開一路 RTSP 讀影像,只對 DB 做唯讀查詢。

用法:
    # 錄 5 分鐘(邊錄邊顯示進度),錄完會印出 report 指令
    python3 scripts/count_check.py record --camera 3 --minutes 5

    # 人工數完之後填進去(可只填一個框)
    python3 scripts/count_check.py report --camera 3 \
        --start '2026-08-15 10:20:00' --end '2026-08-15 10:25:00' \
        --truth '上匝道=42/40'
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "violations.db")


def _load_camera(camera_id: int):
    """取相機來源與車流 zone(與線上同一條 scope 規則)。"""
    from api.models import SessionLocal, Camera
    from api.utils.roi_scope import SCOPE_TRAFFIC, SCOPE_CONGESTION, select_zones
    db = SessionLocal()
    try:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if not cam:
            raise SystemExit(f"找不到 camera_id={camera_id}")
        zones = select_zones(cam.zones or [], scope=SCOPE_TRAFFIC,
                             allowed_types=("detection", "flow_detection"),
                             fallback_scopes=(SCOPE_CONGESTION,))
        src = cam.source or (f"rtsp://{cam.username}:{cam.password}@"
                             f"{cam.ip}:{cam.port}{cam.stream_path}")
        return cam.name, src, zones
    finally:
        db.close()


def _draw_zones(frame, zones):
    """畫出框與進出線 —— 人工計數時要看得出邊界在哪、哪條是進哪條是出。"""
    for z in zones:
        pts = np.array(z.get("points") or [], dtype=np.int32)
        if len(pts) < 3:
            continue
        cv2.polylines(frame, [pts], True, (80, 220, 80), 2)
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            tag = ""
            if str(z.get("in_edge")) == str(i):
                tag, col = "IN", (255, 200, 0)
            elif str(z.get("out_edge")) == str(i):
                tag, col = "OUT", (0, 160, 255)
            if not tag:
                continue
            cv2.line(frame, tuple(a), tuple(b), col, 4)
            m = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            cv2.putText(frame, f"{z.get('name')} {tag}", (m[0] - 60, m[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return frame


def cmd_record(args) -> int:
    name, src, zones = _load_camera(args.camera)
    if not zones:
        print("這台沒有車流 zone,不用驗")
        return 1
    print(f"相機 {args.camera} {name}")
    for z in zones:
        print(f"  框 {z.get('name')!r}  進=邊{z.get('in_edge')} 出=邊{z.get('out_edge')}")

    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"開不了串流: {src}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    # 人工計數用,不需要原始幀率;降到 8fps 檔案小很多又不會漏看車
    out_fps = 8.0
    step = max(1, int(round(fps / out_fps)))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    t_start = time.time()
    started_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t_start))
    total = args.minutes * 60
    n = 0
    while time.time() - t_start < total:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        n += 1
        if n % step:
            continue
        vw.write(_draw_zones(frame, zones))
        left = total - (time.time() - t_start)
        if int(left) % 30 == 0:
            print(f"  錄影中… 剩 {left / 60:.1f} 分", flush=True)
            time.sleep(1)
    ended_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    cap.release()
    vw.release()

    print(f"\n影片: {args.out}")
    print(f"時窗(UTC): {started_utc} ~ {ended_utc}")
    print("\n人工數完各框「進幾台/出幾台」之後,跑:")
    truth = " ".join(f"--truth '{z.get('name')}=進/出'" for z in zones)
    print(f"  python3 scripts/count_check.py report --camera {args.camera} \\\n"
          f"      --start '{started_utc}' --end '{ended_utc}' {truth}")
    return 0


def cmd_report(args) -> int:
    truth = {}
    for item in args.truth or []:
        try:
            k, v = item.split("=", 1)
            i, o = v.split("/", 1)
            truth[k.strip()] = (int(i), int(o))
        except ValueError:
            print(f"⚠️ --truth 格式應為 '框名=進/出',收到 {item!r},略過")

    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = c.execute(
        "select entered_zones, direction, count(*) from traffic_events "
        "where camera_id=? and created_at >= ? and created_at < ? group by 1,2",
        (args.camera, args.start, args.end)).fetchall()
    agg: dict[str, dict] = {}
    for z, d, n in rows:
        try:
            nm = (json.loads(z) or [""])[0]
        except (TypeError, ValueError):
            nm = str(z)
        agg.setdefault(nm, {})[d] = n

    print(f"時窗(UTC) {args.start} ~ {args.end}   相機 {args.camera}")
    if not agg:
        print("  這個時窗沒有任何事件 —— 確認時間是 UTC(台灣時間要減 8 小時)")
        return 1
    print(f"  {'框':22} {'系統進':>7} {'系統出':>7} {'一般流量':>9}   誤差(對人工)")
    for nm, v in agg.items():
        si, so = v.get("IN", 0), v.get("EXIT", 0)
        fl = v.get("INOUT", 0) + v.get("straight", 0)
        err = ""
        if nm in truth:
            ti, to = truth[nm]
            ei = (si - ti) / ti * 100 if ti else float("nan")
            eo = (so - to) / to * 100 if to else float("nan")
            err = f"進 {ei:+.0f}% (人工 {ti})   出 {eo:+.0f}% (人工 {to})"
        elif truth:
            err = "(沒填人工數)"
        print(f"  {nm:22} {si:>7} {so:>7} {fl:>9}   {err}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="進出計數準確度驗證")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="錄一段帶 ROI 標示的影片供人工計數")
    r.add_argument("--camera", type=int, required=True)
    r.add_argument("--minutes", type=float, default=5.0)
    r.add_argument("--out", default="/tmp/count_check.mp4")
    r.set_defaults(func=cmd_record)

    p = sub.add_parser("report", help="比對系統計數與人工計數")
    p.add_argument("--camera", type=int, required=True)
    p.add_argument("--start", required=True, help="UTC 起始 'YYYY-MM-DD HH:MM:SS'")
    p.add_argument("--end", required=True, help="UTC 結束")
    p.add_argument("--truth", action="append", help="'框名=進/出',可重複")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
