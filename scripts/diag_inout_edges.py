#!/usr/bin/env python3
"""診斷:車輛實際跨越 ROI 的哪一條邊。

IN/OUT 計數恆為 0 時,分不出是「轉場沒發生」還是「跨線判定不過」——
兩者都沒有錯誤訊息。這支腳本用獨立程序重跑同一套判定,把過程印出來。

🛑 完全不碰執行中的 traffic-api:自己開一路 RTSP、自己跑偵測,
   看完就結束。不要為了診斷去改線上偵測迴圈(2026-08-15 那樣做過,
   把事件寫入弄停了)。

用法:
    python3 scripts/diag_inout_edges.py --camera 3 --seconds 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from api.routes.stream import (  # noqa: E402
    _seg_intersect,
    _vehicle_center,
    _zone_edge_segment,
)


def load_zones(camera_id: int):
    """從 DB 取該相機的車流 zone(含 scope 過濾,與線上同一條規則)。"""
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
        src = cam.source or f"rtsp://{cam.username}:{cam.password}@{cam.ip}:{cam.port}{cam.stream_path}"
        return cam.name, src, zones
    finally:
        db.close()


def point_in_zone(pt, zone) -> bool:
    pts = zone.get("points") or []
    if len(pts) < 3:
        return False
    poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def _verdict(zone, direction, prev_pt, cur_pt) -> str:
    """套用線上 stream.py 的同一條接受規則,回傳這筆轉場會不會被寫成事件。

    規則(2026-08-15 起):標記的邊代表「哪一側是進、哪一側是出」——
    先試嚴格跨線,跨不到就看車是不是順著「進線中點→出線中點」這條車流軸走。
    """
    in_seg = _zone_edge_segment(zone, zone.get("in_edge"))
    out_seg = _zone_edge_segment(zone, zone.get("out_edge"))
    edge = in_seg if direction == "IN" else out_seg
    if edge is None:
        return "不計:該方向沒標線"
    if prev_pt is None:
        # 第一次被偵測到就已在框內(框的入口比 YOLO 認得出車的距離還遠)。
        # 線上是照算的 —— 擋掉會讓 IN 少算 11/16,連帶它的出場也被擋掉。
        return "計入(無法反證)"
    if _seg_intersect(prev_pt, cur_pt, edge[0], edge[1]):
        return "計入(跨線)"
    if in_seg is None or out_seg is None:
        return "計入(只標一條)"
    axis = ((out_seg[0][0] + out_seg[1][0]) / 2.0 - (in_seg[0][0] + in_seg[1][0]) / 2.0,
            (out_seg[0][1] + out_seg[1][1]) / 2.0 - (in_seg[0][1] + in_seg[1][1]) / 2.0)
    mv = (cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1])
    if axis[0] * mv[0] + axis[1] * mv[1] > 0:
        return "計入(順車流軸)"
    return "不計:逆車流軸"


def _axis(zone):
    """車流軸:進線中點 → 出線中點。回傳 (起點, 單位向量, 長度)。"""
    ins = _zone_edge_segment(zone, zone.get("in_edge"))
    outs = _zone_edge_segment(zone, zone.get("out_edge"))
    if ins is None or outs is None:
        return None
    a = ((ins[0][0] + ins[1][0]) / 2.0, (ins[0][1] + ins[1][1]) / 2.0)
    b = ((outs[0][0] + outs[1][0]) / 2.0, (outs[0][1] + outs[1][1]) / 2.0)
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = (dx * dx + dy * dy) ** 0.5
    if L <= 0:
        return None
    return a, (dx / L, dy / L), L


def _clip_halfplane(pts, origin, normal, cut):
    """Sutherland-Hodgman:只留下 dot(p-origin, normal) >= cut 的那半邊。"""
    def t(p):
        return (p[0] - origin[0]) * normal[0] + (p[1] - origin[1]) * normal[1] - cut
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ta, tb = t(a), t(b)
        if ta >= 0:
            out.append(a)
        if (ta >= 0) != (tb >= 0):
            r = ta / (ta - tb)
            out.append([round(a[0] + r * (b[0] - a[0])), round(a[1] + r * (b[1] - a[1]))])
    return out


def suggest_entry(zone, births_inside):
    """依「車第一次被偵測到的位置」算出入口邊該退到哪裡,回傳新的多邊形。

    框的入口邊比 YOLO 認得出車的距離還遠時,車進框那一刻沒被看到,進場只能
    用推定的 —— 把入口邊退到車已經穩定被偵測到的位置之後,進場就變成真正
    觀察得到的跨線事件。取 p90 而不是最深的那台,免得一個離群值把框砍太多。
    """
    ax = _axis(zone)
    if ax is None or not births_inside:
        return None
    origin, n, L = ax
    ts = sorted((p[0] - origin[0]) * n[0] + (p[1] - origin[1]) * n[1]
                for p in births_inside)
    p90 = ts[min(len(ts) - 1, int(len(ts) * 0.9))]
    cut = p90 + 10.0          # 再退 10px,讓最深的那批也確實落在框外
    if cut >= L * 0.8:
        return None           # 退太多會把框砍到只剩尾巴,不建議
    pts = [[int(p[0]), int(p[1])] for p in (zone.get("points") or [])]
    return {
        "cut": cut, "axis_len": L, "ts": ts,
        "points": _clip_halfplane(pts, origin, n, cut),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, required=True)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--out", default="/tmp/diag_inout.jpg")
    # 預設比照線上參數,診斷結果才有代表性:
    #   fps      線上實測 cam2 3.3 / cam3 3.1 / cam4 1.4 / cam5 0.2
    #   max-dist _nearest_track_id 的 90px
    #   ttl      speed_track_ttl_sec 預設 5 秒
    ap.add_argument("--fps", type=float, default=3.1, help="取樣幀率(比照線上偵測速率)")
    ap.add_argument("--max-dist", type=float, default=90.0, help="track 配對距離上限(px)")
    ap.add_argument("--ttl", type=float, default=5.0, help="track 存活秒數")
    args = ap.parse_args()

    name, src, zones = load_zones(args.camera)
    print(f"相機 {args.camera} {name}")
    for z in zones:
        n = len(z.get("points") or [])
        print(f"  zone {z.get('name')!r}: {n} 個點  direction={z.get('direction')!r} "
              f"in_edge={z.get('in_edge')!r} out_edge={z.get('out_edge')!r}")
    if not zones:
        print("沒有車流 zone,無法診斷")
        return 1

    from detection.vehicle_detector import VehicleDetector
    det = VehicleDetector()

    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"開不了串流: {src}")
        return 1

    # track_id 用最近鄰配對(與線上 _nearest_track_id 同精神,這裡簡化)
    tracks: dict[int, dict] = {}
    births = []          # (第一次被偵測到的位置, 當下是否已在框內)
    last_frame = None
    next_id = 1
    crossed = Counter()      # (zone, 方向, 邊號) → 次數
    transitions = Counter()  # (zone, 方向) → 次數
    verdicts = Counter()     # (zone, 方向, 結論) → 次數  ← 線上實際會不會寫入事件
    died_inside = Counter()  # zone → track 在框內就過期的次數(這些車的 EXIT 觀察不到)
    lost_exit = Counter()    # zone → 算過進場、track 卻在框內過期 → 掉掉的 out_flow
    frames = 0
    period = 1.0 / max(0.1, args.fps)
    t0 = time.time()
    next_sample = t0

    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frames += 1
        last_frame = frame
        now = time.time()
        if now < next_sample:   # 依 --fps 取樣,比照線上偵測速率
            continue
        next_sample = now + period
        vehicles = det.detect(frame) or []
        # 過期清除(比照線上 track_ttl_sec):在框內就過期 = 線上看不到那筆 EXIT
        for tid in [t for t, tr in tracks.items() if now - tr["t"] > args.ttl]:
            _gone = tracks.pop(tid)
            for zk in _gone["inside"]:
                died_inside[zk] += 1
            for zk in _gone["counted"]:
                lost_exit[zk] += 1   # 算過進場、卻沒等到出場 → 這筆 out_flow 就是這樣掉的
        seen = set()
        for v in vehicles:
            c = _vehicle_center(v)
            # 最近鄰配對(比照線上 _nearest_track_id:取整體最近,超過 max_dist 就開新 track)
            best, bestd = None, 1e9
            for tid, tr in tracks.items():
                if tid in seen:
                    continue
                d = ((tr["pt"][0] - c[0]) ** 2 + (tr["pt"][1] - c[1]) ** 2) ** 0.5
                if d < bestd:
                    best, bestd = tid, d
            fresh = best is None or bestd > args.max_dist
            if fresh:
                best = next_id
                next_id += 1
                tracks[best] = {"pt": None, "inside": set(), "counted": set(), "t": now}
                # 第一次被偵測到的位置 —— 已經在框內的話,線上沒有「框外那一段」可連線
                births.append((c, [z for z in zones if point_in_zone(c, z)]))
            seen.add(best)
            tr = tracks[best]
            prev_pt, prev_in = tr["pt"], tr["inside"]
            cur_in = {str(z.get("name") or id(z)) for z in zones if point_in_zone(c, z)}
            tr["pt"] = c
            tr["t"] = now
            if cur_in != prev_in:
                for z in zones:
                    zk = str(z.get("name") or id(z))
                    if zk in (cur_in - prev_in):
                        d = "IN"
                    elif zk in (prev_in - cur_in):
                        d = "EXIT"
                    else:
                        continue
                    transitions[(zk, d)] += 1
                    # 線上還有一道閘門:EXIT 必須是「同一個 track 算過進場」才計。
                    # 這裡照抄,才看得出線上的 EXIT 是被哪一關擋掉的。
                    if d == "EXIT" and zk not in tr["counted"]:
                        verdicts[(zk, d, "不計:這個 track 沒算過進場")] += 1
                        continue
                    hits = [i for i in range(len(z.get("points") or []))
                            if prev_pt is not None
                            and (lambda s: s is not None and _seg_intersect(prev_pt, c, s[0], s[1]))(
                                _zone_edge_segment(z, i))]
                    for i in hits:
                        crossed[(zk, d, i + 1)] += 1
                    if not hits:
                        crossed[(zk, d, "無(跳幀/框內)")] += 1
                    v = _verdict(z, d, prev_pt, c)
                    verdicts[(zk, d, v)] += 1
                    if v.startswith("計入"):
                        if d == "IN":
                            tr["counted"].add(zk)
                        else:
                            tr["counted"].discard(zk)
            tr["inside"] = cur_in

    cap.release()

    # 把「車輛第一次被偵測到的位置」畫出來 —— 要調 ROI 的進入側邊界,
    # 就是要看這些點落在框內還是框外。落在框內 = 來不及看到跨線 = IN 恆 0。
    if last_frame is not None:
        vis = last_frame.copy()
        for z in zones:
            pts = np.array(z.get("points") or [], dtype=np.int32)
            if len(pts) >= 3:
                cv2.polylines(vis, [pts], True, (80, 200, 80), 2)
                for i in range(len(pts)):
                    a, b = pts[i], pts[(i + 1) % len(pts)]
                    m = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                    tag = f"E{i+1}"
                    if str(z.get("in_edge")) == str(i):
                        tag += "=IN"
                    if str(z.get("out_edge")) == str(i):
                        tag += "=OUT"
                    cv2.rectangle(vis, (m[0] - 26, m[1] - 12), (m[0] + 34, m[1] + 10), (0, 0, 0), -1)
                    cv2.putText(vis, tag, (m[0] - 22, m[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for pt, in_zs in births:
            # 紅=第一次就在框內(問題點)  藍=在框外(正常,追得到跨線)
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 7,
                       (0, 0, 255) if in_zs else (255, 160, 0), -1)
        cv2.imwrite(args.out, vis)
        inside_n = sum(1 for _, i in births if i)
        print(f"\n已輸出 {args.out}")
        print(f"  紅點 = 第一次被偵測到時已在框內 ({inside_n}/{len(births)})  ← 這些的 IN 永遠算不到")
        print(f"  藍點 = 第一次在框外 ({len(births) - inside_n}/{len(births)})  ← 這些才追得到跨線")

    print(f"\n讀入 {frames} 幀 / {args.seconds} 秒,"
          f"取樣 {args.fps} fps、配對 {args.max_dist:.0f}px、TTL {args.ttl:.0f}s(比照線上)")
    print("\n=== 轉場次數(車輛進出 ROI) ===")
    for (zk, d), n in sorted(transitions.items()):
        print(f"  {zk} {d}: {n}")
    if not transitions:
        print("  (完全沒有轉場 —— 車輛沒有進出這個框,或框的位置不在車道上)")
    print("\n=== 轉場當下實際跨越的邊 ===")
    for (zk, d, e), n in sorted(crossed.items(), key=lambda x: -x[1]):
        print(f"  {zk} {d} → 邊{e}: {n} 次")
    print("\n=== 線上規則下會不會寫成事件 ===")
    for (zk, d, v), n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {zk} {d} → {v}: {n} 次")
    print("\n=== track 在框內就過期(這些車的 EXIT 觀察不到) ===")
    if died_inside:
        for zk, n in died_inside.most_common():
            print(f"  {zk}: {n} 次")
    else:
        print("  (沒有)")

    print("\n=== 算過進場、卻沒等到出場(掉掉的 out_flow) ===")
    if lost_exit:
        for zk, n in lost_exit.most_common():
            print(f"  {zk}: {n} 次")
    else:
        print("  (沒有)")

    print("\n=== 入口邊建議(依車第一次被偵測到的位置) ===")
    for z in zones:
        nm = str(z.get("name") or "")
        pts_in = [pt for pt, in_zs in births if any(zz is z for zz in in_zs)]
        if not pts_in:
            print(f"  {nm}: 沒有「第一次出現就在框內」的車 → 入口邊位置沒問題")
            continue
        sug = suggest_entry(z, pts_in)
        if sug is None:
            print(f"  {nm}: {len(pts_in)} 台第一次出現就在框內,"
                  f"但沒標進出線、或退太多會把框砍掉 → 不建議自動調")
            continue
        ts = sug["ts"]
        print(f"  {nm}: {len(pts_in)} 台第一次出現就在框內")
        print(f"    沿車流軸位置 最淺 {ts[0]:.0f}px / 中位 {ts[len(ts) // 2]:.0f}px / "
              f"最深 {ts[-1]:.0f}px  (框全長 {sug['axis_len']:.0f}px)")
        print(f"    建議入口邊退到 {sug['cut']:.0f}px,新多邊形:")
        print(f"    {sug['points']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
