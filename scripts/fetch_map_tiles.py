#!/usr/bin/env python3
"""下載離線地圖圖磚 — 現場是封閉網段，線上圖磚連不到，地圖會整片空白。

把站台周邊的圖磚抓下來放進 `web/tiles/{z}/{x}/{y}.jpg`，
前端 `L.tileLayer('/web/tiles/{z}/{x}/{y}.jpg')` 直接吃本地檔。

用法（換站台就改座標重跑）:
    python3 scripts/fetch_map_tiles.py --lat 23.063772 --lon 120.279169
    python3 scripts/fetch_map_tiles.py --lat X --lon Y --dry-run    # 只估數量
    python3 scripts/fetch_map_tiles.py --lat X --lon Y --source nlsc-photo   # 正射影像

圖磚來源預設「國土測繪中心 WMTS 通用電子地圖」（公開介接，允許取用）。

🛑 不要改回 OpenStreetMap 官方 tile server。
   2026-08-13 實測：osm.org 對大量下載回**HTTP 200 + 一張「Access blocked」
   告示圖**，不是回錯誤碼。腳本會判定全部成功，1193 張下載完才發現每張都
   一模一樣。他們的 tile usage policy 明文禁止 bulk download。
   本腳本因此加了內容重複偵測（見 _check_variety）——任何來源都適用。
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
import urllib.request

# {z}/{y}/{x} 或 {z}/{x}/{y} 的順序各家不同，寫在 template 裡不要硬記
SOURCES = {
    # 國土測繪中心 通用電子地圖（注意是 z/y/x 順序，回 jpeg）
    "nlsc": ("https://wmts.nlsc.gov.tw/wmts/EMAP/default/GoogleMapsCompatible/{z}/{y}/{x}",
             "jpg", "© 內政部國土測繪中心"),
    # 國土測繪中心 正射影像（衛星圖）
    "nlsc-photo": ("https://wmts.nlsc.gov.tw/wmts/PHOTO2/default/GoogleMapsCompatible/{z}/{y}/{x}",
                   "jpg", "© 內政部國土測繪中心"),
}
UA = ("traffic-violation-detection/1.0 "
      "(Taiwan traffic monitoring; offline field deployment)")

ZOOM_MIN, ZOOM_MAX = 10, 18
# 🛑 範圍要用「圖磚張數」定，不能用公里數。
#    每張圖磚在低 zoom 涵蓋的實際距離大得多（z15 一張≈1.1km、z18 一張≈140m），
#    用固定公里半徑會讓低 zoom 只有中間幾張、視窗邊緣全是破洞。
#    實測:半徑 2km 在 z15 只有 ±1.8 張,1920x1080 視窗需要 ±4 張。
# ±5 張 = 11x11，足夠填滿 1920x1080（需 9x7）再留一圈平移餘裕。
TILE_RADIUS = 5
MAX_TILES = 3000          # 安全上限,超過就中止


def deg2tile(lat: float, lon: float, z: int) -> tuple:
    """經緯度 → tile 座標 (Web Mercator / slippy map)。"""
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def tile_span_m(lat: float, z: int) -> float:
    """一張圖磚在該緯度涵蓋的實際距離(公尺)。"""
    return 156543.03 * math.cos(math.radians(lat)) / (2 ** z) * 256


def plan(lat: float, lon: float, r: int) -> list:
    jobs = []
    for z in range(ZOOM_MIN, ZOOM_MAX + 1):
        cx, cy = deg2tile(lat, lon, z)
        n = 2 ** z
        for x in range(cx - r, cx + r + 1):
            for y in range(cy - r, cy + r + 1):
                if 0 <= x < n and 0 <= y < n:
                    jobs.append((z, x, y))
    return jobs


def fetch(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _check_variety(samples: list, label: str) -> bool:
    """抽樣的圖磚內容必須有差異。

    全部一樣代表對方回的是同一張告示/佔位圖（HTTP 200 也可能是被擋），
    不是真的地圖資料。這種失敗不會有錯誤碼，只能靠比對內容抓出來。
    """
    digests = {hashlib.md5(d).hexdigest() for d in samples if d}
    if len(digests) <= 1:
        print(f"❌ {label}: 抽樣 {len(samples)} 張內容完全相同 —— "
              f"對方回的是佔位圖或封鎖告示，不是地圖資料。已中止。", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--source", default="nlsc", choices=sorted(SOURCES))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "web", "tiles"))
    ap.add_argument("--delay", type=float, default=0.08, help="每張之間的延遲秒數")
    ap.add_argument("--tile-radius", type=int, default=TILE_RADIUS,
                    help="每個 zoom 以中心往外抓幾張(預設 5 = 11x11)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    url_tpl, ext, attribution = SOURCES[args.source]
    jobs = plan(args.lat, args.lon, args.tile_radius)
    per_zoom = {}
    for z, _, _ in jobs:
        per_zoom[z] = per_zoom.get(z, 0) + 1
    print(f"站台: {args.lat}, {args.lon}（每 zoom ±{args.tile_radius} 張）")
    print(f"來源: {args.source}  {attribution}")
    for z in range(ZOOM_MIN, ZOOM_MAX + 1):
        span = tile_span_m(args.lat, z) * args.tile_radius / 1000
        print(f"  z{z:<3} {per_zoom.get(z, 0):>4} 張 → 涵蓋中心往外 {span:>6.2f} km")
    print(f"  合計 {len(jobs)} 張")

    if len(jobs) > MAX_TILES:
        print(f"❌ 超過上限 {MAX_TILES} 張", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    # ── 先抽樣 4 張確認來源真的給資料，再開始整批下載 ──
    cx, cy = deg2tile(args.lat, args.lon, 17)
    probes = [(17, cx, cy), (17, cx + 1, cy), (17, cx, cy + 1), (17, cx + 2, cy + 2)]
    samples = []
    for z, x, y in probes:
        try:
            samples.append(fetch(url_tpl.format(z=z, x=x, y=y)))
        except Exception as e:
            print(f"❌ 抽樣失敗 z{z}/{x}/{y}: {e}", file=sys.stderr)
            return 1
    if not _check_variety(samples, "抽樣"):
        return 1
    print(f"✓ 抽樣 4 張內容各異，來源正常（{len(samples[0])} bytes / 張）\n")

    out_root = os.path.abspath(args.out)
    ok = skip = fail = 0
    digests = set()
    t0 = time.time()
    for i, (z, x, y) in enumerate(jobs, 1):
        path = os.path.join(out_root, str(z), str(x), f"{y}.{ext}")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            skip += 1
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data = fetch(url_tpl.format(z=z, x=x, y=y))
            if not data:
                raise ValueError("空回應")
            with open(path, "wb") as f:
                f.write(data)
            digests.add(hashlib.md5(data).hexdigest())
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  ✗ z{z}/{x}/{y}: {e}", flush=True)
        time.sleep(args.delay)
        if i % 100 == 0:
            print(f"  ... {i}/{len(jobs)} (新增 {ok} / 已有 {skip} / 失敗 {fail} / "
                  f"相異內容 {len(digests)})", flush=True)

    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fs in os.walk(out_root) for f in fs) / 1024 / 1024
    print(f"\n完成: 新增 {ok} / 已有 {skip} / 失敗 {fail}，"
          f"耗時 {time.time() - t0:.0f}s，總容量 {size_mb:.1f} MB")

    # 整批下載完再驗一次：相異內容太少一樣是被回佔位圖
    if ok and len(digests) < max(2, ok * 0.5):
        print(f"❌ 新下載 {ok} 張但只有 {len(digests)} 種內容 —— 疑似佔位圖，請檢查來源。",
              file=sys.stderr)
        return 1
    if ok:
        print(f"✓ 相異內容 {len(digests)}/{ok} 張，來源資料正常")
    print(f"※ 前端須標示圖資來源：{attribution}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
