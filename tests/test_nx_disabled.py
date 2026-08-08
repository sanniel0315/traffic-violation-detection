#!/usr/bin/env python3
"""NX 未啟用時,端點必須立刻回應,不可以對連不到的主機做連線嘗試。

回歸案例（2026-08-09 實測）：production 網路連不到 NX 伺服器
（10.26.4.123:7001），/api/nx/devices 要 **60 秒**才回 502 ——
多種認證策略各吃一次 12 秒連線逾時累加起來。
而前端 NVR 頁一載入就呼叫它（index.html 兩處），使用者開那頁整整卡 60 秒。

規則：
  enabled=false 或沒設任何伺服器位址 → 視為未啟用
  /devices          回 200 + 空清單 + enabled:false（刻意關掉的功能不該報錯）
  其他媒體類端點    回 503 快速失敗
🛑 新增 NX 設定欄位要同時改 _default_nx_settings 與 _normalize_nx_settings ——
   後者是「用預設值重建」,沒列到的欄位存了會直接消失。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from fastapi import HTTPException  # noqa: E402

from api.routes.nx import _nx_enabled, _require_nx, nx_devices  # noqa: E402
from api.routes.system import _normalize_nx_settings  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


ON = {"enabled": True, "server_base_url": "https://10.26.4.123:7001", "proxy_base_url": ""}
OFF = {"enabled": False, "server_base_url": "https://10.26.4.123:7001", "proxy_base_url": ""}
NOURL = {"enabled": True, "server_base_url": "", "proxy_base_url": ""}
PROXY = {"enabled": True, "server_base_url": "", "proxy_base_url": "http://127.0.0.1:9"}

print("啟用判定")
check("有位址且 enabled=true → 啟用", _nx_enabled(ON), True)
check("enabled=false → 未啟用", _nx_enabled(OFF), False)
check("沒有任何位址 → 未啟用", _nx_enabled(NOURL), False)
check("只有 proxy 位址 → 啟用", _nx_enabled(PROXY), True)
check("沒有 enabled 欄位時預設啟用（向後相容）",
      _nx_enabled({"server_base_url": "https://x:7001"}), True)

print("\n_require_nx 守衛")
try:
    _require_nx(OFF)
    check("未啟用要丟 503", "沒有丟例外", "HTTPException 503")
except HTTPException as e:
    check("未啟用要丟 503", e.status_code, 503)
try:
    _require_nx(ON)
    check("啟用時不擋", True, True)
except HTTPException as e:
    check("啟用時不擋", f"被擋 {e.status_code}", True)

print("\n設定欄位不可被正規化吃掉")
check("enabled=false 存得住", _normalize_nx_settings(OFF)["enabled"], False)
check("enabled=true 存得住", _normalize_nx_settings(ON)["enabled"], True)
check("舊設定檔沒有 enabled → 預設 true",
      _normalize_nx_settings({"server_base_url": "https://x:7001"})["enabled"], True)
check("其他欄位沒被影響",
      _normalize_nx_settings(OFF)["server_base_url"], "https://10.26.4.123:7001")

print("\n/devices 未啟用時的回應")
import api.routes.nx as nxmod  # noqa: E402

_orig = nxmod._nx_settings
nxmod._nx_settings = lambda: dict(OFF)
try:
    resp = nx_devices()
    check("回 200（不丟例外）", isinstance(resp, dict), True)
    check("devices 是空清單", resp.get("devices"), [])
    check("標明 enabled=false", resp.get("enabled"), False)
    check("ok=true，前端不會跳錯誤 toast", resp.get("ok"), True)
finally:
    nxmod._nx_settings = _orig

print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
