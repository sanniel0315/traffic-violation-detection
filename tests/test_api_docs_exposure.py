#!/usr/bin/env python3
"""API 文件的曝光範圍:客戶只能看到對外那幾條,完整規格要登入。

回歸案例（2026-08-09 實測）：/docs、/redoc、/openapi.json 三者
**不需要任何憑證**就能從網路上開啟，會把 178 條路徑全部列出來 ——
其中 173 條是內部端點，包含 /api/auth/users、/api/auth/users/{id}/password、
/api/io/do/{ch}、/api/frigate/restart、/api/io_tcp/simulate_detection …
而對外客戶實際只需要 5 條 /api/v1/external/*。

分層：
  /docs /redoc /openapi.json                 → 要管理者登入（cookie）
  /api/v1/external/docs   + openapi.json     → 要 API 金鑰（?token= 或 header）
                                                且只回 /api/v1/external/* 這幾條
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")
os.environ.setdefault("EXTERNAL_API_KEY", "unit-test-key-abcdef")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

fails = []
KEY = "unit-test-key-abcdef"


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


client = TestClient(app)

print("完整文件:未登入一律擋")
for path in ("/docs", "/redoc", "/openapi.json"):
    check(f"{path} 未登入", client.get(path).status_code, 401)

print("\n對外文件:沒有金鑰擋、金鑰錯也擋")
check("/api/v1/external/docs 無 token", client.get("/api/v1/external/docs").status_code, 401)
check("/api/v1/external/docs token 錯",
      client.get("/api/v1/external/docs", params={"token": "wrong"}).status_code, 401)
check("/api/v1/external/openapi.json 無 token",
      client.get("/api/v1/external/openapi.json").status_code, 401)

print("\n對外文件:金鑰正確")
r = client.get("/api/v1/external/docs", params={"token": KEY})
check("?token= 可開 Swagger UI", r.status_code, 200)
r = client.get("/api/v1/external/openapi.json", headers={"X-API-Key": KEY})
check("X-API-Key header 也可以", r.status_code, 200)

spec = client.get("/api/v1/external/openapi.json", params={"token": KEY}).json()
paths = sorted(spec.get("paths", {}))
check("只含對外端點", [p for p in paths if not p.startswith("/api/v1/external")], [])
# 🛑 不要用關鍵字子字串比對 —— "io" 會命中 "congest-io-n",誤判成內部端點。
# 直接拿完整規格裡真正的內部路徑來比。
full = client.get("/openapi.json", headers={"X-API-Key": KEY})   # 無 cookie → 應該 401
check("完整規格不會因為有 API 金鑰就開放", full.status_code, 401)
for internal in ("/api/auth/users", "/api/auth/users/{user_id}/password",
                 "/api/frigate/restart", "/api/io/do/{ch}"):
    check(f"對外規格不含 {internal}", internal in paths, False)
check("對外端點都在", len(paths) >= 5, True)
check("標題是對外用的", spec["info"]["title"], "交通資料對外 API")

print("\n對外 API 本身的認證沒被文件改動影響")
check("vd-report/latest 無金鑰",
      client.get("/api/v1/external/vd-report/latest").status_code, 401)
check("vd-report/latest 帶金鑰",
      client.get("/api/v1/external/vd-report/latest",
                 headers={"X-API-Key": KEY}).status_code, 200)

print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
