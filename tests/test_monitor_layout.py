#!/usr/bin/env python3
"""即時監控方塊排版 API:順序正規化與權限。

排版全系統共用一份,讀取開放所有登入者、修改僅限管理員。
order 只存 camera id 順序,必須去重且濾掉非法值,否則前端排序會錯亂。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.routes import system as sysmod  # noqa: E402


def _normalize(order):
    """複製端點內的 order 正規化邏輯。"""
    seen = set()
    clean = []
    for v in order:
        try:
            cid = int(v)
        except (TypeError, ValueError):
            continue
        if cid in seen:
            continue
        seen.add(cid)
        clean.append(cid)
    return clean


def test_order_dedup():
    """同一台攝影機出現兩次只保留第一次 —— 否則排序會有重複位置。"""
    assert _normalize([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_order_drops_invalid():
    assert _normalize([1, "x", None, 2, "", 3.9]) == [1, 2, 3]


def test_order_accepts_numeric_strings():
    assert _normalize(["2", "10", 3]) == [2, 10, 3]


def test_order_preserves_sequence():
    """順序就是畫面排列，不可被排序打亂。"""
    assert _normalize([6, 2, 4, 3]) == [6, 2, 4, 3]


def test_default_layout_is_locked():
    """預設必須是鎖定,避免任何人一進來就能拖亂。"""
    d = sysmod._default_monitor_layout()
    assert d["locked"] is True
    assert d["order"] == []


def test_endpoints_registered():
    paths = [r.path for r in sysmod.router.routes if "monitor-layout" in r.path]
    assert len(paths) == 2, f"應有 GET 與 PUT 兩個端點，實際 {paths}"


def test_put_requires_admin():
    """PUT 必須掛 get_admin_user;GET 只需登入。"""
    import inspect
    put_route = [r for r in sysmod.router.routes
                 if "monitor-layout" in r.path and "PUT" in getattr(r, "methods", set())]
    assert put_route, "找不到 PUT 端點"
    sig = inspect.signature(put_route[0].endpoint)
    deps = [str(p.default) for p in sig.parameters.values()]
    assert any("get_admin_user" in d for d in deps), \
        f"PUT 端點必須要求管理員，實際依賴 {deps}"

    get_route = [r for r in sysmod.router.routes
                 if "monitor-layout" in r.path and "GET" in getattr(r, "methods", set())]
    sig2 = inspect.signature(get_route[0].endpoint)
    deps2 = [str(p.default) for p in sig2.parameters.values()]
    assert any("get_current_user" in d for d in deps2), \
        f"GET 端點應要求登入，實際依賴 {deps2}"


def test_read_layout_survives_missing_file():
    """設定檔不存在時要回預設值,不能拋例外讓監控頁掛掉。"""
    orig = sysmod.MONITOR_LAYOUT_PATH
    try:
        sysmod.MONITOR_LAYOUT_PATH = "/nonexistent/nope/monitor_layout.json"
        d = sysmod._read_monitor_layout()
        assert d["locked"] is True and d["order"] == []
    finally:
        sysmod.MONITOR_LAYOUT_PATH = orig


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{'全部通過' if not failed else str(failed) + ' 項失敗'}")
    sys.exit(1 if failed else 0)
