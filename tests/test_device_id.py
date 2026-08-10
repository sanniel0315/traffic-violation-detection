#!/usr/bin/env python3
"""終端控制器識別碼（設備編號）—— 規範 (C) 識別碼設定。

規範：每一個終端控制器須具有個別之通訊識別碼（設備編號）設定，
      指撥開關（DIP SWITCH）16 位元 **或以軟體控制**。

本系統採「以軟體控制」：
  `.env` 的 DEVICE_ID 優先（可設定）→ 未設定則由板載網卡 MAC 自動產生（開箱即唯一）

🛑 MAC 選取規則必須穩定，否則拔插網線就換編號：
  - 只取實體介面，排除 docker0 / l4tbr0 / tailscale0 / lo
  - 排除「本地管理位址」—— Jetson 的 usb0/usb1 是 USB gadget，
    MAC 每次開機隨機產生（實測 f2:12:58:...），拿來當編號會漂移
  - 取最小的 MAC，與哪張網卡有接線、接哪個孔無關

回歸案例：原本 `_DEVICE_ID = "jetson-nx-001"` 是寫死的常數，
第二台架起來會回一模一樣的編號 —— 正是規範要防的事。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.utils import device_id as mod  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}  got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


# ── 1. 本地管理位址判定（決定哪些 MAC 不可用） ────────────────────────
print("本地管理位址判定（這種 MAC 每次開機可能都不同，不可當編號）")
check("74:fe:48:.. 板載網卡 → 全域管理，可用",
      mod._is_locally_administered("74:fe:48:be:6b:2e"), False)
check("f2:12:58:.. USB gadget → 本地管理，不可用",
      mod._is_locally_administered("f2:12:58:d5:5f:41"), True)
check("52:49:ca:.. docker0 → 本地管理，不可用",
      mod._is_locally_administered("52:49:ca:ec:fb:51"), True)
check("3e:c7:e7:.. l4tbr0 → 本地管理，不可用",
      mod._is_locally_administered("3e:c7:e7:79:f3:87"), True)
check("格式壞掉 → 保守視為不可用", mod._is_locally_administered("xx"), True)

# ── 2. DEVICE_ID 設定優先於 MAC ──────────────────────────────────────
print("\n設定優先序（規範要求「可設定」）")
saved = os.environ.get("DEVICE_ID")
try:
    os.environ["DEVICE_ID"] = "VD-0301"
    check("有設定 → 用設定值", mod.get_device_id(), "VD-0301")
    check("來源標示為 configured", mod.device_id_info()["source"], "configured")

    os.environ["DEVICE_ID"] = "  VD-0302  "
    check("前後空白會去掉", mod.get_device_id(), "VD-0302")

    os.environ["DEVICE_ID"] = "12345"
    check("可填 16 位元數字（等同 DIP 開關）", mod.get_device_id(), "12345")

    os.environ["DEVICE_ID"] = ""
    check("設成空字串 → 退回 MAC 自動值", mod.get_device_id(), mod.default_device_id())
    check("來源標示為 mac", mod.device_id_info()["source"], "mac")

    os.environ.pop("DEVICE_ID", None)
    check("完全沒設 → 退回 MAC 自動值", mod.get_device_id(), mod.default_device_id())
finally:
    if saved is None:
        os.environ.pop("DEVICE_ID", None)
    else:
        os.environ["DEVICE_ID"] = saved

# ── 3. MAC 自動值的格式（16 位元） ───────────────────────────────────
print("\nMAC 自動產生值的格式")
auto = mod.default_device_id()
check("有前綴 TVD-", auto.startswith("TVD-"), True)
tail = auto.split("-", 1)[1]
check("末段是 4 個十六進位字元 = 16 位元", len(tail), 4)
check("末段可解析為 0~65535", 0 <= int(tail, 16) <= 0xFFFF, True)
check("多次呼叫結果一致（不會漂移）", mod.default_device_id(), auto)

# ── 4. 沒有任何合格網卡時要有安全預設 ────────────────────────────────
print("\n取不到 MAC 時的行為")
orig = mod._physical_macs
try:
    mod._physical_macs = lambda: []
    check("無合格網卡 → 回固定預設值而不是爆掉", mod.default_device_id(), "TVD-0000")
finally:
    mod._physical_macs = orig

# ── 5. 對外 API 不可再用寫死常數 ─────────────────────────────────────
print("\n對外 API 必須用動態編號")
ext = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "api", "routes", "external.py"), encoding="utf-8").read()
check("已移除寫死的 jetson-nx-001 常數", "_DEVICE_ID = " in ext, False)
check("改用 get_device_id()", ext.count("get_device_id()") >= 2, True)

print()
if fails:
    print(f"FAIL {len(fails)} 項: {fails}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
