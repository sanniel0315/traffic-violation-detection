#!/usr/bin/env python3
"""終端控制器識別碼（設備編號）。

規範要求：每一個終端控制器須具有個別之通訊識別碼（設備編號）設定，
指撥開關（DIP SWITCH）16 位元 **或以軟體控制**。

本系統採「以軟體控制」：
  1. `.env` 的 `DEVICE_ID` 為準 —— 可設定，等同 DIP 開關的角色
  2. 未設定時，由板載網卡 MAC 自動產生唯一預設值 —— 開箱即不重複
  3. 預設值取 MAC 末 2 bytes = **16 位元**，與規範的 16 位元 DIP 開關等價

🛑 MAC 的選取規則必須穩定，否則拔插網線就換編號：
  - 只看實體介面（有 /sys/class/net/<n>/device），排除 docker0 / l4tbr0 /
    tailscale0 / lo 等虛擬介面
  - 排除「本地管理位址」（第一個 byte 的 bit1 = 1）—— Jetson 的 usb0/usb1
    是 USB gadget，MAC 每次開機隨機產生（實測 f2:12:58:...），拿來當編號會漂移
  - 在合格者中取**最小的 MAC**，與哪張網卡有沒有接線、接哪個孔都無關

板子送修更換會導致 MAC 改變 → 屆時把原編號填進 `.env` 的 `DEVICE_ID` 即可，
交控中心的歷史資料不會斷。
"""
from __future__ import annotations

import os
from pathlib import Path

_SYS_NET = Path("/sys/class/net")
_PREFIX = "TVD"


def _is_locally_administered(mac: str) -> bool:
    """本地管理位址（自行產生、可能每次開機都不同）→ 不可當識別碼。"""
    try:
        return bool(int(mac.split(":")[0], 16) & 0b10)
    except (ValueError, IndexError):
        return True


def _physical_macs() -> list[str]:
    """板載實體網卡的永久 MAC，由小到大。"""
    macs = []
    try:
        names = sorted(p.name for p in _SYS_NET.iterdir())
    except OSError:
        return []
    for name in names:
        iface = _SYS_NET / name
        if not (iface / "device").exists():      # 虛擬介面（docker0 / lo / tailscale0…）
            continue
        try:
            mac = (iface / "address").read_text().strip().lower()
        except OSError:
            continue
        if not mac or mac == "00:00:00:00:00:00":
            continue
        if _is_locally_administered(mac):        # usb0 / usb1 這種每次開機都變的
            continue
        macs.append(mac)
    return sorted(macs)


def default_device_id() -> str:
    """未設定 DEVICE_ID 時的自動編號：MAC 末 2 bytes（16 位元）。"""
    macs = _physical_macs()
    if not macs:
        return f"{_PREFIX}-0000"
    tail = "".join(macs[0].split(":")[-2:]).upper()
    return f"{_PREFIX}-{tail}"


def get_device_id() -> str:
    """本機的設備編號。`.env` 的 DEVICE_ID 優先，未設定則由 MAC 產生。"""
    configured = str(os.getenv("DEVICE_ID", "") or "").strip()
    return configured or default_device_id()


def device_id_info() -> dict:
    """給網頁/API 顯示用：目前編號、來源、可選的 MAC。"""
    configured = str(os.getenv("DEVICE_ID", "") or "").strip()
    macs = _physical_macs()
    return {
        "device_id": configured or default_device_id(),
        "source": "configured" if configured else "mac",
        "source_label": "軟體設定（.env DEVICE_ID）" if configured else "板載網卡 MAC 自動產生",
        "mac_based_default": default_device_id(),
        "physical_macs": macs,
    }
