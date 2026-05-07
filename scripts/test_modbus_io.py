#!/usr/bin/env python3
"""
tM-PD3R3 (3 DI + 3 DO) RS-485 / Modbus RTU P1 通訊驗證腳本

使用情境：
  $ pip install --user pymodbus pyserial
  $ sudo systemctl stop nvgetty.service     # 釋放 ttyTHS0 console（若有用到）
  $ sudo chmod 666 /dev/ttyTHS1
  $ python3 scripts/test_modbus_io.py --diagnose
  $ python3 scripts/test_modbus_io.py --port /dev/ttyTHS1 --slave 1 --baudrate 9600

進階：
  --force-rs485-mode  Jetson UART 自動方向控制失靈時用 ioctl TIOCSRS485 強制切
  --port /dev/ttyUSB0 改用 USB-RS485 轉換器（Plan B）

設計：4 階段，任一段失敗印對應排查清單，避免直接跑後段白忙。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional


# ---------- 階段 0：環境診斷 ----------
def diagnose(port: str) -> bool:
    print("=" * 60)
    print("階段 0：環境診斷")
    print("=" * 60)

    ok = True

    # 0.1 裝置存在
    if not os.path.exists(port):
        print(f"❌ 找不到 {port}")
        print("   排查：")
        print("     ls -l /dev/tty* | grep -E 'THS|USB|ACM'")
        print("     dmesg | grep -i tty | tail -20")
        ok = False
    else:
        print(f"✅ 裝置存在：{port}")

    # 0.2 讀寫權限
    if os.path.exists(port) and not os.access(port, os.R_OK | os.W_OK):
        print(f"⚠️  目前使用者沒有讀寫權限")
        print(f"   臨時：sudo chmod 666 {port}")
        print(f"   永久：sudo usermod -a -G dialout $USER && 重新登入")
        ok = False
    elif os.path.exists(port):
        print(f"✅ 讀寫權限 OK")

    # 0.3 是否被 console 佔用
    try:
        import subprocess
        result = subprocess.run(
            ["systemctl", "is-active", "nvgetty.service"],
            capture_output=True, text=True, timeout=3,
        )
        nvgetty_state = result.stdout.strip()
        if nvgetty_state == "active":
            print(f"⚠️  nvgetty.service 還在跑（可能佔住 ttyTHS0）")
            print(f"   sudo systemctl stop nvgetty && sudo systemctl disable nvgetty")
        else:
            print(f"✅ nvgetty.service: {nvgetty_state}")
    except Exception as e:
        print(f"ℹ️  nvgetty 狀態查不到: {e}")

    # 0.4 lsof 看誰佔用
    try:
        import subprocess
        result = subprocess.run(
            ["lsof", port],
            capture_output=True, text=True, timeout=3,
        )
        if result.stdout.strip():
            print(f"⚠️  {port} 被以下程序佔用：")
            print(result.stdout)
            ok = False
        else:
            print(f"✅ {port} 沒被其他程序佔用")
    except FileNotFoundError:
        print(f"ℹ️  lsof 沒裝（apt install lsof 或忽略）")
    except Exception as e:
        print(f"ℹ️  lsof 查不到: {e}")

    print()
    return ok


# ---------- 強制 RS-485 模式 (Jetson 雷點) ----------
def force_rs485_mode(port: str) -> bool:
    """用 TIOCSRS485 ioctl 強制切換 RS-485 方向控制；某些 Jetson UART 驅動不會自動切"""
    try:
        import fcntl
        import struct

        SER_RS485_ENABLED = 0b00000001
        SER_RS485_RTS_ON_SEND = 0b00000010
        SER_RS485_RTS_AFTER_SEND = 0b00000100

        TIOCSRS485 = 0x542F
        flags = SER_RS485_ENABLED | SER_RS485_RTS_ON_SEND | SER_RS485_RTS_AFTER_SEND
        # struct serial_rs485: flags(u32), delay_rts_before(u32), delay_rts_after(u32), padding(5*u32)
        rs485_conf = struct.pack("IIIIIIII", flags, 0, 0, 0, 0, 0, 0, 0)
        with open(port, "rb+") as f:
            fcntl.ioctl(f.fileno(), TIOCSRS485, rs485_conf)
        print(f"✅ 已對 {port} 設定 TIOCSRS485（方向自動 by RTS）")
        return True
    except Exception as e:
        print(f"⚠️  TIOCSRS485 失敗（驅動可能不支援）：{e}")
        return False


# ---------- 連線 + 通訊驗證 ----------
def connect(port: str, baudrate: int, slave: int, timeout: float = 1.0):
    """建立 ModbusSerialClient 並連線"""
    from pymodbus.client import ModbusSerialClient

    client = ModbusSerialClient(
        port=port,
        baudrate=baudrate,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=timeout,
    )
    if not client.connect():
        print(f"❌ 連不上 {port}")
        return None
    print(f"✅ 已連接 {port} @ {baudrate}bps, slave={slave}")
    return client


def read_di(client, slave: int) -> Optional[List[bool]]:
    """讀 DI x3（Modbus FC=02 Read Discrete Input，位址 0x00000）"""
    try:
        rr = client.read_discrete_inputs(address=0, count=3, device_id=slave)
    except TypeError:
        # 舊版 pymodbus 用 unit= 參數
        rr = client.read_discrete_inputs(address=0, count=3, slave=slave)
    if rr.isError():
        print(f"❌ 讀 DI 失敗: {rr}")
        return None
    return list(rr.bits[:3])


def write_do(client, slave: int, do_index: int, on: bool) -> bool:
    """寫單一 DO（FC=05 Write Single Coil）"""
    try:
        rr = client.write_coil(address=do_index, value=on, device_id=slave)
    except TypeError:
        rr = client.write_coil(address=do_index, value=on, slave=slave)
    return not rr.isError()


# ---------- 階段 1：通訊驗證 ----------
def step1_comm_check(client, slave: int) -> bool:
    print("=" * 60)
    print("階段 1：通訊驗證（讀 DI）")
    print("=" * 60)

    succ = 0
    times = []
    for i in range(3):
        t0 = time.time()
        di = read_di(client, slave)
        elapsed_ms = (time.time() - t0) * 1000
        if di is not None:
            print(f"  讀取 #{i + 1}: DI={di}  ({elapsed_ms:.1f} ms)")
            succ += 1
            times.append(elapsed_ms)
        time.sleep(0.1)

    if succ == 0:
        print("\n❌ 完全讀不到 — 通訊掛了。排查順序：")
        print("  1) 硬體：D+/D- 有沒有接反？GND 有沒有共地？終端電阻 120Ω 兩端都裝？")
        print("  2) BIOS：AFE-R750 COM 模式真的切到 RS-485 嗎？")
        print("  3) tM-PD3R3：slave ID 對嗎？(出廠 1，按過 reset 鍵會跑)")
        print("            baudrate 對嗎？(預設 9600，可能被改過)")
        print("  4) 軟體：加 --force-rs485-mode 試 ioctl 強制方向")
        print("  5) Plan B：插 USB-RS485 轉換器，--port /dev/ttyUSB0")
        return False

    avg = sum(times) / len(times) if times else 0
    print(f"\n✅ 通訊正常（{succ}/3 成功，平均 {avg:.1f} ms）")
    return True


# ---------- 階段 2：DO 跑馬燈 ----------
def step2_do_chase(client, slave: int, cycles: int = 3) -> bool:
    print("\n" + "=" * 60)
    print(f"階段 2：DO 跑馬燈（{cycles} 圈）")
    print("=" * 60)
    print("⚡ 看面板：3 顆燈會輪流亮一下（DO0 → DO1 → DO2）")

    try:
        for cycle in range(cycles):
            print(f"\n  圈 {cycle + 1}/{cycles}")
            for i in range(3):
                # 全滅
                for j in range(3):
                    write_do(client, slave, j, False)
                # 點亮 i
                write_do(client, slave, i, True)
                print(f"    DO{i} = ON  (其他 OFF)")
                time.sleep(0.6)
        # 結束全滅
        for j in range(3):
            write_do(client, slave, j, False)
        print("\n✅ DO 跑馬燈完成（已全部關閉）")
        return True
    except Exception as e:
        print(f"❌ DO 寫入失敗: {e}")
        return False


# ---------- 階段 3：按鍵監聽 ----------
def step3_button_listener(client, slave: int, duration_sec: int = 30) -> None:
    print("\n" + "=" * 60)
    print(f"階段 3：按鍵監聽（{duration_sec} 秒，Ctrl+C 提早結束）")
    print("=" * 60)
    print("  按面板上按鍵，會印 edge event")
    print("  DI0=遠端下載  DI1=Reset  DI2=保留\n")

    labels = ["遠端下載", "Reset    ", "保留    "]
    last = read_di(client, slave) or [False, False, False]
    end_at = time.time() + duration_sec
    try:
        while time.time() < end_at:
            cur = read_di(client, slave)
            if cur is None:
                time.sleep(0.05)
                continue
            for i, (a, b) in enumerate(zip(last, cur)):
                if a != b:
                    arrow = "🔽 按下" if b else "🔼 放開"
                    print(f"  [{time.strftime('%H:%M:%S')}] DI{i} [{labels[i]}] {arrow}")
            last = cur
            time.sleep(0.05)  # 20 Hz
    except KeyboardInterrupt:
        print("\n  使用者中止")
    print("\n✅ 按鍵監聽結束")


# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="tM-PD3R3 RS-485 P1 測試")
    p.add_argument("--port", default="/dev/ttyTHS1", help="serial port (default /dev/ttyTHS1)")
    p.add_argument("--slave", type=int, default=1, help="Modbus slave ID (default 1)")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--diagnose", action="store_true", help="只做環境診斷不開連線")
    p.add_argument("--force-rs485-mode", action="store_true", help="ioctl 強制切 RS-485 模式")
    p.add_argument("--listen-sec", type=int, default=30, help="按鍵監聽秒數")
    p.add_argument("--skip-listen", action="store_true", help="略過按鍵監聽（自動化用）")
    args = p.parse_args()

    # 階段 0
    if not diagnose(args.port):
        if args.diagnose:
            sys.exit(1)
        print("⚠️  環境有問題，仍嘗試連線...")

    if args.diagnose:
        sys.exit(0)

    # 強制 RS-485 模式（可選）
    if args.force_rs485_mode:
        force_rs485_mode(args.port)

    # 連線
    client = connect(args.port, args.baudrate, args.slave)
    if not client:
        print("\n❌ 連線失敗，跳過後續階段")
        sys.exit(2)

    try:
        # 階段 1
        if not step1_comm_check(client, args.slave):
            sys.exit(3)

        # 階段 2
        if not step2_do_chase(client, args.slave):
            sys.exit(4)

        # 階段 3
        if not args.skip_listen:
            step3_button_listener(client, args.slave, duration_sec=args.listen_sec)

        print("\n" + "=" * 60)
        print("🎉 P1 全部通過 — 可以進 P2 寫正式 driver")
        print("=" * 60)
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
