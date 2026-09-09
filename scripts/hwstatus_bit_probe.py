#!/usr/bin/env python3
"""HardwareStatus 逐位元實測工具(中央端位元對照)。

用途:用 force 模式把**指定的值**送上中央,等中央那份 1 分鐘週期的
scmXMLData.xmL 更新,記下 eq_hw_status 顯示什麼。一次一個位元,
就能得到「我方 bitN → 中央告警名稱」的實證對照表,不必靠推論。

🛑 中央畫面上的「硬體狀態燈號」不在這份 XML 裡(它是另一條解析路徑),
   那一欄只能請現場的人看畫面回報,腳本量不到。

用法:
    python3 scripts/hwstatus_bit_probe.py 0x4000            # 測單一值
    python3 scripts/hwstatus_bit_probe.py --bits 14 13 9    # 測指定位元
    python3 scripts/hwstatus_bit_probe.py --all             # bit0~bit15 全掃

結束(含 Ctrl-C / 例外)一定會還原成常設模式,見 RESTORE。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sqlite3
import sys
import time
import urllib.request

DAEMON = "http://127.0.0.1:8012"
CENTER_XML = "http://10.105.6.73/api/smg/r24a/device/scmXMLData.xmL"
DB = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "violations.db")
LOG = pathlib.Path(__file__).resolve().parents[1] / "docs" / "reports" / "hwstatus_bit_probe.jsonl"
# 🛑 常設組態(2026-09-08 定案)。測完一定要回到這裡。
RESTORE = {"mode": "swap", "mask": 8192}


def _post(path: str) -> dict:
    req = urllib.request.Request(DAEMON + path, method="POST")
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode("utf-8"))


def set_mode(mode: str, value: int = 0, mask: int = 0) -> dict:
    return _post(f"/api/signal/control/hwstatus-mode?mode={mode}"
                 f"&value={value}&mask={mask}")


def read_center() -> tuple[str, str, str]:
    """回傳 (XML 時間, eq_hw_status, message)。讀不到就回空字串。"""
    try:
        with urllib.request.urlopen(CENTER_XML, timeout=8) as r:
            body = r.read().decode("utf-8", "ignore")
    except Exception as exc:                    # 中央端不通不該中斷整輪測試
        return ("", f"<讀取失敗:{exc}>", "")
    t = re.search(r'file_attribute[^>]*time="([^"]*)"', body)
    hw = re.search(r'<sig[^>]*eq_hw_status="([^"]*)"', body)
    msg = re.search(r'<sig[^>]*message="([^"]*)"', body)
    return (t.group(1) if t else "", hw.group(1) if hw else "",
            msg.group(1) if msg else "")


def wait_on_wire(expect: int, timeout: float = 90.0) -> tuple[bool, float]:
    """等到真的有一框 0F04/0FC1 帶著 expect 送上線路(sent_hw 是實際送出值)。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            conn = sqlite3.connect(DB, timeout=20)
            conn.execute("PRAGMA busy_timeout=20000")
            row = conn.execute(
                "SELECT ts, sent_hw FROM signal_frames "
                "WHERE code IN ('0F04','0FC1') AND src='controller' "
                "ORDER BY id DESC LIMIT 1").fetchone()
            conn.close()
            if row and row[1] is not None and int(row[1]) == expect \
                    and float(row[0]) > t0:
                return True, float(row[0])
        except Exception:
            pass
        time.sleep(3)
    return False, 0.0


def probe(value: int, ticks: int = 2) -> dict:
    """送出 value,等線路確認 + ticks 個 XML 週期,回傳這一輪的紀錄。"""
    label = f"0x{value:04X}"
    bits = [b for b in range(16) if value >> b & 1]
    print(f"\n=== {label}  (bit {', '.join(map(str, bits)) or '無'}) ===", flush=True)
    before_t, before_hw, _ = read_center()
    set_mode("force", value=value, mask=0)
    t_set = time.time()
    on_wire, t_wire = wait_on_wire(value)
    print(f"  上線路: {'✔ ' + time.strftime('%H:%M:%S', time.localtime(t_wire)) if on_wire else '✘ 90 秒內沒看到'}",
          flush=True)
    # 等 XML 換過 ticks 次(它是 1 分鐘週期,不等滿會讀到舊值)
    seen, last_t = 0, before_t
    deadline = time.time() + 60 * (ticks + 1) + 40
    xml_t, hw, msg = before_t, before_hw, ""
    while seen < ticks and time.time() < deadline:
        time.sleep(10)
        xml_t, hw, msg = read_center()
        if xml_t and xml_t != last_t:
            seen += 1
            last_t = xml_t
            print(f"  XML {xml_t}  eq_hw_status={hw!r}", flush=True)
    rec = {"probe_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "sent": label, "sent_int": value, "bits": bits,
           "on_wire": on_wire,
           "wire_ts": time.strftime("%H:%M:%S", time.localtime(t_wire)) if on_wire else None,
           "xml_time": xml_t, "eq_hw_status": hw, "message": msg,
           "before_eq_hw_status": before_hw,
           "xml_ticks_waited": seen}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → 結果 eq_hw_status={hw!r}(前一輪 {before_hw!r})", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("values", nargs="*", help="要送的值,例如 0x4000")
    ap.add_argument("--bits", nargs="*", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="bit0~bit15 逐位")
    ap.add_argument("--ticks", type=int, default=2, help="每個值等幾個 XML 週期")
    a = ap.parse_args()

    todo: list[int] = [int(v, 0) for v in a.values]
    if a.bits:
        todo += [1 << b for b in a.bits]
    if a.all:
        todo += [1 << b for b in range(16)]
    if not todo:
        ap.error("要給值、--bits 或 --all")

    out = []
    try:
        for v in todo:
            out.append(probe(v, a.ticks))
    finally:
        # 🛑 不管怎麼結束,一定回到常設組態 —— 測試值留在線路上是事故。
        r = set_mode(RESTORE["mode"], mask=RESTORE["mask"])
        print(f"\n已還原:mode={r.get('mode')} mask={r.get('mask_hex')}", flush=True)

    print("\n| 送出 | 位元 | 中央 eq_hw_status |")
    print("|---|---|---|")
    for r in out:
        print(f"| `{r['sent']}` | bit{','.join(map(str, r['bits'])) or '—'} "
              f"| {r['eq_hw_status'] or '(空)'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
