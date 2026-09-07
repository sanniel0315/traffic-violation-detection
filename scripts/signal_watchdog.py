#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""動態號誌看門狗:每 N 分鐘確認演算法是否真的在控制,異常就推播。

為什麼要有這支:先前的「每 10 分鐘確認」是靠人(或 AI)在線上手動查,
人一離線就沒有人看。這支跑在 Jetson 上,由 systemd timer 驅動,
不依賴任何外部連線。

🛑 它**只觀察與通報,不會自己動控制**。看門狗自己去「修」路口是最危險的
   設計:故障當下最不該做的事就是再送一則命令。要不要恢復由人決定。

檢查項目(與驗收條文的「運作狀態/故障情形」同一組):
  1. 控制策略是否仍是時相控制(bit4)
  2. 動態控制總開關與降階級數
  3. 授權續約是否還在跑(斷了授權會在一分鐘內過期,路口回定時)
  4. 我方 5F1C 有沒有被控制器拒絕(送出成功 ≠ 被接受)
  5. 決策迴圈是否還在產生樣本
  6. 三個服務是否 active、號誌通道連線數

用法:
    python3 scripts/signal_watchdog.py            # 檢查一次,印出結果
    python3 scripts/signal_watchdog.py --quiet    # 正常時不輸出(給 timer 用)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from typing import Optional

def _find_root() -> str:
    """專案根目錄。🛑 不能只靠 __file__ 的上層 —— 從別的位置(例如 /tmp)執行時
    會算成 '/',接著所有 DB 路徑都錯,看門狗會誤報「讀不到資料庫」而狂推播。
    以「該目錄下有 data/」為準,依序試:環境變數 → 腳本上層 → 目前工作目錄。"""
    for cand in (os.getenv("SIGNAL_WATCHDOG_ROOT"),
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 os.getcwd()):
        if cand and os.path.isdir(os.path.join(cand, "data")):
            return cand
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _find_root()
DAEMON = os.getenv("SIGNAL_DAEMON_URL", "http://127.0.0.1:8012")
VIOL_DB = os.path.join(ROOT, "data", "violations.db")
SHADOW_DB = os.path.join(ROOT, "data", "signal_shadow.db")
WINDOW_SEC = float(os.getenv("SIGNAL_WATCHDOG_WINDOW", "900") or 900)
# 續約週期是 45 秒。🛑 門檻取 2 個週期多一點,不是剛好 45 ——
#    一次網路抖動或 GC 造成的延遲不該當成故障。
RENEW_STALE_SEC = float(os.getenv("SIGNAL_WATCHDOG_RENEW_STALE", "100") or 100)
SERVICES = ("traffic-api", "traffic-signal", "traffic-io")


def _get(path: str) -> dict:
    try:
        with urllib.request.urlopen(DAEMON + path, timeout=5) as r:
            return json.load(r)
    except Exception as exc:
        return {"_error": "%s: %s" % (type(exc).__name__, exc)}


def _ro(db: str) -> Optional[sqlite3.Connection]:
    try:
        return sqlite3.connect("file:%s?mode=ro" % db, uri=True, timeout=8)
    except Exception:
        return None


def check() -> dict:
    now = time.time()
    cut = now - WINDOW_SEC
    out: dict = {"ts": time.strftime("%H:%M:%S"), "problems": [], "info": {}}

    def bad(msg: str) -> None:
        out["problems"].append(msg)

    # ── 1/2 控制權 ──────────────────────────────────────────────
    safety = _get("/api/signal/safety")
    dyn = _get("/api/signal/control/dynamic")
    strategy = safety.get("strategy")
    out["info"]["strategy"] = safety.get("strategy_text") or safety.get("_error") or "?"
    out["info"]["dynamic"] = "%s %s" % (dyn.get("enabled"), dyn.get("level"))
    if "_error" in safety or "_error" in dyn:
        bad("問不到 signal daemon(%s)" % (safety.get("_error") or dyn.get("_error")))
    else:
        if not (isinstance(strategy, int) and strategy & 0x10):
            bad("控制策略不含時相控制:%s" % out["info"]["strategy"])
        if not dyn.get("enabled"):
            # 🛑 traffic-signal 一重啟就會回到關閉,那是刻意的 fail-safe。
            #    這裡只通報,不自動打開 —— 重啟的原因可能正是有人在處理故障。
            bad("動態控制總開關為關閉(traffic-signal 重啟過?需人工確認後開啟)")
        if dyn.get("level") not in (None, "", "L0"):
            bad("降階中:%s(%s)" % (dyn.get("level"), dyn.get("reason") or ""))

    # ── 3/4 下發與 ACK ─────────────────────────────────────────
    conn = _ro(VIOL_DB)
    if conn is None:
        bad("讀不到 %s" % VIOL_DB)
    else:
        sends = {}
        for code, user, n, last in conn.execute(
                "SELECT code,user,count(*),max(ts) FROM signal_frames "
                "WHERE src='self' AND ts>? GROUP BY code,user", (cut,)):
            sends["%s/%s" % (code, user)] = {"n": n, "last": last}
        out["info"]["sends"] = {k: v["n"] for k, v in sends.items()}

        renew_last = max([v["last"] for k, v in sends.items() if k.startswith("5F10")]
                         or [0])
        out["info"]["renew_age_sec"] = round(now - renew_last, 1) if renew_last else None
        if isinstance(strategy, int) and strategy & 0x10:
            # 只有在我方持有授權時,續約中斷才是問題
            if not renew_last:
                bad("持有時相控制但近 %.0f 分鐘沒有續約紀錄" % (WINDOW_SEC / 60))
            elif now - renew_last > RENEW_STALE_SEC:
                bad("授權續約已中斷 %.0f 秒(超過 %.0f 秒,授權會過期回定時)"
                    % (now - renew_last, RENEW_STALE_SEC))

        # 🛑 送出成功不等於被接受。5F1C 的 NAK 率實測 42%,只看送出會讓人
        #    以為在控制,實際上控制器一則都沒吃。
        nacks = []
        for ts, raw in conn.execute(
                "SELECT ts,raw FROM signal_frames WHERE code='0F81' AND ts>?", (cut,)):
            try:
                b = bytes.fromhex(str(raw).replace(" ", ""))
            except Exception:
                continue
            i = b.find(b"\x0f\x81")
            if i >= 0 and len(b) >= i + 6:
                nacks.append(("%02X%02X" % (b[i + 2], b[i + 3]), b[i + 4]))
        out["info"]["nacks"] = len(nacks)
        our_1c = [x for x in nacks if x[0] == "5F1C"]
        if our_1c:
            bad("近 %.0f 分鐘有 %d 則 5F1C 被拒(ErrorCode %s)——控制沒生效"
                % (WINDOW_SEC / 60, len(our_1c),
                   ",".join(sorted({str(e) for _, e in our_1c}))))
        conn.close()

    # ── 5 決策 ──────────────────────────────────────────────────
    sconn = _ro(SHADOW_DB)
    if sconn is None:
        bad("讀不到 %s" % SHADOW_DB)
    else:
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(cut))
        try:
            row = list(sconn.execute(
                "SELECT count(*), sum(ours='SWITCH'), "
                "sum(COALESCE(queue_m_1,0)>0 OR COALESCE(queue_m_2,0)>0) "
                "FROM signal_shadow_log WHERE ts>?", (iso,)))[0]
            out["info"]["samples"] = row[0]
            out["info"]["switch"] = row[1] or 0
            out["info"]["with_veh"] = row[2] or 0
            if not row[0]:
                bad("決策迴圈近 %.0f 分鐘沒有產生任何樣本" % (WINDOW_SEC / 60))
        except Exception as exc:
            bad("讀決策紀錄失敗:%s" % exc)
        try:
            degr = list(sconn.execute(
                "SELECT ts,level,reason FROM signal_degrade_log WHERE epoch>? "
                "ORDER BY epoch DESC LIMIT 5", (cut,)))
            out["info"]["degrade_events"] = len(degr)
            for ts, lv, reason in degr:
                if lv != "L0":
                    bad("降階事件 %s %s:%s" % (ts, lv, (reason or "")[:80]))
        except Exception:
            pass                      # 表還沒建立(從未降階)不算問題
        sconn.close()

    # ── 6 服務與連線 ───────────────────────────────────────────
    for unit in SERVICES:
        try:
            st = subprocess.run(["systemctl", "is-active", unit],
                                capture_output=True, text=True, timeout=8).stdout.strip()
        except Exception as exc:
            st = "unknown(%s)" % exc
        out["info"][unit] = st
        if st != "active":
            bad("%s 不是 active(%s)" % (unit, st))
    try:
        ss = subprocess.run(["ss", "-tn", "state", "established"],
                            capture_output=True, text=True, timeout=8).stdout
        n1001 = sum(1 for l in ss.splitlines() if ":1001" in l)
    except Exception:
        n1001 = -1
    out["info"]["conn_1001"] = n1001
    if n1001 != 2:
        # 上游控制器 + 中央中繼,正常是 2 條
        bad("號誌通道連線數 %s(正常 2:上游控制器 + 中央中繼)" % n1001)

    out["ok"] = not out["problems"]
    return out


def one_line(res: dict) -> str:
    i = res["info"]
    return ("%s %s · %s · 動態 %s · 下發 %s · NAK %s · 決策 %s(SWITCH %s/有車 %s)"
            % (res["ts"], "正常" if res["ok"] else "異常",
               i.get("strategy"), i.get("dynamic"),
               sum((i.get("sends") or {}).values()) or 0,
               i.get("nacks"), i.get("samples"), i.get("switch"), i.get("with_veh")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="正常時不輸出(給 timer 用)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    res = check()
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    elif not (args.quiet and res["ok"]):
        print(one_line(res))
        for p in res["problems"]:
            print("  ⚠ " + p)

    # 🛑 異常才推播。正常也推會讓人麻痺,真的出事時反而被忽略。
    if not res["ok"]:
        try:
            sys.path.insert(0, ROOT)
            os.environ.setdefault("AUTH_SECRET", "watchdog")
            from api.routes.push import push_alert
            push_alert("動態號誌異常",
                       one_line(res) + "\n" + "\n".join(res["problems"]),
                       level="critical")
        except Exception as exc:
            print("  (推播失敗:%s)" % exc, file=sys.stderr)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
