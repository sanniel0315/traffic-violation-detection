# -*- coding: utf-8 -*-
"""動態號誌影子模式 —— 我方決策全速運轉，但**只記錄不下發**。

用途（bypass OPAC 的前置驗證）：
    OPAC 正在控制路口時，我方每 N 秒用同一份現場資料算出「換我會怎麼切」，
    與控制器**實際發生的動作**並排記錄。累積夠多之後用成效指標
    （總延滯／排隊／主線回堵次數）比較兩套控制的好壞，而不是比逐筆一致。

🛑 **絕對不下發。** 本模組只讀 congestion 的排隊與 signal_tc3 抄錄的燈態，
   算完寫進 DB 就結束。任何下發都要走 signal_tc3 的 control/prepare +
   control/send，那條路預設關閉（SIGNAL_TC3_CONTROL）。

🛑 為什麼要趁現在做：OPAC 停掉之後就沒有對照組了。將來要接管，
   必須拿得出「我方演算法在真實車流上表現如何」的依據。

實際動作怎麼判斷：我方 signal_tc3 抄錄的 5F03 帶 sub_phase_id，
分相編號一變就是發生了切換（SWITCH），沒變就是 KEEP。
這與 icagent 讀的是同一顆控制器的同一份訊框，所以是權威值。
"""
from __future__ import annotations

import os
import sqlite3 as _sqlite3
import threading
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.routes.auth import get_current_user
from api.routes.logs import add_log
from api.routes.push import push_alert

router = APIRouter(prefix="/api/signal/shadow", tags=["signal-shadow"])

# 取樣週期(秒)。OPAC 是 5 秒一次決策，對齊它才好比對。
SHADOW_INTERVAL_SEC = float(os.getenv("SIGNAL_SHADOW_INTERVAL_SEC", "5") or 5)
# 影子模式開關。預設關 —— 要明確開啟才跑（雖然它不下發，仍是背景負載）。
SHADOW_ENABLED = os.getenv("SIGNAL_SHADOW_ENABLED", "0") != "0"
# 分相 → 提供該分相排隊量測的相機 id（對照 ramp_timing_baseline.json 的
# phases[].constraint_camera：分相1=ID3、分相2=ID4）
PHASE_CAMERA = {
    1: int(os.getenv("SIGNAL_SHADOW_CAM_PHASE1", "3") or 3),
    2: int(os.getenv("SIGNAL_SHADOW_CAM_PHASE2", "4") or 4),
}

# 抄錄器所在的獨立服務(traffic-signal.service)。燈態只有它有。
SIGNAL_DAEMON_URL = os.getenv("SIGNAL_DAEMON_URL", "http://127.0.0.1:8012").rstrip("/")
# 自動回報週期(秒)。影子跑再久，沒人去撈 DB 就等於沒回報 —— 這是實際踩到的問題。
SHADOW_REPORT_SEC = float(os.getenv("SIGNAL_SHADOW_REPORT_SEC", "3600") or 3600)
# 一致率低於此值就推播(只在「有車」的樣本上算，夜間無車不觸發)。
SHADOW_ALERT_RATE = float(os.getenv("SIGNAL_SHADOW_ALERT_RATE", "0.75") or 0.75)
# 一小時內「有車樣本」少於這個數就不評分(車太少，比率沒有意義)。
SHADOW_MIN_ACTIVE = int(os.getenv("SIGNAL_SHADOW_MIN_ACTIVE", "60") or 60)
_DB_PATH = os.getenv("SIGNAL_SHADOW_DB",
                     os.getenv("SIGNAL_TC3_QDB", "data/signal_shadow.db"))
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_stats = {"started_at": None, "samples": 0, "last_error": "", "last_at": None}
_db_ready = False
_last_report = [0.0]   # 上次自動回報的時刻(list 才好在 _loop 內改;0=尚未從 DB 讀回)


def _db():
    global _db_ready
    conn = _sqlite3.connect(_DB_PATH, timeout=20)
    conn.execute("PRAGMA busy_timeout=20000")
    if not _db_ready:
        conn.execute("""CREATE TABLE IF NOT EXISTS signal_shadow_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, green_phase INTEGER, green_elapsed REAL,
            queue_m_1 REAL, queue_m_2 REAL,
            ours TEXT, actual TEXT, agree INTEGER,
            switch_gain REAL, keep_gain REAL, change_cost REAL,
            forced INTEGER, blocked INTEGER, reason TEXT,
            step_id INTEGER, clearance INTEGER, control_mode TEXT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_shadow_ts "
                     "ON signal_shadow_log(ts)")
        # 🛑 上次回報時刻必須落地。放行程內變數的話,每次部署重啟都把一小時的
        #    計時歸零 —— 2026-09-03 部署頻繁,14:04 與 14:56 兩次重啟就讓
        #    14 點那個小時整個沒有回報,使用者以為影子壞了。
        conn.execute("CREATE TABLE IF NOT EXISTS signal_shadow_meta ("
                     "k TEXT PRIMARY KEY, v TEXT)")
        # 既有 DB(9/2 起累積的那份)沒有這三欄,補上 —— 舊列留 NULL,
        # summarize 會把 control_mode IS NULL 的樣本當「前提不明」排除。
        have = {r[1] for r in conn.execute("PRAGMA table_info(signal_shadow_log)")}
        for col, typ in (("step_id", "INTEGER"), ("clearance", "INTEGER"),
                         ("control_mode", "TEXT")):
            if col not in have:
                conn.execute(f"ALTER TABLE signal_shadow_log ADD COLUMN {col} {typ}")
        conn.commit()
        _db_ready = True
    return conn


def _last_report_at() -> float:
    """讀回上次回報時刻(跨重啟保留)。讀不到就當「從未回報」。"""
    try:
        conn = _db()
        row = conn.execute("SELECT v FROM signal_shadow_meta WHERE k='last_report'"
                           ).fetchone()
        conn.close()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0


def _mark_reported(ts: float) -> None:
    try:
        conn = _db()
        conn.execute("INSERT INTO signal_shadow_meta(k,v) VALUES('last_report',?) "
                     "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (str(ts),))
        conn.commit()
        conn.close()
    except Exception:
        pass


def _queue_m(camera_id: int) -> Optional[float]:
    """取該相機當下的排隊公尺（我方壅塞偵測的量測值）。"""
    try:
        from api.routes.congestion import congestion_results
        r = congestion_results.get(camera_id) or {}
        v = r.get("estimated_queue_length_m")
        return float(v) if v is not None else None
    except Exception:
        return None


def _flow_vpm(camera_id: int) -> Optional[float]:
    """取該相機當下的到達流量(輛/分)。綠側價值要靠它,不能只看靜態排隊。"""
    try:
        from api.routes.congestion import congestion_results
        r = congestion_results.get(camera_id) or {}
        v = r.get("flow_vpm")
        return float(v) if v is not None else None
    except Exception:
        return None


def _live_phase() -> Optional[dict]:
    """取控制器當下的分相/步階（5F03）。

    🛑 **必須跟 signal_daemon 要，不可以讀 traffic-api 行程內的 _by_addr。**
       抄錄器是獨立服務 `traffic-signal.service`
       (`uvicorn services.signal_daemon:app --port 8012`)，
       它才是真正連著 MiiNePort :1001 在抄錄的那一個。traffic-api 這個行程
       的 _by_addr 永遠是空的 —— 影子模式第一版就是讀它，結果空轉沒資料。
       而且 services/signal_daemon.py 明寫「traffic-api 那邊絕不可再
       start_recorder，否則搶 :1001 / 雙抄錄」，所以也不能自己開一份。
    """
    try:
        import urllib.request as _u
        with _u.urlopen(f"{SIGNAL_DAEMON_URL}/api/signal/status", timeout=3) as r:
            import json as _j
            data = _j.load(r)
        for xn in (data.get("intersections") or []):
            ph = xn.get("phase") or {}
            if ph.get("sub_phase_id") is None:
                continue
            lights = ph.get("lights") or []
            # 清道判定不寫死步階編號 —— 直接看有沒有任何方向亮綠
            # (協定 bit2 圓頭綠 / bit3 左 / bit4 直 / bit5 右)。
            # 黃燈與全紅期間控制器已經committed,那時的 KEEP/SWITCH 判斷沒有意義。
            any_green = any((int(l.get("value") or 0) & 0x3C) for l in lights)
            cm = xn.get("control_mode") or {}
            return {"sub_phase_id": int(ph["sub_phase_id"]),
                    "step_id": ph.get("step_id"),
                    "clearance": not any_green,
                    "stale": bool(xn.get("stale")),
                    "age_sec": xn.get("age_sec"),
                    "control_mode": cm.get("code")}
    except Exception:
        pass
    return None


def _loop():
    """影子迴圈：取樣 → 算我方決策 → 與實際動作對照 → 落 DB。不下發。"""
    from detection.signal_decision_engine import ApproachState, decide
    from detection.signal_timing_lookup import (
        current_base_plan, plan_params, phase_role,
    )

    prev_phase: Optional[int] = None
    green_since: float = time.time()
    while not _stop.is_set():
        try:
            live = _live_phase()
            if live is None:
                _stop.wait(SHADOW_INTERVAL_SEC)
                continue
            # 🛑 抄錄斷線時 latest 會凍結在最後一幀,sub_phase_id 不再變,
            #    green_elapsed 會無限累加,樣本卻照記 —— 一致率被污染而且看不出來。
            #    (9/3 實測有 73 筆 green_elapsed>210s,若不排除無從分辨真假長綠。)
            if live.get("stale"):
                with _lock:
                    _stats["skipped_stale"] = _stats.get("skipped_stale", 0) + 1
                prev_phase = None       # 重連後不要拿斷線前的分相當基準
                _stop.wait(SHADOW_INTERVAL_SEC)
                continue
            cur_phase = live["sub_phase_id"]
            now = time.time()
            # 分相變了 = 控制器實際發生了切換
            actual = "KEEP"
            if prev_phase is not None and cur_phase != prev_phase:
                actual = "SWITCH"
                green_since = now
            elif prev_phase is None:
                green_since = now
            prev_phase = cur_phase
            green_elapsed = max(0.0, now - green_since)

            q1 = _queue_m(PHASE_CAMERA.get(1, 3))
            q2 = _queue_m(PHASE_CAMERA.get(2, 4))
            f1 = _flow_vpm(PHASE_CAMERA.get(1, 3))
            f2 = _flow_vpm(PHASE_CAMERA.get(2, 4))
            # 綠燈側 = 當下分相；紅燈側 = 另一相
            g_no, r_no = (cur_phase, 2 if cur_phase == 1 else 1)
            q_map = {1: q1, 2: q2}
            f_map = {1: f1, 2: f2}
            g_role = phase_role(g_no) or {}
            r_role = phase_role(r_no) or {}
            pp = plan_params(current_base_plan()) or {}
            mins = pp.get("min_green") or [15, 15]
            min_green = float(mins[g_no - 1] if len(mins) >= g_no else 15)

            d = decide(
                green_phase=g_no, green_elapsed_sec=green_elapsed,
                green_side=ApproachState(
                    g_no, queue_m=q_map.get(g_no),
                    flow_vpm=f_map.get(g_no),
                    storage_m=g_role.get("storage_m"),
                    priority=bool(g_role.get("priority"))),
                red_side=ApproachState(
                    r_no, queue_m=q_map.get(r_no),
                    storage_m=r_role.get("storage_m"),
                    waiting_sec=green_elapsed),
                min_green_sec=min_green,
                max_green_sec=float(pp.get("max_green") or 210),
            )

            conn = _db()
            conn.execute(
                "INSERT INTO signal_shadow_log(ts,green_phase,green_elapsed,"
                "queue_m_1,queue_m_2,ours,actual,agree,switch_gain,keep_gain,"
                "change_cost,forced,blocked,reason,step_id,clearance,control_mode)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), g_no,
                 round(green_elapsed, 1), q1, q2, d.action, actual,
                 # 🛑 切換剛發生的那一筆不列入一致率(agree=NULL)。
                 #    偵測到分相變了才記 actual=SWITCH,但那一刻 green_elapsed
                 #    已重設為 0,我方引擎因 min-green 未滿必然回 KEEP ——
                 #    這是取樣時序造成的假不一致,不是真的決策分歧。
                 #    (實測:18 筆裡 2 筆不一致全都是 green_elapsed=0 那筆)
                 # 🛑 三種樣本不列入一致率(agree=NULL),因為前提根本不成立:
                 #  (a) 切換剛發生那一筆:偵測到分相變了才記 actual=SWITCH,
                 #      但那一刻 green_elapsed 已重設為 0,我方因 min-green
                 #      未滿必然回 KEEP —— 取樣時序造成的假不一致。
                 #  (b) 清道期間(黃燈/全紅):控制器已經committed要換相,
                 #      這時候問「該不該切」沒有意義。
                 #  (c) 不是外部動態控制:定時/手動時 actual 不是 OPAC 的決策,
                 #      拿我方演算法去比一個根本沒在做決策的控制器毫無意義。
                 (None if (
                     (actual == "SWITCH" and green_elapsed < 1.0)
                     or live.get("clearance")
                     or live.get("control_mode") != "external_dynamic"
                 ) else (1 if d.action == actual else 0)),
                 d.switch_gain, d.keep_gain,
                 d.change_cost, 1 if d.forced_by_max_green else 0,
                 1 if d.blocked_by_priority else 0, d.reason,
                 live.get("step_id"), 1 if live.get("clearance") else 0,
                 live.get("control_mode")))
            conn.commit()
            conn.close()
            with _lock:
                _stats["samples"] += 1
                _stats["last_at"] = datetime.now().isoformat(timespec="seconds")
                _stats["last_error"] = ""
        except Exception as e:
            with _lock:
                _stats["last_error"] = str(e)
        # 到點就自己回報一次 —— 不必等人去撈 DB。
        # 上次時刻從 DB 讀回,重啟不會把計時歸零。
        if not _last_report[0]:
            _last_report[0] = _last_report_at() or time.time()
        if time.time() - _last_report[0] >= SHADOW_REPORT_SEC:
            _last_report[0] = time.time()
            _mark_reported(_last_report[0])
            try:
                _report()
            except Exception:
                pass
        _stop.wait(SHADOW_INTERVAL_SEC)


def summarize(minutes: int = 60) -> dict:
    """把最近 N 分鐘的影子結果壓成一份摘要。

    🛑 一致率一定要分「有車/無車」算。夜間兩側排隊都是 0，兩邊都 KEEP，
       一致率會漂到 98% —— 那個數字沒有資訊量，會蓋掉尖峰的真實表現。
       實測 13.5 小時:整體 87.4%，但只看有車樣本，08 時只有 54.7%。
    """
    since = time.time() - minutes * 60
    since_iso = datetime.fromtimestamp(since).isoformat(timespec="seconds")
    out = {"minutes": minutes, "samples": 0, "judged_samples": 0,
           "active_samples": 0,
           "agree_rate": None, "active_agree_rate": None,
           "excluded_clearance": 0, "excluded_not_opac": 0,
           "excluded_switch_instant": 0,
           "disagree_switch_early": 0, "disagree_switch_late": 0,
           "keep_gain_zero": 0, "forced": 0, "blocked": 0,
           "max_green_elapsed": None}
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT green_phase,green_elapsed,queue_m_1,queue_m_2,ours,actual,"
            "agree,keep_gain,forced,blocked,clearance,control_mode "
            "FROM signal_shadow_log WHERE ts>=?",
            (since_iso,)).fetchall()
        conn.close()
    except Exception as e:
        out["error"] = str(e)
        return out
    if not rows:
        return out
    out["samples"] = len(rows)
    # 🛑 一致率只能在「前提成立」的樣本上算。把排除掉的量攤開,
    #    否則看到一個裸數字沒人知道它是拿什麼比出來的。
    out["excluded_clearance"] = sum(1 for r in rows if r[10])
    out["excluded_not_opac"] = sum(
        1 for r in rows if r[11] is not None and r[11] != "external_dynamic")
    out["excluded_switch_instant"] = sum(
        1 for r in rows if r[6] is None and not r[10]
        and (r[11] == "external_dynamic" or r[11] is None))
    judged = [r for r in rows if r[6] is not None]
    out["judged_samples"] = len(judged)
    # 有車 = 任一側量到排隊。只有這些樣本的一致率才有意義。
    active = [r for r in judged if (r[2] or 0) > 0 or (r[3] or 0) > 0]
    out["active_samples"] = len(active)
    if judged:
        out["agree_rate"] = round(sum(r[6] for r in judged) / len(judged), 3)
    if active:
        out["active_agree_rate"] = round(sum(r[6] for r in active) / len(active), 3)
    out["disagree_switch_early"] = sum(
        1 for r in rows if r[4] == "SWITCH" and r[5] == "KEEP")
    out["disagree_switch_late"] = sum(
        1 for r in rows if r[4] == "KEEP" and r[5] == "SWITCH" and r[6] is not None)
    out["keep_gain_zero"] = sum(
        1 for r in rows if r[4] == "SWITCH" and r[5] == "KEEP" and not r[7])
    out["forced"] = sum(1 for r in rows if r[8])
    out["blocked"] = sum(1 for r in rows if r[9])
    out["max_green_elapsed"] = round(max((r[1] or 0) for r in rows), 1)
    return out


def _report() -> None:
    """把摘要寫進系統日誌;有車而一致率偏低就推播。"""
    s = summarize(int(SHADOW_REPORT_SEC // 60) or 60)
    if not s["samples"]:
        return
    ar = s["active_agree_rate"]
    pct = "—" if ar is None else f"{ar * 100:.1f}%"
    detail = (f"樣本 {s['samples']} → 可比對 {s['judged_samples']}"
              f"(排除:清道 {s['excluded_clearance']}、"
              f"非外部動態 {s['excluded_not_opac']}、"
              f"切換瞬間 {s['excluded_switch_instant']})、"
              f"其中有車 {s['active_samples']}、有車一致率 {pct}、"
              f"我方提早切 {s['disagree_switch_early']} 次"
              f"(其中綠側價值=0 佔 {s['keep_gain_zero']})、"
              f"最大綠強制 {s['forced']}、最長綠 {s['max_green_elapsed']}s")
    try:
        add_log("info", f"號誌影子決策回報 — {detail}", source="signal-shadow")
    except Exception:
        pass
    # 車太少不評分:比率會被少數樣本帶著跳。
    if ar is not None and s["active_samples"] >= SHADOW_MIN_ACTIVE and ar < SHADOW_ALERT_RATE:
        try:
            push_alert("號誌影子決策:與現行控制差異偏大",
                       f"有車時一致率僅 {pct} — {detail}",
                       {"active_agree_rate": ar,
                        "active_samples": s["active_samples"]},
                       category="signal")
        except Exception:
            pass


def start_shadow() -> bool:
    """啟動影子執行緒（冪等）。回傳是否真的啟動。"""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _stop.clear()
        _stats["started_at"] = datetime.now().isoformat(timespec="seconds")
        _thread = threading.Thread(target=_loop, name="signal-shadow", daemon=True)
        _thread.start()
        return True


def stop_shadow() -> None:
    _stop.set()


@router.get("", summary="影子模式狀態與最近決策")
async def shadow_status(limit: int = Query(50, ge=1, le=500),
                        _user=Depends(get_current_user)):
    rows = []
    try:
        conn = _db()
        cur = conn.execute(
            "SELECT ts,green_phase,green_elapsed,queue_m_1,queue_m_2,ours,actual,"
            "agree,switch_gain,keep_gain,forced,blocked,reason "
            "FROM signal_shadow_log ORDER BY id DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        with _lock:
            _stats["last_error"] = str(e)
    with _lock:
        st = dict(_stats)
    running = _thread is not None and _thread.is_alive()
    agree = [r["agree"] for r in rows if r.get("agree") is not None]
    return {
        "enabled": SHADOW_ENABLED,
        "running": running,
        "interval_sec": SHADOW_INTERVAL_SEC,
        "note": "影子模式只記錄不下發，路口仍由現行控制方(OPAC)控制",
        **st,
        "agree_rate": round(sum(agree) / len(agree), 3) if agree else None,
        "recent": rows,
    }


@router.get("/outcome", summary="成效比較(總延滯/排隊/回堵次數)")
async def shadow_outcome(minutes: int = Query(60, ge=1, le=1440),
                         _user=Depends(get_current_user)):
    """用 evaluate_outcome 算這段時間的成效指標。

    這是比較兩套控制的正確方式 —— 比結果好壞，不是比逐筆一致。
    """
    from detection.signal_decision_engine import evaluate_outcome
    from detection.signal_timing_lookup import phase_role
    samples = []
    try:
        conn = _db()
        cur = conn.execute(
            "SELECT queue_m_1,queue_m_2,actual FROM signal_shadow_log "
            "WHERE ts >= datetime('now','localtime',?) ORDER BY id",
            (f"-{int(minutes)} minutes",))
        st2 = (phase_role(2) or {}).get("storage_m")
        for q1, q2, actual in cur.fetchall():
            samples.append({"queue_m_1": q1 or 0, "queue_m_2": q2 or 0,
                            "storage_2": st2, "interval_sec": SHADOW_INTERVAL_SEC,
                            "switched": (actual == "SWITCH")})
        conn.close()
    except Exception as e:
        return {"error": str(e), "minutes": minutes}
    return {"minutes": minutes, "outcome": evaluate_outcome(samples)}


@router.post("/start", summary="啟動影子模式(只記錄不下發)")
async def shadow_start(_user=Depends(get_current_user)):
    started = start_shadow()
    return {"started": started, "running": True,
            "note": "影子模式不會對號誌控制器送出任何指令"}


@router.post("/stop", summary="停止影子模式")
async def shadow_stop(_user=Depends(get_current_user)):
    stop_shadow()
    return {"stopped": True}


@router.get("/summary", summary="影子結果摘要(有車/無車分開算一致率)")
async def shadow_summary(minutes: int = Query(60, ge=5, le=1440),
                         _user=Depends(get_current_user)):
    return summarize(minutes)
