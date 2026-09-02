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
_DB_PATH = os.getenv("SIGNAL_SHADOW_DB",
                     os.getenv("SIGNAL_TC3_QDB", "data/signal_shadow.db"))
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_stats = {"started_at": None, "samples": 0, "last_error": "", "last_at": None}
_db_ready = False


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
            forced INTEGER, blocked INTEGER, reason TEXT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_shadow_ts "
                     "ON signal_shadow_log(ts)")
        conn.commit()
        _db_ready = True
    return conn


def _queue_m(camera_id: int) -> Optional[float]:
    """取該相機當下的排隊公尺（我方壅塞偵測的量測值）。"""
    try:
        from api.routes.congestion import congestion_results
        r = congestion_results.get(camera_id) or {}
        v = r.get("estimated_queue_length_m")
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
            if ph.get("sub_phase_id") is not None:
                return {"sub_phase_id": int(ph["sub_phase_id"]),
                        "ts": xn.get("age_sec")}
        latest = data.get("latest") or {}
        ph = latest.get("phase") or {}
        if ph.get("sub_phase_id") is not None:
            return {"sub_phase_id": int(ph["sub_phase_id"]), "ts": latest.get("ts")}
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
            # 綠燈側 = 當下分相；紅燈側 = 另一相
            g_no, r_no = (cur_phase, 2 if cur_phase == 1 else 1)
            q_map = {1: q1, 2: q2}
            g_role = phase_role(g_no) or {}
            r_role = phase_role(r_no) or {}
            pp = plan_params(current_base_plan()) or {}
            mins = pp.get("min_green") or [15, 15]
            min_green = float(mins[g_no - 1] if len(mins) >= g_no else 15)

            d = decide(
                green_phase=g_no, green_elapsed_sec=green_elapsed,
                green_side=ApproachState(
                    g_no, queue_m=q_map.get(g_no),
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
                "change_cost,forced,blocked,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), g_no,
                 round(green_elapsed, 1), q1, q2, d.action, actual,
                 1 if d.action == actual else 0, d.switch_gain, d.keep_gain,
                 d.change_cost, 1 if d.forced_by_max_green else 0,
                 1 if d.blocked_by_priority else 0, d.reason))
            conn.commit()
            conn.close()
            with _lock:
                _stats["samples"] += 1
                _stats["last_at"] = datetime.now().isoformat(timespec="seconds")
                _stats["last_error"] = ""
        except Exception as e:
            with _lock:
                _stats["last_error"] = str(e)
        _stop.wait(SHADOW_INTERVAL_SEC)


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
