# -*- coding: utf-8 -*-
"""號誌整合驗收清單 —— 每一項都**實際去量**,不是宣稱。

為什麼要有這個檔:功能一路加上去,散在五個頁面與兩個服務裡,沒有任何地方
能回答「哪些做完了、怎麼驗、現在的實測結果是什麼」。要準備實際上線就必須
有一份可以逐條對照的驗收表,而且狀態要由程式去打端點量出來,不能靠人宣稱。

🛑 三條原則:
   1. 量不到就回 unknown,**不要當成 pass**。缺資料和通過是兩件事。
   2. 每一項都要留 evidence(實際數值),讓人可以自己複核。
   3. 安全項(下發封鎖)不通過就是不通過,不因為「還沒要用」而略過。

🛑 prefix 刻意不放在 /api/signal 底下 —— 那個路徑有 {sub_path:path} 萬用
   代理會把它整個吃掉轉去 daemon(2026-09-03 影子端點就是這樣全部 404)。
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends

from api.routes.auth import get_current_user

router = APIRouter(prefix="/api/acceptance", tags=["acceptance"])

SIGNAL_DAEMON_URL = os.getenv("SIGNAL_DAEMON_URL", "http://127.0.0.1:8012").rstrip("/")

PASS, FAIL, WARN, UNKNOWN = "pass", "fail", "warn", "unknown"


def _get(path: str, timeout: float = 4.0) -> Optional[dict]:
    """打 daemon 的端點。連不上回 None(→ unknown),不要吞成空 dict 當 pass。"""
    try:
        import json as _j
        import urllib.request as _u
        with _u.urlopen(f"{SIGNAL_DAEMON_URL}{path}", timeout=timeout) as r:
            return _j.load(r)
    except Exception:
        return None


def _item(key: str, group: str, title: str, criteria: str,
          state: str, evidence: str, detail: str = "") -> dict:
    return {"key": key, "group": group, "title": title, "criteria": criteria,
            "state": state, "evidence": evidence, "detail": detail}


def _check_recorder(st: Optional[dict]) -> list:
    out = []
    if st is None:
        return [_item("recorder", "資料源", "控制器抄錄連線",
                      "connected=true 且資料未過期(stale=false)",
                      UNKNOWN, "號誌 daemon 無回應 —— 量不到就不算通過")]
    connected = bool(st.get("connected"))
    stale = bool(st.get("stale"))
    ok = connected and not stale
    out.append(_item(
        "recorder", "資料源", "控制器抄錄連線",
        "connected=true 且 stale=false",
        PASS if ok else FAIL,
        f"connected={connected} stale={stale} age={st.get('age_sec')}s "
        f"frames={st.get('frames_total')}"))

    dropped = st.get("frames_dropped")
    out.append(_item(
        "frames_dropped", "資料源", "訊框持久化無丟棄",
        "frames_dropped = 0",
        UNKNOWN if dropped is None else (PASS if dropped == 0 else WARN),
        f"frames_dropped={dropped}",
        "佇列滿會靜默丟棄,丟了會讓事後的涵蓋率分析低估"))

    cks = st.get("cks_bad")
    total = st.get("frames_total") or 0
    if total:
        rate = (cks or 0) / total
        out.append(_item(
            "cks", "資料源", "訊框完整性",
            "CKS 錯誤率 < 0.1%",
            PASS if rate < 0.001 else WARN,
            f"cks_bad={cks} / {total} = {rate*100:.3f}%"))
    return out


def _check_phase(st: Optional[dict]) -> list:
    out = []
    xs = (st or {}).get("intersections") or []
    if not xs:
        return [_item("phase", "即時監看", "燈態解析(5F03)",
                      "解得出分相/步階/逐方向燈色",
                      UNKNOWN, "沒有路口資料")]
    x = xs[0]
    ph = x.get("phase") or {}
    lights = ph.get("lights") or []
    ok = ph.get("sub_phase_id") is not None and len(lights) > 0
    out.append(_item(
        "phase", "即時監看", "燈態解析(5F03)",
        "解得出分相/步階/逐方向燈色",
        PASS if ok else FAIL,
        f"分相={ph.get('sub_phase_id')} 步階={ph.get('step_id')} "
        f"方向數={len(lights)}"))

    el = x.get("phase_elapsed_sec")
    out.append(_item(
        "phase_elapsed", "即時監看", "已亮秒數精確度",
        "由抄錄器逐框追蹤(精確到訊框),非外部輪詢推算",
        PASS if el is not None else FAIL,
        f"phase_elapsed_sec={el}",
        "min/max green 安全閘門拿這個數字去比,誤差會直接變成違反最小綠的風險"))

    cm = (x.get("control_mode") or {}).get("code")
    out.append(_item(
        "control_mode", "即時監看", "控制模式判讀",
        "解得出誰在控制(不可只看 roadSideManual)",
        PASS if cm else FAIL,
        f"control_mode={cm or '尚未收到 5FC0'}"))
    return out


def _check_config() -> list:
    d = _get("/api/signal/config")
    if d is None:
        return [_item("config", "號誌設定", "設定總覽可讀",
                      "11 類設定各能取得最新回報",
                      UNKNOWN, "端點無回應")]
    secs = d.get("sections") or []
    got = [s for s in secs if s.get("received")]
    missing = [s["title"] for s in secs if not s.get("received")]
    return [_item(
        "config", "號誌設定", "設定總覽可讀",
        f"{len(secs)} 類設定各能取得最新回報",
        PASS if len(got) == len(secs) else WARN,
        f"已收到 {len(got)}/{len(secs)}"
        + (f";未收到:{'、'.join(missing)}" if missing else ""),
        "未收到者可用該類別的查詢碼主動取回")]


def _check_device() -> list:
    d = _get("/api/signal/device-status")
    if d is None or not d.get("available"):
        return [_item("device", "即時監看", "設備硬體狀態解讀",
                      "HardwareStatus 逐位元解出並分組",
                      UNKNOWN, (d or {}).get("reason", "端點無回應"))]
    return [
        _item("device", "即時監看", "設備硬體狀態解讀",
              "HardwareStatus 逐位元解出並分組",
              PASS, f"值={d.get('value_hex')} 已實證異常={d.get('fault_count')}"),
        _item("device_polarity", "即時監看", "硬體位元極性實證",
              "16 個位元的極性都經現場實證",
              WARN if d.get("pending_count") else PASS,
              f"待確認 {d.get('pending_count')} / 16",
              "未實證的位元一律不做正常異常判定,避免假警報;"
              "要轉正需走硬體狀態碼測試逐 bit 對照中央"),
    ]


def _check_shadow() -> list:
    """影子引擎與一致率 —— L5 接管的前置門檻。"""
    from api.routes import signal_shadow as ss
    out = []
    running = ss._thread is not None and ss._thread.is_alive()
    out.append(_item(
        "shadow_running", "決策引擎", "影子引擎運轉",
        "執行緒存活且持續取樣",
        PASS if running else FAIL,
        f"running={running} 本次啟動後樣本={ss._stats.get('samples')}"))

    s1 = ss.summarize(60)
    out.append(_item(
        "shadow_sampling", "決策引擎", "近一小時有樣本",
        "最近 60 分鐘取得樣本",
        PASS if s1.get("samples") else FAIL,
        f"樣本={s1.get('samples')} 可比對={s1.get('judged_samples')} "
        f"有車={s1.get('active_samples')}"))

    # L5 門檻:有車一致率連續達標
    rate = s1.get("active_agree_rate")
    act = s1.get("active_samples") or 0
    if rate is None or act < 30:
        st, ev = UNKNOWN, f"有車樣本僅 {act} 筆,不足以判定(需 ≥30)"
    elif rate >= 0.95:
        st, ev = PASS, f"有車一致率 {rate*100:.1f}%(門檻 95%)"
    else:
        st, ev = WARN, f"有車一致率 {rate*100:.1f}%,未達 95% 門檻"
    out.append(_item(
        "agree_rate", "決策引擎", "安全性:與現行控制一致率",
        "有車樣本一致率 ≥ 95%(接管前置的**安全門檻**)",
        st, ev,
        "🛑 這是安全門檻不是優越門檻:一致率高只代表我方接手不會亂來,"
        "100% 一致等於一點也沒有比較好。要證明「優於現今」必須做 A/B 交替時段,"
        "影子資料無法證明(反事實量不到)—— 見 docs/上線報告_骨架.md"))

    # 成效基準:現行控制下的實際表現。這是日後 A/B 比較的基準線,
    # 本身不能拿來宣稱我方比較好。
    try:
        from api.routes.signal_shadow import _outcome_window
        since = (datetime.now() - timedelta(hours=6)).isoformat(timespec="seconds")
        until = datetime.now().isoformat(timespec="seconds")
        ow = _outcome_window(since, until)
        has = bool(ow.get("samples"))
        out.append(_item(
            "outcome_baseline", "決策引擎", "成效基準可量測",
            "能算出現行控制下的總延滯/排隊/回堵/切換頻率",
            PASS if has else FAIL,
            (f"近6h 總延滯={ow.get('total_delay_veh_sec')} 車·秒、"
             f"回堵事件={ow.get('spillback_events_2')}、"
             f"切換={ow.get('switch_per_min')} 次/分") if has else "無樣本",
            "這是基準線,不是我方演算法的成效 —— 我方沒真的控制過,反事實量不到"))
    except Exception as e:
        out.append(_item("outcome_baseline", "決策引擎", "成效基準可量測",
                         "能算出現行控制下的總延滯/排隊/回堵/切換頻率",
                         UNKNOWN, f"計算失敗:{e}"))

    st2 = None
    try:
        st2 = ss.summarize(360)
    except Exception:
        pass
    if st2:
        out.append(_item(
            "shadow_premise", "決策引擎", "比對前提可稽核",
            "清道/非外部動態/切換瞬間都排除且數量攤開",
            PASS,
            f"近6h 排除:清道{st2.get('excluded_clearance')}、"
            f"非外部動態{st2.get('excluded_not_opac')}、"
            f"切換瞬間{st2.get('excluded_switch_instant')}"))
    return out


def _check_stats() -> list:
    from api.routes import signal_shadow as ss
    try:
        import asyncio
        d = asyncio.get_event_loop()
    except Exception:
        d = None
    # summarize 是同步的,stats 端點是 async;直接呼叫內部重建函式較穩
    since = (datetime.now() - timedelta(hours=6)).isoformat(timespec="seconds")
    until = datetime.now().isoformat(timespec="seconds")
    try:
        conn = ss._db()
        rows = conn.execute(
            "SELECT ts,green_phase,green_elapsed,forced,queue_m_1,queue_m_2 "
            "FROM signal_shadow_log WHERE ts>=? AND ts<=? "
            "AND control_mode='external_dynamic' ORDER BY ts",
            (since, until)).fetchall()
        conn.close()
    except Exception as e:
        return [_item("stats", "歷史與統計", "運作統計可重建",
                      "能從燈態重建每一次綠燈的長度",
                      UNKNOWN, f"讀取失敗:{e}")]
    runs = ss._green_runs([(r[0], r[1], r[2], r[3], r[4], r[5], None, None)
                           for r in rows])
    used = [r for r in runs[1:] if r["green_sec"] > 0]
    return [_item(
        "stats", "歷史與統計", "運作統計可重建",
        "能從燈態重建每一次綠燈的長度",
        PASS if used else FAIL,
        f"近6h 取樣={len(rows)} 綠燈段={len(used)}")]


def _check_safety() -> list:
    d = _get("/api/signal/safety")
    out = []
    if d is None:
        out.append(_item("safety", "安全", "安全網監看",
                         "策略變更/現場操作/故障步階能記事件",
                         UNKNOWN, "端點無回應"))
    else:
        evs = d.get("events") or []
        out.append(_item(
            "safety", "安全", "安全網監看",
            "策略變更/現場操作/故障步階能記事件",
            PASS, f"事件 {len(evs)} 則,目前策略={d.get('strategy_text') or '尚未收到'}"))

    # 下發封鎖 —— 這是安全項,沒到 L5 就必須是關的
    ctl = _get("/api/signal/control/status")
    if ctl is None:
        out.append(_item("control_locked", "安全", "TC3 下發封鎖",
                         "未進入 L5 前,設定命令通道必須關閉",
                         UNKNOWN, "端點無回應"))
    else:
        enabled = bool(ctl.get("enabled") or ctl.get("control_enabled"))
        qonly = bool(ctl.get("query_only", True))
        safe = (not enabled) or qonly
        out.append(_item(
            "control_locked", "安全", "TC3 下發封鎖",
            "未進入 L5 前,設定命令通道必須關閉",
            PASS if safe else FAIL,
            f"control_enabled={enabled} query_only={qonly}",
            "送錯一則 5F10/5F15 會直接改變路口實際運轉"))
    return out


def _check_params() -> list:
    """決策參數是不是「落地的準確值」(使用者要求),以及量測的前提有沒有成立。"""
    out: list = []
    plan = _get("/api/signal/shadow/plan") or {}
    c = plan.get("constants") or {}
    src = str(c.get("saturation_source") or "")
    st = PASS if src.startswith("measured") else (UNKNOWN if not c else WARN)
    out.append(_item(
        "param_saturation", "決策參數", "飽和流量:現場實測",
        "saturation_source = measured(24h 視窗、只從有隊伍的綠燈量、落地檔重啟載回)",
        st, f"{c.get('saturation_vph')} vph,來源 {src or '—'},量於 {c.get('saturation_measured_at')}",
        "退回預設 1800 會讓 change_cost 從 ~3.5 變 12.5,引擎明顯不願意換相。"
        "09-05 凌晨就是這樣把整個早尖峰比對作廢的。"))

    lsrc = str(c.get("lost_time_source") or "")
    if "5F03" in lsrc and "實測" in lsrc:
        st, ev = PASS, f"{c.get('lost_time_sec')} s,{lsrc}"
    elif "5FC4" in lsrc:
        st, ev = PASS, f"{c.get('lost_time_sec')} s,{lsrc}(控制器回報的設定值,非假設)"
    elif not c:
        st, ev = UNKNOWN, "決策引擎無資料"
    else:
        st, ev = WARN, f"{c.get('lost_time_sec')} s,{lsrc or '預設'}"
    out.append(_item(
        "param_lost_time", "決策參數", "換相損失時間:控制器數值",
        "來源為控制器 5F03 每秒回報實測,或控制器 5FC4 時制設定;不可為假設值",
        st, ev,
        "🛑 5F03 不是每秒一框時不可用框量:2026-09-05 量出 8.5 秒(真值 5),差的是一個框距。"))

    msrc = str(c.get("meters_per_vehicle_source") or "")
    st = PASS if msrc.startswith("實測") else (UNKNOWN if not c else WARN)
    out.append(_item(
        "param_mpv", "決策參數", "每車佔用長度:現場實測",
        "停止線相機 排隊公尺 ÷ 停等車數(停等 ≥2 台)取中位數,≥100 筆才採信",
        st, f"{c.get('meters_per_vehicle')} m,{msrc or '—'}"))

    fi = c.get("frame_interval_sec")
    if fi is None:
        st, ev = UNKNOWN, "無法量框距(抄錄框不足)"
    elif float(fi) <= 1.5:
        st, ev = PASS, f"5F03 中位框距 {fi} 秒(每秒一框)"
    else:
        st, ev = WARN, f"5F03 中位框距 {fi} 秒 —— 09-04 14:22:48 起控制器 TransmitCycle 被改為 0(僅變化時回報)"
    out.append(_item(
        "frame_interval", "資料源", "控制器燈態回報週期",
        "5F03 每秒一框(5F6F TransmitCycle=1),配對法秒數精度才有 ±1 秒",
        st, ev,
        "框距變大時:配對法精度只剩一個框距、損失時間不可用框量。回復需送 5F6F,目前 QUERY_ONLY=1,要使用者同意。"))

    # 逐時評估:今天到目前為止每一小時都有列(已結束的小時不可缺)
    try:
        from datetime import date as _date
        today = _date.today().isoformat()
        hr = _get(f"/api/signal/shadow/hourly?date={today}") or {}
        rows = hr.get("rows") or []
        done = [r for r in rows if not r.get("partial")]
        expect = max(0, datetime.now().hour)          # 00..(now-1) 小時應已結束
        missing = expect - len(done)
        if not rows:
            st, ev = UNKNOWN, "今天尚無逐時列"
        elif missing <= 0:
            st, ev = PASS, f"今天 {len(done)} 個已結束小時全部有列(進行中 {len(rows) - len(done)})"
        else:
            st, ev = WARN, f"今天缺 {missing} 個小時的列({len(done)}/{expect}),背景回填中或有錯"
    except Exception as e:
        st, ev = UNKNOWN, f"取逐時列失敗: {e}"
    out.append(_item(
        "hourly_eval", "歷史與統計", "逐時評估無缺漏",
        "每整點自動算前一小時(配對/成效/一致率/參數),存表可查",
        st, ev, "使用者要求「每小時都要有」。"))
    return out


@router.get("", summary="號誌整合驗收清單(每項實測)")
async def acceptance(_user=Depends(get_current_user)):
    """逐項實際量測目前狀態。量不到一律 unknown,不當成通過。"""
    st = _get("/api/signal/status")
    items: list = []
    items += _check_recorder(st)
    items += _check_phase(st)
    items += _check_config()
    items += _check_device()
    items += _check_safety()
    items += _check_shadow()
    items += _check_params()
    items += _check_stats()

    counts = {k: 0 for k in (PASS, FAIL, WARN, UNKNOWN)}
    for it in items:
        counts[it["state"]] = counts.get(it["state"], 0) + 1
    total = len(items) or 1
    groups: dict = {}
    for it in items:
        groups.setdefault(it["group"], []).append(it)
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "total": len(items),
        "counts": counts,
        "pass_rate": round(counts[PASS] / total, 3),
        "ready_for_l5": counts[FAIL] == 0 and counts[UNKNOWN] == 0
                        and all(i["state"] == PASS for i in items
                                if i["key"] in ("agree_rate", "control_locked")),
        "groups": [{"title": g, "items": v} for g, v in groups.items()],
        "note": "每一項都是打端點實際量出來的。量不到回 unknown,不當成通過 —— "
                "缺資料和通過是兩件事。",
    }
