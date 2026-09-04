# -*- coding: utf-8 -*-
"""號誌控制模擬器 —— 用同一份到達流量餵兩套演算法,比較成效。

為什麼需要它:影子模式只能證明「我方接手不會亂來」(安全),**證明不了更好**。
我方沒有真的控制過,「如果當時是我方在控」是反事實 —— 拿影子決策去回推成效
會踩到沒有回饋迴路的問題:我方若提早換相,之後的車流就與實際不同,但我們手上
只有現行控制那一條時間線的排隊量測。

模擬可以補這個洞,但**有一個前提**:

🛑 模型必須先用現場資料校準過,否則只是把假設換個地方放。
   校準方式:把「現場實際發生的換相序列」餵進模型,看模擬出來的排隊
   能不能重現實際量到的排隊。重現不了,就代表模型不能用來預測
   「換另一套控制會怎樣」—— 這時候必須誠實說模擬結論不成立,
   不可以因為想要結論就放寬標準。

模型刻意做得簡單(確定性排隊模型),因為:
  1. 參數越多越難校準,越容易用「調參數」把想要的結論調出來
  2. 這個路口只有兩個分相、單點控制,不需要路網模型
  3. 簡單模型的失準會誠實地顯示在校準指標上,複雜模型會把失準藏起來
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from detection.signal_decision_engine import (
    DEFAULT_LOST_TIME_SEC,
    DEFAULT_METERS_PER_VEHICLE,
    DEFAULT_SATURATION_VPH,
)

# 分相數(這個路口固定兩相:1=上匝道 2=下匝道)
PHASES = (1, 2)


@dataclass
class SimConfig:
    """模擬參數。全部沿用決策引擎的常數,不另外訂一套 ——
    模擬用一套、決策用另一套的話,比出來的差異可能只是參數不同。"""
    dt_sec: float = 5.0
    saturation_vph: float = DEFAULT_SATURATION_VPH
    meters_per_vehicle: float = DEFAULT_METERS_PER_VEHICLE
    lost_time_sec: float = DEFAULT_LOST_TIME_SEC
    min_green_sec: dict = field(default_factory=lambda: {1: 10.0, 2: 20.0})
    max_green_sec: float = 210.0


def estimate_arrivals(rows: list, mpv: float = DEFAULT_METERS_PER_VEHICLE) -> dict:
    """從「紅燈期間排隊成長」估各分相的到達率(輛/秒)。

    rows: [(ts_iso, green_phase, queue_m_1, queue_m_2), ...] 依時間排序

    為什麼挑紅燈期間:紅燈時沒有車離開,排隊成長速率**就是**到達率。
    綠燈期間量到的變化是「到達 − 離開」,分不出來。這是標準的輸入輸出法。

    🛑 回傳裡帶 samples 與 spread,呈現時要一起看 —— 樣本太少或離散太大
       就代表這個估計撐不起後面的模擬。
    """
    out: dict = {}
    for ph in PHASES:
        idx = 2 if ph == 1 else 3
        rates: list = []
        run = None          # [t0, q0, t1, q1]
        for r in rows:
            q = r[idx]
            if q is None:
                continue
            t = datetime.fromisoformat(r[0]).timestamp()
            if r[1] != ph:                      # 這一相是紅燈
                if run is None:
                    run = [t, q, t, q]
                else:
                    run[2], run[3] = t, q
            else:
                if run and (run[2] - run[0]) >= 15:
                    dt = run[2] - run[0]
                    dq = (run[3] - run[1]) / mpv
                    if dq > 0:
                        rates.append(dq / dt)
                run = None
        rates.sort()
        n = len(rates)
        if n:
            med = rates[n // 2]
            lo = rates[max(0, int(n * 0.25))]
            hi = rates[min(n - 1, int(n * 0.75))]
        else:
            med = lo = hi = None
        out[ph] = {"veh_per_sec": med, "samples": n, "p25": lo, "p75": hi}
    return out


def arrival_profile(rows: list, window_sec: float = 900.0,
                    mpv: float = DEFAULT_METERS_PER_VEHICLE) -> dict:
    """分段估到達率(預設 15 分鐘一段)。

    🛑 為什麼一定要時變:用單一中位數代表數小時,配上實際換相序列只會產生
       規律鋸齒波,跟真實排隊自然不相關 —— 2026-09-04 首次校準就是這樣,
       相關係數 -0.006 / -0.04(等於完全沒跟上動態),MAE 反而看不出問題。
       真實需求隨時段變化很大,固定到達率是站不住的假設。

    這是「用更多真實資料」不是「調參數湊答案」:每一段的率仍然是從該段的
    紅燈成長算出來的,沒有任何自由參數。
    """
    if not rows:
        return {"windows": [], "window_sec": window_sec}
    t0 = datetime.fromisoformat(rows[0][0]).timestamp()
    tn = datetime.fromisoformat(rows[-1][0]).timestamp()
    wins = []
    start = t0
    while start < tn:
        end = min(start + window_sec, tn)
        seg = [r for r in rows
               if start <= datetime.fromisoformat(r[0]).timestamp() <= end]
        if len(seg) >= 12:
            a = estimate_arrivals(seg, mpv)
            wins.append({"t_start": round(start - t0, 1),
                         "t_end": round(end - t0, 1),
                         "rates": {p: (a[p]["veh_per_sec"] or 0.0) for p in PHASES},
                         "samples": {p: a[p]["samples"] for p in PHASES}})
        start = end
    # 某段估不到(該段沒有夠長的紅燈)就沿用前一段,不要當成 0
    last = {p: 0.0 for p in PHASES}
    for w in wins:
        for p in PHASES:
            if w["rates"][p] > 0:
                last[p] = w["rates"][p]
            else:
                w["rates"][p] = last[p]
    return {"windows": wins, "window_sec": window_sec}


def profile_rate_fn(profile: dict) -> Callable:
    """把分段到達率包成 rate(t, phase) 供 simulate 使用。"""
    wins = profile.get("windows") or []

    def rate(t: float, phase: int) -> float:
        for w in wins:
            if w["t_start"] <= t <= w["t_end"]:
                return float(w["rates"].get(phase) or 0.0)
        return float(wins[-1]["rates"].get(phase) or 0.0) if wins else 0.0
    return rate


def simulate(arrivals, switch_fn: Callable, duration_sec: float,
             cfg: Optional[SimConfig] = None,
             init_queue_veh: Optional[dict] = None,
             start_phase: int = 1) -> dict:
    """跑一段模擬。

    switch_fn(state) -> bool:給控制器決定「這一刻要不要換相」。
      state = {"t", "green_phase", "green_elapsed", "queue_veh": {1:..,2:..}}
      🛑 回 True 也不一定會換 —— 最小綠/最大綠由模擬器強制,
         這樣兩套控制受到的安全約束才一致,比較才公平。

    回傳逐步的軌跡與成效指標。
    """
    cfg = cfg or SimConfig()
    sat = cfg.saturation_vph / 3600.0
    q = dict(init_queue_veh or {p: 0.0 for p in PHASES})
    green = start_phase
    green_elapsed = 0.0
    t = 0.0
    lost_remaining = 0.0        # 換相後的損失時間,這段時間沒有車能走
    total_delay = 0.0
    switches = 0
    forced = 0
    traj: list = []

    while t < duration_sec:
        dt = cfg.dt_sec
        # ① 到達。arrivals 可以是固定率的 dict,也可以是 rate(t, phase) 函式
        #    (時變到達率 —— 固定率撐不起數小時的模擬,見 arrival_profile)
        for p in PHASES:
            if callable(arrivals):
                rate = float(arrivals(t, p) or 0.0)
            else:
                rate = (arrivals.get(p) or {}).get("veh_per_sec") or 0.0
            q[p] += rate * dt
        # ② 放行(只有綠燈側,且要先扣掉換相損失時間)
        eff = max(0.0, dt - lost_remaining)
        lost_remaining = max(0.0, lost_remaining - dt)
        if eff > 0:
            q[green] = max(0.0, q[green] - sat * eff)
        # ③ 延滯 = 這段時間內排隊的車 × 時間
        total_delay += (q[1] + q[2]) * dt
        t += dt
        green_elapsed += dt

        traj.append({"t": round(t, 1), "green_phase": green,
                     "green_elapsed": round(green_elapsed, 1),
                     "q1": round(q[1], 2), "q2": round(q[2], 2)})

        # ④ 換相決策 —— 安全約束由模擬器統一強制,兩套控制條件一致
        want = bool(switch_fn({"t": t, "green_phase": green,
                               "green_elapsed": green_elapsed,
                               "queue_veh": dict(q)}))
        min_g = cfg.min_green_sec.get(green, 10.0)
        if green_elapsed >= cfg.max_green_sec:
            want, is_forced = True, True
        elif green_elapsed < min_g:
            want, is_forced = False, False
        else:
            is_forced = False
        if want:
            green = 2 if green == 1 else 1
            green_elapsed = 0.0
            lost_remaining = cfg.lost_time_sec
            switches += 1
            if is_forced:
                forced += 1

    mpv = cfg.meters_per_vehicle
    qs1 = [x["q1"] for x in traj]
    qs2 = [x["q2"] for x in traj]
    return {
        "duration_sec": round(t, 1),
        "total_delay_veh_sec": round(total_delay, 1),
        "avg_queue_m_1": round(sum(qs1) / len(qs1) * mpv, 1) if qs1 else None,
        "avg_queue_m_2": round(sum(qs2) / len(qs2) * mpv, 1) if qs2 else None,
        "max_queue_m_1": round(max(qs1) * mpv, 1) if qs1 else None,
        "max_queue_m_2": round(max(qs2) * mpv, 1) if qs2 else None,
        "switch_count": switches,
        "forced_switch_count": forced,
        "switch_per_min": round(switches / (t / 60.0), 2) if t else None,
        "trajectory": traj,
    }


def replay_actual(rows: list, arrivals,
                  cfg: Optional[SimConfig] = None) -> dict:
    """把「現場實際的換相序列」餵進模型 —— 這是校準用的。

    做法:照著 rows 裡 green_phase 的變化時刻換相,不做任何決策。
    模擬出來的排隊若能重現實際量到的排隊,模型才可信。
    """
    cfg = cfg or SimConfig()
    marks = []
    prev = None
    t0 = datetime.fromisoformat(rows[0][0]).timestamp()
    for r in rows:
        t = datetime.fromisoformat(r[0]).timestamp() - t0
        if prev is not None and r[1] != prev:
            marks.append(t)
        prev = r[1]
    marks_set = sorted(marks)
    idx = {"i": 0}

    def switch_fn(state):
        # 走到下一個實際換相時刻就換
        while idx["i"] < len(marks_set) and marks_set[idx["i"]] <= state["t"]:
            idx["i"] += 1
            return True
        return False

    dur = datetime.fromisoformat(rows[-1][0]).timestamp() - t0
    return simulate(arrivals, switch_fn, dur, cfg,
                    start_phase=rows[0][1] or 1)


def calibrate(rows: list, sim: dict,
              mpv: float = DEFAULT_METERS_PER_VEHICLE) -> dict:
    """比對「模擬排隊」與「實際量到的排隊」,回傳擬合指標。

    🛑 這是整個模擬路線的關卡。指標沒過就是模型不能用,
       不可以拿它去宣稱任何「我方比較好」的結論。

    判準(刻意訂得保守):
      - 平均絕對誤差 MAE ≤ 12 公尺(約 1.7 台車)
      - 相關係數 r ≥ 0.5(至少要抓得到漲跌的方向)
      兩項都過才算 usable。
    """
    traj = sim.get("trajectory") or []
    if not traj or not rows:
        return {"usable": False, "reason": "沒有資料可比對"}
    t0 = datetime.fromisoformat(rows[0][0]).timestamp()
    meas = []
    for r in rows:
        meas.append((datetime.fromisoformat(r[0]).timestamp() - t0,
                     r[2], r[3]))

    def pair(idx_meas, key):
        xs, ys = [], []
        j = 0
        for (tm, m1, m2) in meas:
            m = m1 if idx_meas == 0 else m2
            if m is None:
                continue
            while j + 1 < len(traj) and traj[j + 1]["t"] <= tm:
                j += 1
            xs.append(float(m))
            ys.append(float(traj[j][key]) * mpv)
        return xs, ys

    def stats(xs, ys):
        n = len(xs)
        if n < 10:
            return {"n": n, "mae": None, "r": None}
        mae = sum(abs(a - b) for a, b in zip(xs, ys)) / n
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs)
        vy = sum((b - my) ** 2 for b in ys)
        r = cov / ((vx * vy) ** 0.5) if vx > 0 and vy > 0 else None
        return {"n": n, "mae": round(mae, 1),
                "r": round(r, 3) if r is not None else None}

    s1 = stats(*pair(0, "q1"))
    s2 = stats(*pair(1, "q2"))
    MAE_MAX, R_MIN = 12.0, 0.5

    def ok(s):
        return (s["mae"] is not None and s["mae"] <= MAE_MAX
                and s["r"] is not None and s["r"] >= R_MIN)

    usable = ok(s1) and ok(s2)
    reasons = []
    for name, s in (("分相1", s1), ("分相2", s2)):
        if s["mae"] is None:
            reasons.append(f"{name} 樣本不足({s['n']})")
        else:
            if s["mae"] > MAE_MAX:
                reasons.append(f"{name} MAE {s['mae']}m > {MAE_MAX}m")
            if s["r"] is None or s["r"] < R_MIN:
                reasons.append(f"{name} 相關係數 {s['r']} < {R_MIN}")
    return {
        "usable": usable,
        "thresholds": {"mae_max_m": MAE_MAX, "r_min": R_MIN},
        "phase_1": s1, "phase_2": s2,
        "reason": "校準通過" if usable else "；".join(reasons),
        "note": "校準沒過就代表模型無法在已知控制下重現現場排隊,"
                "更不可能預測『換另一套控制會怎樣』—— 此時模擬結論一律不成立。",
    }
