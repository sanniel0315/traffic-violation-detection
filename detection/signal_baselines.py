# -*- coding: utf-8 -*-
"""基準控制器 —— 我方演算法的對照組。

🛑 為什麼不拿現行控制(OPAC)當驗收基準:
   使用者 2026-09-05 明確指出「OPAC 是最差的控制,沒法驗收」。
   拿一個沒調好的系統當對照,贏了也不能證明我方演算法好 ——
   只證明了對方沒調好。而且 OPAC 的參數不在我方手上,它變差變好
   我方無法控制,基準會漂移。

   OPAC 在這條路線上只剩一個角色:**校準**。把它實際發生的換相序列
   餵進模型,看模擬排隊能不能重現現場量到的排隊(signal_sim.calibrate)。
   校準是驗證模型,不是驗收演算法 —— 兩件事分開。

## 對照組(全部是交通工程教科書/文獻裡的標準控制)

| 基準 | 說明 | 為什麼選它 |
|------|------|-----------|
| 固定時制(現行計畫) | 週期 85 s、綠燈 35/40,現場原本的時制表 | 這是動態控制上線前的真實狀態,最有說服力的「之前」 |
| Webster 最佳固定時制 | 用實測流量比與損失時間算最佳週期與分配 | 固定時制的理論上限;贏不過它就沒有做動態的必要 |
| 感應控制(vehicle-actuated) | 綠燈側車放完且間距超過門檻就切(gap-out) | 最普及的動態控制,是「動態」這件事的基本盤 |
| MaxPressure | 切到壓力(排隊×飽和流)較大的一相 | 近十年文獻的標準對照,有穩定性證明 |

四個對照組與我方演算法跑**同一份到達流量、同一組安全約束**
(最小綠、最大綠、損失時間都由模擬器統一強制),差異只來自換相決策。

## 誠實邊界
- 這是確定性排隊模型(point queue):不模擬車輛長度佔用、轉向、行人。
- 校準沒過就不可以引用任何比較結果 —— 由呼叫端強制(見 /benchmark)。
- 到達率是「該時段紅燈排隊成長」實測出來的,不是假設值。
"""
from __future__ import annotations

from typing import Callable, Optional

from detection.signal_sim import PHASES, SimConfig, simulate  # noqa: F401


# ── 基準控制器:每個都回傳 switch_fn(state) -> bool ────────────────────

def fixed_time(green_sec: dict) -> Callable:
    """固定時制:各相綠燈固定秒數,時間到就換。

    green_sec: {1: 35.0, 2: 40.0} —— 現行計畫 1 的綠燈分配。
    最小綠/最大綠仍由模擬器強制,所以這裡只表達「意圖」。
    """
    def switch_fn(state) -> bool:
        g = state["green_phase"]
        return state["green_elapsed"] >= float(green_sec.get(g, 30.0))
    return switch_fn


def webster_split(flow_veh_per_sec: dict, sat_veh_per_sec: dict,
                  lost_time_sec: float, n_phases: int = 2,
                  min_green: Optional[dict] = None,
                  max_cycle: float = 150.0) -> dict:
    """Webster 最佳固定時制:算最佳週期與綠燈分配。

        y_i = q_i / s_i            (各相流量比)
        Y   = Σ y_i
        L   = 損失時間 × 相數      (整個週期的總損失)
        C0  = (1.5 L + 5) / (1 − Y)
        g_i = (C0 − L) × y_i / Y

    回傳 {"cycle","green","flow_ratio","Y","feasible","note"}。
    Y ≥ 1 表示需求超過容量,固定時制在理論上無解 —— 這種情況要照實說,
    不可以硬套公式產生一個看起來合理的數字。
    """
    y = {}
    for p in (1, 2):
        s = float(sat_veh_per_sec.get(p) or 0.0)
        q = float(flow_veh_per_sec.get(p) or 0.0)
        y[p] = (q / s) if s > 0 else 0.0
    Y = sum(y.values())
    L = float(lost_time_sec) * n_phases
    if Y >= 0.95:
        return {"cycle": None, "green": None, "flow_ratio": y, "Y": round(Y, 3),
                "feasible": False,
                "note": f"流量比合計 Y={Y:.2f} ≥ 0.95,已達容量上限,"
                        "Webster 公式無解(固定時制無法消化此需求)"}
    cycle = (1.5 * L + 5.0) / (1.0 - Y)
    cycle = min(cycle, max_cycle)
    total_green = max(cycle - L, 1.0)
    green = {p: (total_green * (y[p] / Y) if Y > 0 else total_green / 2)
             for p in (1, 2)}
    if min_green:
        for p in (1, 2):
            green[p] = max(green[p], float(min_green.get(p, 0.0)))
    return {"cycle": round(cycle, 1),
            "green": {p: round(green[p], 1) for p in (1, 2)},
            "flow_ratio": {p: round(y[p], 3) for p in (1, 2)},
            "Y": round(Y, 3), "feasible": True,
            "note": "Webster (1958) 最小延滯週期"}


def actuated(gap_sec: float = 3.0, mpv_threshold_veh: float = 0.5) -> Callable:
    """感應控制(vehicle-actuated):gap-out。

    真實感應控制看的是車輛通過偵測器的時間間距,間距大於門檻就結束綠燈。
    在排隊模型裡等價的判準是:**綠燈側排隊已經放完**(低於半台車),
    而且到達間距大於 gap —— 排隊放完後只要再等 gap 秒沒有新車,就切。

    這裡用「排隊 < 門檻且持續 gap 秒」近似,是感應控制的標準排隊模型對應。
    """
    state_box = {"empty_since": None}

    def switch_fn(state) -> bool:
        g = state["green_phase"]
        q = float(state["queue_veh"].get(g) or 0.0)
        t = float(state["t"])
        if q >= mpv_threshold_veh:
            state_box["empty_since"] = None
            return False
        if state_box["empty_since"] is None:
            state_box["empty_since"] = t
            return False
        return (t - state_box["empty_since"]) >= gap_sec
    return switch_fn


def max_pressure(sat_veh_per_sec: dict) -> Callable:
    """MaxPressure:切到壓力較大的那一相。

        pressure_p = 排隊車數_p × 飽和流_p

    這個路口是兩相單點,下游沒有其他路口,所以壓力就是自身排隊的
    加權值(沒有下游排隊要扣)。文獻上有佇列穩定性證明,是近年的標準對照。
    """
    def switch_fn(state) -> bool:
        g = state["green_phase"]
        r = 2 if g == 1 else 1
        qv = state["queue_veh"]
        pg = float(qv.get(g) or 0.0) * float(sat_veh_per_sec.get(g) or 0.0)
        pr = float(qv.get(r) or 0.0) * float(sat_veh_per_sec.get(r) or 0.0)
        return pr > pg
    return switch_fn


def cycle_adaptive(sat_veh_per_sec: dict, lost_time_sec: float,
                   min_green: dict, max_green: float,
                   cycle_min: float = 60.0, cycle_max: float = 120.0,
                   window_sec: float = 300.0,
                   arrival_rate: Optional[Callable] = None) -> Callable:
    """週期制動態控制 —— 使用者 2026-09-06 給的規格。

        週期長度 = 綠燈_A + 綠燈_B + 損失時間(黃燈 + 全紅)
        車流大 → 延長週期(上限 cycle_max);車流小 → 縮短(下限 cycle_min)
        各相綠燈受最小綠 / 最大綠限制

    做法:每次回到週期起點,用最近 window_sec 的實測到達率重算 Webster 最佳週期,
    夾在 [cycle_min, cycle_max] 之間,再依流量比分配綠燈並套上下限。
    週期內就照分配好的秒數走(這是週期制與「每 5 秒重新決策」的根本差別)。

    🛑 週期下限是**約束不是優點**:需求遠低於容量時,最小延滯的週期可能比
       下限還短,這時下限會直接製造延滯。要不要設下限、設多少,應該用
       benchmark 對照最佳解來決定,不是先訂了再說。
    """
    st = {"green": {}, "cycle": None, "elapsed_in_plan": 0.0, "last_plan_t": -1e9}

    def replan(t: float, arrivals: dict):
        w = webster_split(arrivals, sat_veh_per_sec, lost_time_sec,
                          min_green=min_green, max_cycle=cycle_max)
        if w.get("feasible"):
            cyc = min(max(float(w["cycle"]), cycle_min), cycle_max)
            g = dict(w["green"])
        else:
            # 需求超過容量:Webster 無解,直接用最大週期把綠燈按流量比分
            cyc = cycle_max
            tot = sum(max(0.0, arrivals.get(p, 0.0)) for p in (1, 2)) or 1.0
            g = {p: max(1.0, cyc - lost_time_sec * 2) * max(0.0, arrivals.get(p, 0.0)) / tot
                 for p in (1, 2)}
        # 🛑 模擬器把損失時間放在**綠燈區間之內**(換相後前幾秒沒有車能走),
        #    沒有另外的全紅區間。所以「一個綠燈區間」= Webster 的有效綠 + 損失時間,
        #    兩個區間相加才等於週期長度。先前寫成 cyc - lost*2 再配比,
        #    等於把損失時間扣兩次,實際週期只有設定值的 5/6
        #    (設 60 秒實測跑出 50 秒、切換 2.4 次/分而不是 2.0)。
        tot_g = sum(g.values()) or 1.0
        eff_total = max(1.0, cyc - lost_time_sec * 2)
        for p in (1, 2):
            interval = eff_total * g[p] / tot_g + lost_time_sec
            g[p] = max(float(min_green.get(p, 10.0)), min(float(max_green), interval))
        st["green"], st["cycle"], st["last_plan_t"] = g, cyc, t
        return g

    def switch_fn(state) -> bool:
        t = float(state["t"])
        g = state["green_phase"]
        if not st["green"] or (t - st["last_plan_t"]) >= window_sec:
            if arrival_rate is not None:
                # 給它**準確的**近期需求(取樣平均),這樣比的是週期制這個策略本身,
                # 不是它的需求估計器 —— 對照組要給最有利的條件。
                n = max(1, int(window_sec // 10))
                arr = {p: sum(float(arrival_rate(max(0.0, t - window_sec) + i * 10.0, p) or 0.0)
                              for i in range(n)) / n for p in (1, 2)}
            else:
                qv = state["queue_veh"]
                arr = {p: max(0.001, float(qv.get(p) or 0.0) / max(window_sec, 1.0) * 60.0)
                       for p in (1, 2)}
            replan(t, arr)
        return state["green_elapsed"] >= float(st["green"].get(g, 30.0))

    return switch_fn


# ── 成效彙整 ──────────────────────────────────────────────────────────

def _demand_veh(rate_fn, duration_sec: float, dt: float) -> float:
    """把到達率對時間積分,得到這段時間的總需求(輛)。
    四個對照組與我方跑同一份需求,所以這個數字對所有人都一樣。"""
    total = 0.0
    t = 0.0
    while t < duration_sec:
        for p in (1, 2):
            total += float(rate_fn(t, p) or 0.0) * dt
        t += dt
    return total


def metrics(sim: dict, demand_veh: float, cfg: SimConfig) -> dict:
    """把 simulate() 的輸出換成可比較的成效指標。

    🛑 平均延滯的分母用「實際放行車數」而不是需求 ——
       控制器若把車擋著不放,延滯總量會低估;殘留排隊要一起看。
    """
    traj = sim.get("trajectory") or []
    residual = 0.0
    if traj:
        residual = float(traj[-1]["q1"]) + float(traj[-1]["q2"])
    served = max(0.0, demand_veh - residual)
    dur = float(sim.get("duration_sec") or 0.0)
    delay = float(sim.get("total_delay_veh_sec") or 0.0)
    q1 = sim.get("avg_queue_m_1")
    q2 = sim.get("avg_queue_m_2")
    return {
        "total_delay_veh_sec": round(delay, 1),
        "delay_per_veh_sec": round(delay / served, 2) if served > 0 else None,
        "avg_queue_m": round(((q1 or 0) + (q2 or 0)) / 2, 1),
        "avg_queue_m_1": q1, "avg_queue_m_2": q2,
        "max_queue_m_1": sim.get("max_queue_m_1"),
        "max_queue_m_2": sim.get("max_queue_m_2"),
        "throughput_vph": round(served / dur * 3600, 1) if dur > 0 else None,
        "residual_queue_veh": round(residual, 2),
        "switch_per_min": sim.get("switch_per_min"),
        "forced_switch_count": sim.get("forced_switch_count"),
    }


def run_benchmark(rate_fn, duration_sec: float, cfg: SimConfig,
                  ours_switch_fn: Callable, start_phase: int = 1,
                  plan_green: Optional[dict] = None,
                  flow_veh_per_sec: Optional[dict] = None) -> dict:
    """跑我方演算法與四個基準,回傳成效表與相對改善。

    plan_green:現行時制表的綠燈分配(例:{1:35,2:40});沒有就跳過該基準。
    flow_veh_per_sec:實測到達率(算 Webster 用);沒有就跳過 Webster。
    """
    sat = cfg.saturation_by_phase or {}
    demand = _demand_veh(rate_fn, duration_sec, cfg.dt_sec)

    controllers: list = [("ours", "我方演算法(延滯成本)", ours_switch_fn, {})]
    if plan_green:
        controllers.append(("fixed_plan", f"固定時制(現行計畫 {plan_green[1]:.0f}/{plan_green[2]:.0f})",
                            fixed_time(plan_green), {"green_sec": plan_green}))
    web = None
    if flow_veh_per_sec:
        web = webster_split(flow_veh_per_sec, sat, cfg.lost_time_sec,
                            min_green=cfg.min_green_sec)
        if web.get("feasible"):
            controllers.append(("webster", "Webster 最佳固定時制",
                                fixed_time(web["green"]), web))
    controllers.append(("actuated", "感應控制(gap-out 3 s)", actuated(3.0), {"gap_sec": 3.0}))
    controllers.append(("max_pressure", "MaxPressure", max_pressure(sat), {}))

    out: dict = {"demand_veh": round(demand, 1),
                 "duration_sec": round(duration_sec, 1),
                 "webster": web, "results": {}}
    for key, label, fn, params in controllers:
        sim = simulate(rate_fn, fn, duration_sec, cfg, start_phase=start_phase)
        m = metrics(sim, demand, cfg)
        m["label"] = label
        m["params"] = params
        out["results"][key] = m

    ours = out["results"].get("ours") or {}
    comp: dict = {}
    for key, m in out["results"].items():
        if key == "ours":
            continue
        row = {"label": m["label"]}
        for fld, better_low in (("delay_per_veh_sec", True),
                                ("avg_queue_m", True),
                                ("throughput_vph", False),
                                ("switch_per_min", None)):
            a, b = ours.get(fld), m.get(fld)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
                pct = (a - b) / b * 100
                row[fld] = {"ours": a, "baseline": b,
                            "diff": round(a - b, 2), "pct": round(pct, 1),
                            "better": None if better_low is None
                                      else (pct < 0 if better_low else pct > 0)}
        comp[key] = row
    out["vs_baseline"] = comp
    out["note"] = ("我方演算法與四個公認基準跑同一份實測到達流量、"
                   "同一組安全約束(最小綠/最大綠/損失時間由模擬器統一強制)。"
                   "現行控制(OPAC)不在對照組內 —— 它只用於模型校準。")
    return out
