# -*- coding: utf-8 -*-
"""演算法最佳化的**驗證平台** —— 讓「改好了」這件事可以被證明。

使用者 2026-09-05:「要能驗證比較 演算法最佳化」。

在這之前的問題是:改一個參數、跑一次合成需求、看到數字變好就當成改進。
那不算驗證 —— 單一情境的改善可能只是把參數調到剛好合那個情境。
這個模組提供三件缺的東西:

## 一、天花板:這個模型的真正最佳解(beam search)

不知道最佳解是多少,就不知道「差 13%」是還有很大空間、還是已經接近極限。
`optimal_policy()` 用 beam search 在同一個排隊模型上找近似最佳的換相序列:
逐步展開「換/不換」兩個分支,依成本剪枝,保留前 K 個狀態。
K 夠大時結果非常接近真正最佳(K=1 就退化成貪婪)。

🛑 它是**離線的、看得到未來所有到達**的最佳解,任何線上控制都不可能達到 ——
   它的用途是當上界,不是當對照組。線上控制拿它比,只該問「差多少」。

## 二、多情境:一次跑一整組需求,不是單一情境

`scenario_set()` 產生一組有系統差異的需求(整體流量高低、兩相不對稱程度、
尖峰錯開與否、突發湧入),每個都是決定性的(給定 seed 就固定),
所以任何人重跑都會得到同一組數字。

## 三、訓練/驗證分離:改進必須在沒調過的情境上也成立

`tune()` 只在**訓練情境**上搜參數,`evaluate()` 在**驗證情境**上報結果。
訓練集上變好、驗證集上沒變好,就是過度擬合,不算改進。
這是「可驗證」的核心 —— 沒有這一條,參數搜尋只是把數字調給自己看。
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from detection.signal_baselines import _demand_veh, actuated, fixed_time, max_pressure, metrics
from detection.signal_decision_engine import ApproachState, decide
from detection.signal_sim import SimConfig, simulate

# 終端懲罰:時域結束時仍在排隊的車,每輛折算幾秒。
# 最佳解若沒有這一項,會學會「把車留到最後」讓帳面延滯變好看。
TERMINAL_WEIGHT_SEC = 25.0


# ── 一、天花板:beam search 最佳解 ────────────────────────────────────

def optimal_policy(rate_fn: Callable, duration_sec: float, cfg: SimConfig,
                   start_phase: int = 1, beam: int = 300,
                   terminal_weight_sec: float = TERMINAL_WEIGHT_SEC) -> dict:
    """在同一個排隊模型上找近似最佳的換相序列(離線、已知未來到達)。

    回傳與 simulate() 相容的欄位,外加 switch_times 與 beam。
    """
    dt = cfg.dt_sec
    sat = {p: (cfg.saturation_by_phase or {}).get(p)
           or cfg.saturation_vph / 3600.0 for p in (1, 2)}
    # 狀態:(cost, q1, q2, green, elapsed, lost_remaining, switch_times, q_series)
    states = [(0.0, 0.0, 0.0, start_phase, 0.0, 0.0, (), ())]
    t = 0.0
    while t < duration_sec:
        a1 = float(rate_fn(t, 1) or 0.0) * dt
        a2 = float(rate_fn(t, 2) or 0.0) * dt
        nxt = []
        for (cost, q1, q2, g, el, lost, sw, series) in states:
            q1n, q2n = q1 + a1, q2 + a2
            eff = max(0.0, dt - lost)
            lost_n = max(0.0, lost - dt)
            if eff > 0:
                if g == 1:
                    q1n = max(0.0, q1n - sat[1] * eff)
                else:
                    q2n = max(0.0, q2n - sat[2] * eff)
            c = cost + (q1n + q2n) * dt
            t2 = t + dt
            el2 = el + dt
            ser = series + (round(q1n, 2), round(q2n, 2))
            min_g = cfg.min_green_sec.get(g, 10.0)
            # 換相是否可行:最小綠已滿;最大綠到頂則強制
            can_switch = el2 >= min_g
            must_switch = el2 >= cfg.max_green_sec
            if not must_switch:
                nxt.append((c, q1n, q2n, g, el2, lost_n, sw, ser))
            if can_switch or must_switch:
                ng = 2 if g == 1 else 1
                nxt.append((c, q1n, q2n, ng, 0.0, cfg.lost_time_sec,
                            sw + (round(t2, 1),), ser))
        # 剪枝:同一個「格點」只留成本最低的,再取前 beam 名
        best_by_key = {}
        for s in nxt:
            key = (s[3], int(s[4] // dt), round(s[1], 1), round(s[2], 1))
            if key not in best_by_key or s[0] < best_by_key[key][0]:
                best_by_key[key] = s
        states = sorted(best_by_key.values(), key=lambda s: s[0])[:beam]
        t += dt

    # 終端懲罰後選最佳
    scored = sorted(states, key=lambda s: s[0] + (s[1] + s[2]) * terminal_weight_sec)
    best = scored[0]
    cost, q1, q2, g, el, lost, sw, series = best
    qs1 = series[0::2]
    qs2 = series[1::2]
    mpv = cfg.meters_per_vehicle
    n = max(1, len(qs1))
    return {
        "duration_sec": round(t, 1),
        "total_delay_veh_sec": round(cost, 1),
        "avg_queue_m_1": round(sum(qs1) / n * mpv, 1),
        "avg_queue_m_2": round(sum(qs2) / n * mpv, 1),
        "max_queue_m_1": round(max(qs1) * mpv, 1) if qs1 else None,
        "max_queue_m_2": round(max(qs2) * mpv, 1) if qs2 else None,
        "switch_count": len(sw),
        "forced_switch_count": 0,
        "switch_per_min": round(len(sw) / (t / 60.0), 2) if t else None,
        "trajectory": [{"t": t, "q1": q1, "q2": q2}],   # 只留終態,供殘留計算
        "switch_times": list(sw),
        "beam": beam,
        "note": "離線最佳解(已知未來所有到達),是上界不是對照組",
    }


# ── 二、多情境 ────────────────────────────────────────────────────────

def scenario_set(kind: str = "all") -> list:
    """產生一組決定性的需求情境。給定名稱就固定,任何人重跑都一樣。

    每個情境回 {"name", "rate", "flows"}:
      rate(t, phase) -> 輛/秒;flows 是時段平均(算 Webster 用)。
    """
    def make(name, base1, base2, period, phase_shift, floor=0.35, surge=None):
        def rate(t, p, base1=base1, base2=base2, period=period,
                 phase_shift=phase_shift, floor=floor, surge=surge):
            b = base1 if p == 1 else base2
            ph = 0.0 if p == 1 else phase_shift
            f = 0.5 + 0.5 * math.sin(2 * math.pi * t / period + ph)
            v = b * (floor + (1.0 - floor) * f)
            if surge and surge[0] <= t < surge[1] and p == surge[2]:
                v *= surge[3]
            return v
        flows = {p: sum(rate(x, p) for x in range(0, 3600, 5)) / 720 for p in (1, 2)}
        return {"name": name, "rate": rate, "flows": flows}

    train = [
        make("平穩對稱", 0.10, 0.10, 1800.0, 0.0),
        make("尖峰錯開", 0.16, 0.13, 1800.0, math.pi),
        make("分相1 偏重", 0.20, 0.08, 2400.0, math.pi / 2),
        make("低流量", 0.05, 0.04, 1800.0, math.pi),
        make("快速起伏", 0.14, 0.12, 600.0, math.pi),
    ]
    test = [
        make("尖峰同步", 0.15, 0.14, 1800.0, 0.0),
        make("分相2 偏重", 0.07, 0.19, 2400.0, math.pi / 2),
        make("接近容量", 0.22, 0.17, 1800.0, math.pi, floor=0.55),
        make("突發湧入", 0.11, 0.10, 1800.0, math.pi,
             surge=(1200.0, 1800.0, 2, 2.5)),
        make("長波動", 0.13, 0.11, 3600.0, math.pi),
    ]
    if kind == "train":
        return train
    if kind == "test":
        return test
    return train + test


# ── 三、可調參數的規則版 ──────────────────────────────────────────────

DEFAULT_PARAMS = {
    "change_cost_mult": 1.0,    # 換相成本倍率(越大越不願換相)
    "keep_weight": 1.0,         # 綠側價值倍率
    "clear_gap_sec": 0.0,       # 綠側排隊清空後再等幾秒才准換(模仿感應控制)
}


def rule_switch_fn(cfg: SimConfig, roles: dict, params: Optional[dict] = None):
    """把決策引擎包成 switch_fn,並開放幾個參數給最佳化搜尋。

    🛑 參數只加乘數與一個 gap,不改成本函數的結構 —— 結構要能講得出道理,
       乘數是現場校正。參數多了就會變成「調到剛好合這批情境」。
    """
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    mpv = cfg.meters_per_vehicle
    box = {"empty_since": None}

    def switch_fn(state) -> bool:
        g = state["green_phase"]
        r = 2 if g == 1 else 1
        qv = state["queue_veh"]
        sat = (cfg.saturation_by_phase or {}).get(g) or cfg.saturation_vph / 3600.0
        d = decide(
            green_phase=g, green_elapsed_sec=state["green_elapsed"],
            green_side=ApproachState(g, queue_m=qv[g] * mpv,
                                     storage_m=roles[g].get("storage_m"),
                                     priority=bool(roles[g].get("priority"))),
            red_side=ApproachState(r, queue_m=qv[r] * mpv,
                                   storage_m=roles[r].get("storage_m"),
                                   priority=bool(roles[r].get("priority")),
                                   waiting_sec=state["green_elapsed"]),
            min_green_sec=cfg.min_green_sec.get(g, 10.0),
            max_green_sec=cfg.max_green_sec,
            saturation_vph=sat * 3600.0,
            meters_per_vehicle=mpv,
            lost_time_sec=cfg.lost_time_sec,
            keep_weight=p["keep_weight"])
        if d.forced_by_max_green or d.blocked_by_priority:
            return d.action == "SWITCH"
        # 🛑 keep_weight 交給引擎算(2026-09-06 起引擎原生支援),這裡只再處理
        #    change_cost_mult —— 驗證平台與線上引擎必須用**同一套算法**,
        #    在外面各算一次門檻,兩邊遲早會漂移,那時驗證出來的參數就不能上線了。
        if p["change_cost_mult"] == 1.0:
            want = d.action == "SWITCH"
        else:
            thr = d.keep_gain * p["keep_weight"] + d.change_cost * p["change_cost_mult"]
            want = d.switch_gain > thr
        # 清空後等待(模仿感應控制的 gap-out):綠側還有車就不准換
        gap = float(p["clear_gap_sec"] or 0.0)
        if gap > 0:
            t = float(state["t"])
            if qv[g] >= 0.5:
                box["empty_since"] = None
            elif box["empty_since"] is None:
                box["empty_since"] = t
            if qv[g] >= 0.5 or (box["empty_since"] is not None
                                and t - box["empty_since"] < gap):
                # 綠側還沒放完(或剛放完不到 gap 秒):除非紅側壓力已經很大
                if not want:
                    return False
        return want
    return switch_fn


# ── 評估與搜尋 ────────────────────────────────────────────────────────

def _controllers(cfg: SimConfig, roles: dict, params: dict, sc: dict) -> dict:
    sat = cfg.saturation_by_phase or {}
    return {
        "ours": rule_switch_fn(cfg, roles, params),
        "actuated": actuated(3.0),
        "max_pressure": max_pressure(sat),
        "fixed_plan": fixed_time({1: 35.0, 2: 40.0}),
    }


def evaluate(scenarios: list, cfg: SimConfig, roles: dict,
             params: Optional[dict] = None, duration_sec: float = 3600.0,
             with_optimal: bool = False, beam: int = 200) -> dict:
    """在一組情境上評估各控制器,回每情境與平均的每車延滯。"""
    params = params or DEFAULT_PARAMS
    rows = []
    for sc in scenarios:
        rate = sc["rate"]
        demand = _demand_veh(rate, duration_sec, cfg.dt_sec)
        row = {"scenario": sc["name"], "demand_veh": round(demand, 1)}
        for key, fn in _controllers(cfg, roles, params, sc).items():
            sim = simulate(rate, fn, duration_sec, cfg, start_phase=1)
            row[key] = metrics(sim, demand, cfg)["delay_per_veh_sec"]
        if with_optimal:
            opt = optimal_policy(rate, duration_sec, cfg, beam=beam)
            row["optimal"] = metrics(opt, demand, cfg)["delay_per_veh_sec"]
        rows.append(row)
    keys = [k for k in rows[0] if k not in ("scenario", "demand_veh")]
    mean = {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in keys}
    out = {"rows": rows, "mean_delay_per_veh": mean, "params": dict(params)}
    if with_optimal:
        # 離最佳還有多遠(越小越好)
        out["gap_to_optimal_pct"] = {
            k: round((mean[k] - mean["optimal"]) / mean["optimal"] * 100, 1)
            for k in keys if k != "optimal" and mean["optimal"]}
    return out


def tune(cfg: SimConfig, roles: dict, grid: Optional[dict] = None,
         duration_sec: float = 3600.0) -> dict:
    """只在**訓練情境**上搜參數。回最佳參數與訓練集成績。

    座標下降:一次只動一個參數,掃完一輪再來一輪,直到不再變好。
    刻意不用複雜的最佳化器 —— 參數只有三個,而且要能解釋每一步為什麼變。
    """
    grid = grid or {
        "change_cost_mult": [0.5, 1.0, 2.0, 4.0, 8.0],
        "keep_weight": [0.5, 1.0, 1.5, 2.0, 3.0],
        "clear_gap_sec": [0.0, 3.0, 6.0, 10.0],
    }
    train = scenario_set("train")
    best = dict(DEFAULT_PARAMS)
    best_score = evaluate(train, cfg, roles, best, duration_sec)["mean_delay_per_veh"]["ours"]
    history = [{"params": dict(best), "train_delay": best_score}]
    improved = True
    rounds = 0
    while improved and rounds < 3:
        improved = False
        rounds += 1
        for name, values in grid.items():
            for v in values:
                if v == best[name]:
                    continue
                cand = dict(best)
                cand[name] = v
                s = evaluate(train, cfg, roles, cand, duration_sec)["mean_delay_per_veh"]["ours"]
                if s < best_score - 1e-9:
                    best, best_score = cand, s
                    history.append({"params": dict(cand), "train_delay": round(s, 3)})
                    improved = True
    return {"best_params": best, "train_delay_per_veh": round(best_score, 3),
            "rounds": rounds, "history": history}
