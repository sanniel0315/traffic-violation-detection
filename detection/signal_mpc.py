# -*- coding: utf-8 -*-
"""滾動時域最佳化控制(MPC)—— 我方演算法的進階版本。

## 為什麼是 MPC 而不是強化學習(RL)

使用者 2026-09-05:「我們要跑 RL 演算法或更適合的」。這個路口的條件是:

  · 動作空間只有兩個(續綠/換相),狀態只有四維(兩側排隊、綠燈相、已亮秒數)
  · 有一個用現場資料校準的排隊模型
  · 產出要能給公路主管機關稽核

在這種條件下 RL 是**不划算而且不安全**的:

| | RL(DQN/PPO) | MPC(本模組) |
|---|---|---|
| 需要模擬器 | 要,而且要跑幾十萬步訓練 | 要,但只在每次決策往前推 120 秒 |
| 模型誤差的後果 | 政策會**學會鑽模型的漏洞**,誤差被放大且看不見 | 每 5 秒用最新的實測排隊重新規劃,誤差每步被修正 |
| 可稽核 | 權重是黑箱,說不出為什麼切 | 攤得開:每個候選方案的延滯數字都列得出來 |
| 安全約束 | 要靠獎勵塑形間接壓 | 直接寫在候選方案的產生規則裡,不可能違反 |
| 現場資料需求 | 大量,且要能重複互動 | 只要當下的排隊與到達率 |

RL 真正有價值的場景是「狀態高維、動作組合爆炸、模型寫不出來」——
例如整個路網幾十個路口連鎖。單點兩相不是那個場景。

🛑 更關鍵的一點:**不論 RL 還是 MPC,在模擬器裡最佳化出來的東西都會繼承
   模擬器的誤差**。目前模型的排隊平均誤差已進門檻(MAE 9.9/11.7 m),但
   相關係數只有 0.19/0.15 —— 抓得到水準、抓不到起伏。在那之前,任何
   「最佳化後改善 X%」都只能當模擬結論,不能當現場承諾。

## 🛑 現況:還沒有贏過規則版,不要上線也不要寫進報告

2026-09-05 在合成的起伏需求(分相1/2 尖峰錯開,整體約容量七成)上實測,
每車延滯(規則版 15.37 秒為基準):

| 控制 | 延滯/車 | 切換/分 | vs 規則版 |
|------|---------|---------|-----------|
| 感應控制(gap-out) | 13.29 | 2.33 | −13.5% |
| 規則版(延滯成本) | 15.37 | 3.13 | — |
| MaxPressure | 15.55 | 3.22 | +1.2% |
| MPC H=120 | 19.52 | 2.33 | +27.0% |
| 固定時制 | 42.18 | 1.60 | +174% |
| Webster 固定 | 43.93 | 2.00 | +186% |

已修掉兩個 bug(空候選方案被讀成 KEEP、換相時點用遞增後的時間),
但仍輸。時域長度掃過 120/180/240/360 秒 × 終端懲罰 0/25:
**時域越長越差**(H=360 是 +57%),所以不是時域截斷造成的。
研判是候選集合只列兩次換相,時域一長,尾段延續策略就佔掉大部分成本,
候選之間變得無從區分。下一步要做的是先用動態規劃算出**這個模型的真正最佳解**,
拿它當天花板,才知道 MPC 差在哪、以及規則版離最佳還有多遠。

另一個從 benchmark 直接得到、不需要 MPC 的發現:**感應控制比規則版好 13.5%**,
而它的切換次數更少(2.33 vs 3.13)。規則版換相過於積極,沒有等綠燈側真的放完。
這是規則版本身可以馬上改的方向。

## 做法

每一次決策(每 5 秒),用當下實測的排隊與到達率,把模型往前推 `horizon_sec`:
  1. 產生候選方案:未來要在第幾秒換相(t1 = 0 就是「現在換」)
     —— 只列**下一次**與**再下一次**換相時點,因為再遠的決策下一輪會重算
     —— 時點列舉完之後的尾段,一律套同一個延續策略(滿最小綠就換),
        讓每個候選的尾巴一樣真實
  2. 每個候選方案跑一次前推,累計總延滯(車·秒)
  3. 加上終端懲罰:時域結束時還在排的車,會延續到下一個時域
  4. 選總成本最低的方案;若它的第一步是「現在換」,就回 SWITCH

最小綠、最大綠、主線保護在候選方案產生時就過濾掉,不可能被最佳化繞過。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 終端懲罰:時域結束時仍在排隊的車,每一輛折算幾秒延滯。
# 取「對向最小綠 + 損失時間」的量級 —— 那是它至少還要等的時間。
# 沒有這一項,最佳化會學會「把車留到時域外」來讓帳面延滯變好看。
DEFAULT_TERMINAL_WEIGHT_SEC = 25.0


@dataclass
class MPCConfig:
    horizon_sec: float = 120.0
    dt_sec: float = 5.0
    lost_time_sec: float = 5.0
    min_green_sec: Optional[dict] = None      # {1: 10.0, 2: 20.0}
    max_green_sec: float = 100.0
    saturation_veh_per_sec: Optional[dict] = None   # {1: .., 2: ..}
    terminal_weight_sec: float = DEFAULT_TERMINAL_WEIGHT_SEC
    # 主線保護:優先相排隊達儲車上限這個比例時不可被切走
    spillback_ratio: float = 0.8

    def min_green(self, phase: int) -> float:
        return float((self.min_green_sec or {}).get(phase, 10.0))

    def sat(self, phase: int) -> float:
        return float((self.saturation_veh_per_sec or {}).get(phase, 0.5))


def _rollout(q: dict, green: int, elapsed: float, switch_at: list,
             arrival_rate, t0: float, cfg: MPCConfig) -> float:
    """把模型往前推一個時域,回傳總成本(車·秒)。

    switch_at: 這個候選方案要換相的時點(相對於現在的秒數),已排序。
    arrival_rate(t_abs, phase) -> 輛/秒。

    🛑 列舉完的時點之後,尾段一律套同一個**延續策略**(到最小綠就換),
       不是「維持到時域結束」。沒有延續策略的話,所有候選方案的尾段都在
       比「誰把綠燈拖得久」,而候選集合最多只能表達兩次換相 ——
       2026-09-05 實測:MPC 因此只切 1.13 次/分(規則版 3.42),
       延滯反而高 113%。尾段策略讓每個候選的尾巴一樣真實,
       決策才只反映「接下來這一兩次要不要換」的差別。
    """
    q = dict(q)
    lost_remaining = 0.0
    cost = 0.0
    t = 0.0
    nxt = list(switch_at)
    while t < cfg.horizon_sec:
        dt = cfg.dt_sec
        t_start = t          # 這一步開始的時刻,判定換相時點要用它(見下方註解)
        for p in (1, 2):
            q[p] += float(arrival_rate(t0 + t, p) or 0.0) * dt
        eff = max(0.0, dt - lost_remaining)
        lost_remaining = max(0.0, lost_remaining - dt)
        if eff > 0:
            q[green] = max(0.0, q[green] - cfg.sat(green) * eff)
        cost += (q[1] + q[2]) * dt          # 延滯 = 排隊車數 × 時間
        t += dt
        elapsed += dt
        if nxt:
            # 🛑 用「這一步開始的時刻」比,不是遞增後的時刻。用遞增後的時刻的話
            #    t1=0 與 t1=5 會算出完全一樣的軌跡(兩者都在第一步結束時換相),
            #    等於每次換相都比計畫晚一步 —— 2026-09-05 實測:切換頻率被壓到
            #    2.37 次/分(規則版 3.13),延滯高 26%。
            do_switch = t_start >= nxt[0]
            if do_switch:
                nxt.pop(0)
        else:
            # 尾段延續策略:滿最小綠就換(中性、可行、對所有候選一致)
            do_switch = elapsed >= cfg.min_green(green)
        # 最大綠在 rollout 內也強制,候選方案繞不過安全約束
        if elapsed >= cfg.max_green_sec:
            do_switch = True
        if do_switch:
            green = 2 if green == 1 else 1
            elapsed = 0.0
            lost_remaining = cfg.lost_time_sec
    # 終端懲罰:時域結束時還在排的車,會延續到下一個時域
    cost += (q[1] + q[2]) * cfg.terminal_weight_sec
    return cost


def _candidates(green: int, elapsed: float, cfg: MPCConfig) -> list:
    """產生候選換相時點。安全約束在這裡就過濾掉,最佳化繞不過去。

    只列下一次與再下一次換相:更遠的決策下一輪(5 秒後)會重新規劃。
    """
    dt = cfg.dt_sec
    H = cfg.horizon_sec
    min_g = cfg.min_green(green)
    other = 2 if green == 1 else 1
    min_o = cfg.min_green(other)

    # 第一次換相最早可以在哪:要滿足目前這一相的最小綠
    first_earliest = max(0.0, min_g - elapsed)
    # 最晚:最大綠到頂就一定得換
    first_latest = max(first_earliest, cfg.max_green_sec - elapsed)

    # 🛑 不要放「空方案」(接下來不主動換,全交給尾段策略)。它在前推時其實會
    #    照尾段策略很快換相,但 decide 會把它讀成 KEEP —— 計畫說要切、動作卻
    #    不切,下一步再重算又是同一個結論,換相被無限延後。
    #    2026-09-05 實測:留著它 MPC 比規則版差 60%,拿掉後才正常。
    #    列舉的 t1 已經涵蓋「現在就換(t1=0)」,空方案本來就是多餘的。
    out = []
    t1 = first_earliest
    while t1 <= min(first_latest, H):
        # 第二次換相要滿足對向的最小綠
        opts = [[t1]]
        t2 = t1 + max(min_o, dt)
        while t2 <= H:
            opts.append([t1, t2])
            t2 += dt
        out.extend(opts)
        t1 += dt
    return out or [[]]


def decide_mpc(*, green_phase: int, green_elapsed_sec: float,
               queue_veh: dict, arrival_rate, cfg: MPCConfig,
               t_abs: float = 0.0,
               storage_veh: Optional[dict] = None,
               priority_phase: Optional[int] = None) -> dict:
    """算出這一刻該 KEEP 還是 SWITCH,並附上每個候選方案的成本(可稽核)。

    queue_veh: {1: 輛, 2: 輛} 當下實測排隊換算的車數
    storage_veh / priority_phase: 主線保護用(可省略)
    """
    g = green_phase
    # ① 最小綠未滿 → 一律不可切(與規則版一致)
    if green_elapsed_sec < cfg.min_green(g):
        return {"action": "KEEP", "reason": f"未滿最小綠 {cfg.min_green(g):.0f}s",
                "forced": True, "candidates": []}
    # ② 最大綠到頂 → 強制切
    if green_elapsed_sec >= cfg.max_green_sec:
        return {"action": "SWITCH", "reason": f"已達最大綠 {cfg.max_green_sec:.0f}s",
                "forced": True, "candidates": []}
    # ③ 主線保護:優先相排隊逼近儲車上限 → 不可切走
    if priority_phase is not None and g == priority_phase and storage_veh:
        cap = float(storage_veh.get(g) or 0.0)
        if cap > 0 and float(queue_veh.get(g) or 0.0) / cap >= cfg.spillback_ratio:
            return {"action": "KEEP", "reason": "主線保護:排隊逼近儲車上限",
                    "forced": True, "candidates": []}

    cands = _candidates(g, green_elapsed_sec, cfg)
    scored = []
    for c in cands:
        cost = _rollout(queue_veh, g, green_elapsed_sec, c,
                        arrival_rate, t_abs, cfg)
        scored.append({"switch_at": c, "cost": round(cost, 1)})
    scored.sort(key=lambda x: x["cost"])
    best = scored[0]
    # 最佳方案的第一次換相若落在下一個決策點之前,就是「現在換」
    switch_now = bool(best["switch_at"]) and best["switch_at"][0] < cfg.dt_sec
    keep_best = next((s for s in scored
                      if not (s["switch_at"] and s["switch_at"][0] < cfg.dt_sec)), None)
    margin = round(((keep_best or best)["cost"] - best["cost"]), 1)
    return {
        "action": "SWITCH" if switch_now else "KEEP",
        "reason": (f"最佳方案:{'現在換相' if switch_now else '續綠'},"
                   f"時域 {cfg.horizon_sec:.0f}s 總成本 {best['cost']:.0f} 車·秒"
                   f"(次佳差 {margin:.0f})"),
        "forced": False,
        "best": best,
        "margin": margin,
        "candidates": scored[:8],       # 只留前 8 名,夠稽核又不會灌爆回應
        "n_candidates": len(scored),
    }


def mpc_switch_fn(cfg: MPCConfig, arrival_rate,
                  storage_veh: Optional[dict] = None,
                  priority_phase: Optional[int] = None):
    """包成 signal_sim.simulate 用的 switch_fn,讓 MPC 能進 benchmark 對照。"""
    def switch_fn(state) -> bool:
        d = decide_mpc(green_phase=state["green_phase"],
                       green_elapsed_sec=state["green_elapsed"],
                       queue_veh=state["queue_veh"],
                       arrival_rate=arrival_rate, cfg=cfg,
                       t_abs=state["t"],
                       storage_veh=storage_veh, priority_phase=priority_phase)
        return d["action"] == "SWITCH"
    return switch_fn
