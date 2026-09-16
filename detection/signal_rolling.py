"""滾動時程決策:用校準過的模型看未來一個週期,選總延滯最小的切換時機。

🛑 為什麼要有這支:現行 decide() 是**瞬時成本比較** —— 只看「此刻續綠 vs 切走」
   誰的成本低。它看不到「再等 8 秒這一波車就放完了」這種事,也看不到
   「現在切走,對向 3 秒後才有車到」。2026-09-16 的模擬對照就打在這一點上:
   我方贏固定時制 68%、贏 Webster 72%,但**輸感應控制 97%** —— 感應控制正是
   用「車走完就放掉綠燈」這個未來資訊在決策的。

做法(標準的 rolling horizon / model predictive control):
  每次決策時,對「本相再給 g 秒」逐一試 g = 0, 5, 10, ...,
  用模型往前推 horizon 秒(之後各相用「放完排隊」的啟發式接續),
  算整段的總延滯,挑最小的那個 g。g = 0 就是現在切。

🛑 這支**只算不送**。安全約束(最小綠/最大綠/主線保護)一律由呼叫端與模擬器強制,
   這裡回傳的只是「想不想切」。
"""
from __future__ import annotations

from typing import Callable, Optional

PHASES = (1, 2)


def _other(p: int) -> int:
    return 2 if p == 1 else 1


def _clear_time(q_veh: float, sat: float, rate: float, min_green: float,
                max_green: float) -> float:
    """放完這條排隊要多久(邊放邊有車來),夾在最小綠與最大綠之間。"""
    net = sat - rate
    if net <= 0:
        return max_green
    return max(min_green, min(max_green, q_veh / net))


def forward_delay(rate_fn: Callable, sat: dict, cfg, t0: float, green: int,
                  q0: dict, first_green_sec: float, horizon_sec: float,
                  dt: float = 5.0) -> float:
    """從現在起往前推 horizon 秒的總延滯(車·秒)。

    first_green_sec = 本相還要再給幾秒(候選值);之後每一相都給「放完排隊」的時間。
    """
    q = {p: float(q0.get(p) or 0.0) for p in PHASES}
    t = 0.0
    g = green
    left = float(first_green_sec)
    lost = 0.0
    delay = 0.0
    min_green = getattr(cfg, "min_green_sec", {}) or {}
    max_green = float(getattr(cfg, "max_green_sec", 100.0) or 100.0)
    lost_time = float(getattr(cfg, "lost_time_sec", 5.0) or 5.0)
    while t < horizon_sec:
        step = min(dt, horizon_sec - t)
        for p in PHASES:
            q[p] += float(rate_fn(t0 + t, p) or 0.0) * step
        eff = max(0.0, step - lost)
        lost = max(0.0, lost - step)
        if eff > 0:
            q[g] = max(0.0, q[g] - float(sat.get(g) or 0.0) * eff)
        delay += (q[1] + q[2]) * step
        t += step
        left -= step
        if left <= 0:
            g = _other(g)
            lost = lost_time
            left = _clear_time(q[g], float(sat.get(g) or 0.0),
                               float(rate_fn(t0 + t, g) or 0.0),
                               float(min_green.get(g, 10.0)), max_green)
    return delay


def plan_extra_green(rate_fn: Callable, sat: dict, cfg, t0: float, green: int,
                     q0: dict, elapsed: float, horizon_sec: float = 120.0,
                     grid_sec: float = 2.0,
                     switch_margin_veh_sec: float = 0.0) -> dict:
    """本相還該再給幾秒?回傳 {best_sec, delay_by_sec, switch_now}。

    🛑 grid_sec 預設 2 秒,不是 5 秒。切換時機的解析度直接決定成效:
       2026-09-16 實測(09-14 全天)網格 10 秒 45.8 秒/車、5 秒 33.5、2 秒 32.3;
       網格 5 秒時晨峰還會輸感應控制 8~16%,改 2 秒之後各到達率都贏。

    🛑 switch_margin_veh_sec 預設 0(不啟用)。曾經想用「要明顯比較好才切」來抑制
       切換次數,實測是**反效果**:門檻 4 車·秒就幾乎不切了(1 小時只切 36 次 =
       全部跑到最大綠),每車延滯從 15.8 秒惡化到 43.9 秒。
       真正的病因是網格太粗,不是切太多。參數留著但別再打開,除非有新證據。
    """
    min_green = float((getattr(cfg, "min_green_sec", {}) or {}).get(green, 10.0))
    max_green = float(getattr(cfg, "max_green_sec", 100.0) or 100.0)
    lo = max(0.0, min_green - elapsed)                 # 最小綠還沒滿就不能切
    hi = max(lo, max_green - elapsed)
    cands = []
    g = lo
    while g <= hi + 1e-9:
        cands.append(round(g, 1))
        g += grid_sec
    if not cands:
        cands = [lo]
    best, best_d, table = None, None, {}
    for c in cands:
        d = forward_delay(rate_fn, sat, cfg, t0, green, q0, c, horizon_sec, grid_sec)
        table[c] = round(d, 1)
        # 平手時偏向**續綠**(候選由小到大,所以要嚴格小於才換掉最佳解)
        if best_d is None or d < best_d - 1e-9:
            best, best_d = c, d
    # 立刻切的代價 vs 再給一點時間的最好結果 —— 差得夠多才值得切
    d_now = table.get(0.0)
    d_keep = min([v for k, v in table.items() if k > 0.0] or [None])
    switch_now = bool(d_now is not None and d_keep is not None
                      and d_now <= d_keep - switch_margin_veh_sec)
    if d_keep is None:                      # 沒有「再給」的選項(已到最大綠)
        switch_now = d_now is not None
    return {"best_sec": 0.0 if switch_now else (best if best else 0.0),
            "delay": round(best_d or 0.0, 1), "delay_by_sec": table,
            "switch_now": switch_now, "margin_veh_sec": switch_margin_veh_sec}


def rolling_horizon(rate_fn: Callable, sat: dict, cfg,
                    horizon_sec: float = 120.0, grid_sec: float = 2.0,
                    replan_sec: float = 5.0,
                    switch_margin_veh_sec: float = 0.0) -> Callable:
    """給模擬器用的 switch_fn:滾動時程控制。

    🛑 重新規劃的頻率(replan_sec)是效能與品質的取捨 —— 每步都重算在
       數小時的模擬裡太慢,預設每 5 秒重算一次(= 決策取樣間隔)。
    """
    box = {"t": None, "want": False}

    def switch_fn(state) -> bool:
        t = float(state["t"])
        if box["t"] is not None and t - box["t"] < replan_sec:
            return box["want"]
        box["t"] = t
        plan = plan_extra_green(rate_fn, sat, cfg, t, int(state["green_phase"]),
                                state["queue_veh"], float(state["green_elapsed"]),
                                horizon_sec, grid_sec, switch_margin_veh_sec)
        box["want"] = plan["switch_now"]
        return box["want"]

    return switch_fn
