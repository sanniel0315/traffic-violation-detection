# -*- coding: utf-8 -*-
"""動態號誌影子決策引擎 —— 算出「如果是我，我會怎麼切」，**不下發**。

用途（bypass OPAC 的前置驗證）：
    OPAC(中心端適應性控制)正在控制路口時，我方同步用同一份輸入算出自己的決策，
    與 OPAC 的實際決策逐筆比對一致率。一致率夠高、差異點都能解釋，
    才有底氣接手控制權（見 docs/OPAC控制融入我方系統_規劃.md 的 L4→L5）。

🛑 這個模組**只算不送**。任何下發都要走 signal_tc3 的 control/prepare +
   control/send，且那條路預設關閉（SIGNAL_TC3_CONTROL）。

## 決策法則:我方自己的延滯成本(不複製 OPAC)

沿用 OPAC 已驗證的**規則骨架**(比較兩側成本、大的一方切、min/max green、
主線保護),但**成本函數換成我方自己定義的「車輛等待時間」** ——
這是交通工程的標準目標(最小化總延滯),而且每一項我方都量得到、語意清楚。

    切換效益 = 紅側繼續等下去的成本   switch_gain
    保持效益 = 綠側還能消化的價值     keep_gain
    SWITCH ⟺ switch_gain > keep_gain + 換相成本

    switch_gain = 紅側車數 × 已等待秒數        (等越久、車越多 → 越該切)
    keep_gain   = 綠側剩餘可放行車數 × 單位放行價值
    換相成本    = 換相損失時間 × 飽和流         (切一次要付的代價)

🛑 為什麼不複製 OPAC 的 pn1/pn2:2026-09-02 用 400 筆連續序列測過 15 種假設,
   全部失敗(最佳 58%)。pn1/pn2 是 OPAC 內部有界狀態機的耦合計數器,
   無法從外部欄位重建 —— 缺的是狀態機轉移規則,只在其原始碼裡。
   我方有自己的排隊量測(而且比 OPAC 拿到的更完整),用自己的成本函數更合理。

## 與 OPAC 的比較方式:比「結果好壞」不比「逐筆一致」

逐筆一致率沒有意義(不同成本函數本來就會有不同決策)。要比的是成效指標:
總延滯、平均排隊長度、通過量、主線回堵次數。見 evaluate_outcome()。

安全約束（與官方時制表對齊，見 signal_timing_lookup）：
    - min_green 未滿一律 KEEP（不可提早切）
    - max_green 到頂強制 SWITCH（forced）
    - 主線保護：分相2(下匝道)排隊逼近儲車上限時不可被切走
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# 飽和流率(輛/小時)→ 每秒可放行輛數。與 OPAC constant.yaml 的 1800 vph 對齊。
DEFAULT_SATURATION_VPH = 1800.0
# 公尺→輛換算(停等每輛佔用長度)。與 OPAC 的 meters-per-vehicle 對齊。
DEFAULT_METERS_PER_VEHICLE = 7.0
# 主線保護:下匝道排隊達儲車上限這個比例時，視為逼近回堵主線。
DEFAULT_SPILLBACK_RATIO = 0.8
# 換相損失時間(秒)。與官方時制表 ramp_timing_baseline.json 的
# lost_time_per_change_sec 對齊(現場值 5)。切一次相位要付的代價。
DEFAULT_LOST_TIME_SEC = 5.0


@dataclass
class ApproachState:
    """單一方向(分相)的即時狀態。"""
    phase_no: int                    # 分相編號(1=上匝道 2=下匝道)
    queue_m: Optional[float] = None  # 排隊長度(公尺)，來自我方壅塞偵測
    arrivals: float = 0.0            # 觀測窗內到達車數
    waiting_sec: float = 0.0         # 這一側已經紅燈等待幾秒(紅側才有意義)
    storage_m: Optional[int] = None  # 該匝道儲車上限(公尺)
    priority: bool = False           # 是否主線保護優先

    def queue_vehicles(self, meters_per_vehicle: float) -> float:
        """排隊公尺換算成輛。queue_m 為 None(未量到)時以 0 計。"""
        if not self.queue_m or meters_per_vehicle <= 0:
            return 0.0
        return self.queue_m / meters_per_vehicle

    def spillback_ratio(self) -> Optional[float]:
        """排隊佔儲車上限的比例。無資料回 None。"""
        if not self.queue_m or not self.storage_m:
            return None
        return self.queue_m / float(self.storage_m)


@dataclass
class Decision:
    action: str                      # "KEEP" / "SWITCH"
    switch_gain: float               # 切換效益:紅側繼續等下去的成本(車·秒)
    keep_gain: float                 # 保持效益:綠側還能消化的價值(車·秒)
    reason: str                      # 人看得懂的理由
    change_cost: float = 0.0         # 換相成本(車·秒),切一次要付的代價
    forced_by_max_green: bool = False
    blocked_by_priority: bool = False # 因主線保護而不切
    green_phase: Optional[int] = None
    green_elapsed: float = 0.0
    detail: dict = field(default_factory=dict)


def decide(
    *,
    green_phase: int,
    green_elapsed_sec: float,
    green_side: ApproachState,
    red_side: ApproachState,
    min_green_sec: float,
    max_green_sec: float,
    saturation_vph: float = DEFAULT_SATURATION_VPH,
    meters_per_vehicle: float = DEFAULT_METERS_PER_VEHICLE,
    spillback_ratio: float = DEFAULT_SPILLBACK_RATIO,
    lost_time_sec: float = DEFAULT_LOST_TIME_SEC,
) -> Decision:
    """算出這一刻該 KEEP 還是 SWITCH。純函式，不碰 IO、不下發。"""
    mpv = meters_per_vehicle
    sat_per_sec = saturation_vph / 3600.0      # 每秒可放行輛數

    # 切換效益 = 紅側車數 × 已等待秒數(車·秒延滯)
    #   等越久、車越多 → 繼續紅燈的代價越高 → 越該切
    #   紅側等待秒數若無資料,退回用綠燈已亮秒數(對向紅燈至少亮了這麼久)
    red_veh = red_side.queue_vehicles(mpv) + max(0.0, red_side.arrivals)
    red_wait = red_side.waiting_sec if red_side.waiting_sec > 0 else green_elapsed_sec
    switch_gain = red_veh * max(0.0, red_wait)

    # 保持效益 = 綠側「當下仍在排隊」的車數 × 若被迫停下要多等的時間
    #   🛑 不要用「需求 − 飽和流×綠燈秒」去推剩餘 —— 飽和流是「車連續通過時」
    #      的速率,不能假設綠燈全程滿載放行。這樣推會讓長綠燈的 keep_gain 恆為 0,
    #      造成「明明還在排隊卻判該切」(2026-09-02 實測:下匝道 77m/11台 排隊、
    #      綠燈 60 秒,被算成剩餘 0 → 誤判 SWITCH)。
    #      queue_m 是**當下實測的排隊長度**,本身就已經反映「還沒走完的車」,
    #      直接用它即可,不需要再自行扣除。
    #   單位必須與 switch_gain 對等(都是 車·秒),否則等待時間一長,
    #   紅側必然壓過綠側,不管綠側多塞(2026-09-02 實測:綠側11台排隊 vs
    #   紅側2台等60秒 → 誤判 SWITCH)。
    #   綠側被切走的代價 = 這些車要多等「換相損失 + 對向一輪綠燈」才輪得到。
    green_remain = green_side.queue_vehicles(mpv)
    wait_if_switched = lost_time_sec + max(min_green_sec, 0.0)
    keep_gain = green_remain * wait_if_switched
    # 理論放行量僅供觀察(不參與決策),用來看綠燈是否給得過長
    discharged = sat_per_sec * max(0.0, green_elapsed_sec)
    green_demand = green_remain + max(0.0, green_side.arrivals)

    # 換相成本 = 換相損失時間 × 飽和流(這段時間誰都不能走)
    change_cost = lost_time_sec * sat_per_sec * lost_time_sec

    d = Decision(action="KEEP",
                 switch_gain=round(switch_gain, 2),
                 keep_gain=round(keep_gain, 2),
                 change_cost=round(change_cost, 2),
                 reason="", green_phase=green_phase,
                 green_elapsed=green_elapsed_sec,
                 detail={"red_veh": round(red_veh, 2),
                         "red_wait_sec": round(red_wait, 1),
                         "green_demand": round(green_demand, 2),
                         "green_remain": round(green_remain, 2),
                         "discharged": round(discharged, 2)})

    # ① min-green 未滿 → 一律不可切
    if green_elapsed_sec < min_green_sec:
        d.action = "KEEP"
        d.reason = f"未滿最小綠 {min_green_sec:.0f}s(已亮 {green_elapsed_sec:.0f}s)"
        return d

    # ② max-green 到頂 → 強制切(即使綠燈側還有需求)
    if green_elapsed_sec >= max_green_sec:
        d.action = "SWITCH"
        d.forced_by_max_green = True
        d.reason = f"已達最大綠 {max_green_sec:.0f}s,強制切換"
        return d

    # ③ 主線保護:綠燈側是優先相且排隊逼近儲車上限 → 不可切走
    #    (分相2 下匝道回堵會溢到國道主線,比地面道路回堵嚴重)
    gsr = green_side.spillback_ratio()
    if green_side.priority and gsr is not None and gsr >= spillback_ratio:
        d.action = "KEEP"
        d.blocked_by_priority = True
        d.reason = (f"主線保護:分相{green_side.phase_no}排隊已達儲車上限 "
                    f"{gsr*100:.0f}%,不切走")
        return d

    # ④ 一般規則:切換效益 > 保持效益 + 換相成本 → 才值得切
    #    加上換相成本是防抖動:小幅優勢不值得付一次換相的代價
    threshold = keep_gain + change_cost
    if switch_gain > threshold:
        d.action = "SWITCH"
        d.reason = (f"紅側延滯 {switch_gain:.0f} > 綠側價值 {keep_gain:.0f}"
                    f" + 換相成本 {change_cost:.0f}")
    else:
        d.action = "KEEP"
        d.reason = (f"紅側延滯 {switch_gain:.0f} ≤ 綠側價值 {keep_gain:.0f}"
                    f" + 換相成本 {change_cost:.0f},續綠")
    return d


def compare(ours: Decision, theirs_action: Optional[str]) -> dict:
    """把我方影子決策與 OPAC 實際決策比對。theirs_action 為 None 表無對照。"""
    if theirs_action is None:
        return {"match": None, "ours": ours.action, "theirs": None}
    same = (ours.action == theirs_action)
    return {
        "match": same,
        "ours": ours.action,
        "theirs": theirs_action,
        "switch_gain": ours.switch_gain,
        "keep_gain": ours.keep_gain,
        "reason": ours.reason,
    }


# ── 成效評估:比「結果好壞」而不是「逐筆一致」 ──────────────────────
# 不同成本函數本來就會給出不同決策,逐筆一致率沒有意義。
# 要回答的是:哪一套控制讓路口整體表現更好?

def evaluate_outcome(samples: list) -> dict:
    """把一段時間的觀測樣本算成成效指標,用來比較兩套控制的好壞。

    samples: [{"queue_m_1":.., "queue_m_2":.., "storage_1":.., "storage_2":..,
               "interval_sec":.., "switched": bool}, ...]
      queue_m_1/2 = 分相1(上匝道)/分相2(下匝道)當下排隊公尺
      interval_sec = 這筆樣本代表的時間長度
      switched = 這筆是否發生相位切換

    回傳指標(越小越好,除了 throughput_proxy):
      total_delay_veh_sec  總延滯(車·秒) —— 最主要的目標
      avg_queue_m_1/2      各匝道平均排隊
      max_queue_m_1/2      各匝道最大排隊
      spillback_events_2   分相2(下匝道)排隊逼近儲車上限的次數 ← 主線回堵風險
      switch_count         切換次數(太頻繁 = 浪費在換相損失)
      switch_per_min       每分鐘切換次數
    """
    if not samples:
        return {}
    mpv = DEFAULT_METERS_PER_VEHICLE
    total_delay = 0.0
    q1 = []
    q2 = []
    spill2 = 0
    switches = 0
    total_sec = 0.0
    for s in samples:
        dt = float(s.get("interval_sec") or 0)
        total_sec += dt
        a = float(s.get("queue_m_1") or 0)
        b = float(s.get("queue_m_2") or 0)
        q1.append(a)
        q2.append(b)
        # 延滯 = 排隊車數 × 這段時間(車·秒)
        total_delay += ((a + b) / mpv) * dt
        st2 = s.get("storage_2")
        if st2 and b / float(st2) >= DEFAULT_SPILLBACK_RATIO:
            spill2 += 1
        if s.get("switched"):
            switches += 1
    n = len(samples)
    return {
        "samples": n,
        "duration_sec": round(total_sec, 1),
        "total_delay_veh_sec": round(total_delay, 1),
        "avg_queue_m_1": round(sum(q1) / n, 1),
        "avg_queue_m_2": round(sum(q2) / n, 1),
        "max_queue_m_1": round(max(q1), 1),
        "max_queue_m_2": round(max(q2), 1),
        "spillback_events_2": spill2,
        "switch_count": switches,
        "switch_per_min": round(switches / (total_sec / 60.0), 2) if total_sec else 0,
    }
