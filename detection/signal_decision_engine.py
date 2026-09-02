# -*- coding: utf-8 -*-
"""動態號誌影子決策引擎 —— 算出「如果是我，我會怎麼切」，**不下發**。

用途（bypass OPAC 的前置驗證）：
    OPAC(中心端適應性控制)正在控制路口時，我方同步用同一份輸入算出自己的決策，
    與 OPAC 的實際決策逐筆比對一致率。一致率夠高、差異點都能解釋，
    才有底氣接手控制權（見 docs/OPAC控制融入我方系統_規劃.md 的 L4→L5）。

🛑 這個模組**只算不送**。任何下發都要走 signal_tc3 的 control/prepare +
   control/send，且那條路預設關閉（SIGNAL_TC3_CONTROL）。

決策規則（2026-09-02 從 OPAC decision.log 41 筆樣本反推，零例外）：

    SWITCH ⟺ pn1 > pn2      否則 KEEP（含平手）

    pn1 = 紅燈側累積的等待需求（切換效益）
    pn2 = 綠燈側尚未消化完的放行需求（保持效益，隨綠燈以飽和流遞減）

⚠️ **本引擎目前與 OPAC 的一致率只有 42%（40 筆實測），尚不可用於接管。**
   已知落差在 pn2：

   - pn1 = 紅側 swl + fa + sa —— 27 筆命中 22 筆，方向正確、細節待補
   - pn2 **不是**綠側 swl+fa+sa（27 筆命中 0 筆），也不是「需求 − 飽和流×綠燈秒」
     （本檔初版這樣估，導致 pn2 恆為 0、全部誤判成 SWITCH）

   實測反例（綠側量沒變，pn2 卻一路遞減，且數值遠大於綠側 swl）：
       swlG=4 saG=3 el=15 → pn2=11
       swlG=3 faG=3 el=20 → pn2=9
       swlG=3 saG=2 el=25 → pn2=8
       swlG=3 faG=2 el=30 → pn2=7
       swlG=3        el=35 → pn2=5
   → pn2 是跨週期累積/消耗的**有狀態量**，無法從單筆瞬時值重建，
     需要放行歷史（OPAC 內部狀態）。這是它的本質，不是參數沒調對。

   **下一步**：不要再用瞬時值硬湊。要嘛(a)蒐集連續序列自行建立狀態遞迴模型，
   要嘛(b)請中心端提供 OPAC 演算法文件/原始碼。在一致率達標前，
   本引擎只能當「觀察與比對工具」，不可作為接管依據。

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


@dataclass
class ApproachState:
    """單一方向(分相)的即時狀態。"""
    phase_no: int                    # 分相編號(1=上匝道 2=下匝道)
    queue_m: Optional[float] = None  # 排隊長度(公尺)，來自我方壅塞偵測
    arrivals: float = 0.0            # 觀測窗內到達車數
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
    pn1: float                       # 切換效益(紅燈側需求)
    pn2: float                       # 保持效益(綠燈側殘量)
    reason: str                      # 人看得懂的理由
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
) -> Decision:
    """算出這一刻該 KEEP 還是 SWITCH。純函式，不碰 IO、不下發。"""
    mpv = meters_per_vehicle
    # pn1:紅燈側累積等待需求(排隊 + 觀測窗內到達)
    pn1 = red_side.queue_vehicles(mpv) + max(0.0, red_side.arrivals)
    # pn2:綠燈側需求扣掉綠燈期間已放行的量(飽和流 × 已亮綠秒數)
    demand = green_side.queue_vehicles(mpv) + max(0.0, green_side.arrivals)
    discharged = (saturation_vph / 3600.0) * max(0.0, green_elapsed_sec)
    pn2 = max(0.0, demand - discharged)

    d = Decision(action="KEEP", pn1=round(pn1, 3), pn2=round(pn2, 3),
                 reason="", green_phase=green_phase,
                 green_elapsed=green_elapsed_sec,
                 detail={"demand_green": round(demand, 3),
                         "discharged": round(discharged, 3)})

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

    # ④ 一般規則:紅側需求 > 綠側殘量 → 切
    if pn1 > pn2:
        d.action = "SWITCH"
        d.reason = f"紅側需求 {pn1:.1f} > 綠側殘量 {pn2:.1f}"
    else:
        d.action = "KEEP"
        d.reason = f"綠側殘量 {pn2:.1f} ≥ 紅側需求 {pn1:.1f},續綠"
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
        "pn1": ours.pn1,
        "pn2": ours.pn2,
        "reason": ours.reason,
    }
