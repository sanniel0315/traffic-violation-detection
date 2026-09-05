"""號誌成效評估:工程報告 / 技術報告 / 完整報告 三個等級的指標。

使用者定義的指標規格:
  最低要求(工程報告):平均延滯、平均排隊長度、每小時通過車輛數
  標準組合(技術報告):+ 平均旅行時間、平均停車次數、95 分位延滯
  完整組合:核心 6 項 + 進階 7 項 + t-test/p-value + Cohen's d + 分時段(尖峰 vs 離峰)

🛑 每個指標都帶 method 欄位:measured(直接量到)/ approx(由可量到的東西推算)/
   unavailable(現場沒有這種資料)。報告不可以把近似值寫成實測值。
   現場能真量的只有:通過車數(traffic_events)、排隊長度與停等車數
   (congestion_samples 每 5 秒)、燈態秒數(控制器 5F03 框)。
   逐車的延滯、旅行時間、停車次數現場沒有軌跡層級紀錄,只能推算。

統計單位是「一個號誌週期」(由控制器 5F03 重建的綠燈段),兩個時段的比較
以週期為樣本做 Welch t-test 與 Cohen's d。用 5 秒取樣當樣本會嚴重高估自由度
(同一週期內的取樣不獨立),p 值會假小。
"""
from __future__ import annotations

import math
import sqlite3
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from typing import Optional

UTC_OFFSET_H = 8            # congestion_samples / traffic_events 存 UTC,號誌側存本地
MPV = 7.0                   # 每車佔用長度(公尺),與決策引擎同值
LOCAL_SAMPLE_SEC = 5.0      # congestion 取樣週期


def _to_utc(local_iso: str) -> str:
    return (datetime.fromisoformat(local_iso) - timedelta(hours=UTC_OFFSET_H)).strftime("%Y-%m-%d %H:%M:%S")


def _local_ts(utc_str: str) -> float:
    return (datetime.strptime(utc_str[:19], "%Y-%m-%d %H:%M:%S") + timedelta(hours=UTC_OFFSET_H)).timestamp()


# ── 統計 ─────────────────────────────────────────────────────────────
def _mean(v):
    return sum(v) / len(v) if v else None


def _pctl(v, p):
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _var(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def welch_t(a: list, b: list) -> dict:
    """Welch t-test(不假設同變異)+ Cohen's d(pooled SD)。scipy 有就用,沒有就手算 p。"""
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "p": None, "df": None, "cohen_d": None, "n_a": len(a), "n_b": len(b),
                "note": "樣本不足(每組至少 2 個週期)"}
    ma, mb = _mean(a), _mean(b)
    va, vb = _var(a), _var(b)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        # 兩組各自都是常數:t 無定義。差值照給,但不能宣稱顯著 —— 零變異多半是
        # 資料太少或量測值被夾住(例如排隊一直是 0),要在報告裡講清楚。
        return {"t": None, "p": None, "df": None, "cohen_d": None, "n_a": na, "n_b": nb,
                "mean_a": round(ma, 3), "mean_b": round(mb, 3), "diff": round(mb - ma, 3),
                "note": "兩組零變異,無法檢定"}
    t = (mb - ma) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = None
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:
        # 常態近似(df 大時可用;小樣本時標註)
        p = float(math.erfc(abs(t) / math.sqrt(2)))
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) if (na + nb - 2) > 0 else 0.0
    d = (mb - ma) / sp if sp > 0 else 0.0
    mag = "negligible" if abs(d) < 0.2 else ("small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large"))
    return {"t": round(t, 3), "p": round(p, 4), "df": round(df, 1), "cohen_d": round(d, 3),
            "effect": mag, "n_a": na, "n_b": nb, "mean_a": round(ma, 3), "mean_b": round(mb, 3),
            "diff": round(mb - ma, 3)}


# ── 資料讀取 ─────────────────────────────────────────────────────────
def load_congestion(db_path: str, cams: list, since_local: str, until_local: str) -> dict:
    """{cam: [(local_ts, stopped, vehicles, queue_m)]},每台依時間排序。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    out = {c: [] for c in cams}
    q = ("SELECT camera_id, created_at, stopped_vehicle_count, vehicle_count, estimated_queue_length_m "
         "FROM congestion_samples WHERE is_overall=1 AND camera_id IN (%s) AND created_at>=? AND created_at<? "
         "ORDER BY created_at" % ",".join("?" * len(cams)))
    for cam, ca, st, vc, qm in conn.execute(q, (*cams, _to_utc(since_local), _to_utc(until_local))):
        try:
            out[cam].append((_local_ts(ca), float(st or 0), float(vc or 0), float(qm or 0)))
        except Exception:
            continue
    conn.close()
    return out


def load_passes(db_path: str, cams: list, since_local: str, until_local: str) -> dict:
    """{cam: [(local_ts, speed_kmh|None)]} —— traffic_events 每筆一輛通過。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    out = {c: [] for c in cams}
    q = ("SELECT camera_id, created_at, speed_kmh FROM traffic_events WHERE camera_id IN (%s) "
         "AND created_at>=? AND created_at<? ORDER BY created_at" % ",".join("?" * len(cams)))
    for cam, ca, sp in conn.execute(q, (*cams, _to_utc(since_local), _to_utc(until_local))):
        try:
            out[cam].append((_local_ts(ca), float(sp) if sp and sp > 0 else None))
        except Exception:
            continue
    conn.close()
    return out


# ── 指標 ─────────────────────────────────────────────────────────────
def per_cycle_metrics(cycles: list, cong: list, passes: list, storage_m: Optional[float],
                      approach_len_m: Optional[float]) -> list:
    """一個進場道、逐週期的指標。

    cycles: [{phase, start, green_end, end, green_sec}](該相的綠燈段,epoch)
            一個「週期」= 這一相從綠燈開始到下一次綠燈開始。
    cong:   [(ts, stopped, vehicles, queue_m)] 停止線相機
    passes: [(ts, speed)] 停止線相機通過事件
    """
    ct = [c[0] for c in cong]
    pt = [p[0] for p in passes]
    out = []
    for i, cy in enumerate(cycles):
        t0 = cy["start"]
        t1 = cycles[i + 1]["start"] if i + 1 < len(cycles) else cy["end"]
        if t1 - t0 < 10:
            continue
        a, b = bisect_left(ct, t0), bisect_left(ct, t1)
        seg = cong[a:b]
        pa, pb = bisect_left(pt, t0), bisect_left(pt, t1)
        pas = passes[pa:pb]
        n_pass = len(pas)
        # 停等延滯(車·秒):每筆取樣的停等車數 × 取樣間隔
        stopped_vs = 0.0
        present_vs = 0.0
        for k in range(len(seg)):
            dt = (seg[k + 1][0] - seg[k][0]) if k + 1 < len(seg) else LOCAL_SAMPLE_SEC
            dt = min(max(dt, 0.0), LOCAL_SAMPLE_SEC * 3)
            stopped_vs += seg[k][1] * dt
            present_vs += seg[k][2] * dt
        queues = [s[3] for s in seg]
        speeds = [p[1] for p in pas if p[1]]
        # 綠燈利用率:綠燈期間停止線有車在場的時間比例。
        # 🛑 第一版用「每 2 秒放一台為滿載」,現場通過量 1200~1600 輛/h 一律算到 0.99,
        #    沒有鑑別力。有車比例才看得出「綠燈亮著卻沒車可放」。
        ga, gb = bisect_left(ct, t0), bisect_left(ct, cy["green_end"])
        gseg = cong[ga:gb]
        g_present = 0.0
        g_total = 0.0
        for k in range(len(gseg)):
            dt = (gseg[k + 1][0] - gseg[k][0]) if k + 1 < len(gseg) else LOCAL_SAMPLE_SEC
            dt = min(max(dt, 0.0), LOCAL_SAMPLE_SEC * 3)
            g_total += dt
            if gseg[k][2] > 0:
                g_present += dt
        green_util = (g_present / g_total) if g_total > 0 else None
        delay_per_veh = (stopped_vs / n_pass) if n_pass else None
        # 停車率:到達的車裡有多少遇到隊伍/紅燈 —— 用「停等車·秒 / 在場車·秒」近似
        stop_ratio = (stopped_vs / present_vs) if present_vs > 0 else None
        travel = None
        if approach_len_m and speeds:
            v = _mean(speeds) / 3.6
            if v > 0.5:
                travel = approach_len_m / v + (delay_per_veh or 0.0)
        spill = bool(storage_m and queues and max(queues) >= 0.8 * storage_m)
        out.append({
            "start": t0, "cycle_sec": round(t1 - t0, 1), "green_sec": cy["green_sec"],
            "passes": n_pass,
            "throughput_vph": round(n_pass * 3600.0 / (t1 - t0), 1),
            "delay_veh_sec": round(stopped_vs, 1),
            "delay_per_veh": None if delay_per_veh is None else round(delay_per_veh, 2),
            "queue_avg_m": None if not queues else round(_mean(queues), 2),
            "queue_max_m": None if not queues else round(max(queues), 1),
            "stop_ratio": None if stop_ratio is None else round(min(stop_ratio, 1.0), 3),
            "travel_sec": None if travel is None else round(travel, 2),
            "speed_avg_kmh": None if not speeds else round(_mean(speeds), 1),
            "spillback": spill,
            "green_util": None if green_util is None else round(green_util, 3),
        })
    return out


def summarize_cycles(rows: list, tier: str = "full") -> dict:
    """把逐週期指標彙整成報告欄位。每欄帶 method。"""
    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]
    tp = col("throughput_vph")
    dl = col("delay_per_veh")
    core = {
        "avg_delay_sec": {"value": _r(_mean(dl)), "unit": "秒/車", "method": "approx",
                          "how": "停等車數×取樣間隔積分 ÷ 通過車數(停等延滯,不含減速延滯)", "n": len(dl)},
        "avg_queue_m": {"value": _r(_mean(col("queue_avg_m"))), "unit": "公尺", "method": "measured",
                        "how": "停止線相機 5 秒取樣的排隊長度平均", "n": len(col("queue_avg_m"))},
        "throughput_vph": {"value": _r(_mean(tp)), "unit": "輛/小時", "method": "measured",
                           "how": "停止線相機通過事件(traffic_events)換算", "n": len(tp)},
    }
    if tier in ("standard", "full"):
        core.update({
            "avg_travel_sec": {"value": _r(_mean(col("travel_sec"))), "unit": "秒", "method": "approx",
                               "how": "進場道長度 ÷ 區間平均車速 + 停等延滯;LPR 跨相機再辨識樣本太少(3h 僅 3 組)未採用",
                               "n": len(col("travel_sec")), "confidence": "low"},
            "avg_stops": {"value": _r(_mean(col("stop_ratio"))), "unit": "次/車(0~1)", "method": "approx",
                          "how": "停等車·秒 ÷ 在場車·秒;現場無逐車軌跡的停車計數", "n": len(col("stop_ratio"))},
            "delay_p95_sec": {"value": _r(_pctl(dl, 0.95)), "unit": "秒/車", "method": "approx",
                              "how": "逐週期平均延滯的 95 分位(週期層級,不是逐車)", "n": len(dl)},
        })
    adv = {}
    if tier == "full":
        adv = {
            "queue_max_m": {"value": _r(max(col("queue_max_m")) if col("queue_max_m") else None), "unit": "公尺", "method": "measured"},
            "spillback_cycles": {"value": sum(1 for r in rows if r.get("spillback")), "unit": "週期", "method": "measured",
                                 "how": "該週期最大排隊 ≥ 儲車上限 80%"},
            "cycle_sec_avg": {"value": _r(_mean(col("cycle_sec"))), "unit": "秒", "method": "measured"},
            "cycle_sec_std": {"value": _r(math.sqrt(_var(col("cycle_sec")))) if len(col("cycle_sec")) > 1 else None, "unit": "秒", "method": "measured"},
            "green_sec_avg": {"value": _r(_mean(col("green_sec"))), "unit": "秒", "method": "measured", "how": "控制器 5F03 秒數"},
            "green_util": {"value": _r(_mean(col("green_util"))), "unit": "0~1", "method": "measured",
                           "how": "綠燈期間停止線有車在場的時間比例(5 秒取樣)"},
            "speed_avg_kmh": {"value": _r(_mean(col("speed_avg_kmh"))), "unit": "km/h", "method": "measured", "n": len(col("speed_avg_kmh"))},
        }
    return {"cycles": len(rows), "core": core, "advanced": adv}


def compare(rows_a: list, rows_b: list, keys=("delay_per_veh", "queue_avg_m", "throughput_vph",
                                              "travel_sec", "stop_ratio", "queue_max_m", "green_sec")) -> dict:
    out = {}
    for k in keys:
        a = [r[k] for r in rows_a if r.get(k) is not None]
        b = [r[k] for r in rows_b if r.get(k) is not None]
        out[k] = welch_t(a, b)
    return out


def _r(v, nd=2):
    return None if v is None else round(v, nd)
