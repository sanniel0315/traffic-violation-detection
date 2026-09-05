"""把 /report(成效)與 /paired(配對)的 JSON 排成 Markdown 報告。

三個等級跟使用者的規格一一對應:
  min      工程報告:平均延滯、平均排隊、每小時通過車輛數
  standard 技術報告:+ 平均旅行時間、平均停車次數、95 分位延滯
  full     完整報告:核心 6 + 進階 7 + t-test/p/Cohen's d + 尖峰/離峰 + 配對法

🛑 每個數字後面跟著 method 標記:實測 / 近似 / 近似(低信心)。這是報告的
   誠信底線 —— 現場只能真量通過車數、排隊、燈態秒數,其餘是推算。
純函式,不碰網路與 DB;抓資料在 scripts/signal_report.py。
"""
from __future__ import annotations

from datetime import datetime

METHOD_TAG = {"measured": "實測", "approx": "近似", "unavailable": "無資料"}
CORE_LABEL = [
    ("avg_delay_sec", "平均延滯", "秒/車"),
    ("avg_queue_m", "平均排隊長度", "公尺"),
    ("throughput_vph", "每小時通過車輛數", "輛/h"),
    ("avg_travel_sec", "平均旅行時間(進場道)", "秒"),
    ("avg_stops", "平均停車次數", "次/車"),
    ("delay_p95_sec", "95 分位延滯", "秒/車"),
]
ADV_LABEL = [
    ("queue_max_m", "最大排隊長度", "公尺"),
    ("spillback_cycles", "回堵週期數", "週期"),
    ("cycle_sec_avg", "週期長度平均", "秒"),
    ("cycle_sec_std", "週期長度標準差", "秒"),
    ("green_sec_avg", "綠燈秒數平均", "秒"),
    ("green_util", "綠燈利用率", "0~1"),
    ("speed_avg_kmh", "區間車速", "km/h"),
]
AB_LABEL = {"delay_per_veh": "平均延滯", "queue_avg_m": "平均排隊", "throughput_vph": "通過車數",
            "travel_sec": "旅行時間", "stop_ratio": "停車率", "queue_max_m": "最大排隊", "green_sec": "綠燈秒數"}


def _v(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def _tag(m: dict) -> str:
    t = METHOD_TAG.get(m.get("method"), m.get("method", ""))
    if m.get("confidence") == "low":
        t += "(低信心)"
    return t


def _core_table(a: dict, b: dict | None, tier: str) -> str:
    keys = CORE_LABEL[:3] if tier == "min" else CORE_LABEL
    head = "| 指標 | A | B | 差(B−A) | 方法 |" if b else "| 指標 | 值 | 方法 |"
    sep = "|---|---|---|---|---|" if b else "|---|---|---|"
    rows = [head, sep]
    for k, label, unit in keys:
        ma = (a.get("core") or {}).get(k)
        if not ma:
            continue
        if b:
            mb = (b.get("core") or {}).get(k) or {}
            va, vb = ma.get("value"), mb.get("value")
            d = None if (va is None or vb is None) else vb - va
            rows.append(f"| {label}({unit}) | {_v(va)} | {_v(vb)} | {_v(d)} | {_tag(ma)} |")
        else:
            rows.append(f"| {label}({unit}) | {_v(ma.get('value'))} | {_tag(ma)} |")
    return "\n".join(rows)


def _adv_table(a: dict, b: dict | None) -> str:
    head = "| 進階指標 | A | B | 方法 |" if b else "| 進階指標 | 值 | 方法 |"
    sep = "|---|---|---|---|" if b else "|---|---|---|"
    rows = [head, sep]
    for k, label, unit in ADV_LABEL:
        ma = (a.get("advanced") or {}).get(k)
        if not ma:
            continue
        if b:
            mb = (b.get("advanced") or {}).get(k) or {}
            rows.append(f"| {label}({unit}) | {_v(ma.get('value'))} | {_v(mb.get('value'))} | {_tag(ma)} |")
        else:
            rows.append(f"| {label}({unit}) | {_v(ma.get('value'))} | {_tag(ma)} |")
    return "\n".join(rows)


def _ab_table(ab: dict) -> str:
    rows = ["| 指標 | A 平均 | B 平均 | 差 | t | p | Cohen's d | 效果量 | n(A/B) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, label in AB_LABEL.items():
        r = ab.get(k)
        if not r:
            continue
        rows.append(f"| {label} | {_v(r.get('mean_a'))} | {_v(r.get('mean_b'))} | {_v(r.get('diff'))} | "
                    f"{_v(r.get('t'), 3)} | {_v(r.get('p'), 4)} | {_v(r.get('cohen_d'), 3)} | "
                    f"{r.get('effect') or r.get('note') or '—'} | {r.get('n_a')}/{r.get('n_b')} |")
    return "\n".join(rows)


def _paired_block(p: dict, tag: str) -> str:
    if not p or "runs_usable" not in p:
        return f"（{tag} 無配對資料）"
    dm = p.get("delta_meaningful") or {}
    hc = p.get("hold_compare") or {}
    src = "控制器 5F03 秒數" if p.get("source") == "controller_5F03" else "影子取樣(退回)"
    return (f"| {tag} | {p['runs_usable']} | {p['earlier']}（有車 {p.get('earlier_meaningful', '—')}） | {p['same']} | "
            f"{p['hold']} | {p['later']} | {_v(dm.get('avg'), 1)} | {_v(p.get('waste_sec_total'), 0)} | "
            f"{_v((hc.get('margin_at_switch') or {}).get('avg'), 1)} | "
            f"{_v(100 * (hc.get('red_waiting_ratio') or 0), 0)}% | {src} |")


def _tdx_block(t: dict | None) -> str:
    if not t or not t.get("n"):
        return "尚無 TDX 資料（未設金鑰或該時段未抓到）。"
    rows = ["| 配對 | 方向 | 筆數 | 旅行時間(秒) | 車速(km/h) | 車數 |", "|---|---|---|---|---|---|"]
    for q in t.get("pairs", []):
        rows.append(f"| {q['pair_id']} | {q.get('direction') or '—'} | {q['n']} | {_v(q.get('travel_time_sec'), 1)} | "
                    f"{_v(q.get('speed_kmh'), 1)} | {_v(q.get('vehicles'))} |")
    rows.append(f"\n平均旅行時間 **{_v(t.get('avg_travel_time_sec'), 1)} 秒**（{t.get('source')}，實測）")
    return "\n".join(rows)


def render(report: dict, paired_a: dict | None = None, paired_b: dict | None = None,
           meta: dict | None = None) -> str:
    meta = meta or {}
    tier = report.get("tier", "full")
    tier_name = {"min": "工程報告（最低要求）", "standard": "技術報告（標準組合）", "full": "完整報告"}[tier]
    a, b = report.get("a") or {}, report.get("b")
    site = meta.get("site", "國道 8 號新市交流道 0xFFFF")
    lines = [f"# 動態號誌成效{tier_name}", "",
             f"- 站點：{site}",
             f"- A：{a.get('since')} ～ {a.get('until')}（{meta.get('a_label', '基準')}）"]
    if b:
        lines.append(f"- B：{b.get('since')} ～ {b.get('until')}（{meta.get('b_label', '對照')}）")
    lines += [f"- 統計單位：號誌週期（控制器 5F03 重建）；A {a.get('all', {}).get('cycles', '—')} 週期"
              + (f"、B {b.get('all', {}).get('cycles', '—')} 週期" if b else ""),
              f"- 產出：{meta.get('generated_at') or datetime.now().isoformat(timespec='minutes')}",
              "", "> 方法標記：**實測** = 直接量到（通過車數、排隊、燈態秒數）；**近似** = 由可量到的東西推算。"
              "報告不把近似寫成實測。", ""]
    lines += ["## 一、核心指標", "", _core_table(a.get("all", {}), (b or {}).get("all") if b else None, tier), ""]
    if tier != "min":
        lines += ["### 分相拆解", ""]
        for ph in ("1", "2"):
            pa = (a.get("by_phase") or {}).get(ph, {})
            pb = ((b or {}).get("by_phase") or {}).get(ph) if b else None
            lines += [f"**分相 {ph}**（{'上匝道' if ph == '1' else '下匝道'}）", "",
                      _core_table(pa, pb, tier), ""]
        lines += ["### 平均旅行時間（國道主線，TDX eTag 實測）", "",
                  "A：", _tdx_block(a.get("travel_time_tdx")), ""]
        if b:
            lines += ["B：", _tdx_block(b.get("travel_time_tdx")), ""]
        lines += ["> 進場道旅行時間與主線旅行時間量的是不同路段，分開呈現，不合併。", ""]
    if tier == "full":
        lines += ["## 二、進階指標", "", _adv_table(a.get("all", {}), (b or {}).get("all") if b else None), ""]
        if b and report.get("ab_test"):
            lines += ["## 三、統計檢定（Welch t-test，樣本 = 號誌週期）", "",
                      _ab_table(report["ab_test"].get("all", {})), "",
                      "> Cohen's d：<0.2 negligible、<0.5 small、<0.8 medium、≥0.8 large。"
                      "延滯／旅行時間／停車率為近似指標，其檢定結果的解讀強度低於排隊與通過車數。", ""]
        pk, off = a.get("peak"), a.get("offpeak")
        if pk or off:
            lines += ["## 四、分時段（尖峰 09–12、17–20 vs 離峰）", "",
                      f"A 尖峰 {pk.get('cycles', 0) if pk else 0} 週期／離峰 {off.get('cycles', 0) if off else 0} 週期", ""]
            if pk and off and pk.get("cycles") and off.get("cycles"):
                lines += [_core_table(off, pk, "standard").replace("| A |", "| 離峰 |").replace("| B |", "| 尖峰 |"), ""]
        lines += ["## 五、決策配對（逐次綠燈：我方會早幾秒切）", "",
                  "| 時段 | 綠燈段 | 早切 | 同時 | 續綠 | 晚切 | 早切平均Δ(秒) | 有代價空放(秒) | 續綠裕度 | 續綠時紅側有車 | 秒數來源 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|",
                  _paired_block(paired_a, "A")]
        if paired_b:
            lines.append(_paired_block(paired_b, "B"))
        lines += ["", "> Δ 為負代表我方會比現行控制早切；「有代價空放」只計紅側真的有車在等的早切。"
                  "晚切必須為 0 —— 我方不會比現行控制更慢放人。", ""]
    return "\n".join(lines)
