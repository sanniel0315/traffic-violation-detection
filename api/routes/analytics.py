#!/usr/bin/env python3
"""行為優化分析 API - 從歷史 violations + traffic_events 挖洞察

5 個 endpoint:
- /hotspots          違規熱點 GROUP BY (location, hour, type) top N
- /recommendations   推薦引擎規則層 (基於熱點 + 閾值產生中文建議)
- /safety_score      各 cam 安全分數 0-100 (越高越安全)
- /flow_forecast     24h 流量歷史 + 下 1 小時預測
- /vehicle_type_violations  車種×違規 交叉統計

純讀 violations + traffic_events,不阻塞 detector pipeline。
"""
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from api.models import get_db, Violation, Camera

router = APIRouter(prefix="/api/analytics", tags=["行為優化分析"])


# ---------- 1. 違規熱點 ----------
def _camera_name_map(db: Session) -> Dict[int, str]:
    """cache cam_id → name 對應 (violations.location 多數空,fallback 用 cam name 當地點)"""
    return {c.id: c.name for c in db.query(Camera).all()}


@router.get("/hotspots")
def get_hotspots(
    days: int = Query(30, ge=1, le=365),
    top: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """熱點 = (camera_id/location, hour, violation_type) top N。
    location 空時 fallback 用 camera_id+name 當地點 (現有 violations.location 全空)。"""
    start = datetime.utcnow() - timedelta(days=days)
    cam_map = _camera_name_map(db)
    rows = (
        db.query(
            Violation.camera_id,
            Violation.location,
            Violation.violation_type,
            Violation.violation_name,
            func.strftime("%H", Violation.created_at).label("hour"),
            func.count(Violation.id).label("cnt"),
        )
        .filter(Violation.created_at >= start)
        .filter(Violation.camera_id.isnot(None))
        .group_by(Violation.camera_id, Violation.location, Violation.violation_type,
                  Violation.violation_name, "hour")
        .order_by(func.count(Violation.id).desc())
        .limit(top)
        .all()
    )
    return {
        "days": days,
        "items": [
            {
                "camera_id": int(r.camera_id),
                "location": (r.location or "").strip() or cam_map.get(r.camera_id) or f"攝影機 {r.camera_id}",
                "violation_type": r.violation_type,
                "violation_name": r.violation_name,
                "hour": int(r.hour) if r.hour else None,
                "count": int(r.cnt),
            }
            for r in rows
        ],
    }


# ---------- 2. 推薦引擎 (規則層) ----------
_REC_THRESHOLDS = {
    # type → (per_month_threshold, suggestion_template)
    "ILLEGAL_PARKING":     (50,  "違規停車頻發 → 建議加裝告示牌 / 增設臨停格"),
    "RED_LINE_STOP":       (30,  "紅線臨停頻發 → 建議加強標線 / 巡邏取締"),
    "RED_LINE_PARKING":    (20,  "紅線停車嚴重 → 建議拖吊執法"),
    "YELLOW_LINE_PARKING": (50,  "黃線停車頻發 → 建議標示「禁止停車」告示"),
    "SIDEWALK":            (10,  "駕車侵入人行道 → 建議設置車阻"),
    "SIDEWALK_PARKING":    (30,  "人行道停車頻發 → 建議加裝車阻防護"),
    "BUS_STOP_STOP":       (20,  "公車招呼站臨停 → 建議劃設候車區"),
    "BUS_STOP_PARKING":    (15,  "公車招呼站停車 → 建議拖吊"),
    "CROSSWALK_STOP":      (10,  "斑馬線停車 → 建議畫設停止線後移"),
    "SPEEDING":            (100, "超速密集 → 建議增設測速告示 / 調速限"),
    "RED_LIGHT":           (30,  "闖紅燈頻發 → 建議調整號誌秒數"),
}


@router.get("/recommendations")
def get_recommendations(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """規則層推薦: 從 hotspots 找出超閾值組合,組合中文建議。
    location 空時 fallback 用 camera name 當地點。"""
    start = datetime.utcnow() - timedelta(days=days)
    cam_map = _camera_name_map(db)
    rows = (
        db.query(
            Violation.camera_id,
            Violation.location,
            Violation.violation_type,
            Violation.violation_name,
            func.count(Violation.id).label("cnt"),
        )
        .filter(Violation.created_at >= start)
        .filter(Violation.camera_id.isnot(None))
        .group_by(Violation.camera_id, Violation.location, Violation.violation_type, Violation.violation_name)
        .all()
    )
    recs: List[Dict[str, Any]] = []
    for r in rows:
        thr, tmpl = _REC_THRESHOLDS.get(r.violation_type, (1000, ""))
        monthly = int(r.cnt) * (30.0 / max(1, days))
        if monthly >= thr and tmpl:
            location = (r.location or "").strip() or cam_map.get(r.camera_id) or f"攝影機 {r.camera_id}"
            recs.append({
                "camera_id": int(r.camera_id),
                "location": location,
                "violation_type": r.violation_type,
                "violation_name": r.violation_name,
                "count": int(r.cnt),
                "monthly_estimate": round(monthly, 1),
                "threshold_per_month": thr,
                "severity": "high" if monthly >= thr * 2 else "medium",
                "suggestion": f"{location} · {r.violation_name} {int(r.cnt)} 起 (約 {round(monthly)}/月) → {tmpl}",
            })
    recs.sort(key=lambda x: -x["monthly_estimate"])
    return {"days": days, "items": recs[:50]}


# ---------- 3. 安全分數 ----------
@router.get("/safety_score")
def get_safety_score(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """各 cam 安全分數 0-100 (越高越安全)。
    分數 = 100 - min(50, 違規率) - min(30, 平均超速幅度) - min(20, 高風險違規比例)
    """
    start = datetime.utcnow() - timedelta(days=days)
    # 1. 各 cam 違規數 + 超速幅度 + 高風險比
    rows = db.execute(text("""
        SELECT
            camera_id,
            COUNT(*) AS total,
            AVG(COALESCE(overspeed_kmh, 0)) AS avg_overspeed,
            SUM(CASE WHEN violation_type IN ('RED_LIGHT','SPEEDING','SIDEWALK','CROSSWALK_STOP')
                     THEN 1 ELSE 0 END) AS high_risk
        FROM violations
        WHERE created_at >= :start
          AND camera_id IS NOT NULL
        GROUP BY camera_id
    """), {"start": start}).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        total = int(r.total or 0)
        avg_over = float(r.avg_overspeed or 0)
        high_risk = int(r.high_risk or 0)
        # 違規率扣分 (越多扣越多,封頂 50)
        violation_penalty = min(50, total / 30.0)  # 每月 30 起扣 1
        # 超速幅度扣分 (avg 10 km/h 超扣 10,封頂 30)
        speed_penalty = min(30, avg_over)
        # 高風險違規比例扣分 (封頂 20)
        hr_ratio = (high_risk / total) if total > 0 else 0
        hr_penalty = min(20, hr_ratio * 20)
        score = max(0, 100 - violation_penalty - speed_penalty - hr_penalty)
        out.append({
            "camera_id": int(r.camera_id),
            "score": round(score, 1),
            "level": "safe" if score >= 80 else "warn" if score >= 60 else "danger",
            "total_violations": total,
            "avg_overspeed_kmh": round(avg_over, 1),
            "high_risk_count": high_risk,
            "high_risk_ratio": round(hr_ratio, 3),
        })
    out.sort(key=lambda x: x["score"])
    return {"days": days, "items": out}


# ---------- 4. 24h 流量預測 ----------
@router.get("/flow_forecast")
def get_flow_forecast(
    camera_id: int = Query(...),
    history_days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """24h 流量歷史 + 下 1 小時預測 (簡單同小時平均法,不需 ML lib)"""
    start = datetime.utcnow() - timedelta(days=history_days)
    # 近 N 天 hourly 流量
    rows = db.execute(text("""
        SELECT strftime('%H', created_at) AS hour, COUNT(*) AS cnt
        FROM traffic_events
        WHERE camera_id = :cid AND created_at >= :start
        GROUP BY hour
        ORDER BY hour
    """), {"cid": camera_id, "start": start}).fetchall()
    hourly_avg: Dict[int, float] = {int(r.hour): int(r.cnt) / history_days for r in rows}
    history = [{"hour": h, "avg_count": round(hourly_avg.get(h, 0.0), 1)} for h in range(24)]

    # 下 1 小時預測 = 該小時近 N 天平均
    now_hour = datetime.utcnow().hour
    next_hour = (now_hour + 1) % 24
    forecast = {
        "next_hour": next_hour,
        "predicted_count": round(hourly_avg.get(next_hour, 0.0), 1),
        "method": "same-hour mean of recent {} days".format(history_days),
    }

    # 找出 peak 跟低谷
    if hourly_avg:
        peak_hour = max(hourly_avg, key=hourly_avg.get)
        low_hour = min(hourly_avg, key=hourly_avg.get)
        peak = {"hour": peak_hour, "count": round(hourly_avg[peak_hour], 1)}
        low = {"hour": low_hour, "count": round(hourly_avg[low_hour], 1)}
    else:
        peak = low = None

    return {
        "camera_id": camera_id,
        "history_days": history_days,
        "history_24h": history,
        "forecast": forecast,
        "peak": peak,
        "low": low,
    }


# ---------- 6. 違規 30 天 trend (每日違規數) ----------
@router.get("/violation_trend")
def get_violation_trend(
    days: int = Query(30, ge=7, le=180),
    db: Session = Depends(get_db),
):
    """過去 N 天每日違規數,給 line chart 用。
    top 5 type 各一條線 + 「其他」一條合計。"""
    start = datetime.utcnow() - timedelta(days=days)
    # 1. 找 top 5 type
    top_types = db.execute(text("""
        SELECT violation_type, COUNT(*) AS cnt
        FROM violations
        WHERE created_at >= :start
        GROUP BY violation_type
        ORDER BY cnt DESC
        LIMIT 5
    """), {"start": start}).fetchall()
    top_type_names = [r.violation_type for r in top_types]
    # 2. 每日每 type 計數
    rows = db.execute(text("""
        SELECT
            DATE(created_at) AS day,
            violation_type,
            COUNT(*) AS cnt
        FROM violations
        WHERE created_at >= :start
        GROUP BY day, violation_type
        ORDER BY day
    """), {"start": start}).fetchall()
    # 3. 整理 day → {type: cnt}
    by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        d = str(r.day)
        t = r.violation_type if r.violation_type in top_type_names else "OTHER"
        by_day[d][t] += int(r.cnt)
    # 4. 補齊缺少的天
    days_list: List[str] = []
    cursor = (datetime.utcnow() - timedelta(days=days)).date()
    end_date = datetime.utcnow().date()
    while cursor <= end_date:
        days_list.append(str(cursor))
        cursor += timedelta(days=1)
    # 5. 組 series
    series: List[Dict[str, Any]] = []
    for tname in top_type_names + ["OTHER"]:
        data = [by_day[d].get(tname, 0) for d in days_list]
        if sum(data) > 0:
            series.append({"type": tname, "data": data})
    return {
        "days": days,
        "labels": days_list,
        "series": series,
        "total_per_day": [sum(by_day[d].values()) for d in days_list],
    }


# ---------- 5. 車種 × 違規 交叉統計 ----------
@router.get("/vehicle_type_violations")
def get_vehicle_type_violations(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """車種 × 違規類型 交叉表 (堆疊長條 chart 用)"""
    start = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(
            Violation.vehicle_type,
            Violation.violation_type,
            Violation.violation_name,
            func.count(Violation.id).label("cnt"),
        )
        .filter(Violation.created_at >= start)
        .filter(Violation.vehicle_type.isnot(None))
        .filter(Violation.vehicle_type != "")
        .group_by(Violation.vehicle_type, Violation.violation_type, Violation.violation_name)
        .order_by(func.count(Violation.id).desc())
        .all()
    )
    # 整理成 vehicle_type → [{violation_type, name, count}]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    totals: Dict[str, int] = defaultdict(int)
    for r in rows:
        grouped[r.vehicle_type].append({
            "violation_type": r.violation_type,
            "violation_name": r.violation_name,
            "count": int(r.cnt),
        })
        totals[r.vehicle_type] += int(r.cnt)
    items = [
        {
            "vehicle_type": vt,
            "total": totals[vt],
            "breakdown": sorted(grouped[vt], key=lambda x: -x["count"]),
        }
        for vt in sorted(totals, key=lambda v: -totals[v])
    ]
    return {"days": days, "items": items}
