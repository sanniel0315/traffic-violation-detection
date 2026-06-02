#!/usr/bin/env python3
"""違規事件 API"""
import calendar
import os
import subprocess
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from api.models import get_db, Violation
from api.routes.logs import add_log

router = APIRouter(prefix="/api/violations", tags=["違規事件"])

# 違規 4 張時間軸快照 cache dir (mount 在 /files/violations/snapshots/)
_SNAPSHOT_CACHE_DIR = Path("./output/violations/snapshots").resolve()
_SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# 4 張時間軸偏移 (秒)，相對 violation_time
_SNAPSHOT_OFFSETS = [("before", -2.0), ("mid_a", -0.5), ("mid_b", 0.5), ("after", 2.0)]
_SNAPSHOT_LABELS = {"before": "事件前 t-2s", "mid_a": "中段 t-0.5s",
                    "mid_b": "中段 t+0.5s", "after": "事件後 t+2s"}


def _find_unicode_font(size: int):
    """跨平台找中文字體 (PIL ImageFont)，找不到 fall back default。"""
    from PIL import ImageFont
    candidates = [
        "/workspace/web/fonts/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in candidates:
        try:
            if os.path.exists(fp):
                return ImageFont.truetype(fp, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _find_plate_crop_disk_path(v: Violation, db: Session) -> Optional[Path]:
    """違規 → LPRRecord 配對 → 推算 plate crop disk path。
    跟前端 /match-plate-snapshots 同邏輯: plate_number + camera_id + ±10min。"""
    if not v.license_plate or not v.camera_id or not v.created_at:
        return None
    try:
        from api.models import LPRRecord
        from api.routes.lpr_stream import SNAPSHOT_DIR
        lo = v.created_at - timedelta(minutes=10)
        hi = v.created_at + timedelta(minutes=10)
        rec = (db.query(LPRRecord)
               .filter(LPRRecord.plate_number == v.license_plate)
               .filter(LPRRecord.camera_id == int(v.camera_id))
               .filter(LPRRecord.created_at >= lo, LPRRecord.created_at <= hi)
               .order_by(LPRRecord.created_at.desc())
               .first())
        if not rec or not rec.snapshot:
            return None
        plate_name = rec.snapshot.rsplit(".", 1)[0] + "_plate.png"
        p = Path(SNAPSHOT_DIR) / plate_name
        return p if p.exists() else None
    except Exception:
        return None


def _find_nearest_lpr_plate(v: Violation, db: Session) -> Optional[dict]:
    """violation.license_plate 空時，從 LPR 同攝影機 ±2 分鐘最近一筆抓車牌號碼 + plate crop。
    confidence >= 0.5 才信。回 {plate, plate_path}。"""
    if not v.camera_id or not v.created_at:
        return None
    try:
        from api.models import LPRRecord
        from api.routes.lpr_stream import SNAPSHOT_DIR
        lo = v.created_at - timedelta(minutes=2)
        hi = v.created_at + timedelta(minutes=2)
        rec = (db.query(LPRRecord)
               .filter(LPRRecord.camera_id == int(v.camera_id))
               .filter(LPRRecord.created_at >= lo, LPRRecord.created_at <= hi)
               .filter(LPRRecord.plate_number.isnot(None))
               .filter(LPRRecord.plate_number != "")
               .filter(LPRRecord.confidence >= 0.5)
               .order_by(LPRRecord.created_at.desc())
               .first())
        if not rec:
            return None
        out = {"plate": rec.plate_number, "plate_path": None}
        if rec.snapshot:
            pn = rec.snapshot.rsplit(".", 1)[0] + "_plate.png"
            p = Path(SNAPSHOT_DIR) / pn
            if p.exists():
                out["plate_path"] = p
        return out
    except Exception:
        return None


def _build_composite_image(vid: int, v: Violation, plate_disk: Optional[Path], out_path: Path) -> bool:
    """拼 2x2 (4 張時間軸) → 1 張高解析度 JPG (2560x1440 2K)。
    每 cell 1280x720 (從 1920x1080 LANCZOS 一次 downscale，避免兩次縮放模糊)。
    乾淨版: 不加時間標籤/info bar/plate overlay。
    plate_disk 參數保留 (相容呼叫端) 但不使用 (LPR 跟 violation 不嚴格 1:1)。"""
    try:
        from PIL import Image
        cells = []
        any_loaded = False
        cell_w, cell_h = 1280, 720
        for tag, _ in _SNAPSHOT_OFFSETS:
            p = _SNAPSHOT_CACHE_DIR / f"{vid}_{tag}.jpg"
            if p.exists() and p.stat().st_size > 1024:
                im = Image.open(p).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
                any_loaded = True
            else:
                im = None
            cells.append(im)
        if not any_loaded:
            return False
        canvas = Image.new("RGB", (cell_w * 2, cell_h * 2), (15, 23, 42))
        for i, im in enumerate(cells):
            cx = (i % 2) * cell_w
            cy = (i // 2) * cell_h
            if im is not None:
                canvas.paste(im, (cx, cy))

        canvas.save(out_path, "JPEG", quality=92, subsampling=0)
        return True
    except Exception as e:
        print(f"⚠️ composite build failed for {vid}: {e}", flush=True)
        return False


class ViolationCreate(BaseModel):
    violation_type: str
    violation_name: str
    license_plate: Optional[str] = None
    vehicle_type: str
    location: str
    camera_id: int
    confidence: float
    track_id: Optional[int] = None
    bbox: Optional[dict] = None
    image_path: Optional[str] = None
    fine_amount: Optional[int] = None
    points: Optional[int] = None
    speed_kmh: Optional[float] = None
    speed_limit_kmh: Optional[float] = None
    overspeed_kmh: Optional[float] = None
    flow_roi_hit: Optional[bool] = None
    speed_roi_hit: Optional[bool] = None


class ViolationReview(BaseModel):
    status: str  # confirmed, rejected
    comment: Optional[str] = None


@router.get("")
async def get_violations(
    status: Optional[str] = None,
    violation_type: Optional[str] = None,
    license_plate: Optional[str] = None,
    camera_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """取得違規列表"""
    query = db.query(Violation)
    
    if status:
        query = query.filter(Violation.status == status)
    if violation_type:
        query = query.filter(Violation.violation_type == violation_type)
    if license_plate:
        query = query.filter(Violation.license_plate.ilike(f"%{license_plate}%"))
    if camera_id:
        query = query.filter(Violation.camera_id == camera_id)
    if start_date:
        query = query.filter(Violation.violation_time >= start_date)
    if end_date:
        query = query.filter(Violation.violation_time <= end_date)
    
    total = query.count()
    items = query.order_by(desc(Violation.created_at)).offset((page-1)*page_size).limit(page_size).all()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [_to_dict(v) for v in items]
    }


@router.get("/statistics")
async def get_statistics(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """取得統計資料"""
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    
    # 總數統計
    total = db.query(Violation).count()
    pending = db.query(Violation).filter(Violation.status == "pending").count()
    confirmed = db.query(Violation).filter(Violation.status == "confirmed").count()
    
    # 今日統計
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.query(Violation).filter(Violation.created_at >= today_start).count()
    
    # 按類型統計
    by_type = db.query(
        Violation.violation_type,
        Violation.violation_name,
        func.count(Violation.id)
    ).filter(
        Violation.created_at >= start
    ).group_by(Violation.violation_type, Violation.violation_name).all()
    
    # 按日期統計
    by_date = db.query(
        func.date(Violation.created_at).label('date'),
        func.count(Violation.id)
    ).filter(
        Violation.created_at >= start
    ).group_by(func.date(Violation.created_at)).all()
    
    return {
        "total": total,
        "pending": pending,
        "confirmed": confirmed,
        "today": today_count,
        "by_type": [{"type": t, "name": n, "count": c} for t, n, c in by_type],
        "by_date": [{"date": str(d), "count": c} for d, c in by_date]
    }


@router.get("/repeat-offenders")
def repeat_offenders(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    threshold: int = 2,
    violation_type: Optional[str] = None,
    plate_kw: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """車輛累犯查詢: GROUP BY 車牌 + 套用時間/類型/車牌關鍵字篩選。
    必須定義在 /{violation_id} 之前 — FastAPI 按順序 match，否則 'repeat-offenders'
    會被當 violation_id 嘗試 parse 成 int → 422。"""
    base = db.query(Violation).filter(
        Violation.license_plate.isnot(None),
        Violation.license_plate != "",
    )
    if start_time:
        try:
            base = base.filter(Violation.created_at >= datetime.fromisoformat(start_time.replace("Z", "+00:00")))
        except Exception:
            pass
    if end_time:
        try:
            base = base.filter(Violation.created_at < datetime.fromisoformat(end_time.replace("Z", "+00:00")))
        except Exception:
            pass
    if violation_type:
        base = base.filter(Violation.violation_type == violation_type)
    if plate_kw:
        base = base.filter(Violation.license_plate.like(f"%{plate_kw}%"))

    th = max(1, int(threshold or 1))
    rows = (
        base.with_entities(
            Violation.license_plate,
            func.count(Violation.id).label("cnt"),
            func.min(Violation.created_at).label("first_at"),
            func.max(Violation.created_at).label("last_at"),
            func.sum(Violation.fine_amount).label("fine_total"),
        )
        .group_by(Violation.license_plate)
        .having(func.count(Violation.id) >= th)
        .order_by(desc("cnt"))
        .limit(500)
        .all()
    )
    plates = [r[0] for r in rows]
    types_map: dict = {}
    if plates:
        tq = base.with_entities(
            Violation.license_plate,
            Violation.violation_type,
            func.count(Violation.id),
        ).filter(Violation.license_plate.in_(plates)).group_by(
            Violation.license_plate, Violation.violation_type
        ).all()
        for p, t, c in tq:
            types_map.setdefault(p, {})[t or "UNKNOWN"] = int(c)
    items = [
        {
            "plate": p,
            "count": int(cnt),
            "first_at": fa.isoformat() if fa else None,
            "last_at": la.isoformat() if la else None,
            "fine_total": int(ft or 0),
            "types": types_map.get(p, {}),
        }
        for (p, cnt, fa, la, ft) in rows
    ]
    return {"items": items, "total": len(items)}


def _fetch_frigate_clip(camera_name: str, start_unix: int, end_unix: int) -> Optional[bytes]:
    """從 frigate 抓 mp4 clip bytes (失敗回 None)。"""
    import requests as _rq
    base_urls = ["http://127.0.0.1:5000", "http://frigate:5000"]
    for base in base_urls:
        try:
            r = _rq.get(
                f"{base}/api/{camera_name}/start/{start_unix}/end/{end_unix}/clip.mp4",
                timeout=15,
            )
            if r.status_code == 200 and len(r.content) > 1024:
                return r.content
        except Exception:
            continue
    return None


def _extract_frame_with_ffmpeg(clip_path: Path, seek_sec: float, out_path: Path) -> bool:
    """ffmpeg seek to `seek_sec` in clip → 抽 1 frame 存 out_path。
    Robust 多策略: 1) accurate seek (-ss after -i) → 2) fast seek (-ss before -i) →
    3) 容差 -0.3s fallback。前策略失敗 fall through 下一個。"""
    def _try(seek: float, fast: bool) -> bool:
        try:
            if fast:
                cmd = ["ffmpeg", "-y", "-ss", f"{seek:.3f}", "-i", str(clip_path),
                       "-frames:v", "1", "-q:v", "3", str(out_path)]
            else:
                cmd = ["ffmpeg", "-y", "-i", str(clip_path),
                       "-ss", f"{seek:.3f}", "-frames:v", "1", "-q:v", "3", str(out_path)]
            subprocess.run(cmd, capture_output=True, timeout=10, check=False)
            return out_path.exists() and out_path.stat().st_size > 1024
        except Exception:
            return False
    if _try(seek_sec, fast=False):
        return True
    if _try(seek_sec, fast=True):
        return True
    if seek_sec > 0.3 and _try(seek_sec - 0.3, fast=True):
        return True
    return False


@router.get("/{violation_id}/snapshots")
def get_violation_snapshots(violation_id: int, db: Session = Depends(get_db)):
    """違規 4 張時間軸快照 (前/中/中/後): 從 frigate clip 用 ffmpeg 抽 ±2s/±0.5s 共 4 frame，
    存 ./output/violations/snapshots/{id}_{tag}.jpg，回 /files/... URL。
    Cache：4 張都已存在直接回。第一次 fetch 抓 frigate clip ~1-2s。"""
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="violation not found")
    if not v.created_at or not v.camera_id:
        return {"snapshots": {}, "violation_id": violation_id, "reason": "missing time/camera"}

    # check cache 是否齊全
    cached = {tag: (_SNAPSHOT_CACHE_DIR / f"{violation_id}_{tag}.jpg") for tag, _ in _SNAPSHOT_OFFSETS}
    need_fetch = [tag for tag, p in cached.items() if not (p.exists() and p.stat().st_size > 1024)]

    if need_fetch:
        # frigate camera name 慣例 = f"cam_{camera_id}"
        camera_name = f"cam_{int(v.camera_id)}"
        # violation_time 是 UTC datetime (DB stored as naive UTC via datetime.utcnow)
        ts_unix = int(calendar.timegm(v.created_at.utctimetuple()))
        # 抓 clip ±6s (端 12s) — 用更大 buffer 避免末端 seek lost frame
        clip_bytes = _fetch_frigate_clip(camera_name, ts_unix - 6, ts_unix + 6)
        if clip_bytes:
            clip_path = _SNAPSHOT_CACHE_DIR / f"{violation_id}_clip.mp4"
            clip_path.write_bytes(clip_bytes)
            for tag, offset in _SNAPSHOT_OFFSETS:
                if tag not in need_fetch:
                    continue
                # clip 從 ts-6 開始, seek = 6+offset (4, 5.5, 6.5, 8 → 全部在 clip 中段)
                seek = 6.0 + offset
                _extract_frame_with_ffmpeg(clip_path, seek, cached[tag])
            try:
                clip_path.unlink(missing_ok=True)
            except Exception:
                pass

    result = {}
    for tag, _ in _SNAPSHOT_OFFSETS:
        p = cached[tag]
        if p.exists() and p.stat().st_size > 1024:
            result[tag] = f"/files/violations/snapshots/{p.name}"

    # plate crop 嚴格用該違規本身車牌 (plate_number+camera+±10min)；
    # 不 fallback 用同攝影機其他車的車牌避免「貼錯車」誤導
    plate_disk = _find_plate_crop_disk_path(v, db)
    plate_url = None
    plate_number = v.license_plate or None
    if plate_disk and plate_disk.exists():
        plate_url = f"/api/lpr/stream/snapshot/{plate_disk.name}"
    else:
        # violation 沒車牌 → 不貼 plate crop (composite/dialog overlay 都不貼)
        # 但 plate_number text 仍 fallback 同攝影機 ±2 分鐘最近 LPR 結果，給 dialog info 顯示
        if not v.license_plate:
            nearest = _find_nearest_lpr_plate(v, db)
            if nearest:
                plate_number = nearest.get("plate")
        # plate_disk 維持 None → composite / overlay 不貼

    # 至少 1 張時間軸 frame 就拼 composite (缺的格子留深灰底，避免 frigate 末端 seek 失敗整個拿不到)
    # v2 = 乾淨版 (移除時間標籤/info bar)，新檔名自動讓舊 cache 失效
    composite_url = None
    if len(result) >= 1:
        composite_path = _SNAPSHOT_CACHE_DIR / f"{violation_id}_composite_v9.jpg"
        last_count_attr = f"_last_count_{violation_id}"
        prev_count = getattr(_build_composite_image, last_count_attr, 0)
        existing = composite_path.exists() and composite_path.stat().st_size > 1024
        # 重建條件: 不存在 OR 新 frame 數比上次多 (有抽到新時間點)
        if not existing or len(result) > prev_count:
            if _build_composite_image(violation_id, v, plate_disk, composite_path):
                setattr(_build_composite_image, last_count_attr, len(result))
        if composite_path.exists() and composite_path.stat().st_size > 1024:
            composite_url = f"/files/violations/snapshots/{composite_path.name}"

    return {
        "snapshots": result,
        "composite": composite_url,
        "plate_url": plate_url,
        "plate_number": plate_number,
        "violation_id": violation_id,
        "available": list(result.keys()),
        "violation_time": v.created_at.isoformat() if v.created_at else None,
    }


@router.get("/{violation_id}")
async def get_violation(violation_id: int, db: Session = Depends(get_db)):
    """取得單一違規記錄"""
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="違規記錄不存在")
    return _to_dict(v)


@router.post("")
async def create_violation(data: ViolationCreate, db: Session = Depends(get_db)):
    """建立違規記錄"""
    v = Violation(
        violation_type=data.violation_type,
        violation_name=data.violation_name,
        license_plate=data.license_plate,
        vehicle_type=data.vehicle_type,
        location=data.location,
        camera_id=data.camera_id,
        confidence=data.confidence,
        track_id=data.track_id,
        bbox=data.bbox,
        image_path=data.image_path,
        fine_amount=data.fine_amount,
        points=data.points,
        speed_kmh=data.speed_kmh,
        speed_limit_kmh=data.speed_limit_kmh,
        overspeed_kmh=data.overspeed_kmh,
        flow_roi_hit=bool(data.flow_roi_hit),
        speed_roi_hit=bool(data.speed_roi_hit),
        violation_time=datetime.utcnow()
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    add_log(
        "warning",
        f"新增違規紀錄: {v.violation_name} | 車牌 {v.license_plate or '未知'} | 攝影機 ID={v.camera_id}",
        "violation",
    )
    return _to_dict(v)


@router.put("/{violation_id}/review")
async def review_violation(
    violation_id: int,
    review: ViolationReview,
    db: Session = Depends(get_db)
):
    """審核違規記錄"""
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="違規記錄不存在")
    
    v.status = review.status
    v.review_comment = review.comment
    v.reviewed_at = datetime.utcnow()
    db.commit()
    
    return {"message": "審核完成", "status": review.status}


@router.delete("/{violation_id}")
async def delete_violation(violation_id: int, db: Session = Depends(get_db)):
    """刪除違規記錄"""
    v = db.query(Violation).filter(Violation.id == violation_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="違規記錄不存在")
    
    db.delete(v)
    db.commit()
    return {"message": "已刪除"}


def _to_dict(v: Violation) -> dict:
    return {
        "id": v.id,
        "violation_type": v.violation_type,
        "violation_name": v.violation_name,
        "license_plate": v.license_plate,
        "vehicle_type": v.vehicle_type,
        "location": v.location,
        "camera_id": v.camera_id,
        "violation_time": v.violation_time.isoformat() if v.violation_time else None,
        "confidence": v.confidence,
        "image_path": v.image_path,
        "status": v.status,
        "fine_amount": v.fine_amount,
        "points": v.points,
        "speed_kmh": v.speed_kmh,
        "speed_limit_kmh": v.speed_limit_kmh,
        "overspeed_kmh": v.overspeed_kmh,
        "flow_roi_hit": bool(v.flow_roi_hit),
        "speed_roi_hit": bool(v.speed_roi_hit),
        "created_at": v.created_at.isoformat() if v.created_at else None
    }
