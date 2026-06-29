#!/usr/bin/env python3
"""交通違規偵測系統 - FastAPI 後端"""
import warnings
warnings.filterwarnings('ignore')

# 啟用 Python faulthandler — 觸發 SIGSEGV/SIGABRT 時印 Python 層 stack 到 stderr，
# 由 systemd journalctl 收下，方便定位 native 崩潰是從哪段 Python 呼叫進去的。
import faulthandler
faulthandler.enable()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import os
import threading
from zoneinfo import ZoneInfo
from api.models import init_db
from api.routes import auth, frigate, lpr, lpr_stream, lpr_visual, violations, cameras, stream, traffic, nx
from api.routes import congestion
from api.routes import logs, system
from api.routes import external, api_key_admin
from api.routes import mqtt as mqtt_route
from api.routes import io as io_route
from api.routes import lock as lock_route
from api.routes import sensor_fusion as sensor_fusion_route
from api.routes import analytics as analytics_route
from api.routes import vision_eye as vision_eye_route
from api.routes import io_tcp as io_tcp_route
from api.routes import parking as parking_route
TZ_TAIPEI = ZoneInfo("Asia/Taipei")


def _resume_services_in_background():
    # 三個服務各自獨立恢復，不互相阻塞
    def _resume_detection():
        try:
            det = stream.resume_detection_services()
            logs.add_log("info", f"偵測服務恢復: {det.get('resumed',0)}/{det.get('total',0)}", "system")
        except Exception as e:
            logs.add_log("error", f"偵測恢復失敗: {e}", "system")

    def _resume_congestion():
        try:
            cong = congestion.resume_congestion_services()
            logs.add_log("info", f"壅塞偵測恢復: {cong.get('resumed',0)}/{cong.get('total',0)}", "system")
        except Exception as e:
            logs.add_log("error", f"壅塞恢復失敗: {e}", "system")

    def _resume_lpr():
        try:
            lpr_resume = lpr_stream.resume_lpr_streams()
            logs.add_log("info", f"LPR 恢復: {lpr_resume.get('resumed',0)}/{lpr_resume.get('total',0)}", "system")
        except Exception as e:
            logs.add_log("error", f"LPR 恢復失敗: {e}", "system")

    threading.Thread(target=_resume_detection, daemon=True, name="resume-detection").start()
    threading.Thread(target=_resume_congestion, daemon=True, name="resume-congestion").start()
    threading.Thread(target=_resume_lpr, daemon=True, name="resume-lpr").start()


import time as _time

_WATCHDOG_INTERVAL = 15  # 每 15 秒檢查一次


def _service_watchdog():
    """定期監控 detection / LPR / congestion 服務，掛掉自動重啟。"""
    from api.models import SessionLocal, Camera
    from api.utils.feature_state import get_feature_enabled

    # 等待初始啟動完成
    _time.sleep(45)
    print("🐕 [watchdog] 服務監控啟動", flush=True)

    while True:
        try:
            db = SessionLocal()
            cameras = db.query(Camera).filter(Camera.enabled == True).all()
            restarted = []

            for cam in cameras:
                cam_id = cam.id

                # --- Detection watchdog ---
                want_det = get_feature_enabled("detection", cam_id, default=bool(cam.detection_enabled))
                if want_det:
                    svc = stream.detection_services.get(cam_id, {})
                    t = svc.get("_thread")
                    if t is not None and not t.is_alive():
                        stream.detection_services.pop(cam_id, None)
                        stream._start_detection_service(cam)
                        restarted.append(f"detection-{cam_id}")

                # --- LPR watchdog ---
                want_lpr = get_feature_enabled("lpr", cam_id, default=False)
                if want_lpr:
                    task = lpr_stream._lpr_tasks.get(cam_id)
                    need_restart = False
                    if task is None:
                        need_restart = True
                    elif not task.running:
                        need_restart = True
                    elif task.thread is not None and not task.thread.is_alive():
                        need_restart = True
                    elif task.last_frame_at and (_time.time() - task.last_frame_at) > 20:
                        # 超過 60 秒沒有新幀 → 串流卡住，重啟
                        need_restart = True
                    if need_restart:
                        if task:
                            try:
                                task.stop()
                            except Exception:
                                pass
                        lpr_stream._lpr_tasks.pop(cam_id, None)
                        lpr_stream._start_lpr_task(cam)
                        restarted.append(f"lpr-{cam_id}")

                # --- Congestion watchdog ---
                want_cong = get_feature_enabled("congestion", cam_id, default=False)
                if want_cong:
                    cong_svc = congestion.congestion_services.get(cam_id, {})
                    ct = cong_svc.get("_thread")
                    if ct is not None and not ct.is_alive():
                        congestion.congestion_services.pop(cam_id, None)
                        congestion._start_congestion_service(cam)
                        restarted.append(f"congestion-{cam_id}")

            db.close()

            if restarted:
                msg = f"Watchdog 自動重啟服務: {', '.join(restarted)}"
                print(f"🔄 {msg}", flush=True)
                logs.add_log("warning", msg, "watchdog")

        except Exception as e:
            print(f"⚠️ [watchdog] 監控異常: {e}", flush=True)

        _time.sleep(_WATCHDOG_INTERVAL)


def _assert_gpu_ready():
    """強制檢查 GPU 是否可用，避免推論誤落到 CPU。"""
    force_gpu = os.getenv("FORCE_GPU", "true").lower() in ("1", "true", "yes", "on")
    if not force_gpu:
        print("ℹ️ FORCE_GPU=False，略過 GPU 強制檢查")
        return
    device = os.getenv("DEVICE", "cuda:0")
    model_dir = os.getenv("MODEL_DIR", "/workspace/models")
    detect_model_pt = os.getenv("DETECT_MODEL_PT", "yolov8n.pt")
    model_path = detect_model_pt if os.path.isabs(detect_model_pt) else f"{model_dir}/{detect_model_pt}"
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"FORCE_GPU=True 但無法匯入 torch: {e}")

    if not torch.cuda.is_available():
        raise RuntimeError("FORCE_GPU=True 但 CUDA 不可用，拒絕啟動")

    if torch.cuda.device_count() < 1:
        raise RuntimeError("FORCE_GPU=True 但未偵測到任何 CUDA 裝置，拒絕啟動")

    try:
        _ = torch.zeros((1,), device=device)
    except Exception as e:
        raise RuntimeError(f"FORCE_GPU=True 但無法使用裝置 {device}: {e}")

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"FORCE_GPU=True 但偵測模型不存在: {model_path}"
        )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU 檢查通過: {gpu_name} ({device})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 啟動交通違規影像分析系統")
    _assert_gpu_ready()
    init_db()
    logs.add_log("info", "系統日誌服務啟動", "system")
    os.makedirs("./output/violations", exist_ok=True)
    threading.Thread(target=_resume_services_in_background, daemon=True, name="resume-services").start()
    threading.Thread(target=_service_watchdog, daemon=True, name="service-watchdog").start()
    # 啟動 MQTT bridge (讀 config，自動連 broker)
    try:
        from services.mqtt_bridge import start as _mqtt_start
        _mqtt_start()
    except Exception as _e:
        print(f"⚠️ MQTT bridge 啟動失敗: {_e}", flush=True)
    try:
        from services.io_service import start as _io_start
        _io_start()
    except Exception as _e:
        print(f"?? IO service ????: {_e}", flush=True)
    print("✅ 系統初始化完成")
    yield
    # 通知所有 streaming generator 退出，避免 sync while True 卡到 SIGKILL
    from api.utils.shutdown import shutdown_event
    shutdown_event.set()
    print("👋 系統關閉")


app = FastAPI(
    title="交通違規影像分析系統",
    description="Jetson NX AI 邊緣運算平台",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_web_html(request: Request, call_next):
    response = await call_next(request)
    path = str(request.url.path or "")
    if path.startswith("/web"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# 註冊路由
app.include_router(violations.router)
app.include_router(cameras.router)
app.include_router(stream.router)
app.include_router(traffic.router)
app.include_router(auth.router)
app.include_router(frigate.router)
app.include_router(nx.router)
app.include_router(lpr.router)
app.include_router(lpr_stream.router)
app.include_router(lpr_visual.router)
app.include_router(congestion.router)
app.include_router(logs.router)
app.include_router(system.router)
app.include_router(external.router)
app.include_router(api_key_admin.router)
app.include_router(mqtt_route.router)
app.include_router(io_route.router)
app.include_router(lock_route.router)
app.include_router(sensor_fusion_route.router)
app.include_router(analytics_route.router)
app.include_router(vision_eye_route.router)
app.include_router(io_tcp_route.router)
app.include_router(parking_route.router)
# 啟動時把 io_tcp_modules.json 內所有 module 自動連起來
try:
    from services.modbus_tcp_io import init_from_config as _init_io_tcp
    _init_io_tcp()
except Exception as _e:
    print(f"[main] init_from_config (io_tcp) failed: {_e}", flush=True)
# 靜態檔案
if os.path.exists("./output"):
    app.mount("/files", StaticFiles(directory="./output"), name="files")
if os.path.exists("./web"):
    app.mount("/web", StaticFiles(directory="./web", html=True), name="web")


@app.get("/")
async def root():
    return {"name": "交通違規影像分析系統", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(TZ_TAIPEI).isoformat()}


@app.get("/api/system/info")
async def system_info():
    import torch
    return {
        "platform": "Jetson NX",
        "pytorch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.get("/api/dashboard")
async def dashboard():
    """主頁儀錶板 — KPI + 24h trend buckets"""
    from api.models import SessionLocal, Violation, Camera
    from sqlalchemy import func
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        d24h_ago = now - timedelta(hours=24)

        # KPI
        today_v = db.query(Violation).filter(Violation.created_at >= today).count()
        pending = db.query(Violation).filter(Violation.status == "pending").count()
        total_v = db.query(Violation).count()

        # Camera 狀態統計 — 區分「啟用」(user 想跑的) 跟「故障離線」(實際斷線)
        # 注意：enabled=False 表 user 主動關閉，不算故障離線
        all_cams = db.query(Camera).all()
        total_cam = len(all_cams)
        enabled_cams = [c for c in all_cams if getattr(c, "enabled", True)]
        enabled_count = len(enabled_cams)
        online_cam = sum(1 for c in enabled_cams if (c.status or "").lower() == "online")
        # offline_cameras 只計算「啟用但離線」(真實故障，需立即檢查)
        offline_cam = max(0, enabled_count - online_cam)
        disabled_cam = total_cam - enabled_count

        # 24h trend buckets (hour=0..23 按本地時間 = 台北 UTC+8)
        rows = (db.query(Violation)
                  .filter(Violation.created_at >= d24h_ago)
                  .with_entities(Violation.created_at)
                  .all())
        bucket_map: dict = {}
        for (ts,) in rows:
            if ts is None:
                continue
            # 台北時區 hour-of-day (UTC + 8)
            local = ts + timedelta(hours=8)
            key = local.hour
            bucket_map[key] = bucket_map.get(key, 0) + 1
        # 從現在往前 24 小時順序 (從 24h 前 → 現在)
        cursor = (now + timedelta(hours=8) - timedelta(hours=23)).hour
        hourly_buckets = []
        for i in range(24):
            h = (cursor + i) % 24
            hourly_buckets.append({"hour": h, "count": bucket_map.get(h, 0)})

        # 今日違規分析分布 (戰情室視覺化用) — 依 count 由多到少
        def _today_dist(col):
            return [
                {"key": k, "count": cnt}
                for k, cnt in (db.query(col, func.count(Violation.id))
                                 .filter(Violation.created_at >= today)
                                 .group_by(col)
                                 .order_by(func.count(Violation.id).desc())
                                 .all())
            ]
        cam_name_map = {c.id: (c.name or f"攝影機 {c.id}") for c in all_cams}
        type_dist = _today_dist(Violation.violation_type)
        vehicle_dist = _today_dist(Violation.vehicle_type)
        camera_dist = [
            {"key": cam_name_map.get(d["key"], f"攝影機 {d['key']}"), "count": d["count"]}
            for d in _today_dist(Violation.camera_id)
        ]
        # 超速幅度分桶 (今日，依 overspeed_kmh)
        _speed_edges = [("<10", 0, 10), ("10-20", 10, 20), ("20-40", 20, 40),
                        ("40-60", 40, 60), ("60+", 60, 10 ** 9)]
        _speed_counts = {e[0]: 0 for e in _speed_edges}
        for (ov,) in (db.query(Violation.overspeed_kmh)
                        .filter(Violation.created_at >= today,
                                Violation.overspeed_kmh.isnot(None))
                        .all()):
            val = float(ov or 0)
            for name, lo, hi in _speed_edges:
                if lo <= val < hi:
                    _speed_counts[name] += 1
                    break
        speed_dist = [{"key": e[0], "count": _speed_counts[e[0]]} for e in _speed_edges]

        # 昨日同時段對比 (KPI 趨勢 % + 24h 趨勢圖比較線)
        yesterday_start = today - timedelta(days=1)
        yesterday_same_time_end = now - timedelta(days=1)
        yesterday_same_time = (db.query(Violation)
                               .filter(Violation.created_at >= yesterday_start,
                                       Violation.created_at < yesterday_same_time_end)
                               .count())
        # 昨日整天 hourly (對齊 hourly_buckets 的 hour 順序)
        ystr_rows = (db.query(Violation.created_at)
                       .filter(Violation.created_at >= yesterday_start,
                               Violation.created_at < today)
                       .all())
        ybucket_map: dict = {}
        for (ts,) in ystr_rows:
            if ts is None:
                continue
            local = ts + timedelta(hours=8)
            ybucket_map[local.hour] = ybucket_map.get(local.hour, 0) + 1
        yesterday_hourly_buckets = [
            {"hour": (cursor + i) % 24, "count": ybucket_map.get((cursor + i) % 24, 0)}
            for i in range(24)
        ]
        # 待審核中最舊一筆距現在多久 (分鐘) — KPI 待審核副資訊
        _oldest = (db.query(Violation.created_at)
                   .filter(Violation.status == "pending")
                   .order_by(Violation.created_at.asc()).first())
        oldest_pending_minutes = (int((now - _oldest[0]).total_seconds() / 60)
                                  if (_oldest and _oldest[0]) else None)

        # 最近 1 小時違規分類 (LAST 1H 卡用) — 前端 hourlyStats 來源
        one_hour_ago = now - timedelta(hours=1)
        recent_hour_dist = [
            {"key": k, "count": cnt}
            for k, cnt in (db.query(Violation.violation_type, func.count(Violation.id))
                             .filter(Violation.created_at >= one_hour_ago)
                             .group_by(Violation.violation_type)
                             .all())
        ]

        return {
            "today_violations": today_v,
            "pending_review": pending,
            "total_violations": total_v,
            "online_cameras": online_cam,
            "offline_cameras": offline_cam,       # 真實故障 (enabled=True 但 status=offline)
            "disabled_cameras": disabled_cam,     # user 主動關閉 (enabled=False)
            "enabled_cameras": enabled_count,
            "total_cameras": total_cam,
            "hourly_buckets": hourly_buckets,
            "type_dist": type_dist,
            "vehicle_dist": vehicle_dist,
            "camera_dist": camera_dist,
            "speed_dist": speed_dist,
            "yesterday_same_time_violations": yesterday_same_time,
            "yesterday_hourly_buckets": yesterday_hourly_buckets,
            "oldest_pending_minutes": oldest_pending_minutes,
            "recent_hour_dist": recent_hour_dist,
        }
    finally:
        db.close()


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
