#!/usr/bin/env python3
"""交通違規偵測系統 - FastAPI 後端"""
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import os
import threading
from zoneinfo import ZoneInfo
from api.models import init_db
from api.routes import auth, frigate, lpr, lpr_stream, lpr_visual, violations, cameras, stream, traffic, nx
from api.routes import congestion
from api.routes import logs, system
from api.routes import external, api_key_admin
from api.routes import mqtt as mqtt_route
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
    print("✅ 系統初始化完成")
    yield
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
    from api.models import SessionLocal, Violation, Camera
    db = SessionLocal()
    try:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        return {
            "today_violations": db.query(Violation).filter(Violation.created_at >= today).count(),
            "pending_review": db.query(Violation).filter(Violation.status == "pending").count(),
            "total_violations": db.query(Violation).count(),
            "online_cameras": db.query(Camera).filter(Camera.status == "online").count(),
            "total_cameras": db.query(Camera).count()
        }
    finally:
        db.close()


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
