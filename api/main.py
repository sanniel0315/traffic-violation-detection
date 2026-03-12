#!/usr/bin/env python3
"""交通違規偵測系統 - FastAPI 後端"""
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from contextlib import asynccontextmanager
from datetime import datetime
import os
from zoneinfo import ZoneInfo
from api.models import init_db
from api.routes import auth, frigate, lpr, lpr_stream, lpr_visual, violations, cameras, stream, traffic
from api.routes import congestion
from api.routes import logs, system
TZ_TAIPEI = ZoneInfo("Asia/Taipei")


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
    det = stream.resume_detection_services()
    cong = congestion.resume_congestion_services()
    lpr_resume = lpr_stream.resume_lpr_streams()
    logs.add_log(
        "info",
        f"服務狀態恢復完成: detection={det.get('resumed',0)}/{det.get('total',0)} "
        f"congestion={cong.get('resumed',0)}/{cong.get('total',0)} "
        f"lpr={lpr_resume.get('resumed',0)}/{lpr_resume.get('total',0)}",
        "system",
    )
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
    is_web_html = path == "/web" or path == "/web/" or path.endswith(".html")
    if path.startswith("/web") and is_web_html:
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
app.include_router(lpr.router)
app.include_router(lpr_stream.router)
app.include_router(lpr_visual.router)
app.include_router(congestion.router)
app.include_router(logs.router)
app.include_router(system.router) 
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
