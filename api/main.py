#!/usr/bin/env python3
"""交通違規偵測系統 - FastAPI 後端"""
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime
import os

from api.models import init_db
from api.routes import frigate, lpr, lpr_stream, lpr_visual, violations, cameras, stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 啟動交通違規影像分析系統")
    init_db()
    os.makedirs("./output/violations", exist_ok=True)
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

# 註冊路由
app.include_router(violations.router)
app.include_router(cameras.router)
app.include_router(stream.router)
app.include_router(frigate.router)
app.include_router(lpr.router)
app.include_router(lpr_stream.router)
app.include_router(lpr_visual.router)

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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
