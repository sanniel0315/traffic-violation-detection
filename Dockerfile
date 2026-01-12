# =============================================================================
# 交通違規影像分析系統 - Dockerfile
# 平台: NVIDIA Jetson Xavier NX (JetPack 6.0 / L4T R36.x)
# =============================================================================

FROM dustynv/l4t-pytorch:r36.2.0

# 環境變數
ENV TZ=Asia/Taipei
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# 設定時區
RUN ln -sf /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    echo "Asia/Taipei" > /etc/timezone

# 降級 NumPy (PyTorch 2.2 需要 NumPy < 2)
RUN pip install --no-cache-dir -i https://pypi.org/simple/ "numpy==1.26.4"

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-tra \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Ultralytics (不安裝 torch 依賴，使用基礎映像的)
RUN pip install --no-cache-dir --no-deps -i https://pypi.org/simple/ ultralytics

# 安裝 OpenCV (相容 NumPy 1.x)
RUN pip install --no-cache-dir --no-deps -i https://pypi.org/simple/ "opencv-python-headless==4.8.0.76"

# 安裝其他 Python 依賴
RUN pip install --no-cache-dir -i https://pypi.org/simple/ \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiofiles \
    pydantic \
    sqlalchemy \
    psycopg2-binary \
    httpx \
    python-dotenv \
    pytesseract \
    onnx \
    onnxslim \
    matplotlib \
    pandas \
    pyyaml \
    tqdm \
    scipy \
    seaborn \
    requests \
    pillow \
    py-cpuinfo \
    psutil \
    lap \
    filterpy

WORKDIR /workspace

EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
