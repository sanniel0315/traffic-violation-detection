FROM dustynv/pytorch:2.1-r36.2.0

# 降級 numpy 避免記憶體問題
RUN pip install --no-cache-dir "numpy<2"

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.0.74 \
    ultralytics \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiofiles \
    pydantic \
    pydantic-settings \
    sqlalchemy \
    requests

WORKDIR /workspace
CMD ["/bin/bash"]
