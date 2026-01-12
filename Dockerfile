FROM dustynv/l4t-pytorch:r36.2.0

# 降級 numpy
RUN pip install --no-cache-dir "numpy<2"

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# 安裝 ultralytics (使用標準 PyPI)
RUN pip install --no-cache-dir --no-deps -i https://pypi.org/simple/ ultralytics

# 安裝其他依賴
RUN pip install --no-cache-dir -i https://pypi.org/simple/ \
    opencv-python-headless \
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
    psutil

WORKDIR /workspace
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
