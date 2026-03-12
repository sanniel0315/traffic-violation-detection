# RUNBOOK

## 1) 專案目的與主要模組

### 專案目的
- 以 Jetson 邊緣運算做交通監控：車輛偵測、車牌 OCR、違規判定、壅塞分析，並整合 Frigate NVR 與 FastAPI。
- 主要入口：`api/main.py`。

### 模組分工
- Frigate 整合
  - 設定檔：`config/frigate/config.yml`
  - API 與同步：`api/routes/frigate.py`
  - 對外呼叫封裝：`recognition/frigate_integration.py`
- 偵測（Detection）
  - 車輛偵測：`detection/vehicle_detector.py`（Ultralytics YOLO）
  - 違規邏輯：`detection/violation_detector.py`
  - 壅塞分析：`detection/congestion_detector.py`
  - 串流偵測路由：`api/routes/stream.py`
- 車牌 OCR（LPR）
  - OCR 核心：`recognition/plate_recognizer.py`（Tesseract）
  - 單張辨識 API：`api/routes/lpr.py`
  - 串流辨識 API：`api/routes/lpr_stream.py`
  - 視覺化串流：`api/routes/lpr_visual.py`
- API / 資料層
  - API 啟動：`api/main.py`
  - 資料模型：`api/models.py`
  - 登入與帳號管理：`api/routes/auth.py`（login/me/logout + users CRUD）
  - 路由：`api/routes/*.py`
- 部署
  - 容器：`docker-compose.yml`, `Dockerfile`
  - 偵測腳本：`scripts/detection_service.py`

---

## 2) 啟動方式與所需環境變數/模型路徑

### A. Docker Compose（主要方式）
- 檔案：`docker-compose.yml`
- 啟動命令：`docker compose up -d`
- 服務
  - `api`（`traffic-api`）
  - `frigate`
  - `dev`（profile `dev`，可選）

#### API 容器環境變數（compose 內定義）
- `TZ=Asia/Taipei`
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=all`
- `FRIGATE_HOST=frigate`
- `FRIGATE_PORT=5000`

#### Frigate 容器環境變數
- `TZ=Asia/Taipei`
- `FRIGATE_RTSP_PASSWORD=${FRIGATE_RTSP_PASSWORD:-}`（來自 `.env`）
- `YOLO_MODELS=yolo7-tiny-416`

#### `.env`（目前 repo 提供 `.env.example`）
- `TZ`
- `FRIGATE_RTSP_PASSWORD`
- `MODEL_DIR`（預設 `/workspace/models`）
- `DETECT_MODEL_ENGINE`（可填絕對路徑或檔名）
- `DETECT_MODEL_PT`（可填絕對路徑或檔名）
- （可選）`DATABASE_URL`
- （可選）`MQTT_HOST`, `MQTT_PORT`

範例：
```env
MODEL_DIR=/workspace/models
DETECT_MODEL_ENGINE=yolov8n.engine
DETECT_MODEL_PT=yolov8n.pt
```

#### 模型路徑（統一規則）
- 模型統一放置：`/workspace/models`（由 compose 將主機 `./models` 掛載進容器）。
- `DETECT_MODEL_ENGINE` / `DETECT_MODEL_PT` 為絕對路徑時直接使用。
- 若為檔名或相對路徑，程式自動拼成 `${MODEL_DIR}/${值}`。

### B. 手動啟動 API（非 compose）
- README 指令：`uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`
- 依賴需自行準備：Python 套件、Tesseract、OpenCV 可讀 RTSP 的 ffmpeg/gstreamer 環境。
- 主要環境變數
  - `DATABASE_URL`（預設 `sqlite:///./data/violations.db`）
  - `FRIGATE_HOST`（預設 `frigate`）
  - `FRIGATE_PORT`（預設 `5000`）
  - `MODEL_DIR`（預設 `/workspace/models`）
  - `DETECT_MODEL_ENGINE`（預設 `yolov8n.engine`）
  - `DETECT_MODEL_PT`（預設 `yolov8n.pt`）
- 模型檔放在 `MODEL_DIR` 對應目錄（預設為 `/workspace/models`）。

### C. scripts 啟動（測試/批次）
- 檔案：`scripts/detection_service.py`
- 指令範例：
  - `python3 scripts/detection_service.py --source ./source/test.mp4 --api http://localhost:8000`
- 參數
  - `--source`
  - `--camera-id`
  - `--location`
  - `--api`
- 模型路徑
  - 預設使用 env 解析後的 `DETECT_MODEL_PT`

### D. systemd
- 目前 repo **沒有** `.service` 檔（已掃描）。
- 可用 systemd 管理的目標通常會是：
  - `uvicorn api.main:app ...`
  - 或 `python3 scripts/detection_service.py ...`
- 若要 systemd 化，需額外提供 unit file 與環境變數檔（repo 內尚未提供）。

---

## 3) Jetson 硬體/依賴在專案中的使用方式

### JetPack / CUDA / TensorRT
- README 宣告目標平台：JetPack 6.0、CUDA 12.2、TensorRT 8.6。
- Docker base image：`dustynv/l4t-pytorch:r36.2.0`（Jetson L4T r36）。
- Compose API/Frigate 皆設 `runtime: nvidia`。
- Frigate image 使用 `ghcr.io/blakeblackshear/frigate:stable-tensorrt-jp6`。
- Frigate 設定含：
  - `detectors.tensorrt.type: tensorrt`
  - `detectors.tensorrt.device: 0`

### ffmpeg hwaccel
- `config/frigate/config.yml` 目前為：`ffmpeg.hwaccel_args: []`（未啟用 Jetson preset）。
- `api/routes/frigate.py` 在 `sync-cameras` 內僅在 `ffmpeg` 缺失時補 `{"hwaccel_args": "preset-jetson-h264"}`；若 `ffmpeg` 已存在且 `hwaccel_args` 為空，不會自動覆蓋。

### OCR / CV 依賴
- Dockerfile 安裝：`tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-chi-tra`, `ffmpeg`。
- OCR 引擎實作為 `pytesseract`（`recognition/plate_recognizer.py`）。
- 串流讀取主要透過 `cv2.VideoCapture(rtsp_url)`。

---

## 4) 常見故障點與建議排查順序

### 建議排查順序（由外到內）
1. 容器與服務是否啟動
- 檢查 `traffic-api`、`frigate` 是否在跑。
- 檢查 API 健康端點：`/api/health`。

2. RTSP 來源是否可讀
- 優先用 API：`POST /api/cameras/test-url`。
- 失敗時先驗證 URL 格式、帳密、埠號、防火牆、攝影機編碼（H264/H265）。

3. Frigate 連線是否正常
- 檢查 `/api/frigate/status`、`/api/frigate/version`。
- 確認 `FRIGATE_HOST`/`FRIGATE_PORT` 與 Docker 網路一致。

4. 模型檔路徑是否對齊程式實際讀取
- 需特別確認以下檔案存在：
  - `/workspace/models/yolov8n.pt`
  - `/workspace/models/yolov8n.engine`
- 若 `DETECT_MODEL_*` 是相對值，會自動組到 `MODEL_DIR`（預設 `/workspace/models`）。

5. Jetson/TensorRT/加速是否真的生效
- Frigate detector 是否為 `tensorrt`。
- `hwaccel_args` 若為空，視串流負載可能造成 CPU 偏高。

6. OCR 準確率與性能問題
- 先檢查影像品質（解析度、角度、夜間噪聲）。
- 再調整辨識幀率與 ROI，降低無效 OCR 次數。

7. 資料庫與儲存路徑
- `DATABASE_URL` 是否正確。
- `./data`, `./output`, `./storage` 是否可寫。

### 目前掃描到的高機率坑點
- `DETECT_MODEL_*` 值與 `MODEL_DIR` 不一致（例如檔名拼錯、副檔名不符）。
- `ffmpeg.hwaccel_args` 在現有 Frigate config 是空值。
- `requirements.txt` 與 Dockerfile 實際安裝套件不完全一致（例如 README/requirements 提到 EasyOCR/PaddleOCR，但主流程實作是 Tesseract）。

---

## 5) 最小檢查清單（值班版）

- API alive：`GET /api/health` 回 `healthy`
- Frigate alive：`GET /api/frigate/status` 為 `online`
- Camera alive：`POST /api/cameras/test-url` 成功
- Model ready：`MODEL_DIR` 目錄下有 `DETECT_MODEL_PT` / `DETECT_MODEL_ENGINE` 對應檔案
- OCR ready：容器內可執行 Tesseract（`pytesseract` 不報錯）
- Storage ready：`output/`、`storage/`、`data/` 可寫

---

## 6) 登入與 users CRUD（admin）

### Web 操作入口
- 登入頁：`/web/`（未登入會先顯示登入畫面）
- 權限管理頁：側欄 `🔐 權限管理`（僅 admin 可見）
- 使用者管理功能：
  - 新增使用者
  - 修改角色（admin/ops/viewer）
  - 啟用/停用
  - 重設密碼
  - 刪除

### API 操作
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`
- `GET /api/auth/users`（admin）
- `POST /api/auth/users`（admin）
- `PUT /api/auth/users/{id}`（admin）
- `PUT /api/auth/users/{id}/password`（admin）
- `DELETE /api/auth/users/{id}`（admin）

### 內建保護規則
- 不可停用目前登入中的管理者。
- 不可刪除目前登入中的管理者。
- 系統至少保留一位啟用中的管理者。
