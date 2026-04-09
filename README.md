# 🚦 交通違規影像分析系統

基於 NVIDIA Jetson 平台的 AI 邊緣運算交通監控系統，整合車輛偵測、車牌辨識、違規偵測、壅塞分析等功能。

![Platform](https://img.shields.io/badge/Platform-Jetson%20NX-green)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.2-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 目錄

- [系統特色](#-系統特色)
- [系統需求](#-系統需求)
- [專案架構](#-專案架構)
- [安裝部署](#-安裝部署)
- [推送流程](#-推送流程)
- [登入與權限](#-登入與權限)
- [API 文件](#-api-文件)
- [模組說明](#-模組說明)
- [使用指南](#-使用指南)
- [開發指南](#-開發指南)

---

## ✨ 系統特色

| 功能 | 說明 | 技術 |
|------|------|------|
| 🚗 **車輛偵測** | 偵測汽車、機車、公車、卡車、自行車、行人 | YOLOv8n + TensorRT |
| 🚛 **大型車分類** | 大貨車/小貨車/大客車二階段細分類 (Top-1 97.7%) | YOLO26s-cls |
| 🔢 **車牌辨識** | 台灣車牌格式辨識，多重預處理提升準確率 | Tesseract OCR |
| 🚨 **違規偵測** | 闖紅燈、超速、違規停車、逆向行駛 | ROI + 規則引擎 |
| 🚦 **壅塞偵測** | 即時車流密度分析，四級壅塞等級判定 | 佔用率演算法 |
| 📹 **NVR 整合** | Frigate NVR 整合，支援動態偵測與錄影 | Frigate + MQTT |
| 🖥️ **NVR 回放** | EZ Pro 深色主題回放介面，多格分割、時間軸、書籤 | Vue 3 + EZ Pro UI |
| 🔐 **登入與權限** | 帳密登入、角色管理、前台權限勾選派放 | Cookie Session + RBAC UI |
| 🌐 **Web 介面** | 響應式 SPA 管理介面 | Vue 3 + Element Plus |
| 📊 **系統日誌** | 即時監控與連線狀態記錄 | FastAPI + WebSocket |

---

## 💻 系統需求

### 硬體需求
```
裝置: NVIDIA Jetson Xavier NX 8GB (或更高)
儲存: 64GB+ SSD/SD Card
網路: 支援 RTSP 攝影機
```

### 軟體環境
```
系統: JetPack 6.0 (Ubuntu 22.04)
CUDA: 12.2
TensorRT: 8.6
Python: 3.10
Docker: 24.0+
```

---

## 📁 專案架構
```
traffic-violation-detection/
│
├── 📄 Dockerfile                    # Docker 映像建置
├── 📄 docker-compose.yml            # 容器編排設定
├── 📄 requirements.txt              # Python 依賴
├── 📄 README.md                     # 本文件
│
├── 📂 api/                          # FastAPI 後端服務
│   ├── 📄 main.py                   # API 入口點
│   ├── 📄 models.py                 # SQLAlchemy 資料模型
│   └── 📂 routes/                   # API 路由模組
│       ├── 📄 auth.py               # 登入/登出/目前使用者
│       ├── 📄 cameras.py            # 攝影機 CRUD + 連線測試
│       ├── 📄 violations.py         # 違規事件管理
│       ├── 📄 stream.py             # 即時串流 + 偵測服務
│       ├── 📄 frigate.py            # Frigate NVR 整合
│       ├── 📄 lpr.py                # 車牌辨識 (單張)
│       ├── 📄 lpr_stream.py         # 車牌辨識串流
│       ├── 📄 lpr_visual.py         # LPR 視覺化串流
│       ├── 📄 congestion.py         # 壅塞偵測服務
│       └── 📄 logs.py               # 系統日誌服務
│
├── 📂 detection/                    # 偵測模組
│   ├── 📄 vehicle_detector.py       # YOLOv8 車輛偵測 (含大型車分類整合)
│   ├── 📄 truck_classifier.py       # YOLO26s 大型車細分類器
│   ├── 📄 violation_detector.py     # 違規偵測邏輯
│   └── 📄 congestion_detector.py    # 壅塞偵測器
│
├── 📂 recognition/                  # 辨識模組
│   ├── 📄 plate_recognizer.py       # Tesseract 車牌 OCR
│   └── 📄 frigate_integration.py    # Frigate 事件整合
│
├── 📂 web/                          # 前端介面
│   ├── 📄 index.html                # Vue 3 SPA 主頁
│   ├── 📄 nvr_playback.html         # NVR 回放介面 (EZ Pro 深色主題)
│   ├── 📄 roi_editor.html           # ROI 編輯器
│   └── 📂 fonts/                    # 字型檔（含 CJK 疊加字型）
│
├── 📂 config/                       # 設定檔
│   └── 📂 frigate/
│       └── 📄 config.yml            # Frigate NVR 設定
│
├── 📂 models/                       # AI 模型 (不納入版控)
│   ├── 📄 yolov8n.pt                # YOLOv8 PyTorch 偵測模型
│   ├── 📄 yolov8n.engine            # TensorRT 加速模型
│   └── 📄 truck_cls_yolo26s.pt      # YOLO26s 大型車分類模型
│
├── 📂 storage/                      # 資料儲存 (不納入版控)
│   ├── 📂 violations/               # 違規截圖
│   ├── 📂 lpr_snapshots/            # 車牌辨識截圖
│   └── 📂 frigate/                  # Frigate 錄影
│
└── 📂 data/                         # 資料庫
    └── 📄 violations.db             # SQLite 資料庫
```

---

## 🔄 系統架構圖
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              使用者介面層                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    web/index.html (Vue 3 SPA)                        │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ 儀表板   │ │ 攝影機   │ │ 違規管理 │ │ 車牌辨識 │ │ 系統日誌 │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │ HTTP / WebSocket
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API 服務層 (FastAPI :8000)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ /api/cameras      攝影機管理 (CRUD, 連線測試)                        │   │
│  │ /api/violations   違規事件 (查詢, 審核, 統計)                        │   │
│  │ /api/stream       即時串流 (MJPEG, 偵測啟停)                         │   │
│  │ /api/lpr          車牌辨識 (單張/串流/視覺化)                        │   │
│  │ /api/congestion   壅塞偵測 (啟停/狀態/串流)                          │   │
│  │ /api/frigate      NVR 整合 (設定/事件/錄影)                          │   │
│  │ /api/traffic      交通報表 (VD 報表/事件查詢)                        │   │
│  │ /api/system       系統管理 (NTP/NX/硬體狀態)                        │   │
│  │ /api/logs         系統日誌 (即時/查詢/清除)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   偵測模組       │       │   辨識模組       │       │   儲存層         │
│ ┌─────────────┐ │       │ ┌─────────────┐ │       │ ┌─────────────┐ │
│ │VehicleDetect│ │       │ │PlateRecogniz│ │       │ │  SQLite DB  │ │
│ │ (YOLOv8)    │ │       │ │ (Tesseract) │ │       │ │  violations │ │
│ └─────────────┘ │       │ └─────────────┘ │       │ │  cameras    │ │
│ ┌─────────────┐ │       │ ┌─────────────┐ │       │ └─────────────┘ │
│ │Congestion   │ │       │ │ Frigate     │ │       │ ┌─────────────┐ │
│ │ Detector    │ │       │ │ Integration │ │       │ │ File Storage│ │
│ └─────────────┘ │       │ └─────────────┘ │       │ │ screenshots │ │
│ ┌─────────────┐ │       └─────────────────┘       │ └─────────────┘ │
│ │Violation    │ │                                 └─────────────────┘
│ │ Detector    │ │
│ └─────────────┘ │
└─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              外部服務層                                      │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ Frigate NVR (:5000)         │  │ IP 攝影機 (RTSP)                    │  │
│  │ ├─ 動態偵測                 │  │ ├─ rtsp://user:pass@ip:port/path    │  │
│  │ ├─ 事件錄影                 │  │ └─ H.264/H.265 編碼                 │  │
│  │ └─ MQTT 推送                │  └─────────────────────────────────────┘  │
│  └─────────────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 安裝部署

### 方式一：Docker 部署 (推薦)
```bash
# 1. 克隆專案
git clone https://github.com/your-repo/traffic-violation-detection.git
cd traffic-violation-detection

# 2. 建立環境變數
cp .env.example .env

# 3. 啟動服務
docker compose up -d

# 4. 查看日誌
docker logs -f traffic-api
```

### 方式二：手動安裝
```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 安裝 Tesseract OCR
sudo apt install tesseract-ocr tesseract-ocr-chi-tra

# 3. 下載 YOLOv8 模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
# 4. (可選) 放置 TensorRT 模型
# cp /path/to/yolov8n.engine models/yolov8n.engine

# 5. 啟動 API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 存取服務

| 服務 | URL | 說明 |
|------|-----|------|
| Web 介面 | http://localhost:8000/web/ | 管理介面 |
| NVR 回放 | http://localhost:8000/web/nvr_playback.html | EZ Pro 風格回放介面 |
| API 文件 | http://localhost:8000/docs | Swagger UI |
| API Reference | [docs/API_REFERENCE.md](./docs/API_REFERENCE.md) | 完整 API 文件 (95+ 端點) |
| Frigate NVR | http://localhost:5000 | NVR 介面 |

**預設登入帳號（首次初始化）**
```
username: admin
password: admin123
```

### 部署後自動檢查（建議每次必跑）
```bash
# API 健康 + 交通事件區間查詢 + VD 前端關鍵標記
python3 scripts/smoke_check.py --base-url http://127.0.0.1:8000 --timeout 60

# 一鍵：重啟 API 並自動驗證
./scripts/restart_and_verify.sh http://127.0.0.1:8000 60
```

### 現場上板部署（無網路 Jetson）
1. 開發機打包映像
```bash
docker compose build api
docker save -o traffic-api_latest.tar traffic-api:latest
python3 scripts/settings_backup.py export
```

2. 將檔案複製到板端（可用 `scp` / 隨身碟）
```bash
scp traffic-api_latest.tar <board_user>@<board_ip>:/home/<board_user>/deploy/
scp config/settings_backup.json <board_user>@<board_ip>:/home/<board_user>/deploy/
```

3. 板端套版（進入專案根目錄）
```bash
# 可選：先備份板端設定
python3 scripts/settings_backup.py export

# 載入新映像
docker load -i /home/<board_user>/deploy/traffic-api_latest.tar

# 套用設定（不含辨識資料）
python3 scripts/settings_backup.py import --file /home/<board_user>/deploy/settings_backup.json

# 重啟並驗證
./scripts/restart_and_verify.sh http://127.0.0.1:8000 60
```

4. 回滾（若新版本異常）
```bash
docker images | rg traffic-api
docker tag traffic-api:<old_tag> traffic-api:latest
./scripts/restart_and_verify.sh http://127.0.0.1:8000 60
```

---

## 📤 推送流程

已整理完整操作文件：[`推送流程.md`](./推送流程.md)

內容包含：
- GitHub SSH 金鑰設定與驗證
- 專案推送標準步驟（branch/commit/push）
- 設定備份流程（`config/settings_backup.json`）
- Docker 有網打包、現場離線部署流程

---

## 🔐 登入與權限

### 登入機制
- Web 首頁未登入時會顯示登入頁。
- 後端透過 `HttpOnly` Cookie 維持 Session。
- 支援 API：
  - `POST /api/auth/login`
  - `GET /api/auth/me`
  - `POST /api/auth/logout`

### 角色模型
- `admin`：可管理權限派放與所有功能
- `ops`：可使用營運/維運功能（預設不含權限管理）
- `viewer`：依派放權限只讀/部分可見

### 權限派放（前台）
- Web 側欄新增 `🔐 權限管理`（僅 `admin` 可見）。
- 可對 `admin / ops / viewer` 勾選功能可見權限。
- 權限即時影響：
  - 側欄功能顯示
  - 頁面可存取性（無權限頁會自動導回可用頁）

---

## 📡 API 文件

> 完整 API 文件（含 Request/Response 範例）請參閱 **[docs/API_REFERENCE.md](./docs/API_REFERENCE.md)**

以下為各模組端點摘要：

### 認證 `/api/auth`

| 方法 | 端點 | 說明 |
|------|------|------|
| `POST` | `/api/auth/login` | 使用帳密登入 |
| `GET` | `/api/auth/me` | 取得目前登入使用者 |
| `POST` | `/api/auth/logout` | 登出 |
| `GET` | `/api/auth/users` | 取得使用者列表（admin） |
| `POST` | `/api/auth/users` | 新增使用者（admin） |
| `PUT` | `/api/auth/users/{id}` | 更新角色/啟停用（admin） |
| `PUT` | `/api/auth/users/{id}/password` | 重設密碼（admin） |
| `DELETE` | `/api/auth/users/{id}` | 刪除使用者（admin） |

**users CRUD 規則**
- 只有 `admin` 可管理使用者。
- 不可刪除或停用目前登入中的管理者。
- 系統至少保留一位啟用中的 `admin`。

### 攝影機管理 `/api/cameras`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/cameras` | 取得所有攝影機 |
| `GET` | `/api/cameras/{id}` | 取得單一攝影機 |
| `POST` | `/api/cameras` | 新增攝影機 |
| `PUT` | `/api/cameras/{id}` | 更新攝影機 |
| `DELETE` | `/api/cameras/{id}` | 刪除攝影機 |
| `POST` | `/api/cameras/{id}/test` | 測試連線 |
| `POST` | `/api/cameras/test-url` | 測試 RTSP URL |

**新增攝影機範例：**
```bash
curl -X POST http://localhost:8000/api/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "name": "前門攝影機",
    "source": "rtsp://admin:password@192.168.1.100:554/stream1",
    "location": "大門入口",
    "detection_config": {
      "red_light": true,
      "speeding": true,
      "illegal_parking": true
    }
  }'
```

---

### 違規管理 `/api/violations`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/violations` | 查詢違規列表 |
| `GET` | `/api/violations/{id}` | 取得違規詳情 |
| `PUT` | `/api/violations/{id}/review` | 審核違規 |
| `GET` | `/api/violations/statistics` | 違規統計 |

**查詢參數：**
```
?status=pending          # 狀態過濾
&violation_type=RED_LIGHT # 類型過濾
&license_plate=ABC-1234   # 車牌過濾
&page=1&page_size=20      # 分頁
```

---

### 即時串流 `/api/stream`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/stream/{id}/live` | MJPEG 即時串流 |
| `GET` | `/api/stream/{id}/live-overlay` | MJPEG 疊加串流（ROI/辨識） |
| `GET` | `/api/stream/{id}/snapshot` | 取得截圖 |
| `POST` | `/api/stream/{id}/detection/start` | 啟動偵測 |
| `POST` | `/api/stream/{id}/detection/stop` | 停止偵測 |
| `GET` | `/api/stream/detection/all` | 所有偵測狀態 |

---

### 車牌辨識 `/api/lpr`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/lpr/status` | LPR 服務狀態 |
| `POST` | `/api/lpr/recognize-upload` | 上傳圖片辨識 |
| `POST` | `/api/lpr/recognize-camera/{id}` | 攝影機截圖辨識 |
| `POST` | `/api/lpr/stream/start/{id}` | 啟動串流辨識 |
| `POST` | `/api/lpr/stream/stop/{id}` | 停止串流辨識 |
| `GET` | `/api/lpr/stream/status/{id}` | 串流辨識狀態 |
| `GET` | `/api/lpr/stream/results/{id}` | 取得辨識結果 |
| `GET` | `/api/lpr/visual/stream/{id}` | 視覺化串流 |

**上傳辨識範例：**
```bash
curl -X POST http://localhost:8000/api/lpr/recognize-upload \
  -F "file=@plate_image.jpg"
```

**回應：**
```json
{
  "plate_number": "ABC-1234",
  "confidence": 0.92,
  "valid": true,
  "type": "一般",
  "vehicle_type": "car"
}
```

---

### 壅塞偵測 `/api/congestion`

| 方法 | 端點 | 說明 |
|------|------|------|
| `POST` | `/api/congestion/{id}/start` | 啟動壅塞偵測 |
| `POST` | `/api/congestion/{id}/stop` | 停止壅塞偵測 |
| `GET` | `/api/congestion/{id}/status` | 取得壅塞狀態 |
| `GET` | `/api/congestion/status/all` | 所有壅塞狀態 |
| `GET` | `/api/congestion/{id}/snapshot` | 壅塞分析截圖 |
| `GET` | `/api/congestion/{id}/stream` | 壅塞視覺化串流 |

**壅塞狀態回應：**
```json
{
  "running": true,
  "result": {
    "vehicle_count": 15,
    "occupancy": 0.42,
    "level": "medium",
    "level_name": "中等",
    "vehicle_stats": {"car": 10, "motorcycle": 5}
  }
}
```

**壅塞等級定義：**
| 等級 | 佔用率 | 說明 |
|------|--------|------|
| `low` | < 20% | 暢通 |
| `medium` | 20-40% | 中等 |
| `high` | 40-60% | 擁擠 |
| `critical` | > 60% | 嚴重壅塞 |

---

### NVR 整合 `/api/frigate`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/frigate/status` | NVR 狀態 |
| `GET` | `/api/frigate/cameras` | NVR 攝影機列表 |
| `POST` | `/api/frigate/camera` | 新增 NVR 攝影機 |
| `DELETE` | `/api/frigate/camera/{name}` | 刪除攝影機 |
| `PUT` | `/api/frigate/camera/{name}/switch` | 單台錄影/偵測開關 |
| `GET` | `/api/frigate/camera/{name}/motion-roi` | 取得 Motion ROI |
| `PUT` | `/api/frigate/camera/{name}/motion-roi` | 更新 Motion ROI |
| `GET` | `/api/frigate/events` | 取得事件 |
| `POST` | `/api/frigate/sync-cameras` | 同步攝影機 |
| `POST` | `/api/frigate/restart` | 重啟 NVR |

---

### 系統日誌 `/api/logs`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/logs` | 取得日誌 |
| `GET` | `/api/logs/query` | 查詢日誌 (含篩選與分頁) |
| `DELETE` | `/api/logs` | 清除日誌 |

### 交通報表 `/api/traffic`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/traffic/vd-report` | 車輛偵測報表 (含聚合) |
| `GET` | `/api/traffic/events` | 列出交通事件 |

### 系統管理 `/api/system`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/system/status` | 系統硬體狀態 (CPU/GPU/Memory/Disk) |
| `GET` | `/api/system/ntp/settings` | 取得 NTP 設定 |
| `PUT` | `/api/system/ntp/settings` | 更新 NTP 設定 |
| `POST` | `/api/system/ntp/sync-now` | 手動 NTP 同步 |
| `GET` | `/api/system/nx/settings` | 取得 NX/VMS 設定 |
| `PUT` | `/api/system/nx/settings` | 更新 NX/VMS 設定 |

### NX VMS `/api/nx`

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/api/nx/devices` | 列出 NX VMS 設備 |
| `GET` | `/api/nx/stream/{device_id}` | 從 NX 設備串流 |

---

## 🔧 模組說明

### 1. 車輛偵測模組 `detection/vehicle_detector.py`
```python
class VehicleDetector:
    """YOLOv8 車輛偵測器 + 大型車二階段分類"""
    
    VEHICLE_CLASSES = {
        0: 'person',      # 行人
        1: 'bicycle',     # 自行車
        2: 'car',         # 汽車
        3: 'motorcycle',  # 機車
        5: 'bus',         # 公車
        7: 'truck'        # 卡車
    }
    
    def __init__(self, model_path=None, conf_threshold=0.5, enable_truck_cls=True):
        """初始化偵測器，自動載入 TruckClassifier"""
        
    def detect(self, frame) -> List[Dict]:
        """
        偵測影像中的車輛
        偵測到 truck/bus 時自動觸發二階段分類
        
        Returns:
            [{'class_name': 'heavy_truck', 'confidence': 0.85, 'bbox': {...},
              'truck_cls': {'label': '大貨車', 'confidence': 0.92, 'group': 'large'}}, ...]
        """
        
    def detect_with_draw(self, frame) -> Tuple[ndarray, List]:
        """偵測並繪製標註框（含大型車分類標籤）"""
```

### 1.1 大型車分類模組 `detection/truck_classifier.py`

YOLO26s-cls 二階段細分類器，對偵測出的 truck/bus 做精細分類。

```python
class TruckClassifier:
    """大型車輛細分類器 (YOLO26s-cls, Top-1 Acc: 97.7%)"""
    
    def classify(self, frame, bbox) -> Dict:
        """
        對 bounding box 區域做分類
        
        Returns:
            {'class_name': 'heavy_truck', 'label': '大貨車',
             'confidence': 0.92, 'group': 'large', 'length_m': 12.0}
        """
```

**分類類別：**

| 類別 | 中文 | 等效長度 | 分組 |
|------|------|---------|------|
| `heavy_truck` | 大貨車 | 12.0m | large |
| `light_truck` | 小貨車 | 6.0m | small |
| `bus` | 大客車 | 12.0m | large |
| `non_truck` | 非目標 | 6.0m | other |

**模型訓練結果：**
- 訓練資料：8,648 張標註圖片 (train 6,723 / val 840 / test 846)
- **Val Top-1 Accuracy: 97.74%** | Test Top-1: 96.93%
- 推論速度：1.7ms/張 (GTX 1660 SUPER)
- 模型大小：11MB (`truck_cls_yolo26s.pt`)

---

### 2. 壅塞偵測模組 `detection/congestion_detector.py`
```python
class CongestionDetector:
    """壅塞偵測器 - 計算車流密度與佔用率"""
    
    LEVEL_NAMES = {
        'low': '暢通',
        'medium': '中等', 
        'high': '擁擠',
        'critical': '嚴重壅塞'
    }
    
    def __init__(self, vehicle_detector=None):
        """初始化，可共用 VehicleDetector 實例"""
        
    def analyze(self, frame, zones=None) -> Dict:
        """
        分析壅塞程度
        
        Args:
            frame: BGR 影像
            zones: ROI 區域設定 (來自攝影機設定)
            
        Returns:
            {
                'vehicle_count': 15,
                'occupancy': 0.42,
                'level': 'medium',
                'level_name': '中等',
                'vehicle_stats': {'car': 10, 'motorcycle': 5},
                'vehicles': [...]
            }
        """
```

**演算法流程：**
```
1. YOLOv8 偵測車輛
2. 過濾 ROI 區域內車輛 (無 ROI 則全景)
3. 計算車輛佔用面積
4. 佔用率 = 車輛面積 / ROI 面積
5. 歷史平滑 (10 幀移動平均)
6. 判定壅塞等級
```

---

### 3. 車牌辨識模組 `recognition/plate_recognizer.py`
```python
class PlateRecognizer:
    """台灣車牌辨識器 - 多重預處理 + Tesseract OCR"""
    
    PLATE_PATTERNS = [
        r'^[A-Z]{3}-\d{4}$',    # ABC-1234 (新式)
        r'^[A-Z]{2}-\d{4}$',    # AB-1234 (舊式)
        r'^\d{4}-[A-Z]{2}$',    # 1234-AB
        r'^[A-Z]{3}-\d{3}$',    # ABC-123 (機車)
        # ... 更多格式
    ]
    
    def recognize(self, img) -> Dict:
        """
        辨識車牌
        
        Returns:
            {
                'plate_number': 'ABC-1234',
                'confidence': 0.92,
                'valid': True,
                'type': '一般'
            }
        """
        
    def preprocess(self, img) -> List[ndarray]:
        """多重預處理 (6 種方式)"""
        # 1. 原圖灰階
        # 2. CLAHE 增強
        # 3. Otsu 二值化
        # 4. 反轉二值化
        # 5. 自適應二值化
        # 6. 銳化
        
    def perspective_transform(self, img) -> ndarray:
        """透視變換校正傾斜車牌"""
        
    def _validate(self, plate) -> bool:
        """驗證台灣車牌格式"""
```

**LPR 處理流程：**
```
RTSP 輸入
    │
    ▼
YOLOv8n (車輛偵測)
    │ car, motorcycle, bus, truck
    ▼
ROI 裁切 (車牌區域定位)
    │
    ▼
多重預處理 (6 種方式)
    │
    ▼
Tesseract OCR
    │
    ▼
格式驗證 (台灣車牌)
    │
    ▼
輸出結果
```

---

### 4. 違規偵測模組 `detection/violation_detector.py`
```python
class ViolationType(Enum):
    """違規類型"""
    RED_LIGHT = "闖紅燈"
    SPEEDING = "超速"
    ILLEGAL_PARKING = "違規停車"
    WRONG_WAY = "逆向行駛"
    NO_HELMET = "未戴安全帽"
    SIDEWALK = "騎樓違停"

class ViolationEvent:
    """違規事件"""
    violation_type: ViolationType
    vehicle_type: str
    license_plate: str
    confidence: float
    bbox: Dict
    timestamp: datetime
    
class VehicleTracker:
    """車輛追蹤器 (簡易版)"""
    
class ViolationDetector:
    """違規偵測器"""
    
    def detect_violations(self, frame, detections, zones) -> List[ViolationEvent]:
        """偵測違規行為"""
```

---

### 5. 系統日誌模組 `api/routes/logs.py`
```python
def add_log(level: str, message: str, source: str = "system"):
    """
    新增日誌 (供其他模組呼叫)
    
    Args:
        level: info / warning / error / success
        message: 日誌訊息
        source: 來源 (system / camera / lpr / congestion)
    """

# 使用範例
from api.routes.logs import add_log

add_log("info", "開始測試攝影機連線", "camera")
add_log("success", "連線成功: 前門攝影機 (1920x1080)", "camera")
add_log("error", "無法連線: 後門攝影機", "camera")
```

---

### 6. NVR 回放介面 `web/nvr_playback.html`

參考 EZ Pro NVR 設計的深色主題回放頁面。

**功能特色：**

| 區域 | 功能 |
|------|------|
| 左側 Resource Tree | 攝影機搜尋、Server/NVR/歷史三層分組、拖放到 Grid |
| 中央 Viewing Grid | 1x1 / 2x2 / 3x3 分割、Camera name + 時間疊圖 |
| 右側 Panel | 事件/通知/書籤三分頁、統計摘要、事件跳轉 |
| 底部 Timeline | 分布圖、事件標記、Playhead、時間刻度 |
| 底部 Controls | 播放/暫停、速度 0.5x-8x、截圖、書籤、時間篩選 |

**快捷鍵：**
- `Space` — 播放/暫停
- `B` — 加入書籤
- `←` / `→` — 快進/後退 5 秒

訪問：`http://localhost:8000/web/nvr_playback.html`

---

## 📖 使用指南

### 新增攝影機

1. 進入「攝影機管理」頁面
2. 點擊「新增」按鈕
3. 填寫資訊：
   - 名稱：攝影機識別名稱
   - IP：攝影機 IP 位址
   - 帳號/密碼：RTSP 認證資訊
   - 埠號：RTSP 埠號 (預設 554)
   - 路徑：串流路徑
4. 點擊「測試」確認連線
5. 點擊「儲存」

### 設定 ROI 區域

1. 在攝影機管理點擊「設定」
2. 切換到「偵測區域」分頁
3. 在預覽畫面點擊設定多邊形頂點
4. 選擇區域類型（偵測區域/排除區域）
5. 點擊「儲存區域」

### 啟動壅塞偵測

1. 進入「即時監控」頁面
2. 找到目標攝影機
3. 點擊「🚦壅塞」按鈕
4. 進入「系統日誌」查看分析結果

### 查看系統日誌

1. 點擊左側選單「系統監控日誌」
2. 日誌會即時更新
3. 可依等級過濾 (info/warning/error/success)
4. 點擊「清除」清空日誌

---

## 🛠️ 開發指南

### 新增 API 路由
```python
# api/routes/my_feature.py
from fastapi import APIRouter
from api.routes.logs import add_log

router = APIRouter(prefix="/api/my-feature", tags=["我的功能"])

@router.get("/status")
async def get_status():
    add_log("info", "查詢狀態", "my-feature")
    return {"status": "ok"}
```
```python
# api/main.py 註冊路由
from api.routes import my_feature
app.include_router(my_feature.router)
```

### 新增偵測模組
```python
# detection/my_detector.py
class MyDetector:
    def __init__(self):
        print("✅ MyDetector 初始化完成")
        
    def detect(self, frame):
        # 實作偵測邏輯
        return results
```

### 資料庫模型
```python
# api/models.py
class MyModel(Base):
    __tablename__ = "my_table"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## 📝 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `DATABASE_URL` | `sqlite:///./data/violations.db` | 資料庫連線 |
| `FRIGATE_HOST` | `frigate` | Frigate 主機 |
| `FRIGATE_PORT` | `5000` | Frigate 埠號 |
| `MODEL_DIR` | `/home/ubuntu/traffic-violation-detection/models` | 模型目錄 |
| `DETECT_MODEL_ENGINE` | `yolov8n.engine` | 偵測 engine 模型（可填絕對路徑或檔名） |
| `DETECT_MODEL_PT` | `yolov8n.pt` | 偵測 pt 模型（可填絕對路徑或檔名） |
| `TRUCK_CLS_MODEL` | `truck_cls_yolo26s.pt` | 大型車分類模型（可填絕對路徑或檔名） |
| `DEVICE` | `cuda:0` | 推論裝置 |
| `TZ` | `Asia/Taipei` | 時區 |

### 模型路徑規則

- 所有偵測模型統一放置於容器 `/workspace/models`（主機端 `./models`）。
- `DETECT_MODEL_ENGINE` / `DETECT_MODEL_PT` 若為絕對路徑（`/` 開頭）則直接使用。
- 若為檔名或相對路徑，程式會自動拼成 `${MODEL_DIR}/${值}`。

範例 `.env`：
```env
MODEL_DIR=/workspace/models
DETECT_MODEL_ENGINE=yolov8n.engine
DETECT_MODEL_PT=yolov8n.pt
```

---

## 🔍 故障排除

### 攝影機連線失敗
```bash
# 測試 RTSP 連線
docker exec traffic-api python3 -c "
import cv2
cap = cv2.VideoCapture('rtsp://user:pass@ip:port/path')
print('Connected:', cap.isOpened())
cap.release()
"
```

### 查看 API 日誌
```bash
docker logs -f traffic-api
```

### 重啟服務
```bash
docker restart traffic-api
```

---

## 📄 授權

MIT License

---

## 👥 貢獻

歡迎提交 Issue 和 Pull Request！

---

*最後更新: 2026-04-09*
