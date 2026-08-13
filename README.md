# 🚦 交通違規影像分析系統

基於 NVIDIA Jetson 平台的 AI 邊緣運算交通監控系統，整合車輛偵測、車牌辨識、違規偵測、壅塞分析等功能。

![Platform](https://img.shields.io/badge/Platform-Jetson%20AGX%20Orin-green)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.6-brightgreen)
![TensorRT](https://img.shields.io/badge/TensorRT-10.3-brightgreen)
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
- [環境變數](#-環境變數)
- [故障排除](#-故障排除)

**延伸文件**：[`ocr 流程.md`](./ocr%20流程.md)（車牌辨識架構）、
[`RUNBOOK.md`](./RUNBOOK.md)（運維）、
[`DEPLOY_NEW_SITE.md`](./DEPLOY_NEW_SITE.md)（新站台部署）、
[`電子鎖_README.md`](./電子鎖_README.md)、
[`API整合文件.md`](./API整合文件.md)（對外規格）

---

## ✨ 系統特色

| 功能 | 說明 | 技術 |
|------|------|------|
| 🚗 **車輛偵測** | 偵測汽車、機車、公車、卡車、自行車、行人 | YOLOv8n + TensorRT |
| 🚛 **大型車分類** | 大貨車/小貨車/大客車二階段細分類 (Top-1 97.7%) | YOLO26s-cls |
| 🔢 **車牌辨識** | 台灣車牌格式辨識，5 變體 ensemble + 多幀投票 + 格式修復 | YOLO 字元偵測微服務 (:8010) |
| 🔒 **電子鎖 / IO** | 箱門門磁、刷卡紀錄、三色燈號、DI/DO 控制 | RS-485 Modbus + 獨立 daemon |
| 🅿️ **停車場管理** | 車位佔用判定、幾何編輯器、VLM 仲裁 | YOLO + 分類器 + Qwen2-VL |
| 🚨 **違規偵測** | 闖紅燈、超速、違規停車、逆向行駛 | ROI + 規則引擎 + 信心度 sanity gate |
| 🎯 **速度精準測量** | Trip wire ±1-2 km/h / Homography ±2-5 km/h / Vanishing-point auto-cal | OpenCV findHomography + Kalman filter |
| 🚦 **壅塞偵測** | 即時車流密度分析，四級壅塞等級判定 | 佔用率演算法 |
| 📹 **NVR 整合** | Frigate NVR 整合，支援動態偵測與錄影 | Frigate + MQTT |
| 🖥️ **NVR 回放** | EZ Pro 深色主題回放介面，多格分割、時間軸、書籤 | Vue 3 + EZ Pro UI |
| 🔐 **登入與權限** | 帳密登入、角色管理、前台權限勾選派放 | Cookie Session + RBAC UI |
| 🌐 **Web 介面** | 響應式 SPA 管理介面 | Vue 3 + Element Plus |
| 📊 **系統日誌** | 即時監控與連線狀態記錄 | FastAPI + WebSocket |

---

## 🆕 最近更新 (2026-08)

### 車牌辨識架構（已驗收）
- OCR 主線是 **YOLO 字元偵測微服務 `:8010`**，不是 Tesseract；Tesseract 相關程式碼仍在檔案內但不執行
- 5 變體 ensemble（original / clahe / upscale_2x / bilateral / gray_otsu）加權投票
- 字元模型 2026-07-13 finetune 版上線（`Charcter-LP.pt`），可回滾
- 格式修復兩層：微服務內 `_repair_plate` + 存 DB 前 `_enforce_plate_format`（只修不丟、idempotent）
- 完整流程見 [`ocr 流程.md`](./ocr%20流程.md)

### 電子鎖 / IO 模組
- IO RS-485 拆成獨立 systemd unit（`traffic-io`，`127.0.0.1:8011`），SEGV 不會拖垮主服務
- 同匯流排支援兩顆電子鎖（addr 2 = 後門 / addr 3 = 前門），含位址掃描與變更
- 門磁為 NO 常開接點：**門關 = 接點閉合 = raw 1**（與協議文件相反，以現場實測為準）
- 三色燈號解耦：通訊故障亮紅燈時，綠燈（運作中）不受影響，但壓制「下載中」白燈閃爍

### 攝影機設定自動同步
- 攝影機新增/修改/刪除後自動改寫 Frigate `go2rtc.streams` 與 `cameras`，去抖後重啟
- 修掉「DB 是新 IP、config.yml 還是舊 IP」造成的半死狀態

### NTP 校時
- 設定改寫 systemd **drop-in**（主檔會被 drop-in 蓋掉）
- drop-in 檔名必須 `zz-` 開頭，才排在 Jetson 出廠的 `nv-fallback-ntp.conf` 之後
- 現場為封閉網段，**不留外網 fallback**

### 保存政策
- 照片 30 天 / 錄影 3 天 / `congestion_samples` 30 天
- 測試用攝影機媒體只留 3 天（`--camera-days`），每日 02:30 釋出空間

---

## 先前更新 (2026-05)

### Dashboard V3 主頁重做
- 移除資訊重複（hero meta / KPI / service health 三處不再顯示同資料）
- 3 個焦點 KPI（今日違規 / 待審核 / 系統健康）+ 4 服務 dot 縮寫
- 24h 違規趨勢柱狀圖（後端 `hourly_buckets`，台北時區 hour-of-day）
- 違規明細 full-width table（時間 / 車牌 / 違規類型 / 地點）
- 後端 `/api/dashboard` 新欄位：`offline_cameras` (真實故障) / `disabled_cameras` (主動關閉) / `enabled_cameras` / `hourly_buckets`
- 系統健康 KPI 自動 unlock guard（Frigate uptime > 60s 自動 reset 卡死的 NVR 重啟 overlay）

### Design System 全站套用
- 公務藍 `#0b5ea8` + Noto Sans TC + IBM Plex Mono 統一 token
- 16 個 page hero block（編號 / 標題 / sub / meta tags）
- Element Plus 全元件覆寫：button / tag / input / table / dialog / message-box / scrollbar / selection
- 11 個 dialog 統一視覺：公務藍漸層 header + 黃 accent stripe + backdrop blur
- camera-card / list-item hover micro-interaction
- 全域 inline color tokenization：`[style*="color:#xxx"]` 抓 hardcode 顏色強制換 design tokens

### LPR 辨識優化
- OCR service 加台灣車牌格式修復 `_repair_plate`（含 I/O/Q blacklist）
- LPR pipeline `_is_plausible_plate` 雙層過濾，擋 OCR 常見誤判
- 配合既有 `_PLATE_FORMATS`（7 種台灣車牌格式組合）

### MQTT 整合強化
- TCP probe + reason_code 翻譯：broker 未啟動時顯示 actionable hint
- publish endpoint 503 訊息明確（提示用戶安裝 mosquitto 或檢查網路）
- subscribe pattern 預設 `traffic/#`（含自己 publish 的 echo + 外部設備上報）
- embedded mode + external mode 切換

### CI/CD 自動部署
- Jetson self-hosted runner (label: `self-hosted, linux, arm64, jetson, gpu, cuda`)
- workflow `Jetson Device Verification`：GPU/CUDA/Python/YOLO/DB/API/Frigate 8 steps verify
- 自動 deploy step：verify 通過 + main branch push → `git pull production` + `sudo systemctl restart traffic-api` + 驗 commit hash
- Trivy security scan：Dockerfile USER + permissions security-events:write 已過

### Frigate / NVR 修補
- `/api/frigate/recordings/play` 加 HEAD method 支援（前端 onError 探錯用，舊版 405）
- video tag 加 `muted` 屬性繞 browser autoplay policy
- offline_cameras 邏輯區分「真實故障」vs「主動關閉」
- camera record.enabled 預設 false 修補腳本（cam_2/3/4/6）

### Health Probe 防誤判
- `/api/health` probe timeout 4s → 8s
- 連續 3 次失敗才切 fault（避免 Jetson CPU spike 誤判）

完整 commit log: `git log --oneline -30`

---


## 🎯 速度偵測精度方案

本系統提供 **三層速度測量** + **自動校準**，依精度由高至低：

### 1. Trip Wire 跨線測速（最準 ±1-2 km/h）⭐
原理：同物理感應線圈 — 路面畫兩條已知間距的線，車跨第一條記時間 t1、跨第二條記 t2 → speed = distance ÷ (t2 - t1)

**設定**：在 ROI 編輯器新增 `speed_line_in` + `speed_line_out` 兩個 zone（同 `lane_no`），輸入 `line_distance_m`（兩線間實際距離 m）

**使用 API**：
- `POST /api/cameras/{id}/calibrate/suggest-tw-distance` — 用畫面車輛寬度 (1.8m car / 2.5m truck) 反推建議距離
- `POST /api/cameras/{id}/calibrate/apply-tw-distance?distance_m=X` — 套用真實距離

### 2. Homography 透視校正（±2-5 km/h）
原理：4 點對應已知矩形 → `cv2.findHomography` 算透視矩陣 → bbox 底中點投影到 world (m, m) 平面 → 連續測速

**設定**：
- 畫 `speed_roi` zone（4 點 TL→TR→BR→BL 順時針）
- 設定 `calibration_width_m`（橫向真實寬）+ `calibration_length_m`（沿車流方向真實長）

### 3. Vanishing Point 自動校正
**API**：`POST /api/cameras/{id}/auto-calibrate/apply` — 偵測車道線 → 算消失點 → 自動建立 4-point speed_roi

**UI**：speed_roi 編輯區「🎯 自動校正 (vanishing point)」按鈕一鍵建 zone

### 4. Sanity Gate（共用）
所有 SPEEDING violation 必須通過：
- speed ∈ [5, 200] km/h
- track ≥ 5 frames（沒 trip wire 校準時要 ≥ 8 frames）
- speed > limit + margin（真超速）
- 同 track 5 秒 dedup（避免 mp4 loop 灌爆）

### 5. 平滑與容錯
- **5-frame 滑動窗口**：取最舊→最新的 displacement / 總時間
- **Median 多 sample**：window 內 instantaneous speeds 取中位數
- **Kalman filter (CV)**：常速度模型 (x, y, vx, vy)，per-track，世界座標時啟用
- **Outlier reject**：raw > 2× prev + 30 km/h 視為錯抓
- **Per-class bbox bottom offset**：car 2%、truck 5%、heavy_truck 6%、bus 7%（接地點補償）

---

## 💻 系統需求

### 硬體需求（現場實機）
```
裝置: NVIDIA Jetson AGX Orin Developer Kit
系統碟: eMMC 54GB          ← 容量吃緊，媒體與模型一律放 NVMe
資料碟: NVMe 938GB 掛在 /mnt/nvme
網路: RTSP 攝影機 + RS-485（IO 模組 / 電子鎖）
```

`data/` `models/` `output/` `storage/` 四個目錄是 symlink 指向 `/mnt/nvme/traffic/`，
由 `scripts/init_dirs.sh` 建立。**不要把媒體寫回 eMMC**，54GB 很快就滿。

### 軟體環境（現場實測值）
```
系統: JetPack 6.1 (L4T R36.4.0 / Ubuntu 22.04)
CUDA: 12.6
TensorRT: 10.3
Python: 3.10.12
Docker: 只用來跑 Frigate；主程式跑 host systemd
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
│   ├── 📄 main.py                   # API 入口點（註冊 23 個 router）
│   ├── 📄 models.py                 # SQLAlchemy 資料模型
│   └── 📂 routes/                   # API 路由模組
│       ├── 📄 auth.py               # 登入/登出/目前使用者
│       ├── 📄 cameras.py            # 攝影機 CRUD + 連線測試 + Frigate 同步
│       ├── 📄 violations.py         # 違規事件管理
│       ├── 📄 stream.py             # 即時串流 + 偵測服務
│       ├── 📄 frigate.py            # Frigate NVR 整合
│       ├── 📄 frigate_camera_endpoints.py  # Frigate 逐台端點
│       ├── 📄 lpr.py                # 車牌辨識 (單張)
│       ├── 📄 lpr_stream.py         # 車牌辨識串流（主流程都在這）
│       ├── 📄 lpr_visual.py         # LPR 視覺化串流
│       ├── 📄 congestion.py         # 壅塞偵測服務
│       ├── 📄 traffic.py            # 交通報表 / VD 報表
│       ├── 📄 analytics.py          # 分析統計
│       ├── 📄 external.py           # 對外 API（X-API-Key）
│       ├── 📄 api_key_admin.py      # API 金鑰管理
│       ├── 📄 io.py / io_tcp.py     # IO 模組（RS-485 / TCP）
│       ├── 📄 lock.py               # 電子鎖（雙位址、刷卡、事件）
│       ├── 📄 parking.py            # 停車場車位
│       ├── 📄 sensor_fusion.py      # 感測器融合
│       ├── 📄 vision_eye.py         # VisionEye
│       ├── 📄 mqtt.py               # MQTT 橋接
│       ├── 📄 nx.py                 # NX VMS 整合
│       ├── 📄 system.py             # 硬體監測 / NTP / 識別碼
│       └── 📄 logs.py               # 系統日誌服務
│
├── 📂 services/                     # 常駐服務與背景模組
│   ├── 📄 ocr_service.py            # ⭐ YOLO 字元 OCR 微服務 (:8010)
│   ├── 📄 io_daemon.py              # ⭐ IO/電子鎖獨立 daemon (:8011)
│   ├── 📄 io_service.py             # IO 邏輯（燈號、門磁、刷卡）
│   ├── 📄 io_module.py / pd3r3.py   # RS-485 Modbus 底層
│   ├── 📄 frigate_sync.py           # 攝影機設定 → Frigate/go2rtc 同步
│   ├── 📄 network_health.py         # 通訊故障判定（網卡 link 層）
│   ├── 📄 mqtt_bridge.py            # MQTT
│   └── 📄 parking_*.py              # 停車場（分類器 / VLM / SAHI / 佔用）
│
├── 📂 detection/                    # 偵測模組
│   ├── 📄 vehicle_detector.py       # YOLOv8 車輛偵測 (含大型車分類整合)
│   ├── 📄 truck_classifier.py       # YOLO26s 大型車細分類器
│   ├── 📄 violation_detector.py     # 違規偵測邏輯
│   ├── 📄 congestion_detector.py    # 壅塞偵測器
│   ├── 📄 wrong_way.py              # 逆向
│   ├── 📄 no_helmet.py              # 未戴安全帽
│   ├── 📄 pedestrian_yield.py       # 未禮讓行人
│   ├── 📄 parking_violation.py      # 違規停車
│   ├── 📄 speed_calib.py            # 測速校正
│   ├── 📄 auto_calibration.py       # 自動校正
│   ├── 📄 radar_track.py            # 雷達軌跡
│   └── 📄 gpu_lock.py               # GPU 推論互斥鎖
│
├── 📂 recognition/                  # 辨識模組
│   ├── 📄 plate_detector.py         # 車牌框 YOLO
│   ├── 📄 plate_recognizer.py       # 車牌 OCR（呼叫 :8010 微服務）
│   └── 📄 frigate_integration.py    # Frigate 事件整合
│
├── 📂 web/                          # 前端介面（單檔 inline-template SPA）
│   ├── 📄 index.html                # Vue 3 SPA 主頁（11000+ 行）
│   ├── 📄 nvr_playback.html         # NVR 回放介面 (EZ Pro 深色主題)
│   ├── 📄 roi_editor.html           # ROI 編輯器
│   ├── 📄 io_panel.html             # IO 控制面板
│   ├── 📄 lock_panel.html           # 電子鎖面板
│   ├── 📄 lpr_verify.html           # 車牌人工複驗
│   ├── 📄 parking_editor.html       # 停車格幾何編輯器
│   ├── 📄 parking_label.html        # 停車格標註
│   ├── 📄 agency_landing.html       # 機關交付入口頁
│   └── 📂 fonts/                    # 字型檔（含 CJK 疊加字型）
│
├── 📂 deploy/                       # 部署設定（進版控）
│   ├── 📂 systemd/                  # traffic-api / -ocr / -io / -frigate / -cleanup
│   ├── 📂 timesyncd/                # NTP drop-in 模板
│   └── 📂 journald/                 # journal 保存政策
│
├── 📂 scripts/                      # 工具腳本
│   ├── 📄 setup_new_site.sh         # 新站台一鍵初始化
│   ├── 📄 init_dirs.sh              # 建立四個資料目錄 symlink
│   ├── 📄 lint_vue_template.py      # ⭐ 改 web 前必跑
│   ├── 📄 smoke_check.py            # 部署後煙霧測試
│   ├── 📄 aggregate_reports.py      # 報表聚合
│   ├── 📄 cleanup_storage.py        # 保存政策清理
│   └── 📂 lpr_finetune/             # 車牌字元模型 finetune 流程
│
├── 📂 config/                       # 設定檔
│   ├── 📂 frigate/config.yml        # Frigate NVR 設定（執行期會被寫）
│   └── 📂 system/                   # NTP / NX / 版面 / 功能開關
│
├── 📂 models/                       # AI 模型 (不納入版控)
│   ├── 📄 yolov8n.pt / .engine      # 車輛偵測（engine 優先）
│   ├── 📄 truck_cls_yolo26s.pt/.engine  # 大型車分類
│   └── 📂 lpr/
│       ├── 📄 plate_yolov8n.pt      # 車牌框（現場無 .engine，走 PyTorch）
│       ├── 📄 Charcter-LP.pt        # 字元偵測（線上 = 07-13 finetune 版）
│       └── 📄 Charcter-LP.pt.before_finetune   # 回滾用
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
│  │ /api/system       系統管理 (NTP/NX/硬體狀態/識別碼)                  │   │
│  │ /api/io /api/lock IO 模組與電子鎖 (轉發到 :8011 daemon)              │   │
│  │ /api/parking      停車場車位                                         │   │
│  │ /api/v1/external  對外資料 API (X-API-Key)                           │   │
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
│ │ (YOLOv8)    │ │       │ │ →:8010 OCR  │ │       │ │  violations │ │
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
│                        同機常駐服務（獨立 process）                          │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ traffic-ocr (:8010)         │  │ traffic-io (127.0.0.1:8011)         │  │
│  │ └─ YOLO 字元偵測 OCR        │  │ ├─ RS-485 Modbus (DI/DO/電子鎖)     │  │
│  │    拆開避免與主 YOLO 搶 GPU │  │ └─ 拆開避免 SEGV 拖垮主服務         │  │
│  └─────────────────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              外部服務層                                      │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ Frigate NVR (:5000, Docker) │  │ IP 攝影機 (RTSP)                    │  │
│  │ ├─ go2rtc restream (:8554)  │  │ ├─ rtsp://user:pass@ip:port/path    │  │
│  │ ├─ 事件錄影                 │  │ └─ H.264/H.265 編碼                 │  │
│  │ └─ MQTT 推送                │  └─────────────────────────────────────┘  │
│  └─────────────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**process 邊界是刻意的**：字元 OCR 與 RS-485 各自跑在獨立 systemd unit。
OCR 拆開是為了不跟主偵測搶 GPU；IO 拆開是因為 RS-485 的 native 例外會直接
SEGV，同 process 會把整套偵測拉下水。

---

## 🚀 安裝部署

> ⚠️ **正式機（Jetson）跑的是 host systemd，不是容器。**
> 只有 Frigate 在 Docker。容器版 compose 是給開發機 / staging 用的，
> 兩者搶同一個 `:8000`，**不能同時啟動**。

### 方式一：正式部署（Jetson，systemd）

新站台一次帶起來：

```bash
git clone git@github.com:sanniel0315/traffic-violation-detection.git
cd traffic-violation-detection

FIELD_NTP=<現場NTP位址> TRAFFIC_STORAGE_ROOT=/mnt/nvme/traffic \
  bash scripts/setup_new_site.sh

sudo systemctl enable --now traffic-frigate traffic-ocr traffic-io traffic-api
```

四個服務的分工：

| unit | 埠 | 職責 |
|---|---|---|
| `traffic-api` | `:8000` | 主程式（偵測、LPR 串流、報表、Web） |
| `traffic-ocr` | `:8010` | YOLO 字元 OCR 微服務 |
| `traffic-io` | `127.0.0.1:8011` | RS-485 IO / 電子鎖 daemon |
| `traffic-frigate` | `:5000` | Frigate NVR（Docker compose 包一層） |
| `traffic-cleanup.timer` | — | 每日 02:30 依保存政策清理 |

`AUTH_SECRET` **必填**，沒設或用公開預設值會**直接拒絕啟動**
（避免 session token 被偽造）。產生方式：

```bash
python3 -c "import secrets;print(secrets.token_urlsafe(48))"
```

### 方式二：容器部署（開發機 / staging）
```bash
cp .env.example .env        # 至少要有 AUTH_SECRET
docker compose up -d
docker logs -f traffic-api
```

程式碼是 bind-mount 進容器的，改完 `docker restart traffic-api` 就生效，
不用 rebuild image（除非動到 `requirements.txt` / `Dockerfile`）。

### 方式三：本機直跑
```bash
pip install -r requirements.txt
bash scripts/init_dirs.sh                      # 建立四個資料目錄
python3 scripts/download_plate_model.py        # 車牌模型
AUTH_SECRET=$(python3 -c "import secrets;print(secrets.token_urlsafe(48))") \
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 另開一個 terminal：OCR 微服務（不起的話車牌辨識全部拿不到結果）
python3 services/ocr_service.py
```

### 存取服務

| 服務 | URL | 說明 |
|------|-----|------|
| Web 介面 | http://localhost:8000/web/ | 管理介面 |
| NVR 回放 | http://localhost:8000/web/nvr_playback.html | EZ Pro 風格回放介面 |
| API 文件 | http://localhost:8000/docs | Swagger UI |
| API Reference | [docs/API_REFERENCE.md](./docs/API_REFERENCE.md) | 完整 API 文件（全系統 208 個端點） |
| Frigate NVR | http://localhost:5000 | NVR 介面 |

**預設登入帳號（首次初始化，可用 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 覆寫）**
```
username: admin
password: admin123
```
⚠️ 上線前務必改掉。

### 部署後自動檢查（建議每次必跑）
```bash
# API 健康 + 交通事件區間查詢 + VD 前端關鍵標記
python3 scripts/smoke_check.py --base-url http://127.0.0.1:8000 --timeout 60

# 一鍵：重啟 API 並自動驗證
./scripts/restart_and_verify.sh http://127.0.0.1:8000 60
```

### 現場上板部署（無網路 Jetson，**容器版**）

> 正式機是 systemd 版，離線更新請直接 `scp` 檔案或 `git bundle`，
> 不要用下面的 docker save/load 流程 —— 那是容器版才適用。

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

### 一鍵自動部署（推薦）

CI/CD pipeline 已設好 — 開發機 `git push origin main` 後**完全自動**：

```
dev PC                  GitHub                    Jetson (self-hosted runner)
   │                       │                              │
   │ git push ────────────►│                              │
   │                       │ trigger workflow ────────────►│
   │                       │                              │ ✓ GPU/CUDA check
   │                       │                              │ ✓ Python/YOLO test
   │                       │                              │ ✓ DB schema
   │                       │                              │ ✓ API endpoints
   │                       │                              │ ✓ Frigate connect
   │                       │                              │ ✓ Smoke check
   │                       │                              │
   │                       │ verify pass → auto deploy ──►│ git pull production
   │                       │                              │ sudo restart traffic-api
   │                       │                              │ verify commit hash
   │                       │◄─────────── result ──────────│
```

**workflow 配置**：[.github/workflows/jetson-verify.yml](.github/workflows/jetson-verify.yml)

**runner 狀態**：
```bash
gh api repos/<owner>/<repo>/actions/runners
# 預期 jetson-agx-orin status=online
```

**手動觸發 workflow**：
```bash
gh workflow run "Jetson Device Verification"
```

### 手動 deploy（fallback）

> 🛑 **不要寫死 LAN IP。** 現場的 LAN 位址是 DHCP，實測換過好幾次
> （`.108` → `.102` → `192.168.84.87` → `192.168.1.3`）。
> 一律走 Tailscale 或 Cloudflare Tunnel。

```bash
# 從 dev PC ssh 進 Jetson（Tailscale 固定位址）
ssh ubuntu@100.92.17.87 "cd ~/traffic-violation-detection && git pull origin main && sudo systemctl restart traffic-api"

# 驗證部署版本
ssh ubuntu@100.92.17.87 "cd ~/traffic-violation-detection && git log --oneline -1"

# 外網（固定網址）
curl https://tvd.name-car-box.com/api/health
```

**pull 前先看 `git status`。** `config/frigate/config.yml` 與
`config/system/*.json` 是執行期會被寫的檔案，會擋住 pull。
CI 的部署流程已經處理（deploy 前後保留執行期設定檔），手動 pull 要自己顧。

### 完整流程文件

詳細步驟與離線部署：[`推送流程.md`](./推送流程.md)

內容包含：
- GitHub SSH 金鑰設定與驗證（dev PC + Jetson 雙邊）
- self-hosted runner 安裝與 systemd service
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
| `POST` | `/api/lpr/recognize-base64` | base64 圖片辨識 |
| `POST` | `/api/lpr/recognize-camera/{id}` | 攝影機截圖辨識 |
| `POST` | `/api/lpr/stream/start/{id}` | 啟動串流辨識 |
| `POST` | `/api/lpr/stream/stop/{id}` | 停止串流辨識 |
| `GET` | `/api/lpr/stream/status/{id}` | 串流辨識狀態 + 七個排查計數器 |
| `GET` | `/api/lpr/stream/results/{id}` | 取得辨識結果 |
| `GET` | `/api/lpr/stream/history` | 歷史紀錄（支援 `min_confidence`） |
| `GET` | `/api/lpr/stream/camera-options` | 可選攝影機清單 |
| `GET` | `/api/lpr/stream/snapshot/{filename}` | 車牌截圖 |
| `GET` | `/api/lpr/stream/all` | 全部任務總覽 |
| `GET` | `/api/lpr/visual/stream/{id}` | 視覺化串流 |

> 車牌辨識完整架構（模型、5 變體 ensemble、投票門檻、格式修復兩層、
> 排查漏斗）見 **[`ocr 流程.md`](./ocr%20流程.md)**。

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

### 3. 車牌辨識模組 `recognition/plate_recognizer.py` + `services/ocr_service.py`

> ⚠️ 舊版本文件寫「Tesseract + 6 種預處理」，**那不是現在的路徑**。
> `import pytesseract` 還在檔案裡，但主線不呼叫它。
> 完整說明見 [`ocr 流程.md`](./ocr%20流程.md)。

辨識拆成兩個 process：

```
traffic-api  :8000    PlateRecognizer.recognize_easy()  ← 呼叫端
traffic-ocr  :8010    ocr_service.py  YOLO 字元偵測      ← 模型常駐
```

拆開的原因：**字元 YOLO 不要跟主偵測 YOLO 搶 GPU**。

```python
# recognition/plate_recognizer.py
class PlateRecognizer:
    def recognize_easy(self, img) -> Dict:
        """5 變體 ensemble：每個變體各 POST 一次到 :8010，加權投票取勝者。

        變體: original / clahe / upscale_2x / bilateral / gray_otsu
        投票: 先比出現次數，同票再比平均 conf
        加成: 多變體同意每票 +0.05（上限 +0.15，總分封頂 0.99）
        失敗: 全部 fail → recognize_chars() 字元分割 fallback
        """
```

```python
# services/ocr_service.py — 微服務內部
def ocr_plate(img_bytes) -> dict:
    """YOLO 出字元框後：
       1. y 座標過濾 —— 只留主要那一行（濾掉牌框上下雜訊）
       2. 單字元 conf < 0.4 丟掉 —— 一個糊字不該拖低整體
       3. 依 x 排序組字，avg_conf = 各字元平均
       4. 漏字偵測 —— gap / 字寬中位數 > 1.5 給 penalty
       5. _repair_plate() 台灣格式修復 + 相似字 swap
    """
```

**實際 LPR 處理流程：**
```
RTSP / frigate latest.jpg
    │
    ▼
YOLOv8n 車輛偵測 + 追蹤
    │ car, motorcycle, bus, truck
    ▼
車道 ROI 過濾（排除禁停/人行道/紅線）
    │
    ▼
PlateDetector 找車牌框 (conf 0.12)
    │
    ▼
5 變體 ensemble → :8010 YOLO 字元偵測 ×5
    │
    ▼
微服務內格式修復 _repair_plate
    │
    ▼
多幀空間投票（bucket 160px，TTL 3.5s）
    │
    ▼
confirm（票數 ≥2 或 score ≥1.8）→ commit（score ≥1.5、conf ≥0.40）
    │
    ▼
存 DB 前 _enforce_plate_format 邊界修復（只修不丟）
    │
    ▼
lpr_records（含 vehicle_bbox，供違規關聯用）
```

**台灣車牌格式**（兩層修復共用同一組規則）：

- 依總長度比對格式表，字母位／數字位分開檢查
- 相似字 swap 上限 2 次（`5↔S`、`0↔O`、`1↔I` 等）
- 字母位排除 `I` / `O` / `Q`（台灣車牌規範不使用）
- 修不成合法格式**不丟棄**，原樣保留交由信心度門檻過濾

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

### 必填

| 變數 | 說明 |
|------|------|
| `AUTH_SECRET` | Session 簽章金鑰。**未設或用公開預設值會拒絕啟動**。`python3 -c "import secrets;print(secrets.token_urlsafe(48))"` |

### 常用

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `DATABASE_URL` | `sqlite:///./data/violations.db` | 資料庫連線 |
| `TZ` | `Asia/Taipei` | 時區 |
| `DEVICE` | `cuda:0` | 推論裝置 |
| `DEVICE_ID` | — | 終端控制器識別碼（交付規範用，例 `R34_動態號誌VD`） |
| `STREAM_HOST` | — | 前端組串流網址用的主機位址 |
| `EXTERNAL_API_KEY` | — | 對外 API 的 `X-API-Key` |
| `AUTH_TTL_HOURS` | — | Session 有效時數 |

### 模型路徑

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MODEL_DIR` | `/home/ubuntu/traffic-violation-detection/models` | 模型根目錄 |
| `LPR_MODEL_DIR` | `${MODEL_DIR}/lpr` | 車牌模型目錄 |
| `DETECT_MODEL_ENGINE` | `yolov8n.engine` | 車輛偵測 engine |
| `DETECT_MODEL_PT` | `yolov8n.pt` | 車輛偵測 pt（engine 不存在時用） |
| `LPR_PLATE_MODEL_ENGINE` | `plate_yolov8n.engine` | 車牌框 engine |
| `LPR_PLATE_MODEL_PT` | `plate_yolov8n.pt` | 車牌框 pt（現場實際走這個） |
| `TRUCK_CLS_MODEL` | `truck_cls_yolo26s.pt` | 大型車分類模型 |
| `DISABLE_TRT` / `FORCE_GPU` | — | 除錯用：停用 TensorRT / 強制 GPU |

### Frigate / NVR

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `FRIGATE_HOST` | `frigate` | Frigate 主機（host 部署要設 `localhost`） |
| `FRIGATE_PORT` | `5000` | Frigate 埠號 |
| `FRIGATE_CONFIG_PATH` | `/workspace/config/frigate/config.yml` | 自動同步要改寫的設定檔 |
| `FRIGATE_RESTART_CMD` | `sudo -n systemctl restart traffic-frigate` | 套用設定的重啟指令 |
| `FRIGATE_RESTART_DEBOUNCE_SEC` | `8` | 連續改多台時的去抖秒數 |

### IO / 電子鎖

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `IO_DAEMON_URL` | `http://127.0.0.1:8011` | 主程式連 IO daemon 的位址 |
| `LOCK_SERIAL_PORT` | — | RS-485 裝置（現場改用 USB-485 `/dev/ttyUSB0` 才讀得到完整卡號） |
| `LOCK_MODBUS_ADDR` | `0`（停用） | 電子鎖位址，支援多顆與命名：`2:後門,3:前門` |
| `LOCK_RETRY_SEC` | `5` | 讀不到的鎖退避秒數（不退避會拖慢整個輪詢迴圈） |
| `IO_ADDR` / `IO_BAUD` / `IO_PORT` | — | PD3R3 IO 模組參數 |

### 系統

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `NET_IFACE` | 自動偵測 default route | 通訊故障要監看的網卡（可逗號分隔多張） |
| `NTP_DROPIN_PATH` | `/etc/systemd/timesyncd.conf.d/zz-field-ntp.conf` | NTP 設定落點（**檔名必須排在出廠 drop-in 之後**） |
| `TRAFFIC_STORAGE_ROOT` | `/mnt/nvme/traffic` | 四個資料目錄的實體位置 |
| `SYSTEM_CONFIG_DIR` | `/workspace/config/system` | 執行期設定檔目錄 |

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

> 以下是 **systemd（正式機）** 的指令。容器版把 `journalctl -u traffic-api`
> 換成 `docker logs -f traffic-api`、`systemctl restart` 換成 `docker restart`。

### 攝影機連線失敗
```bash
# 直接探 RTSP（比 cv2 快，也看得到 codec）
ffprobe -rtsp_transport tcp -i "rtsp://user:pass@ip:port/path" 2>&1 | head -20
```

### 查看日誌
```bash
journalctl -u traffic-api -f              # 主程式
journalctl -u traffic-ocr -n 50           # OCR 微服務
journalctl -u traffic-io  -n 50           # IO / 電子鎖
```

journal 已設持久化（`/var/log/journal` → NVMe，20G / 保留 6 個月）。

### 重啟服務
```bash
sudo systemctl restart traffic-api
```

> ⚠️ 偵測、LPR、壅塞都在 `traffic-api` 這一個 process 裡，重啟會全部中斷。
> 小改動請批次做完再一次重啟。使用者回報「壞了」時，先確認是不是剛重啟
> 或瀏覽器快取（hard reload），不要急著改程式碼。

### 網頁顯示離線但後端是好的
```bash
ss -tn | grep :8000 | awk '{print $5}' | cut -d: -f1 | sort | uniq -c
```

同一個瀏覽器 IP 出現 6 條就是**瀏覽器連線數上限被 MJPEG 串流佔滿**，
API 請求排不進去。重啟服務永遠無效。

### 車牌辨識沒有結果
```bash
curl -s http://127.0.0.1:8010/                        # OCR 微服務活著嗎
curl -s http://127.0.0.1:8000/api/lpr/stream/status/6 # 看七個計數器斷在哪
```

排查漏斗與各段意義見 [`ocr 流程.md`](./ocr%20流程.md) 第 10 節。

### 改完 web 頁面整段 UI 消失 / 白屏
```bash
python3 scripts/lint_vue_template.py --all
```

單檔 inline-template SPA 對 HTML balance 極度敏感，**改動 `web/` 下任何
含 `createApp` 的頁面後 commit 前必跑**。詳見 `CLAUDE.md` 的強制 SOP。

---

## 📄 授權

MIT License

---

## 👥 貢獻

歡迎提交 Issue 和 Pull Request！

---

*最後更新: 2026-08-13 — 硬體/環境數值、部署方式、車牌辨識架構、環境變數、
故障排除均已對回程式碼與現場實機核對*
