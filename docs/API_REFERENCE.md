# API Reference

> Traffic Violation Detection System - REST API 文件
>
> Base URL: `http://<host>:<port>/api`

---

## 目錄

- [認證 (Auth)](#認證-auth)
- [攝影機 (Cameras)](#攝影機-cameras)
- [串流與偵測 (Stream & Detection)](#串流與偵測-stream--detection)
- [違規紀錄 (Violations)](#違規紀錄-violations)
- [交通事件 (Traffic)](#交通事件-traffic)
- [壅塞偵測 (Congestion)](#壅塞偵測-congestion)
- [車牌辨識 (LPR)](#車牌辨識-lpr)
- [車牌串流 (LPR Stream)](#車牌串流-lpr-stream)
- [車牌視覺化 (LPR Visual)](#車牌視覺化-lpr-visual)
- [NVR / Frigate](#nvr--frigate)
- [NX VMS](#nx-vms)
- [系統日誌 (Logs)](#系統日誌-logs)
- [系統管理 (System)](#系統管理-system)
- [根端點 (Root)](#根端點-root)
- [大型車分類 (Truck Classifier)](#大型車分類-truck-classifier)

---

## 認證 (Auth)

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/api/auth/login` | 使用者登入 |
| `POST` | `/api/auth/logout` | 使用者登出 |
| `GET` | `/api/auth/me` | 取得目前登入使用者資訊 |
| `GET` | `/api/auth/users` | 列出所有使用者 (管理員) |
| `POST` | `/api/auth/users` | 新增使用者 (管理員) |
| `PUT` | `/api/auth/users/{user_id}` | 更新使用者角色/狀態 (管理員) |
| `PUT` | `/api/auth/users/{user_id}/password` | 更新使用者密碼 (管理員) |
| `DELETE` | `/api/auth/users/{user_id}` | 刪除使用者 (管理員) |

### POST /api/auth/login

```json
// Request
{
  "username": "admin",
  "password": "password"
}

// Response 200
{
  "token": "eyJhbGciOi...",
  "user": {
    "id": 1,
    "username": "admin",
    "role": "admin"
  }
}
```

---

## 攝影機 (Cameras)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/cameras` | 列出所有攝影機 |
| `GET` | `/api/cameras/statistics` | 攝影機統計資訊 |
| `GET` | `/api/cameras/{camera_id}` | 取得單一攝影機詳情 |
| `POST` | `/api/cameras` | 新增攝影機 |
| `PUT` | `/api/cameras/{camera_id}` | 更新攝影機設定 |
| `DELETE` | `/api/cameras/{camera_id}` | 刪除攝影機 |
| `POST` | `/api/cameras/{camera_id}/test` | 測試攝影機連線 |
| `POST` | `/api/cameras/test-url` | 測試攝影機 URL |
| `POST` | `/api/cameras/upload-source` | 上傳影片來源檔案 |
| `GET` | `/api/cameras/source-files` | 列出可用來源檔案 |
| `POST` | `/api/cameras/import-source-file` | 匯入來源檔案為攝影機 |
| `POST` | `/api/cameras/analyze-source` | 快速分析任意來源 |
| `POST` | `/api/cameras/analyze-frame` | 分析單幀影像 |

### GET /api/cameras

```json
// Response 200
{
  "data": [
    {
      "id": 1,
      "name": "路口攝影機A",
      "location": "中正路/忠孝路口",
      "source_url": "rtsp://192.168.1.100:554/stream1",
      "status": "online",
      "type": "rtsp"
    }
  ]
}
```

### POST /api/cameras

```json
// Request
{
  "name": "新攝影機",
  "location": "中正路",
  "source_url": "rtsp://192.168.1.101:554/stream1",
  "type": "rtsp"
}

// Response 201
{
  "id": 2,
  "name": "新攝影機",
  "status": "online"
}
```

### POST /api/cameras/analyze-frame

```json
// Request (multipart/form-data)
// file: 影像檔案

// Response 200
{
  "detections": [
    {
      "class_name": "car",
      "confidence": 0.92,
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
    }
  ]
}
```

---

## 串流與偵測 (Stream & Detection)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/stream/{camera_id}/live` | 即時串流 (MJPEG) |
| `GET` | `/api/stream/{camera_id}/live-overlay` | 帶偵測疊圖的即時串流 |
| `GET` | `/api/stream/{camera_id}/snapshot` | 單張快照 |
| `POST` | `/api/stream/{camera_id}/detection/start` | 啟動偵測服務 |
| `POST` | `/api/stream/{camera_id}/detection/stop` | 停止偵測服務 |
| `GET` | `/api/stream/{camera_id}/detection/status` | 取得偵測狀態 |
| `GET` | `/api/stream/detection/all` | 取得所有偵測服務狀態 |

### POST /api/stream/{camera_id}/detection/start

```json
// Request (optional body)
{
  "conf_threshold": 0.5,
  "enable_truck_cls": true
}

// Response 200
{
  "status": "started",
  "camera_id": 1,
  "message": "偵測服務已啟動"
}
```

### GET /api/stream/{camera_id}/detection/status

```json
// Response 200
{
  "camera_id": 1,
  "running": true,
  "fps": 15.2,
  "detections_count": 342,
  "uptime_sec": 1800
}
```

---

## 違規紀錄 (Violations)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/violations` | 列出違規紀錄 (支援篩選) |
| `GET` | `/api/violations/statistics` | 違規統計 |
| `GET` | `/api/violations/{violation_id}` | 取得單一違規詳情 |
| `POST` | `/api/violations` | 新增違規紀錄 |
| `PUT` | `/api/violations/{violation_id}/review` | 審核/批准違規 |
| `DELETE` | `/api/violations/{violation_id}` | 刪除違規紀錄 |

### GET /api/violations

```
Query Parameters:
  camera_id   (int)    - 篩選攝影機
  type        (string) - 違規類型
  start_time  (string) - 起始時間 (ISO 8601)
  end_time    (string) - 結束時間 (ISO 8601)
  status      (string) - 狀態: pending / approved / rejected
  page        (int)    - 頁碼 (預設 1)
  page_size   (int)    - 每頁筆數 (預設 20)
```

```json
// Response 200
{
  "data": [
    {
      "id": 101,
      "camera_id": 1,
      "type": "red_light",
      "plate_number": "ABC-1234",
      "confidence": 0.95,
      "timestamp": "2026-04-08T14:30:00",
      "status": "pending",
      "snapshot_url": "/api/violations/101/snapshot"
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20
}
```

---

## 交通事件 (Traffic)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/traffic/vd-report` | 車輛偵測報表 (含聚合) |
| `GET` | `/api/traffic/events` | 列出交通事件 |

### GET /api/traffic/vd-report

```
Query Parameters:
  camera_id   (int)    - 攝影機 ID
  start_time  (string) - 起始時間
  end_time    (string) - 結束時間
  interval    (string) - 聚合區間: 5min / 15min / 1h / 1d
```

```json
// Response 200
{
  "data": [
    {
      "time_slot": "2026-04-08T14:00:00",
      "car_count": 45,
      "truck_count": 8,
      "motorcycle_count": 12,
      "bus_count": 3,
      "total": 68,
      "avg_speed": 42.5
    }
  ]
}
```

---

## 壅塞偵測 (Congestion)

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/api/congestion/{camera_id}/start` | 啟動壅塞偵測 |
| `POST` | `/api/congestion/{camera_id}/stop` | 停止壅塞偵測 |
| `GET` | `/api/congestion/{camera_id}/params` | 取得壅塞偵測參數 |
| `PUT` | `/api/congestion/{camera_id}/params` | 更新壅塞偵測參數 |
| `GET` | `/api/congestion/{camera_id}/status` | 取得單一攝影機壅塞狀態 |
| `GET` | `/api/congestion/status/all` | 取得所有壅塞偵測狀態 |
| `GET` | `/api/congestion/samples` | 查詢壅塞樣本 |
| `GET` | `/api/congestion/{camera_id}/snapshot` | 壅塞分析快照 |
| `GET` | `/api/congestion/{camera_id}/stream` | 壅塞偵測視覺化串流 |

### GET /api/congestion/{camera_id}/status

```json
// Response 200
{
  "camera_id": 1,
  "running": true,
  "level": "moderate",
  "vehicle_count": 23,
  "occupancy": 0.65,
  "avg_speed": 18.3,
  "updated_at": "2026-04-08T14:30:00"
}
```

### PUT /api/congestion/{camera_id}/params

```json
// Request
{
  "occupancy_threshold_moderate": 0.5,
  "occupancy_threshold_severe": 0.8,
  "speed_threshold_kmh": 15,
  "detection_interval_sec": 5
}
```

---

## 車牌辨識 (LPR)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/lpr/status` | 取得 LPR 引擎狀態 |
| `POST` | `/api/lpr/recognize-upload` | 上傳圖片辨識車牌 |
| `POST` | `/api/lpr/recognize-base64` | Base64 圖片辨識車牌 |
| `POST` | `/api/lpr/recognize-camera/{camera_id}` | 從攝影機辨識車牌 |

### POST /api/lpr/recognize-upload

```json
// Request (multipart/form-data)
// file: 影像檔案

// Response 200
{
  "plates": [
    {
      "plate_number": "ABC-1234",
      "confidence": 0.97,
      "bbox": {"x1": 150, "y1": 300, "x2": 350, "y2": 380}
    }
  ]
}
```

---

## 車牌串流 (LPR Stream)

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/api/lpr/stream/start/{camera_id}` | 啟動 LPR 串流服務 |
| `POST` | `/api/lpr/stream/stop/{camera_id}` | 停止 LPR 串流服務 |
| `GET` | `/api/lpr/stream/status/{camera_id}` | 取得 LPR 串流狀態 |
| `GET` | `/api/lpr/stream/results/{camera_id}` | 取得 LPR 辨識結果 |
| `GET` | `/api/lpr/stream/history` | 查詢 LPR 歷史紀錄 |
| `GET` | `/api/lpr/stream/camera-options` | 取得可用攝影機選項 |
| `GET` | `/api/lpr/stream/snapshot/{filename}` | 取得 LPR 快照檔案 |
| `GET` | `/api/lpr/stream/all` | 取得所有 LPR 串流資料 |

### GET /api/lpr/stream/history

```
Query Parameters:
  camera_id   (int)    - 攝影機 ID
  plate       (string) - 車牌號碼 (模糊搜尋)
  start_time  (string) - 起始時間
  end_time    (string) - 結束時間
  limit       (int)    - 筆數上限 (預設 100)
```

```json
// Response 200
{
  "records": [
    {
      "id": 1,
      "camera_id": 1,
      "plate_number": "ABC-1234",
      "confidence": 0.96,
      "timestamp": "2026-04-08T14:25:00",
      "snapshot_path": "/api/lpr/stream/snapshot/lpr_20260408_142500.jpg"
    }
  ]
}
```

---

## 車牌視覺化 (LPR Visual)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/lpr/visual/stream/{camera_id}` | LPR 視覺化串流 (MJPEG) |
| `GET` | `/api/lpr/visual/snapshot/{camera_id}` | LPR 視覺化快照 |

---

## NVR / Frigate

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/frigate/status` | Frigate NVR 連線狀態 |
| `GET` | `/api/frigate/version` | Frigate 版本 |
| `GET` | `/api/frigate/config` | 取得 Frigate 設定 |
| `POST` | `/api/frigate/config` | 更新 Frigate 設定 |
| `GET` | `/api/frigate/settings` | 取得 UI 設定 |
| `PUT` | `/api/frigate/settings` | 更新 UI 設定 |
| `GET` | `/api/frigate/cameras` | 列出 Frigate 攝影機 |
| `PUT` | `/api/frigate/cameras/{camera_name}` | 更新 Frigate 攝影機 |
| `DELETE` | `/api/frigate/cameras/{camera_name}` | 刪除 Frigate 攝影機 |
| `POST` | `/api/frigate/camera` | 新增攝影機到 Frigate |
| `DELETE` | `/api/frigate/camera/{name}` | 從 Frigate 移除攝影機 |
| `GET` | `/api/frigate/events` | 列出 Frigate 事件 |
| `GET` | `/api/frigate/events/{event_id}/snapshot` | 取得事件快照 |
| `GET` | `/api/frigate/events/{event_id}/clip` | 取得事件影片 |
| `GET` | `/api/frigate/events/{event_id}/clip.mp4` | 下載事件 MP4 |
| `GET` | `/api/frigate/recordings` | 查詢錄影紀錄 |
| `GET` | `/api/frigate/recordings/play` | 播放錄影 |
| `POST` | `/api/frigate/sync-cameras` | 同步攝影機到 Frigate |
| `POST` | `/api/frigate/restart` | 重啟 Frigate 服務 |
| `PUT` | `/api/frigate/camera/{name}/switch` | 切換攝影機錄影模式 |
| `GET` | `/api/frigate/camera/{name}/motion-roi` | 取得 Motion ROI |
| `PUT` | `/api/frigate/camera/{name}/motion-roi` | 更新 Motion ROI |
| `GET` | `/api/frigate/camera/{name}/latest.jpg` | 取得最新攝影機快照 |

### GET /api/frigate/status

```json
// Response 200
{
  "status": "online",
  "version": "0.13.2",
  "cameras": [
    {"name": "front_door", "recording": true, "detecting": true}
  ]
}
```

### GET /api/frigate/events

```
Query Parameters:
  camera      (string) - 攝影機名稱
  label       (string) - 偵測標籤 (car, person, etc.)
  start_time  (string) - 起始時間
  end_time    (string) - 結束時間
  limit       (int)    - 筆數上限
  all_pages   (string) - "1" 表示取回全部
  batch_size  (int)    - 批次大小
```

```json
// Response 200
{
  "events": [
    {
      "id": "1712345678.123456-abc123",
      "camera": "front_door",
      "label": "car",
      "start_time": 1712345678.12,
      "end_time": 1712345690.45,
      "has_clip": true,
      "has_snapshot": true
    }
  ],
  "total": 50,
  "truncated": false
}
```

### GET /api/frigate/recordings

```
Query Parameters:
  camera      (string) - 攝影機名稱
  start_time  (string) - 起始時間
  end_time    (string) - 結束時間
  limit       (int)    - 筆數上限
```

```json
// Response 200
{
  "items": [
    {
      "id": "rec-001",
      "camera": "front_door",
      "start_time": 1712345600,
      "end_time": 1712345900,
      "duration": 300,
      "play_url": "/api/frigate/recordings/play?src=..."
    }
  ]
}
```

---

## NX VMS

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/nx/devices` | 列出 NX VMS 設備 |
| `GET` | `/api/nx/stream/{device_id}` | 從 NX 設備串流 |

---

## 系統日誌 (Logs)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/logs` | 取得最近日誌 |
| `GET` | `/api/logs/query` | 查詢日誌 (含篩選與分頁) |
| `DELETE` | `/api/logs` | 清除所有日誌 |

### GET /api/logs/query

```
Query Parameters:
  level       (string) - 日誌等級: debug / info / warning / error
  module      (string) - 模組名稱
  keyword     (string) - 關鍵字搜尋
  start_time  (string) - 起始時間
  end_time    (string) - 結束時間
  page        (int)    - 頁碼
  page_size   (int)    - 每頁筆數
```

```json
// Response 200
{
  "logs": [
    {
      "id": 1,
      "timestamp": "2026-04-08T14:30:00",
      "level": "info",
      "module": "detection",
      "message": "偵測服務已啟動"
    }
  ],
  "total": 500,
  "page": 1
}
```

---

## 系統管理 (System)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/api/system/ntp/settings` | 取得 NTP 同步設定 |
| `PUT` | `/api/system/ntp/settings` | 更新 NTP 設定 |
| `POST` | `/api/system/ntp/sync-now` | 手動同步 NTP 時間 |
| `GET` | `/api/system/nx/settings` | 取得 NX/VMS 設定 |
| `PUT` | `/api/system/nx/settings` | 更新 NX/VMS 設定 |
| `GET` | `/api/system/status` | 取得系統硬體狀態 |

### GET /api/system/status

```json
// Response 200
{
  "cpu_percent": 35.2,
  "memory": {
    "total_gb": 16.0,
    "used_gb": 8.5,
    "percent": 53.1
  },
  "disk": {
    "total_gb": 500,
    "used_gb": 120,
    "percent": 24.0
  },
  "gpu": {
    "name": "NVIDIA GeForce GTX 1660 SUPER",
    "memory_total_mb": 6144,
    "memory_used_mb": 2048,
    "utilization_percent": 45
  }
}
```

---

## 根端點 (Root)

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/` | API 根資訊 |
| `GET` | `/api/health` | 健康檢查 |
| `GET` | `/api/system/info` | 系統資訊 (平台、PyTorch、CUDA、GPU) |
| `GET` | `/api/dashboard` | 儀表板摘要 |

### GET /api/health

```json
// Response 200
{
  "status": "ok",
  "timestamp": "2026-04-08T14:30:00"
}
```

### GET /api/dashboard

```json
// Response 200
{
  "violations_today": 12,
  "cameras_online": 5,
  "cameras_total": 6,
  "detections_running": 3,
  "recent_violations": [...]
}
```

---

## 大型車分類 (Truck Classifier)

大型車細分類功能整合在 `VehicleDetector` 中，無獨立 API 端點。偵測服務啟動時自動啟用。

### 分類類別

| 類別 | 中文 | 等效長度 | 分組 |
|------|------|---------|------|
| `heavy_truck` | 大貨車 | 12.0m | large |
| `light_truck` | 小貨車 | 6.0m | small |
| `bus` | 大客車 | 12.0m | large |
| `non_truck` | 非目標 | 6.0m | other |

### 偵測回應中的分類欄位

當偵測到 truck 或 bus 時，回應會附加 `truck_cls` 欄位：

```json
{
  "class_id": 7,
  "class_name": "heavy_truck",
  "confidence": 0.89,
  "bbox": {"x1": 100, "y1": 200, "x2": 500, "y2": 600, "width": 400, "height": 400},
  "truck_cls": {
    "class_name": "heavy_truck",
    "label": "大貨車",
    "confidence": 0.92,
    "group": "large",
    "length_m": 12.0
  }
}
```

### 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MODEL_DIR` | `/home/ubuntu/traffic-violation-detection/models` | 模型目錄 |
| `TRUCK_CLS_MODEL` | `truck_cls_yolo26s.pt` | 分類模型檔名 |
| `DETECT_MODEL_PT` | `yolov8n.pt` | 偵測模型檔名 |
| `DEVICE` | `cuda:0` | 推論裝置 |

---

## 通用說明

### 認證

需要在 Header 帶 Token：
```
Authorization: Bearer <token>
```

### 錯誤回應格式

```json
{
  "error": "錯誤訊息",
  "detail": "詳細說明 (選填)"
}
```

### HTTP 狀態碼

| 狀態碼 | 說明 |
|--------|------|
| 200 | 成功 |
| 201 | 建立成功 |
| 400 | 請求參數錯誤 |
| 401 | 未認證 |
| 403 | 權限不足 |
| 404 | 資源不存在 |
| 500 | 伺服器內部錯誤 |

### 串流端點

以下端點回傳 MJPEG 串流 (`Content-Type: multipart/x-mixed-replace`)：
- `GET /api/stream/{camera_id}/live`
- `GET /api/stream/{camera_id}/live-overlay`
- `GET /api/congestion/{camera_id}/stream`
- `GET /api/lpr/visual/stream/{camera_id}`

在 `<img>` 標籤中直接使用：
```html
<img src="/api/stream/1/live-overlay">
```

### NVR 回放介面

Web UI 位於 `/web/nvr_playback.html`，使用以下 API：
- `/api/frigate/status` - 連線狀態
- `/api/frigate/cameras` - 攝影機列表
- `/api/frigate/recordings` - 錄影查詢
- `/api/frigate/events` - 事件查詢
- `/api/frigate/recordings/play` - 影片播放
