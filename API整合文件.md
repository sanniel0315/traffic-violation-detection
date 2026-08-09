# 對外報表 API 文件

> 版本：1.1 | 更新日期：2026-08-09

## 概述

交通影像分析系統提供對外 REST API，供政府/交通局系統及內部系統取得 VD 車流報表與壅塞報表資料。

- **Base URL：** `http://{host}:8000/api/v1/external`
- **認證方式：** API Key（`X-API-Key` Header）
- **輸出格式：** JSON / CSV
- **Swagger UI：** `http://{host}:8000/api/v1/external/docs?token={API_KEY}`
  （瀏覽器開文件頁沒辦法帶 Header，所以用 `?token=`；程式呼叫請用 Header）

> `http://{host}:8000/docs` 是**內部**完整文件，需管理者登入，對外不提供。

---

## 定期輪詢（建議做法）

要持續取得最新車流，用 `vd-report/latest`，不必自己算時間區間：

```
GET /api/v1/external/vd-report/latest?minutes=5&interval=1m
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `minutes` | 5 | 回最近幾個**已結束**的時間桶 |
| `interval` | `1m` | 桶大小：`1m` / `5m` / `1h` |
| `detector_id` | 全部 | 指定攝影機 |
| `include_records` | true | false = 只回統計摘要 |

**三個實作要點**（照做可避免拿到錯誤數字）：

1. **`minutes` 給 3～5，並依 `time_start` 去重 upsert。**
   聚合是背景每 60 秒跑一次，**最新的桶可能還沒算完**。多抓幾個桶、
   重複的以新值覆蓋，晚到的數字會自己補正。只抓最新一桶會偏低。
2. **桶大小的參數名是 `interval`，不是 `bucket_size`。**
   傳錯會被忽略並套用預設值，看起來像「查不到資料」。
3. **時間帶時區或不帶都可以。** 不帶視為 UTC；帶 `+08:00` 為台北時間，
   兩者指到同一時刻會得到同一批資料。

輪詢頻率 20 秒沒有問題（實測單次回應約 40 毫秒）。速率限制預設 120 次/分。

---

## 認證

所有對外 API 端點皆需在 HTTP Header 帶入 API Key：

```
X-API-Key: tvd_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 錯誤回應

| HTTP Code | 說明 |
|-----------|------|
| 401 | 缺少或無效的 API Key |
| 403 | API Key 無此報表的存取權限（scope 不足） |
| 429 | 超過速率限制 |

### 取得 API Key

**方式一：固定 Key（推薦）**

在 `.env` 設定固定 API Key，不需透過管理端點建立：

```env
EXTERNAL_API_KEY=tvd_hwacom_traffic_2026
```

此 Key 擁有所有報表權限（`vd_report` + `congestion_report`），速率限制 120 req/min。

**方式二：動態 Key**

由系統管理者透過管理端點建立，詳見 [API Key 管理](#api-key-管理)。適合需要多組 Key、個別權限控制的場景。

---

## 統一回應格式

### JSON 成功回應

```json
{
  "status": "success",
  "data": { ... },
  "meta": {
    "request_time": "2026-04-07T10:00:00+08:00",
    "api_version": "1.0",
    "device_id": "jetson-nx-001",
    "format": "json"
  }
}
```

### JSON 錯誤回應

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "end_time 必須大於 start_time"
  }
}
```

### CSV 回應

加 `?format=csv` 參數，回傳 `Content-Type: text/csv` 檔案下載。

---

## 報表端點

### 1. VD 車流報表

取得指定時間範圍的車流量、車速、佔用率等聚合數據。

```
GET /api/v1/external/vd-report
```

#### 請求參數

| 參數 | 類型 | 必填 | 預設 | 說明 |
|------|------|------|------|------|
| `start_time` | datetime | 是 | — | 起始時間（ISO 8601，如 `2026-04-07T00:00:00+08:00`） |
| `end_time` | datetime | 是 | — | 結束時間（ISO 8601） |
| `detector_id` | int | 否 | 全部 | 攝影機 ID |
| `interval` | string | 否 | `5m` | 聚合間隔：`1m` / `5m` / `1h` |
| `format` | string | 否 | `json` | 輸出格式：`json` / `csv` |

#### 時間範圍限制

| 間隔 | 最大查詢範圍 |
|------|-------------|
| `1m` | 24 小時 |
| `5m` | 7 天 |
| `1h` | 90 天 |

#### 請求範例

```bash
curl -H "X-API-Key: tvd_xxxxxxxx" \
  "http://{host}:8000/api/v1/external/vd-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00&interval=5m"
```

#### JSON 回應範例

```json
{
  "status": "success",
  "data": {
    "interval": "5m",
    "period": {
      "start": "2026-04-07T00:00:00+08:00",
      "end": "2026-04-07T12:00:00+08:00"
    },
    "records": [
      {
        "detector_id": "台62基隆段隧道口",
        "road_name": "台62線",
        "time_start": "2026-04-07T08:00:00+08:00",
        "time_end": "2026-04-07T08:05:00+08:00",
        "direction": "N2S",
        "direction_label": "北向南",
        "total_flow": 45,
        "small_vehicle_flow": 38,
        "large_vehicle_flow": 7,
        "avg_speed_kmh": 42.3,
        "avg_occupancy_pct": 18.5,
        "lane_count": 2,
        "lanes": [
          {
            "lane_no": 1,
            "flow": 25,
            "small_vehicle_flow": 22,
            "large_vehicle_flow": 3,
            "avg_speed_kmh": 44.1,
            "avg_occupancy_pct": 16.2,
            "avg_queue_length_m": 12.5,
            "max_queue_length_m": 38.0
          },
          {
            "lane_no": 2,
            "flow": 20,
            "small_vehicle_flow": 16,
            "large_vehicle_flow": 4,
            "avg_speed_kmh": 40.5,
            "avg_occupancy_pct": 20.8,
            "avg_queue_length_m": 14.8,
            "max_queue_length_m": 52.5
          }
        ]
      }
    ]
  },
  "meta": {
    "request_time": "2026-04-07T10:00:00+08:00",
    "api_version": "1.0",
    "device_id": "jetson-nx-001",
    "format": "json"
  }
}
```

#### 欄位說明

| 欄位 | 型別 | 說明 |
|------|------|------|
| `detector_id` | string | 攝影機名稱/ID |
| `road_name` | string | 道路名稱 |
| `time_start` | string | 時間區間起始（ISO 8601） |
| `time_end` | string | 時間區間結束 |
| `direction` | string | 行車方向代碼 |
| `direction_label` | string | 行車方向中文 |
| `total_flow` | int | 總車流量（**不含**進出場事件，見下方說明） |
| `in_flow` | int | 進場車數（車輛駛入偵測框） |
| `out_flow` | int | 出場車數（車輛駛離偵測框） |
| `direction_counts` | object | 各方向原始計數，供需要自行拆分時使用 |
| `small_vehicle_flow` | int | 小型車流量（小客車、機車） |
| `large_vehicle_flow` | int | 大型車流量（公車、貨車） |
| `avg_speed_kmh` | float | 平均車速 (km/h) |
| `avg_occupancy_pct` | float | 平均佔用率 (%) |
| `lane_count` | int | 車道數 |
| `lanes` | array | 各車道明細 |
| `lanes[].lane_no` | int | 車道編號 |
| `lanes[].flow` | int | 該車道車流量 |
| `lanes[].avg_queue_length_m` | float | 平均排隊長度 (公尺) |
| `lanes[].max_queue_length_m` | float | 最大排隊長度 (公尺) |

#### 🛑 `total_flow` 與 `in_flow` / `out_flow` 的關係

進出場是**另一條計數路徑**，兩者不可相加：

| 事件 | 計入 `total_flow` | 計入 `in_flow` / `out_flow` |
|------|------------------|---------------------------|
| 一般通過偵測框 | ✅ | ❌ |
| 駛入偵測框 | ❌ | ✅ `in_flow` |
| 駛離偵測框 | ❌ | ✅ `out_flow` |

`total_flow` 已是該時段的完整車流量，**再加上 `in_flow`／`out_flow` 會重複計算**
（實測某 5 分鐘桶：正確為 `in=40 / out=39`，相加的錯誤算法會得到 `in=80 / out=79`）。

未設定進出線的攝影機，`in_flow` 與 `out_flow` 恆為 0，不影響 `total_flow`。

#### CSV 格式

每一筆展平為一行（per lane），欄位順序：

```
detector_id, road_name, time_start, time_end, direction,
lane_no, flow, small_vehicle_flow, large_vehicle_flow,
avg_speed_kmh, avg_occupancy_pct, avg_queue_length_m, max_queue_length_m,
queue_duration_sec, max_queue_duration_sec, in_flow, out_flow
```

> `in_flow` / `out_flow` 是**整框進出**的量、不分車道，因此只出現在該筆的
> **第一條車道列**，其餘車道列留空。依 `detector_id + time_start` 分組加總
> 即得正確值；若複製到每條車道，加總會變成車道數的倍數。

---

### 2. 壅塞報表

取得指定時間範圍的壅塞偵測聚合數據。

```
GET /api/v1/external/congestion-report
```

#### 請求參數

| 參數 | 類型 | 必填 | 預設 | 說明 |
|------|------|------|------|------|
| `start_time` | datetime | 是 | — | 起始時間（ISO 8601） |
| `end_time` | datetime | 是 | — | 結束時間（ISO 8601） |
| `detector_id` | int | 否 | 全部 | 攝影機 ID |
| `interval` | string | 否 | `5m` | 聚合間隔：`1m` / `5m` / `1h` |
| `format` | string | 否 | `json` | 輸出格式：`json` / `csv` |

#### 請求範例

```bash
curl -H "X-API-Key: tvd_xxxxxxxx" \
  "http://{host}:8000/api/v1/external/congestion-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00"
```

#### JSON 回應範例

```json
{
  "status": "success",
  "data": {
    "interval": "5m",
    "period": {
      "start": "2026-04-07T00:00:00+08:00",
      "end": "2026-04-07T12:00:00+08:00"
    },
    "records": [
      {
        "detector_id": "2",
        "camera_name": "台62基隆段隧道口",
        "time_start": "2026-04-07T08:00:00+08:00",
        "time_end": "2026-04-07T08:05:00+08:00",
        "zone_name": "車流區 1",
        "lane_no": 1,
        "direction": "straight",
        "avg_occupancy_pct": 35.2,
        "max_occupancy_pct": 58.0,
        "avg_vehicle_count": 12.3,
        "avg_stopped_vehicle_count": 4.1,
        "avg_queue_length_m": 22.5,
        "max_queue_length_m": 45.0,
        "queue_active_duration_sec": 120.0,
        "sample_count": 60
      }
    ]
  },
  "meta": {
    "request_time": "2026-04-07T10:00:00+08:00",
    "api_version": "1.0",
    "device_id": "jetson-nx-001",
    "format": "json"
  }
}
```

#### 欄位說明

| 欄位 | 型別 | 說明 |
|------|------|------|
| `detector_id` | string | 攝影機 ID |
| `camera_name` | string | 攝影機名稱 |
| `time_start` | string | 時間區間起始 |
| `time_end` | string | 時間區間結束 |
| `zone_name` | string | 偵測區域名稱 |
| `lane_no` | int | 車道編號 |
| `direction` | string | 行車方向 |
| `avg_occupancy_pct` | float | 平均佔用率 (%) |
| `max_occupancy_pct` | float | 最大佔用率 (%) |
| `avg_vehicle_count` | float | 平均車輛數 |
| `avg_stopped_vehicle_count` | float | 平均停滯車輛數 |
| `avg_queue_length_m` | float | 平均排隊長度 (公尺) |
| `max_queue_length_m` | float | 最大排隊長度 (公尺) |
| `queue_active_duration_sec` | float | 排隊持續時間 (秒) |
| `sample_count` | int | 取樣數量 |

#### CSV 格式

欄位順序：

```
detector_id, camera_name, time_start, time_end,
zone_name, lane_no, direction,
avg_occupancy_pct, max_occupancy_pct,
avg_vehicle_count, avg_stopped_vehicle_count,
avg_queue_length_m, max_queue_length_m,
queue_active_duration_sec, sample_count
```

---

## API Key 管理

> 以下端點需 admin 登入 session，非 API Key 認證。

### 建立 API Key

```
POST /api/auth/api-keys
```

**Request Body：**

```json
{
  "name": "交通局正式環境",
  "scopes": ["vd_report", "congestion_report"],
  "rate_limit_per_min": 60,
  "expires_at": "2027-01-01T00:00:00+08:00"
}
```

| 欄位 | 必填 | 說明 |
|------|------|------|
| `name` | 是 | 用途說明（最多 100 字元） |
| `scopes` | 否 | 授權範圍，預設 `["vd_report","congestion_report"]` |
| `rate_limit_per_min` | 否 | 每分鐘最大請求數，預設 60 |
| `expires_at` | 否 | 過期時間（ISO 8601），不填則永不過期 |

**可用 scopes：**

| Scope | 對應端點 |
|-------|---------|
| `vd_report` | `/api/v1/external/vd-report` |
| `congestion_report` | `/api/v1/external/congestion-report` |

**Response：**

```json
{
  "status": "success",
  "item": {
    "id": 1,
    "name": "交通局正式環境",
    "api_key": "tvd_G3iEzmqVzvcGyDj2cndQqb0P5EMoDvDru3q3ngrmUZE",
    "key_prefix": "tvd_G3iE",
    "scopes": ["vd_report", "congestion_report"],
    "rate_limit_per_min": 60,
    "expires_at": "2027-01-01T00:00:00+08:00"
  },
  "warning": "此 API Key 僅顯示一次，請妥善保存"
}
```

> **重要：** `api_key` 欄位僅在建立時回傳一次，之後無法再查看。

### 列出所有 API Keys

```
GET /api/auth/api-keys
```

### 更新 API Key

```
PUT /api/auth/api-keys/{id}
```

**Request Body（皆為可選）：**

```json
{
  "name": "新名稱",
  "scopes": ["vd_report"],
  "enabled": false,
  "rate_limit_per_min": 30
}
```

### 刪除 API Key

```
DELETE /api/auth/api-keys/{id}
```

---

## 錯誤代碼

| Code | HTTP | 說明 |
|------|------|------|
| `MISSING_API_KEY` | 401 | 未提供 X-API-Key Header |
| `INVALID_API_KEY` | 401 | API Key 無效或已過期 |
| `INSUFFICIENT_SCOPE` | 403 | API Key 無此報表的存取權限 |
| `RATE_LIMITED` | 429 | 超過速率限制 |
| `INVALID_PARAMETER` | 400 | 參數錯誤（如時間格式不正確） |
| `RANGE_TOO_LARGE` | 400 | 查詢時間範圍超過限制 |
| `TOO_MANY_RECORDS` | 413 | 結果超過 10,000 筆上限 |

---

## 使用範例

### Python

```python
import requests

API_KEY = "tvd_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
BASE = "http://{host}:8000/api/v1/external"

# VD 報表 (JSON)
resp = requests.get(f"{BASE}/vd-report", headers={"X-API-Key": API_KEY}, params={
    "start_time": "2026-04-07T00:00:00+08:00",
    "end_time": "2026-04-07T12:00:00+08:00",
    "interval": "5m",
})
data = resp.json()
print(f"車流記錄: {len(data['data']['records'])} 筆")

# 壅塞報表 (CSV 下載)
resp = requests.get(f"{BASE}/congestion-report", headers={"X-API-Key": API_KEY}, params={
    "start_time": "2026-04-07T00:00:00+08:00",
    "end_time": "2026-04-07T12:00:00+08:00",
    "format": "csv",
})
with open("congestion.csv", "w") as f:
    f.write(resp.text)
```

### cURL

```bash
# VD 報表 JSON
curl -H "X-API-Key: tvd_xxx..." \
  "http://{host}:8000/api/v1/external/vd-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00"

# VD 報表 CSV 下載
curl -H "X-API-Key: tvd_xxx..." -o vd_report.csv \
  "http://{host}:8000/api/v1/external/vd-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00&format=csv"

# 壅塞報表
curl -H "X-API-Key: tvd_xxx..." \
  "http://{host}:8000/api/v1/external/congestion-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00"
```
