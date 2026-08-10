# 對外報表 API 文件

> 版本：1.1 | 更新日期：2026-08-09

## 概述

交通影像分析系統提供對外 REST API，供政府/交通局系統及內部系統取得 VD 車流報表與壅塞報表資料。

- **Base URL：** `http://10.42.38.35:8000/api/v1/external`
- **認證方式：** API Key（`X-API-Key` Header）
- **輸出格式：** JSON / CSV
- **Swagger UI：** `http://10.42.38.35:8000/api/v1/external/docs?token={API_KEY}`
  （瀏覽器開文件頁沒辦法帶 Header，所以用 `?token=`；程式呼叫請用 Header）

> `http://10.42.38.35:8000/docs` 是**內部**完整文件，需管理者登入，對外不提供。

---

## 即時 VD 報表（每 5 秒輪詢用這支）

**與 `/vd-report` 是同一種記錄格式**，差別只在「時間怎麼取」：

| | 時間 | 同一分鐘內重複查 |
|---|---|---|
| `/vd-report`、`/vd-report/latest` | 已結束的**固定桶**（最小 1 分鐘） | 得到同一份 |
| **`/realtime`（預設）** | **從當前整分累積到現在** | **累積值持續增加，跨分歸零** |

`records[]` 的欄位完全相同（18 個共同欄位），客戶**只要寫一套解析**，兩支都能吃。
即時版另外多一個 `flow_per_hour`。

```
GET /api/v1/external/realtime
```

| 參數 | 預設 | 範圍 | 說明 |
|------|------|------|------|
| `mode` | **`minute`** | `minute` / `window` | 見下方兩種模式 |
| `window_sec` | 60 | 10～600 | 往回推幾秒（`mode=window` 時有效） |
| `detector_id` | 全部 | — | 指定攝影機 |

#### 兩種模式

**`mode=window`（滾動視窗，需明確指定）** —— 每次都往回看 `window_sec` 秒，起點終點一起移動。
適合畫面即時顯示。⚠️ 視窗會重疊，**不可拿來存檔加總**。

**`mode=minute`（分鐘內累積，預設）** —— 起點釘在**當前整分**，終點是「現在」，跨分自動歸零：

```
18:49:14   起點 18:49:00   經過 14 秒   累積  2
18:49:54   起點 18:49:00   經過 54 秒   累積 18   ← 同一起點，只增不減
18:50:15   起點 18:50:00   經過 15 秒   累積  6   ← 跨分歸零
```

回應的 `elapsed_sec` 是這一分鐘已經過的秒數，`flow_per_hour` 依它換算
（`total_flow × 3600 ÷ elapsed_sec`）。

⚠️ **5 秒輪詢每分鐘必定會取到 `elapsed_sec` 為 0～5 的那幾筆**，此時外推值毫無意義
（3 秒過 1 台 → 1200.0；3 秒沒車 → 0.0）。`elapsed_sec` 為 `0` 時回 `null` 而非 `0`。
建議 `elapsed_sec` 小於 15 時不顯示 `flow_per_hour`，或直接改用累積計數
`total_flow` / `in_flow` / `out_flow` —— 這些是實際計數，任何 `elapsed_sec` 都正確。

> **分鐘結束時的累積值 = `/vd-report` 該分鐘桶的值**（實測 0 不符，含大車）。
> 所以想要「一分鐘統計」又想在分鐘內看到進度，用這個模式；
> 只要最終值、不需要中途進度，用 `/vd-report/latest`（不重疊的固定桶，適合存檔）。

```json
{
  "status": "success",
  "data": {
    "mode": "minute",
    "elapsed_sec": 38,
    "period": { "start": "2026-08-09T14:52:58+08:00", "end": "2026-08-09T14:53:58+08:00" },
    "stats": {
      "record_count": 4, "bucket_count": 1, "detector_count": 4,
      "overall": { "total_flow": 81, "in_flow": 12, "out_flow": 10, "...": "..." }
    },
    "records": [
      {
        "detector_id": "台62基隆段隧道口",
        "time_start": "2026-08-09T14:52:58+08:00",
        "time_end":   "2026-08-09T14:53:58+08:00",
        "direction": "straight", "direction_label": "直行",
        "total_flow": 29, "flow_per_hour": 1740.0,
        "small_vehicle_flow": 29, "large_vehicle_flow": 0,
        "avg_speed_kmh": 0, "avg_occupancy_pct": 33.8,
        "avg_queue_length_m": null, "max_queue_length_m": null,
        "lane_count": 2,
        "lanes": [ { "lane_no": 1, "flow": 17 }, { "lane_no": 2, "flow": 12 } ],
        "in_flow": 12, "out_flow": 10
      }
    ]
  }
}
```

資料直接讀原始表、不經聚合，**沒有聚合延遲**。
實測單次 12～92 毫秒（平均約 27 毫秒）；以 5 秒節奏連續量測佔用約 0.5%，無壓力。

`time_start` 是該分鐘的整分起點，`time_end` 是你查詢的時刻。
⚠️ 同一分鐘內多次查詢會得到同一個 `time_start`、不同的累積值。
要存檔請以 `detector_id + time_start` 做 upsert（覆蓋而非累加），
或直接改用下面的報表端點（不重疊的固定桶）。

---

## 定期報表輪詢（要統計區間才用這支）

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

#### 任何時間查詢都會拿到完整資料

**本端點只回「已經聚合完成」的時間桶**，不會把還在計算中的桶送出來。
因此不需要挑時間查詢，也不會出現「查到 0，過一分鐘又變成 5」的情況。

回應中的 `data.aggregated_through` 就是資料截止時刻，`period.end` 不會超過它：

```json
"period": { "start": "2026-08-09T11:48:00+08:00", "end": "2026-08-09T11:52:00+08:00" },
"aggregated_through": "2026-08-09T11:52:00+08:00"
```

代價是最新資料會落後一小段時間：**時間桶結束後約 15 秒可取得**（實測 14～15 秒）。
這是刻意的取捨：寧可晚十幾秒給正確數字，也不要立刻給一個稍後會變動的數字。
若 `aggregated_through` 明顯落後（超過 5 分鐘），代表背景聚合有異常，可據此告警。

以 `interval=1m` 搭配 5 秒輪詢為例，實際節奏是：每分鐘產生一個新桶，
桶結束後約 15 秒可取得；十二次輪詢中約有一次拿到新桶，其餘為同一份資料
（這是桶大小決定的，不是延遲）。若需要分鐘內的即時進度，請改用上面的
`/realtime?mode=minute`。若希望每次輪詢都有新資料，
可改用較大的 `minutes` 值一次取回多個桶，或依需求調整 `interval`。

#### 其他兩個要點

1. **桶大小的參數名是 `interval`，不是 `bucket_size`。**
   傳錯會被忽略並套用預設值，看起來像「查不到資料」。
2. **時間帶時區或不帶都可以。** 不帶視為 UTC，帶 `+08:00` 為台北時間；
   兩種寫法指到同一時刻時，回傳的資料相同。

輪詢頻率 5 秒沒有問題（實測單次回應約 12～92 毫秒，每分鐘 12 次）。速率限制預設 120 次/分。
仍建議依 `detector_id + time_start` 做 upsert，重複輪詢到同一個桶時直接覆蓋即可。

---

## 認證

所有對外 API 端點皆需在 HTTP Header 帶入 API Key：

```
X-API-Key: {Key}
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
EXTERNAL_API_KEY={Key}
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
curl -H "X-API-Key: {Key}" \
  "http://10.42.38.35:8000/api/v1/external/vd-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00&interval=5m"
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
curl -H "X-API-Key: {Key}" \
  "http://10.42.38.35:8000/api/v1/external/congestion-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00"
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

### 3. 壅塞報表（快捷）

免自己算時間，回最近 N 個已結束的時間桶 + 統計摘要。

```
GET /api/v1/external/congestion-report/latest?minutes=5&interval=1m
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `minutes` | 5 | 回最近幾個已結束的桶 |
| `interval` | `1m` | 桶大小：`1m` / `5m` / `1h` |
| `detector_id` | 全部 | 指定攝影機 |
| `include_records` | true | false = 只回統計摘要 |

回應結構同「壅塞報表」，另含 `stats`（`record_count` / `bucket_count` /
`detector_count` / `overall` / `by_detector` / `peak_bucket`）。

---

### 4. 影像串流清單

取得各攝影機的即時串流網址。

```
GET /api/v1/external/streams
```

```json
{
  "status": "success",
  "data": {
    "device_id": "jetson-nx-001",
    "host": "192.168.0.102",
    "ports": { "rtsp": 8554, "http": 1984 },
    "stream_count": 4,
    "streams": [
      {
        "stream_id": "cam_2",
        "camera_id": 2,
        "name": "台62基隆段隧道口",
        "online": true,
        "codec": "h264",
        "resolution": "1920x1080",
        "fps": 30.0,
        "urls": {
          "rtsp":  "rtsp://10.42.38.35:8554/cam_2",
          "hls":   "http://10.42.38.35:1984/api/stream.m3u8?src=cam_2",
          "mjpeg": "http://10.42.38.35:1984/api/stream.mjpeg?src=cam_2",
          "webrtc_signal": "http://10.42.38.35:1984/api/ws?src=cam_2"
        }
      }
    ]
  }
}
```

| 欄位 | 說明 |
|------|------|
| `stream_id` / `camera_id` | 串流代號／攝影機編號 |
| `online` | 該串流目前是否可用 |
| `codec` / `resolution` / `fps` | 編碼、解析度、幀率（依攝影機本機設定） |
| `urls.rtsp` / `hls` / `mjpeg` / `webrtc_signal` | 四種取流方式 |

> 影像為 H.264 直通（passthrough），不重新編碼。

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

API_KEY = "{Key}"
BASE = "http://10.42.38.35:8000/api/v1/external"

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
curl -H "X-API-Key: {Key}" \
  "http://10.42.38.35:8000/api/v1/external/vd-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00"

# VD 報表 CSV 下載
curl -H "X-API-Key: {Key}" -o vd_report.csv \
  "http://10.42.38.35:8000/api/v1/external/vd-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00&format=csv"

# 壅塞報表
curl -H "X-API-Key: {Key}" \
  "http://10.42.38.35:8000/api/v1/external/congestion-report?start_time=2026-04-07T00:00:00%2B08:00&end_time=2026-04-07T12:00:00%2B08:00"
```
