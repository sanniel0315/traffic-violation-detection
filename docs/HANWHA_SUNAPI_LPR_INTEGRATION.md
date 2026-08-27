# Hanwha SUNAPI 球機追蹤、車牌放大與 LPR 整合文件

## 目的

本文件說明如何把 Hanwha 球機 SUNAPI 整合進 TRAFFIC 專案，達成以下流程：

```text
攝影機追蹤車輛
  -> 取得車牌座標
  -> 車牌太小時控制球機 Area Zoom 放大
  -> 本系統進行 LPR/OCR
  -> 目標到指定位置後停止追蹤
```

## 適用環境

```text
專案：traffic-violation-detection
後端：FastAPI
LPR：既有 /api/lpr 與 /api/lpr/stream 流程
攝影機：Hanwha 支援 SUNAPI 的 PTZ / AI 型號
環境：staging 已建立，攝影機已加入系統
```

## SUNAPI 需求能力

攝影機至少需要支援以下 SUNAPI 能力：

```text
PTZ Query              讀取 Pan / Tilt / Zoom
Area Zoom              依影像座標放大指定區域
Digital Auto Tracking  攝影機端自動追蹤物件
PTZ Stop               停止 PTZ 移動
```

可用以下 API 確認：

```text
GET http://<camera-ip>/stw-cgi/attributes.cgi/attributes
GET http://<camera-ip>/stw-cgi/ptzcontrol.cgi?msubmenu=supportedptzactions&action=view&Channel=0
```

需要確認支援：

```text
Query.Pan
Query.Tilt
Query.Zoom
AreaZoom
DigitalAutoTracking
AIAutoTracking
```

## 專案新增模組

目前 TRAFFIC 專案內新增以下檔案：

```text
services/hanwha_sunapi.py
api/routes/hanwha.py
tests/test_hanwha_sunapi_workflow.py
docs/hanwha_sunapi_lpr_tracking/README.md
docs/HANWHA_SUNAPI_LPR_INTEGRATION.md
```

`api/main.py` 已掛上 Hanwha router：

```python
from api.routes import hanwha

app.include_router(hanwha.router)
```

## Camera 設定方式

攝影機已加入 staging 後，需要在該攝影機的 `detection_config` 加入 SUNAPI 設定。

最小設定：

```json
{
  "sunapi": {
    "base_url": "http://192.168.1.100",
    "username": "admin",
    "password": "password",
    "timeout": 4
  }
}
```

若 `Camera.ip`、`Camera.username`、`Camera.password` 已經有資料，可改用：

```json
{
  "sunapi": {
    "http_port": 80,
    "timeout": 4
  }
}
```

若要讓 LPR stream 自動套用追蹤、放大與停止條件，加入：

```json
{
  "sunapi": {
    "base_url": "http://192.168.1.100",
    "username": "admin",
    "password": "password",
    "timeout": 4
  },
  "hanwha_lpr_tracking": {
    "enabled": true,
    "auto_start": true,
    "channel": 0,
    "profile": 2,
    "min_lpr_plate_width": 160,
    "min_lpr_plate_height": 48,
    "zoom_padding_ratio": 0.35,
    "zoom_cooldown_sec": 1.2,
    "stop_zone": {
      "x1": 1300,
      "y1": 650,
      "x2": 1700,
      "y2": 900
    }
  }
}
```

## 停止條件設定

停止追蹤有兩種方式。

### 方式一：畫面停止區

當車牌中心點進入 `stop_zone`，後端停止追蹤。

```json
{
  "stop_zone": {
    "x1": 1300,
    "y1": 650,
    "x2": 1700,
    "y2": 900
  }
}
```

適合用在「車輛到畫面某區域後就不追」。

### 方式二：PTZ 座標停止

當球機目前 Pan/Tilt/Zoom 進入指定誤差範圍，後端停止追蹤。

```json
{
  "ptz_stop_window": {
    "pan": 180,
    "tilt": 25,
    "zoom": 8,
    "pan_tolerance": 2,
    "tilt_tolerance": 1,
    "zoom_tolerance": 1
  }
}
```

適合用在「追蹤到某個實體方向或路口位置後就不追」。

## 後端 API

### 查 PTZ 座標

```text
GET /api/hanwha/{camera_id}/ptz?channel=0
```

回傳範例：

```json
{
  "camera_id": 1,
  "camera_name": "Hanwha PTZ",
  "position": {
    "pan": 180,
    "tilt": 25,
    "zoom": 8,
    "zoom_pulse": 1789
  }
}
```

### 查支援功能

```text
GET /api/hanwha/{camera_id}/supported-ptz-actions?channel=0
```

### 啟動追蹤

```text
POST /api/hanwha/{camera_id}/tracking/start
```

```json
{
  "channel": 0,
  "profile": 2
}
```

### 停止追蹤

```text
POST /api/hanwha/{camera_id}/tracking/stop
```

```json
{
  "channel": 0,
  "profile": 2
}
```

### 放大車牌區域

```text
POST /api/hanwha/{camera_id}/plate/zoom
```

```json
{
  "plate_bbox": {
    "x1": 900,
    "y1": 500,
    "x2": 980,
    "y2": 526
  },
  "frame_width": 1920,
  "frame_height": 1080,
  "channel": 0,
  "profile": 2,
  "padding_ratio": 0.35
}
```

### 單步流程判斷

```text
POST /api/hanwha/{camera_id}/workflow/step
```

用畫面停止區：

```json
{
  "plate_bbox": {
    "x1": 900,
    "y1": 500,
    "x2": 980,
    "y2": 526
  },
  "frame_width": 1920,
  "frame_height": 1080,
  "stop_zone": {
    "x1": 1300,
    "y1": 650,
    "x2": 1700,
    "y2": 900
  },
  "channel": 0,
  "profile": 2,
  "execute": true
}
```

用 PTZ 座標停止：

```json
{
  "plate_bbox": {
    "x1": 900,
    "y1": 500,
    "x2": 980,
    "y2": 526
  },
  "frame_width": 1920,
  "frame_height": 1080,
  "ptz_stop_window": {
    "pan": 180,
    "tilt": 25,
    "pan_tolerance": 2,
    "tilt_tolerance": 1
  },
  "channel": 0,
  "profile": 2,
  "execute": true
}
```

回傳判斷：

```text
state = zooming    車牌太小，已呼叫或應呼叫 Area Zoom
state = lpr_ready  車牌尺寸足夠，可以進 LPR
state = stopped    已到停止條件，停止追蹤
```

## LPR Stream 整合點

整合位置在：

```text
api/routes/lpr_stream.py
```

建議插入點：

```text
車輛偵測
  -> 車牌 bbox 偵測完成
  -> OCR 前
```

整合邏輯：

```python
workflow = build_plate_lpr_workflow(
    plate_bbox=BBox(px1, py1, px2, py2),
    frame_width=iw,
    frame_height=ih,
    stop_zone=hanwha_stop_zone,
    ptz_position=ptz_position,
    ptz_stop_window=hanwha_ptz_stop_window,
    min_lpr_plate_width=160,
    min_lpr_plate_height=48,
    zoom_padding_ratio=0.35,
)

if "stop_tracking" in workflow["actions"]:
    client.stop_digital_autotracking(channel=0, profile=2)
    return

if "area_zoom" in workflow["actions"]:
    client.area_zoom(
        bbox=BBox(*workflow["zoom_bbox"]),
        frame_width=iw,
        frame_height=ih,
        channel=0,
        profile=2,
    )
    return

# lpr_ready 才進入原本 OCR
```

## Staging 驗收流程

### 1. 確認攝影機資料

確認 staging 的 camera 有：

```text
ip
username
password
detection_config.sunapi
```

### 2. 查支援功能

```text
GET /api/hanwha/{camera_id}/supported-ptz-actions
```

驗收：

```text
HTTP 200
支援 query / areazoom / digitalautotracking / stop
```

### 3. 查 PTZ

```text
GET /api/hanwha/{camera_id}/ptz
```

驗收：

```text
可取得 pan / tilt / zoom
```

### 4. 啟動追蹤

```text
POST /api/hanwha/{camera_id}/tracking/start
```

驗收：

```text
球機開始追蹤 Vehicle
```

### 5. 測試車牌放大

用目前畫面車牌 bbox 呼叫：

```text
POST /api/hanwha/{camera_id}/plate/zoom
```

驗收：

```text
球機將車牌區域放大
放大後 LPR 辨識率提高
```

### 6. 測試停止條件

用 `workflow/step` 丟入 stop_zone 或 ptz_stop_window。

驗收：

```text
目標進入指定位置後停止 digital auto tracking
PTZ stop 有送出
後續不再追該目標
```

## 常見問題

### 1. `ptz` 查不到座標

可能原因：

```text
攝影機不支援 Query.Pan / Query.Tilt / Query.Zoom
SUNAPI 帳密錯誤
HTTP port 設錯
staging 後端連不到攝影機內網 IP
```

### 2. `areazoom` 沒反應

檢查：

```text
AreaZoom 是否支援
plate_bbox 是否為實際影像像素座標
TileWidth / TileHeight 是否等於當前影像解析度
球機是否正在 busy
```

### 3. 追蹤不停止

檢查：

```text
stop_zone 座標是否使用同一個影像解析度
車牌中心點是否真的進入 stop_zone
ptz_stop_window tolerance 是否太小
```

### 4. 放大後 OCR 還是不準

建議：

```text
放大後等待 0.5 到 1 秒再截圖
提高 min_lpr_plate_width / min_lpr_plate_height
確認快門速度，避免動態模糊
確認 IR / 補光是否造成車牌過曝
```

## 測試指令

```bash
python -m pytest tests/test_hanwha_sunapi_workflow.py
python -m py_compile services/hanwha_sunapi.py api/routes/hanwha.py api/main.py
```

## 交付狀態

```text
SUNAPI 控制 service：已加入
FastAPI 路由：已加入
流程判斷測試：已加入
整合文件：已加入
LPR stream 自動掛載：依 staging camera 設定啟用
```
