# Hanwha SUNAPI 追蹤、車牌放大與 LPR 整合

本文件說明本專案新增的 Hanwha SUNAPI 控制層，目標流程是：

```text
啟動攝影機 Vehicle 追蹤
  -> 取得車牌 bbox
  -> 車牌太小就呼叫 Area Zoom
  -> 車牌夠大就交給本系統 LPR/OCR
  -> 目標到指定畫面區域或 PTZ 座標後停止追蹤
```

## 新增檔案

```text
services/hanwha_sunapi.py
api/routes/hanwha.py
tests/test_hanwha_sunapi_workflow.py
```

`api/main.py` 已掛上 `hanwha.router`。

## Camera 設定

系統會優先使用 `cameras` 表既有欄位：

```text
ip
username
password
```

若攝影機 HTTP port 不是 80，或需要指定 base URL，可在 `detection_config` 加上：

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

或：

```json
{
  "sunapi": {
    "scheme": "http",
    "http_port": 8080,
    "timeout": 4
  }
}
```

## API

查 PTZ 座標：

```text
GET /api/hanwha/{camera_id}/ptz?channel=0
```

查支援功能：

```text
GET /api/hanwha/{camera_id}/supported-ptz-actions?channel=0
```

啟動追蹤：

```text
POST /api/hanwha/{camera_id}/tracking/start
```

```json
{
  "channel": 0,
  "profile": 2
}
```

停止追蹤：

```text
POST /api/hanwha/{camera_id}/tracking/stop
```

放大車牌區域：

```text
POST /api/hanwha/{camera_id}/plate/zoom
```

```json
{
  "plate_bbox": {"x1": 900, "y1": 500, "x2": 980, "y2": 526},
  "frame_width": 1920,
  "frame_height": 1080,
  "channel": 0,
  "profile": 2,
  "padding_ratio": 0.35
}
```

## 單步流程控制

主要整合入口：

```text
POST /api/hanwha/{camera_id}/workflow/step
```

### 用畫面停止區停止追蹤

```json
{
  "plate_bbox": {"x1": 900, "y1": 500, "x2": 980, "y2": 526},
  "frame_width": 1920,
  "frame_height": 1080,
  "stop_zone": {"x1": 1300, "y1": 650, "x2": 1700, "y2": 900},
  "channel": 0,
  "profile": 2,
  "execute": true
}
```

判斷結果：

```text
車牌中心進入 stop_zone -> digitalautotracking Stop + ptz stop
車牌未進 stop_zone 且太小 -> areazoom
車牌未進 stop_zone 且夠大 -> next_step = run_lpr
```

### 用 PTZ 座標停止追蹤

如果「特定位置」是球機座標，可以用 `ptz_stop_window`。

```json
{
  "plate_bbox": {"x1": 900, "y1": 500, "x2": 980, "y2": 526},
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

後端會先讀取目前 PTZ：

```text
/stw-cgi/ptzcontrol.cgi?msubmenu=query&action=view&Channel=0&Query=Pan,Tilt,Zoom
```

若目前 `Pan/Tilt/Zoom` 落在容許範圍內，就停止追蹤。

## 與既有 LPR 串接

目前這層只負責攝影機控制，不直接改動既有 LPR 主流程。

建議串接順序：

```text
1. 從 Hanwha metadata 或本系統 YOLO 取得 plate_bbox
2. 呼叫 /api/hanwha/{camera_id}/workflow/step
3. 若 next_step = zooming，等待 0.5 到 1 秒後重新取影像
4. 若 next_step = run_lpr，呼叫既有 /api/lpr/recognize-camera/{camera_id} 或 LPR stream
5. 若 state = stopped，不再追蹤該目標
```

## SUNAPI 對應

```text
查能力:
/stw-cgi/attributes.cgi/attributes

查目前 PTZ:
/stw-cgi/ptzcontrol.cgi?msubmenu=query&action=view&Channel=0&Query=Pan,Tilt,Zoom

Area Zoom:
/stw-cgi/ptzcontrol.cgi?msubmenu=areazoom&action=control&Type=ZoomIn&X1=...&Y1=...&X2=...&Y2=...&TileWidth=...&TileHeight=...

啟動追蹤:
/stw-cgi/ptzcontrol.cgi?msubmenu=digitalautotracking&action=control&Profile=2&Mode=Start

停止追蹤:
/stw-cgi/ptzcontrol.cgi?msubmenu=digitalautotracking&action=control&Profile=2&Mode=Stop

停止 PTZ:
/stw-cgi/ptzcontrol.cgi?msubmenu=stop&action=control&Channel=0
```

## 驗證

```bash
python -m pytest tests/test_hanwha_sunapi_workflow.py
python -m py_compile services/hanwha_sunapi.py api/routes/hanwha.py api/main.py
```
