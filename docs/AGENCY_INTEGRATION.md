# 交通違規影像分析系統 — 對機關既設數位影像平台整合文件

> **適用範圍**：本系統部署主機 → 機關既設 NVR / VMS / 影像平台
> **更新日期**：2026-05-08
> **裝置編號**：jetson-nx-001
> **聯絡**：sannielshi@gmail.com

---

## 1. 規格符合性

| 機關採購規格要求 | 本系統實際規格 | 達成 |
|---|---|---|
| 影像編碼 H.264 | H.264 (Main / High Profile @ Level 4.0)；go2rtc passthrough 不重編碼 | ✅ |
| 影像大小 ≥ 1280×720 | **1920×1080** (Full HD) | ✅ |
| 每秒 ≥ 15 FPS | **30 FPS** | ✅ |
| 即時影像傳送至既設平台 | 提供 RTSP / HLS / MJPEG 三協定，平台主動拉流 | ✅ |

---

## 2. RTSP 串流（推薦的整合方式）

### 2.1 串流列表

> 主機 IP：`192.168.0.108`，RTSP port `8554`，**無需帳號密碼**。

| Stream ID | 攝影機名稱 | RTSP URL | 編碼 | 解析度 | FPS |
|---|---|---|---|---|---|
| cam_1 | 匝道攝影機 (Axis) | `rtsp://192.168.0.108:8554/cam_1` | H.264 | 1920×1080 | 30 |
| cam_2 | 台 62 基隆段隧道口 (Samsung) | `rtsp://192.168.0.108:8554/cam_2` | H.264 Main | 1920×1080 | 30 |
| cam_6 | 台 62 LPR (Samsung) | `rtsp://192.168.0.108:8554/cam_6` | H.264 High | 1920×1080 | 30 |

### 2.2 機關平台設定範例

絕大多數 NVR / VMS（Hikvision、Dahua、NX Witness、Milestone、Genetec、Axis 之類）支援「Generic RTSP Source」，設定如下：

```
RTSP URL:        rtsp://192.168.0.108:8554/cam_2
Transport:       TCP (建議)
Authentication:  None
Codec:           H.264 (auto-detect)
```

### 2.3 命令列驗證

```bash
ffprobe -rtsp_transport tcp rtsp://192.168.0.108:8554/cam_2
# 預期輸出：H.264 Main, 1920x1080, 30 fps
```

---

## 3. 動態 API 查詢（給整合系統用）

需要程式化拿串流資訊（含線上狀態 / bytes 統計），打 External API：

### 3.1 端點

```
GET  http://192.168.0.108:8000/api/v1/external/streams
Header:  X-API-Key: <你的 API Key>
```

### 3.2 API Key 申請

請聯絡系統管理員建立 API Key，申請時告知所需 scopes：

| Scope | 用途 |
|---|---|
| `streams` | 查詢即時影像串流列表 |
| `vd_report` | 取 VD 車流報表 |
| `congestion_report` | 取壅塞報表 |

### 3.3 回應範例

```json
{
  "meta": {
    "request_time": "2026-05-08T14:13:10+08:00",
    "api_version": "1.0",
    "device_id": "jetson-nx-001",
    "format": "json"
  },
  "status": "success",
  "data": {
    "device_id": "jetson-nx-001",
    "host": "192.168.0.108",
    "ports": {"rtsp": 8554, "http": 1984},
    "stream_count": 3,
    "streams": [
      {
        "stream_id": "cam_2",
        "camera_id": 2,
        "name": "台62基隆段隧道口",
        "online": true,
        "codec": "h264",
        "profile": "Main",
        "resolution": "1920x1080",
        "fps": 30.0,
        "bytes_received": 5691793511,
        "urls": {
          "rtsp":  "rtsp://192.168.0.108:8554/cam_2",
          "hls":   "http://192.168.0.108:1984/api/stream.m3u8?src=cam_2",
          "mjpeg": "http://192.168.0.108:1984/api/stream.mjpeg?src=cam_2",
          "webrtc_signal": "http://192.168.0.108:1984/api/ws?src=cam_2"
        }
      }
    ],
    "spec_requirement": {
      "codec": "H.264 (passthrough，無重編碼)",
      "min_resolution": "1280x720",
      "min_fps": 15
    }
  }
}
```

### 3.4 cURL 範例

```bash
curl -H "X-API-Key: <your-key>" \
     http://192.168.0.108:8000/api/v1/external/streams | jq .
```

---

## 4. 備援串流協定

如機關平台不支援 RTSP，可改用：

| 協定 | URL | 適用 |
|---|---|---|
| **HLS** (HTTP Live Streaming) | `http://192.168.0.108:1984/api/stream.m3u8?src=cam_2` | Web 瀏覽器、行動裝置 |
| **MJPEG over HTTP** | `http://192.168.0.108:1984/api/stream.mjpeg?src=cam_2` | 老舊 NVR、低資源裝置（注意頻寬大） |
| **WebRTC** | `http://192.168.0.108:1984/api/ws?src=cam_2` | 現代瀏覽器低延遲場景 |
| **RTSP** | `rtsp://192.168.0.108:8554/cam_2` | **首選**（多數 VMS 標準支援） |

---

## 5. 健康檢查 / 監控

### 5.1 公開健康檢查

```
GET  http://192.168.0.108:8000/api/health
回應：{"status":"ok","version":"1.0.0"}
```

### 5.2 Swagger UI（互動式 API 文件）

```
http://192.168.0.108:8000/docs
http://192.168.0.108:8000/redoc
```

### 5.3 OpenAPI 規格 (JSON)

```
http://192.168.0.108:8000/openapi.json
```

---

## 6. 網路需求

### 6.1 同 LAN 整合

機關平台主機 IP 在 `192.168.0.0/24` 子網 → **無需任何網路設定，直接連線**。

### 6.2 跨子網 / 跨單位整合

需網路管理員協助：

```
[機關平台]  →  [機關 router]  →  [本系統 router 192.168.0.254]  →  192.168.0.108

需開放：
  TCP/UDP 8000     (External API + Web UI)
  TCP      8554    (RTSP)
  TCP      1984    (HLS / MJPEG / WebRTC signaling)
  UDP      8555    (WebRTC media — 視 NVR 是否走 WebRTC)
```

實作方式（由網管擇一）：
- VPN 通道（IPsec / WireGuard）
- 路由表加 static route（兩端 router 互通）
- NAT port forwarding（公網場景）

---

## 7. 攝影機原始 RTSP 規格

> 僅供問題排查，**機關平台請勿直接連這些 URL**（會跟本系統爭搶上游連線造成不穩）。

| Cam | 廠牌 | 原始 RTSP URL | 編碼 |
|---|---|---|---|
| cam_1 | Axis | `rtsp://root:****@111.70.33.189:554/axis-media/media.amp` | H.264 |
| cam_2 | Samsung | `rtsp://admin:****@111.70.34.183:6554/profile2/media.smp` | H.264 Main 4.0 |
| cam_6 | Samsung | `rtsp://admin:****@111.70.34.184:7554/profile2/media.smp` | H.264 High 4.0 |

本系統用 go2rtc 做**單一上游 → 多消費者** 多路分發，相機端只承受 1 個連線。

---

## 8. 故障排查

| 症狀 | 可能原因 | 處置 |
|---|---|---|
| RTSP 連不上 | 防火牆 / 路由 | 確認 `nc -zv 192.168.0.108 8554` 通 |
| 影像卡頓 | 平台端 buffer / 頻寬 | 改 TCP transport；確認頻寬 >10 Mbps/cam |
| 無影像但連得上 | 上游攝影機掛 | 打 `/api/v1/external/streams` 看 `online` 欄位 |
| API 401 | API Key 錯 / scope 不足 | 確認 Header `X-API-Key` 跟所需 scope |
| API 429 | 速率限制 | 降低呼叫頻率（預設 60 req/min） |

---

## 9. 變更紀錄

| 日期 | 變更 |
|---|---|
| 2026-05-08 | 初版 — RTSP 對機關平台開放、新增 `/api/v1/external/streams` |

---

## 10. 附錄：產生 PDF

```bash
# 使用 pandoc
pandoc docs/AGENCY_INTEGRATION.md -o AGENCY_INTEGRATION.pdf \
       --pdf-engine=xelatex \
       -V CJKmainfont="Noto Sans CJK TC"

# 或 VS Code 安裝「Markdown PDF」擴充套件，右鍵 Export
# 或 npx md-to-pdf docs/AGENCY_INTEGRATION.md
```
