# 匝道車輛分析系統

**國道匝道即時車輛計數 / 車種分類 / 車速估算**
平台：Jetson Xavier NX｜YOLOv8-TRT + ByteTrack + Homography

---

## 目錄

- [系統架構](#系統架構)
- [檔案說明](#檔案說明)
- [安裝依賴](#安裝依賴)
- [快速啟動](#快速啟動)
- [ROI 校正](#roi-校正)
- [速度校正](#速度校正)
- [FastAPI 整合](#fastapi-整合)
- [Vue 3 前端整合](#vue-3-前端整合)
- [參數說明](#參數說明)
- [輸出格式](#輸出格式)
- [速度區間定義](#速度區間定義)
- [車種對照表](#車種對照表)
- [常見問題](#常見問題)

---

## 系統架構

```
RTSP / 影片檔
      │
      ▼
 ROI Mask（多邊形遮罩，限定匝道範圍）
      │
      ▼
 YOLOv8-TRT（偵測 + 分類）
      │
      ▼
 ByteTrack（跨幀 ID 追蹤）
      │
      ▼
 Homography 速度估算（像素位移 → km/h）
      │
      ▼
 統計聚合（車種數量 / 均速 / 最高速 / 超速）
      │
      ▼
 FastAPI REST + MJPEG 推流 → Vue 3 儀表板
```

---

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `ramp_analyzer.py` | 核心分析模組，包含 `RampAnalyzer` 類別與 FastAPI router |
| `roi_calibrator.py` | 互動式 ROI 框選工具，輸出頂點座標 |
| `RampDashboard.vue` | Vue 3 儀表板元件，整合即時影像與統計數據 |

---

## 安裝依賴

```bash
# Jetson Xavier NX（PyTorch 2.2.0 已安裝，NumPy 鎖定 1.26.4）
pip install ultralytics --break-system-packages --no-cache-dir

# FastAPI 後端（若尚未安裝）
pip install fastapi uvicorn --break-system-packages
```

> **注意**：Jetson 上請勿升級 NumPy，保持 `1.26.4` 以相容 PyTorch 2.2.0。

---

## 快速啟動

### 測試影片檔

```bash
python3 ramp_analyzer.py --source 國8東.mp4 --model yolov8s.pt
```

### 接 RTSP 串流

```bash
python3 ramp_analyzer.py \
  --source rtsp://192.168.1.100:8554/camera1 \
  --model yolov8s.engine \
  --conf 0.35
```

### 指定自訂 ROI（JSON 格式，順時針四頂點）

```bash
python3 ramp_analyzer.py \
  --source rtsp://192.168.1.100:8554/camera1 \
  --model yolov8s.engine \
  --roi "[[390,150],[680,150],[811,443],[300,443]]"
```

---

## ROI 校正

先執行校正工具，用滑鼠在畫面上點出匝道四個角落（左上 → 右上 → 右下 → 左下）：

```bash
python3 roi_calibrator.py rtsp://192.168.1.100:8554/camera1
# 或用影片檔測試
python3 roi_calibrator.py 國8東.mp4
```

執行後終端機會輸出：

```
✅ ROI 座標 (貼入 DEFAULT_ROI_POINTS):
np.array([[390, 150], [680, 150], [811, 443], [300, 443]], dtype=np.float32)

JSON 格式:
[[390, 150], [680, 150], [811, 443], [300, 443]]
```

將座標貼入 `ramp_analyzer.py` 的 `DEFAULT_ROI_POINTS`，或透過 `--roi` 參數傳入。

---

## 速度校正

速度估算依賴 **Homography 矩陣**，需要在現場量測後填入準確座標。

### 步驟

**1. 選 4 個地面控制點（GCP）**

在匝道地面選取 4 個可從監控鏡頭清晰辨識的點，例如：
- 路面標線的交叉點
- 地面反光標記
- 道路邊線與橫線的交點

**2. 記錄像素座標**

暫停影像，記錄這 4 個點在畫面中的像素座標 `(x, y)`。

**3. 實際量測距離**

在現場用捲尺量測這 4 個點的相對位置（公尺），以其中一點為原點 `(0, 0)` 建立座標系。

**4. 填入 `ramp_analyzer.py`**

```python
DEFAULT_PIXEL_PTS = np.array([
    [440, 180],   # GCP-1 像素座標
    [620, 180],   # GCP-2
    [700, 380],   # GCP-3
    [380, 380],   # GCP-4
], dtype=np.float32)

DEFAULT_WORLD_PTS = np.array([
    [0.0, 50.0],  # GCP-1 實際座標（公尺）
    [4.0, 50.0],  # GCP-2
    [4.0,  0.0],  # GCP-3
    [0.0,  0.0],  # GCP-4
], dtype=np.float32)
```

> **提示**：一般單線匝道寬約 3.5–4 公尺，可利用車道線間距輔助估算。

---

## FastAPI 整合

在你現有的 `main.py` 加入以下幾行即可掛載：

```python
from fastapi import FastAPI
from ramp_analyzer import RampAnalyzer, make_fastapi_router

app = FastAPI()
analyzer = RampAnalyzer(
    model_path="yolov8s.engine",
    conf=0.35,
)

app.include_router(make_fastapi_router(analyzer), prefix="/ramp")
```

啟動：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/ramp/stats` | 回傳最新統計 JSON |
| `GET` | `/ramp/stream` | MJPEG 即時影像串流 |

---

## Vue 3 前端整合

```vue
<template>
  <RampDashboard
    api-base="http://jetson-ip:8000"
    cam-label="國8東 9K+768 新市交流道"
    :poll-ms="1000"
  />
</template>

<script setup>
import RampDashboard from './RampDashboard.vue'
</script>
```

前端需要 **Element Plus**：

```bash
npm install element-plus
```

---

## 參數說明

### `RampAnalyzer` 建構子

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `model_path` | `str` | `"yolov8s.pt"` | YOLOv8 模型路徑（`.pt` 或 `.engine`） |
| `roi_points` | `np.ndarray` | 見程式碼 | 匝道 ROI 四頂點像素座標 |
| `pixel_pts` | `np.ndarray` | 見程式碼 | Homography 像素控制點 |
| `world_pts` | `np.ndarray` | 見程式碼 | Homography 實際世界座標（公尺） |
| `conf` | `float` | `0.35` | YOLO 偵測置信度門檻 |
| `iou` | `float` | `0.5` | NMS IoU 門檻 |
| `track_timeout` | `float` | `3.0` | 追蹤 ID 消失後保留秒數 |

### 命令列參數

| 參數 | 說明 |
|------|------|
| `--source` | 影像來源（RTSP URL / 影片路徑 / `0` 代表 webcam） |
| `--model` | 模型路徑 |
| `--conf` | 偵測置信度（預設 `0.35`） |
| `--roi` | ROI 頂點 JSON 字串 |

---

## 輸出格式

`/ramp/stats` 回傳 JSON：

```json
{
  "timestamp": 1742480000.123,
  "total": 5,
  "by_type": {
    "car": 3,
    "truck": 1,
    "motorcycle": 1
  },
  "avg_speed": 38.4,
  "max_speed": 62.1,
  "overspeed_count": 0,
  "tracks": [
    {
      "id": 12,
      "type": "car",
      "speed_kmh": 35.2,
      "zone": "正常",
      "bbox": [420, 180, 510, 240]
    }
  ]
}
```

---

## 速度區間定義

| 區間 | 判定 | 顯示顏色 |
|------|------|----------|
| 0–40 km/h | 正常 | 綠色 |
| 40–60 km/h | 偏快 | 藍色 |
| 60–80 km/h | 快速 | 橘色 |
| > 80 km/h | 超速 | 紅色 |

> 匝道設計車速通常為 40 km/h，請依實際標誌調整門檻。

---

## 車種對照表

| COCO Class ID | 英文 | 中文 |
|:---:|------|------|
| 2 | car | 轎車 / 小客車 |
| 3 | motorcycle | 機車 / 重機 |
| 5 | bus | 客運 / 遊覽車 |
| 7 | truck | 卡車 / 聯結車 |

---

## 常見問題

**Q：速度顯示 `0.0 km/h` 或明顯偏低？**
需至少累積 3 幀才開始計算速度。若偏低請檢查 `DEFAULT_WORLD_PTS` 是否正確填入實際公尺距離。

**Q：ROI 外的車輛也被偵測到？**
YOLO 會偵測整幀，系統以**車輛中心點**判斷是否在 ROI 內，邊緣車輛若中心點剛好在 ROI 外會被濾除，此為正常行為。

**Q：Jetson 上出現 `nvbufsurface: Failed to create EGLImage`？**
在無頭環境（headless / VNC）已知問題。在 `ramp_analyzer.py` 中加入：
```python
import os
os.environ["DISPLAY"] = ""
```
或改用 `cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_FFMPEG)`。

**Q：想換成 YOLOv8m 提升精度？**
直接修改 `--model yolov8m.engine`，其餘不變。Xavier NX 上 `yolov8s` 約 30+ FPS，`yolov8m` 約 15–20 FPS。

**Q：如何同時分析多支攝影機？**
為每支攝影機各建一個 `RampAnalyzer` 實例，並在 FastAPI 以不同 prefix 分別掛載，例如 `/ramp/cam1`、`/ramp/cam2`。
