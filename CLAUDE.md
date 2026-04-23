# 交通違規影像分析系統 — Claude 指引

Jetson NX 邊緣運算：車輛偵測、車牌辨識、違規偵測、**壅塞偵測**、NVR 整合。

## 技術棧
- **後端**：FastAPI（:8000）+ SQLAlchemy + SQLite（`data/violations.db`）
- **前端**：Vue 3 SPA（`web/index.html`）+ Element Plus
- **AI**：YOLOv8n + TensorRT（車輛）、YOLO26s-cls（大型車細分）、Tesseract（車牌 OCR）
- **NVR**：Frigate + MQTT
- **平台**：JetPack 6.0 / CUDA 12.2 / TensorRT 8.6 / Python 3.10

## 關鍵目錄
| 路徑 | 用途 |
|------|------|
| `api/routes/` | FastAPI 路由（stream、lpr、congestion、frigate、violations …） |
| `api/routes/congestion.py` | **壅塞偵測服務**（狀態、啟停、串流） |
| `detection/congestion_detector.py` | **壅塞偵測核心演算法**（佔用率、四級等級） |
| `detection/vehicle_detector.py` | YOLOv8 車輛偵測（整合大型車分類） |
| `detection/violation_detector.py` | 違規規則引擎（闖紅燈/超速/停車/逆向） |
| `recognition/plate_recognizer.py` | 車牌 OCR |
| `config/frigate/config.yml` | Frigate NVR 設定 |

## 重要文件
- `README.md` — 系統總覽、架構圖、安裝部署
- `RUNBOOK.md` — 運維手冊
- `API整合文件.md` — 對外 API 規格
- `DB_ERMODEL.md` — 資料庫 ER Model
- `ocr 流程.md` — 車牌 OCR pipeline
- `ramp_analyzer_README.md` — 匝道分析器
- `MODEL_PATHS_PROPOSAL.md` / `model_paths.py` — 模型路徑管理

## 攝影機命名慣例
格式：`<camera_id>_<lane_id>`，例如 `62_1` 表示 62 號攝影機第 1 車道。

## 開發注意
- 模型檔（`models/*.pt`, `*.engine`）與 `storage/` 不納入版控。
- 即時串流除錯：直接從 live overlay 抓 frame 診斷，不要等使用者截圖。
- 自動行為必須對應使用者設定 toggle，不要 hardcode。
