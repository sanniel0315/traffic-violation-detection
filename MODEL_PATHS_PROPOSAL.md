# MODEL_PATHS_PROPOSAL

> 文件狀態（2026-02-26）：本文件聚焦「模型路徑統一」提案。  
> 專案目前已另外落地：
> - 登入/Auth（`/api/auth/*`）
> - 角色權限管理 UI
> - users CRUD（admin）
> 以上不屬於模型路徑範圍，已記錄於 `README.md` 與 `RUNBOOK.md`。

## 1) `rg` 命中清單（檔案與行號）

搜尋模式：
- `yolov8n.engine`
- `yolov8n.pt`
- `./yolov8n.pt`
- `models/yolov8n.pt`
- `/workspace/yolov8n.pt`
- `/workspace/yolov8n.engine`
- `/workspace/models/yolov8n.pt`
- `/workspace/models/yolov8n.engine`

命中結果：
- `api/routes/lpr_stream.py:28` `YOLO('/workspace/yolov8n.engine', task='detect')`
- `api/routes/lpr_visual.py:27` `YOLO('/workspace/yolov8n.pt')`
- `api/routes/stream.py:146` `VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.5)`
- `detection/vehicle_detector.py:24` `def __init__(self, model_path: str = 'yolov8n.pt', ...)`
- `scripts/detection_service.py:25` `VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.5)`
- `config/settings.py:24` `YOLO_MODEL: str = "yolov8n.pt"`
- `test_detection.py:9` `model = YOLO('yolov8n.pt')`

文件命中（非執行路徑，但需同步更新）：
- `README.md:100`
- `README.md:101`
- `README.md:199`
- `README.md:392`
- `RUNBOOK.md:63`
- `RUNBOOK.md:64`
- `RUNBOOK.md:65`
- `RUNBOOK.md:67`
- `RUNBOOK.md:76`
- `RUNBOOK.md:88`
- `RUNBOOK.md:138`
- `RUNBOOK.md:139`
- `RUNBOOK.md:166`

---

## 2) 統一路徑策略（提案）

### 目標
- 所有偵測模型統一放在容器內：`/workspace/models`
- 程式不再硬編碼 `/workspace/yolov8n.*` 或相對 `yolov8n.pt`
- 路徑由 `.env` 提供，並有安全預設值

### `.env` 變數
- `MODEL_DIR=/workspace/models`
- `DETECT_MODEL_ENGINE=yolov8n.engine`
- `DETECT_MODEL_PT=yolov8n.pt`

解析規則（建議）：
1. 若 `DETECT_MODEL_*` 是絕對路徑（例如 `/workspace/models/custom.engine`），直接使用。
2. 若是相對檔名（例如 `yolov8n.engine`），組合為 `${MODEL_DIR}/${DETECT_MODEL_*}`。

### LPR/車牌模型規劃
- 目前 LPR 為 `pytesseract`（非 YOLO 模型檔）；主要依賴是系統語言包與 `tessdata`。
- 建議在文件先保留以下可選變數（不必本輪落地）：
  - `TESSDATA_PREFIX`（指定 tesseract 語言資料目錄）
  - `OCR_LANGS`（例如 `eng+chi_tra`）
- 若未來 LPR 改為深度模型（例如 plate detector），可新增：
  - `LPR_MODEL_DIR=/workspace/models/lpr`
  - `LPR_DETECT_MODEL=xxx.engine|xxx.onnx`

---

## 3) `docker-compose.yml` volume 掛載策略（提案）

### API 服務
- 保留或新增：`./models:/workspace/models`
- 在 `environment` 補上：
  - `MODEL_DIR=${MODEL_DIR:-/workspace/models}`
  - `DETECT_MODEL_ENGINE=${DETECT_MODEL_ENGINE:-yolov8n.engine}`
  - `DETECT_MODEL_PT=${DETECT_MODEL_PT:-yolov8n.pt}`

### dev 服務
- 同步掛載：`./models:/workspace/models`
- 同步放入同組 `MODEL_*` 變數，避免開發環境與 API 執行行為不一致。

### Frigate 服務
- 本提案不更動 Frigate detector 模型設定（其模型路徑策略與 API 服務分離），但需在下一輪檢查是否有獨立模型/label 路徑需求。

---

## 4) 風險清單

- 風險 1：漏改某個 hardcoded 路徑，runtime 才報找不到模型。
- 風險 2：`MODEL_DIR` 與 `DETECT_MODEL_*` 組合錯誤（檔名拼錯、副檔名不對）。
- 風險 3：只有 API 服務更新 env，dev 或測試腳本未同步，導致行為不一致。
- 風險 4：Frigate 使用自己的模型配置，不應被 API 的 `MODEL_*` 變數誤導。
- 風險 5：Jetson 上檔案權限/掛載路徑不同，導致容器內可見但不可讀。

---

## 5) 回滾方法（不重建容器版）

1. 還原程式中的 env 讀取改動（回到原本硬編碼路徑）。
2. 還原 `docker-compose.yml` 新增的 `MODEL_*` env（保留 volume 可不動）。
3. 還原 `.env.example` 與文件說明。
4. 若已改 `.env`，可暫時設回舊路徑對應，降低切換風險。

---

## 6) 驗證步驟（不啟動服務）

### A. 檔案存在檢查
- 主機端確認：`models/yolov8n.pt`、`models/yolov8n.engine` 存在。

### B. 靜態路徑檢查
- `rg` 確認程式碼不再有：
  - `/workspace/yolov8n.pt`
  - `/workspace/yolov8n.engine`
  - 裸 `yolov8n.pt`（非文件案例除外）

### C. 組態檢查
- `docker-compose.yml` 應包含：
  - `./models:/workspace/models`
  - `MODEL_DIR`, `DETECT_MODEL_ENGINE`, `DETECT_MODEL_PT`

### D. 程式實際讀取路徑（靜態驗證）
- 檢查模型載入函式/初始化處：
  - 先讀 env
  - 再組出絕對路徑
  - 送入 `YOLO(...)` / `VehicleDetector(...)`

### E. 容器內路徑驗證（僅命令提案，本輪不執行）
- 可用 `docker compose run --rm api ls -l /workspace/models` 驗證掛載（本輪不執行）。

---

## 7) 下一輪要改動的檔案清單（本輪不改）

### 程式碼
- `api/routes/lpr_stream.py`
- `api/routes/lpr_visual.py`
- `detection/vehicle_detector.py`
- `api/routes/stream.py`
- `scripts/detection_service.py`
- `test_detection.py`（若希望測試腳本也一致）
- `config/settings.py`（可選，若要集中管理預設值）

### 部署設定
- `docker-compose.yml`
- `.env.example`

### 文件
- `RUNBOOK.md`
- `README.md`（建議同步，避免文件與實作落差）

---

## 8) 最小改動落地原則（下一輪）

- 優先改「模型載入點」與 `compose/.env`，不動業務流程。
- 保留舊值可回退（以 env 預設值承接）。
- 先改 API 相關執行路徑，再視需要補測試腳本與文件。
