# 車牌辨識（LPR）架構 — 實際線上版

> 本文對照 2026-08-13 的程式碼逐行核對後改寫，描述的是**現場已驗收、正在跑的**流程。
> 舊版文件寫的「Tesseract 雙路徑 / 切字 fallback / 224×72 normalize」已經不是實際路徑，
> 相關程式碼還留在檔案裡但不會執行，見文末〈已停用的舊路徑〉。

---

## 1. 系統組成：兩個 process

```
traffic-api   :8000    主程式。LPR 串流任務跑在 process 內（每台攝影機一個 thread）
traffic-ocr   :8010    services/ocr_service.py — YOLO 字元偵測微服務
```

拆成兩個 process 的原因（`recognition/plate_recognizer.py:41`）：**字元 YOLO 不要跟主
偵測 YOLO 搶 GPU**。微服務是裸 `http.server`，模型 load 一次常駐，收 PNG bytes 回 JSON。

兩個都是 systemd unit，`traffic-api.service` 宣告 `After/Wants=traffic-ocr.service`。

```bash
systemctl status traffic-ocr traffic-api
curl -s http://127.0.0.1:8010/          # 健康檢查
```

---

## 2. 模型

| 用途 | 檔案 | 載入位置 |
|---|---|---|
| 車牌框偵測 | `models/lpr/plate_yolov8n.engine`，**不存在才退回** `.pt` | `recognition/plate_detector.py:17` |
| 字元偵測 | `models/lpr/Charcter-LP.pt` | `services/ocr_service.py:114` |

**現場實況**：`models/lpr/` 只有 `plate_yolov8n.pt`（19MB），**沒有 `.engine`**
→ 車牌框走的是 PyTorch，不是 TensorRT。

**字元模型是 2026-07-13 finetune 後的版本**（`Charcter-LP.pt` 與
`Charcter-LP_tw_v1.pt` 同為 52,029,963 bytes）。回滾方式：

```bash
cp models/lpr/Charcter-LP.pt.before_finetune models/lpr/Charcter-LP.pt
sudo systemctl restart traffic-ocr.service
```

`.pt` 可跨機器搬（`.engine` 不行，綁 GPU/TensorRT 版本）。

---

## 3. 對外端點

**串流辨識**（`api/routes/lpr_stream.py`，prefix `/api/lpr/stream`）

| 方法 | 路徑 | 用途 |
|---|---|---|
| POST | `/start/{camera_id}` | 起一個背景辨識 thread |
| POST | `/stop/{camera_id}` | 停 |
| GET | `/status/{camera_id}` | 任務狀態 + debug 計數器 |
| GET | `/results/{camera_id}` | 該台最近結果 |
| GET | `/history` | 歷史紀錄查詢（支援 `min_confidence` 過濾） |
| GET | `/camera-options` | 可選攝影機清單 |
| GET | `/snapshot/{filename}` | 車牌截圖 |
| GET | `/all` | 全部任務總覽 |

**單張辨識**（`api/routes/lpr.py`，prefix `/api/lpr`）

`/recognize-upload`、`/recognize-base64`、`/recognize-camera/{camera_id}`
—— 三者最後都收斂到同一個 `_recognize_plate_on_crop()`。

---

## 4. 串流逐幀流程（實際）

```
frame
  ├─ cam_2 / cam_6 → shared_frames（detection worker 經 frigate latest.jpg 拿的 1080p）
  │                   取不到 → 直接 GET frigate /api/cam_N/latest.jpg
  └─ 其他攝影機     → cap.read()
        ↓
車輛偵測 + VehicleTracker 追蹤（只留車類別）
        ↓
車道 ROI 過濾（見 §7）
        ↓
PlateDetector.detect(conf=0.12) 找車牌框 → crop
        ↓
_recognize_plate_on_crop(crop)  →  recognizer.recognize_easy(crop)
        ↓
【5 變體 ensemble】original / clahe / upscale_2x / bilateral / gray_otsu
   每個變體各 POST 一次到 :8010
   加權投票：先比出現次數，同票再比平均 conf
   多變體同意加 bonus（每多 1 票 +0.05，上限 +0.15，總分封頂 0.99）
   全部 fail → recognize_chars() 字元分割 fallback
        ↓
【多幀空間投票】以車牌中心切 bucket，bucket 邊長 160px、TTL 3.5s
        ↓
confirm 判定（§6）
        ↓
commit 判定（§6）+ 同車牌 cooldown 20s
        ↓
_enforce_plate_format 邊界修復（§5）
        ↓
寫入 lpr_records（含 vehicle_bbox）+ 更新 lpr_camera_stats
```

### 影像來源分流為什麼要分

`_SHARED_FRAME_LPR_CAMS = {2, 6}`（`lpr_stream.py:2842`）。

這兩台**不可以走 `cap.read()`** —— 對 go2rtc 來源會偶發 native SEGV，
把整個 traffic-api process 拉垮。改吃 detection worker 已經取好的
frigate `latest.jpg`（1080p），取不到才自己去 GET 一次。

新增攝影機若也走 go2rtc，要一併加進這個集合。

---

## 5. 台灣車牌格式修復（兩層）

### 第一層：微服務內 `_repair_plate()`（`ocr_service.py:45`）

字元 YOLO 出框之後：

1. **y 座標過濾** —— 只留主要那一行（濾掉牌框上下的雜訊字）
2. **單字元 conf < 0.4 丟掉** —— 避免一個糊字拖低整體
3. 依 x 排序組成 `raw_text`，`avg_conf` = 各字元平均
4. **漏字偵測** —— 相鄰字元 gap / 字寬中位數 > 1.5 就判定可能漏字，
   依比例給 penalty（1.5→×0.7、2.0→×0.5、2.5+→×0.35）
5. **格式比對 + 相似字 swap** —— 對到台灣車牌格式（swap ≤ 2、字母位不含 I/O/Q）
   就用修復後的字串，`conf = avg_conf × (0.6 + 0.4×repair_score) × penalty`
6. 對不到格式 —— 仍回傳（含 dash 方便人眼看），但 **conf × 0.35 大幅降權**，
   交給下游門檻濾掉

### 第二層：存 DB 前 `_enforce_plate_format()`（`lpr_stream.py:146`）

投票層本身**沒有格式檢查**，所以在寫進 DB 前再擋一次。

原則（寫在 docstring 裡，改的時候不要破壞）：

- **只修不丟** —— 修不成合法格式就原樣回傳，由 `min_confidence` 去濾
- **對已合法車牌 idempotent** —— 重複套用結果不變
- 一樣是 swap ≤ 2、字母位排除 I/O/Q

> 這層是 2026-07 修「23% 無效格式高信心存進 DB」加的。
> 典型案例：真牌 `BKA-5681` 被讀成 `BKAS-681`（5→S 邊界誤讀）。
> 根因是 `lpr_stream` in-process 路徑與 `:8010` 兩套後處理不一致，OCR 模型本身沒錯。

---

## 6. 門檻常數（`lpr_stream.py:58-73`）

| 常數 | 值 | 意義 |
|---|---|---|
| `_PLATE_VOTE_BUCKET_SIZE` | 160 | 空間投票 bucket 邊長（px） |
| `_PLATE_VOTE_TTL_SEC` | 3.5 | 投票桶存活秒數 |
| `_PLATE_CONFIRM_MIN_COUNT` | **2** | confirm 需要的票數 |
| `_PLATE_CONFIRM_MIN_SCORE` | **1.8** | 或加權分數達標 |
| `_PLATE_COMMIT_MIN_SCORE` | **1.5** | 寫 DB 的最低分數 |
| `_PLATE_COMMIT_MIN_CONF` | 0.40 | 寫 DB 的最低信心 |
| `_PLATE_COMMIT_MIN_QUALITY` | 0.08 | 寫 DB 的最低影像品質 |
| `_PLATE_COMMIT_COOLDOWN_SEC` | 20.0 | 同車牌重複寫入冷卻 |

**confirm** = `vote_count ≥ 2` **或** `score ≥ 1.8` **或** 強單幀 **或** 極強單幀。

程式碼註解留著調整痕跡（`3 → 2`、`2.4 → 1.8`、`2.0 → 1.5`、`0.12 → 0.08`）——
整體策略是**放寬收錄、靠格式修復把關**，不是靠高門檻。改門檻前先看懂這個取捨。

---

## 7. 車道 ROI 過濾（`lpr_stream.py:2765`）

```python
_exclude = {"no_parking", "sidewalk", "red_line"}
```

LPR 只用**車道相關**的 zone（車流/測速）。

🛑 **禁停區 / 人行道 / 紅線不是車道偵測範圍**。以前沒排除時，cam_6 只畫了禁停區
→ 主車道的車全被濾掉 → `vehicles_detected` 恆為 0，看起來像模型壞了。

---

## 8. 資料落地

| 表 | 內容 |
|---|---|
| `lpr_records` | 每筆確認車牌：`plate_number`、`confidence`、`valid`、`vehicle_type`、`snapshot`、`raw`、`vehicle_bbox`、`lane_no`、`created_at` |
| `lpr_camera_stats` | 每台累計計數器：`total_frames`、`vehicles_detected`、`plate_boxes_detected`、`ocr_candidates_detected`、`vote_candidates_detected`、`confirmed_candidates`、`committed_candidates` |
| `lpr_report_aggs` | 報表聚合桶 |

`vehicle_bbox` 是給「違規 ↔ 車牌」關聯用的：違規當下就在該車 bbox 內做 OCR，
關聯靠 bbox IoU（`track_id` 跨模組不通用）。

`lpr_camera_stats` 的七個計數器是**排查漏斗**用的 —— 哪一段掉到 0 就知道問題在哪：

```
total_frames → vehicles_detected → plate_boxes_detected → ocr_candidates
             → vote_candidates → confirmed → committed
```

---

## 9. 已停用的舊路徑（程式碼還在，但不執行）

`_recognize_plate_on_crop()`（`lpr_stream.py:942`）在**第 963 行就 `return`**，
底下標著 `# === 以下為舊 Tesseract 邏輯（停用）===` 的 200 多行不會被執行，包括：

- `_tighten_plate_crop_with_bbox()` / `_flatten_plate_roi_with_bbox()` / `_enhance_plate_snapshot()`
- normalize 到 224×72
- `recognize_chars` 切字路徑（**注意**：這個在 `recognize_easy` 內部仍是 fallback，
  只是 `_recognize_plate_on_crop` 這一層的呼叫不會走到）
- 所有 `pytesseract` 呼叫

`import pytesseract` 還在檔案頂端，但**主線已經沒有 Tesseract**。

保留現狀是刻意的：這是已驗收的狀態，清死碼會動到 `lpr_stream.py` 主檔，
要做的話必須排在現場驗證一起，不可以單獨 deploy。

---

## 10. 排查順序

1. `GET /api/lpr/stream/status/{camera_id}` 看七個計數器，找出漏斗斷在哪一段
2. `total_frames = 0` → 影像來源問題，看 §4 的來源分流
3. `vehicles_detected = 0` → 先看 ROI 是不是只畫了禁停區（§7）
4. `plate_boxes = 0` 但有車 → 車牌框模型 / 解析度不足
5. `ocr_candidates = 0` 但有 plate box → `curl http://127.0.0.1:8010/` 看微服務是否活著
6. 有 committed 但車牌是錯的 → 看 `raw` 欄位與 `matched_format`，判斷是 OCR 讀錯
   還是格式修復修歪
