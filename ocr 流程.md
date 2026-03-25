# OCR 流程

## 入口

- 單張辨識 API: [api/routes/lpr.py](/home/mic-711/projects/traffic-violation-detection/api/routes/lpr.py)
- 串流辨識 API: [api/routes/lpr_stream.py](/home/mic-711/projects/traffic-violation-detection/api/routes/lpr_stream.py)
- 車牌 detector: [recognition/plate_detector.py](/home/mic-711/projects/traffic-violation-detection/recognition/plate_detector.py)
- OCR recognizer: [recognition/plate_recognizer.py](/home/mic-711/projects/traffic-violation-detection/recognition/plate_recognizer.py)

目前主流程分成兩種：

1. 單張辨識 `api/routes/lpr.py`
2. 串流辨識與歷史紀錄 `api/routes/lpr_stream.py`

## 單張辨識流程

### 1. 取得影像

- 來源可能是 upload、base64、camera snapshot。
- camera snapshot 會先透過 `resolve_analysis_source(camera)` 取得分析來源。

### 2. 找車牌框

- 使用 `PlateDetector.detect(frame, conf=0.12)` 找候選車牌框。
- 每個候選框會被裁成 `crop`。

### 3. OCR

- 每個 `crop` 會送進 `_recognize_plate_on_crop()`。
- 這個函式實作在 `lpr_stream.py`，單張 API 直接共用。

### 4. 候選文字評分

- OCR 輸出會做：
  - 正規化
  - 產生 plate variants
  - 格式分數 `_plate_layout_score()`
  - 最終分數 `_score_ocr_result()`
- 分數最高的候選 plate 會回傳。

## 串流辨識主流程

### 1. 啟動任務

- `POST /api/lpr/stream/start/{camera_id}`
- 建立 `LPRStreamTask`
- 每台 camera 一個背景 thread

任務內的重要狀態：

- `vehicle_tracker`: 用來追蹤同一台車
- `vehicle_track_states`: 記錄每個 track 的狀態
- `pending_plate_votes`: OCR 投票桶
- `last_committed_plates`: history commit cooldown 用

## 目前實際 11 步流程

```text
frame
-> vehicle detect
-> vehicle track
-> plate detect
-> plate crop + padding + tighten
-> quality check + normalize
-> main whole-plate OCR
-> low-confidence fallback char OCR
-> syntax repair + candidate scoring
-> char-level voting
-> confirmed
-> commit history
```

## 串流逐幀處理流程

### 1. 開串流

- `_open_capture(source)` 依來源選 backend
- 持續讀 frame

### 2. 車輛偵測

- 先抓車輛 bbox
- 只保留車類別
- 用 `VehicleTracker` 追蹤成 track

### 3. 對每台車找車牌

對每個 vehicle：

- 先取車輛 crop
- 用 `PlateDetector` 直接找 plate bbox
- 若 detector 不穩，還會走 heuristic plate proposal

### 4. 車牌 crop 微調

候選 plate 會再經過：

- `_tighten_plate_crop_with_bbox()`
- `_flatten_plate_roi_with_bbox()`
- `_enhance_plate_snapshot()`

目的：

- 把 plate 框收緊
- 修正傾斜
- 放大與增強字元邊緣

### 5. 品質檢查與 normalize

`_recognize_plate_on_crop()` 進 OCR 前會先做：

- 品質指標 `_plate_quality_metrics()`
- 品質分級 `_plate_quality_level()`
- 品質分數 `_plate_quality_score()`
- normalize 到固定尺寸 `224 x 72`

目前會看：

- 寬高
- blur
- brightness
- contrast
- aspect ratio
- angle

太差的 crop 不會直接往下 commit。

### 6. OCR 路徑

`_recognize_plate_on_crop()` 目前是雙路徑：

1. 主路徑：整牌 OCR
2. 備援路徑：切字 OCR

#### 主路徑

- `_recognize_plate_fast()`
- 或 `PlateRecognizer.recognize()`

先直接讀整張 plate ROI。

#### 備援切字路徑

- `_segment_plate_characters()`
- `_ocr_single_character_candidates()`
- `_rebuild_plate_from_char_candidates()`

只有主路徑不夠穩時才啟動。

### 7. OCR 結果清理

OCR 結果會做：

- `_clean_plate_text()`
- `_validate_plate_text()`
- `_plate_text_candidates()`
- `_plate_variants()`
- `_score_ocr_result()`
- `_is_plausible_plate()`

只有通過基本合理性檢查的 plate 才會往下走。

### 8. 多幀字元級投票

為了避免單幀誤判，串流流程有 plate vote bucket：

- `_PLATE_VOTE_BUCKET_SIZE = 160`
- `_PLATE_VOTE_TTL_SEC = 3.5`

同一空間 bucket 內的候選會累積：

- 次數 `count`
- 最佳信心度 `best_conf`
- 是否 valid
- 原始 raw OCR
- 字元位置投票 `char_votes`

投票權重不是只看次數，也會看：

- OCR confidence
- plate detector confidence
- quality score
- syntax score
- center score

### 9. confirmed 條件

目前 confirmed 代表：

- 已經通過候選合理性檢查
- 有進投票桶
- `vote_count` 或 `vote_score` 已達標

確認條件目前為：

- `vote_count >= 3`
- 或 `vote_score >= 2.2`
- 或 `valid 且 conf >= 0.50`
- 或 `conf >= 0.72`

若投票還不夠，也允許強單幀 fallback：

- `conf >= 0.55`
- `score >= 3.10`
- `layout >= 1.20`

### 10. commit history 條件

confirmed 不等於直接寫 history。

目前 commit 需要再通過：

- `plate != UNKNOWN`
- `raw != vehicle_only`
- `valid = true`
- `vote_score >= 2.8`
- `confidence >= 0.42`
- `quality_score >= 0.40`，除非 `conf` 很高
- 非單幀弱候選
- 同 plate cooldown 通過

cooldown 目前是：

- `30 秒`
- key 是 `plate`

### 11. 寫入內容

`_store_history_record()` 會寫：

- `plate_number`
- `confidence`
- `valid`
- `vehicle_type`
- `snapshot`
- `raw`
- `camera_id`
- `camera_name`
- `created_at`

DB model 是 `LPRRecord`。

## UNKNOWN / vehicle_only

若某個 vehicle track 消失前都沒成功辨識到 plate：

- `_flush_inactive_vehicle_tracks()` 會補寫一筆 `UNKNOWN`
- `raw = vehicle_only`

這些資料預設不會出現在歷史頁，除非查詢時加 `include_unknown=true`。

## 頁面狀態欄位怎麼看

LPR 頁面的即時統計目前大致對應：

- `已處理幀`: 已進入逐幀處理
- `車輛數`: vehicle detect + track 有命中
- `車牌框`: plate detector 有命中
- `OCR 候選`: OCR 至少產出過 1 個候選字串
- `投票命中`: 有候選進入 confirmed

注意：

- `投票命中` 不等於辨識正確
- `投票命中` 也不等於已寫進 history
- 真正 history commit 還要再過 commit 條件

## 歷史查詢

查詢 API:

- `GET /api/lpr/stream/history`

預設行為：

- 只顯示有 plate 的記錄
- 排除 `plate_number = UNKNOWN`
- 排除 `raw = vehicle_only`

因此你在 UI 看到的「車牌歷史」通常會比實際總寫入數少。

## 為什麼會覺得歷史偏少

常見原因有三個：

1. 預設查詢把 `UNKNOWN` 全排掉
2. 候選有進 confirmed，但 commit 被擋掉
3. crop 品質差，OCR 被 syntax repair 硬修或直接淘汰

目前已調整：

- 主流程改成整牌 OCR 優先
- 備援切字改成 top-k 候選
- 多幀投票改成字元級投票
- history 改成 `confirmed -> commit` 分離
- commit 目前改成偏保守，先擋掉假牌

## 目前實際判斷順序

1. 讀 frame
2. 偵測 vehicle
3. 追蹤 vehicle track
4. 對 vehicle 內找 plate bbox
5. 微調 plate crop + normalize
6. quality check
7. 主 OCR / 備援切字 OCR
8. syntax repair + 候選評分
9. bucket 投票
10. confirmed
11. commit history
12. 沒 plate 的 stale track 視情況補 `UNKNOWN`

## 建議排查順序

如果辨識數量異常少，先看：

1. `GET /api/lpr/stream/status/{camera_id}`
2. `last_candidate_plate`
3. `last_confirmed_plate`
4. `confirmed_candidates`
5. `committed_candidates`
6. `/api/lpr/stream/history?include_unknown=true`

判讀方式：

- `vehicles_detected` 高，但 `vote_candidates_detected` 低：plate detector / crop 品質有問題
- `vote_candidates_detected` 高，但 `committed_candidates` 低：commit 條件或品質門檻擋住
- `include_unknown=true` 很多，但正常 history 很少：OCR 成功率不足，不是沒抓到車
