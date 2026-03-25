# 模糊車牌調參

## 目的

- 保留原始 plate ROI 細節
- 主 OCR 優先吃灰階增強圖
- 低信心時才走 fallback 切字
- 用多幀最佳 frame 與字元級投票收斂

## 建議流程

```text
Raw Frame
-> Plate Detector
-> Crop from Raw Frame
-> Padding
-> Perspective / Rotation Rectification
-> Normalize Size
-> Gray
-> CLAHE
-> Light Denoise
-> Light Sharpen
-> Main OCR
-> Syntax Check
-> If Low Confidence:
   -> Adaptive Threshold
   -> Char Segmentation
   -> Char OCR
-> Char-Level Voting
-> Confirmed Result
-> Cooldown
-> History Commit
```

## 目前已落地的起始值

```yaml
plate_crop:
  padding_x: 0.12
  padding_y: 0.18

roi_filter:
  aspect_ratio_min: 2.2
  aspect_ratio_max: 6.2
  min_plate_width: 90
  min_plate_height: 28
  min_plate_area: 2200

rectify:
  normalize_width: 256
  normalize_height: 80

quality:
  blur_too_low: 55
  blur_low: 85
  blur_good: 120
  brightness_min: 45
  brightness_max: 215
  contrast_min: 28

main_ocr:
  conf_threshold_accept: 0.68
  min_len: 5
  max_len: 8

fallback_ocr:
  adaptive_block_size: 19
  adaptive_C: 6
  char_min_w: 8
  char_max_w: 54
  char_min_h: 22
  char_max_h: 78
  char_min_area: 130
  char_aspect_min: 0.15
  char_aspect_max: 1.05
  char_topk: 3

voting:
  vote_window_frames: 10
  confirm_vote_min_count: 3
  confirm_vote_min_score: 2.4

history:
  commit_vote_score: 2.9
  cooldown_seconds_same_plate_same_cam: 30
```

## 備註

- 目前 live code 預設已關閉拖板車尾專用 `rear ROI` 搜尋限制。
- 若某支固定鏡頭確實只看車尾，再另外開啟專用 ROI 與 candidate ranking。
- 模糊場景下，主路徑不建議先二值化。
