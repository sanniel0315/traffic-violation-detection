# LPR Fine-tune Toolkit

把 `models/lpr/Charcter-LP.pt` (YOLOv8 字元偵測) **fine-tune 到台灣車牌專用**,
解決 production 上「N→F」「M→G」之類固定字符誤判問題。

---

## 完整流程

### 1. 建立訓練資料集 (10-30 分鐘)

```bash
cd /home/ubuntu/traffic-violation-detection
python3 scripts/lpr_finetune/build_dataset.py \
    --db data/violations.db \
    --snapshot-dir output/violations/snapshots \
    --output dataset_finetune \
    --max-samples 5000
```

**策略 (weak label,不需大量人工標註)**:
- 從 DB 拿 `license_plate` (既有 OCR 高信心結果視為 pseudo ground truth)
- 對 `{vid}_violation_plate.png` 跑 `:8010` OCR 服務拿 char bbox
- OCR 結果 **完全等於** DB plate → 該圖標籤可信,寫 YOLO label
- 不等 → 拋棄 (避免 reinforce 既有錯誤)
- 8/2 split train/val

**輸出**:
```
dataset_finetune/
├── images/train/    # ~3500 張 plate.png
├── images/val/      # ~700 張
├── labels/train/    # YOLO label .txt
├── labels/val/
└── data.yaml        # ultralytics config
```

### 2. 人工補標 difficult cases (可選,效果加成)

build_dataset.py 拋棄的「mismatched」是真正難 case (OCR 認錯)。
找 10-50 張代表性的人工標 char bbox + class,複製到 dataset:

```bash
# 用 labelImg / Roboflow / CVAT 標
# 標完輸出 YOLO format,複製到 dataset_finetune/images/train/ + labels/train/
```

### 3. 訓練

```bash
yolo detect train \
    data=dataset_finetune/data.yaml \
    model=models/lpr/Charcter-LP.pt \
    epochs=80 \
    imgsz=320 \
    batch=32 \
    device=0 \
    project=runs_finetune \
    name=tw_plates_v1
```

Jetson AGX Orin 上預估:
- 4000 image / 80 epoch → 約 2-4 小時
- GPU 利用率 ~85%

### 4. 驗證

訓練完看 `runs_finetune/tw_plates_v1/results.png`:
- mAP50 應該 > 0.95 (字元偵測本身不難)
- 重點看 **confusion matrix** — 之前易誤判的 (N/F, M/G) 是否分得開

### 5. Swap weights

```bash
# 備份原 model
sudo cp models/lpr/Charcter-LP.pt models/lpr/Charcter-LP.pt.before_finetune

# 替換成新 weights
sudo cp runs_finetune/tw_plates_v1/weights/best.pt models/lpr/Charcter-LP.pt

# restart OCR 服務
sudo systemctl restart traffic-ocr.service

# 確認服務起來
curl http://127.0.0.1:8010/ 2>&1 | head -5
```

### 6. A/B 驗證

跑同一批 plate.png 比較 fine-tune 前後 OCR 結果:

```python
# scripts/lpr_finetune/eval.py (TODO 待寫)
# 比較 production 既有 violations + 新 model 結果
```

---

## 回滾 (萬一新 model 變差)

```bash
sudo cp models/lpr/Charcter-LP.pt.before_finetune models/lpr/Charcter-LP.pt
sudo systemctl restart traffic-ocr.service
```

---

## Notes

- **Weak label 天花板**: 只用 DB pseudo label 訓出來的 model **無法超越原 model 的天花板** (對 N→F 認錯的 case 仍會錯),因為 mismatched 都被丟掉。要真正提升必須人工標 difficult cases (步驟 2)。
- **資料量**: 4000+ samples 對 36-class YOLO detector 足夠。少於 1000 容易 overfit。
- **OCR 服務必須 running**: build_dataset 跑時需要 `:8010` 服務在線 (否則拿不到 char bbox)。
- **DB 持續累積中**: 每多 1 天 production 就多 ~3000 違規 plate.png,訓練資料越來越多。
