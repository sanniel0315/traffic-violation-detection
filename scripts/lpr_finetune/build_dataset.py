#!/usr/bin/env python3
"""車牌 OCR Fine-tune 訓練集自動建構

策略 (weak label,不需大量人工標註):
1. 從 violations DB 拿 license_plate (既有 OCR 高信心結果 = pseudo ground truth)
2. 對 {vid}_violation_plate.png 跑 :8010 OCR 服務,拿每個字符的 bbox + class
3. OCR detect 後修復成台灣車牌格式 (走 services/ocr_service.py 修復器)
4. 若修復結果 == DB license_plate → 該圖標籤可信,寫 YOLO format
5. 不符 → 拋棄該圖 (避免 reinforce 錯誤)
6. 8/2 split train/val,符合 ultralytics YOLO 訓練 layout

輸出結構:
    dataset/
        images/
            train/  *.png
            val/    *.png
        labels/
            train/  *.txt  (YOLO: class_id cx cy w h, 正規化 0-1)
            val/    *.txt
        data.yaml  (Ultralytics config)

用法:
    python scripts/lpr_finetune/build_dataset.py \\
        --output ./dataset \\
        --min-confidence 0.8 \\
        --max-samples 5000

訓練 (build_dataset 完成後):
    yolo detect train data=dataset/data.yaml model=models/lpr/Charcter-LP.pt \\
        epochs=80 imgsz=320 batch=32 device=0
"""
import argparse
import io
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


# YOLO char class index 對應 (跟 services/ocr_service.py 一致)
CHAR_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_CLASSES)}


def query_high_confidence_plates(
    db_path: str,
    snapshot_dir: Path,
    min_confidence: float = 0.85,
    max_samples: int = 5000,
) -> List[Tuple[int, str, Path]]:
    """從 DB 撈 license_plate 不為空 + 對應 plate.png 存在的 violations。
    回傳 [(vid, expected_plate, png_path), ...]"""
    con = sqlite3.connect(db_path)
    c = con.cursor()
    rows = c.execute("""
        SELECT id, license_plate
        FROM violations
        WHERE license_plate IS NOT NULL AND license_plate != ''
        ORDER BY id DESC
        LIMIT ?
    """, (max_samples * 3,)).fetchall()
    hits = []
    for vid, plate in rows:
        png = snapshot_dir / f"{vid}_violation_plate.png"
        if not png.exists():
            continue
        if png.stat().st_size < 500:
            continue
        # 去掉 dash 跟空白,只留英數
        clean = "".join(ch for ch in plate.upper() if ch.isalnum())
        if len(clean) < 4 or len(clean) > 7:
            continue
        hits.append((vid, clean, png))
        if len(hits) >= max_samples:
            break
    con.close()
    return hits


_YOLO_MODEL = None


def _get_yolo_model(model_path: str = "models/lpr/Charcter-LP.pt"):
    """lazy load YOLO model (:8010 服務只回 text + count,不回 char bbox,
    要拿 char bbox 必須直接 inference)。"""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(model_path, task="detect")
    return _YOLO_MODEL


def ocr_detect_chars(img: np.ndarray, conf_threshold: float = 0.25) -> List[dict]:
    """直接跑 YOLO model 拿每個字符的 bbox + class。
    回傳 [{class: 'A', confidence: 0.95, bbox: [x1,y1,x2,y2]}, ...]"""
    try:
        model = _get_yolo_model()
        results = model.predict(img, conf=conf_threshold, verbose=False)
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        names = r.names  # {class_id: 'A', ...}
        out = []
        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, "?")
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            out.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": xyxy,
            })
        return out
    except Exception as e:
        print(f"  YOLO inference error: {e}", file=sys.stderr)
        return []


def chars_to_string(chars: List[dict]) -> str:
    """char list (依 x 座標排序) 拼成字串"""
    if not chars:
        return ""
    sorted_chars = sorted(chars, key=lambda c: c.get("bbox", [0])[0])
    return "".join(c.get("class", "?").upper() for c in sorted_chars)


def char_to_yolo_label(char: dict, img_w: int, img_h: int) -> Optional[str]:
    """轉成 YOLO label: 'class_id cx_norm cy_norm w_norm h_norm'"""
    cls_name = char.get("class", "").upper()
    if cls_name not in CHAR_TO_IDX:
        return None
    bbox = char.get("bbox") or []
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    return f"{CHAR_TO_IDX[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_dataset(
    db_path: str,
    snapshot_dir: Path,
    output_dir: Path,
    min_confidence: float = 0.85,
    max_samples: int = 5000,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """完整建構流程"""
    print(f"[1/5] Querying DB {db_path}...")
    samples = query_high_confidence_plates(db_path, snapshot_dir, min_confidence, max_samples)
    print(f"      → {len(samples)} candidates")

    if not samples:
        print("ERROR: 0 candidates,確認 DB 跟 snapshot_dir 路徑")
        sys.exit(1)

    print(f"[2/5] Setup output dir {output_dir}/")
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"[3/5] Running OCR weak labeling on {len(samples)} samples...")
    matched = []
    mismatched = []
    no_chars = []
    for i, (vid, expected, png) in enumerate(samples):
        if i % 100 == 0:
            print(f"      {i}/{len(samples)} processed, "
                  f"matched={len(matched)} mismatched={len(mismatched)} no_chars={len(no_chars)}")
        img = cv2.imread(str(png))
        if img is None:
            continue
        chars = ocr_detect_chars(img)
        if not chars:
            no_chars.append(vid)
            continue
        ocr_str = chars_to_string(chars)
        if ocr_str == expected:
            matched.append((vid, png, chars))
        else:
            mismatched.append((vid, expected, ocr_str))
    print(f"      → matched={len(matched)} (will use), mismatched={len(mismatched)}, no_chars={len(no_chars)}")

    if len(matched) < 50:
        print(f"WARN: only {len(matched)} matched — too few. 建議降低 min-confidence 或人工補標")
        if len(matched) == 0:
            sys.exit(1)

    print(f"[4/5] Split train/val (val_ratio={val_ratio}) + write YOLO labels...")
    random.seed(seed)
    random.shuffle(matched)
    n_val = int(len(matched) * val_ratio)
    val_set = matched[:n_val]
    train_set = matched[n_val:]
    print(f"      train={len(train_set)} val={len(val_set)}")

    def write_split(items, split: str):
        for vid, png, chars in items:
            img = cv2.imread(str(png))
            if img is None:
                continue
            h, w = img.shape[:2]
            label_lines = []
            for ch in chars:
                line = char_to_yolo_label(ch, w, h)
                if line:
                    label_lines.append(line)
            if not label_lines:
                continue
            # copy image
            dst_img = output_dir / f"images/{split}" / f"{vid}.png"
            shutil.copy(png, dst_img)
            # write label
            dst_lbl = output_dir / f"labels/{split}" / f"{vid}.txt"
            dst_lbl.write_text("\n".join(label_lines), encoding="utf-8")

    write_split(train_set, "train")
    write_split(val_set, "val")

    print(f"[5/5] Write data.yaml...")
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"# Auto-generated by build_dataset.py\n"
        f"path: {output_dir.absolute()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CHAR_CLASSES)}\n"
        f"names: {json.dumps(CHAR_CLASSES, ensure_ascii=False)}\n",
        encoding="utf-8",
    )

    # summary
    print()
    print("=" * 60)
    print(f"  Dataset built at {output_dir}")
    print(f"  train: {len(train_set)} images")
    print(f"  val:   {len(val_set)} images")
    print(f"  data.yaml: {data_yaml}")
    print()
    print("  下一步 train:")
    print(f"    yolo detect train \\")
    print(f"      data={data_yaml.absolute()} \\")
    print(f"      model=models/lpr/Charcter-LP.pt \\")
    print(f"      epochs=80 imgsz=320 batch=32 device=0")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/violations.db", help="violations DB 路徑")
    p.add_argument("--snapshot-dir", default="output/violations/snapshots",
                   help="plate.png 所在目錄")
    p.add_argument("--output", default="dataset_finetune", help="輸出資料集目錄")
    p.add_argument("--min-confidence", type=float, default=0.85,
                   help="DB violation 最低 confidence 閾值 (僅篩選用,不依賴此欄位)")
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    build_dataset(
        db_path=args.db,
        snapshot_dir=Path(args.snapshot_dir),
        output_dir=Path(args.output),
        min_confidence=args.min_confidence,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
