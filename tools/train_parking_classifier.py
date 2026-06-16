#!/usr/bin/env python3
"""fine-tune 本地車位逐格分類器 (P4)。

用標註頁累積的裁切 data/parking_train/{occupied,empty}/ 訓練 YOLOv8 classify,
輸出 models/parking/local_cls.pt → parking_slot_classifier 自動啟用,取代不泛化的巴西 PKLot。

用法 (在專案根目錄,Jetson 上需加 LD_LIBRARY_PATH):
  python3 tools/train_parking_classifier.py --epochs 40 --imgsz 96

資料量建議: 每類 (occupied/empty) 至少各 ~200 張,跨時段/天氣/來源越多越泛化。
"""
import argparse
import os
import random
import shutil

SRC = os.path.join("data", "parking_train")          # 標註裁切來源
DATASET = os.path.join("data", "parking_cls_dataset")  # train/val split (暫存)
OUT = os.path.join("models", "parking", "local_cls.pt")
CLASSES = ("occupied", "empty")


def build_dataset(val_ratio: float = 0.2) -> dict:
    if os.path.isdir(DATASET):
        shutil.rmtree(DATASET)
    counts = {}
    for cls in CLASSES:
        src_d = os.path.join(SRC, cls)
        files = [f for f in os.listdir(src_d) if f.lower().endswith(".jpg")] if os.path.isdir(src_d) else []
        random.shuffle(files)
        nv = max(1, int(len(files) * val_ratio)) if files else 0
        for split, fs in (("val", files[:nv]), ("train", files[nv:])):
            d = os.path.join(DATASET, split, cls)
            os.makedirs(d, exist_ok=True)
            for f in fs:
                shutil.copy(os.path.join(src_d, f), os.path.join(d, f))
        counts[cls] = len(files)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--base", default="yolov8n-cls.pt")
    args = ap.parse_args()

    counts = build_dataset()
    print(f"資料量: {counts}")
    if any(c < 20 for c in counts.values()):
        print("⚠ 每類至少建議 20 張以上,目前太少,訓練意義不大。先去標註頁多標幾批。")

    from ultralytics import YOLO
    model = YOLO(args.base)
    results = model.train(
        data=os.path.abspath(DATASET),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=64,
        patience=10,
        verbose=True,
    )
    # 找 best.pt 複製到 models/parking/local_cls.pt
    best = None
    try:
        best = str(model.trainer.best)
    except Exception:
        sd = getattr(results, "save_dir", None)
        if sd:
            cand = os.path.join(str(sd), "weights", "best.pt")
            if os.path.exists(cand):
                best = cand
    if best and os.path.exists(best):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        shutil.copy(best, OUT)
        print(f"✅ 已輸出本地分類器: {OUT}")
        print("   重啟 traffic-api 後 parking_slot_classifier 會自動啟用 (取代 PKLot)。")
    else:
        print("⚠ 找不到 best.pt,請手動從 runs/classify/train*/weights/best.pt 複製到", OUT)


if __name__ == "__main__":
    main()
