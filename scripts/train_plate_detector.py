#!/usr/bin/env python3
"""
台灣車牌偵測模型訓練腳本 — YOLOv8n fine-tune

使用方式:
    # 1) 準備資料集（自動下載 + 轉換）
    python scripts/train_plate_detector.py prepare

    # 2) 開始訓練
    python scripts/train_plate_detector.py train

    # 3) 評估模型
    python scripts/train_plate_detector.py eval

    # 4) 匯出給 plate_detector.py 使用
    python scripts/train_plate_detector.py export

    # 一鍵全流程
    python scripts/train_plate_detector.py all

輸出:
    models/lpr/plate_yolov8n.pt          — PyTorch 權重
    models/lpr/plate_yolov8n.engine      — TensorRT 引擎 (Jetson 上才會產生)
"""
import argparse
import os
import shutil
import sys
import json
import random
import yaml
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 路徑設定
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "plate_detect"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "plate_detect"
MODEL_EXPORT_DIR = PROJECT_ROOT / "models" / "lpr"
BEST_PT = "plate_yolov8n.pt"

# ---------------------------------------------------------------------------
# 資料集來源設定
# ---------------------------------------------------------------------------
# 方案 A：Roboflow 公開台灣車牌資料集（推薦，品質穩定）
# 方案 B：自備資料集（放在 datasets/plate_detect/custom/）
# 方案 C：從現有 LPR snapshots 自動標註（半自動）

ROBOFLOW_DATASETS = [
    {
        "name": "taiwan-license-plate",
        "url": "https://universe.roboflow.com/ds/PLACEHOLDER",
        "desc": "Taiwan license plate detection (Roboflow Universe)",
    },
]


# ===================================================================
# 準備資料集
# ===================================================================
def cmd_prepare(args):
    """準備訓練資料集"""
    print("=" * 60)
    print("步驟 1：準備車牌偵測資料集")
    print("=" * 60)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 檢查是否已有自備資料
    # ------------------------------------------------------------------
    custom_dir = DATASET_DIR / "custom"
    if custom_dir.exists() and _count_images(custom_dir) > 0:
        n = _count_images(custom_dir)
        print(f"✅ 偵測到自備資料集: {custom_dir} ({n} 張圖片)")
        print("   將使用自備資料集進行訓練")
        _prepare_custom_dataset(custom_dir)
        return

    # ------------------------------------------------------------------
    # 嘗試從 Roboflow 下載
    # ------------------------------------------------------------------
    rf_ok = _try_roboflow_download()
    if rf_ok:
        return

    # ------------------------------------------------------------------
    # 嘗試從現有 snapshots 半自動建立
    # ------------------------------------------------------------------
    snapshot_dir = PROJECT_ROOT / "storage" / "lpr_snapshots"
    if snapshot_dir.exists() and _count_images(snapshot_dir) > 20:
        print(f"📸 偵測到 LPR snapshots: {snapshot_dir}")
        _prepare_from_snapshots(snapshot_dir)
        return

    # ------------------------------------------------------------------
    # 都沒有 → 印出手動指引
    # ------------------------------------------------------------------
    _print_manual_instructions()


def _count_images(d: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for f in d.rglob("*") if f.suffix.lower() in exts)


def _try_roboflow_download() -> bool:
    """嘗試用 roboflow pip 套件下載公開車牌資料集"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ℹ️  未安裝 roboflow 套件，跳過自動下載")
        print("   安裝方式: pip install roboflow")
        return False

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("ℹ️  未設定 ROBOFLOW_API_KEY 環境變數，跳過 Roboflow 下載")
        print("   設定方式: export ROBOFLOW_API_KEY=你的API金鑰")
        print("   免費取得: https://app.roboflow.com/settings/api")
        return False

    print("📥 從 Roboflow 下載台灣車牌資料集...")
    try:
        rf = Roboflow(api_key=api_key)

        # 公開的台灣車牌偵測資料集
        # 使用者可替換為自己的 workspace/project/version
        workspace = os.getenv("RF_WORKSPACE", "taiwan-plate")
        project_name = os.getenv("RF_PROJECT", "license-plate-detection")
        version_num = int(os.getenv("RF_VERSION", "1"))

        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version_num).download(
            "yolov8",
            location=str(DATASET_DIR / "roboflow"),
        )
        print(f"✅ 下載完成: {dataset.location}")
        _merge_roboflow_dataset(Path(dataset.location))
        return True
    except Exception as e:
        print(f"⚠️  Roboflow 下載失敗: {e}")
        return False


def _merge_roboflow_dataset(rf_dir: Path):
    """將 Roboflow 下載的 YOLO 格式資料集整理到標準結構"""
    for split in ("train", "valid", "test"):
        src_images = rf_dir / split / "images"
        src_labels = rf_dir / split / "labels"
        if not src_images.exists():
            continue

        dst_split = "val" if split == "valid" else split
        dst_images = DATASET_DIR / dst_split / "images"
        dst_labels = DATASET_DIR / dst_split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for img in src_images.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(img, dst_images / img.name)
                label = src_labels / f"{img.stem}.txt"
                if label.exists():
                    # 確保 class_id 統一為 0 (plate)
                    _normalize_label(label, dst_labels / label.name)

    _write_dataset_yaml()
    _print_dataset_stats()


def _normalize_label(src: Path, dst: Path):
    """確保所有 class_id 都是 0 (plate)"""
    lines = []
    for line in src.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[0] = "0"
            lines.append(" ".join(parts))
    dst.write_text("\n".join(lines) + "\n")


def _prepare_custom_dataset(custom_dir: Path):
    """
    處理自備資料集。支援兩種結構：

    結構 A（YOLO 格式，已有 labels）:
        custom/
        ├── images/
        │   ├── 001.jpg
        │   └── ...
        └── labels/
            ├── 001.txt      # YOLO 格式: class_id cx cy w h
            └── ...

    結構 B（VOC/COCO 格式）:
        custom/
        ├── images/
        │   ├── 001.jpg
        │   └── ...
        └── annotations/
            ├── 001.xml       # Pascal VOC
            └── ...
        或
        └── annotations.json  # COCO 格式
    """
    images_dir = custom_dir / "images"
    labels_dir = custom_dir / "labels"
    annotations_dir = custom_dir / "annotations"
    coco_file = custom_dir / "annotations.json"

    if labels_dir.exists():
        print("📁 偵測到 YOLO 格式標註")
        _split_dataset(images_dir, labels_dir)
    elif coco_file.exists():
        print("📁 偵測到 COCO 格式標註，轉換中...")
        _convert_coco_to_yolo(coco_file, images_dir)
        _split_dataset(images_dir, custom_dir / "labels_yolo")
    elif annotations_dir.exists():
        print("📁 偵測到 VOC 格式標註，轉換中...")
        _convert_voc_to_yolo(annotations_dir, images_dir)
        _split_dataset(images_dir, custom_dir / "labels_yolo")
    else:
        print("⚠️  找不到標註檔案，請參考以下結構:")
        print("    custom/images/  — 圖片")
        print("    custom/labels/  — YOLO 格式 txt (class_id cx cy w h)")
        return

    _write_dataset_yaml()
    _print_dataset_stats()


def _convert_coco_to_yolo(coco_json: Path, images_dir: Path):
    """COCO JSON → YOLO txt"""
    with open(coco_json) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    out_dir = coco_json.parent / "labels_yolo"
    out_dir.mkdir(exist_ok=True)

    for ann in coco["annotations"]:
        img = img_map.get(ann["image_id"])
        if not img:
            continue
        iw, ih = img["width"], img["height"]
        x, y, w, h = ann["bbox"]  # COCO: x,y,w,h (左上角)
        cx = (x + w / 2) / iw
        cy = (y + h / 2) / ih
        nw = w / iw
        nh = h / ih
        label_file = out_dir / f"{Path(img['file_name']).stem}.txt"
        with open(label_file, "a") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"   轉換完成: {len(list(out_dir.glob('*.txt')))} 個標註檔")


def _convert_voc_to_yolo(annotations_dir: Path, images_dir: Path):
    """Pascal VOC XML → YOLO txt"""
    import xml.etree.ElementTree as ET

    out_dir = annotations_dir.parent / "labels_yolo"
    out_dir.mkdir(exist_ok=True)

    for xml_file in annotations_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find("size")
        iw = int(size.find("width").text)
        ih = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            cx = (xmin + xmax) / 2 / iw
            cy = (ymin + ymax) / 2 / ih
            w = (xmax - xmin) / iw
            h = (ymax - ymin) / ih
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_file = out_dir / f"{xml_file.stem}.txt"
        label_file.write_text("\n".join(lines) + "\n")

    print(f"   轉換完成: {len(list(out_dir.glob('*.txt')))} 個標註檔")


def _split_dataset(images_dir: Path, labels_dir: Path):
    """將資料切分為 train/val/test (80/15/5)"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [f for f in images_dir.iterdir() if f.suffix.lower() in exts]
    # 只取有標註的
    paired = []
    for img in all_images:
        label = labels_dir / f"{img.stem}.txt"
        if label.exists() and label.stat().st_size > 0:
            paired.append((img, label))

    if not paired:
        print("⚠️  沒有找到圖片+標註的配對")
        return

    random.seed(42)
    random.shuffle(paired)
    n = len(paired)
    n_train = int(n * 0.80)
    n_val = int(n * 0.15)

    splits = {
        "train": paired[:n_train],
        "val": paired[n_train : n_train + n_val],
        "test": paired[n_train + n_val :],
    }

    for split_name, items in splits.items():
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in items:
            shutil.copy2(img_path, img_dir / img_path.name)
            _normalize_label(lbl_path, lbl_dir / lbl_path.name)

    print(f"   資料切分: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


def _prepare_from_snapshots(snapshot_dir: Path):
    """
    從現有 LPR snapshots 半自動建立標註。
    找出有 _plate 後綴的裁切圖，推算車牌 bbox。
    """
    print("🔧 從 LPR snapshots 建立訓練資料（半自動）...")
    print("   ⚠️  這只產生粗略標註，建議後續用標註工具校正")

    images_dir = DATASET_DIR / "custom" / "images"
    labels_dir = DATASET_DIR / "custom" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    exts = {".jpg", ".jpeg", ".png"}
    # 找出完整車輛截圖（不含 _plate）
    snapshots = [
        f
        for f in snapshot_dir.iterdir()
        if f.suffix.lower() in exts and "_plate" not in f.stem
    ]

    count = 0
    for snap in snapshots:
        plate_snap = snapshot_dir / f"{snap.stem}_plate{snap.suffix}"
        if not plate_snap.exists():
            continue

        img = cv2.imread(str(snap))
        plate_img = cv2.imread(str(plate_snap))
        if img is None or plate_img is None:
            continue

        ih, iw = img.shape[:2]
        ph, pw = plate_img.shape[:2]
        if iw < 50 or ih < 50 or pw < 20 or ph < 10:
            continue

        # 用 template matching 定位車牌在完整圖中的位置
        result = cv2.matchTemplate(img, plate_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.5:
            continue

        x, y = max_loc
        cx = (x + pw / 2) / iw
        cy = (y + ph / 2) / ih
        nw = pw / iw
        nh = ph / ih

        if nw < 0.02 or nh < 0.02 or nw > 0.9 or nh > 0.9:
            continue

        out_name = f"snap_{count:05d}"
        shutil.copy2(snap, images_dir / f"{out_name}{snap.suffix}")
        (labels_dir / f"{out_name}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
        )
        count += 1

    print(f"   產生 {count} 筆半自動標註")

    if count > 20:
        _split_dataset(images_dir, labels_dir)
        _write_dataset_yaml()
        _print_dataset_stats()
    else:
        print("   ⚠️  標註數量太少，建議手動補充資料")
        _print_manual_instructions()


def _write_dataset_yaml():
    """產生 YOLO 格式的 dataset.yaml"""
    config = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["plate"],
    }
    DATASET_YAML.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    print(f"✅ 資料集配置已寫入: {DATASET_YAML}")


def _print_dataset_stats():
    for split in ("train", "val", "test"):
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"   {split:>5}: {n_img} 圖片, {n_lbl} 標註")


def _print_manual_instructions():
    print()
    print("=" * 60)
    print("📋 手動準備資料集指南")
    print("=" * 60)
    print()
    print("方案 A：使用 Roboflow 公開資料集（最簡單）")
    print("  1. 前往 https://universe.roboflow.com 搜尋 'taiwan license plate'")
    print("  2. 推薦資料集:")
    print("     - 'Taiwan License Plate Detection'")
    print("     - 'License Plate Recognition' (含亞洲車牌)")
    print("  3. 下載 YOLOv8 格式，解壓到:")
    print(f"     {DATASET_DIR}/")
    print("     目錄結構應為: train/images/, train/labels/, val/images/, val/labels/")
    print()
    print("方案 B：自備標註資料")
    print(f"  將圖片和標註放到: {DATASET_DIR / 'custom'}/")
    print("  支援格式: YOLO txt / COCO JSON / Pascal VOC XML")
    print("  最少需要 100 張標註圖片，建議 500+ 張")
    print()
    print("方案 C：使用 Roboflow API 自動下載")
    print("  1. pip install roboflow")
    print("  2. export ROBOFLOW_API_KEY=你的金鑰")
    print("  3. export RF_WORKSPACE=你的workspace")
    print("  4. export RF_PROJECT=你的project")
    print("  5. 重新執行 python scripts/train_plate_detector.py prepare")
    print()
    print("方案 D：從其他開源資料集")
    print("  - CCPD (中國車牌): https://github.com/detectRecog/CCPD")
    print("    含 250k+ 張圖，需轉換格式，車牌樣式與台灣不同但可輔助")
    print("  - OpenALPR benchmark: https://github.com/openalpr/benchmarks")
    print()
    print("準備好資料後重新執行:")
    print("  python scripts/train_plate_detector.py prepare")
    print()


# ===================================================================
# 訓練
# ===================================================================
def cmd_train(args):
    """訓練 YOLOv8n 車牌偵測模型"""
    print("=" * 60)
    print("步驟 2：訓練車牌偵測模型")
    print("=" * 60)

    if not DATASET_YAML.exists():
        print("❌ 找不到資料集配置，請先執行: python scripts/train_plate_detector.py prepare")
        sys.exit(1)

    # 統計資料量
    train_imgs = DATASET_DIR / "train" / "images"
    if not train_imgs.exists() or _count_images(train_imgs) == 0:
        print("❌ 訓練圖片目錄為空，請先準備資料集")
        sys.exit(1)

    n_train = _count_images(train_imgs)
    print(f"📊 訓練資料: {n_train} 張圖片")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 未安裝 ultralytics，請執行: pip install ultralytics")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 超參數（針對車牌小物件偵測最佳化）
    # ------------------------------------------------------------------
    epochs = args.epochs or _auto_epochs(n_train)
    batch = args.batch or _auto_batch()
    imgsz = args.imgsz or 640

    print(f"⚙️  超參數: epochs={epochs}, batch={batch}, imgsz={imgsz}")
    print(f"📁 輸出目錄: {OUTPUT_DIR}")

    # 使用 YOLOv8n 作為 base model
    base_model = args.base_model or "yolov8n.pt"
    print(f"🔧 基礎模型: {base_model}")

    model = YOLO(base_model)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(OUTPUT_DIR),
        name="plate_v1",
        exist_ok=True,
        # 車牌偵測最佳化參數
        patience=20,            # early stopping
        save=True,
        save_period=10,
        plots=True,
        # 資料增強（車牌場景適配）
        hsv_h=0.01,            # 色相：車牌顏色變化小
        hsv_s=0.4,             # 飽和度
        hsv_v=0.5,             # 明度：隧道/夜間光線大幅變化
        degrees=5.0,           # 旋轉：車牌傾斜角度有限
        translate=0.15,        # 平移
        scale=0.5,             # 縮放：遠近距離變化大
        shear=3.0,             # 剪切
        perspective=0.001,     # 透視
        flipud=0.0,            # 上下翻轉：車牌不會倒
        fliplr=0.5,            # 左右翻轉
        mosaic=1.0,            # Mosaic 增強
        mixup=0.1,             # MixUp
        erasing=0.3,           # 隨機遮擋（模擬部分遮擋）
        crop_fraction=0.8,     # 裁切比例
    )

    best_pt = OUTPUT_DIR / "plate_v1" / "weights" / "best.pt"
    if best_pt.exists():
        MODEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        dst = MODEL_EXPORT_DIR / BEST_PT
        shutil.copy2(best_pt, dst)
        print(f"✅ 最佳模型已複製到: {dst}")
        print(f"   檔案大小: {dst.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("⚠️  找不到 best.pt，請檢查訓練輸出")

    return results


def _auto_epochs(n_train: int) -> int:
    """根據資料量自動決定 epochs"""
    if n_train < 100:
        return 150
    if n_train < 500:
        return 120
    if n_train < 2000:
        return 80
    return 60


def _auto_batch() -> int:
    """根據 GPU 記憶體自動決定 batch size"""
    try:
        import torch

        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if mem_gb >= 16:
                return 32
            if mem_gb >= 8:
                return 16
            return 8
    except Exception:
        pass
    return 8


# ===================================================================
# 評估
# ===================================================================
def cmd_eval(args):
    """評估訓練好的模型"""
    print("=" * 60)
    print("步驟 3：評估模型")
    print("=" * 60)

    model_path = MODEL_EXPORT_DIR / BEST_PT
    if not model_path.exists():
        # 嘗試從訓練輸出找
        alt = OUTPUT_DIR / "plate_v1" / "weights" / "best.pt"
        if alt.exists():
            model_path = alt
        else:
            print(f"❌ 找不到模型: {model_path}")
            sys.exit(1)

    if not DATASET_YAML.exists():
        print("❌ 找不到資料集配置")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    print(f"📊 模型: {model_path}")
    print(f"   參數量: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

    # 在 val set 上評估
    results = model.val(
        data=str(DATASET_YAML),
        split="val",
        imgsz=640,
        batch=16,
        plots=True,
        project=str(OUTPUT_DIR),
        name="plate_eval",
        exist_ok=True,
    )

    print()
    print("📊 評估結果:")
    print(f"   mAP@0.5     : {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision    : {results.box.mp:.4f}")
    print(f"   Recall       : {results.box.mr:.4f}")
    print()

    # 判斷品質
    map50 = results.box.map50
    if map50 >= 0.90:
        print("✅ 模型品質: 優秀 — 可直接部署")
    elif map50 >= 0.80:
        print("✅ 模型品質: 良好 — 建議補充更多隧道/夜間資料")
    elif map50 >= 0.65:
        print("⚠️  模型品質: 普通 — 建議增加訓練資料到 1000+ 張")
    else:
        print("❌ 模型品質: 不足 — 需要更多標註資料或調整超參數")

    # 在 test set 上評估（如果有）
    test_dir = DATASET_DIR / "test" / "images"
    if test_dir.exists() and _count_images(test_dir) > 0:
        print()
        print("📊 Test set 評估:")
        test_results = model.val(
            data=str(DATASET_YAML),
            split="test",
            imgsz=640,
            batch=16,
            project=str(OUTPUT_DIR),
            name="plate_eval_test",
            exist_ok=True,
        )
        print(f"   mAP@0.5     : {test_results.box.map50:.4f}")
        print(f"   Precision    : {test_results.box.mp:.4f}")
        print(f"   Recall       : {test_results.box.mr:.4f}")

    return results


# ===================================================================
# 匯出
# ===================================================================
def cmd_export(args):
    """匯出模型供 plate_detector.py 使用"""
    print("=" * 60)
    print("步驟 4：匯出模型")
    print("=" * 60)

    model_path = MODEL_EXPORT_DIR / BEST_PT
    if not model_path.exists():
        alt = OUTPUT_DIR / "plate_v1" / "weights" / "best.pt"
        if alt.exists():
            MODEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alt, model_path)
        else:
            print(f"❌ 找不到模型: {model_path}")
            sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # 驗證 class names 包含 plate
    names = getattr(model.model, "names", None) or getattr(model, "names", None) or {}
    print(f"📋 模型類別: {names}")
    has_plate = any("plate" in str(v).lower() for v in names.values())
    if not has_plate:
        print("⚠️  注意: 模型類別名稱不含 'plate'")
        print("   plate_detector.py 會使用所有類別的偵測結果")

    # PyTorch 格式（已有）
    print(f"✅ PyTorch 模型: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ONNX 匯出
    try:
        onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
        onnx_dst = MODEL_EXPORT_DIR / "plate_yolov8n.onnx"
        if Path(onnx_path).exists():
            shutil.copy2(onnx_path, onnx_dst)
            print(f"✅ ONNX 模型: {onnx_dst}")
    except Exception as e:
        print(f"⚠️  ONNX 匯出失敗: {e}")

    # TensorRT 匯出（需要在 Jetson 或有 TensorRT 的環境）
    try:
        import tensorrt
        engine_path = model.export(
            format="engine",
            imgsz=640,
            half=True,       # FP16 加速
            device=0,
        )
        engine_dst = MODEL_EXPORT_DIR / "plate_yolov8n.engine"
        if Path(engine_path).exists():
            shutil.copy2(engine_path, engine_dst)
            print(f"✅ TensorRT 引擎: {engine_dst}")
    except ImportError:
        print("ℹ️  未偵測到 TensorRT，跳過 .engine 匯出")
        print("   請在 Jetson NX 上執行以下指令進行 TensorRT 轉換:")
        print(f"   python -c \"from ultralytics import YOLO; YOLO('{BEST_PT}').export(format='engine', imgsz=640, half=True)\"")
    except Exception as e:
        print(f"⚠️  TensorRT 匯出失敗: {e}")

    print()
    print("🚀 部署方式:")
    print(f"   1. 將 {MODEL_EXPORT_DIR}/ 下的模型複製到 Jetson NX:")
    print(f"      scp {MODEL_EXPORT_DIR / BEST_PT} user@jetson:/workspace/models/lpr/")
    print(f"   2. 設定環境變數:")
    print(f"      LPR_PLATE_MODEL_PT=plate_yolov8n.pt")
    print(f"   3. 重啟 API 服務，PlateDetector 會自動載入 YOLO 模型")


# ===================================================================
# 全流程
# ===================================================================
def cmd_all(args):
    cmd_prepare(args)
    # 檢查資料集是否就緒
    train_dir = DATASET_DIR / "train" / "images"
    if not train_dir.exists() or _count_images(train_dir) == 0:
        print()
        print("⏸️  資料集尚未就緒，請先準備資料後再執行 train")
        return
    cmd_train(args)
    cmd_eval(args)
    cmd_export(args)


# ===================================================================
# CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="台灣車牌偵測模型訓練工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
    python scripts/train_plate_detector.py prepare
    python scripts/train_plate_detector.py train --epochs 100 --batch 16
    python scripts/train_plate_detector.py eval
    python scripts/train_plate_detector.py export
    python scripts/train_plate_detector.py all
        """,
    )
    sub = parser.add_subparsers(dest="command", help="子指令")

    sub.add_parser("prepare", help="準備資料集")

    p_train = sub.add_parser("train", help="訓練模型")
    p_train.add_argument("--epochs", type=int, default=None, help="訓練輪數 (自動決定)")
    p_train.add_argument("--batch", type=int, default=None, help="Batch size (自動決定)")
    p_train.add_argument("--imgsz", type=int, default=640, help="輸入影像大小 (預設 640)")
    p_train.add_argument("--base-model", type=str, default="yolov8n.pt", help="基礎模型 (預設 yolov8n.pt)")

    sub.add_parser("eval", help="評估模型")
    sub.add_parser("export", help="匯出模型")

    p_all = sub.add_parser("all", help="全流程 (prepare → train → eval → export)")
    p_all.add_argument("--epochs", type=int, default=None)
    p_all.add_argument("--batch", type=int, default=None)
    p_all.add_argument("--imgsz", type=int, default=640)
    p_all.add_argument("--base-model", type=str, default="yolov8n.pt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cmds = {
        "prepare": cmd_prepare,
        "train": cmd_train,
        "eval": cmd_eval,
        "export": cmd_export,
        "all": cmd_all,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
