#!/usr/bin/env python3
"""
下載預訓練車牌偵測模型 — 最快部署方案

用法:
    python scripts/download_plate_model.py

效果:
    自動下載 → 驗證 → 部署到 models/lpr/plate_yolov8n.pt
    plate_detector.py 重啟後自動載入，無需改任何程式碼
"""
import os
import sys
import shutil
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "lpr"
TARGET_PT = MODEL_DIR / "plate_yolov8n.pt"


# ------------------------------------------------------------------
# 候選來源（依優先度排列）
# ------------------------------------------------------------------
SOURCES = [
    # 1) Hugging Face: keremberke 的 yolov8 車牌偵測（廣泛使用）
    {
        "name": "keremberke/yolov8n-license-plate (HuggingFace)",
        "method": "huggingface",
        "repo_id": "keremberke/yolov8n-license-plate-detection",
        "filename": "best.pt",
    },
    # 2) Ultralytics HUB 社群模型
    {
        "name": "Roboflow Universe pre-trained (pip roboflow)",
        "method": "roboflow_model",
        "workspace": "roboflow-universe-projects",
        "project": "license-plate-recognition-rxg4e",
        "version": 4,
        "model_format": "yolov8",
    },
]


def download_from_huggingface(source: dict) -> bool:
    """從 HuggingFace 下載預訓練模型"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("   安裝 huggingface_hub...")
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("   ❌ 無法安裝 huggingface_hub")
            return False

    repo_id = source["repo_id"]
    filename = source["filename"]
    print(f"   📥 下載中: {repo_id}/{filename}")

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        downloaded = Path(downloaded)
        if downloaded.exists() and downloaded.stat().st_size > 1_000_000:
            if downloaded != TARGET_PT:
                shutil.copy2(downloaded, TARGET_PT)
            return True
        print(f"   ⚠️  檔案太小或不存在: {downloaded}")
        return False
    except Exception as e:
        print(f"   ⚠️  下載失敗: {e}")
        return False


def download_from_roboflow(source: dict) -> bool:
    """從 Roboflow 下載預訓練模型"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("   跳過 Roboflow（未安裝）")
        return False

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("   跳過 Roboflow（未設定 ROBOFLOW_API_KEY）")
        return False

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(source["workspace"]).project(source["project"])
        model = project.version(source["version"]).model
        # Roboflow 模型下載
        print(f"   📥 下載 Roboflow 模型...")
        # 用 predict 觸發模型下載，然後取得權重
        return False  # Roboflow 模型 API 不直接提供 .pt 下載
    except Exception as e:
        print(f"   ⚠️  Roboflow 失敗: {e}")
        return False


def validate_model(model_path: Path) -> bool:
    """驗證模型可正常載入且包含正確類別"""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        names = getattr(model.model, "names", None) or getattr(model, "names", None) or {}
        print(f"   📋 模型類別: {names}")
        print(f"   📊 參數量: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

        # 確認 plate_detector.py 相容性
        has_plate = any("plate" in str(v).lower() or "license" in str(v).lower()
                        for v in names.values())
        if has_plate:
            print("   ✅ 類別名稱包含 'plate'，與 plate_detector.py 完全相容")
        else:
            print("   ⚠️  類別名稱不含 'plate'")
            print("      plate_detector.py 的 allowed_class_ids 篩選會失效")
            print("      需要修改 plate_detector.py 或重新命名類別")
            return False
        return True
    except Exception as e:
        print(f"   ❌ 模型驗證失敗: {e}")
        return False


def quick_benchmark(model_path: Path) -> None:
    """快速推論測試"""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import time

        model = YOLO(str(model_path))

        # 產生模擬車輛圖片 (640x480)
        dummy = np.random.randint(80, 200, (480, 640, 3), dtype=np.uint8)
        # 畫一個類似車牌的矩形
        cv2.rectangle(dummy, (200, 300), (440, 360), (255, 255, 255), -1)
        cv2.putText(dummy, "ABC-1234", (210, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # 暖機
        model(dummy, verbose=False, conf=0.1)

        # 計時
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            results = model(dummy, verbose=False, conf=0.1)
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        print(f"   ⚡ 推論速度: {avg_ms:.1f} ms/frame (avg of 10)")

        # 檢查偵測結果
        n_det = sum(len(r.boxes) for r in results)
        if n_det > 0:
            print(f"   ✅ 模擬測試偵測到 {n_det} 個物件")
        else:
            print("   ℹ️  模擬圖片未偵測到物件（正常，真實車牌會有結果）")

    except Exception as e:
        print(f"   ⚠️  跳過推論測試: {e}")


def main():
    print("=" * 60)
    print("🚗 車牌偵測模型快速部署")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 檢查是否已有模型
    if TARGET_PT.exists() and TARGET_PT.stat().st_size > 1_000_000:
        print(f"ℹ️  已存在模型: {TARGET_PT} ({TARGET_PT.stat().st_size / 1024 / 1024:.1f} MB)")
        print("   重新驗證中...")
        if validate_model(TARGET_PT):
            quick_benchmark(TARGET_PT)
            print()
            print("✅ 模型已就緒，無需重新下載")
            return
        print("   模型驗證失敗，重新下載...")

    # 依序嘗試各來源
    success = False
    for i, source in enumerate(SOURCES):
        print()
        print(f"[{i + 1}/{len(SOURCES)}] 嘗試: {source['name']}")
        method = source["method"]

        if method == "huggingface":
            success = download_from_huggingface(source)
        elif method == "roboflow_model":
            success = download_from_roboflow(source)

        if success and TARGET_PT.exists():
            print(f"   ✅ 下載成功: {TARGET_PT}")
            if validate_model(TARGET_PT):
                quick_benchmark(TARGET_PT)
                break
            else:
                print("   模型驗證失敗，嘗試下一個來源...")
                TARGET_PT.unlink(missing_ok=True)
                success = False

    if not success:
        print()
        print("=" * 60)
        print("⚠️  自動下載失敗，請手動下載:")
        print("=" * 60)
        print()
        print("最快方案（HuggingFace，推薦）:")
        print("  1. pip install huggingface_hub")
        print("  2. 重新執行此腳本")
        print()
        print("手動下載:")
        print("  1. 前往 https://huggingface.co/keremberke/yolov8n-license-plate-detection")
        print("  2. 下載 best.pt")
        print(f"  3. 放到 {TARGET_PT}")
        print()
        print("或用 Python 下載:")
        print('  from huggingface_hub import hf_hub_download')
        print('  hf_hub_download(')
        print('      repo_id="keremberke/yolov8n-license-plate-detection",')
        print('      filename="best.pt",')
        print(f'      local_dir="{MODEL_DIR}",')
        print('  )')
        return

    print()
    print("=" * 60)
    print("🚀 部署完成！")
    print("=" * 60)
    print(f"   模型位置: {TARGET_PT}")
    print(f"   檔案大小: {TARGET_PT.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("   plate_detector.py 會自動偵測並載入此模型")
    print("   重啟 API 服務即可生效:")
    print("     docker-compose restart api")
    print()
    print("   部署到 Jetson NX:")
    print(f"     scp {TARGET_PT} user@jetson:/workspace/models/lpr/")
    print()
    print("   在 Jetson 上轉 TensorRT（可選，加速推論）:")
    print("     python -c \"from ultralytics import YOLO; "
          "YOLO('/workspace/models/lpr/plate_yolov8n.pt')"
          ".export(format='engine', imgsz=640, half=True)\"")


if __name__ == "__main__":
    main()
